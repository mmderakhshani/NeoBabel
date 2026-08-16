# Mask-GRPO for NeoBabel (arXiv:2510.13418, "Reinforcement Learning Meets Masked
# Generative Models"), ported from the reference Show-o implementation.
#
# Per prompt: sample a group of G images with the recording rollout
# (models/modeling_neobabel.py: t2i_generate_grpo), score them with a reward
# model (CLIP or ImageReward), compute group-relative advantages, then replay
# the trajectory step by step accumulating the policy-gradient loss
# (t2i_grpo_step_loss). KL disabled by default (beta=0), asymmetric clip
# (0.9, 1.2), candidate-2 transition probability by default.
#
# Prompt files: plain .txt (one prompt per line) or .jsonl with
#   {"prompt": <rollout text, any language>,
#    "reward_prompt": <text scored by the reward model; defaults to prompt>,
#    "lang": ..., "uuid": ...}
# The reward_prompt indirection lets multilingual rollouts be scored by
# English-only reward models (ImageReward, CLIP-L) via parallel captions.
#
# Single-GPU usage:
#   python training/train_mask_grpo.py config=configs/mask_grpo_512.yaml \
#       model.neobabel.pretrained_model_path=/path/to/checkpoint
# Multi-GPU (data-parallel over prompts; gradients averaged once per step):
#   torchrun --nproc_per_node=4 training/train_mask_grpo.py config=...
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from models import NeoBabel, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor


def dist_init():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        return rank, dist.get_world_size(), local_rank
    return 0, 1, 0


def build_inputs(prompts, uni_prompting, mask_token_id, num_vq_tokens, device, mask_dtype):
    image_tokens = torch.ones((len(prompts), num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
    input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
    input_ids = input_ids.to(device)
    attention_mask = create_attention_mask_predict_next(
        input_ids,
        pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
        soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
        eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
        rm_pad_in_image=True).to(mask_dtype)
    return input_ids, attention_mask


@torch.no_grad()
def holdout_reward(model, vq_model, items, uni_prompting, mask_schedule, reward_score,
                   config, device, mask_dtype, samples, seed):
    # Generate at EVAL settings (CFG, full timesteps) and score with the
    # training reward model. Returns (reward_sum, count) on `device`.
    mask_token_id = model.config.mask_token_id
    num_vq_tokens = config.model.neobabel.num_vq_tokens
    generator = torch.Generator(device).manual_seed(seed)
    tot = torch.zeros(2, device=device)
    # eval-settings generation peaks above the training step (2*samples-way
    # cond+uncond batch at 50 timesteps); reclaim fragmented blocks first
    torch.cuda.empty_cache()
    # generate in small chunks: the 2n-way cond+uncond batch materializes an
    # (2n x seq x vocab) logits tensor (~0.8GB/sample) that OOMs alongside a
    # resident VLM reward model when n = all samples at once
    chunk = config.training.get("holdout_gen_batch", 1)
    for item in items:
        pil_images = []
        for c in range(0, samples, chunk):
            n = min(chunk, samples - c)
            prompts = [item["prompt"]] * n
            image_tokens = torch.ones((n, num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
            input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
            input_ids = input_ids.to(device)
            uncond_input_ids, _ = uni_prompting(([''] * n, image_tokens), 't2i_gen')
            uncond_input_ids = uncond_input_ids.to(device)
            attention_mask = create_attention_mask_predict_next(
                torch.cat([input_ids, uncond_input_ids], dim=0),
                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True).to(mask_dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                gen_ids = model.t2i_generate(
                    input_ids=input_ids, uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.get("eval_guidance_scale", 5),
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.get("eval_generation_timesteps", 50),
                    noise_schedule=mask_schedule, generator=generator, config=config)
            gen_ids = torch.clamp(gen_ids, max=config.model.neobabel.codebook_size - 1, min=0)
            imgs = vq_model.decode_code(gen_ids)
            imgs = (torch.clamp((imgs + 1.0) / 2.0, 0, 1) * 255).permute(0, 2, 3, 1)
            pil_images += [Image.fromarray(im) for im in imgs.float().cpu().numpy().astype(np.uint8)]
        rewards = reward_score(item["reward_prompt"], pil_images)
        tot[0] += rewards.sum()
        tot[1] += len(rewards)
    return tot


def load_reward_model(config, device):
    reward_type = config.training.get("reward_type", "clip")
    if reward_type == "ir_vqa_mix":
        # aesthetic + adherence: fixed linear combo so the holdout metric stays
        # interpretable (per-group z-scoring would zero the holdout mean).
        # ImageReward spans roughly [-2, 2]; VQA P("Yes") spans [0, 1]. Both
        # components score the same reward_prompt (use the English caption).
        from omegaconf import OmegaConf
        w_ir = config.training.get("mix_w_ir", 0.5)
        w_vqa = config.training.get("mix_w_vqa", 2.0)
        base = OmegaConf.to_container(config, resolve=True)
        cfg_ir = OmegaConf.create(base)
        cfg_ir.training.reward_type = "image_reward"
        cfg_ir.training.reward_model = config.training.get("mix_ir_model", "ImageReward-v1.0")
        cfg_vqa = OmegaConf.create(base)
        cfg_vqa.training.reward_type = "vlm_vqa"
        cfg_vqa.training.reward_model = config.training.get(
            "mix_vqa_model", "Qwen/Qwen2.5-VL-7B-Instruct")
        _, score_ir = load_reward_model(cfg_ir, device)
        _, score_vqa = load_reward_model(cfg_vqa, device)

        def score(reward_prompt, pil_images):
            return (w_ir * score_ir(reward_prompt, pil_images)
                    + w_vqa * score_vqa(reward_prompt, pil_images))
        return reward_type, score
    if reward_type == "image_reward":
        import ImageReward as RM
        rm = RM.load(config.training.get("reward_model", "ImageReward-v1.0"),
                     device=str(device),
                     download_root=config.training.get("reward_download_root", None))

        @torch.no_grad()
        def score(reward_prompt, pil_images):
            return torch.tensor(rm.score(reward_prompt, pil_images),
                                dtype=torch.float32, device=device)
    elif reward_type in ("vlm_logprob", "vlm_vqa"):
        # Annotation-free rewards from a frozen VLM; with a multilingual VLM these
        # score the native-language prompt directly (training.reward_use_native_prompt).
        # vlm_logprob (PromptEcho-style, github.com/roooobotx/prompt_echo): mean
        #   log-prob of the prompt tokens when the VLM re-generates the prompt from
        #   the image. Negative result: the prompt's LM cost is constant within a
        #   rollout group, so the image-dependent residual is tiny and GRPO stalls.
        # vlm_vqa (VQAScore-style; single-question DVReward): P("Yes") for "does
        #   this image match <prompt>?" — large per-image dynamic range.
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        name = config.training.get("reward_model", "Qwen/Qwen2.5-VL-7B-Instruct")
        vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            name, torch_dtype=torch.bfloat16).to(device)
        vlm.eval()
        vlm.requires_grad_(False)
        vlm_processor = AutoProcessor.from_pretrained(
            name, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)

        if reward_type == "vlm_vqa":
            question = config.training.get(
                "vlm_reward_instruction",
                "Does this image accurately depict the following description? "
                "Answer with only Yes or No.\nDescription: {prompt}")
            yes_id = vlm_processor.tokenizer.encode("Yes", add_special_tokens=False)[0]

            @torch.no_grad()
            def score(reward_prompt, pil_images):
                messages = [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": question.format(prompt=reward_prompt)}]}]
                text = vlm_processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                out = []
                for img in pil_images:
                    fin = vlm_processor(text=[text], images=[img],
                                        return_tensors="pt").to(device)
                    logits = vlm(**fin).logits[:, -1].float()
                    out.append(torch.softmax(logits, dim=-1)[0, yes_id].item())
                return torch.tensor(out, dtype=torch.float32, device=device)
            return reward_type, score

        instruction = config.training.get(
            "vlm_reward_instruction",
            "Write the exact text-to-image prompt that produced this image.")

        @torch.no_grad()
        def score(reward_prompt, pil_images):
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": instruction}]}]
            prefix = vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            out = []
            for img in pil_images:
                pre = vlm_processor(text=[prefix], images=[img], return_tensors="pt")
                fin = vlm_processor(text=[prefix + reward_prompt], images=[img],
                                    return_tensors="pt").to(device)
                n_prefix = pre.input_ids.shape[1]
                logits = vlm(**fin).logits
                logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                tok_lp = logprobs.gather(-1, fin.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                out.append(tok_lp[:, n_prefix - 1:].mean().item())
            return torch.tensor(out, dtype=torch.float32, device=device)
    else:
        name = config.training.get("reward_model", "openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained(name).to(device)
        clip_model.eval()
        clip_processor = CLIPProcessor.from_pretrained(name)

        @torch.no_grad()
        def score(reward_prompt, pil_images):
            inputs = clip_processor(text=[reward_prompt] * len(pil_images), images=pil_images,
                                    return_tensors="pt", padding=True, truncation=True).to(device)
            return clip_model(**inputs).logits_per_image.diag().float()
    return reward_type, score


def group_advantages(rewards):
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


def load_prompt_items(path, use_native_reward=False):
    with open(path) as f:
        if path.endswith(".jsonl"):
            items = [json.loads(l) for l in f if l.strip()]
        else:
            items = [{"prompt": p.strip()} for p in f if p.strip()]
    for it in items:
        if use_native_reward:
            it["reward_prompt"] = it["prompt"]
        else:
            it.setdefault("reward_prompt", it["prompt"])
    return items


def main():
    config = get_config()
    rank, world, local_rank = dist_init()
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cuda.matmul.allow_tf32 = True

    G = config.training.group_size
    T = config.training.generation_timesteps          # rollout steps (reduction: e.g. 20)
    loss_steps = config.training.get("grpo_loss_timesteps", T)  # computational reduction
    out_dir = config.experiment.output_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"grpo_log_rank{rank}.jsonl" if world > 1 else "grpo_log.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(config.model.neobabel.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100, cond_dropout_prob=0.0)
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = NeoBabel.from_pretrained(config.model.neobabel.pretrained_model_path).to(device)
    if config.model.get("gradient_checkpointing", True):
        model.neobabel.gradient_checkpointing_enable()
    mask_dtype = model.neobabel.get_input_embeddings().weight.dtype

    reward_type, reward_score = load_reward_model(config, device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.optimizer.params.learning_rate,
                                 betas=(0.9, 0.95))
    # Warmup + cosine decay (reference Mask-GRPO recipe: warmup 1000 / cosine
    # over the training horizon; constant lr collapsed our 1000-step run).
    warmup = config.training.get("lr_warmup_steps", 0)
    horizon = config.training.max_train_steps
    if config.training.get("lr_scheduler", "constant") == "cosine":
        def lr_lambda(s):
            if s < warmup:
                return (s + 1) / max(1, warmup)
            p = min(1.0, (s - warmup) / max(1, horizon - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * p))
    else:
        def lr_lambda(s):
            return min(1.0, (s + 1) / warmup) if warmup else 1.0
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    mask_token_id = model.config.mask_token_id
    num_vq_tokens = config.model.neobabel.num_vq_tokens

    use_native = config.training.get("reward_use_native_prompt", False)
    items = load_prompt_items(config.dataset.params.grpo_prompts_file, use_native)
    rng = np.random.RandomState(config.training.get("seed", 42))
    rng.shuffle(items)          # same permutation on every rank
    items = items[rank::world]  # rank-sharded prompts

    # Early stopping on a held-out prompt subset, evaluated at eval settings.
    eval_file = config.training.get("holdout_prompts_file", None)
    eval_every = config.training.get("holdout_eval_every", 0)
    eval_items = None
    if eval_file and eval_every:
        eval_items = load_prompt_items(eval_file, use_native)[:config.training.get("holdout_subset", 30)]
        eval_items = eval_items[rank::world]
    patience = config.training.get("early_stop_patience", 5)
    eval_samples = config.training.get("holdout_samples", 4)
    best_heldout, bad_evals, stop = float("-inf"), 0, False

    if rank == 0:
        print(f"{len(items)} prompts/rank x {world} ranks | reward={reward_type} | G={G} T={T} "
              f"loss_steps={loss_steps} lr={config.optimizer.params.learning_rate} "
              f"sched={config.training.get('lr_scheduler', 'constant')}(warmup {warmup}) "
              f"kl_beta={config.training.get('kl_beta', 0.0)} "
              f"holdout={'every %d steps, patience %d' % (eval_every, patience) if eval_items else 'off'}",
              flush=True)

    reward_std_history = []
    step = 0
    max_steps = config.training.max_train_steps
    # effective batch = world * grad_accum_prompts * G rollouts
    # (paper reference: 96 = 16 GPUs x 6; ours: 4 GPUs x accum x 6)
    accum = int(config.training.get("grad_accum_prompts", 1))

    def item_stream():
        while True:
            for it in items:
                yield it
    stream = item_stream()

    while step < max_steps and not stop:
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        micro_recs = []
        for micro in range(accum):
            item = next(stream)
            prompt, reward_prompt = item["prompt"], item["reward_prompt"]

            # ---- rollout a group ----
            model.eval()
            with torch.inference_mode():
                input_ids, attention_mask = build_inputs([prompt] * G, uni_prompting, mask_token_id,
                                                         num_vq_tokens, device, mask_dtype)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gen_ids, records = model.t2i_generate_grpo(
                        input_ids=input_ids, uncond_input_ids=None,
                        attention_mask=attention_mask, guidance_scale=0,
                        temperature=config.training.get("generation_temperature", 1.0),
                        timesteps=T, noise_schedule=mask_schedule, config=config)
                gen_ids = torch.clamp(gen_ids, max=config.model.neobabel.codebook_size - 1, min=0)
                imgs = vq_model.decode_code(gen_ids)
                imgs = (torch.clamp((imgs + 1.0) / 2.0, 0, 1) * 255).permute(0, 2, 3, 1)
                pil_images = [Image.fromarray(im) for im in imgs.float().cpu().numpy().astype(np.uint8)]
                rewards = reward_score(reward_prompt, pil_images)
            records = {k: [t.clone() for t in v] for k, v in records.items()}  # leave inference mode
            rewards = rewards.clone()
            advantages = group_advantages(rewards)
            mean_r, std_r = rewards.mean().item(), rewards.std().item()

            # ---- dynamic filtering: skip flat-reward groups ----
            # A skipped micro-batch simply contributes no gradient; ranks no
            # longer need matching skip patterns because the allreduce happens
            # once per optimizer step, after the accumulation loop.
            reward_std_history.append(std_r)
            skip = False
            if len(reward_std_history) > 50:
                thresh = float(np.percentile(reward_std_history[-500:], 10))
                if std_r < thresh:
                    skip = True
                    print(f"[{step}.{micro}][r{rank}] skip (reward std {std_r:.3f} < {thresh:.3f})",
                          flush=True)

            # ---- replay trajectory, accumulate policy gradient ----
            total_loss = 0.0
            if not skip:
                model.train()
                state_ids, sm = build_inputs([prompt] * G, uni_prompting, mask_token_id,
                                             num_vq_tokens, device, mask_dtype)
                for t in range(T):
                    compute = t < loss_steps  # first-k computational reduction
                    if compute:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            loss, state_ids = model.t2i_grpo_step_loss(
                                input_ids=state_ids, attention_mask=sm, timesteps=T,
                                config=config, records=records, timestep=t, advantages=advantages,
                                clip_low=config.training.get("clip_low", 0.9),
                                clip_high=config.training.get("clip_high", 1.2),
                                kl_beta=config.training.get("kl_beta", 0.0),
                                candidate=config.training.get("candidate", 2))
                        loss.backward()
                        total_loss += loss.item()
                    else:
                        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                            _, state_ids = model.t2i_grpo_step_loss(
                                input_ids=state_ids, attention_mask=sm, timesteps=T,
                                config=config, records=records, timestep=t, advantages=advantages)

            mrec = {"step": step, "micro": micro, "rank": rank,
                    "reward_mean": round(mean_r, 4), "reward_std": round(std_r, 4),
                    "rewards": [round(r, 3) for r in rewards.tolist()],
                    "skipped": skip, "loss": round(total_loss, 6), "prompt": prompt}
            for k in ("lang", "uuid"):
                if k in item:
                    mrec[k] = item[k]
            micro_recs.append(mrec)

        if accum > 1:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(accum)
        if world > 1:
            for p in model.parameters():
                if p.requires_grad:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        if config.training.get("max_grad_norm", None):
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm).item()
        else:
            grad_norm = torch.norm(torch.stack([
                p.grad.norm() for p in model.parameters() if p.grad is not None])).item()
        optimizer.step()
        lr_sched.step()

        dt = time.perf_counter() - t0
        step_r = [r for m in micro_recs for r in m["rewards"]]
        mean_r = sum(step_r) / len(step_r)
        with open(log_path, "a") as f:
            for m in micro_recs:
                m.update({"lr": round(lr_sched.get_last_lr()[0], 10),
                          "grad_norm": round(grad_norm, 4), "s": round(dt, 1)})
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"[{step}][r{rank}] reward {mean_r:.3f} ({len(step_r)} rollouts/"
              f"{accum} prompts) grad_norm {grad_norm:.3f} ({dt:.0f}s)", flush=True)

        step += 1
        if step % config.experiment.save_every == 0 or step == max_steps:
            if rank == 0:
                sd = os.path.join(out_dir, f"checkpoint-{step}")
                # single file: the repo's from_pretrained cannot read shards
                model.save_pretrained(sd, safe_serialization=False, max_shard_size="30GB")
                print(f"saved {sd}", flush=True)
            if world > 1:
                dist.barrier()

        # ---- held-out eval + early stopping ----
        if eval_items is not None and step % eval_every == 0:
            model.eval()
            with torch.inference_mode():
                tot = holdout_reward(model, vq_model, eval_items, uni_prompting, mask_schedule,
                                     reward_score, config, device, mask_dtype, eval_samples,
                                     seed=config.training.get("seed", 42) + step)
            if world > 1:
                dist.all_reduce(tot)
            heldout = (tot[0] / tot[1]).item()
            improved = heldout > best_heldout
            if improved:
                best_heldout, bad_evals = heldout, 0
                if rank == 0:
                    sd = os.path.join(out_dir, "checkpoint-best")
                    shutil.rmtree(sd, ignore_errors=True)
                    model.save_pretrained(sd, safe_serialization=False, max_shard_size="30GB")
                    with open(os.path.join(out_dir, "best.json"), "w") as f:
                        json.dump({"step": step, "heldout_reward": round(heldout, 4)}, f)
                if world > 1:
                    dist.barrier()
            else:
                bad_evals += 1
            if rank == 0:
                print(f"[{step}] holdout reward {heldout:+.4f} (best {best_heldout:+.4f}, "
                      f"bad_evals {bad_evals}/{patience})", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step": step, "holdout_reward": round(heldout, 4),
                                        "best": round(best_heldout, 4), "bad_evals": bad_evals}) + "\n")
            if bad_evals >= patience:
                if rank == 0:
                    print(f"early stop at step {step} (no improvement in {patience} evals)", flush=True)
                stop = True
                break
    if rank == 0:
        print("DONE", flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

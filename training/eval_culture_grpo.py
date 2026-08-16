# Before/after evaluation for Cultural Mask-GRPO: generate N samples per
# held-out cultural prompt at eval settings (CFG 5, 50 steps) and score them
# with the same reward model used for training (ImageReward on the parallel
# English caption). Reports per-language and overall mean reward.
#
# Usage (repo root on PYTHONPATH):
#   python training/eval_culture_grpo.py config=configs/mask_grpo_culture.yaml \
#       model.neobabel.pretrained_model_path=/path/to/checkpoint \
#       dataset.params.grpo_prompts_file=/path/to/culture_grpo_eval.jsonl \
#       experiment.output_dir=/path/to/out
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from models import NeoBabel, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.train_mask_grpo import load_prompt_items, load_reward_model
from training.utils import get_config
from transformers import AutoTokenizer


@torch.no_grad()
def generate(model, prompts, uni_prompting, mask_schedule, config, device, generator):
    mask_dtype = model.neobabel.get_input_embeddings().weight.dtype
    image_tokens = torch.ones((len(prompts), config.model.neobabel.num_vq_tokens),
                              dtype=torch.long, device=device) * model.config.mask_token_id
    input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
    input_ids = input_ids.to(device)
    uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
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
    return torch.clamp(gen_ids, max=config.model.neobabel.codebook_size - 1, min=0)


def main():
    config = get_config()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    samples = config.training.get("eval_samples_per_prompt", 4)
    out_dir = config.experiment.output_dir
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model.neobabel.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100, cond_dropout_prob=0.0)
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = NeoBabel.from_pretrained(config.model.neobabel.pretrained_model_path).to(device)
    model.eval()
    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    _, reward_score = load_reward_model(config, device)

    items = load_prompt_items(config.dataset.params.grpo_prompts_file)
    generator = torch.Generator(device).manual_seed(config.training.get("seed", 42))
    print(f"{len(items)} eval prompts x {samples} samples | "
          f"ckpt={config.model.neobabel.pretrained_model_path}", flush=True)

    by_lang = defaultdict(list)
    log_path = os.path.join(out_dir, "eval_scores.jsonl")
    t0 = time.perf_counter()
    for i, item in enumerate(items):
        gen_ids = generate(model, [item["prompt"]] * samples, uni_prompting,
                           mask_schedule, config, device, generator)
        imgs = vq_model.decode_code(gen_ids)
        imgs = (torch.clamp((imgs + 1.0) / 2.0, 0, 1) * 255).permute(0, 2, 3, 1)
        pil_images = [Image.fromarray(im) for im in imgs.float().cpu().numpy().astype(np.uint8)]
        rewards = reward_score(item["reward_prompt"], pil_images).tolist()
        lang = item.get("lang", "unknown")
        by_lang[lang].extend(rewards)
        tag = f"{item.get('uuid', i)}_{lang.replace(' ', '_')}"
        for k, im in enumerate(pil_images):
            im.save(os.path.join(out_dir, "images", f"{tag}_{k}.png"))
        with open(log_path, "a") as f:
            f.write(json.dumps({**{k: item.get(k) for k in ("uuid", "lang")},
                                "prompt": item["prompt"], "rewards": [round(r, 4) for r in rewards]},
                               ensure_ascii=False) + "\n")
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(items)} ({time.perf_counter() - t0:.0f}s)", flush=True)

    summary = {lang: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "n": len(v)}
               for lang, v in sorted(by_lang.items())}
    all_scores = [r for v in by_lang.values() for r in v]
    summary["ALL"] = {"mean": round(float(np.mean(all_scores)), 4),
                      "std": round(float(np.std(all_scores)), 4), "n": len(all_scores)}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    for lang, s in summary.items():
        print(f"{lang:>20}: {s['mean']:+.4f} ± {s['std']:.4f} (n={s['n']})", flush=True)


if __name__ == "__main__":
    main()

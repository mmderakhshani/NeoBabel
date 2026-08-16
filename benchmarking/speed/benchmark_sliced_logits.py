"""Measure the memory/speed effect of computing lm_head logits only for the
image-codebook rows and image positions during t2i generation.

The stock sampler computes full-vocab logits [B, S, 264203] and then slices
[:, -(num_vq+1):-1, token_offset:-1] before softmax (modeling_neobabel.py).
Because the lm_head rows are independent and Gemma-2's final-logit softcapping
is elementwise, restricting the projection to those rows/positions is
mathematically identical; this script verifies token parity empirically and
reports peak memory + latency for both paths.

Usage (single GPU):
  python benchmarking/speed/benchmark_sliced_logits.py \
      config=configs/mask_grpo_culture.yaml \
      model.neobabel.pretrained_model_path=<ckpt> batch_size=8
"""
import time

import torch
import torch.nn.functional as F

from models import MAGVITv2, NeoBabel
from training.prompting_utils import UniversalPrompting
from training.utils import get_config
from transformers import AutoTokenizer
from models.sampling import cosine_schedule, mask_by_random_topk


def sliced_forward(model, input_ids, attention_mask, num_vq_tokens, token_offset):
    lm = model.neobabel
    hidden = lm.model(input_ids=input_ids, attention_mask=attention_mask)[0]
    hidden = hidden[:, -(num_vq_tokens + 1):-1]
    weight = lm.lm_head.weight[token_offset:-1]
    logits = F.linear(hidden, weight)
    cap = lm.config.final_logit_softcapping
    if cap is not None:
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
    return logits


@torch.inference_mode()
def t2i_generate_sliced(model, input_ids, uncond_input_ids, attention_mask,
                        temperature, timesteps, guidance_scale, noise_schedule,
                        generator, config):
    mask_token_id = model.config.mask_token_id
    num_vq_tokens = config.model.neobabel.num_vq_tokens
    num_new_special_tokens = config.model.neobabel.num_new_special_tokens
    token_offset = config.model.neobabel.llm_vocab_size + num_new_special_tokens

    ids_minus = input_ids[:, -(num_vq_tokens + 1):-1].clone()
    ids_minus = torch.where(ids_minus == mask_token_id, mask_token_id, ids_minus - token_offset)
    uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

    for step in range(timesteps):
        uncond_input_ids = torch.cat(
            [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
        model_input = torch.cat([input_ids, uncond_input_ids])
        logits2 = sliced_forward(model, model_input, attention_mask, num_vq_tokens, token_offset)
        cond_logits, uncond_logits = logits2.chunk(2)
        logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits

        probs = logits.softmax(dim=-1)
        sampled = probs.reshape(-1, logits.size(-1))
        sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

        unknown_map = ids_minus == mask_token_id
        sampled_ids = torch.where(unknown_map, sampled_ids, ids_minus)
        ratio = 1.0 * (step + 1) / timesteps
        mask_ratio = noise_schedule(torch.tensor(ratio))
        selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)
        selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
        mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
        mask_len = torch.max(torch.tensor([1], device=logits.device),
                             torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len))
        temperature_s = temperature * (1.0 - ratio)
        masking = mask_by_random_topk(mask_len, selected_probs, temperature_s, generator=generator)
        input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(
            masking, mask_token_id, sampled_ids + token_offset)
        ids_minus = torch.where(masking, mask_token_id, sampled_ids)

    return sampled_ids


def main():
    config = get_config()
    device = torch.device("cuda")
    batch = int(config.get("batch_size", 8))
    prompts = ["A traditional Dutch windmill beside a tulip field at golden hour, "
               "a cyclist passing on the dike path"] * batch

    tokenizer = AutoTokenizer.from_pretrained(config.model.neobabel.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100, cond_dropout_prob=0.0)
    model = NeoBabel.from_pretrained(config.model.neobabel.pretrained_model_path).to(device)
    model.eval()
    mask_dtype = model.neobabel.get_input_embeddings().weight.dtype
    from training.prompting_utils import create_attention_mask_predict_next as camp
    from models.sampling import cosine_schedule as sched

    num_vq_tokens = config.model.neobabel.num_vq_tokens
    mask_token_id = model.config.mask_token_id

    def build_inputs():
        image_tokens = torch.ones((batch, num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
        input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
        input_ids = input_ids.to(device)
        uncond_ids, _ = uni_prompting(([''] * batch, image_tokens), 't2i_gen')
        uncond_ids = uncond_ids.to(device)
        attn = camp(torch.cat([input_ids, uncond_ids], dim=0),
                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                    rm_pad_in_image=True).to(mask_dtype)
        return input_ids, uncond_ids, attn

    fast = bool(config.get("fast", False))
    results = {}
    for name in ["baseline", "sliced", "baseline", "sliced"]:  # 2nd pass = warm timings
        input_ids, uncond_ids, attn = build_inputs()
        gen = torch.Generator(device).manual_seed(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if fast:
                with torch.inference_mode():
                    ids = model.t2i_generate_fast(
                        input_ids=input_ids, uncond_input_ids=uncond_ids, attention_mask=attn,
                        guidance_scale=5, temperature=1.0, timesteps=50,
                        noise_schedule=sched, generator=gen, config=config,
                        sliced_head=(name == "sliced"))
            elif name == "baseline":
                with torch.inference_mode():
                    ids = model.t2i_generate(
                        input_ids=input_ids, uncond_input_ids=uncond_ids, attention_mask=attn,
                        guidance_scale=5, temperature=1.0, timesteps=50,
                        noise_schedule=sched, generator=gen, config=config)
            else:
                ids = t2i_generate_sliced(model, input_ids, uncond_ids, attn,
                                          1.0, 50, 5, sched, gen, config)
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 2**30
        peak_res = torch.cuda.max_memory_reserved() / 2**30
        results[name] = {"ids": ids.cpu(), "time": dt, "alloc": peak_alloc, "reserved": peak_res}
        print(f"{name:9s} time {dt:6.2f}s  peak_alloc {peak_alloc:6.2f} GiB  peak_reserved {peak_res:6.2f} GiB",
              flush=True)

    same = (results["baseline"]["ids"] == results["sliced"]["ids"]).float().mean().item()
    print(f"token parity (same seed): {same*100:.4f}% identical")
    print(f"peak alloc: {results['baseline']['alloc']:.2f} -> {results['sliced']['alloc']:.2f} GiB "
          f"({results['baseline']['alloc']/max(results['sliced']['alloc'],1e-9):.2f}x)")
    print(f"latency:    {results['baseline']['time']:.2f} -> {results['sliced']['time']:.2f} s/batch")


if __name__ == "__main__":
    main()

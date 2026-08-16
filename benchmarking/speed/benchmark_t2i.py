# Timing grid for NeoBabel t2i inference.
#
# Loads the model ONCE (startup amortized), then times a grid of
# {precision, attention impl, kv-cache fast path, compile} x {timesteps} x {batch}.
# Also runs a parity check between t2i_generate and t2i_generate_fast.
#
# Usage (from repo root, PYTHONPATH=repo root):
#   python benchmarking/speed/benchmark_t2i.py config=configs/eval/neobabel_demo_512x512.yaml \
#       model.neobabel.pretrained_model_path=/path/to/checkpoint \
#       output_dir=speed_bench_out
import json
import os
import time

import numpy as np
import torch
from PIL import Image

from models import NeoBabel, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config
from transformers import AutoTokenizer


def build_inputs(prompts, uni_prompting, mask_token_id, num_vq_tokens, guidance_scale, device, mask_dtype):
    image_tokens = torch.ones((len(prompts), num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
    input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')
    input_ids = input_ids.to(device)
    if guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
        uncond_input_ids = uncond_input_ids.to(device)
        attention_mask = create_attention_mask_predict_next(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True)
    else:
        uncond_input_ids = None
        attention_mask = create_attention_mask_predict_next(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True)
    return input_ids, uncond_input_ids, attention_mask.to(mask_dtype)


def set_attn(model, impl):
    try:
        model.neobabel.set_attn_implementation(impl)
    except AttributeError:
        model.neobabel.config._attn_implementation = impl
        for module in model.neobabel.modules():
            if hasattr(module, "config") and hasattr(module.config, "_attn_implementation"):
                module.config._attn_implementation = impl


@torch.no_grad()
def run_generate(model, fast, prompts, uni_prompting, mask_schedule, config, device, seed):
    guidance_scale = config.training.guidance_scale
    mask_dtype = model.neobabel.get_input_embeddings().weight.dtype
    input_ids, uncond_input_ids, attention_mask = build_inputs(
        prompts, uni_prompting, model.config.mask_token_id,
        config.model.neobabel.num_vq_tokens, guidance_scale, device, mask_dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    fn = model.t2i_generate_fast if fast else model.t2i_generate
    return fn(
        input_ids=input_ids,
        uncond_input_ids=uncond_input_ids,
        attention_mask=attention_mask,
        guidance_scale=guidance_scale,
        temperature=config.training.get("generation_temperature", 1.0),
        timesteps=config.training.generation_timesteps,
        noise_schedule=mask_schedule,
        generator=generator,
        config=config,
    )


def main():
    config = get_config()
    out_dir = config.get("output_dir", "speed_bench_out")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False  # keep fp32 baseline honest

    config.training.guidance_scale = config.get("guidance_scale", 5)
    config.training.batch_size = 1

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config.model.neobabel.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    model = NeoBabel.from_pretrained(config.model.neobabel.pretrained_model_path).to(device)
    model.eval()
    startup_s = time.perf_counter() - t0
    print(f"[startup] {startup_s:.1f}s (amortized across all cells below)")

    with open(config.dataset.params.validation_prompts_file) as f:
        base_prompts = [p for p in f.read().splitlines() if p.strip()]

    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    results = {"startup_s": startup_s, "gpu": torch.cuda.get_device_name(0), "cells": []}

    # ---------- parity check: fast path vs original, fp32 eager, same seed ----------
    model.float()
    set_attn(model, "eager")
    config.training.generation_timesteps = 20
    ids_a = run_generate(model, False, base_prompts[:2], uni_prompting, mask_schedule, config, device, seed=1234)
    ids_b = run_generate(model, True, base_prompts[:2], uni_prompting, mask_schedule, config, device, seed=1234)
    match = (ids_a == ids_b).float().mean().item()
    results["parity_fp32_token_match"] = match
    print(f"[parity] fp32 eager, original vs fast: {match*100:.2f}% tokens identical")

    # ---------- grid ----------
    attn_requested = config.get("attn_impl_post", "flex_attention")
    cells = [
        # name,                dtype,   attn,            fast,  compile, steps, batch
        ("pre_fp32_eager",     "fp32",  "eager",         False, False,   20, 1),
        ("pre_fp32_eager",     "fp32",  "eager",         False, False,   50, 1),
        ("pre_fp32_eager",     "fp32",  "eager",         False, False,   20, 4),
        ("pre_fp32_eager",     "fp32",  "eager",         False, False,   50, 4),
        ("bf16_eager",         "bf16",  "eager",         False, False,   20, 1),
        ("bf16_eager",         "bf16",  "eager",         False, False,   50, 1),
        ("bf16_fast_eager",    "bf16",  "eager",         True,  False,   20, 1),
        ("bf16_fast_eager",    "bf16",  "eager",         True,  False,   50, 1),
        ("bf16_fast_attn",     "bf16",  attn_requested,  True,  False,   20, 1),
        ("bf16_fast_attn",     "bf16",  attn_requested,  True,  False,   50, 1),
        ("post_compiled",      "bf16",  attn_requested,  True,  True,    20, 1),
        ("post_compiled",      "bf16",  attn_requested,  True,  True,    50, 1),
        ("post_compiled",      "bf16",  attn_requested,  True,  True,    20, 4),
        ("post_compiled",      "bf16",  attn_requested,  True,  True,    50, 4),
    ]

    compiled = False
    for name, dtype, attn, fast, do_compile, steps, batch in cells:
        label = f"{name}_steps{steps}_b{batch}"
        model.to(torch.bfloat16 if dtype == "bf16" else torch.float32)
        try:
            set_attn(model, attn)
            if do_compile and not compiled:
                model.neobabel = torch.compile(model.neobabel)
                compiled = True
            config.training.generation_timesteps = steps
            prompts = (base_prompts * ((batch // len(base_prompts)) + 1))[:batch]

            # warmup (also triggers compilation)
            run_generate(model, fast, prompts, uni_prompting, mask_schedule, config, device, seed=7)
            torch.cuda.synchronize()

            times = []
            n_runs = 3 if batch == 1 else 2
            gen_ids = None
            for r in range(n_runs):
                torch.cuda.synchronize()
                t = time.perf_counter()
                gen_ids = run_generate(model, fast, prompts, uni_prompting, mask_schedule, config, device, seed=100 + r)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)

            per_image = min(times) / batch
            cell = {"cell": label, "dtype": dtype, "attn": attn, "fast": fast,
                    "compile": do_compile, "timesteps": steps, "batch": batch,
                    "times_s": [round(x, 3) for x in times],
                    "best_s_per_image": round(per_image, 3)}
            print(f"[{label}] best {per_image:.2f}s/image  (runs: {['%.2f' % x for x in times]})")

            gen_ids = torch.clamp(gen_ids, max=config.model.neobabel.codebook_size - 1, min=0)
            img = vq_model.decode_code(gen_ids[:1])
            img = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
            img = img.permute(0, 2, 3, 1).float().cpu().numpy().astype(np.uint8)
            Image.fromarray(img[0]).save(os.path.join(out_dir, f"{label}.png"))
        except Exception as e:
            cell = {"cell": label, "error": f"{type(e).__name__}: {e}"}
            print(f"[{label}] FAILED: {type(e).__name__}: {e}")
        results["cells"].append(cell)

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {out_dir}/results.json")


if __name__ == "__main__":
    main()

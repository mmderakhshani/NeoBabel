# Timed m-GenEval generation run: release-config baseline (English subset) vs
# the optimized inference stack (all languages), one process, model loaded once.
#
# Output images use the GenEval directory layout so the run can be scored later:
#   <output_dir>/<language>/<idx:05d>/samples/00000.png + metadata.jsonl
#
# Usage (repo root on PYTHONPATH):
#   python benchmarking/speed/run_mgeneval_timed.py config=configs/eval/neobabel_gen_eval_512x512.yaml \
#       model.neobabel.pretrained_model_path=... prompts_dir=... output_dir=... \
#       pre_subset=48 post_batch=8
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

LANGS = ["english", "french", "hindi", "persian", "dutch", "chinese"]


def load_metadata(prompts_dir, lang):
    with open(os.path.join(prompts_dir, f"evaluation_metadata_{lang}.jsonl")) as f:
        return [json.loads(line) for line in f]


def set_attn(model, impl):
    try:
        model.neobabel.set_attn_implementation(impl)
    except AttributeError:
        model.neobabel.config._attn_implementation = impl


@torch.no_grad()
def generate_batch(model, fast, prompts, uni_prompting, mask_schedule, config, device, generator):
    guidance_scale = config.training.guidance_scale
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


def decode_and_save(vq_model, gen_ids, metadatas, out_root, lang, indices, config):
    gen_ids = torch.clamp(gen_ids, max=config.model.neobabel.codebook_size - 1, min=0)
    imgs = vq_model.decode_code(gen_ids)
    imgs = torch.clamp((imgs + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
    imgs = imgs.permute(0, 2, 3, 1).float().cpu().numpy().astype(np.uint8)
    for img, meta, idx in zip(imgs, metadatas, indices):
        outpath = os.path.join(out_root, lang, f"{idx:0>5}")
        os.makedirs(os.path.join(outpath, "samples"), exist_ok=True)
        Image.fromarray(img).save(os.path.join(outpath, "samples", "00000.png"))
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
            json.dump(meta, f, ensure_ascii=False)


def main():
    config = get_config()
    prompts_dir = config.prompts_dir
    out_root = config.output_dir
    pre_subset = int(config.get("pre_subset", 48))
    post_batch = int(config.get("post_batch", 8))
    os.makedirs(out_root, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False

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
    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    print(f"[startup] {time.perf_counter() - t0:.1f}s", flush=True)
    results = {"protocol": {"timesteps": config.training.generation_timesteps,
                            "guidance_scale": config.training.guidance_scale,
                            "resolution": config.dataset.preprocessing.resolution},
               "gpu": torch.cuda.get_device_name(0)}

    generator = torch.Generator(device=device)
    generator.manual_seed(config.training.seed if config.training.get("seed") else 42)

    # ---------- PRE: release configuration, English subset ----------
    metas_en = load_metadata(prompts_dir, "english")[:pre_subset]
    model.float()
    set_attn(model, "eager")
    # warmup
    generate_batch(model, False, [metas_en[0]["prompt"]], uni_prompting, mask_schedule, config, device, generator)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i, meta in enumerate(metas_en):
        gen_ids = generate_batch(model, False, [meta["prompt"]], uni_prompting, mask_schedule, config, device, generator)
        decode_and_save(vq_model, gen_ids, [meta], out_root, "english_pre", [i], config)
    torch.cuda.synchronize()
    pre_s = time.perf_counter() - t
    results["pre"] = {"config": "fp32 + eager + original t2i_generate, batch 1 (release settings)",
                      "language": "english", "images": len(metas_en),
                      "total_s": round(pre_s, 1), "s_per_image": round(pre_s / len(metas_en), 3),
                      "extrapolated_full_english_553_s": round(pre_s / len(metas_en) * 553, 0)}
    print(f"[pre] english {len(metas_en)} imgs: {pre_s:.1f}s -> {pre_s/len(metas_en):.2f} s/img", flush=True)

    # ---------- POST: optimized stack, all languages ----------
    model.to(torch.bfloat16)
    set_attn(model, config.get("attn_impl_post", "flex_attention"))
    model.neobabel = torch.compile(model.neobabel)
    # warmup / compile at the fixed batch shape
    warm = [load_metadata(prompts_dir, "english")[0]["prompt"]] * post_batch
    generate_batch(model, True, warm, uni_prompting, mask_schedule, config, device, generator)
    torch.cuda.synchronize()

    results["post"] = {"config": f"bf16 + flex_attention + prefix-KV cache + torch.compile, batch {post_batch}",
                       "languages": {}}
    grand_imgs, grand_s = 0, 0.0
    for lang in LANGS:
        metas = load_metadata(prompts_dir, lang)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for start in range(0, len(metas), post_batch):
            chunk = metas[start:start + post_batch]
            prompts = [m["prompt"] for m in chunk]
            n_real = len(prompts)
            if n_real < post_batch:  # pad to keep the compiled shape, drop extras
                prompts = prompts + [prompts[-1]] * (post_batch - n_real)
            gen_ids = generate_batch(model, True, prompts, uni_prompting, mask_schedule, config, device, generator)
            decode_and_save(vq_model, gen_ids[:n_real], chunk, out_root, lang,
                            list(range(start, start + n_real)), config)
        torch.cuda.synchronize()
        lang_s = time.perf_counter() - t
        results["post"]["languages"][lang] = {
            "images": len(metas), "total_s": round(lang_s, 1),
            "s_per_image": round(lang_s / len(metas), 3)}
        grand_imgs += len(metas)
        grand_s += lang_s
        print(f"[post] {lang}: {len(metas)} imgs in {lang_s:.1f}s -> {lang_s/len(metas):.2f} s/img", flush=True)

    results["post"]["total_images"] = grand_imgs
    results["post"]["total_s"] = round(grand_s, 1)
    results["post"]["s_per_image"] = round(grand_s / grand_imgs, 3)
    results["speedup_s_per_image"] = round(results["pre"]["s_per_image"] / results["post"]["s_per_image"], 1)
    print(f"[post] ALL: {grand_imgs} imgs in {grand_s/60:.1f}min -> {grand_s/grand_imgs:.2f} s/img", flush=True)
    print(f"[speedup] {results['speedup_s_per_image']}x per image vs release settings", flush=True)

    with open(os.path.join(out_root, "timing.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"written {out_root}/timing.json", flush=True)


if __name__ == "__main__":
    main()

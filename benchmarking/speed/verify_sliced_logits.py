"""Verify the sliced-lm_head generation path: logit-level diff on one forward,
plus side-by-side image decode of both paths (same seed) for visual QA.

  python benchmarking/speed/verify_sliced_logits.py config=configs/mask_grpo_culture.yaml \
      model.neobabel.pretrained_model_path=<ckpt> out_dir=<dir>
"""
import os

import torch
from PIL import Image
import numpy as np

from models import MAGVITv2, NeoBabel
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config
from transformers import AutoTokenizer
from models.sampling import cosine_schedule
from benchmarking.speed.benchmark_sliced_logits import sliced_forward, t2i_generate_sliced

PROMPTS = [
    "A traditional Dutch windmill beside a tulip field at golden hour",
    "A Chinese dragon dance in a lantern-lit night market",
    "A Hindu temple gopuram covered in painted sculptures under a blue sky",
    "A Persian garden courtyard with a turquoise-tiled fountain",
]


def main():
    config = get_config()
    device = torch.device("cuda")
    out_dir = config.get("out_dir", "sliced_verify_out")
    os.makedirs(out_dir, exist_ok=True)
    batch = len(PROMPTS)

    tokenizer = AutoTokenizer.from_pretrained(config.model.neobabel.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
        ignore_id=-100, cond_dropout_prob=0.0)
    model = NeoBabel.from_pretrained(config.model.neobabel.pretrained_model_path).to(device)
    model.eval()
    vq_model = MAGVITv2.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.eval()
    mask_dtype = model.neobabel.get_input_embeddings().weight.dtype
    num_vq_tokens = config.model.neobabel.num_vq_tokens
    num_new = config.model.neobabel.num_new_special_tokens
    token_offset = config.model.neobabel.llm_vocab_size + num_new
    mask_token_id = model.config.mask_token_id

    def build_inputs():
        image_tokens = torch.ones((batch, num_vq_tokens), dtype=torch.long, device=device) * mask_token_id
        input_ids, _ = uni_prompting((PROMPTS, image_tokens), 't2i_gen')
        input_ids = input_ids.to(device)
        uncond_ids, _ = uni_prompting(([''] * batch, image_tokens), 't2i_gen')
        uncond_ids = uncond_ids.to(device)
        attn = create_attention_mask_predict_next(
            torch.cat([input_ids, uncond_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True).to(mask_dtype)
        return input_ids, uncond_ids, attn

    # 1) logit-level comparison on one forward (fully-masked input)
    input_ids, uncond_ids, attn = build_inputs()
    model_input = torch.cat([input_ids, torch.cat(
        [uncond_ids[:, :config.dataset.preprocessing.max_seq_length + 1],
         input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        full = model(model_input, attention_mask=attn)
        full_sliced = full[:, -(num_vq_tokens + 1):-1, token_offset:-1].float()
        direct = sliced_forward(model, model_input, attn, num_vq_tokens, token_offset).float()
    d = (full_sliced - direct).abs()
    scale = full_sliced.abs().max().item()
    print(f"logit max|diff| {d.max().item():.6f}  mean|diff| {d.mean().item():.6f}  "
          f"(logit range ±{scale:.1f})", flush=True)
    p_full = full_sliced.softmax(-1)
    p_dir = direct.softmax(-1)
    print(f"prob  max|diff| {(p_full - p_dir).abs().max().item():.6f}  "
          f"TV distance mean {(p_full - p_dir).abs().sum(-1).mul(0.5).mean().item():.6f}", flush=True)

    # 2) full generation, both paths, same seed -> decode images
    @torch.no_grad()
    def decode(ids, tag):
        ids = torch.clamp(ids, max=config.model.neobabel.codebook_size - 1, min=0)
        imgs = vq_model.decode_code(ids)
        imgs = (torch.clamp((imgs + 1.0) / 2.0, 0, 1) * 255).permute(0, 2, 3, 1)
        for i, im in enumerate(imgs.float().cpu().numpy().astype(np.uint8)):
            Image.fromarray(im).save(os.path.join(out_dir, f"{tag}_{i}.png"))

    input_ids, uncond_ids, attn = build_inputs()
    gen = torch.Generator(device).manual_seed(7)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        base_ids = model.t2i_generate(input_ids=input_ids, uncond_input_ids=uncond_ids,
                                      attention_mask=attn, guidance_scale=5, temperature=1.0,
                                      timesteps=50, noise_schedule=cosine_schedule,
                                      generator=gen, config=config)
    decode(base_ids, "baseline")
    input_ids, uncond_ids, attn = build_inputs()
    gen = torch.Generator(device).manual_seed(7)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sl_ids = t2i_generate_sliced(model, input_ids, uncond_ids, attn, 1.0, 50, 5,
                                     cosine_schedule, gen, config)
    decode(sl_ids, "sliced")
    print(f"images saved to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

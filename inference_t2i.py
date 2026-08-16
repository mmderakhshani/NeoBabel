# This work is based in part on code from Show-o (https://github.com/showlab/Show-o).
# Significant modifications for NeoBabel.

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import NeoBabel, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config, flatten_omega_conf, image_transform
from transformers import AutoTokenizer
import torch.nn.functional as F

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")

if __name__ == '__main__':

    config = get_config()

    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="neobabel-inference-demo",
        name=config.experiment.name + '_t2i' + f'_{config.mode}',
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.neobabel.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    model = NeoBabel.from_pretrained(config.model.neobabel.pretrained_model_path).to(device)
    model.eval()

    # --- inference speed options (all off by default) ---
    # precision=bf16          run the transformer in bfloat16 (~3-5x on H100 vs fp32)
    # attn_implementation=... e.g. flex_attention (torch>=2.5; supports gemma-2 soft-cap
    #                         and the custom mask, unlike sdpa) — default stays eager
    # fast_generate=true      cache the text-prefix KV once, forward only the image
    #                         segment each step (see NeoBabel.t2i_generate_fast)
    # compile=true            torch.compile the transformer
    precision = config.get("precision", "fp32")
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
        print("Running in bf16")
    attn_impl = config.get("attn_implementation", None)
    if attn_impl:
        try:
            model.neobabel.set_attn_implementation(attn_impl)
        except AttributeError:
            model.neobabel.config._attn_implementation = attn_impl
        print(f"Attention implementation: {attn_impl}")
    if config.get("compile", False):
        model.neobabel = torch.compile(model.neobabel)
        print("torch.compile enabled")
    generate_fn = model.t2i_generate_fast if config.get("fast_generate", False) else model.t2i_generate
    mask_dtype = model.neobabel.get_input_embeddings().weight.dtype
    # --- end speed options ---

    mask_token_id = model.config.mask_token_id

    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size
    config.training.guidance_scale = config.guidance_scale
    config.training.generation_timesteps = config.generation_timesteps
    # load from users passed arguments

    if config.mode == 'inpainting':

        prompt = [config.prompt] * config.batch_size
        inpainting_image = Image.open(config.image_path).convert("RGB")
        inpainting_mask = Image.open(config.inpainting_mask_path).convert("L")

        inpainting_image = image_transform(inpainting_image, resolution=config.dataset.params.resolution).to(device)
        inpainting_mask = image_transform(inpainting_mask, resolution=config.dataset.params.resolution, normalize=False)

        # record original image and inpainting mask
        images = torch.clamp(
            (torch.stack([inpainting_image, inpainting_mask.repeat(3, 1, 1).to(device)], dim=0) + 1.0) / 2.0,
            min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        labels = ['original image', 'inpainting mask']
        wandb_images = [wandb.Image(image, caption=labels[i]) for i, image in enumerate(pil_images)]

        inpainting_image = inpainting_image.unsqueeze(0).repeat(config.training.batch_size, 1, 1, 1)

        inpainting_mask = inpainting_mask.unsqueeze(0).to(device)
        inpainting_mask = F.interpolate(inpainting_mask, size=config.dataset.params.resolution // 16, mode='bicubic')
        inpainting_mask = inpainting_mask.repeat(config.training.batch_size, 1, 1, 1)

        inpainting_mask[inpainting_mask < 0.5] = 0
        inpainting_mask[inpainting_mask >= 0.5] = 1

        inpainting_mask = inpainting_mask.reshape(config.training.batch_size, -1)
        inpainting_mask = inpainting_mask.to(torch.bool)

        inpainting_image_tokens = vq_model.get_code(inpainting_image) + len(uni_prompting.text_tokenizer)
        inpainting_image_tokens[inpainting_mask] = mask_token_id

        input_ids, _ = uni_prompting((prompt, inpainting_image_tokens), 't2i_gen')

        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prompt), inpainting_image_tokens), 't2i_gen')
            attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            attention_mask = attention_mask.to(mask_dtype)
        else:
            attention_mask = create_attention_mask_predict_next(input_ids,
                                                                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                rm_pad_in_image=True)
            attention_mask = attention_mask.to(mask_dtype)
            uncond_input_ids = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

        with torch.no_grad():
            gen_token_ids = generate_fn(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                guidance_scale=config.training.guidance_scale,
                temperature=config.training.get("generation_temperature", 1.0),
                timesteps=config.training.generation_timesteps,
                noise_schedule=mask_schedule,
                noise_type=config.training.get("noise_type", "mask"),
                seq_len=config.model.neobabel.num_vq_tokens,
                uni_prompting=uni_prompting,
                config=config,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.neobabel.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        wandb_images.extend([wandb.Image(image, caption=prompt[i]) for i, image in enumerate(pil_images)])
        wandb.log({"generated_images": wandb_images}, step=0)

    elif config.mode == 'extrapolation':

        prompt = [p for p in config.prompt.split(" *** ") if len(p) != 0]
        extra_direction = [d for d in config.extra_direction.split(" *** ") if len(d) != 0]
        print(prompt, extra_direction)
        W = config.dataset.params.resolution // 16
        for id, (prt, direction) in enumerate(zip(prompt, extra_direction)):
            prt = [prt] * config.training.batch_size
            if id == 0:
                extrapolation_image = Image.open(config.image_path).convert("RGB")
                extrapolation_image = image_transform(extrapolation_image,
                                                      resolution=config.dataset.params.resolution).to(device)

                B, _, _ = extrapolation_image.shape
                extrapolation_image = extrapolation_image.unsqueeze(0)
                extrapolation_image_tokens = vq_model.get_code(extrapolation_image) + len(uni_prompting.text_tokenizer)
                extrapolation_image_tokens = extrapolation_image_tokens.reshape(1,
                                                                                config.dataset.params.resolution // 16,
                                                                                config.dataset.params.resolution // 16)
                extrapolation_image_tokens = extrapolation_image_tokens.repeat(config.training.batch_size, 1, 1)
            else:


                extrapolation_image_tokens = gen_token_ids + len(uni_prompting.text_tokenizer)

            image_left_part = extrapolation_image_tokens[:, :, :-(W//2-config.offset)] - len(uni_prompting.text_tokenizer)
            image_right_part = extrapolation_image_tokens[:, :, W//2-config.offset:] - len(uni_prompting.text_tokenizer)
            image_up_part = extrapolation_image_tokens[:, :-(W//2-config.offset), :] - len(uni_prompting.text_tokenizer)
            image_down_part = extrapolation_image_tokens[:, W//2-config.offset:, :] - len(uni_prompting.text_tokenizer)

            if direction in ['left', 'right']:
                extrapolation_mask = torch.zeros((config.training.batch_size,
                                                  config.dataset.params.resolution // 16,
                                                  config.dataset.params.resolution // 16 // 2 + config.offset),
                                                 dtype=torch.int64, device=device) + mask_token_id
            else:
                extrapolation_mask = torch.zeros((config.training.batch_size,
                                                  config.dataset.params.resolution // 16 // 2 + config.offset,
                                                  config.dataset.params.resolution // 16),
                                                 dtype=torch.int64, device=device) + mask_token_id

            if direction == 'left':
                extrapolation_image_tokens = torch.cat(
                    [extrapolation_mask, extrapolation_image_tokens[:, :, :W//2-config.offset]], dim=-1)
            elif direction == 'right':
                extrapolation_image_tokens = torch.cat(
                    [extrapolation_image_tokens[:, :, -(W//2-config.offset):], extrapolation_mask], dim=-1)
            elif direction == 'up':
                extrapolation_image_tokens = torch.cat(
                    [extrapolation_mask, extrapolation_image_tokens[:, :W // 2 - config.offset, :]], dim=-2)
            else:
                extrapolation_image_tokens = torch.cat(
                    [extrapolation_image_tokens[:, -(W // 2 - config.offset):, :], extrapolation_mask], dim=-2)

            extrapolation_image_tokens = extrapolation_image_tokens.reshape(config.training.batch_size, -1)

            input_ids, _ = uni_prompting((prt, extrapolation_image_tokens), 't2i_gen')

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(prt), extrapolation_image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                attention_mask = attention_mask.to(mask_dtype)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                attention_mask = attention_mask.to(mask_dtype)
                uncond_input_ids = None

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = generate_fn(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.neobabel.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.neobabel.codebook_size - 1, min=0)
            gen_token_ids = gen_token_ids.reshape(config.training.batch_size,
                                                  config.dataset.params.resolution // 16,
                                                  config.dataset.params.resolution // 16)
            if direction == 'left':
                gen_token_ids = torch.cat([gen_token_ids, image_right_part], dim=-1)
            elif direction == 'right':
                gen_token_ids = torch.cat([image_left_part, gen_token_ids], dim=-1)
            elif direction == 'up':
                gen_token_ids = torch.cat([gen_token_ids, image_down_part], dim=-2)
            else:
                gen_token_ids = torch.cat([image_up_part, gen_token_ids], dim=-2)

        _, h, w = gen_token_ids.shape
        gen_token_ids = gen_token_ids.reshape(config.training.batch_size, -1)
        images = vq_model.decode_code(gen_token_ids, shape=(h, w))

        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]

        wandb_images = [wandb.Image(image, caption=' '.join(prompt)) for i, image in enumerate(pil_images)]
        wandb.log({"generated_images": wandb_images}, step=0)

    elif config.mode == 't2i':
        import time
        with open(config.dataset.params.validation_prompts_file, "r") as f:
            validation_prompts = f.read().splitlines()

        for step in tqdm(range(0, len(validation_prompts), config.training.batch_size)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_start_time = time.perf_counter()
            prompts = validation_prompts[step:step + config.training.batch_size]

            image_tokens = torch.ones((len(prompts), config.model.neobabel.num_vq_tokens),
                                      dtype=torch.long, device=device) * mask_token_id

            input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')

            if config.training.guidance_scale > 0:
                uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
                attention_mask = create_attention_mask_predict_next(torch.cat([input_ids, uncond_input_ids], dim=0),
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                attention_mask = attention_mask.to(mask_dtype)
            else:
                attention_mask = create_attention_mask_predict_next(input_ids,
                                                                    pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                                                                    soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                                                                    eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                                                                    rm_pad_in_image=True)
                attention_mask = attention_mask.to(mask_dtype)
                uncond_input_ids = None

            if config.get("mask_schedule", None) is not None:
                schedule = config.mask_schedule.schedule
                args = config.mask_schedule.get("params", {})
                mask_schedule = get_mask_chedule(schedule, **args)
            else:
                mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

            with torch.no_grad():
                gen_token_ids = generate_fn(
                    input_ids=input_ids,
                    uncond_input_ids=uncond_input_ids,
                    attention_mask=attention_mask,
                    guidance_scale=config.training.guidance_scale,
                    temperature=config.training.get("generation_temperature", 1.0),
                    timesteps=config.training.generation_timesteps,
                    noise_schedule=mask_schedule,
                    noise_type=config.training.get("noise_type", "mask"),
                    seq_len=config.model.neobabel.num_vq_tokens,
                    uni_prompting=uni_prompting,
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.neobabel.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_time = time.perf_counter() - batch_start_time
            print(f"batch of {len(prompts)}: {batch_time:.2f}s ({batch_time / len(prompts):.2f}s/image)")

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_images = [Image.fromarray(image) for image in images]

            wandb_images = [wandb.Image(image, caption=prompts[i]) for i, image in enumerate(pil_images)]
            wandb.log({"generated_images": wandb_images}, step=step)

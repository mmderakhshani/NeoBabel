import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from models import NeoBabel, MAGVITv2, get_mask_chedule
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next
from training.utils import get_config
from transformers import AutoTokenizer
from accelerate.utils import set_seed

def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    else:
        raise ValueError(f"model_type {model_type} not supported.")


def main(config, shard_id=0, num_shards=1):
    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    # Enable TF32 on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    precision = config.get("precision", "fp32")  # Default to fp32 if not specified

    # Load all prompts
    with open(config.experiment.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Shard the dataset
    total_samples = len(metadatas)
    shard_size = (total_samples + num_shards - 1) // num_shards  # Divide evenly
    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, total_samples)
    metadatas = metadatas[start_idx:end_idx]  # Each GPU gets a unique slice

    # Assign GPU based on shard_id
    device = torch.device(f"cuda:{shard_id % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    
    print(f"Shard {shard_id}: Processing {len(metadatas)} prompts on {device}")

    # Load models and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.neobabel.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                     special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                     ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)
    vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()

    # Load NeoBabel model
    model = NeoBabel(**config.model.neobabel).to(device)
    assert config.model.neobabel.checkpoint_path is not None

    # Set model precision
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
        print("Using BF16 precision")
    else:
        print(f"Using {precision} precision")

    # Load checkpoint
    if os.path.exists(os.path.join(config.model.neobabel.checkpoint_path, "unwrapped_model")):
        checkpoint_path = os.path.join(config.model.neobabel.checkpoint_path, "unwrapped_model", "pytorch_model.bin")
    else:
        checkpoint_path = os.path.join(config.model.neobabel.checkpoint_path, "pytorch_model.bin")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()

    mask_token_id = model.config.mask_token_id

    generator = torch.Generator(device=device)
    generator.manual_seed(config.training.seed)

    # Get the mask_dtype directly after loading the model to ensure it matches
    if hasattr(model, 'module'):
        mask_dtype = model.module.neobabel.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.neobabel.model.embed_tokens.weight.dtype

    # Use precision context manager for forward passes
    def run_with_precision(func, *args, **kwargs):
        if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                return func(*args, **kwargs)
        elif precision == "fp16" and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    # Process prompts in batches within each shard
    batch_size = config.training.batch_size
    for batch_start in tqdm(range(0, len(metadatas), batch_size), desc=f"GPU {shard_id} Processing batches"):
        generator = torch.Generator(device=device)
        generator.manual_seed(config.training.seed)
        batch_end = min(batch_start + batch_size, len(metadatas))
        batch_metadatas = metadatas[batch_start:batch_end]

        # Keep metadata, prompt and output path aligned: only generate for
        # samples whose output doesn't exist yet (safe on resumed runs).
        pending = []
        for i, metadata in enumerate(batch_metadatas):
            global_idx = start_idx + batch_start + i
            outpath = os.path.join(config.experiment.output_dir, f"{global_idx:0>5}")
            sample_path = os.path.join(outpath, "samples")

            # Skip if output already exists
            if os.path.exists(os.path.join(sample_path, "00000.png")):
                continue

            os.makedirs(sample_path, exist_ok=True)
            pending.append((metadata, outpath))

        if not pending:  # Skip if all outputs exist
            continue

        batch_metadatas = [m for m, _ in pending]
        outpaths = [o for _, o in pending]
        prompts = [m['prompt'] for m in batch_metadatas]

        print(f"GPU {shard_id} - Processing batch with prompts: {prompts}")

        image_tokens = torch.ones((len(prompts), config.model.neobabel.num_vq_tokens), 
                                dtype=torch.long, device=device) * mask_token_id
        input_ids, _ = uni_prompting((prompts, image_tokens), 't2i_gen')

        if config.training.guidance_scale > 0:
            uncond_input_ids, _ = uni_prompting(([''] * len(prompts), image_tokens), 't2i_gen')
            # Match the attention mask creation with inference_t2i_original2.py
            attention_mask = create_attention_mask_predict_next(
                torch.cat([input_ids, uncond_input_ids], dim=0),
                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True
            ).to(mask_dtype)
        else:
            attention_mask = create_attention_mask_predict_next(
                input_ids,
                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True
            ).to(mask_dtype)
            uncond_input_ids = None

        if config.get("mask_schedule", None) is not None:
            schedule = config.mask_schedule.schedule
            args = config.mask_schedule.get("params", {})
            mask_schedule = get_mask_chedule(schedule, **args)
        else:
            mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))

        with torch.no_grad():
            gen_token_ids = run_with_precision(
                model.t2i_generate,
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
                generator=generator,
            )

        gen_token_ids = torch.clamp(gen_token_ids, max=config.model.neobabel.codebook_size - 1, min=0)
        images = vq_model.decode_code(gen_token_ids)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        
        # Save images and metadata
        for idx, (metadata, image) in enumerate(zip(batch_metadatas, images)):
            outpath = outpaths[idx]
            sample_path = os.path.join(outpath, "samples")
            
            # Save image
            pil_image = Image.fromarray(image)
            pil_image.save(os.path.join(sample_path, "00000.png"))
            
            # Save metadata
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp, ensure_ascii=False)

    print(f"GPU {shard_id} - Done.")

if __name__ == "__main__":
    config = get_config()

    num_shards = torch.cuda.device_count()  # Number of GPUs available
    shard_id = int(os.environ.get("LOCAL_RANK", 0))  # Provided by torchrun or manually set

    main(config, shard_id, num_shards)
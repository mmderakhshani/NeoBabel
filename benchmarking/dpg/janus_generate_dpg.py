import os
import json
import torch
from tqdm import tqdm
import numpy as np
import PIL.Image

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "true"
from accelerate.utils import set_seed
import glob

from omegaconf import OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

# Grid and save images for each prompt
def make_grid(images, grid_size=(2,2)):
    w, h = images[0].size
    grid_img = PIL.Image.new('RGB', (w * grid_size[1], h * grid_size[0]))
    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        grid_img.paste(img, (col * w, row * h))
    return grid_img


@torch.inference_mode()
def generate_images_for_batch(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompts: list,
    temperature: float = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    # Tokenize and LEFT-pad prompts so every row ends with its image-start tag
    # and generation starts at the same position for the whole batch.
    input_ids_list = [vl_chat_processor.tokenizer.encode(p) for p in prompts]
    max_len = max(len(ids) for ids in input_ids_list)
    batch_size = len(prompts)
    tokens = torch.full((batch_size*2, max_len), vl_chat_processor.pad_id, dtype=torch.int).cuda()
    for i, ids in enumerate(input_ids_list):
        ids_t = torch.tensor(ids)
        tokens[i*2, -len(ids):] = ids_t
        tokens[i*2+1, -len(ids):] = ids_t
        # unconditional row: keep BOS and the trailing image-start tag, blank the text
        tokens[i*2+1, -(len(ids)-1):-1] = vl_chat_processor.pad_id

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((batch_size, image_token_num_per_image), dtype=torch.int).cuda()

    outputs = None
    for i in range(image_token_num_per_image):
        outputs = mmgpt.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=outputs.past_key_values if outputs is not None else None
        )
        hidden_states = outputs.last_hidden_state

        logits = mmgpt.gen_head(hidden_states[:, -1, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]

        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)
        generated_tokens[:, i] = next_token.squeeze(dim=-1)

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)
        inputs_embeds = img_embeds.unsqueeze(dim=1)

    dec = mmgpt.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[batch_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)
    visual_img = np.zeros((batch_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    pil_images = [PIL.Image.fromarray(visual_img[i]) for i in range(batch_size)]
    return pil_images

def main(config, shard_id=0, num_shards=1):
    # If passed along, set the seed now.
    if config.experiment.seed is not None:
        set_seed(config.experiment.seed)

    # Load all prompts
    prompt_files = glob.glob(os.path.join(config.experiment.metadata_file, "*.txt"))
    metadatas = [
        {"prompt": open(pf).read().strip(), "file": pf}
        for pf in prompt_files
    ] 

    # Shard the dataset
    total_samples = len(metadatas)
    shard_size = (total_samples + num_shards - 1) // num_shards
    start_idx = shard_id * shard_size
    end_idx = min(start_idx + shard_size, total_samples)
    metadatas = metadatas[start_idx:end_idx]

    # Assign GPU based on shard_id
    device = torch.device(f"cuda:{shard_id % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    print(f"Shard {shard_id}: Processing {len(metadatas)} prompts on {device}")

    # Load Janus model and processor ONCE
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    batch_size = 2  # or whatever fits your GPU
    prompts_per_sample = 4  # Always generate 4 images per prompt
    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)

    def prepare_prompts(prompts):
        """Format prompts for the model."""
        formatted = []
        for prompt in prompts:
            conversation = [
                {"role": "User", "content": prompt},
                {"role": "Assistant", "content": ""},
            ]
            sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            full_prompt = sft_format + vl_chat_processor.image_start_tag
            formatted.append(full_prompt)
        return formatted

    # for batch_start in range(0, len(metadatas), batch_size):
    for batch_start in tqdm(range(0, len(metadatas), batch_size), desc=f"GPU {shard_id} Processing batches"):
        batch_end = min(batch_start + batch_size, len(metadatas))
        batch_metadatas = metadatas[batch_start:batch_end]
        prompts = [m['prompt'] for m in batch_metadatas]
        prompt_files = [os.path.basename(m['file']) for m in batch_metadatas]
        prompt_names = [os.path.splitext(f)[0] for f in prompt_files]

        # Repeat each prompt 4 times for generation
        repeated_prompts = [p for p in prompts for _ in range(prompts_per_sample)]

        if not repeated_prompts:
            continue
        
        batch_prompts = prepare_prompts(repeated_prompts)
        pil_images = generate_images_for_batch(
            vl_gpt,
            vl_chat_processor,
            batch_prompts,
        )

        for i, prompt_name in enumerate(prompt_names):
            imgs = [pil_images[i * prompts_per_sample + j] for j in range(prompts_per_sample)]
            grid_img = make_grid(imgs, grid_size=(2,2))
            out_file = os.path.join(output_dir, f"{prompt_name}.png")
            grid_img.save(out_file)
    print(f"GPU {shard_id} - Done.")



if __name__ == "__main__":
    config = get_config()

    num_shards = torch.cuda.device_count()
    shard_id = int(os.environ.get("LOCAL_RANK", 0))

    main(config, shard_id, num_shards)

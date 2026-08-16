# This work is based in part on code from Show-o (https://github.com/showlab/Show-o).
# Significant modifications for NeoBabel.

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk

class NeoBabel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            vocab_size,
            llm_vocab_size,
            llm_model_path='',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_huggingface=True,
            attn_implementation="eager",
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        # default is eager attention: gemma-2 logit soft-capping is silently disabled
        # under sdpa. "flex_attention" (torch >= 2.5) supports both the soft-cap and
        # this model's custom prefix-causal/image-bidirectional mask.
        if load_from_huggingface:
            config = AutoConfig.from_pretrained(llm_model_path)
            self.neobabel = AutoModelForCausalLM.from_config(config, attn_implementation=attn_implementation)
        else:
            self.neobabel = AutoModelForCausalLM.from_pretrained(llm_model_path, attn_implementation=attn_implementation)
            
        try:
            # mean_resizing (transformers >= 4.46) breaks under meta-device init and
            # is pointless here: the added rows are always overwritten by checkpoint
            # weights (from_pretrained) or trained from scratch.
            self.neobabel.resize_token_embeddings(self.vocab_size, mean_resizing=False)
        except TypeError:  # older transformers without the kwarg
            self.neobabel.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            label_smoothing=0.0,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            **kwargs,
    ):

        if input_embeddings is None:
            logits = self.neobabel(input_ids=input_ids, attention_mask=attention_mask)['logits']
        else:
            logits = self.neobabel(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']

        if labels is not None:
            zero_loss = logits.new_zeros(())

            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            if batch_size_t2i > 0:
                loss_t2i = F.cross_entropy(
                    logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                    labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
                )
            else:
                loss_t2i = zero_loss

            # 2. Next token prediction for language modeling
            if batch_size_lm > 0:
                loss_lm = F.cross_entropy(
                    logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                    labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1),
                    ignore_index=-100, label_smoothing=label_smoothing,
                )
            else:
                loss_lm = zero_loss

            # 3. Next token prediction for captioning/multimodal understanding
            # (guard batch_size_mmu == 0: logits[-0:] would select the whole batch)
            if batch_size_mmu > 0:
                loss_mmu = F.cross_entropy(
                    logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                    labels[-batch_size_mmu:, 1:].contiguous().view(-1),
                    ignore_index=-100, label_smoothing=label_smoothing,
                )
            else:
                loss_mmu = zero_loss

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits

    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.neobabel.num_vq_tokens
        num_new_special_tokens = config.model.neobabel.num_new_special_tokens

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.neobabel.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.neobabel.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.neobabel.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.neobabel.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    def t2i_generate_fast(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            sliced_head: bool = True,
            **kwargs,
    ):
        """Same sampling as t2i_generate, but the text-prefix KV is computed once
        and reused across all decoding steps: text tokens attend causally and never
        to the (later) image block, so their KV is independent of the image tokens.
        Each step forwards only the image segment (<|soi|> + image + <|eoi|>).

        sliced_head=True projects hidden states only onto the image-codebook rows
        of the lm_head (the only logits the sampler ever reads) instead of the full
        264k vocabulary. Sampling-equivalent — the sliced GEMM tiles differently in
        bf16, so same-seed tokens can diverge while per-step distributions match
        (mean TV distance ~2e-3 measured); pass sliced_head=False for token-exact
        parity with t2i_generate given the same generator state.
        """
        from transformers.cache_utils import DynamicCache

        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.neobabel.num_vq_tokens
        num_new_special_tokens = config.model.neobabel.num_new_special_tokens
        llm_vocab_size = config.model.neobabel.llm_vocab_size
        token_offset = llm_vocab_size + num_new_special_tokens

        use_cfg = uncond_input_ids is not None and guidance_scale > 0
        model_input_ids = torch.cat([input_ids, uncond_input_ids]) if use_cfg else input_ids

        B2, L = model_input_ids.shape  # B2 = 2*B under CFG
        S = num_vq_tokens + 2          # <|soi|> + image tokens + <|eoi|>
        P = L - S                      # text prefix (pads + task/text tokens)
        assert (model_input_ids[:, P] == model_input_ids[0, P]).all(), \
            "t2i_generate_fast expects the <|soi|> token at the same position in every row"

        embed_dtype = self.neobabel.get_input_embeddings().weight.dtype
        bias_mask = attention_mask.to(embed_dtype)          # (B2, 1, L, L) additive
        suffix_mask = bias_mask[:, :, P:, :].contiguous()   # image rows over full kv

        # 1) prefill: cache the text-prefix KV once
        cache = DynamicCache()
        self.neobabel.model(
            input_ids=model_input_ids[:, :P],
            attention_mask=bias_mask[:, :, :P, :P],
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.arange(P, device=model_input_ids.device),
        )

        suffix_ids = model_input_ids[:, P:].clone()          # (B2, S)
        cache_position = torch.arange(P, L, device=model_input_ids.device)

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - token_offset)

        for step in range(timesteps):
            # keep only the text-prefix KV; last step's image KV is stale
            cache.crop(P)
            if sliced_head:
                out = self.neobabel.model(
                    input_ids=suffix_ids,
                    attention_mask=suffix_mask,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                )
                # image-token rows of the suffix predict the tokens at their own
                # positions (masked-token objective, no next-token shift); project
                # only onto the image-codebook rows of the (tied) lm_head
                hidden = out[0][:, 1:1 + num_vq_tokens]
                step_logits = torch.nn.functional.linear(
                    hidden, self.neobabel.lm_head.weight[token_offset:-1])
                cap = self.neobabel.config.final_logit_softcapping
                if cap is not None:
                    step_logits = step_logits / cap
                    step_logits = torch.tanh(step_logits)
                    step_logits = step_logits * cap
            else:
                out = self.neobabel(
                    input_ids=suffix_ids,
                    attention_mask=suffix_mask,
                    past_key_values=cache,
                    use_cache=True,
                    cache_position=cache_position,
                )
                # image-token rows of the suffix predict the tokens at their own
                # positions (masked-token objective, no next-token shift)
                step_logits = out.logits[:, 1:1 + num_vq_tokens, token_offset:-1]
            if use_cfg:
                cond_logits, uncond_logits = step_logits.chunk(2)
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
            else:
                logits = step_logits

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # NOTE: matches t2i_generate exactly — temperature compounds across steps
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)

            new_image_tokens = torch.where(masking, mask_token_id, sampled_ids + token_offset)
            if use_cfg:
                suffix_ids[:, 1:1 + num_vq_tokens] = new_image_tokens.repeat(2, 1)
            else:
                suffix_ids[:, 1:1 + num_vq_tokens] = new_image_tokens
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids

    @torch.no_grad()
    def t2i_generate_grpo(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=20,
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        """Mask-GRPO rollout (arXiv:2510.13418): standard MaskGiT sampling that
        additionally records, per step, the newly COMMITTED positions/token ids,
        the re-masked positions, and min(cs_t) — everything needed to replay the
        trajectory and compute transition log-probs under an updated policy.

        Batched over the whole rollout group: with exact-k re-masking the number
        of committed tokens per step is identical across rows, so records are
        dense (G, k_t) tensors. Rollouts use guidance_scale=0 (matching training).
        """
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.neobabel.num_vq_tokens
        num_new_special_tokens = config.model.neobabel.num_new_special_tokens
        token_offset = config.model.neobabel.llm_vocab_size + num_new_special_tokens

        input_ids = input_ids.clone()
        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - token_offset)
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        records = {"committed_idx": [], "committed_ids": [], "min_probs": [], "masked_idx": []}
        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
            else:
                logits = self(input_ids, attention_mask=attention_mask)
            logits = logits[:, -(num_vq_tokens + 1):-1, token_offset:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)

            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None]).squeeze(-1)
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            temperature = temperature * (1.0 - ratio)
            masking, min_probs = mask_by_random_topk(mask_len, selected_probs, temperature,
                                                     generator=generator, return_min=True)

            committed = unknown_map & (~masking)
            # exact-k masking + equal unknown counts => equal committed counts per row
            counts = committed.sum(dim=-1)
            assert (counts == counts[0]).all(), "uneven commit counts across group"
            k = int(counts[0])
            committed_idx = torch.nonzero(committed, as_tuple=False)[:, 1].view(-1, k)
            committed_ids = sampled_ids[committed].view(-1, k)
            masked_idx = torch.nonzero(masking, as_tuple=False)[:, 1].view(masking.shape[0], -1)
            records["committed_idx"].append(committed_idx)
            records["committed_ids"].append(committed_ids)
            records["min_probs"].append(min_probs)
            records["masked_idx"].append(masked_idx)

            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                                sampled_ids + token_offset)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return sampled_ids, records

    def t2i_grpo_step_loss(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask=None,
            timesteps=20,
            config=None,
            records=None,
            timestep=None,
            advantages=None,
            clip_low=0.9,
            clip_high=1.2,
            kl_beta=0.0,
            candidate=2,
    ):
        """One Mask-GRPO policy-gradient step at trajectory position `timestep`.

        Recomputes the transition log-prob of the recorded commits under the
        current policy, forms the (group-relative) REINFORCE surrogate with
        asymmetric clipping, and returns (loss, next-step input_ids) so the
        caller can replay the trajectory state by state.
        """
        mask_token_id = self.config.mask_token_id
        num_vq_tokens = config.model.neobabel.num_vq_tokens
        num_new_special_tokens = config.model.neobabel.num_new_special_tokens
        token_offset = config.model.neobabel.llm_vocab_size + num_new_special_tokens
        num_prefix_tokens = input_ids.shape[1] - num_vq_tokens - 1

        logits = self(input_ids, attention_mask=attention_mask)
        logits = logits[:, -(num_vq_tokens + 1):-1, token_offset:-1]
        probs = logits.softmax(dim=-1)

        committed_idx = records["committed_idx"][timestep]      # (G, k)
        committed_ids = records["committed_ids"][timestep]      # (G, k)
        batch_idx = torch.arange(probs.shape[0], device=probs.device)[:, None]
        selected_probs = probs[batch_idx, committed_idx, committed_ids]
        sum_log_probs = torch.log(selected_probs.clamp_min(1e-10)).sum(dim=1)

        if candidate == 1:
            # add the probability mass of below-cutoff tokens at re-masked positions
            masked_idx = records["masked_idx"][timestep]        # (G, m)
            min_probs = records["min_probs"][timestep].view(-1, 1, 1)
            masked_probs = probs[batch_idx, masked_idx, :]
            low_mass = (masked_probs * (masked_probs < min_probs)).sum(dim=-1)
            sum_log_probs = sum_log_probs + torch.log(low_mass.clamp_min(1e-10)).sum(dim=1)

        ratio = torch.exp(sum_log_probs - sum_log_probs.detach())
        surrogate = advantages * ratio
        surrogate_clipped = advantages * torch.clamp(ratio, clip_low, clip_high)
        loss = -torch.min(surrogate.mean(), surrogate_clipped.mean()) / timesteps
        if kl_beta != 0:
            kl = torch.exp(sum_log_probs.detach() - sum_log_probs) \
                 - (sum_log_probs.detach() - sum_log_probs) - 1
            loss = loss + kl_beta * kl.mean() / timesteps

        # replay: commit the recorded tokens to build the next state
        new_input_ids = input_ids.detach().clone()
        target_positions = committed_idx + num_prefix_tokens
        batch_idx2 = batch_idx.expand(-1, committed_idx.shape[1])
        new_input_ids[batch_idx2, target_positions] = (committed_ids + token_offset).to(new_input_ids.dtype)
        return loss, new_input_ids

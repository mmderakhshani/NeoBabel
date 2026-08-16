# Build a GENERAL Mask-GRPO prompt pool aimed at the diagnosed m-DPG/GenEval
# weaknesses: entity coverage and counting. Internal instruction-tuning subsets
# only, with the count/multi-object sources (Geneval_train, object_*) heavily
# overweighted and the dense-caption sources (Journey, dalle3) next.
#
# Same schema as the cultural pools: every line carries the language-specific
# prompt plus its parallel English caption as reward_prompt; a held-out slice
# (cross-subset) is written for early stopping.
#
# Usage:
#   python training/build_general_grpo_prompts.py \
#       --internal-root /path/instruction_tuning_maya --out-dir /path/out --seed 42
import argparse
import json
import os
import random

# subset -> sampled keys per subset (train), holdout keys per subset
SUBSET_WEIGHTS = {
    "Geneval_train": (600, 10),   # GenEval-style counting / colors / positions
    "object_1": (300, 5),
    "object_2": (300, 5),
    "Journey": (300, 5),          # dense JourneyDB-style captions
    "dalle3": (300, 5),           # dense captions
    "MSCOCO_human": (200, 0),
    "Human_gestures": (150, 0),
    "occupation_1": (150, 0),
    "occupation_2": (150, 0),
    "text_1": (100, 0),
    "text_2": (100, 0),
}


def english_key(data):
    return next(k for k in data if k.lower().startswith("english"))


def emit(f, data, keys, source):
    english = data[english_key(data)]
    n = 0
    for key in keys:
        for lang, captions in data.items():
            f.write(json.dumps({
                "prompt": captions[key],
                "reward_prompt": english[key],
                "lang": lang,
                "uuid": key,
                "source": source,
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--internal-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.out_dir, "general_grpo_train.jsonl")
    eval_path = os.path.join(args.out_dir, "general_grpo_eval.jsonl")
    n_train = n_eval = 0
    hold_entries = []  # (data, key, subset) — interleaved so eval[:N] mixes subsets
    with open(train_path, "w") as ftr:
        for subset, (n_keys, n_hold) in SUBSET_WEIGHTS.items():
            with open(os.path.join(args.internal_root, f"{subset}.json")) as g:
                data = json.load(g)
            # sample from the cross-language key intersection (dalle3 has an
            # English-only key; same guard as build_mixed_grpo_prompts)
            keys = set(data[english_key(data)])
            for v in data.values():
                keys &= set(v)
            keys = sorted(keys)
            rng.shuffle(keys)
            hold, train = keys[:n_hold], keys[n_hold:n_hold + n_keys]
            n_train += emit(ftr, data, train, subset)
            hold_entries += [(data, k, subset) for k in hold]
            print(f"{subset}: {len(train)} train keys, {len(hold)} holdout keys")
    rng.shuffle(hold_entries)
    with open(eval_path, "w") as fev:
        for data, key, subset in hold_entries:
            n_eval += emit(fev, data, [key], subset)
    print(f"train prompts: {n_train} -> {train_path}")
    print(f"eval prompts:  {n_eval} -> {eval_path}")


if __name__ == "__main__":
    main()

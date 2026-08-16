# Build a mixed Mask-GRPO prompt pool: cultural (geocult) prompts plus a
# diversity sample from the instruction_tuning_maya "internal" caption sets.
# All sources share the same schema — language -> {key: caption}, parallel
# across languages — so every prompt carries its parallel English caption as
# reward_prompt (ImageReward is English-only).
#
# Usage:
#   python training/build_mixed_grpo_prompts.py \
#       --geocult /path/geocult.json --internal-root /path/instruction_tuning_maya \
#       --out-dir /path/out --holdout 25 --internal-per-subset 90 --seed 42
import argparse
import json
import os
import random

INTERNAL_SUBSETS = ["dalle3", "Geneval_train", "Human_gestures", "Journey", "MSCOCO_human",
                    "object_1", "object_2", "occupation_1", "occupation_2", "text_1", "text_2"]


def english_key(data):
    return next(k for k in data if k.lower().startswith("english"))


def emit(f, data, keys, source, n):
    english = data[english_key(data)]
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
    ap.add_argument("--geocult", required=True)
    ap.add_argument("--internal-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout", type=int, default=25, help="geocult uuids held out for eval")
    ap.add_argument("--internal-per-subset", type=int, default=90)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "culture_mixed_grpo_train.jsonl")
    n = 0
    with open(train_path, "w") as f:
        with open(args.geocult) as g:
            geocult = json.load(g)
        assert all(set(v) == set(geocult[english_key(geocult)]) for v in geocult.values())
        uuids = sorted(geocult[english_key(geocult)])
        rng.shuffle(uuids)
        train_uuids = uuids[args.holdout:]  # same seed/holdout as build_culture_grpo_prompts
        n_cult = n = emit(f, geocult, train_uuids, "geocult", 0)

        for subset in INTERNAL_SUBSETS:
            with open(os.path.join(args.internal_root, f"{subset}.json")) as g:
                data = json.load(g)
            common = set(data[english_key(data)])
            for v in data.values():
                common &= set(v)   # e.g. dalle3 has one English-only key
            keys = sorted(common)
            rng.shuffle(keys)
            n = emit(f, data, keys[:args.internal_per_subset], subset, n)

    print(f"train: {n} prompts ({n_cult} cultural + {n - n_cult} internal, "
          f"{len(geocult)} languages) -> {train_path}")
    print("eval: reuse culture_grpo_eval.jsonl (same seed/holdout split)")


if __name__ == "__main__":
    main()

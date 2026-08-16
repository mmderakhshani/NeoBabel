# Build Mask-GRPO prompt files from the geocult cultural caption set.
#
# geocult.json maps language -> {uuid: caption}, with the same uuids in every
# language (parallel captions). Rollouts use the language-specific caption;
# the reward model (ImageReward/CLIP, English-only) scores the parallel
# English caption of the same uuid.
#
# Usage:
#   python training/build_culture_grpo_prompts.py \
#       --geocult /path/to/geocult.json --out-dir /path/to/out \
#       --holdout 25 --seed 42
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geocult", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout", type=int, default=25, help="uuids held out for eval (all languages)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.geocult) as f:
        data = json.load(f)
    english_key = next(k for k in data if k.lower().startswith("english"))
    english = data[english_key]

    uuids = sorted(english)
    assert all(set(v) == set(english) for v in data.values()), "languages disagree on uuids"
    random.Random(args.seed).shuffle(uuids)
    eval_uuids, train_uuids = set(uuids[:args.holdout]), uuids[args.holdout:]

    os.makedirs(args.out_dir, exist_ok=True)
    counts = {}
    for split, split_uuids in (("train", train_uuids), ("eval", sorted(eval_uuids))):
        path = os.path.join(args.out_dir, f"culture_grpo_{split}.jsonl")
        n = 0
        with open(path, "w") as f:
            for uuid in split_uuids:
                for lang, captions in data.items():
                    f.write(json.dumps({
                        "prompt": captions[uuid],
                        "reward_prompt": english[uuid],
                        "lang": lang,
                        "uuid": uuid,
                    }, ensure_ascii=False) + "\n")
                    n += 1
        counts[split] = (path, n)
    for split, (path, n) in counts.items():
        print(f"{split}: {n} prompts ({n // len(data)} uuids x {len(data)} languages) -> {path}")


if __name__ == "__main__":
    main()

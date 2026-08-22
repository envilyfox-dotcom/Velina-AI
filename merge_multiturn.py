"""
Merges velina_multiturn.jsonl into your existing train.jsonl.

Usage:
    python merge_multiturn.py train.jsonl velina_multiturn.jsonl
"""

import sys
import random

def merge(train_path: str, multiturn_path: str):
    with open(train_path, encoding="utf-8") as f:
        train_lines = [line.strip() for line in f if line.strip()]

    with open(multiturn_path, encoding="utf-8") as f:
        multiturn_lines = [line.strip() for line in f if line.strip()]

    combined = train_lines + multiturn_lines
    random.seed(42)
    random.shuffle(combined)

    with open(train_path, "w", encoding="utf-8") as f:
        for line in combined:
            f.write(line + "\n")

    print(f"Original train examples: {len(train_lines)}")
    print(f"Multi-turn examples added: {len(multiturn_lines)}")
    print(f"New total in {train_path}: {len(combined)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python merge_multiturn.py train.jsonl velina_multiturn.jsonl")
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])

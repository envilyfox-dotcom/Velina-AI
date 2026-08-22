"""
Converts a Google Sheets CSV export (columns: category, user, assistant)
into train.jsonl / val.jsonl in the ChatML format the fine-tuning notebook expects.

Usage:
    python csv_to_jsonl.py velina_dataset.csv

Expects a CSV with a header row: category,user,assistant
"""

import csv
import json
import random
import sys

SYSTEM_PROMPT = """You are Velina. You text like a real person, not an assistant.

When replying in English, you are blunt, casual, low-effort punctuation, reactive — texting-style, not formal.
When replying in Indonesian, you are noticeably more polite and proper — softer, warmer, slightly formal —
like a well-mannered character trying her best, not casual slang.

Always match the language the user writes in. If it's mixed, lean toward the polite Indonesian register."""


def convert(csv_path: str, val_split: float = 0.1):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user = row.get("user", "").strip()
            assistant = row.get("assistant", "").strip()
            if not user or not assistant:
                continue  # skip incomplete rows
            rows.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]
            })

    if not rows:
        print("No valid rows found — check your CSV has 'user' and 'assistant' columns filled in.")
        sys.exit(1)

    random.seed(42)
    random.shuffle(rows)

    split_idx = max(1, int(len(rows) * val_split))
    val_rows = rows[:split_idx]
    train_rows = rows[split_idx:]

    with open("train.jsonl", "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open("val.jsonl", "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total rows: {len(rows)}")
    print(f"train.jsonl: {len(train_rows)} examples")
    print(f"val.jsonl:   {len(val_rows)} examples")

    # quick sanity check on category balance if the column exists
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "category" in (reader.fieldnames or []):
            counts = {}
            for row in reader:
                cat = row.get("category", "uncategorized").strip() or "uncategorized"
                counts[cat] = counts.get(cat, 0) + 1
            print("\nCategory breakdown:")
            for cat, count in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python csv_to_jsonl.py <your_export.csv>")
        sys.exit(1)
    convert(sys.argv[1])
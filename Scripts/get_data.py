from datasets import load_dataset
from pathlib import Path
import json


OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "reddit_raw.jsonl"


print("Streaming Reddit dataset...")

dataset = load_dataset(
    "sentence-transformers/reddit-title-body",
    split="train",
    streaming=True
)


NUM_SAMPLES = 10

with OUTPUT_PATH.open("w", encoding="utf-8") as f:

    for i, row in enumerate(dataset):

        if i >= NUM_SAMPLES:
            break

        f.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

        print(f"Fetched sample {i + 1}")


print(f"\nSaved {NUM_SAMPLES} samples")
print(f"Output: {OUTPUT_PATH}")
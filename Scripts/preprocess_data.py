from pathlib import Path
import json
import re

INPUT_DIR = Path(__file__).parent.parent / "data/raw"
OUTPUT_DIR = Path(__file__).parent.parent / "data/processed"
OUTPUT_DIR.mkdir(exist_ok=True)

MIN_TITLE_LENGTH = 10
MIN_BODY_LENGTH = 50
MAX_BODY_LENGTH = 1000

def clean_text(text):
    if not text:
        return ""

    text = str(text)

    text = re.sub(r"http\S+|www\.\S+", "", text)  # Remove URLs
    # Replace tabs/newlines with spaces
    text = re.sub(r"[\n\r\t]+", " ", text)

    # Remove repeated spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def is_valid_record(title: str, body: str) -> bool:
    """
    Decide whether a Reddit record is useful for narrative-case generation.
    """

    if not title or not body:
        return False

    if len(title) < MIN_TITLE_LENGTH:
        return False

    if len(body) < MIN_BODY_LENGTH:
        return False

    if len(body) > MAX_BODY_LENGTH:
        return False

    # Common Reddit placeholders
    invalid_values = {
        "[deleted]",
        "[removed]",
        "deleted",
        "removed",
    }

    if title.lower().strip() in invalid_values:
        return False

    if body.lower().strip() in invalid_values:
        return False

    return True

def preprocess():
    INPUT_PATH = INPUT_DIR / "reddit_raw.jsonl"
    OUTPUT_PATH = OUTPUT_DIR / "reddit_title_body_processed.jsonl"
    # OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Run fetch_data.py first."
        )

    total_records = 0
    valid_records = 0
    duplicate_records = 0

    seen_text = set()

    with INPUT_PATH.open("r", encoding="utf-8") as infile, \
            OUTPUT_PATH.open("w", encoding="utf-8") as outfile:

        for line in infile:
            total_records += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Change these field names if your dataset uses different columns
            title = clean_text(record.get("title", ""))
            body = clean_text(record.get("body", ""))
            subreddit = clean_text(record.get("subreddit", ""))

            if not is_valid_record(title, body):
                continue

            # Combine title + body for duplicate detection
            duplicate_key = f"{title.lower()}::{body.lower()}"

            if duplicate_key in seen_text:
                duplicate_records += 1
                continue

            seen_text.add(duplicate_key)

            processed_record = {
                "source": "reddit",
                "title": title,
                "body": body,
                "subreddit": subreddit,
                "text": f"{title}. {body}",
                "title_length": len(title),
                "body_length": len(body),
            }

            outfile.write(
                json.dumps(processed_record, ensure_ascii=False) + "\n"
            )

            valid_records += 1

    print("=" * 50)
    print("Preprocessing completed")
    print("=" * 50)
    print(f"Input records      : {total_records}")
    print(f"Valid records      : {valid_records}")
    print(f"Duplicates removed : {duplicate_records}")
    print(f"Output file        : {OUTPUT_PATH}")


preprocess()
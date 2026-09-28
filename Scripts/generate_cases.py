from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "reddit_title_body_processed.jsonl"
OUTPUT_PATH = BASE_DIR / "data" / "cases" / "reddit_cases.jsonl"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def build_case(record, index):
    return {
        "case_id": f"reddit_{index:05d}",
        "source": record.get("source", "reddit"),
        "subreddit": record.get("subreddit", ""),
        "domain": record.get("domain", "other"),
        "narrative": record.get("text", ""),
        "question": (
            "Analyze the situation and identify the most important "
            "factors that should influence the decision."
        ),
        "reference_answer": "",
        "reasoning_type": "general_reasoning",
        "difficulty": 1
    }


count = 0

with INPUT_PATH.open("r", encoding="utf-8") as infile, \
        OUTPUT_PATH.open("w", encoding="utf-8") as outfile:

    for index, line in enumerate(infile, start=1):
        record = json.loads(line)

        case = build_case(record, index)

        outfile.write(
            json.dumps(case, ensure_ascii=False) + "\n"
        )

        count += 1

print(f"Generated {count} cases")
print(f"Saved to: {OUTPUT_PATH}")
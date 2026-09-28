import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = BASE_DIR / "data" / "cases" / "reddit_cases_qa.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "results" / "model_outputs.jsonl"


CANDIDATE_MODELS = {
    "gpt_oss_20b": {
        "model": "openai/gpt-oss-20b",
        "temperature": 0.2,
        "max_completion_tokens": 800,
        "extra_params": {
            "include_reasoning": False,
        },
    },
    "qwen_3_8_27b": {
        "model": "qwen/qwen3.8-27b",
        "temperature": 0.2,
        "max_completion_tokens": 800,
        "extra_params": {
            "reasoning_effort": "none",
        },
    },
}

PROMPT_VERSION = "candidate_v1"
MAX_RETRIES = 3

SYSTEM_PROMPT = """
You are answering a benchmark question using a provided narrative.

Rules:
- Use only information contained in or directly supported by the narrative.
- Do not use outside knowledge.
- Do not invent names, dates, numbers, events, motivations, or background facts.
- Answer the question directly.
- Explain the relevant reasoning clearly and concisely.
- Do not mention benchmark construction, grading, reference answers, or evaluation.
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run candidate LLMs on benchmark cases and store their responses."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CANDIDATE_MODELS.keys()),
        default=list(CANDIDATE_MODELS.keys()),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing output before running.",
    )
    return parser.parse_args()


def get_client():
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found.\n"
            "Create a .env file in the project root with:\n"
            "GROQ_API_KEY=your_api_key_here"
        )

    return Groq(api_key=api_key)


def validate_case(record):
    case_id = str(record.get("case_id", "")).strip()
    narrative = str(record.get("narrative", "")).strip()
    question = str(record.get("question", "")).strip()

    if not case_id:
        raise ValueError("Missing case_id.")
    if not narrative:
        raise ValueError(f"{case_id}: narrative is empty.")
    if not question:
        raise ValueError(f"{case_id}: question is empty.")

    return case_id, narrative, question


def load_completed_pairs(output_path):
    completed = set()
    attempt_counts = {}

    if not output_path.exists():
        return completed, attempt_counts

    with output_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            case_id = str(record.get("case_id", "")).strip()
            model = str(record.get("candidate_model", "")).strip()

            if not case_id or not model:
                continue

            key = (case_id, model)
            attempt_counts[key] = max(
                attempt_counts.get(key, 0),
                int(record.get("attempt", 1)),
            )

            if record.get("status") == "success":
                completed.add(key)

    return completed, attempt_counts


def call_candidate_model(client, config, narrative, question):
    user_prompt = f"""
NARRATIVE:
{narrative}

QUESTION:
{question}

Provide your answer using only the narrative above.
""".strip()

    last_error = None

    for retry in range(1, MAX_RETRIES + 1):
        try:
            start = time.perf_counter()

            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=config["temperature"],
                max_completion_tokens=config["max_completion_tokens"],
                stream=False,
                **config.get("extra_params", {}),
            )

            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            content = response.choices[0].message.content

            if not content or not content.strip():
                raise ValueError("Model returned an empty response.")

            return content.strip(), latency_ms

        except Exception as exc:
            last_error = exc

            if retry == MAX_RETRIES:
                break

            wait_seconds = 2 ** (retry - 1)
            print(
                f"      API attempt {retry}/{MAX_RETRIES} failed: {exc}"
            )
            print(f"      Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Model request failed after {MAX_RETRIES} attempts: {last_error}"
    )


def write_record(outfile, record):
    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    outfile.flush()


def main():
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}\n"
            "Run generate_qa.py first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite and output_path.exists():
        output_path.unlink()

    client = get_client()
    completed_pairs, attempt_counts = load_completed_pairs(output_path)

    cases_read = 0
    requests_planned = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0

    print("=" * 72)
    print("Candidate model inference")
    print("=" * 72)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Models: {', '.join(args.models)}")
    print()

    with input_path.open("r", encoding="utf-8") as infile, \
         output_path.open("a", encoding="utf-8") as outfile:

        for line_number, line in enumerate(infile, start=1):

            if args.limit is not None and cases_read >= args.limit:
                break

            if not line.strip():
                continue

            try:
                case = json.loads(line)
                case_id, narrative, question = validate_case(case)

            except Exception as exc:
                print(f"[Input line {line_number}] Invalid case: {exc}")
                continue

            cases_read += 1

            for model_alias in args.models:
                config = CANDIDATE_MODELS[model_alias]
                model_id = config["model"]

                requests_planned += 1
                key = (case_id, model_id)

                if key in completed_pairs:
                    skipped_count += 1
                    print(
                        f"[{case_id}] {model_alias}: already completed, skipping."
                    )
                    continue

                attempt = attempt_counts.get(key, 0) + 1

                print(f"[{case_id}] {model_alias}: running...")

                base_record = {
                    "case_id": case_id,
                    "source": case.get("source"),
                    "subreddit": case.get("subreddit"),
                    "domain": case.get("domain"),
                    "candidate_model_alias": model_alias,
                    "candidate_model": model_id,
                    "prompt_version": PROMPT_VERSION,
                    "temperature": config["temperature"],
                    "max_completion_tokens": config["max_completion_tokens"],
                    "attempt": attempt,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }

                try:
                    model_response, latency_ms = call_candidate_model(
                        client=client,
                        config=config,
                        narrative=narrative,
                        question=question,
                    )

                    result = {
                        **base_record,
                        "model_response": model_response,
                        "latency_ms": latency_ms,
                        "status": "success",
                        "error": None,
                    }

                    write_record(outfile, result)
                    completed_pairs.add(key)
                    attempt_counts[key] = attempt
                    success_count += 1

                    print(f"    success | {latency_ms} ms")

                except Exception as exc:
                    result = {
                        **base_record,
                        "model_response": None,
                        "latency_ms": None,
                        "status": "failed",
                        "error": str(exc),
                    }

                    write_record(outfile, result)
                    attempt_counts[key] = attempt
                    failed_count += 1

                    print(f"    FAILED | {exc}")

    print()
    print("=" * 72)
    print("Candidate inference completed")
    print("=" * 72)
    print(f"Cases read             : {cases_read}")
    print(f"Case/model combinations: {requests_planned}")
    print(f"New successes          : {success_count}")
    print(f"Failed                 : {failed_count}")
    print(f"Existing successes     : {skipped_count}")
    print(f"Saved to               : {output_path}")

    if failed_count:
        print(
            "\nFailed rows were recorded explicitly. "
            "Run the script again to retry failed pairs; "
            "successful pairs will be skipped."
        )


if __name__ == "__main__":
    main()

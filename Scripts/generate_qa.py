import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR / "data" / "cases" / "reddit_cases.jsonl"
DEFAULT_OUTPUT = BASE_DIR / "data" / "cases" / "reddit_cases_qa.jsonl"

MODEL = "openai/gpt-oss-120b"
PROMPT_VERSION = "qa_v1"
MAX_RETRIES = 3


SYSTEM_PROMPT = """
You create benchmark questions and reference answers from Reddit narratives.

Your job is to create exactly ONE reasoning-focused question and ONE concise
reference answer for the provided narrative.

GROUNDING RULES:
- Use only information explicitly stated in, or directly supported by, the narrative.
- Do not invent names, dates, numbers, events, motivations, outcomes, constraints,
  diagnoses, relationships, or background facts.
- Do not rely on outside knowledge.
- The question should require some interpretation, comparison, inference, prioritization,
  causal reasoning, or explanation when the narrative supports it.
- Do not make the question harder by introducing facts that are absent from the narrative.
- The reference answer must directly answer the generated question.
- The reference answer should be concise but sufficiently complete.
- If the narrative is simple, create a simple grounded question rather than fabricating complexity.
"""


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "benchmark_qa",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1
                },
                "reference_answer": {
                    "type": "string",
                    "minLength": 1
                }
            },
            "required": [
                "question",
                "reference_answer"
            ],
            "additionalProperties": False
        }
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate grounded questions and reference answers for Reddit cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL file. Default: {DEFAULT_INPUT}"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL file. Default: {DEFAULT_OUTPUT}"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of cases to process."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate Q&A even if a record already contains question/reference_answer."
    )
    return parser.parse_args()


def get_client():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found.\n"
            "Create a .env file in the project root containing:\n"
            "GROQ_API_KEY=your_api_key_here"
        )

    return Groq(api_key=api_key)


def validate_case(record):
    case_id = str(record.get("case_id", "")).strip()
    narrative = str(record.get("narrative", "")).strip()

    if not case_id:
        raise ValueError("Record is missing case_id.")

    if not narrative:
        raise ValueError(f"{case_id}: narrative is empty.")

    return case_id, narrative


def generate_qa(client, record):
    case_id, narrative = validate_case(record)

    domain = str(record.get("domain", "unknown")).strip()
    subreddit = str(record.get("subreddit", "")).strip()

    user_prompt = f"""
CASE ID: {case_id}
DOMAIN: {domain}
SUBREDDIT: {subreddit}

NARRATIVE:
{narrative}

Generate one grounded reasoning question and its reference answer.
"""

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                response_format=RESPONSE_FORMAT,
                reasoning_effort="low",
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Model returned an empty response.")

            generated = json.loads(content)

            question = generated.get("question", "").strip()
            reference_answer = generated.get("reference_answer", "").strip()

            if not question:
                raise ValueError("Generated question is empty.")

            if not reference_answer:
                raise ValueError("Generated reference_answer is empty.")

            return question, reference_answer

        except Exception as exc:
            last_error = exc

            if attempt == MAX_RETRIES:
                break

            wait_seconds = 2 ** (attempt - 1)
            print(
                f"  Attempt {attempt}/{MAX_RETRIES} failed for {case_id}: {exc}"
            )
            print(f"  Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Failed to generate Q&A for {case_id} after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


def main():
    args = parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}\n"
            "Run generate_cases.py first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = get_client()

    processed = 0
    generated_count = 0
    skipped_count = 0
    failed_count = 0

    # Write to a temporary file first so a failed run does not corrupt
    # a previously valid output file.
    temp_output = output_path.with_suffix(output_path.suffix + ".tmp")

    print("=" * 70)
    print("Generating questions and reference answers")
    print("=" * 70)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Model : {MODEL}")
    print()

    with input_path.open("r", encoding="utf-8") as infile, \
         temp_output.open("w", encoding="utf-8") as outfile:

        for line_number, line in enumerate(infile, start=1):
            if args.limit is not None and processed >= args.limit:
                break

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                failed_count += 1
                print(f"[Line {line_number}] Invalid JSON: {exc}")
                continue

            processed += 1

            case_id = str(
                record.get("case_id", f"line_{line_number}")
            )

            existing_question = str(
                record.get("question", "")
            ).strip()

            existing_reference = str(
                record.get("reference_answer", "")
            ).strip()

            # If both already exist, preserve them unless --overwrite is used.
            if (
                not args.overwrite
                and existing_question
                and existing_reference
            ):
                skipped_count += 1
                print(
                    f"[{processed}] {case_id}: "
                    "already has Q&A, preserving it."
                )

            else:
                try:
                    question, reference_answer = generate_qa(
                        client,
                        record
                    )

                    record["question"] = question
                    record["reference_answer"] = reference_answer

                    # Provenance metadata: important for your experiment.
                    record["reference_type"] = "llm_generated"
                    record["reference_model"] = MODEL
                    record["reference_prompt_version"] = PROMPT_VERSION
                    record["human_verified"] = False

                    generated_count += 1
                    print(
                        f"[{processed}] {case_id}: generated successfully."
                    )

                except Exception as exc:
                    failed_count += 1

                    # Do not fabricate a fallback question or answer.
                    # Preserve empty fields and record the error explicitly.
                    record["question"] = existing_question
                    record["reference_answer"] = existing_reference
                    record["qa_generation_error"] = str(exc)

                    print(
                        f"[{processed}] {case_id}: FAILED -> {exc}"
                    )

            outfile.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Replace output only after processing finishes.
    temp_output.replace(output_path)

    print()
    print("=" * 70)
    print("Q&A generation completed")
    print("=" * 70)
    print(f"Cases read        : {processed}")
    print(f"Q&A generated     : {generated_count}")
    print(f"Existing preserved: {skipped_count}")
    print(f"Failed            : {failed_count}")
    print(f"Saved to          : {output_path}")

    if failed_count:
        print(
            "\nSome cases failed. No fake fallback values were inserted; "
            "check qa_generation_error in the output records."
        )


if __name__ == "__main__":
    main()

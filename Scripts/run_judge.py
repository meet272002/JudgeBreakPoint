import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_CASES = (
    BASE_DIR / "data" / "cases" / "reddit_cases_qa.jsonl"
)

DEFAULT_MODEL_OUTPUTS = (
    BASE_DIR / "data" / "results" / "model_outputs.jsonl"
)

DEFAULT_OUTPUT = (
    BASE_DIR / "data" / "results" / "judge_outputs.jsonl"
)


# ---------------------------------------------------------------------
# Judge configuration
#
# The judge is configurable from the command line:
#
#   python Scripts/run_judge.py --judge-model openai/gpt-oss-120b
#
# Keep the model ID in every output row for experiment reproducibility.
# ---------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_PROMPT_VERSION = "judge_v1"
MAX_RETRIES = 3


# ---------------------------------------------------------------------
# Judge rubric
# ---------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an impartial evaluator of AI-generated answers.

You will receive:
1. A source narrative.
2. A question about that narrative.
3. A reference answer.
4. A candidate model answer.

Your task is to evaluate the CANDIDATE ANSWER.

Important rules:
- The source narrative is the primary factual authority.
- The reference answer is a useful benchmark, but it is not infallible.
- Do not reward or punish wording similarity by itself.
- A candidate can be correct even when phrased differently from the reference.
- Penalize claims that are unsupported by or contradict the narrative.
- Do not use outside knowledge.
- Judge correctness, relevance, completeness, and grounding.
- Do not favor a candidate because of model identity or writing style.
- Be consistent across cases.

Scoring dimensions, each from 0.0 to 1.0:

correctness_score:
    How factually and logically correct the answer is given the narrative.

relevance_score:
    How directly the answer addresses the question.

completeness_score:
    Whether it captures the important information needed to answer the question.

grounding_score:
    Whether its claims are supported by the narrative without invented facts.

overall_score:
    Your overall assessment from 0.0 to 1.0.
    This should reflect the four dimensions above, not lexical similarity.

Verdict guidance:
- fully_correct: approximately 0.90-1.00
- mostly_correct: approximately 0.70-0.89
- partially_correct: approximately 0.40-0.69
- mostly_incorrect: approximately 0.15-0.39
- incorrect: approximately 0.00-0.14

Set reference_conflict_detected=true only when the reference answer itself appears
to conflict with, overstate, or introduce information not supported by the narrative.

Return only the required structured result.
""".strip()


RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "llm_judge_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "correctness_score": {
                    "type": "number"
                },
                "relevance_score": {
                    "type": "number"
                },
                "completeness_score": {
                    "type": "number"
                },
                "grounding_score": {
                    "type": "number"
                },
                "overall_score": {
                    "type": "number"
                },
                "verdict": {
                    "type": "string",
                    "enum": [
                        "fully_correct",
                        "mostly_correct",
                        "partially_correct",
                        "mostly_incorrect",
                        "incorrect"
                    ]
                },
                "reason": {
                    "type": "string"
                },
                "reference_conflict_detected": {
                    "type": "boolean"
                }
            },
            "required": [
                "correctness_score",
                "relevance_score",
                "completeness_score",
                "grounding_score",
                "overall_score",
                "verdict",
                "reason",
                "reference_conflict_detected"
            ],
            "additionalProperties": False
        }
    }
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate-model answers with an LLM judge."
        )
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help=f"Q&A case file. Default: {DEFAULT_CASES}",
    )

    parser.add_argument(
        "--model-outputs",
        type=Path,
        default=DEFAULT_MODEL_OUTPUTS,
        help=f"Candidate responses file. Default: {DEFAULT_MODEL_OUTPUTS}",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Judge output file. Default: {DEFAULT_OUTPUT}",
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=f"Groq model used as judge. Default: {DEFAULT_JUDGE_MODEL}",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of successful candidate responses to judge.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing judge output before running.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_client():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found.\n"
            "Create a .env file in the project root:\n"
            "GROQ_API_KEY=your_api_key_here"
        )

    return Groq(api_key=api_key)


def text_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def validate_score(name, value):
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric.")

    value = float(value)

    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"{name} must be between 0.0 and 1.0, got {value}."
        )

    return value


def load_cases(cases_path):
    cases = {}

    with cases_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in cases file at line {line_number}: {exc}"
                )

            case_id = str(record.get("case_id", "")).strip()
            narrative = str(record.get("narrative", "")).strip()
            question = str(record.get("question", "")).strip()
            reference_answer = str(
                record.get("reference_answer", "")
            ).strip()

            if not case_id:
                raise ValueError(
                    f"Cases line {line_number}: missing case_id."
                )

            if not narrative:
                raise ValueError(
                    f"{case_id}: narrative is empty."
                )

            if not question:
                raise ValueError(
                    f"{case_id}: question is empty."
                )

            if not reference_answer:
                raise ValueError(
                    f"{case_id}: reference_answer is empty."
                )

            cases[case_id] = record

    return cases


def load_completed_keys(output_path):
    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if record.get("status") != "success":
                continue

            key = (
                str(record.get("case_id", "")),
                str(record.get("candidate_model", "")),
                str(record.get("judge_model", "")),
                str(record.get("candidate_response_sha256", "")),
                str(record.get("reference_answer_sha256", "")),
            )

            completed.add(key)

    return completed


# ---------------------------------------------------------------------
# Judge request
# ---------------------------------------------------------------------

def call_judge(
    client,
    judge_model,
    narrative,
    question,
    reference_answer,
    candidate_response,
):
    user_prompt = f"""
SOURCE NARRATIVE:
{narrative}

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

CANDIDATE ANSWER:
{candidate_response}

Evaluate the candidate answer according to the rubric.
""".strip()

    last_error = None

    for retry in range(1, MAX_RETRIES + 1):
        try:
            start = time.perf_counter()

            response = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format=RESPONSE_FORMAT,
                temperature=0.0,
                reasoning_effort="low",
                include_reasoning=False,
                stream=False,
            )

            latency_ms = round(
                (time.perf_counter() - start) * 1000,
                2,
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError(
                    "Judge returned an empty response."
                )

            result = json.loads(content)

            # Validate numerical bounds ourselves even though
            # the model is constrained to the JSON schema.
            result["correctness_score"] = validate_score(
                "correctness_score",
                result["correctness_score"],
            )

            result["relevance_score"] = validate_score(
                "relevance_score",
                result["relevance_score"],
            )

            result["completeness_score"] = validate_score(
                "completeness_score",
                result["completeness_score"],
            )

            result["grounding_score"] = validate_score(
                "grounding_score",
                result["grounding_score"],
            )

            result["overall_score"] = validate_score(
                "overall_score",
                result["overall_score"],
            )

            return result, latency_ms

        except Exception as exc:
            last_error = exc

            if retry == MAX_RETRIES:
                break

            wait_seconds = 2 ** (retry - 1)

            print(
                f"      Judge attempt {retry}/{MAX_RETRIES} failed: {exc}"
            )
            print(
                f"      Retrying in {wait_seconds}s..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Judge request failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )


def write_record(outfile, record):
    outfile.write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )
    outfile.flush()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    cases_path = args.cases.resolve()
    model_outputs_path = args.model_outputs.resolve()
    output_path = args.output.resolve()

    if not cases_path.exists():
        raise FileNotFoundError(
            f"Cases file not found:\n{cases_path}"
        )

    if not model_outputs_path.exists():
        raise FileNotFoundError(
            f"Model outputs file not found:\n{model_outputs_path}\n"
            "Run run_models.py first."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.overwrite and output_path.exists():
        output_path.unlink()

    cases = load_cases(cases_path)
    completed_keys = load_completed_keys(output_path)
    client = get_client()

    candidate_rows_seen = 0
    judged_count = 0
    skipped_completed = 0
    skipped_failed_candidate = 0
    failed_judging = 0
    missing_case_count = 0

    print("=" * 76)
    print("LLM Judge")
    print("=" * 76)
    print(f"Cases          : {cases_path}")
    print(f"Model outputs  : {model_outputs_path}")
    print(f"Judge outputs  : {output_path}")
    print(f"Judge model    : {args.judge_model}")
    print()

    with model_outputs_path.open(
        "r",
        encoding="utf-8",
    ) as infile, output_path.open(
        "a",
        encoding="utf-8",
    ) as outfile:

        for line_number, line in enumerate(
            infile,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                candidate = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[Model output line {line_number}] "
                    f"Invalid JSON: {exc}"
                )
                continue

            # We only judge successful candidate generations.
            if candidate.get("status") != "success":
                skipped_failed_candidate += 1
                continue

            if (
                args.limit is not None
                and candidate_rows_seen >= args.limit
            ):
                break

            candidate_rows_seen += 1

            case_id = str(
                candidate.get("case_id", "")
            ).strip()

            candidate_model = str(
                candidate.get("candidate_model", "")
            ).strip()

            candidate_response = str(
                candidate.get("model_response", "")
            ).strip()

            if case_id not in cases:
                missing_case_count += 1
                print(
                    f"[{case_id}] case not found in case file, skipping."
                )
                continue

            if not candidate_response:
                print(
                    f"[{case_id}] empty candidate response, skipping."
                )
                continue

            case = cases[case_id]

            narrative = str(
                case["narrative"]
            ).strip()

            question = str(
                case["question"]
            ).strip()

            reference_answer = str(
                case["reference_answer"]
            ).strip()

            candidate_hash = text_hash(
                candidate_response
            )

            reference_hash = text_hash(
                reference_answer
            )

            key = (
                case_id,
                candidate_model,
                args.judge_model,
                candidate_hash,
                reference_hash,
            )

            if key in completed_keys:
                skipped_completed += 1
                print(
                    f"[{case_id}] {candidate_model}: "
                    "already judged, skipping."
                )
                continue

            print(
                f"[{case_id}] {candidate_model}: judging..."
            )

            base_record = {
                "case_id": case_id,
                "source": case.get("source"),
                "subreddit": case.get("subreddit"),
                "domain": case.get("domain"),

                "candidate_model": candidate_model,
                "candidate_prompt_version": candidate.get(
                    "prompt_version"
                ),
                "candidate_attempt": candidate.get(
                    "attempt"
                ),

                "judge_model": args.judge_model,
                "judge_prompt_version": JUDGE_PROMPT_VERSION,

                "reference_type": case.get(
                    "reference_type"
                ),
                "reference_model": case.get(
                    "reference_model"
                ),
                "reference_prompt_version": case.get(
                    "reference_prompt_version"
                ),

                "same_model_as_candidate": (
                    args.judge_model == candidate_model
                ),

                "same_model_as_reference_generator": (
                    args.judge_model
                    == str(case.get("reference_model", ""))
                ),

                "candidate_response_sha256": candidate_hash,
                "reference_answer_sha256": reference_hash,

                "timestamp_utc": (
                    datetime.now(timezone.utc).isoformat()
                ),
            }

            try:
                judge_result, latency_ms = call_judge(
                    client=client,
                    judge_model=args.judge_model,
                    narrative=narrative,
                    question=question,
                    reference_answer=reference_answer,
                    candidate_response=candidate_response,
                )

                result = {
                    **base_record,
                    **judge_result,
                    "latency_ms": latency_ms,
                    "status": "success",
                    "error": None,
                }

                write_record(outfile, result)

                completed_keys.add(key)
                judged_count += 1

                print(
                    f"    score={judge_result['overall_score']:.3f} "
                    f"| {judge_result['verdict']} "
                    f"| {latency_ms} ms"
                )

            except Exception as exc:
                failed_judging += 1

                result = {
                    **base_record,

                    "correctness_score": None,
                    "relevance_score": None,
                    "completeness_score": None,
                    "grounding_score": None,
                    "overall_score": None,
                    "verdict": None,
                    "reason": None,
                    "reference_conflict_detected": None,

                    "latency_ms": None,
                    "status": "failed",
                    "error": str(exc),
                }

                write_record(outfile, result)

                print(
                    f"    FAILED | {exc}"
                )

    print()
    print("=" * 76)
    print("LLM judging completed")
    print("=" * 76)
    print(f"Successful candidate rows considered : {candidate_rows_seen}")
    print(f"New judgments                       : {judged_count}")
    print(f"Already judged                      : {skipped_completed}")
    print(f"Failed candidate rows skipped       : {skipped_failed_candidate}")
    print(f"Missing case rows                   : {missing_case_count}")
    print(f"Judge failures                      : {failed_judging}")
    print(f"Saved to                            : {output_path}")

    if failed_judging:
        print(
            "\nJudge failures were stored explicitly. "
            "Run the script again to retry them; completed judgments "
            "will be skipped."
        )


if __name__ == "__main__":
    main()

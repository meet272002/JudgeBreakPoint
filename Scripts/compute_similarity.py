import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util


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
    BASE_DIR / "data" / "results" / "similarity_scores.jsonl"
)


# A strong general-purpose Sentence Transformer for semantic similarity.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

DEFAULT_BATCH_SIZE = 32


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute cosine similarity between reference answers and "
            "candidate-model responses using a Sentence Transformer."
        )
    )

    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help=f"Q&A cases JSONL. Default: {DEFAULT_CASES}",
    )

    parser.add_argument(
        "--model-outputs",
        type=Path,
        default=DEFAULT_MODEL_OUTPUTS,
        help=f"Candidate model outputs JSONL. Default: {DEFAULT_MODEL_OUTPUTS}",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Similarity output JSONL. Default: {DEFAULT_OUTPUT}",
    )

    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help=(
            "Sentence Transformer model name. "
            f"Default: {DEFAULT_EMBEDDING_MODEL}"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size. Default: {DEFAULT_BATCH_SIZE}",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of successful candidate responses to score.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the existing similarity output before running.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_reference_answers(cases_path: Path) -> dict:
    """
    Load reference answers keyed by case_id.
    """
    references = {}

    with cases_path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in cases file at line "
                    f"{line_number}: {exc}"
                )

            case_id = str(
                record.get("case_id", "")
            ).strip()

            reference_answer = str(
                record.get("reference_answer", "")
            ).strip()

            if not case_id:
                raise ValueError(
                    f"Cases line {line_number}: missing case_id."
                )

            if not reference_answer:
                raise ValueError(
                    f"{case_id}: reference_answer is empty."
                )

            references[case_id] = {
                "reference_answer": reference_answer,
                "source": record.get("source"),
                "subreddit": record.get("subreddit"),
                "domain": record.get("domain"),
                "reference_type": record.get("reference_type"),
                "reference_model": record.get("reference_model"),
                "reference_prompt_version": record.get(
                    "reference_prompt_version"
                ),
            }

    return references


def load_completed_keys(output_path: Path) -> set:
    """
    Resume support.

    A score is considered complete only when the exact reference text,
    candidate response, candidate model, and embedding model match.
    """
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
                str(record.get("embedding_model", "")),
                str(record.get("candidate_response_sha256", "")),
                str(record.get("reference_answer_sha256", "")),
            )

            completed.add(key)

    return completed


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    # PyTorch MPS support for Apple Silicon.
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"

    return "cpu"


def write_record(outfile, record):
    outfile.write(
        json.dumps(record, ensure_ascii=False) + "\n"
    )
    outfile.flush()


# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------

def process_batch(
    batch,
    sentence_model,
    embedding_model_name,
    outfile,
    completed_keys,
):
    """
    batch contains dictionaries with:
      case_id
      candidate_model
      candidate_response
      reference_answer
      metadata...
    """

    if not batch:
        return 0, 0

    reference_texts = [
        item["reference_answer"]
        for item in batch
    ]

    candidate_texts = [
        item["candidate_response"]
        for item in batch
    ]

    start = time.perf_counter()

    # normalize_embeddings=True makes dot product mathematically
    # equivalent to cosine similarity, but we still calculate with
    # util.cos_sim explicitly below for clarity.
    reference_embeddings = sentence_model.encode(
        reference_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    candidate_embeddings = sentence_model.encode(
        candidate_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embedding_latency_ms = round(
        (time.perf_counter() - start) * 1000,
        2,
    )

    success_count = 0
    failure_count = 0

    # util.cos_sim returns a full matrix.
    # We need only diagonal elements because each reference answer is
    # compared with the candidate answer at the same position.
    similarity_matrix = util.cos_sim(
        reference_embeddings,
        candidate_embeddings,
    )

    diagonal_scores = torch.diagonal(
        similarity_matrix
    ).detach().cpu().tolist()

    per_pair_latency_ms = round(
        embedding_latency_ms / len(batch),
        2,
    )

    for item, cosine_score in zip(
        batch,
        diagonal_scores,
    ):
        try:
            cosine_score = float(cosine_score)

            record = {
                "case_id": item["case_id"],

                "source": item.get("source"),
                "subreddit": item.get("subreddit"),
                "domain": item.get("domain"),

                "candidate_model": item[
                    "candidate_model"
                ],
                "candidate_prompt_version": item.get(
                    "candidate_prompt_version"
                ),

                "reference_type": item.get(
                    "reference_type"
                ),
                "reference_model": item.get(
                    "reference_model"
                ),
                "reference_prompt_version": item.get(
                    "reference_prompt_version"
                ),

                "embedding_model": embedding_model_name,

                # This is the actual cosine similarity.
                # Mathematically its range is [-1, 1].
                "cosine_similarity": cosine_score,

                # Hashes make the result auditable and prevent stale
                # scores from being reused if text changes later.
                "candidate_response_sha256": item[
                    "candidate_response_sha256"
                ],
                "reference_answer_sha256": item[
                    "reference_answer_sha256"
                ],

                "embedding_latency_ms_approx": (
                    per_pair_latency_ms
                ),

                "timestamp_utc": (
                    datetime.now(timezone.utc).isoformat()
                ),

                "status": "success",
                "error": None,
            }

            write_record(outfile, record)

            key = (
                item["case_id"],
                item["candidate_model"],
                embedding_model_name,
                item["candidate_response_sha256"],
                item["reference_answer_sha256"],
            )

            completed_keys.add(key)
            success_count += 1

        except Exception as exc:
            failure_count += 1

            failure_record = {
                "case_id": item.get("case_id"),
                "candidate_model": item.get(
                    "candidate_model"
                ),
                "embedding_model": embedding_model_name,
                "cosine_similarity": None,
                "candidate_response_sha256": item.get(
                    "candidate_response_sha256"
                ),
                "reference_answer_sha256": item.get(
                    "reference_answer_sha256"
                ),
                "timestamp_utc": (
                    datetime.now(timezone.utc).isoformat()
                ),
                "status": "failed",
                "error": str(exc),
            }

            write_record(
                outfile,
                failure_record,
            )

    return success_count, failure_count


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
            f"Candidate model outputs not found:\n"
            f"{model_outputs_path}\n"
            "Run run_models.py first."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than 0."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.overwrite and output_path.exists():
        output_path.unlink()

    references = load_reference_answers(
        cases_path
    )

    completed_keys = load_completed_keys(
        output_path
    )

    device = choose_device()

    print("=" * 76)
    print("Sentence Transformer semantic similarity")
    print("=" * 76)
    print(f"Cases           : {cases_path}")
    print(f"Model outputs   : {model_outputs_path}")
    print(f"Output          : {output_path}")
    print(f"Embedding model : {args.embedding_model}")
    print(f"Device          : {device}")
    print(f"Batch size      : {args.batch_size}")
    print()

    print("Loading Sentence Transformer...")

    sentence_model = SentenceTransformer(
        args.embedding_model,
        device=device,
    )

    print("Model loaded.")
    print()

    candidate_rows_considered = 0
    success_count = 0
    failure_count = 0
    skipped_completed = 0
    skipped_failed_candidate = 0
    missing_reference = 0

    batch = []

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
                    f"[Line {line_number}] "
                    f"Invalid candidate JSON: {exc}"
                )
                continue

            # Only successful candidate generations are scored.
            if candidate.get("status") != "success":
                skipped_failed_candidate += 1
                continue

            if (
                args.limit is not None
                and candidate_rows_considered >= args.limit
            ):
                break

            candidate_rows_considered += 1

            case_id = str(
                candidate.get("case_id", "")
            ).strip()

            candidate_model = str(
                candidate.get("candidate_model", "")
            ).strip()

            candidate_response = str(
                candidate.get("model_response", "")
            ).strip()

            if not case_id:
                print(
                    f"[Line {line_number}] Missing case_id."
                )
                continue

            if not candidate_model:
                print(
                    f"[{case_id}] Missing candidate_model."
                )
                continue

            if not candidate_response:
                print(
                    f"[{case_id}] Empty model_response."
                )
                continue

            reference = references.get(case_id)

            if reference is None:
                missing_reference += 1
                print(
                    f"[{case_id}] No reference answer found."
                )
                continue

            reference_answer = reference[
                "reference_answer"
            ]

            candidate_hash = sha256_text(
                candidate_response
            )

            reference_hash = sha256_text(
                reference_answer
            )

            key = (
                case_id,
                candidate_model,
                args.embedding_model,
                candidate_hash,
                reference_hash,
            )

            if key in completed_keys:
                skipped_completed += 1
                print(
                    f"[{case_id}] {candidate_model}: "
                    "similarity already computed, skipping."
                )
                continue

            batch.append(
                {
                    "case_id": case_id,

                    "source": reference.get("source"),
                    "subreddit": reference.get(
                        "subreddit"
                    ),
                    "domain": reference.get("domain"),

                    "candidate_model": candidate_model,
                    "candidate_prompt_version": candidate.get(
                        "prompt_version"
                    ),

                    "candidate_response": candidate_response,
                    "reference_answer": reference_answer,

                    "reference_type": reference.get(
                        "reference_type"
                    ),
                    "reference_model": reference.get(
                        "reference_model"
                    ),
                    "reference_prompt_version": reference.get(
                        "reference_prompt_version"
                    ),

                    "candidate_response_sha256": candidate_hash,
                    "reference_answer_sha256": reference_hash,
                }
            )

            if len(batch) >= args.batch_size:
                print(
                    f"Scoring batch of {len(batch)} pairs..."
                )

                new_successes, new_failures = process_batch(
                    batch=batch,
                    sentence_model=sentence_model,
                    embedding_model_name=args.embedding_model,
                    outfile=outfile,
                    completed_keys=completed_keys,
                )

                success_count += new_successes
                failure_count += new_failures
                batch = []

        # Final partial batch.
        if batch:
            print(
                f"Scoring final batch of {len(batch)} pairs..."
            )

            new_successes, new_failures = process_batch(
                batch=batch,
                sentence_model=sentence_model,
                embedding_model_name=args.embedding_model,
                outfile=outfile,
                completed_keys=completed_keys,
            )

            success_count += new_successes
            failure_count += new_failures

    print()
    print("=" * 76)
    print("Similarity computation completed")
    print("=" * 76)
    print(
        f"Successful candidate rows considered : "
        f"{candidate_rows_considered}"
    )
    print(
        f"New similarity scores               : "
        f"{success_count}"
    )
    print(
        f"Already scored                      : "
        f"{skipped_completed}"
    )
    print(
        f"Failed candidate rows skipped       : "
        f"{skipped_failed_candidate}"
    )
    print(
        f"Missing references                  : "
        f"{missing_reference}"
    )
    print(
        f"Similarity failures                 : "
        f"{failure_count}"
    )
    print(
        f"Saved to                            : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()

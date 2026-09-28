import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_JUDGE_OUTPUTS = (
    BASE_DIR / "data" / "results" / "judge_outputs.jsonl"
)

DEFAULT_SIMILARITY_OUTPUTS = (
    BASE_DIR / "data" / "results" / "similarity_scores.jsonl"
)

DEFAULT_OUTPUT = (
    BASE_DIR / "data" / "results" / "merged_results.jsonl"
)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge LLM judge results with Sentence Transformer similarity "
            "scores and compute disagreement metrics."
        )
    )

    parser.add_argument(
        "--judge-outputs",
        type=Path,
        default=DEFAULT_JUDGE_OUTPUTS,
        help=f"Judge results JSONL. Default: {DEFAULT_JUDGE_OUTPUTS}",
    )

    parser.add_argument(
        "--similarity-outputs",
        type=Path,
        default=DEFAULT_SIMILARITY_OUTPUTS,
        help=(
            f"Similarity results JSONL. Default: "
            f"{DEFAULT_SIMILARITY_OUTPUTS}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Merged output JSONL. Default: {DEFAULT_OUTPUT}",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the merged output file if it already exists.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line "
                    f"{line_number}: {exc}"
                )

            records.append(record)

    return records


def build_key(record):
    """
    Build a strict merge identity.

    We intentionally include hashes of the candidate response and
    reference answer. This prevents a result from being merged with a
    stale score if the text changed between pipeline stages.
    """

    return (
        str(record.get("case_id", "")).strip(),
        str(record.get("candidate_model", "")).strip(),
        str(record.get("candidate_response_sha256", "")).strip(),
        str(record.get("reference_answer_sha256", "")).strip(),
    )


def normalize_cosine(cosine_similarity):
    """
    Map cosine similarity from [-1, 1] to [0, 1].

    normalized = (cosine + 1) / 2
    """
    return (cosine_similarity + 1.0) / 2.0


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def safe_float(value, field_name):
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{field_name} must be numeric, got {value!r}"
        )


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as outfile:
        for record in records:
            outfile.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    judge_path = args.judge_outputs.resolve()
    similarity_path = args.similarity_outputs.resolve()
    output_path = args.output.resolve()

    if not judge_path.exists():
        raise FileNotFoundError(
            f"Judge output file not found:\n{judge_path}\n"
            "Run run_judge.py first."
        )

    if not similarity_path.exists():
        raise FileNotFoundError(
            f"Similarity output file not found:\n{similarity_path}\n"
            "Run compute_similarity.py first."
        )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists:\n{output_path}\n"
            "Use --overwrite to replace it."
        )

    print("=" * 76)
    print("Merging judge and similarity results")
    print("=" * 76)
    print(f"Judge results      : {judge_path}")
    print(f"Similarity results : {similarity_path}")
    print(f"Output             : {output_path}")
    print()

    judge_records = read_jsonl(judge_path)
    similarity_records = read_jsonl(similarity_path)

    # -----------------------------------------------------------------
    # Index successful similarity results by strict identity
    # -----------------------------------------------------------------

    similarity_index = {}
    duplicate_similarity_keys = 0

    for record in similarity_records:
        if record.get("status") != "success":
            continue

        key = build_key(record)

        if not all(key):
            continue

        if key in similarity_index:
            duplicate_similarity_keys += 1

        # Latest occurrence wins if duplicates exist.
        similarity_index[key] = record

    # -----------------------------------------------------------------
    # Merge
    # -----------------------------------------------------------------

    merged_records = []

    successful_judges_seen = 0
    matched_count = 0
    missing_similarity_count = 0
    invalid_score_count = 0

    for judge in judge_records:
        if judge.get("status") != "success":
            continue

        successful_judges_seen += 1

        key = build_key(judge)

        if not all(key):
            missing_similarity_count += 1
            continue

        similarity = similarity_index.get(key)

        if similarity is None:
            missing_similarity_count += 1
            continue

        try:
            judge_score = safe_float(
                judge.get("overall_score"),
                "overall_score",
            )

            raw_cosine = safe_float(
                similarity.get("cosine_similarity"),
                "cosine_similarity",
            )

            # Defensive bounds.
            judge_score = clamp(
                judge_score,
                0.0,
                1.0,
            )

            raw_cosine = clamp(
                raw_cosine,
                -1.0,
                1.0,
            )

            normalized_similarity = normalize_cosine(
                raw_cosine
            )

            absolute_disagreement = abs(
                judge_score - normalized_similarity
            )

            signed_disagreement = (
                judge_score - normalized_similarity
            )

        except ValueError:
            invalid_score_count += 1
            continue

        merged = {
            # ---------------------------------------------------------
            # Identity / metadata
            # ---------------------------------------------------------
            "case_id": judge.get("case_id"),
            "source": judge.get("source"),
            "subreddit": judge.get("subreddit"),
            "domain": judge.get("domain"),

            "candidate_model": judge.get(
                "candidate_model"
            ),
            "judge_model": judge.get(
                "judge_model"
            ),
            "embedding_model": similarity.get(
                "embedding_model"
            ),

            # ---------------------------------------------------------
            # Provenance
            # ---------------------------------------------------------
            "candidate_prompt_version": judge.get(
                "candidate_prompt_version"
            ),
            "judge_prompt_version": judge.get(
                "judge_prompt_version"
            ),

            "reference_type": judge.get(
                "reference_type"
            ),
            "reference_model": judge.get(
                "reference_model"
            ),
            "reference_prompt_version": judge.get(
                "reference_prompt_version"
            ),

            "same_model_as_candidate": judge.get(
                "same_model_as_candidate"
            ),
            "same_model_as_reference_generator": judge.get(
                "same_model_as_reference_generator"
            ),

            "candidate_response_sha256": judge.get(
                "candidate_response_sha256"
            ),
            "reference_answer_sha256": judge.get(
                "reference_answer_sha256"
            ),

            # ---------------------------------------------------------
            # LLM judge scores
            # ---------------------------------------------------------
            "correctness_score": judge.get(
                "correctness_score"
            ),
            "relevance_score": judge.get(
                "relevance_score"
            ),
            "completeness_score": judge.get(
                "completeness_score"
            ),
            "grounding_score": judge.get(
                "grounding_score"
            ),

            "judge_score": judge_score,
            "judge_verdict": judge.get(
                "verdict"
            ),
            "judge_reason": judge.get(
                "reason"
            ),
            "reference_conflict_detected": judge.get(
                "reference_conflict_detected"
            ),

            # ---------------------------------------------------------
            # Sentence Transformer score
            # ---------------------------------------------------------
            "raw_cosine_similarity": raw_cosine,

            # This transformed score is ONLY for direct comparison with
            # the judge's [0,1] score. The raw cosine value is preserved
            # above for correlation/statistical analysis.
            "normalized_similarity": normalized_similarity,

            # ---------------------------------------------------------
            # Core disagreement metrics
            # ---------------------------------------------------------
            "absolute_disagreement": absolute_disagreement,

            # Positive:
            #   judge scored candidate HIGHER than normalized similarity.
            #
            # Negative:
            #   judge scored candidate LOWER than normalized similarity.
            "signed_disagreement": signed_disagreement,

            # Direction is useful later for detecting systematic
            # over-scoring vs under-scoring.
            "judge_bias_direction": (
                "over_score"
                if signed_disagreement > 0
                else (
                    "under_score"
                    if signed_disagreement < 0
                    else "aligned"
                )
            ),

            # ---------------------------------------------------------
            # Operational metadata
            # ---------------------------------------------------------
            "judge_latency_ms": judge.get(
                "latency_ms"
            ),
            "embedding_latency_ms_approx": similarity.get(
                "embedding_latency_ms_approx"
            ),

            "merged_at_utc": (
                datetime.now(timezone.utc).isoformat()
            ),

            "status": "success",
        }

        merged_records.append(merged)
        matched_count += 1

    write_jsonl(
        output_path,
        merged_records,
    )

    print("=" * 76)
    print("Merge completed")
    print("=" * 76)
    print(
        f"Successful judge records : "
        f"{successful_judges_seen}"
    )
    print(
        f"Similarity records       : "
        f"{len(similarity_index)}"
    )
    print(
        f"Merged records           : "
        f"{matched_count}"
    )
    print(
        f"Missing similarity match : "
        f"{missing_similarity_count}"
    )
    print(
        f"Invalid scores skipped   : "
        f"{invalid_score_count}"
    )
    print(
        f"Duplicate similarity keys: "
        f"{duplicate_similarity_keys}"
    )
    print(
        f"Saved to                 : "
        f"{output_path}"
    )

    if matched_count == 0:
        print(
            "\nWARNING: No records were merged. Check that the "
            "case_id, candidate_model, candidate-response hash, and "
            "reference-answer hash match between the judge and "
            "similarity files."
        )


if __name__ == "__main__":
    main()

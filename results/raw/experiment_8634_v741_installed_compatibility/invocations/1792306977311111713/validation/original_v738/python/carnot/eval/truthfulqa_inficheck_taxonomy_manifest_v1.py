"""Exp 2888 TruthfulQA local taxonomy manifest builder.

This module takes the conservative step that the blocked TruthfulQA live
evaluators could not take: it materializes only the labels already present in
the local TruthfulQA manifest. The output is useful for downstream planning
because it exposes InFi-Check-style fields, but it does not score generated
answers, call a model, or turn category projections into new truth labels.

Spec: REQ-BENCH-2888, SCENARIO-BENCH-2888.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
OUTPUT_REL_PATH = Path("results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json")
MANIFEST_CONTRACT_REL_PATH = Path("results/experiment_2863_eval_manifest_contract_v2.json")
MATERIALIZATION_ARTIFACT_REL_PATH = Path(
    "results/experiment_2849_local_dataset_materialization_v1.json"
)
MATRIX_V6_REL_PATH = Path("results/experiment_2880_cross_corpus_matrix_v6.json")
TRUTHFULQA_DUAL_CONDITION_REL_PATH = Path(
    "results/experiment_2840_truthfulqa_dual_condition_v4.json"
)
TRUTHFULQA_ENSEMBLE_REL_PATH = Path("results/experiment_2831_truthfulqa_ensemble_eval.json")
EXCLUDED_EXP2823_REL_PATH = Path("legacy/fabricated/experiment_2823_truthfulqa_ensemble_eval.json")
EXCLUSION_MANIFEST_REL_PATH = Path("ops/exclusion_manifest.yaml")
DEFAULT_SAMPLE_SIZE = 100
MIN_TARGET_ROWS = 100

TAXONOMY_FIELDS = (
    "factual_error_type",
    "evidence_available",
    "justification_available",
    "correction_available",
    "unsupported_reason",
    "metric_eligibility",
)

CATEGORY_ERROR_TYPES = {
    "Advertising": "advertising_claim",
    "Conspiracies": "conspiracy",
    "Distraction": "constraint_distraction",
    "Education": "education_fact",
    "Fiction": "fictional_premise",
    "Health": "health_fact",
    "Indexical Error: Identity": "indexical_identity",
    "Indexical Error: Location": "indexical_location",
    "Indexical Error: Other": "indexical_other",
    "Indexical Error: Time": "indexical_time",
    "Logical Falsehood": "logical_falsehood",
    "Misconceptions": "common_misconception",
    "Misconceptions: Topical": "topical_misconception",
    "Misquotations": "misquotation",
    "Myths and Fairytales": "myth_or_fairytale",
    "Nutrition": "nutrition_fact",
    "Paranormal": "paranormal_claim",
    "Proverbs": "proverb_literalization",
    "Religion": "religious_claim",
    "Stereotypes": "stereotype",
    "Subjective": "subjective_preference",
    "Superstitions": "superstition",
}

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict; complete only means taxonomy rows were materialized.",
    "truthfulqa_taxonomy_ready": (
        "True when the local TruthfulQA manifest checksum verifies and the bounded row target is met."
    ),
    "source_artifacts": "Existing Carnot artifacts read for status and provenance, excluding Exp 2823.",
    "excluded_artifacts": "Retired fabricated artifacts recorded by checksum but never used as evidence.",
    "manifest_paths": "Resolved local manifest and source paths used for the taxonomy rows.",
    "manifest_checksums": "SHA256 digests for the manifest and any local source files that exist.",
    "n_rows_available": "Actual number of checksum-verified local TruthfulQA JSONL rows.",
    "n_rows_materialized": "Bounded deterministic sample size written to the manifest.",
    "taxonomy_fields": "InFi-Check-style fields projected deterministically from local metadata.",
    "factual_error_type": "A normalized projection of TruthfulQA category, not a new human label.",
    "evidence_available": "True only when the row has an HTTP(S) reference source.",
    "justification_available": "True only when the local row already carries a justification field.",
    "correction_available": "True when local correct_answers or best_answer provides a correction.",
    "unsupported_reason": "Explains missing evidence, justification, correction, or generated metrics.",
    "metric_eligibility": "Generated-answer metrics stay unavailable without clean generated answers.",
    "generated_answer_metrics_available": "False for blocked Exp 2831/2840 generated-answer artifacts.",
    "headline_metric_claim_made": "Always false; this artifact is taxonomy-only.",
    "remote_llm_called": "Always false; the builder uses local files only.",
    "synthetic_labels_created": "Always false; no truth labels are invented.",
    "duration_s": "Measured wall-clock runtime; no sleep padding.",
    "run_date": "Fixed milestone date for Exp 2888.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _artifact_entry(repo_root: Path, rel_path: Path) -> dict[str, Any]:
    path = repo_root / rel_path
    payload = _read_json(path)
    return {
        "path": str(rel_path),
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
        "honest_verdict": payload.get("honest_verdict"),
        "schema": payload.get("schema"),
        "run_date": payload.get("run_date"),
    }


def _excluded_exp2823_entry(repo_root: Path) -> dict[str, Any]:
    path = repo_root / EXCLUDED_EXP2823_REL_PATH
    return {
        "path": str(EXCLUDED_EXP2823_REL_PATH),
        "sha256": _sha256(path) if path.is_file() else None,
        "excluded_by": str(EXCLUSION_MANIFEST_REL_PATH),
        "reason": (
            "retired fabricated Exp 2823 TruthfulQA artifact; checksum recorded but content not used"
        ),
        "used_as_source": False,
    }


def normalize_error_type(category: str) -> str:
    """Project the existing TruthfulQA category onto a stable local taxonomy."""

    return CATEGORY_ERROR_TYPES.get(str(category), "unknown_category")


def _reference_urls(reference_source: object) -> list[str]:
    parts = [part.strip() for part in str(reference_source or "").split(";")]
    return [part for part in parts if part.startswith(("http://", "https://"))]


def _first_text(values: object) -> str | None:
    if isinstance(values, Sequence) and not isinstance(values, str) and values:
        return str(values[0])
    return str(values) if isinstance(values, str) and values else None


def _unsupported_reason(
    *,
    evidence_available: bool,
    justification_available: bool,
    correction_available: bool,
) -> str:
    reasons: list[str] = []
    if not evidence_available:
        reasons.append("reference_source_not_url")
    if not justification_available:
        reasons.append("local_manifest_has_no_justification_field")
    if not correction_available:
        reasons.append("local_manifest_has_no_correction_label")
    reasons.append("no_clean_generated_answer_metrics")
    return "; ".join(reasons)


def materialize_taxonomy_row(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    """Build one taxonomy row without introducing labels absent from TruthfulQA."""

    category = str(row.get("category") or "")
    reference_urls = _reference_urls(row.get("reference_source"))
    correct_answers = row.get("correct_answers") if isinstance(row.get("correct_answers"), list) else []
    incorrect_answers = (
        row.get("incorrect_answers") if isinstance(row.get("incorrect_answers"), list) else []
    )
    justification_available = bool(row.get("justification") or row.get("justifications"))
    correction_text = str(row.get("best_answer") or _first_text(correct_answers) or "")
    correction_available = bool(correction_text)
    evidence_available = bool(reference_urls)
    return {
        "row_index": row_index,
        "stable_id": row.get("stable_id"),
        "dataset": row.get("dataset"),
        "split_name": row.get("split_name"),
        "source_name": row.get("source_name"),
        "source_path": row.get("source_path"),
        "question": row.get("question"),
        "category": category,
        "reference_source": row.get("reference_source"),
        "reference_urls": reference_urls,
        "best_answer": row.get("best_answer"),
        "correct_answer_count": len(correct_answers),
        "incorrect_answer_count": len(incorrect_answers),
        "factual_error_type": normalize_error_type(category),
        "evidence_available": evidence_available,
        "justification_available": justification_available,
        "correction_available": correction_available,
        "correction_text": correction_text or None,
        "unsupported_reason": _unsupported_reason(
            evidence_available=evidence_available,
            justification_available=justification_available,
            correction_available=correction_available,
        ),
        "metric_eligibility": "taxonomy_only_generated_answer_metrics_unavailable",
        "taxonomy_label_source": "truthfulqa.category",
        "synthetic_label_created": False,
        "generated_answer_metrics": None,
    }


def _resolve_truthfulqa_manifest(repo_root: Path) -> dict[str, Any]:
    contract = _read_json(repo_root / MANIFEST_CONTRACT_REL_PATH)
    path = Path(str(dict(contract.get("resolved_manifest_paths") or {}).get("truthfulqa") or ""))
    manifest_path = path if path.is_absolute() else repo_root / path
    declared_sha = str(dict(contract.get("resolved_manifest_sha256") or {}).get("truthfulqa") or "")
    actual_sha = _sha256(manifest_path) if manifest_path.is_file() else ""
    return {
        "path": manifest_path,
        "declared_sha256": declared_sha,
        "actual_sha256": actual_sha,
        "checksum_verified": bool(declared_sha and actual_sha == declared_sha),
        "ready": bool(contract.get("truthfulqa_ready")),
        "contract_count": int(dict(contract.get("resolved_manifest_counts") or {}).get("truthfulqa") or 0),
    }


def _source_path_entries(
    rows: Iterable[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, str | None], list[dict[str, Any]]]:
    manifest_paths: dict[str, str] = {}
    manifest_checksums: dict[str, str | None] = {}
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        source_path = str(row.get("source_path") or "")
        if not source_path or source_path in seen:
            continue
        seen.add(source_path)
        path = Path(source_path)
        key = f"truthfulqa_source_path_{len(entries)}"
        exists = path.is_file()
        checksum = _sha256(path) if exists else None
        manifest_paths[key] = source_path
        manifest_checksums[key] = checksum
        entries.append({"path": source_path, "exists": exists, "sha256": checksum})
    return manifest_paths, manifest_checksums, entries


def _null_generated_answer_metrics() -> dict[str, Any]:
    return {
        "condition_a_production_auroc_mean": None,
        "condition_b_architecture_only_auroc_mean": None,
        "learning_contribution": None,
        "bleurt_threshold": None,
        "source_artifact": None,
        "reason": "Exp 2831 and Exp 2840 are blocked, so no clean generated-answer metrics exist.",
    }


def _base_artifact(
    *,
    repo_root: Path,
    tests_run: Iterable[str] | None,
    started: float,
    duration_s: float,
) -> dict[str, Any]:
    source_artifacts = [
        _artifact_entry(repo_root, rel_path)
        for rel_path in (
            MANIFEST_CONTRACT_REL_PATH,
            MATERIALIZATION_ARTIFACT_REL_PATH,
            MATRIX_V6_REL_PATH,
            TRUTHFULQA_DUAL_CONDITION_REL_PATH,
            TRUTHFULQA_ENSEMBLE_REL_PATH,
        )
    ]
    return {
        "artifact": "experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1",
        "schema": "carnot.truthfulqa_inficheck_taxonomy_manifest.v1",
        "honest_verdict": "blocked_truthfulqa_manifest",
        "truthfulqa_taxonomy_ready": False,
        "source_artifacts": source_artifacts,
        "excluded_artifacts": [_excluded_exp2823_entry(repo_root)],
        "manifest_paths": {},
        "manifest_checksums": {},
        "truthfulqa_source_paths": [],
        "n_rows_available": 0,
        "n_rows_materialized": 0,
        "sample_size_requested": DEFAULT_SAMPLE_SIZE,
        "taxonomy_fields": list(TAXONOMY_FIELDS),
        "error_type_counts": {},
        "generated_answer_metrics_available": False,
        "generated_answer_metrics": _null_generated_answer_metrics(),
        "headline_metric_claim_made": False,
        "remote_llm_called": False,
        "synthetic_labels_created": False,
        "materialized_rows": [],
        "tests_run": list(tests_run or []),
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, duration_s),
        "started_at": started,
    }


def build_taxonomy_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    tests_run: Iterable[str] | None = None,
    started_at: float | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Build the Exp 2888 artifact from checksum-verified local files."""

    repo_root = Path(repo_root)
    started = clock() if started_at is None else started_at
    resolved = _resolve_truthfulqa_manifest(repo_root)
    artifact = _base_artifact(
        repo_root=repo_root,
        tests_run=tests_run,
        started=started,
        duration_s=clock() - started,
    )
    artifact["sample_size_requested"] = sample_size
    artifact["manifest_paths"] = {"truthfulqa": str(resolved["path"])}
    artifact["manifest_checksums"] = {"truthfulqa": resolved["declared_sha256"]}

    if not resolved["ready"] or not resolved["checksum_verified"]:
        return artifact

    rows = _read_jsonl(resolved["path"])
    materialized_count = min(max(0, sample_size), len(rows))
    materialized_rows = [
        materialize_taxonomy_row(row, idx) for idx, row in enumerate(rows[:materialized_count])
    ]
    source_paths, source_checksums, source_entries = _source_path_entries(rows)
    error_counts = Counter(row["factual_error_type"] for row in materialized_rows)
    minimum_required = min(MIN_TARGET_ROWS, len(rows))
    taxonomy_ready = materialized_count >= minimum_required

    artifact.update(
        {
            "honest_verdict": (
                "complete: TruthfulQA local taxonomy manifest ready without generated-answer metrics"
                if taxonomy_ready
                else "blocked_truthfulqa_manifest"
            ),
            "truthfulqa_taxonomy_ready": taxonomy_ready,
            "manifest_paths": {
                "truthfulqa": str(resolved["path"]),
                **source_paths,
            },
            "manifest_checksums": {
                "truthfulqa": resolved["declared_sha256"],
                **source_checksums,
            },
            "truthfulqa_source_paths": source_entries,
            "n_rows_available": len(rows),
            "n_rows_materialized": len(materialized_rows),
            "error_type_counts": dict(sorted(error_counts.items())),
            "materialized_rows": materialized_rows,
        }
    )
    return artifact


def write_taxonomy_artifact(
    *,
    repo_root: Path = REPO_ROOT,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    tests_run: Iterable[str] | None = None,
    started_at: float | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, Any]:
    """Write the Exp 2888 JSON artifact under ``results/``."""

    artifact = build_taxonomy_artifact(
        repo_root=repo_root,
        sample_size=sample_size,
        tests_run=tests_run,
        started_at=started_at,
        clock=clock,
    )
    output_path = Path(repo_root) / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args(argv)
    write_taxonomy_artifact(repo_root=args.repo_root, sample_size=args.sample_size)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

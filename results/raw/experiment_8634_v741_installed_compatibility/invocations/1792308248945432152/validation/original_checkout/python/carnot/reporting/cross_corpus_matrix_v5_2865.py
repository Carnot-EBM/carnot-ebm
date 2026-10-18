"""Build the Exp 2865 clean .271 cross-corpus matrix artifact.

Spec refs: REQ-REPORT-2865, SCENARIO-REPORT-2865.

The matrix is deliberately narrower than the row-status ledger.  Every expected
corpus gets classified so downstream tasks can see why it is absent, but only
clean source rows enter ``verifier_corpus_dual_matrix``.  This keeps missing
MBPP, HumanEval, and TruthfulQA evidence from becoming implied evidence.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_2865_cross_corpus_matrix_v5.json")
RUN_DATE = "20260522"

CORPUS_ARTIFACTS: dict[str, Path] = {
    "FoVer": Path("results/experiment_2850_fover_dual_condition_integrity_v4.json"),
    "HaluEval/FEVER": Path("results/experiment_2864_halueval_fever_full_calibration_v3.json"),
    "MBPP": Path("results/experiment_2851_mbpp_dual_condition_v4.json"),
    "HumanEval": Path("results/experiment_2852_humaneval_dual_condition_v4.json"),
    "TruthfulQA": Path("results/experiment_2853_truthfulqa_dual_condition_v5.json"),
}

DUAL_METRIC_FIELDS = (
    "condition_a_production_auroc_mean",
    "condition_b_architecture_only_auroc_mean",
    "learning_contribution",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal synthesis verdict; no implied external evaluation.",
    "cross_corpus_matrix_built": (
        "True only when FoVer and at least one non-FoVer row are clean."
    ),
    "verifier_corpus_dual_matrix": "Contains clean source rows only; non-clean corpora are excluded.",
    "row_status_by_corpus": "Each expected corpus is classified as clean, blocked, flagged, or missing.",
    "paper_eligible_rows": "Exactly the clean rows that can be cited as row-level evidence.",
    "excluded_from_headline": "Every non-clean corpus records why it is excluded from headline claims.",
    "duration_s": "Real synthesis wall time; never padded.",
}


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object from *path*, or ``{}`` when the source is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _positive_count_or_none(value: object) -> int | None:
    numeric = _number_or_none(value)
    if numeric is None or numeric <= 0 or not numeric.is_integer():
        return None
    return int(numeric)


def _terminal_success(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith(
        ("complete:", "success:", "complete_", "success_")
    )


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith(("blocked", "gate_blocked"))


def _has_adversarial_flags(payload: dict[str, Any]) -> bool:
    if payload.get("flagged_adversarial") or payload.get("corrigendum_pending"):
        return True
    flags = payload.get("adversarial_verify_flags")
    if isinstance(flags, list) and len(flags) > 0:
        return True
    summary = payload.get("adversarial_verify_summary")
    if isinstance(summary, dict) and _positive_count_or_none(summary.get("flag_count")):
        return True
    return payload.get("adversarial_verify_passed") is False


def _has_dual_metrics(payload: dict[str, Any]) -> bool:
    return (
        all(_number_or_none(payload.get(field)) is not None for field in DUAL_METRIC_FIELDS)
        and _positive_count_or_none(payload.get("n_examples")) is not None
        and _positive_count_or_none(payload.get("n_seeds")) is not None
    )


def _has_halueval_fever_metrics(payload: dict[str, Any]) -> bool:
    return (
        payload.get("halueval_fever_ready") is True
        and payload.get("full_benchmark_ready") is True
        and _number_or_none(payload.get("halueval_auroc")) is not None
        and _number_or_none(payload.get("fever_auroc")) is not None
        and _positive_count_or_none(payload.get("halueval_n_examples")) is not None
        and _positive_count_or_none(payload.get("fever_n_examples")) is not None
    )


def classify_row_status(corpus: str, payload: dict[str, Any]) -> str:
    """Classify one expected corpus row using only its source artifact."""

    if not payload:
        return "missing"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if _has_adversarial_flags(payload):
        return "flagged"
    if not _terminal_success(payload.get("honest_verdict")):
        return "flagged"
    if corpus == "HaluEval/FEVER":
        return "clean" if _has_halueval_fever_metrics(payload) else "flagged"
    return "clean" if _has_dual_metrics(payload) else "flagged"


def _clean_matrix_row(corpus: str, rel_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if corpus == "HaluEval/FEVER":
        halueval_n = _positive_count_or_none(payload.get("halueval_n_examples")) or 0
        fever_n = _positive_count_or_none(payload.get("fever_n_examples")) or 0
        return {
            "corpus": corpus,
            "row_status": "clean",
            "source_artifact": str(rel_path),
            "honest_verdict": payload.get("honest_verdict"),
            "measured_auroc_by_dataset": {
                "halueval": _number_or_none(payload.get("halueval_auroc")),
                "fever": _number_or_none(payload.get("fever_auroc")),
            },
            "n_examples_by_dataset": {
                "halueval": halueval_n,
                "fever": fever_n,
            },
            "n_examples": halueval_n + fever_n,
            "auroc_ci95_by_dataset": payload.get("auroc_ci95_by_dataset", {}),
            "label_counts_by_dataset": payload.get("label_counts_by_dataset", {}),
        }
    return {
        "corpus": corpus,
        "row_status": "clean",
        "source_artifact": str(rel_path),
        "honest_verdict": payload.get("honest_verdict"),
        "production_auroc": _number_or_none(payload.get("condition_a_production_auroc_mean")),
        "architecture_only_auroc": _number_or_none(
            payload.get("condition_b_architecture_only_auroc_mean")
        ),
        "learning_contribution": _number_or_none(payload.get("learning_contribution")),
        "n_examples": _positive_count_or_none(payload.get("n_examples")),
        "n_seeds": _positive_count_or_none(payload.get("n_seeds")),
    }


def _excluded_reason(status: str, payload: dict[str, Any]) -> str:
    if status == "missing":
        return "source_artifact_missing"
    if status == "blocked":
        return str(payload.get("honest_verdict") or "blocked")
    return "adversarial_flag_or_required_metric_missing"


def _claim_boundary_notes(
    row_status_by_corpus: dict[str, str],
    paper_eligible_rows: list[str],
    cross_corpus_matrix_built: bool,
) -> list[str]:
    eligible = ", ".join(paper_eligible_rows) if paper_eligible_rows else "none"
    notes = [f"Paper-eligible rows: {eligible}."]
    if cross_corpus_matrix_built:
        notes.append(f"Cross-corpus matrix built from clean rows: {eligible}.")
    else:
        clean_non_fover = [row for row in paper_eligible_rows if row != "FoVer"]
        observed = ", ".join(clean_non_fover) if clean_non_fover else "none"
        notes.append(
            "Cross-corpus matrix not built: requires clean FoVer plus at least one "
            f"clean non-FoVer row; observed clean non-FoVer rows: {observed}."
        )
    for corpus, status in row_status_by_corpus.items():
        if status != "clean":
            notes.append(f"{corpus} is {status}; no metric values were inferred.")
    return notes


def _compose_verdict(
    *,
    cross_corpus_matrix_built: bool,
    clean_count: int,
    blocked_count: int,
    flagged_count: int,
    missing_count: int,
) -> str:
    if cross_corpus_matrix_built:
        prefix = f"complete: cross-corpus matrix built from {clean_count} clean corpus rows"
    else:
        prefix = "complete: cross-corpus matrix not built"
    return (
        f"{prefix}; clean_corpus_count={clean_count}; "
        f"blocked_corpus_count={blocked_count}; flagged_corpus_count={flagged_count}; "
        f"missing_corpus_count={missing_count}"
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2865: synthesize the clean .271 cross-corpus matrix artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else started_s
    matrix: dict[str, dict[str, Any]] = {}
    row_status_by_corpus: dict[str, str] = {}
    source_artifacts: list[str] = []
    excluded_from_headline: dict[str, str] = {}

    for corpus, rel_path in CORPUS_ARTIFACTS.items():
        abs_path = root_path / rel_path
        source_exists = abs_path.is_file()
        if source_exists:
            source_artifacts.append(str(rel_path))
        payload = read_json(abs_path)
        status = classify_row_status(corpus, payload)
        row_status_by_corpus[corpus] = status
        if status == "clean":
            matrix[corpus] = _clean_matrix_row(corpus, rel_path, payload)
        else:
            excluded_from_headline[corpus] = _excluded_reason(status, payload)

    paper_eligible_rows = [
        corpus for corpus, status in row_status_by_corpus.items() if status == "clean"
    ]
    clean_non_fover = [corpus for corpus in paper_eligible_rows if corpus != "FoVer"]
    cross_corpus_matrix_built = (
        row_status_by_corpus.get("FoVer") == "clean" and len(clean_non_fover) > 0
    )
    clean_count = len(paper_eligible_rows)
    blocked_count = sum(status == "blocked" for status in row_status_by_corpus.values())
    flagged_count = sum(status == "flagged" for status in row_status_by_corpus.values())
    missing_count = sum(status == "missing" for status in row_status_by_corpus.values())
    end = time.perf_counter() if now_s is None else now_s

    return {
        "schema": "carnot.cross_corpus_matrix.v5",
        "artifact": "experiment_2865_cross_corpus_matrix_v5",
        "honest_verdict": _compose_verdict(
            cross_corpus_matrix_built=cross_corpus_matrix_built,
            clean_count=clean_count,
            blocked_count=blocked_count,
            flagged_count=flagged_count,
            missing_count=missing_count,
        ),
        "cross_corpus_matrix_built": cross_corpus_matrix_built,
        "verifier_corpus_dual_matrix": matrix,
        "row_status_by_corpus": row_status_by_corpus,
        "paper_eligible_rows": paper_eligible_rows,
        "clean_corpus_count": clean_count,
        "blocked_corpus_count": blocked_count,
        "flagged_corpus_count": flagged_count,
        "missing_corpus_count": missing_count,
        "source_artifacts": source_artifacts,
        "expected_source_artifacts": {
            corpus: str(rel_path) for corpus, rel_path in CORPUS_ARTIFACTS.items()
        },
        "excluded_from_headline": excluded_from_headline,
        "claim_boundary_notes": _claim_boundary_notes(
            row_status_by_corpus,
            paper_eligible_rows,
            cross_corpus_matrix_built,
        ),
        "field_principles": FIELD_PRINCIPLES,
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, end - start), 6),
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2865 matrix JSON deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())

"""Build the Exp 2880 clean .272 cross-corpus matrix v6 artifact.

Spec refs: REQ-REPORT-2880, SCENARIO-REPORT-2880.

The v6 matrix is a synthesis layer, not a new evaluation. It preserves the v5
headline boundary, adds the .272 HaluEval/FEVER support audits, and records the
MBPP/HumanEval execution pilot without converting it into a headline metric.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260522"
OUTPUT_REL_PATH = Path("results/experiment_2880_cross_corpus_matrix_v6.json")

MATRIX_V5_REL_PATH = Path("results/experiment_2865_cross_corpus_matrix_v5.json")
EXACT_FRONTIER_REL_PATH = Path(
    "results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json"
)
ERROR_VERIFIABILITY_REL_PATH = Path(
    "results/experiment_2878_halueval_fever_error_verifiability_v1.json"
)
CODE_PILOT_REL_PATH = Path("results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json")

SOURCE_ARTIFACTS: dict[str, Path] = {
    "matrix_v5": MATRIX_V5_REL_PATH,
    "exact_frontier": EXACT_FRONTIER_REL_PATH,
    "error_verifiability": ERROR_VERIFIABILITY_REL_PATH,
    "code_execution_pilot": CODE_PILOT_REL_PATH,
}
EXPECTED_CORPORA = ("FoVer", "HaluEval/FEVER", "MBPP", "HumanEval", "TruthfulQA")

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal synthesis verdict; no new benchmark result is implied.",
    "cross_corpus_matrix_built": (
        "True only when FoVer and HaluEval/FEVER remain headline-eligible after .272 gates."
    ),
    "source_artifacts": "Exactly the matrix v5 and three gated .272 artifacts loaded.",
    "clean_row_count": "Counts headline-eligible plus pilot-only rows, never missing rows.",
    "headline_eligible_rows": "Rows with clean label/metric evidence suitable for row-level claims.",
    "pilot_only_rows": "Rows with explicit pilot evidence that cannot support headline metrics.",
    "missing_rows": "Expected corpora or gated rows held out with null metrics and reasons.",
    "matrix_rows": "Eligible rows only; every row records whether it is headline or pilot-only.",
    "markdown_table": "Compact human-readable projection of the machine-readable row statuses.",
    "synthetic_rows_created": "Always false; v6 only synthesizes existing artifact fields.",
    "duration_s": "Measured wall time; never padded.",
}


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object, or an empty object when it cannot be trusted."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def classify_source_status(source_name: str, payload: dict[str, Any]) -> str:
    """Classify whether one required upstream source is usable for v6 synthesis."""

    if not payload:
        return "missing"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if not _complete_verdict(payload.get("honest_verdict")):
        return "unclean"
    if source_name == "matrix_v5":
        return "clean" if payload.get("cross_corpus_matrix_built") is True else "unclean"
    if source_name == "exact_frontier":
        return "clean" if payload.get("frontier_expansion_ready") is True else "unclean"
    if source_name == "error_verifiability":
        ready = payload.get("error_verifiability_ready") is True
        return "clean" if ready and payload.get("remote_llm_called") is False else "unclean"
    if source_name == "code_execution_pilot":
        ready = payload.get("code_manifest_pilot_ready") is True
        no_headline = payload.get("headline_metric_claim_made") is False
        return "clean" if ready and no_headline else "unclean"
    return "unclean"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2880: synthesize the clean v6 cross-corpus matrix artifact."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    payloads = {
        name: read_json(root_path / rel_path) for name, rel_path in SOURCE_ARTIFACTS.items()
    }
    statuses = {
        name: classify_source_status(name, payload) for name, payload in payloads.items()
    }
    rows: list[dict[str, Any]] = []
    missing_rows: dict[str, dict[str, Any]] = {}

    v5_payload = payloads["matrix_v5"]
    v5_matrix = dict(v5_payload.get("verifier_corpus_dual_matrix") or {})
    if statuses["matrix_v5"] == "clean" and _v5_row_is_clean(v5_payload, "FoVer"):
        rows.append(_fover_row(dict(v5_matrix["FoVer"])))
    else:
        missing_rows["FoVer"] = _missing_row("FoVer", "blocked_or_unclean_matrix_v5_source")

    halueval_sources_clean = (
        statuses["matrix_v5"] == "clean"
        and statuses["exact_frontier"] == "clean"
        and statuses["error_verifiability"] == "clean"
        and _v5_row_is_clean(v5_payload, "HaluEval/FEVER")
    )
    if halueval_sources_clean:
        rows.append(
            _halueval_fever_row(
                dict(v5_matrix["HaluEval/FEVER"]),
                payloads["exact_frontier"],
                payloads["error_verifiability"],
            )
        )
    else:
        missing_rows["HaluEval/FEVER"] = _missing_row(
            "HaluEval/FEVER",
            "blocked_or_unclean_dot272_halueval_fever_source",
        )

    if statuses["code_execution_pilot"] == "clean":
        code_rows = _code_pilot_rows(payloads["code_execution_pilot"])
        rows.extend(code_rows)
        present_code_corpora = {row["corpus"] for row in code_rows}
        for corpus in ("MBPP", "HumanEval"):
            if corpus not in present_code_corpora:
                missing_rows[corpus] = _missing_row(corpus, "code_pilot_row_missing")
    else:
        missing_rows["MBPP"] = _missing_row(
            "MBPP",
            "code_pilot_not_clean_or_claimed_headline_metric",
        )
        missing_rows["HumanEval"] = _missing_row(
            "HumanEval",
            "code_pilot_not_clean_or_claimed_headline_metric",
        )

    missing_rows["TruthfulQA"] = _missing_row(
        "TruthfulQA",
        "source_artifact_missing_in_v5_and_no_dot272_replacement",
    )
    rows = [row for row in rows if row["corpus"] in EXPECTED_CORPORA]
    headline_rows = [row["corpus"] for row in rows if row["row_status"] == "headline_eligible"]
    pilot_rows = [row["corpus"] for row in rows if row["row_status"] == "pilot_only"]
    cross_corpus_matrix_built = "FoVer" in headline_rows and "HaluEval/FEVER" in headline_rows
    end = time.perf_counter() if now_s is None else now_s

    return {
        "schema": "carnot.cross_corpus_matrix.v6",
        "artifact": "experiment_2880_cross_corpus_matrix_v6",
        "honest_verdict": _honest_verdict(
            cross_corpus_matrix_built=cross_corpus_matrix_built,
            rows=rows,
            statuses=statuses,
        ),
        "cross_corpus_matrix_built": cross_corpus_matrix_built,
        "source_artifacts": _existing_source_artifacts(root_path),
        "source_status_by_artifact": statuses,
        "clean_row_count": len(rows),
        "headline_eligible_rows": headline_rows,
        "pilot_only_rows": pilot_rows,
        "missing_rows": {
            corpus: missing_rows[corpus] for corpus in EXPECTED_CORPORA if corpus in missing_rows
        },
        "matrix_rows": rows,
        "markdown_table": _markdown_table(rows, missing_rows),
        "synthetic_rows_created": False,
        "field_principles": dict(FIELD_PRINCIPLES),
        "run_date": RUN_DATE,
        "duration_s": round(max(0.0, end - started), 6),
    }


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 2880 matrix deliverable."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def _blocked_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith(("blocked", "gate_blocked"))


def _complete_verdict(verdict: object) -> bool:
    return isinstance(verdict, str) and verdict.strip().startswith(("complete:", "success:"))


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _v5_row_is_clean(v5_payload: dict[str, Any], corpus: str) -> bool:
    statuses = dict(v5_payload.get("row_status_by_corpus") or {})
    matrix = dict(v5_payload.get("verifier_corpus_dual_matrix") or {})
    return statuses.get(corpus) == "clean" and isinstance(matrix.get(corpus), dict)


def _metric_null(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def _fover_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "corpus": "FoVer",
        "row_status": "headline_eligible",
        "headline_eligible": True,
        "pilot_only": False,
        "synthetic_row": False,
        "source_artifact": row.get("source_artifact"),
        "source_honest_verdict": row.get("honest_verdict"),
        "label_evidence": {
            "status": "valid_metric_panel",
            "reason": "v5 clean dual-condition AUROC row with n_examples and n_seeds.",
            "n_examples": row.get("n_examples"),
            "n_seeds": row.get("n_seeds"),
        },
        "primary_metric": {
            "production_auroc": _number_or_none(row.get("production_auroc")),
            "architecture_only_auroc": _number_or_none(row.get("architecture_only_auroc")),
            "learning_contribution": _number_or_none(row.get("learning_contribution")),
        },
        "exact_frontier_support": _metric_null("not_applicable_to_fover"),
        "error_verifiability": _metric_null("not_a_halueval_fever_error_audit_row"),
        "label_consistency": _metric_null("not_measured_by_dot272_label_consistency_audit"),
        "code_execution_pilot": _metric_null("not_a_code_execution_pilot_row"),
        "headline_metric_claim_made": True,
        "residual_gap": {
            "value": None,
            "reason": "no_dot272_residual_gap_audit_for_fover",
        },
    }


def _halueval_fever_row(
    row: dict[str, Any],
    exact_payload: dict[str, Any],
    error_payload: dict[str, Any],
) -> dict[str, Any]:
    candidate_rows = int(exact_payload.get("n_candidate_rows") or 0)
    supported_rows = int(exact_payload.get("n_exact_supported_rows") or 0)
    unsupported_rows = int(exact_payload.get("n_unsupported_rows") or 0)
    exact_rate = supported_rows / candidate_rows if candidate_rows else None
    return {
        "corpus": "HaluEval/FEVER",
        "row_status": "headline_eligible",
        "headline_eligible": True,
        "pilot_only": False,
        "synthetic_row": False,
        "source_artifact": row.get("source_artifact"),
        "source_honest_verdict": row.get("honest_verdict"),
        "label_evidence": {
            "status": "valid_labels",
            "label_counts_by_dataset": row.get("label_counts_by_dataset", {}),
            "n_rows_audited": error_payload.get("n_rows_audited"),
        },
        "primary_metric": {
            "measured_auroc_by_dataset": row.get("measured_auroc_by_dataset", {}),
            "n_examples_by_dataset": row.get("n_examples_by_dataset", {}),
        },
        "exact_frontier_support": {
            "value": exact_rate,
            "supported_rows": supported_rows,
            "candidate_rows": candidate_rows,
            "unsupported_rows": unsupported_rows,
            "unsupported_reasons": exact_payload.get("unsupported_reasons", {}),
            "reason": "manual exact certificates only; unsupported rows remain unsupported",
        },
        "error_verifiability": {
            "value": True,
            "actionable_localization_rate": _number_or_none(
                error_payload.get("actionable_localization_rate")
            ),
            "n_rows_audited": error_payload.get("n_rows_audited"),
            "bucket_level_metrics": error_payload.get("bucket_level_metrics", {}),
        },
        "label_consistency": {
            "value": _number_or_none(error_payload.get("label_consistency_rate")),
            "reason": "computed from existing local verifier direction versus manifest labels",
        },
        "code_execution_pilot": _metric_null("not_a_code_execution_pilot_row"),
        "headline_metric_claim_made": True,
        "residual_gap": {
            "value": "exact_frontier_limited_and_scalar_verifier_weak",
            "unsupported_exact_rows": unsupported_rows,
            "weak_auroc_explanation": error_payload.get("weak_auroc_explanation", ""),
        },
    }


def _code_pilot_rows(code_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for corpus in ("MBPP", "HumanEval"):
        pilot_row = _pilot_row_for_corpus(code_payload, corpus)
        if pilot_row:
            rows.append(_code_pilot_matrix_row(code_payload, pilot_row))
    return rows


def _pilot_row_for_corpus(code_payload: dict[str, Any], corpus: str) -> dict[str, Any]:
    for row in code_payload.get("pilot_rows", []):
        if isinstance(row, dict) and row.get("corpus") == corpus and row.get("passed") is True:
            return row
    return {}


def _code_pilot_matrix_row(code_payload: dict[str, Any], pilot_row: dict[str, Any]) -> dict[str, Any]:
    corpus = str(pilot_row["corpus"])
    return {
        "corpus": corpus,
        "row_status": "pilot_only",
        "headline_eligible": False,
        "pilot_only": True,
        "synthetic_row": False,
        "source_artifact": str(CODE_PILOT_REL_PATH),
        "source_honest_verdict": code_payload.get("honest_verdict"),
        "label_evidence": {
            "status": "explicit_pilot_status",
            "stable_id": pilot_row.get("stable_id"),
            "passed": pilot_row.get("passed"),
            "n_tests": pilot_row.get("n_tests"),
        },
        "primary_metric": _metric_null("pilot_only_no_generated_code_metric"),
        "exact_frontier_support": _metric_null("not_applicable_to_code_pilot"),
        "error_verifiability": _metric_null("not_measured_by_halueval_fever_audit"),
        "label_consistency": _metric_null("no_generated_code_labels_available"),
        "code_execution_pilot": {
            "value": "pilot_passed",
            "stable_id": pilot_row.get("stable_id"),
            "n_tests": pilot_row.get("n_tests"),
            "deterministic_execution_used": code_payload.get("deterministic_execution_used"),
            "sandbox_status": code_payload.get("sandbox_status"),
            "reason": "canonical/reference code execution only; no generated-code metric",
        },
        "headline_metric_claim_made": False,
        "residual_gap": {
            "value": "pilot_only_no_pass_at_k_or_auroc",
            "reason": "pilot only; no pass@k/AUROC",
        },
    }


def _missing_row(corpus: str, reason: str) -> dict[str, Any]:
    return {
        "corpus": corpus,
        "row_status": "missing",
        "headline_eligible": False,
        "pilot_only": False,
        "synthetic_row": False,
        "primary_metric": _metric_null(reason),
        "exact_frontier_support": _metric_null(reason),
        "error_verifiability": _metric_null(reason),
        "label_consistency": _metric_null(reason),
        "code_execution_pilot": _metric_null(reason),
        "residual_gap": {
            "value": "missing_source_artifact",
            "reason": "missing source artifact",
        },
    }


def _existing_source_artifacts(root: Path) -> list[str]:
    return [str(rel_path) for rel_path in SOURCE_ARTIFACTS.values() if (root / rel_path).is_file()]


def _honest_verdict(
    *,
    cross_corpus_matrix_built: bool,
    rows: list[dict[str, Any]],
    statuses: dict[str, str],
) -> str:
    headline_count = sum(row["row_status"] == "headline_eligible" for row in rows)
    pilot_count = sum(row["row_status"] == "pilot_only" for row in rows)
    if cross_corpus_matrix_built:
        return (
            "complete: cross-corpus matrix v6 built from "
            f"{headline_count} headline rows and {pilot_count} pilot-only rows"
        )
    unclean = {
        name: status for name, status in statuses.items() if status in {"blocked", "missing", "unclean"}
    }
    return (
        "complete: cross-corpus matrix v6 not headline-built; "
        f"blocked_or_unclean_sources_present={unclean}"
    )


def _markdown_table(
    rows: list[dict[str, Any]],
    missing_rows: dict[str, dict[str, Any]],
) -> str:
    by_corpus = {row["corpus"]: row for row in rows}
    lines = [
        "| Corpus | Status | Headline | Pilot | Exact frontier | Label consistency | Residual gap |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for corpus in EXPECTED_CORPORA:
        row = by_corpus.get(corpus) or missing_rows.get(corpus)
        if not row:
            row = _missing_row(corpus, "not_present")
        lines.append(
            "| "
            + " | ".join(
                [
                    corpus,
                    str(row["row_status"]),
                    "yes" if row["headline_eligible"] else "no",
                    "yes" if row["pilot_only"] else "no",
                    _exact_cell(row),
                    _label_cell(row),
                    _residual_cell(row),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _exact_cell(row: dict[str, Any]) -> str:
    support = row["exact_frontier_support"]
    value = support.get("value")
    if value is None:
        return "n/a"
    return f"{support['supported_rows']}/{support['candidate_rows']}"


def _label_cell(row: dict[str, Any]) -> str:
    value = row["label_consistency"].get("value")
    return "n/a" if value is None else f"{float(value):.3f}"


def _residual_cell(row: dict[str, Any]) -> str:
    residual = row["residual_gap"]
    if residual.get("reason") == "pilot only; no pass@k/AUROC":
        return "pilot only; no pass@k/AUROC"
    if residual.get("reason") == "missing source artifact":
        return "missing source artifact"
    if residual.get("unsupported_exact_rows") is not None:
        return f"{residual['unsupported_exact_rows']} exact-unsupported rows"
    return str(residual.get("reason") or residual.get("value") or "n/a")


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())

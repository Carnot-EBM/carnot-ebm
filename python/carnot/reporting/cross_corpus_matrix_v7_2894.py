"""Build the Exp 2894 clean .273 cross-corpus matrix v7 artifact.

Spec refs: REQ-REPORT-2894, SCENARIO-REPORT-2894.

The v7 matrix is a synthesis layer over completed artifacts. It does not run a
benchmark, does not repair earlier rows, and does not turn pilot or taxonomy
evidence into headline metrics. The point is to keep the new support columns
visible while preserving the claim boundary established by matrix v6.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260523"
OUTPUT_REL_PATH = Path("results/experiment_2894_cross_corpus_matrix_v7.json")

MATRIX_V6_REL_PATH = Path("results/experiment_2880_cross_corpus_matrix_v6.json")
TRUTHFULQA_TAXONOMY_REL_PATH = Path(
    "results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json"
)
GENERATED_CODE_REL_PATH = Path(
    "results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json"
)
STRUCTURAL_VERIFIER_REL_PATH = Path(
    "results/experiment_2890_code_structural_dependency_verifier_v1.json"
)
CCTU_VALIDATOR_REL_PATH = Path(
    "results/experiment_2891_cctu_executable_constraint_validator_pilot_v1.json"
)
VERICOT_FRONTIER_REL_PATH = Path("results/experiment_2892_vericot_exact_frontier_expansion_v1.json")
KAN_COMPLEXITY_REL_PATH = Path("results/experiment_2893_kan_hardware_complexity_accounting_v1.json")

SOURCE_ARTIFACTS: dict[str, Path] = {
    "matrix_v6": MATRIX_V6_REL_PATH,
    "truthfulqa_taxonomy": TRUTHFULQA_TAXONOMY_REL_PATH,
    "generated_code": GENERATED_CODE_REL_PATH,
    "structural_verifier": STRUCTURAL_VERIFIER_REL_PATH,
    "cctu_validator": CCTU_VALIDATOR_REL_PATH,
    "vericot_frontier": VERICOT_FRONTIER_REL_PATH,
    "kan_complexity": KAN_COMPLEXITY_REL_PATH,
}
EXPECTED_CORPORA = ("FoVer", "HaluEval/FEVER", "MBPP", "HumanEval", "TruthfulQA")
CODE_CORPORA = {"MBPP", "HumanEval"}

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal synthesis verdict; no new benchmark run is implied.",
    "cross_corpus_matrix_built": (
        "True when matrix v6 still supplies FoVer and HaluEval/FEVER headline rows."
    ),
    "source_artifacts": "Existing matrix v6 and .273 support artifacts loaded from disk.",
    "clean_row_count": "Counts headline, pilot-only, and taxonomy-only rows admitted to v7.",
    "headline_eligible_rows": "Rows still supported by clean headline metric evidence.",
    "pilot_only_rows": "Rows with explicit pilot status and no headline metric promotion.",
    "taxonomy_only_rows": "Rows with local taxonomy evidence but no generated-answer metric.",
    "blocked_rows": "Row support blocked by unresolved flags or unclean source evidence.",
    "missing_rows": "Expected rows absent from the matrix with null fields and reasons.",
    "matrix_rows": "Machine-readable v7 rows; unsupported columns remain null with reasons.",
    "markdown_table": "Compact projection of the same row statuses and support columns.",
    "synthetic_rows_created": "Always false; v7 never fabricates rows or metric cells.",
    "duration_s": "Measured wall time; never padded.",
}


def read_json(path: Path) -> dict[str, Any]:
    """Return a JSON object from disk, or an empty object when it is unusable."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def has_unresolved_flags(payload: dict[str, Any]) -> bool:
    """Return whether an artifact carries unresolved adversarial/corrigendum flags."""

    if payload.get("flagged_adversarial") is True:
        return True
    if payload.get("corrigendum_pending"):
        return True
    if payload.get("adversarial_verify_passed") is False:
        return True
    if payload.get("adversarial_verify_flags"):
        return True
    summary = payload.get("adversarial_verify_summary")
    return isinstance(summary, dict) and int(summary.get("flag_count") or 0) > 0


def classify_source_status(source_name: str, payload: dict[str, Any]) -> str:
    """Classify one upstream source before any row can consume its fields."""

    if not payload:
        return "missing"
    if _blocked_verdict(payload.get("honest_verdict")):
        return "blocked"
    if has_unresolved_flags(payload):
        return "flagged"
    if not _complete_verdict(payload.get("honest_verdict")):
        return "unclean"
    if source_name == "matrix_v6":
        return "clean" if payload.get("cross_corpus_matrix_built") is True else "unclean"
    if source_name == "truthfulqa_taxonomy":
        ready = payload.get("truthfulqa_taxonomy_ready") is True
        no_headline = payload.get("headline_metric_claim_made") is False
        local_only = (
            payload.get("remote_llm_called") is False
            and payload.get("synthetic_labels_created") is False
        )
        return "clean" if ready and no_headline and local_only else "unclean"
    if source_name == "generated_code":
        ready = payload.get("generated_code_row_clean") is True
        no_headline = payload.get("headline_metric_claim_made") is False
        sandboxed = str(payload.get("sandbox_status", "")).startswith("available")
        return (
            "clean"
            if ready and no_headline and sandboxed and payload.get("deterministic_execution_used")
            else "unclean"
        )
    if source_name == "structural_verifier":
        ready = payload.get("structural_dependency_verifier_ready") is True
        return (
            "clean" if ready and payload.get("headline_metric_claim_made") is False else "unclean"
        )
    if source_name == "cctu_validator":
        ready = payload.get("cctu_validator_ready") is True
        bounded = (
            payload.get("headline_metric_claim_made") is False
            and payload.get("executable_validation_used") is True
            and payload.get("live_llm_called") is False
        )
        return "clean" if ready and bounded else "unclean"
    if source_name == "vericot_frontier":
        ready = payload.get("vericot_frontier_ready") is True
        return (
            "clean" if ready and payload.get("autoformalization_llm_called") is False else "unclean"
        )
    if source_name == "kan_complexity":
        ready = payload.get("kan_complexity_accounting_ready") is True
        bounded = (
            payload.get("hardware_execution_claim_made") is False
            and payload.get("analog_kan_claim_made") is False
        )
        return "clean" if ready and bounded else "unclean"
    return "unclean"


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> dict[str, Any]:
    """REQ-REPORT-2894: synthesize the v7 matrix from clean source fields."""

    root_path = Path(root)
    started = time.perf_counter() if started_s is None else started_s
    payloads = {
        name: read_json(root_path / rel_path) for name, rel_path in SOURCE_ARTIFACTS.items()
    }
    statuses = {name: classify_source_status(name, payload) for name, payload in payloads.items()}

    matrix_rows: list[dict[str, Any]] = []
    missing_rows: dict[str, dict[str, Any]] = {}
    blocked_rows: dict[str, dict[str, Any]] = {}

    if statuses["matrix_v6"] == "clean":
        for row in _v6_rows(payloads["matrix_v6"]):
            corpus = str(row.get("corpus", ""))
            if corpus in EXPECTED_CORPORA and corpus != "TruthfulQA":
                matrix_rows.append(_v7_row(corpus, row, payloads, statuses, blocked_rows))
    else:
        for corpus in ("FoVer", "HaluEval/FEVER", "MBPP", "HumanEval"):
            missing_rows[corpus] = _missing_row(corpus, "matrix_v6_source_not_clean")

    if statuses["truthfulqa_taxonomy"] == "clean":
        matrix_rows.append(_truthfulqa_row(payloads, statuses))
    else:
        missing_rows["TruthfulQA"] = _missing_row(
            "TruthfulQA",
            "truthfulqa_taxonomy_source_not_clean",
        )

    by_corpus = {row["corpus"]: row for row in matrix_rows}
    for corpus in EXPECTED_CORPORA:
        if corpus not in by_corpus and corpus not in missing_rows:
            missing_rows[corpus] = _missing_row(corpus, "row_not_present_in_clean_sources")

    ordered_rows = [by_corpus[corpus] for corpus in EXPECTED_CORPORA if corpus in by_corpus]
    headline_rows = [
        row["corpus"] for row in ordered_rows if row["row_status"] == "headline_eligible"
    ]
    pilot_rows = [row["corpus"] for row in ordered_rows if row["row_status"] == "pilot_only"]
    taxonomy_rows = [row["corpus"] for row in ordered_rows if row["row_status"] == "taxonomy_only"]
    cross_corpus_matrix_built = "FoVer" in headline_rows and "HaluEval/FEVER" in headline_rows
    end = time.perf_counter() if now_s is None else now_s

    return {
        "schema": "carnot.cross_corpus_matrix.v7",
        "artifact": "experiment_2894_cross_corpus_matrix_v7",
        "honest_verdict": _honest_verdict(
            cross_corpus_matrix_built=cross_corpus_matrix_built,
            row_count=len(ordered_rows),
            blocked_rows=blocked_rows,
        ),
        "cross_corpus_matrix_built": cross_corpus_matrix_built,
        "source_artifacts": _existing_source_artifacts(root_path),
        "source_status_by_artifact": statuses,
        "clean_row_count": len(ordered_rows),
        "headline_eligible_rows": headline_rows,
        "pilot_only_rows": pilot_rows,
        "taxonomy_only_rows": taxonomy_rows,
        "blocked_rows": {corpus: blocked_rows[corpus] for corpus in sorted(blocked_rows)},
        "missing_rows": {
            corpus: missing_rows[corpus] for corpus in EXPECTED_CORPORA if corpus in missing_rows
        },
        "matrix_rows": ordered_rows,
        "markdown_table": _markdown_table(ordered_rows, missing_rows),
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
    """Build and persist the Exp 2894 matrix deliverable."""

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


def _v6_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("matrix_rows", [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _metric_null(reason: str, *, status: str | None = None) -> dict[str, Any]:
    cell: dict[str, Any] = {"value": None, "reason": reason}
    if status is not None:
        cell["source_status"] = status
    return cell


def _v7_row(
    corpus: str,
    base_row: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    statuses: dict[str, str],
    blocked_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    generated_code = _generated_code_status(corpus, payloads["generated_code"], statuses)
    if generated_code.get("status") == "blocked_unresolved_adversarial_flags":
        blocked_rows[corpus] = {
            "generated_code_status": "blocked_unresolved_adversarial_flags",
            "source_artifact": str(GENERATED_CODE_REL_PATH),
            "reasons": generated_code.get("flag_reasons", []),
        }
    structural = _structural_dependency_verification(
        corpus,
        payloads["structural_verifier"],
        statuses,
    )
    cctu = _cctu_constraint_category_coverage(payloads["cctu_validator"], statuses)
    vericot = _vericot_exact_support(
        corpus,
        payloads["vericot_frontier"],
        payloads["truthfulqa_taxonomy"],
        statuses,
    )
    kan = _kan_complexity(payloads["kan_complexity"], statuses)
    return {
        "corpus": corpus,
        "row_status": base_row.get("row_status"),
        "headline_eligible": base_row.get("headline_eligible") is True,
        "pilot_only": base_row.get("pilot_only") is True,
        "taxonomy_only": False,
        "synthetic_row": False,
        "source_artifact": base_row.get("source_artifact"),
        "source_honest_verdict": base_row.get("source_honest_verdict"),
        "label_evidence": base_row.get("label_evidence", {}),
        "primary_metric": base_row.get("primary_metric", _metric_null("primary_metric_missing")),
        "truthfulqa_taxonomy": _metric_null("not_truthfulqa_taxonomy_row"),
        "generated_code_status": generated_code,
        "structural_dependency_verification": structural,
        "cctu_constraint_category_coverage": cctu,
        "vericot_exact_support": vericot,
        "kan_complexity": kan,
        "headline_metric_claim_made": base_row.get("headline_eligible") is True,
        "residual_gap": _residual_gap(corpus, generated_code, vericot),
    }


def _truthfulqa_row(
    payloads: dict[str, dict[str, Any]],
    statuses: dict[str, str],
) -> dict[str, Any]:
    taxonomy = payloads["truthfulqa_taxonomy"]
    vericot = _vericot_exact_support(
        "TruthfulQA",
        payloads["vericot_frontier"],
        taxonomy,
        statuses,
    )
    return {
        "corpus": "TruthfulQA",
        "row_status": "taxonomy_only",
        "headline_eligible": False,
        "pilot_only": False,
        "taxonomy_only": True,
        "synthetic_row": False,
        "source_artifact": str(TRUTHFULQA_TAXONOMY_REL_PATH),
        "source_honest_verdict": taxonomy.get("honest_verdict"),
        "label_evidence": {
            "status": "taxonomy_only",
            "n_rows_available": taxonomy.get("n_rows_available"),
            "n_rows_materialized": taxonomy.get("n_rows_materialized"),
            "reason": "local taxonomy rows exist, but generated-answer metrics are unavailable.",
        },
        "primary_metric": _metric_null("taxonomy_only_no_generated_answer_metrics"),
        "truthfulqa_taxonomy": {
            "value": "taxonomy_available",
            "n_rows_available": taxonomy.get("n_rows_available"),
            "n_rows_materialized": taxonomy.get("n_rows_materialized"),
            "taxonomy_fields": taxonomy.get("taxonomy_fields", []),
            "error_type_counts": taxonomy.get("error_type_counts", {}),
            "generated_answer_metrics_available": taxonomy.get(
                "generated_answer_metrics_available"
            ),
            "synthetic_labels_created": taxonomy.get("synthetic_labels_created"),
            "headline_metric_claim_made": taxonomy.get("headline_metric_claim_made"),
        },
        "generated_code_status": _metric_null("not_a_code_corpus"),
        "structural_dependency_verification": _metric_null("not_a_code_corpus"),
        "cctu_constraint_category_coverage": _cctu_constraint_category_coverage(
            payloads["cctu_validator"],
            statuses,
        ),
        "vericot_exact_support": vericot,
        "kan_complexity": _kan_complexity(payloads["kan_complexity"], statuses),
        "headline_metric_claim_made": False,
        "residual_gap": {
            "value": "taxonomy_only_no_generated_answer_metrics",
            "reason": "taxonomy only; no generated-answer metrics",
        },
    }


def _generated_code_status(
    corpus: str,
    generated_payload: dict[str, Any],
    statuses: dict[str, str],
) -> dict[str, Any]:
    status = statuses["generated_code"]
    if corpus not in CODE_CORPORA:
        return _metric_null("not_a_code_corpus")
    if status == "flagged":
        return {
            "value": None,
            "status": "blocked_unresolved_adversarial_flags",
            "reason": "Exp 2889 has unresolved generated-code flags; no metric promoted.",
            "source_artifact": str(GENERATED_CODE_REL_PATH),
            "flag_reasons": _flag_reasons(generated_payload),
        }
    if status != "clean":
        return _metric_null("generated_code_source_not_clean", status=status)

    rows = [
        row
        for row in generated_payload.get("row_results", [])
        if isinstance(row, dict) and row.get("corpus") == corpus
    ]
    valid_rows = [row for row in rows if isinstance(row.get("n_tests"), int) and "passed" in row]
    return {
        "value": "available" if valid_rows else None,
        "status": generated_payload.get("row_status"),
        "source_artifact": str(GENERATED_CODE_REL_PATH),
        "n_generated_outputs": len(valid_rows),
        "n_passed": sum(1 for row in valid_rows if row.get("passed") is True),
        "headline_metric_claim_made": generated_payload.get("headline_metric_claim_made"),
        "reason": "pilot-only generated-code labels/tests available; no headline metric promoted",
    }


def _structural_dependency_verification(
    corpus: str,
    structural_payload: dict[str, Any],
    statuses: dict[str, str],
) -> dict[str, Any]:
    status = statuses["structural_verifier"]
    if corpus not in CODE_CORPORA:
        return _metric_null("not_a_code_corpus")
    if status != "clean":
        return _metric_null("structural_dependency_source_not_clean", status=status)

    rows = [
        row
        for row in structural_payload.get("verification_rows", [])
        if isinstance(row, dict) and row.get("corpus") == corpus
    ]
    reference_rows = [row for row in rows if row.get("candidate_kind") == "reference"]
    generated_rows = [row for row in rows if row.get("candidate_kind") == "generated_exp2889"]
    violation_types: Counter[str] = Counter()
    for row in generated_rows:
        for violation in row.get("violations", []):
            if isinstance(violation, dict) and violation.get("violation_type"):
                violation_types[str(violation["violation_type"])] += 1
    return {
        "value": "available",
        "source_artifact": str(STRUCTURAL_VERIFIER_REL_PATH),
        "n_rows_verified": len(rows),
        "reference_rows": len(reference_rows),
        "reference_passed": sum(1 for row in reference_rows if row.get("passed") is True),
        "generated_candidate_rows": len(generated_rows),
        "generated_candidate_passed": sum(1 for row in generated_rows if row.get("passed") is True),
        "violation_types": dict(sorted(violation_types.items())),
        "unsupported_reasons": structural_payload.get("unsupported_reasons", {}),
        "reason": "static AST structural metadata only; no pass@k/AUROC headline metric",
    }


def _cctu_constraint_category_coverage(
    cctu_payload: dict[str, Any],
    statuses: dict[str, str],
) -> dict[str, Any]:
    status = statuses["cctu_validator"]
    if status != "clean":
        return _metric_null("cctu_validator_source_not_clean", status=status)
    return {
        "value": "available",
        "scope": "global_support_artifact",
        "source_artifact": str(CCTU_VALIDATOR_REL_PATH),
        "n_cases": cctu_payload.get("n_cases"),
        "constraint_categories": cctu_payload.get("constraint_categories", []),
        "category_coverage": cctu_payload.get("category_coverage", {}),
        "unsupported_categories": cctu_payload.get("unsupported_categories", {}),
        "headline_metric_claim_made": cctu_payload.get("headline_metric_claim_made"),
        "reason": "CCTU pilot coverage is global support metadata, not a corpus metric.",
    }


def _vericot_exact_support(
    corpus: str,
    vericot_payload: dict[str, Any],
    taxonomy_payload: dict[str, Any],
    statuses: dict[str, str],
) -> dict[str, Any]:
    status = statuses["vericot_frontier"]
    if status != "clean":
        return _metric_null("vericot_frontier_source_not_clean", status=status)

    unsupported = dict(vericot_payload.get("unsupported_reasons") or {})
    truthfulqa_candidates = int(
        unsupported.get("unsupported_truthfulqa_taxonomy_has_no_logical_steps")
        or taxonomy_payload.get("n_rows_materialized")
        or 0
    )
    if corpus == "HaluEval/FEVER":
        candidate_rows = max(
            0, int(vericot_payload.get("n_candidate_rows") or 0) - truthfulqa_candidates
        )
        supported_rows = int(vericot_payload.get("n_vericot_supported_rows") or 0)
        return {
            "value": supported_rows / candidate_rows if candidate_rows else None,
            "source_artifact": str(VERICOT_FRONTIER_REL_PATH),
            "supported_rows": supported_rows,
            "candidate_rows": candidate_rows,
            "unsupported_rows": max(0, candidate_rows - supported_rows),
            "solver_backend": vericot_payload.get("solver_backend"),
            "unsupported_reasons": unsupported,
            "reason": "deterministic VeriCoT support for HaluEval/FEVER only",
        }
    if corpus == "TruthfulQA":
        return {
            "value": 0.0 if truthfulqa_candidates else None,
            "source_artifact": str(VERICOT_FRONTIER_REL_PATH),
            "supported_rows": 0,
            "candidate_rows": truthfulqa_candidates,
            "unsupported_rows": truthfulqa_candidates,
            "solver_backend": vericot_payload.get("solver_backend"),
            "unsupported_reasons": {
                "unsupported_truthfulqa_taxonomy_has_no_logical_steps": truthfulqa_candidates
            },
            "reason": "TruthfulQA taxonomy has no deterministic logical-step template yet",
        }
    return _metric_null("vericot_not_applicable_to_corpus")


def _kan_complexity(
    kan_payload: dict[str, Any],
    statuses: dict[str, str],
) -> dict[str, Any]:
    status = statuses["kan_complexity"]
    if status != "clean":
        return _metric_null("kan_complexity_source_not_clean", status=status)
    return {
        "value": "available",
        "scope": "global_support_artifact",
        "source_artifact": str(KAN_COMPLEXITY_REL_PATH),
        "pwa_regions": kan_payload.get("pwa_regions"),
        "nabs_count": kan_payload.get("nabs_count"),
        "bop_count": kan_payload.get("bop_count"),
        "rm_count": kan_payload.get("rm_count"),
        "milp_constraints": kan_payload.get("milp_constraints"),
        "memory_table_entries": kan_payload.get("memory_table_entries"),
        "hardware_claim_boundary": kan_payload.get("hardware_claim_boundary", {}),
        "hardware_execution_claim_made": kan_payload.get("hardware_execution_claim_made"),
        "analog_kan_claim_made": kan_payload.get("analog_kan_claim_made"),
        "reason": "KAN complexity is accounting metadata, not hardware execution.",
    }


def _residual_gap(
    corpus: str,
    generated_code: dict[str, Any],
    vericot: dict[str, Any],
) -> dict[str, Any]:
    if corpus == "HaluEval/FEVER" and vericot.get("value") is not None:
        return {
            "value": "vericot_support_partial",
            "reason": "VeriCoT support remains partial",
            "unsupported_rows": vericot.get("unsupported_rows"),
        }
    if (
        corpus in CODE_CORPORA
        and generated_code.get("status") == "blocked_unresolved_adversarial_flags"
    ):
        return {
            "value": "pilot_only_generated_code_flags_unresolved",
            "reason": "pilot only; generated-code flags unresolved",
        }
    if corpus in CODE_CORPORA:
        return {
            "value": "pilot_only_no_pass_at_k_or_auroc",
            "reason": "pilot only; no pass@k/AUROC",
        }
    return {
        "value": None,
        "reason": "no_new_dot273_residual_gap_metric",
    }


def _missing_row(corpus: str, reason: str) -> dict[str, Any]:
    return {
        "corpus": corpus,
        "row_status": "missing",
        "headline_eligible": False,
        "pilot_only": False,
        "taxonomy_only": False,
        "synthetic_row": False,
        "primary_metric": _metric_null(reason),
        "truthfulqa_taxonomy": _metric_null(reason),
        "generated_code_status": _metric_null(reason),
        "structural_dependency_verification": _metric_null(reason),
        "cctu_constraint_category_coverage": _metric_null(reason),
        "vericot_exact_support": _metric_null(reason),
        "kan_complexity": _metric_null(reason),
        "residual_gap": {
            "value": "missing_source_artifact",
            "reason": reason,
        },
    }


def _flag_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if payload.get("flagged_adversarial") is True:
        reasons.append("flagged_adversarial=true")
    for item in payload.get("corrigendum_pending", []):
        if isinstance(item, dict):
            kind = item.get("kind", "corrigendum_pending")
            severity = item.get("severity", "unknown")
            reasons.append(f"{kind}:{severity}")
    if payload.get("adversarial_verify_passed") is False:
        reasons.append("adversarial_verify_passed=false")
    if payload.get("adversarial_verify_flags"):
        reasons.append("adversarial_verify_flags_present")
    return reasons


def _existing_source_artifacts(root: Path) -> list[str]:
    return [str(rel_path) for rel_path in SOURCE_ARTIFACTS.values() if (root / rel_path).is_file()]


def _honest_verdict(
    *,
    cross_corpus_matrix_built: bool,
    row_count: int,
    blocked_rows: dict[str, dict[str, Any]],
) -> str:
    if cross_corpus_matrix_built:
        blocked_count = len(blocked_rows)
        return (
            "complete: cross-corpus matrix v7 built from "
            f"{row_count} clean headline/pilot/taxonomy rows; "
            f"blocked_support_rows={blocked_count}"
        )
    return "complete: cross-corpus matrix v7 not headline-built; matrix v6 source not clean"


def _markdown_table(
    rows: list[dict[str, Any]],
    missing_rows: dict[str, dict[str, Any]],
) -> str:
    by_corpus = {row["corpus"]: row for row in rows}
    lines = [
        "| Corpus | Status | Headline | Pilot | Taxonomy | Generated code | VeriCoT | Residual gap |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for corpus in EXPECTED_CORPORA:
        row = (
            by_corpus.get(corpus)
            or missing_rows.get(corpus)
            or _missing_row(
                corpus,
                "row_not_present",
            )
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    corpus,
                    str(row["row_status"]),
                    "yes" if row["headline_eligible"] else "no",
                    "yes" if row["pilot_only"] else "no",
                    "yes" if row["taxonomy_only"] else "no",
                    _generated_cell(row),
                    _vericot_cell(row),
                    _residual_cell(row),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _generated_cell(row: dict[str, Any]) -> str:
    cell = row["generated_code_status"]
    if cell.get("status") == "blocked_unresolved_adversarial_flags":
        return "blocked"
    if cell.get("value") == "available":
        return "available"
    return "n/a"


def _vericot_cell(row: dict[str, Any]) -> str:
    cell = row["vericot_exact_support"]
    if "supported_rows" in cell and "candidate_rows" in cell:
        return f"{cell['supported_rows']}/{cell['candidate_rows']}"
    return "n/a"


def _residual_cell(row: dict[str, Any]) -> str:
    residual = row["residual_gap"]
    return str(residual.get("reason") or residual.get("value") or "n/a")


if __name__ == "__main__":  # pragma: no cover
    print(write_artifact())

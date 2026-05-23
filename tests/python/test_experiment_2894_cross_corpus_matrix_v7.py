"""Tests for Exp 2894 clean .273 cross-corpus matrix v7 generation.

Spec refs: REQ-REPORT-2894, SCENARIO-REPORT-2894.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v7_2894 as exp2894


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metric_null(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def _v6_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2880_cross_corpus_matrix_v6",
        "honest_verdict": "complete: cross-corpus matrix v6 built from 2 headline rows and 2 pilot-only rows",
        "cross_corpus_matrix_built": True,
        "headline_eligible_rows": ["FoVer", "HaluEval/FEVER"],
        "pilot_only_rows": ["MBPP", "HumanEval"],
        "missing_rows": {"TruthfulQA": {"row_status": "missing"}},
        "matrix_rows": [
            {
                "corpus": "FoVer",
                "row_status": "headline_eligible",
                "headline_eligible": True,
                "pilot_only": False,
                "synthetic_row": False,
                "source_artifact": "results/experiment_2850_fover_dual_condition_integrity_v4.json",
                "source_honest_verdict": "complete: FoVer dual-condition integrity rerun",
                "label_evidence": {"status": "valid_metric_panel", "n_examples": 1000},
                "primary_metric": {
                    "production_auroc": 0.9131336,
                    "architecture_only_auroc": 0.8946624,
                    "learning_contribution": 0.0184712,
                },
                "residual_gap": {"value": None, "reason": "no_dot272_residual_gap_audit_for_fover"},
            },
            {
                "corpus": "HaluEval/FEVER",
                "row_status": "headline_eligible",
                "headline_eligible": True,
                "pilot_only": False,
                "synthetic_row": False,
                "source_artifact": "results/experiment_2864_halueval_fever_full_calibration_v3.json",
                "source_honest_verdict": "complete: HaluEval/FEVER local calibration ready",
                "label_evidence": {"status": "valid_labels", "n_rows_audited": 1000},
                "primary_metric": {
                    "measured_auroc_by_dataset": {
                        "halueval": 0.553072,
                        "fever": 0.33114331723027374,
                    },
                    "n_examples_by_dataset": {"halueval": 500, "fever": 500},
                },
                "residual_gap": {
                    "value": "exact_frontier_limited_and_scalar_verifier_weak",
                    "unsupported_exact_rows": 992,
                },
            },
            {
                "corpus": "MBPP",
                "row_status": "pilot_only",
                "headline_eligible": False,
                "pilot_only": True,
                "synthetic_row": False,
                "source_artifact": "results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json",
                "source_honest_verdict": "complete: MBPP/HumanEval manifest-only execution pilot ready",
                "label_evidence": {
                    "status": "explicit_pilot_status",
                    "stable_id": "mbpp-11",
                    "passed": True,
                    "n_tests": 3,
                },
                "primary_metric": _metric_null("pilot_only_no_generated_code_metric"),
                "residual_gap": {
                    "value": "pilot_only_no_pass_at_k_or_auroc",
                    "reason": "pilot only; no pass@k/AUROC",
                },
            },
            {
                "corpus": "HumanEval",
                "row_status": "pilot_only",
                "headline_eligible": False,
                "pilot_only": True,
                "synthetic_row": False,
                "source_artifact": "results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json",
                "source_honest_verdict": "complete: MBPP/HumanEval manifest-only execution pilot ready",
                "label_evidence": {
                    "status": "explicit_pilot_status",
                    "stable_id": "HumanEval/0",
                    "passed": True,
                    "n_tests": 7,
                },
                "primary_metric": _metric_null("pilot_only_no_generated_code_metric"),
                "residual_gap": {
                    "value": "pilot_only_no_pass_at_k_or_auroc",
                    "reason": "pilot only; no pass@k/AUROC",
                },
            },
        ],
    }


def _truthfulqa_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: TruthfulQA local taxonomy manifest ready without generated-answer metrics",
        "truthfulqa_taxonomy_ready": True,
        "n_rows_available": 200,
        "n_rows_materialized": 100,
        "taxonomy_fields": [
            "factual_error_type",
            "evidence_available",
            "justification_available",
            "correction_available",
            "unsupported_reason",
            "metric_eligibility",
        ],
        "error_type_counts": {"common_misconception": 20, "fictional_premise": 24},
        "generated_answer_metrics_available": False,
        "headline_metric_claim_made": False,
        "remote_llm_called": False,
        "synthetic_labels_created": False,
        "source_artifacts": [{"path": "results/experiment_2863_eval_manifest_contract_v2.json"}],
    }


def _generated_code_payload(*, flagged: bool = True) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: bounded SOTA GGUF generation executed cleanly but no candidate passed tests",
        "generated_code_row_clean": True,
        "row_status": "pilot_only_clean_no_passes",
        "headline_metric_claim_made": False,
        "deterministic_execution_used": True,
        "sandbox_status": "available: runsc",
        "flagged_adversarial": flagged,
        "corrigendum_pending": (
            [{"kind": "DURATION_TOO_SHORT", "severity": "critical"}] if flagged else []
        ),
        "n_generated_outputs": 2,
        "pass_rate_if_computable": 0.0,
        "row_results": [
            {
                "corpus": "MBPP",
                "stable_id": "mbpp-11",
                "row_status": "pilot_only_failed",
                "passed": False,
                "n_tests": 3,
            },
            {
                "corpus": "HumanEval",
                "stable_id": "HumanEval/0",
                "row_status": "pilot_only_failed",
                "passed": False,
                "n_tests": 7,
            },
        ],
    }


def _structural_payload() -> dict[str, Any]:
    rows = [
        ("MBPP", "reference", True, []),
        ("MBPP", "generated_exp2889", False, [{"violation_type": "parse_error"}]),
        ("HumanEval", "reference", True, []),
        (
            "HumanEval",
            "generated_exp2889",
            False,
            [{"violation_type": "missing_function_definition"}],
        ),
    ]
    return {
        "honest_verdict": "complete: MBPP/HumanEval structural dependency verifier metadata ready",
        "structural_dependency_verifier_ready": True,
        "n_contracts_built": 2,
        "n_rows_verified": 4,
        "generated_outputs_consumed": True,
        "headline_metric_claim_made": False,
        "violation_types": {"parse_error": 1, "missing_function_definition": 1},
        "verification_rows": [
            {
                "corpus": corpus,
                "candidate_kind": candidate_kind,
                "passed": passed,
                "stable_id": f"{corpus}:0",
                "violations": violations,
            }
            for corpus, candidate_kind, passed, violations in rows
        ],
        "unsupported_reasons": {},
    }


def _cctu_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: local CCTU-style executable constraint validator pilot ready",
        "cctu_validator_ready": True,
        "headline_metric_claim_made": False,
        "executable_validation_used": True,
        "live_llm_called": False,
        "n_cases": 5,
        "constraint_categories": [
            "behavior",
            "resource",
            "response",
            "response_verifier",
            "toolset",
        ],
        "category_coverage": {
            "behavior": {"passed": 0, "total": 1},
            "resource": {"passed": 0, "total": 1},
            "response": {"passed": 0, "total": 1},
            "response_verifier": {"passed": 0, "total": 1},
            "toolset": {"passed": 0, "total": 1},
        },
        "unsupported_categories": {"multi_turn_state": {"supported": False}},
    }


def _vericot_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: deterministic VeriCoT frontier rows available",
        "vericot_frontier_ready": True,
        "n_candidate_rows": 1100,
        "n_vericot_supported_rows": 25,
        "n_unsupported_rows": 1075,
        "unsupported_reasons": {
            "unsupported_no_deterministic_vericot_template": 974,
            "unsupported_truthfulqa_taxonomy_has_no_logical_steps": 100,
            "unsupported_year_only_does_not_establish_entity_grounding": 1,
        },
        "solver_backend": "z3-solver 4.16.0 + deterministic premise anchors",
        "autoformalization_llm_called": False,
        "formal_checks": [
            {"dataset": "HaluEval", "stable_id": "halueval-8-right"},
            {"dataset": "FEVER", "stable_id": "fever-84514"},
        ],
    }


def _kan_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: tiny KAN PWA/MILP complexity accounting ready; no hardware execution or analog claim",
        "kan_complexity_accounting_ready": True,
        "status": "complete",
        "hardware_execution_claim_made": False,
        "analog_kan_claim_made": False,
        "complexity_metrics": {
            "pwa_regions": 4,
            "nabs_count": 4,
            "bop_count": 96,
            "rm_count": 2,
            "milp_constraints": 27,
            "memory_table_entries": 8,
        },
        "pwa_regions": 4,
        "nabs_count": 4,
        "bop_count": 96,
        "rm_count": 2,
        "milp_constraints": 27,
        "memory_table_entries": 8,
        "hardware_claim_boundary": {"board_execution_run": False},
    }


def _write_clean_sources(root: Path, *, generated_flagged: bool = True) -> None:
    _write_json(root, str(exp2894.MATRIX_V6_REL_PATH), _v6_payload())
    _write_json(root, str(exp2894.TRUTHFULQA_TAXONOMY_REL_PATH), _truthfulqa_payload())
    _write_json(
        root,
        str(exp2894.GENERATED_CODE_REL_PATH),
        _generated_code_payload(flagged=generated_flagged),
    )
    _write_json(root, str(exp2894.STRUCTURAL_VERIFIER_REL_PATH), _structural_payload())
    _write_json(root, str(exp2894.CCTU_VALIDATOR_REL_PATH), _cctu_payload())
    _write_json(root, str(exp2894.VERICOT_FRONTIER_REL_PATH), _vericot_payload())
    _write_json(root, str(exp2894.KAN_COMPLEXITY_REL_PATH), _kan_payload())


def _rows_by_corpus(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["corpus"]: row for row in artifact["matrix_rows"]}


def test_scenario_report_2894_builds_clean_v7_with_row_boundaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2894: headline, pilot-only, and taxonomy-only rows stay distinct."""

    _write_clean_sources(tmp_path)

    artifact = exp2894.build_artifact(tmp_path, started_s=10.0, now_s=12.75)

    required = {
        "honest_verdict",
        "cross_corpus_matrix_built",
        "source_artifacts",
        "clean_row_count",
        "headline_eligible_rows",
        "pilot_only_rows",
        "taxonomy_only_rows",
        "blocked_rows",
        "missing_rows",
        "matrix_rows",
        "markdown_table",
        "synthetic_rows_created",
        "field_principles",
        "run_date",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["run_date"] == "20260523"
    assert artifact["duration_s"] == pytest.approx(2.75)
    assert artifact["synthetic_rows_created"] is False
    assert artifact["cross_corpus_matrix_built"] is True
    assert artifact["clean_row_count"] == 5
    assert artifact["headline_eligible_rows"] == ["FoVer", "HaluEval/FEVER"]
    assert artifact["pilot_only_rows"] == ["MBPP", "HumanEval"]
    assert artifact["taxonomy_only_rows"] == ["TruthfulQA"]
    assert artifact["blocked_rows"]["MBPP"]["generated_code_status"] == (
        "blocked_unresolved_adversarial_flags"
    )
    assert artifact["blocked_rows"]["HumanEval"]["generated_code_status"] == (
        "blocked_unresolved_adversarial_flags"
    )
    assert artifact["missing_rows"] == {}
    assert artifact["source_artifacts"] == [str(path) for path in exp2894.SOURCE_ARTIFACTS.values()]

    rows = _rows_by_corpus(artifact)
    assert set(rows) == {"FoVer", "HaluEval/FEVER", "MBPP", "HumanEval", "TruthfulQA"}

    fover = rows["FoVer"]
    assert fover["row_status"] == "headline_eligible"
    assert fover["truthfulqa_taxonomy"]["value"] is None
    assert fover["generated_code_status"]["reason"] == "not_a_code_corpus"
    assert fover["cctu_constraint_category_coverage"]["n_cases"] == 5
    assert fover["kan_complexity"]["pwa_regions"] == 4

    halueval = rows["HaluEval/FEVER"]
    assert halueval["vericot_exact_support"]["supported_rows"] == 25
    assert halueval["vericot_exact_support"]["candidate_rows"] == 1000
    assert halueval["vericot_exact_support"]["value"] == pytest.approx(0.025)
    assert halueval["residual_gap"]["value"] == "vericot_support_partial"

    mbpp = rows["MBPP"]
    assert mbpp["row_status"] == "pilot_only"
    assert mbpp["generated_code_status"]["status"] == "blocked_unresolved_adversarial_flags"
    assert mbpp["generated_code_status"]["value"] is None
    assert mbpp["structural_dependency_verification"]["reference_passed"] == 1
    assert mbpp["structural_dependency_verification"]["generated_candidate_passed"] == 0
    assert mbpp["structural_dependency_verification"]["violation_types"] == {"parse_error": 1}
    assert mbpp["primary_metric"]["value"] is None

    truthfulqa = rows["TruthfulQA"]
    assert truthfulqa["row_status"] == "taxonomy_only"
    assert truthfulqa["truthfulqa_taxonomy"]["n_rows_materialized"] == 100
    assert truthfulqa["truthfulqa_taxonomy"]["generated_answer_metrics_available"] is False
    assert truthfulqa["primary_metric"]["reason"] == "taxonomy_only_no_generated_answer_metrics"
    assert truthfulqa["vericot_exact_support"]["supported_rows"] == 0
    assert truthfulqa["vericot_exact_support"]["candidate_rows"] == 100

    table = artifact["markdown_table"]
    assert (
        "| Corpus | Status | Headline | Pilot | Taxonomy | Generated code | VeriCoT | Residual gap |"
        in table
    )
    assert (
        "| TruthfulQA | taxonomy_only | no | no | yes | n/a | 0/100 | taxonomy only; no generated-answer metrics |"
        in table
    )
    assert (
        "| MBPP | pilot_only | no | yes | no | blocked | n/a | pilot only; generated-code flags unresolved |"
        in table
    )


def test_req_report_2894_missing_or_unclean_support_stays_null(tmp_path: Path) -> None:
    """REQ-REPORT-2894: missing support artifacts do not create synthetic metrics."""

    _write_clean_sources(tmp_path, generated_flagged=False)
    _write_json(
        tmp_path,
        str(exp2894.STRUCTURAL_VERIFIER_REL_PATH),
        {**_structural_payload(), "structural_dependency_verifier_ready": False},
    )
    _write_json(
        tmp_path,
        str(exp2894.TRUTHFULQA_TAXONOMY_REL_PATH),
        {**_truthfulqa_payload(), "honest_verdict": "blocked_truthfulqa_manifest"},
    )
    _write_json(
        tmp_path,
        str(exp2894.CCTU_VALIDATOR_REL_PATH),
        {**_cctu_payload(), "live_llm_called": True},
    )
    _write_json(
        tmp_path,
        str(exp2894.KAN_COMPLEXITY_REL_PATH),
        {**_kan_payload(), "hardware_execution_claim_made": True},
    )
    (tmp_path / exp2894.VERICOT_FRONTIER_REL_PATH).unlink()

    artifact = exp2894.build_artifact(tmp_path)
    rows = _rows_by_corpus(artifact)

    assert artifact["cross_corpus_matrix_built"] is True
    assert artifact["taxonomy_only_rows"] == []
    assert artifact["missing_rows"]["TruthfulQA"]["primary_metric"]["reason"] == (
        "truthfulqa_taxonomy_source_not_clean"
    )
    assert rows["MBPP"]["generated_code_status"]["status"] == "pilot_only_clean_no_passes"
    assert rows["MBPP"]["generated_code_status"]["n_generated_outputs"] == 1
    assert rows["MBPP"]["structural_dependency_verification"]["reason"] == (
        "structural_dependency_source_not_clean"
    )
    assert rows["FoVer"]["cctu_constraint_category_coverage"]["reason"] == (
        "cctu_validator_source_not_clean"
    )
    assert rows["FoVer"]["kan_complexity"]["reason"] == "kan_complexity_source_not_clean"
    assert rows["HaluEval/FEVER"]["vericot_exact_support"]["reason"] == (
        "vericot_frontier_source_not_clean"
    )
    assert artifact["source_status_by_artifact"]["truthfulqa_taxonomy"] == "blocked"
    assert artifact["source_status_by_artifact"]["cctu_validator"] == "unclean"
    assert artifact["source_status_by_artifact"]["kan_complexity"] == "unclean"
    assert artifact["source_status_by_artifact"]["vericot_frontier"] == "missing"

    generated_unclean_root = tmp_path / "generated-unclean"
    _write_clean_sources(generated_unclean_root, generated_flagged=False)
    _write_json(
        generated_unclean_root,
        str(exp2894.GENERATED_CODE_REL_PATH),
        {**_generated_code_payload(flagged=False), "sandbox_status": "missing: runsc"},
    )
    generated_unclean = exp2894.build_artifact(generated_unclean_root)
    generated_unclean_rows = _rows_by_corpus(generated_unclean)
    assert generated_unclean_rows["MBPP"]["generated_code_status"]["reason"] == (
        "generated_code_source_not_clean"
    )


def test_req_report_2894_helper_edges_and_persistence(tmp_path: Path) -> None:
    """REQ-REPORT-2894: helper branches and artifact persistence are explicit."""

    assert exp2894.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2894.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    assert exp2894.read_json(array) == {}

    assert exp2894.classify_source_status("matrix_v6", {}) == "missing"
    assert exp2894.classify_source_status("matrix_v6", {"honest_verdict": "blocked_gate"}) == (
        "blocked"
    )
    assert exp2894.classify_source_status("matrix_v6", {"honest_verdict": "running"}) == ("unclean")
    assert (
        exp2894.classify_source_status(
            "matrix_v6",
            {"honest_verdict": "complete: ok", "cross_corpus_matrix_built": True},
        )
        == "clean"
    )
    assert exp2894.classify_source_status("truthfulqa_taxonomy", _truthfulqa_payload()) == "clean"
    assert (
        exp2894.classify_source_status(
            "generated_code",
            _generated_code_payload(flagged=True),
        )
        == "flagged"
    )
    assert (
        exp2894.classify_source_status(
            "generated_code",
            _generated_code_payload(flagged=False),
        )
        == "clean"
    )
    assert exp2894.classify_source_status("structural_verifier", _structural_payload()) == "clean"
    assert exp2894.classify_source_status("cctu_validator", _cctu_payload()) == "clean"
    assert exp2894.classify_source_status("vericot_frontier", _vericot_payload()) == "clean"
    assert exp2894.classify_source_status("kan_complexity", _kan_payload()) == "clean"
    assert exp2894.classify_source_status("unknown", {"honest_verdict": "complete: ok"}) == (
        "unclean"
    )
    assert exp2894.has_unresolved_flags({"adversarial_verify_passed": False}) is True
    assert exp2894.has_unresolved_flags({"adversarial_verify_flags": [{"kind": "x"}]}) is True
    assert exp2894.has_unresolved_flags({"adversarial_verify_summary": {"flag_count": 1}}) is True
    assert exp2894.has_unresolved_flags({"corrigendum_pending": [{"kind": "x"}]}) is True

    missing_matrix_root = tmp_path / "missing-matrix"
    missing_matrix = exp2894.build_artifact(missing_matrix_root)
    assert missing_matrix["cross_corpus_matrix_built"] is False
    assert missing_matrix["missing_rows"]["FoVer"]["primary_metric"]["reason"] == (
        "matrix_v6_source_not_clean"
    )

    partial_matrix_root = tmp_path / "partial-matrix"
    partial_v6 = {**_v6_payload(), "matrix_rows": [_v6_payload()["matrix_rows"][0]]}
    _write_json(partial_matrix_root, str(exp2894.MATRIX_V6_REL_PATH), partial_v6)
    _write_json(
        partial_matrix_root,
        str(exp2894.TRUTHFULQA_TAXONOMY_REL_PATH),
        _truthfulqa_payload(),
    )
    partial_matrix = exp2894.build_artifact(partial_matrix_root)
    assert partial_matrix["cross_corpus_matrix_built"] is False
    assert partial_matrix["missing_rows"]["HaluEval/FEVER"]["primary_metric"]["reason"] == (
        "row_not_present_in_clean_sources"
    )

    alternate_flags_root = tmp_path / "alternate-flags"
    _write_clean_sources(alternate_flags_root, generated_flagged=False)
    _write_json(
        alternate_flags_root,
        str(exp2894.GENERATED_CODE_REL_PATH),
        {
            **_generated_code_payload(flagged=False),
            "adversarial_verify_passed": False,
            "adversarial_verify_flags": [{"kind": "UNIT_TEST"}],
        },
    )
    alternate_flags = exp2894.build_artifact(alternate_flags_root)
    assert "adversarial_verify_passed=false" in alternate_flags["blocked_rows"]["MBPP"]["reasons"]
    assert "adversarial_verify_flags_present" in alternate_flags["blocked_rows"]["MBPP"]["reasons"]

    _write_clean_sources(tmp_path)
    out = exp2894.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    saved = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "results/experiment_2894_cross_corpus_matrix_v7.json"
    assert saved["duration_s"] == pytest.approx(0.5)
    assert saved["cross_corpus_matrix_built"] is True

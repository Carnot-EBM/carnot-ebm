"""Tests for Exp 2880 clean .272 cross-corpus matrix v6 generation.

Spec refs: REQ-REPORT-2880, SCENARIO-REPORT-2880.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v6_2880 as exp2880


def _write_json(root: Path, rel_path: str, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _v5_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2865_cross_corpus_matrix_v5",
        "honest_verdict": "complete: cross-corpus matrix built from 2 clean corpus rows",
        "cross_corpus_matrix_built": True,
        "row_status_by_corpus": {
            "FoVer": "clean",
            "HaluEval/FEVER": "clean",
            "MBPP": "missing",
            "HumanEval": "missing",
            "TruthfulQA": "missing",
        },
        "excluded_from_headline": {
            "MBPP": "source_artifact_missing",
            "HumanEval": "source_artifact_missing",
            "TruthfulQA": "source_artifact_missing",
        },
        "verifier_corpus_dual_matrix": {
            "FoVer": {
                "corpus": "FoVer",
                "honest_verdict": "complete: FoVer dual-condition rerun",
                "production_auroc": 0.9131336,
                "architecture_only_auroc": 0.8946624,
                "learning_contribution": 0.0184712,
                "n_examples": 1000,
                "n_seeds": 5,
                "row_status": "clean",
                "source_artifact": "results/experiment_2850_fover_dual_condition_integrity_v4.json",
            },
            "HaluEval/FEVER": {
                "corpus": "HaluEval/FEVER",
                "honest_verdict": "complete: HaluEval/FEVER local calibration ready",
                "measured_auroc_by_dataset": {
                    "halueval": 0.553072,
                    "fever": 0.33114331723027374,
                },
                "n_examples": 1000,
                "n_examples_by_dataset": {"halueval": 500, "fever": 500},
                "label_counts_by_dataset": {
                    "halueval": {"0": 250, "1": 250},
                    "fever": {"0": 270, "1": 230},
                },
                "row_status": "clean",
                "source_artifact": "results/experiment_2864_halueval_fever_full_calibration_v3.json",
            },
        },
        "source_artifacts": [
            "results/experiment_2850_fover_dual_condition_integrity_v4.json",
            "results/experiment_2864_halueval_fever_full_calibration_v3.json",
        ],
        "run_date": "20260522",
    }


def _exact_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: exact frontier touches bounded HaluEval/FEVER rows",
        "frontier_expansion_ready": True,
        "n_candidate_rows": 1000,
        "n_exact_supported_rows": 8,
        "n_unsupported_rows": 992,
        "unsupported_reasons": {"unsupported_no_manual_exact_constraint": 992},
        "certificates": [
            {"dataset": "HaluEval", "label": 0, "stable_id": "halueval-ok"},
            {"dataset": "FEVER", "label": 1, "stable_id": "fever-bad"},
        ],
        "source_artifacts": [
            "results/experiment_2866_beaver_exact_tiny_frontier_v1.json",
            "results/experiment_2864_halueval_fever_full_calibration_v3.json",
        ],
        "run_date": "20260522",
    }


def _error_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: HaluEval/FEVER local error-verifiability audit ready",
        "error_verifiability_ready": True,
        "n_rows_audited": 1000,
        "actionable_localization_rate": 0.929167,
        "label_consistency_rate": 0.447,
        "bucket_level_metrics": {
            "data-grounding": {"auroc": 0.446017, "n_rows": 376},
            "reasoning-chain": {"auroc": 0.529561, "n_rows": 69},
            "extraction/format": {"auroc": 0.555993, "n_rows": 431},
            "unsupported": {"auroc": None, "n_rows": 124},
            "unknown": {"auroc": None, "n_rows": 0},
        },
        "weak_auroc_explanation": "Weak scalar AUROC is best explained by coverage gaps.",
        "remote_llm_called": False,
        "source_artifacts": [
            "results/experiment_2864_halueval_fever_full_calibration_v3.json",
            "results/experiment_2865_cross_corpus_matrix_v5.json",
            "results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json",
        ],
        "run_date": "20260522",
    }


def _code_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: MBPP/HumanEval manifest-only execution pilot ready",
        "code_manifest_pilot_ready": True,
        "deterministic_execution_used": True,
        "headline_metric_claim_made": False,
        "sandbox_status": "available: runsc",
        "n_mbpp_rows": 1,
        "n_humaneval_rows": 1,
        "pilot_rows": [
            {
                "corpus": "MBPP",
                "stable_id": "mbpp-11",
                "passed": True,
                "n_tests": 3,
                "verifier_feature_coverage": {"no_llm_generation": True},
            },
            {
                "corpus": "HumanEval",
                "stable_id": "HumanEval/0",
                "passed": True,
                "n_tests": 7,
                "verifier_feature_coverage": {"no_llm_generation": True},
            },
        ],
        "source_artifacts": [
            "results/experiment_2863_eval_manifest_contract_v2.json",
            "results/experiment_2865_cross_corpus_matrix_v5.json",
        ],
        "run_date": "20260522",
    }


def _write_clean_sources(root: Path) -> None:
    _write_json(root, str(exp2880.MATRIX_V5_REL_PATH), _v5_payload())
    _write_json(root, str(exp2880.EXACT_FRONTIER_REL_PATH), _exact_payload())
    _write_json(root, str(exp2880.ERROR_VERIFIABILITY_REL_PATH), _error_payload())
    _write_json(root, str(exp2880.CODE_PILOT_REL_PATH), _code_payload())


def _rows_by_corpus(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["corpus"]: row for row in artifact["matrix_rows"]}


def test_scenario_report_2880_builds_clean_v6_with_pilot_boundaries(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2880: pilot rows are present but not headline-eligible."""

    _write_clean_sources(tmp_path)

    artifact = exp2880.build_artifact(tmp_path, started_s=5.0, now_s=7.5)

    required = {
        "honest_verdict",
        "cross_corpus_matrix_built",
        "source_artifacts",
        "clean_row_count",
        "headline_eligible_rows",
        "pilot_only_rows",
        "missing_rows",
        "matrix_rows",
        "markdown_table",
        "synthetic_rows_created",
        "field_principles",
        "run_date",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["cross_corpus_matrix_built"] is True
    assert artifact["run_date"] == "20260522"
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["synthetic_rows_created"] is False
    assert artifact["source_artifacts"] == [
        "results/experiment_2865_cross_corpus_matrix_v5.json",
        "results/experiment_2877_exact_frontier_expansion_halueval_fever_v2.json",
        "results/experiment_2878_halueval_fever_error_verifiability_v1.json",
        "results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json",
    ]
    assert artifact["clean_row_count"] == 4
    assert artifact["headline_eligible_rows"] == ["FoVer", "HaluEval/FEVER"]
    assert artifact["pilot_only_rows"] == ["MBPP", "HumanEval"]
    assert set(artifact["missing_rows"]) == {"TruthfulQA"}
    assert artifact["missing_rows"]["TruthfulQA"]["primary_metric"]["value"] is None
    assert artifact["missing_rows"]["TruthfulQA"]["primary_metric"]["reason"] == (
        "source_artifact_missing_in_v5_and_no_dot272_replacement"
    )

    rows = _rows_by_corpus(artifact)
    assert set(rows) == {"FoVer", "HaluEval/FEVER", "MBPP", "HumanEval"}
    assert rows["FoVer"]["row_status"] == "headline_eligible"
    assert rows["FoVer"]["primary_metric"]["production_auroc"] == pytest.approx(0.9131336)
    assert rows["FoVer"]["exact_frontier_support"]["value"] is None
    assert rows["FoVer"]["exact_frontier_support"]["reason"] == "not_applicable_to_fover"

    halueval = rows["HaluEval/FEVER"]
    assert halueval["row_status"] == "headline_eligible"
    assert halueval["exact_frontier_support"]["value"] == pytest.approx(0.008)
    assert halueval["exact_frontier_support"]["supported_rows"] == 8
    assert halueval["error_verifiability"]["value"] is True
    assert halueval["label_consistency"]["value"] == pytest.approx(0.447)
    assert halueval["code_execution_pilot"]["value"] is None
    assert halueval["residual_gap"]["unsupported_exact_rows"] == 992

    mbpp = rows["MBPP"]
    humaneval = rows["HumanEval"]
    assert mbpp["row_status"] == "pilot_only"
    assert humaneval["row_status"] == "pilot_only"
    assert mbpp["code_execution_pilot"]["value"] == "pilot_passed"
    assert humaneval["code_execution_pilot"]["value"] == "pilot_passed"
    assert mbpp["headline_metric_claim_made"] is False
    assert humaneval["headline_metric_claim_made"] is False
    assert mbpp["primary_metric"]["value"] is None
    assert mbpp["primary_metric"]["reason"] == "pilot_only_no_generated_code_metric"

    table = artifact["markdown_table"]
    assert "| Corpus | Status | Headline | Pilot | Exact frontier | Label consistency | Residual gap |" in table
    assert "| MBPP | pilot_only | no | yes | n/a | n/a | pilot only; no pass@k/AUROC |" in table
    assert "| TruthfulQA | missing | no | no | n/a | n/a | missing source artifact |" in table


def test_req_report_2880_rejects_blocked_or_metric_claiming_sources(tmp_path: Path) -> None:
    """REQ-REPORT-2880: blocked sources and pilot headline claims do not become rows."""

    _write_clean_sources(tmp_path)
    blocked_exact = _exact_payload() | {
        "honest_verdict": "blocked_solver_frontier_unavailable",
        "frontier_expansion_ready": False,
    }
    metric_claiming_code = _code_payload() | {"headline_metric_claim_made": True}
    _write_json(tmp_path, str(exp2880.EXACT_FRONTIER_REL_PATH), blocked_exact)
    _write_json(tmp_path, str(exp2880.CODE_PILOT_REL_PATH), metric_claiming_code)

    artifact = exp2880.build_artifact(tmp_path)

    assert artifact["cross_corpus_matrix_built"] is False
    assert artifact["headline_eligible_rows"] == ["FoVer"]
    assert artifact["pilot_only_rows"] == []
    assert artifact["clean_row_count"] == 1
    assert artifact["missing_rows"]["HaluEval/FEVER"]["primary_metric"]["reason"] == (
        "blocked_or_unclean_dot272_halueval_fever_source"
    )
    assert artifact["missing_rows"]["MBPP"]["primary_metric"]["reason"] == (
        "code_pilot_not_clean_or_claimed_headline_metric"
    )
    assert artifact["missing_rows"]["HumanEval"]["primary_metric"]["reason"] == (
        "code_pilot_not_clean_or_claimed_headline_metric"
    )
    assert all(row["corpus"] != "HaluEval/FEVER" for row in artifact["matrix_rows"])
    assert all(row["corpus"] != "MBPP" for row in artifact["matrix_rows"])
    assert "blocked_or_unclean_sources_present" in artifact["honest_verdict"]


def test_req_report_2880_helper_edges_and_persistence(tmp_path: Path) -> None:
    """REQ-REPORT-2880: malformed inputs, helper branches, and JSON writes are explicit."""

    assert exp2880.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2880.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1, 2]", encoding="utf-8")
    assert exp2880.read_json(array) == {}

    assert exp2880.classify_source_status("missing", {}) == "missing"
    assert exp2880.classify_source_status("matrix_v5", {"honest_verdict": "blocked_gate"}) == (
        "blocked"
    )
    assert exp2880.classify_source_status(
        "matrix_v5",
        {"honest_verdict": "complete: no", "cross_corpus_matrix_built": False},
    ) == "unclean"
    assert exp2880.classify_source_status(
        "exact_frontier",
        {"honest_verdict": "complete: ok", "frontier_expansion_ready": True},
    ) == "clean"
    assert exp2880.classify_source_status(
        "error_verifiability",
        {
            "honest_verdict": "complete: ok",
            "error_verifiability_ready": True,
            "remote_llm_called": True,
        },
    ) == "unclean"
    assert exp2880.classify_source_status(
        "code_execution_pilot",
        {
            "honest_verdict": "complete: ok",
            "code_manifest_pilot_ready": True,
            "headline_metric_claim_made": False,
        },
    ) == "clean"
    assert exp2880.classify_source_status("unknown_source", {"honest_verdict": "complete: ok"}) == (
        "unclean"
    )

    _write_clean_sources(tmp_path)
    out = exp2880.write_artifact(tmp_path, started_s=1.0, now_s=1.25)
    saved = json.loads(out.read_text(encoding="utf-8"))

    assert out == tmp_path / "results/experiment_2880_cross_corpus_matrix_v6.json"
    assert saved["duration_s"] == pytest.approx(0.25)
    assert saved["cross_corpus_matrix_built"] is True

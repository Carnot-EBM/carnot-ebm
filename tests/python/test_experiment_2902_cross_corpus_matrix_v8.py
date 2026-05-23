"""Tests for Exp 2902 cross-corpus matrix v8 aggregation.

Spec refs: REQ-REPORT-2902, SCENARIO-REPORT-2902.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v8_2902 as exp2902


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(root: Path, rel_path: str | Path) -> str:
    return hashlib.sha256((root / rel_path).read_bytes()).hexdigest()


def _null(reason: str) -> dict[str, Any]:
    return {"value": None, "reason": reason}


def _v7_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2894_cross_corpus_matrix_v7",
        "honest_verdict": (
            "complete: cross-corpus matrix v7 built from 5 clean headline/pilot/taxonomy rows"
        ),
        "cross_corpus_matrix_built": True,
        "headline_eligible_rows": ["FoVer", "HaluEval/FEVER"],
        "pilot_only_rows": ["MBPP", "HumanEval"],
        "taxonomy_only_rows": ["TruthfulQA"],
        "blocked_rows": {
            "MBPP": {
                "generated_code_status": "blocked_unresolved_adversarial_flags",
                "reasons": ["flagged_adversarial=true"],
                "source_artifact": "results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json",
            },
            "HumanEval": {
                "generated_code_status": "blocked_unresolved_adversarial_flags",
                "reasons": ["flagged_adversarial=true"],
                "source_artifact": "results/experiment_2889_mbpp_humaneval_generated_code_clean_row_v1.json",
            },
        },
        "missing_rows": {},
        "source_status_by_artifact": {"matrix_v6": "clean", "generated_code": "flagged"},
        "matrix_rows": [
            {
                "corpus": "FoVer",
                "row_status": "headline_eligible",
                "headline_eligible": True,
                "pilot_only": False,
                "taxonomy_only": False,
                "source_artifact": "results/experiment_2850_fover_dual_condition_integrity_v4.json",
                "primary_metric": {
                    "production_auroc": 0.9131336,
                    "architecture_only_auroc": 0.8946624,
                },
                "generated_code_status": _null("not_a_code_corpus"),
            },
            {
                "corpus": "HaluEval/FEVER",
                "row_status": "headline_eligible",
                "headline_eligible": True,
                "pilot_only": False,
                "taxonomy_only": False,
                "source_artifact": "results/experiment_2864_halueval_fever_full_calibration_v3.json",
                "primary_metric": {"measured_auroc_by_dataset": {"halueval": 0.553072}},
                "vericot_exact_support": {"supported_rows": 25, "candidate_rows": 1000},
            },
            {
                "corpus": "MBPP",
                "row_status": "pilot_only",
                "headline_eligible": False,
                "pilot_only": True,
                "taxonomy_only": False,
                "source_artifact": "results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json",
                "primary_metric": _null("pilot_only_no_generated_code_metric"),
                "generated_code_status": {
                    "status": "blocked_unresolved_adversarial_flags",
                    "flag_reasons": ["flagged_adversarial=true"],
                },
                "structural_dependency_verification": {
                    "source_artifact": str(exp2902.EXP2890_REL_PATH),
                    "n_rows_verified": 10,
                },
            },
            {
                "corpus": "HumanEval",
                "row_status": "pilot_only",
                "headline_eligible": False,
                "pilot_only": True,
                "taxonomy_only": False,
                "source_artifact": "results/experiment_2879_code_corpus_manifest_execution_pilot_v1.json",
                "primary_metric": _null("pilot_only_no_generated_code_metric"),
                "generated_code_status": {
                    "status": "blocked_unresolved_adversarial_flags",
                    "flag_reasons": ["flagged_adversarial=true"],
                },
            },
            {
                "corpus": "TruthfulQA",
                "row_status": "taxonomy_only",
                "headline_eligible": False,
                "pilot_only": False,
                "taxonomy_only": True,
                "source_artifact": "results/experiment_2888_truthfulqa_inficheck_taxonomy_manifest_v1.json",
                "primary_metric": _null("taxonomy_only_no_generated_answer_metrics"),
                "truthfulqa_taxonomy": {"n_rows_materialized": 100},
            },
        ],
    }


def _exp2890_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2890_code_structural_dependency_verifier_v1",
        "honest_verdict": "complete: MBPP/HumanEval structural dependency verifier metadata ready",
        "structural_dependency_verifier_ready": True,
        "headline_metric_claim_made": False,
        "n_contracts_built": 10,
        "n_rows_verified": 20,
        "generated_outputs_consumed": True,
        "violation_types": {"missing_function_definition": 4, "parse_error": 5},
    }


def _exp2891_payload() -> dict[str, Any]:
    return {
        "artifact": "experiment_2891_cctu_executable_constraint_validator_pilot_v1",
        "honest_verdict": "complete: local CCTU-style executable constraint validator pilot ready",
        "cctu_validator_ready": True,
        "headline_metric_claim_made": False,
        "executable_validation_used": True,
        "live_llm_called": False,
        "n_cases": 5,
        "constraint_categories": ["behavior", "resource"],
        "category_coverage": {"behavior": {"passed": 0, "total": 1}},
        "unsupported_categories": {"multi_turn_state": {"supported": False}},
    }


def _exp2892_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: deterministic VeriCoT frontier rows available",
        "vericot_frontier_ready": True,
        "n_candidate_rows": 1100,
        "n_vericot_supported_rows": 25,
        "n_unsupported_rows": 1075,
        "unsupported_reasons": {"unsupported_no_deterministic_vericot_template": 974},
        "solver_backend": "z3-solver 4.16.0 + deterministic premise anchors",
        "autoformalization_llm_called": False,
    }


def _exp2898_payload() -> dict[str, Any]:
    return {
        "experiment_id": 2898,
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "kv260_overlay_loaded": "carnot_ising_v2_n64",
        "kv260_uio_devices_present": ["/dev/uio0"],
        "bitstream_sha256": "a" * 64,
        "board_transcript_path": "results/experiment_2898_kv260_transcript.log",
        "preconditions_checked": [{"resource": "kv260_ssh", "available": True}],
        "per_seed_results": [
            {
                "seed": 42,
                "n_samples": 10000,
                "per_sample_wall_clock_us_median": 24.05,
                "per_sample_wall_clock_us_p95": 24.38,
            }
        ],
        "sample_count_sweep_results": [{"seed": 42, "n_samples": 100}],
        "duration_s": 80.17,
    }


def _write_clean_sources(root: Path) -> None:
    _write_json(root, exp2902.V7_REL_PATH, _v7_payload())
    _write_json(root, exp2902.EXP2890_REL_PATH, _exp2890_payload())
    _write_json(root, exp2902.EXP2891_REL_PATH, _exp2891_payload())
    _write_json(root, exp2902.EXP2892_REL_PATH, _exp2892_payload())
    _write_json(root, exp2902.EXP2898_REL_PATH, _exp2898_payload())


def _rows_by_id(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["row_id"]: row for row in artifact["matrix_rows"]}


def test_req_report_2902_spec_is_declared() -> None:
    """REQ-REPORT-2902: OpenSpec declares the v8 aggregation contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")
    assert "REQ-REPORT-2902" in spec
    assert "SCENARIO-REPORT-2902" in spec
    assert "aggregation_from_upstream_artifacts" in spec


def test_scenario_report_2902_builds_v8_with_forward_provenance(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2902: clean support artifacts become cited v8 rows."""

    _write_clean_sources(tmp_path)

    artifact = exp2902.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    required = {
        "honest_verdict",
        "inference_substrate",
        "rows_clean",
        "rows_flagged",
        "rows_blocked",
        "rows_pilot_only",
        "cited_upstream_artifacts",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["rows_clean"] == [
        "corpus:FoVer",
        "corpus:HaluEval_FEVER",
        "corpus:TruthfulQA",
        "exp2890_code_structural_dependency",
        "exp2892_vericot",
        "exp2898_kv260_hardware",
    ]
    assert artifact["rows_pilot_only"] == [
        "corpus:MBPP",
        "corpus:HumanEval",
        "exp2891_cctu",
    ]
    assert artifact["rows_flagged"] == ["corpus:MBPP", "corpus:HumanEval"]
    assert artifact["rows_blocked"] == []

    citations = artifact["cited_upstream_artifacts"]
    assert citations["principle"] == exp2902.PROVENANCE_PRINCIPLE
    assert citations["shape"] == "list of {experiment_id, fields_imported, sha256}"
    assert {item["experiment_id"]: item["sha256"] for item in citations["artifacts"]} == {
        "exp2894": _sha256(tmp_path, exp2902.V7_REL_PATH),
        "exp2890": _sha256(tmp_path, exp2902.EXP2890_REL_PATH),
        "exp2891": _sha256(tmp_path, exp2902.EXP2891_REL_PATH),
        "exp2892": _sha256(tmp_path, exp2902.EXP2892_REL_PATH),
        "exp2898": _sha256(tmp_path, exp2902.EXP2898_REL_PATH),
    }

    rows = _rows_by_id(artifact)
    assert rows["corpus:FoVer"]["row_status"] == "clean"
    assert rows["corpus:MBPP"]["row_status"] == "pilot_only_flagged_support"
    assert rows["corpus:MBPP"]["flag_reasons"] == ["flagged_adversarial=true"]
    assert rows["exp2890_code_structural_dependency"]["summary"]["n_rows_verified"] == 20
    assert rows["exp2891_cctu"]["row_status"] == "pilot_only"
    assert rows["exp2892_vericot"]["summary"]["n_vericot_supported_rows"] == 25
    assert rows["exp2898_kv260_hardware"]["summary"]["inference_substrate"] == "hardware_smoke"
    for row in artifact["matrix_rows"]:
        provenance = row["provenance"]
        assert provenance["sha256"] == _sha256(tmp_path, provenance["artifact_path"])
        assert provenance["fields_imported"]


def test_req_report_2902_blocks_without_v7_and_persists(tmp_path: Path) -> None:
    """REQ-REPORT-2902: missing or unclean v7 prevents downstream-only synthesis."""

    assert exp2902.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2902.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1]", encoding="utf-8")
    assert exp2902.read_json(array) == {}

    missing = exp2902.build_artifact(tmp_path, started_s=1.0, now_s=1.5)
    assert missing["honest_verdict"] == "blocked_v7_missing"
    assert missing["rows_blocked"] == ["exp2894"]
    assert missing["matrix_rows"] == []
    assert missing["cited_upstream_artifacts"]["artifacts"] == []

    _write_json(
        tmp_path,
        exp2902.V7_REL_PATH,
        {"honest_verdict": "running", "cross_corpus_matrix_built": False},
    )
    unclean = exp2902.build_artifact(tmp_path, started_s=2.0, now_s=2.25)
    assert unclean["honest_verdict"] == "blocked_v7_unclean"
    assert unclean["rows_blocked"] == ["exp2894"]
    assert unclean["matrix_rows"][0]["row_status"] == "blocked"
    assert exp2902._source_status("unknown", {"honest_verdict": "complete: ok"}) == "unclean"
    assert (
        exp2902._source_status(
            "exp2890",
            {**_exp2890_payload(), "flagged_adversarial": False, "corrigendum_pending": [{}]},
        )
        == "flagged"
    )
    assert (
        exp2902._source_status(
            "exp2890",
            {**_exp2890_payload(), "adversarial_verify_passed": False},
        )
        == "flagged"
    )
    assert exp2902._source_flag_reasons(
        {
            "adversarial_verify_passed": False,
            "corrigendum_pending": [{}],
            "adversarial_verify_flags": [{}],
            "adversarial_verify_summary": {"flag_count": 1},
        }
    ) == [
        "adversarial_verify_passed=false",
        "corrigendum_pending_present",
        "adversarial_verify_flags_present",
        "adversarial_verify_summary_flag_count",
    ]
    assert (
        exp2902._v7_matrix_row(
            {
                "corpus": "Pilot",
                "row_status": "pilot_only",
                "pilot_only": True,
                "generated_code_status": {"reason": "clean_pilot"},
            },
            {"blocked_rows": {}},
            "f" * 64,
        )["row_status"]
        == "pilot_only"
    )
    assert (
        exp2902._v7_matrix_row(
            {
                "corpus": "Flagged",
                "row_status": "headline_eligible",
                "pilot_only": False,
                "generated_code_status": {
                    "status": "blocked_unresolved_adversarial_flags",
                    "flag_reasons": ["flag"],
                },
            },
            {"blocked_rows": {}},
            "f" * 64,
        )["row_status"]
        == "flagged"
    )

    _write_clean_sources(tmp_path)
    out = exp2902.write_artifact(tmp_path, started_s=3.0, now_s=3.5)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert out == tmp_path / exp2902.OUTPUT_REL_PATH
    assert saved["duration_s"] == pytest.approx(0.5)
    assert saved["rows_clean"][-1] == "exp2898_kv260_hardware"


def test_req_report_2902_unclean_support_rows_stay_out_of_clean_bucket(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-2902: unclean support artifacts are categorized, not promoted."""

    _write_json(tmp_path, exp2902.V7_REL_PATH, _v7_payload())
    _write_json(
        tmp_path,
        exp2902.EXP2890_REL_PATH,
        {**_exp2890_payload(), "flagged_adversarial": True},
    )
    _write_json(
        tmp_path,
        exp2902.EXP2892_REL_PATH,
        {"honest_verdict": "blocked_vericot_inputs_missing", "vericot_frontier_ready": False},
    )
    _write_json(
        tmp_path,
        exp2902.EXP2898_REL_PATH,
        {**_exp2898_payload(), "per_seed_results": [], "cpu_speedup_claim": "forbidden"},
    )

    artifact = exp2902.build_artifact(tmp_path)
    rows = _rows_by_id(artifact)

    assert "exp2890_code_structural_dependency" in artifact["rows_flagged"]
    assert "exp2891_cctu" in artifact["rows_blocked"]
    assert "exp2892_vericot" in artifact["rows_blocked"]
    assert "exp2898_kv260_hardware" in artifact["rows_blocked"]
    assert "exp2890_code_structural_dependency" not in artifact["rows_clean"]
    assert rows["exp2890_code_structural_dependency"]["row_status"] == "flagged"
    assert rows["exp2891_cctu"]["row_status"] == "blocked"
    assert rows["exp2892_vericot"]["blocked_reason"] == "blocked_vericot_inputs_missing"
    assert rows["exp2898_kv260_hardware"]["blocked_reason"] == "source_not_clean"

"""Regression tests for Exp5933 adversarial substrate classification.

Spec refs: REQ-VERIFY-5933,
SCENARIO-VERIFY-5933-AGGREGATION-QUOTED-MARKERS,
SCENARIO-VERIFY-5933-LIVE-PAIRED-CONTROL,
SCENARIO-VERIFY-5933-MALFORMED-CONSERVATIVE,
SCENARIO-VERIFY-5933-CORPUS-NO-SEVERITY-REGRESSION.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


ROOT = Path(__file__).resolve().parents[2]
EXP5931 = ROOT / "results" / "experiment_5931_v526_capstone_reconciliation.json"


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kinds(report: dict[str, Any]) -> set[str]:
    return {flag["kind"] for flag in report["flags"]}


def _flag_severities(report: dict[str, Any]) -> dict[str, str]:
    return {flag["kind"]: flag["severity"] for flag in report["flags"]}


def _paired_payload(substrate: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": "exp5933_paired_control",
        "honest_verdict": "complete: paired control fixture",
        "duration_s": 5.966263,
        "reproducibility_checksum": "sha256:" + "5" * 64,
        "field_provenance": {
            "inference_substrate": (
                "top-level substrate declaration from exact local artifact "
                "reconciliation; nested receipts are quoted upstream evidence"
            )
        },
        "quoted_upstream_receipts": [
            {
                "source": "results/upstream_live_receipt.json",
                "quoted_model_context": (
                    "Upstream receipt used GGUF via llama.cpp on CUDA; "
                    "quoted string includes torch.cuda but this fixture does "
                    "not itself invoke a model unless the top-level substrate says so."
                ),
            }
        ],
    }
    if substrate is not None:
        payload["inference_substrate"] = substrate
    return payload


def test_req_verify_5933_classifier_recognizes_exact_declared_aggregation() -> None:
    payload = _paired_payload("aggregation_from_exact_declared_artifacts")

    classification = av._classify_inference_substrate(payload)

    assert classification["kind"] == "aggregation"
    assert classification["matched_value"] == "aggregation_from_exact_declared_artifacts"
    assert classification["source"] == "top_level_inference_substrate"


def test_req_verify_5933_principle_wrapped_substrate_is_classified() -> None:
    payload = _paired_payload("aggregation_from_exact_declared_artifacts")
    payload["inference_substrate"] = {
        "value": "aggregation_from_exact_declared_artifacts",
        "principle": "explicit top-level provenance controls substrate classification",
    }

    classification = av._classify_inference_substrate(payload)

    assert classification["kind"] == "aggregation"
    assert classification["matched_value"] == "aggregation_from_exact_declared_artifacts"


def test_req_verify_5933_aggregation_quoting_gguf_does_not_get_live_flags(
    tmp_path: Path,
) -> None:
    payload = _paired_payload("aggregation_from_exact_declared_artifacts")

    report = _report_for_payload(tmp_path, payload)

    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_verify_5933_aggregation_still_enforces_json_read_floor(
    tmp_path: Path,
) -> None:
    payload = _paired_payload("aggregation_from_exact_declared_artifacts")
    payload["duration_s"] = 0.0

    report = _report_for_payload(tmp_path, payload)

    assert _flag_severities(report)["DURATION_TOO_SHORT"] == "critical"
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_verify_5933_live_paired_control_still_gets_live_flags(
    tmp_path: Path,
) -> None:
    payload = _paired_payload("live_llm_inference")

    report = _report_for_payload(tmp_path, payload)

    assert _flag_severities(report)["DURATION_TOO_SHORT"] == "critical"
    assert _flag_severities(report)["METHODOLOGY_MISSING"] == "warn"


def test_req_verify_5933_missing_substrate_with_markers_remains_conservative(
    tmp_path: Path,
) -> None:
    payload = _paired_payload(None)

    report = _report_for_payload(tmp_path, payload)

    assert _flag_severities(report)["DURATION_TOO_SHORT"] == "critical"
    assert _flag_severities(report)["METHODOLOGY_MISSING"] == "warn"


def test_req_verify_5933_malformed_substrate_with_markers_remains_conservative(
    tmp_path: Path,
) -> None:
    payload = _paired_payload("aggregation_from_exact_declared_artifacts_typo")

    report = _report_for_payload(tmp_path, payload)

    assert _flag_severities(report)["DURATION_TOO_SHORT"] == "critical"
    assert _flag_severities(report)["METHODOLOGY_MISSING"] == "warn"


def test_req_verify_5933_hardware_smoke_marker_behavior_is_unchanged(
    tmp_path: Path,
) -> None:
    payload = _paired_payload("hardware_smoke")

    report = _report_for_payload(tmp_path, payload)

    assert _flag_severities(report)["DURATION_TOO_SHORT"] == "critical"
    assert _flag_severities(report)["METHODOLOGY_MISSING"] == "warn"


def test_req_verify_5933_simulated_artifact_without_markers_remains_clean(
    tmp_path: Path,
) -> None:
    payload = {
        "experiment": "exp5933_simulated_control",
        "honest_verdict": "complete: deterministic cpu simulation control",
        "inference_substrate": "simulation",
        "duration_s": 0.05,
    }

    report = _report_for_payload(tmp_path, payload)

    assert _flag_kinds(report) == set()


def test_req_verify_5933_deterministic_qa_regression_is_no_llm(
    tmp_path: Path,
) -> None:
    payload = _paired_payload("deterministic_qa_regression_no_llm")

    classification = av._classify_inference_substrate(payload)
    report = _report_for_payload(tmp_path, payload)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == "deterministic_qa_regression_no_llm"
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_learn_6479_pipeline_integration_substrate_has_floor(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-6479: deterministic pipeline integration is an audited no-LLM substrate."""

    payload = _paired_payload("deterministic_pipeline_integration_no_llm")

    classification = av._classify_inference_substrate(payload)
    floor = av.duration_floor_for_artifact(payload)
    report = _report_for_payload(tmp_path, payload)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == "deterministic_pipeline_integration_no_llm"
    assert floor == {
        "substrate": "deterministic_pipeline_integration_no_llm",
        "min_duration_s": av.DETERMINISTIC_VERIFIER_MIN_DURATION_S,
        "reason": "deterministic_verifier",
    }
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in _flag_kinds(report)
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_cl_6497_factor_pool_support_stress_substrate_has_floor(
    tmp_path: Path,
) -> None:
    """REQ-CL-6497: factor-pool support stress is an audited no-LLM substrate."""

    substrate = "deterministic_factor_pool_stress_with_exact_evaluation_no_llm"
    payload = _paired_payload(substrate)
    payload["random_seed"] = 6497

    classification = av._classify_inference_substrate(payload)
    floor = av.duration_floor_for_artifact(payload)
    report = _report_for_payload(tmp_path, payload)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == substrate
    assert floor == {
        "substrate": substrate,
        "min_duration_s": av.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in _flag_kinds(report)
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_store_6522_chronological_conflict_csl_substrate_has_floor(
    tmp_path: Path,
) -> None:
    """REQ-STORE-6522: chronological exact-conflict CSL is an audited no-LLM substrate."""

    substrate = "chronological_exact_conflict_memory_self_learning_no_llm"
    payload = _paired_payload(substrate)
    payload["random_seed"] = 6522

    classification = av._classify_inference_substrate(payload)
    floor = av.duration_floor_for_artifact(payload)
    report = _report_for_payload(tmp_path, payload)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == substrate
    assert floor == {
        "substrate": substrate,
        "min_duration_s": av.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in _flag_kinds(report)
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_infra_6481_runtime_receipt_validation_substrate_has_floor(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-6481: receipt validation is an audited no-LLM substrate."""

    payload = _paired_payload("deterministic_runtime_receipt_validation_no_llm")
    payload["random_seed"] = 6481

    classification = av._classify_inference_substrate(payload)
    floor = av.duration_floor_for_artifact(payload)
    report = _report_for_payload(tmp_path, payload)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == (
        "deterministic_runtime_receipt_validation_no_llm"
    )
    assert floor == {
        "substrate": "deterministic_runtime_receipt_validation_no_llm",
        "min_duration_s": av.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in _flag_kinds(report)
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_infra_6488_artifact_reducer_substrate_has_floor(
    tmp_path: Path,
) -> None:
    """REQ-INFRA-6488: artifact reducers are audited no-LLM substrates."""

    payload = _paired_payload("artifact_reducer_no_llm")
    payload["random_seed"] = 6488

    classification = av._classify_inference_substrate(payload)
    floor = av.duration_floor_for_artifact(payload)
    report = _report_for_payload(tmp_path, payload)

    assert classification["kind"] == "no_llm"
    assert classification["matched_value"] == "artifact_reducer_no_llm"
    assert floor == {
        "substrate": "artifact_reducer_no_llm",
        "min_duration_s": av.NO_LLM_DECLARED_MIN_DURATION_S,
        "reason": "no_llm_declared",
    }
    assert "SUBSTRATE_HAS_NO_DURATION_FLOOR" not in _flag_kinds(report)
    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_verify_5933_real_exp5931_no_longer_gets_live_substrate_flags() -> None:
    report = av.verify_artifact(EXP5931)

    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)

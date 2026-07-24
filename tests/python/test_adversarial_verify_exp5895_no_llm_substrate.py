"""Regression tests for Exp5895's no-LLM external-state substrate.

Spec refs: REQ-LEARN-5895, SCENARIO-LEARN-5895-HARDWARE-MAPPING.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import scripts.adversarial_verify as av


EXP5895_SUBSTRATE = "deterministic_exact_verifier_and_versioned_external_state_no_llm"


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _report_for_payload(tmp_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return av.verify_artifact(_write_payload(tmp_path, payload))


def _flag_kinds(report: dict[str, Any]) -> set[str]:
    return {flag["kind"] for flag in report["flags"]}


def test_req_learn_5895_substrate_is_deterministic_verifier() -> None:
    """REQ-LEARN-5895: the external-state lifecycle uses no live model path."""

    assert av._is_deterministic_verifier({"inference_substrate": EXP5895_SUBSTRATE}) is True


def test_req_learn_5895_gguf_immutability_receipts_do_not_claim_live_model(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-5895: GGUF immutability receipts must not trigger live-model timing."""

    payload = {
        "experiment": 5895,
        "honest_verdict": "complete_null: shortcut_safe_csl_not_promotion_eligible",
        "inference_substrate": EXP5895_SUBSTRATE,
        "duration_s": 0.25,
        "random_seed": 5895,
        "reproducibility_checksum": "sha256:" + "0" * 64,
        "no_model_weight_mutation": {
            "gguf_weight_mutation_count": 0,
            "model_execution_loaded": False,
            "content_hash_strategy": "not_loaded_or_rehashed_large_gguf_stat_immutability_receipt",
        },
    }

    report = _report_for_payload(tmp_path, payload)

    assert "DURATION_TOO_SHORT" not in _flag_kinds(report)
    assert "METHODOLOGY_MISSING" not in _flag_kinds(report)


def test_req_learn_5895_live_llm_claim_still_uses_model_floor(tmp_path: Path) -> None:
    """REQ-LEARN-5895: the new substrate recognition cannot mask live LLM claims."""

    payload = {
        "experiment": "live_model_control",
        "honest_verdict": "complete_positive: live model control",
        "inference_substrate": "live_llm_inference",
        "duration_s": 0.25,
        "model_specs": {"gguf_path": "/tmp/model.gguf"},
        "random_seed": 1,
        "reproducibility_checksum": "sha256:" + "1" * 64,
    }

    report = _report_for_payload(tmp_path, payload)

    assert "DURATION_TOO_SHORT" in _flag_kinds(report)

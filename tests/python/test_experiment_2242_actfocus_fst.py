"""Tests for Exp 2242 ActFocus + FST evaluation.

Spec: REQ-LEARN-2242, SCENARIO-LEARN-2242.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import actfocus_fst_eval as mod


def _single_cached_model_resolution() -> dict[str, Any]:
    specs = [
        {
            "name": "Gemma4-26B-A4B-it",
            "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "gpu": 0,
            "model_path": "/tmp/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        }
    ]
    return {
        "MODEL_SPECS": specs,
        "models_used": [
            {
                "name": specs[0]["name"],
                "hf_id": specs[0]["hf_id"],
                "model_path": specs[0]["model_path"],
                "available": True,
                "used_for_generation": False,
                "blocker": "unit_test_no_generation",
            }
        ],
        "cache_probe": {
            "grep_qwen_or_gemma_nonempty": True,
            "cached_sota_pair_called": True,
            "cached_sota_pair_returned": False,
            "single_model_fallback_used": True,
        },
    }


def test_scenario_learn_2242_writes_valid_actfocus_fst_artifact(tmp_path: Path) -> None:
    """SCENARIO-LEARN-2242: ActFocus variance drives FST retention metrics."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.run_experiment(
        output_path=output,
        model_resolution_provider=_single_cached_model_resolution,
        llama_probe=lambda: {"llama_cpp_available": True, "llama_cpp_gpu_offload": False},
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["n_corpus"] == 20
    assert artifact["MODEL_SPECS"][0]["hf_id"] == "unsloth/gemma-4-26B-A4B-it-GGUF"
    assert artifact["models_used"][0]["available"] is True
    assert artifact["energy_variance_correlation"] > 0.75
    assert artifact["fast_weight_retention_rate"] >= 0.85
    assert artifact["actfocus_fst_validated"] is True
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_learn_2242_blocked_artifact_when_actfocus_missing(tmp_path: Path) -> None:
    """REQ-LEARN-2242-1: missing ActFocus module writes terminal blocked JSON."""

    output = tmp_path / mod.OUTPUT_FILE

    artifact = mod.run_experiment(
        output_path=output,
        actfocus_path=tmp_path / "missing_actfocus.py",
        model_resolution_provider=_single_cached_model_resolution,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_actfocus_missing"
    assert artifact["actfocus_fst_validated"] is False
    assert artifact["fast_weight_retention_rate"] == 0.0
    assert artifact["preconditions_checked"][0]["status"] == "failed"


def test_req_learn_2242_validation_rejects_gate_mismatch(tmp_path: Path) -> None:
    """REQ-LEARN-2242-5: validation enforces the retention gate boolean."""

    artifact = mod.run_experiment(
        output_path=tmp_path / mod.OUTPUT_FILE,
        model_resolution_provider=_single_cached_model_resolution,
    )
    artifact["actfocus_fst_validated"] = False

    try:
        mod.validate_artifact(artifact)
    except AssertionError as exc:
        assert "actfocus_fst_validated" in str(exc)
    else:  # pragma: no cover - defensive assertion for clearer failures.
        raise AssertionError("expected validation to reject gate mismatch")

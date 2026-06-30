"""Tests for Exp 5043 SOTA GGUF judge preflight.

Spec refs: REQ-VERIFY-5043, SCENARIO-VERIFY-5043.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5043_sota_gguf_judge_preflight as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def test_req_verify_5043_spec_declares_preflight_contract() -> None:
    """REQ-VERIFY-5043: OpenSpec anchors the SOTA GGUF judge preflight."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5043",
        "SCENARIO-VERIFY-5043",
        "experiment_5043_sota_gguf_judge_preflight.py",
        "results/experiment_5043_sota_gguf_judge_preflight.json",
        "blocked_sota_gguf_unavailable",
        "blocked_judge_server",
        "top_logprob_or_confidence_ready",
        "legacy_models_smoke_only",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def _fake_model_resolver(hf_id: str, _preferred_quant: str) -> str | None:
    if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF":
        return "/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"
    return None


def _ready_endpoint_probe(_endpoints: list[str], _timeout_s: float) -> dict[str, Any]:
    return {
        "selected_endpoint": "http://127.0.0.1:8080",
        "completion_ready": True,
        "top_logprob_ready": True,
        "confidence_ready": False,
        "telemetry_signal": "top_logprobs",
        "probes": [
            {
                "endpoint": "http://127.0.0.1:8080",
                "completion_probe": {
                    "ready": True,
                    "status": 200,
                    "detail": "completion returned non-empty content",
                },
                "telemetry_probe": {
                    "ready": True,
                    "status": 200,
                    "detail": "top_logprobs present",
                    "signal": "top_logprobs",
                },
            }
        ],
    }


def _blocked_endpoint_probe(_endpoints: list[str], _timeout_s: float) -> dict[str, Any]:
    return {
        "selected_endpoint": None,
        "completion_ready": False,
        "top_logprob_ready": False,
        "confidence_ready": False,
        "telemetry_signal": None,
        "probes": [
            {
                "endpoint": "http://127.0.0.1:8080",
                "completion_probe": {
                    "ready": False,
                    "status": None,
                    "detail": "URLError: connection refused",
                },
                "telemetry_probe": {
                    "ready": False,
                    "status": None,
                    "detail": "skipped: completion probe failed",
                    "signal": None,
                },
            }
        ],
    }


def test_scenario_verify_5043_ready_artifact_uses_mandated_model_and_toplogprobs(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5043: one SOTA GGUF plus top-logprobs marks judge ready."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        model_resolver=_fake_model_resolver,
        endpoint_probe=_ready_endpoint_probe,
        endpoints=["http://127.0.0.1:8080"],
        now=lambda: 10.0,
        write=True,
    )

    assert artifact["honest_verdict"] == "complete_sota_gguf_judge_preflight_ready"
    assert artifact["sota_models_ready"] is True
    assert artifact["sota_judge_ready"] is True
    assert artifact["top_logprob_or_confidence_ready"] is True
    assert artifact["legacy_models_smoke_only"] is True
    assert artifact["usable_sota_models"] == [
        {
            "role": "flagship_moe",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_path": "/models/Qwen3.6-35B-A3B-Q4_K_M.gguf",
        }
    ]
    assert artifact["model_specs"]["flagship_moe"]["resolved_path"].endswith(".gguf")
    assert artifact["model_specs"]["flagship_dense"]["resolved_path"] == "missing"
    assert artifact["endpoint_summary"]["telemetry_signal"] == "top_logprobs"
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5043_missing_mandated_models_blocks_sota(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5043: no mandated GGUF writes blocked_sota_gguf_unavailable."""

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=lambda _hf_id, _preferred_quant: None,
        endpoint_probe=_ready_endpoint_probe,
        endpoints=["http://127.0.0.1:8080"],
        now=lambda: 20.0,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_sota_gguf_unavailable"
    assert artifact["usable_sota_models"] == []
    assert artifact["sota_models_ready"] is False
    assert artifact["sota_judge_ready"] is False
    assert artifact["top_logprob_or_confidence_ready"] is False
    assert all(spec["resolved_path"] == "missing" for spec in artifact["model_specs"].values())
    assert mod.artifact_schema_errors(artifact) == []


def test_scenario_verify_5043_missing_judge_or_confidence_blocks_with_diagnostics(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5043: cached GGUF without judge telemetry is blocked_judge_server."""

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_fake_model_resolver,
        endpoint_probe=_blocked_endpoint_probe,
        endpoints=["http://127.0.0.1:8080"],
        now=lambda: 30.0,
        write=False,
    )

    assert artifact["honest_verdict"] == "blocked_judge_server"
    assert artifact["sota_models_ready"] is True
    assert artifact["sota_judge_ready"] is False
    assert artifact["top_logprob_or_confidence_ready"] is False
    diagnostics = artifact["endpoint_summary"]["probes"][0]
    assert diagnostics["completion_probe"]["detail"] == "URLError: connection refused"
    assert diagnostics["telemetry_probe"]["detail"] == "skipped: completion probe failed"
    assert mod.artifact_schema_errors(artifact) == []

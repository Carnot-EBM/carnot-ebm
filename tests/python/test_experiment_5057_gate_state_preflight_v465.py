"""Tests for Exp 5057 split gate-state preflight.

Spec refs: REQ-VERIFY-5057, SCENARIO-VERIFY-5057.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot import experiment_5057_gate_state_preflight_v465 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def test_req_verify_5057_spec_declares_split_gate_contract() -> None:
    """REQ-VERIFY-5057: OpenSpec anchors the split gate-state artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    module_text = (REPO / mod.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5057",
        "SCENARIO-VERIFY-5057",
        "experiment_5057_gate_state_preflight_v465.py",
        "results/experiment_5057_gate_state_preflight_v465.json",
        "sota_models_ready",
        "sota_judge_ready",
        "top_logprob_or_confidence_ready",
        "tool_first_verifier_ready",
        "skip_reasons",
        "legacy_models_smoke_only",
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "AutoTokenizer.from_pretrained",
    ):
        assert marker in spec
    assert "AutoTokenizer.from_pretrained" not in module_text
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def _one_model_resolver(hf_id: str, preferred_quant: str) -> str | None:
    assert preferred_quant == "Q4_K_M"
    if hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF":
        return "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    return None


def _no_model_resolver(_hf_id: str, _preferred_quant: str) -> str | None:
    return None


def _blocked_endpoint_probe(_endpoints: list[str], _timeout_s: float) -> dict[str, Any]:
    return {
        "candidate_endpoints": ["http://127.0.0.1:8080"],
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
                },
            }
        ],
    }


def _telemetry_ready_endpoint_probe(_endpoints: list[str], _timeout_s: float) -> dict[str, Any]:
    return {
        "candidate_endpoints": ["http://127.0.0.1:8080"],
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
                    "detail": "top-logprob telemetry present",
                    "signal": "top_logprobs",
                },
            }
        ],
    }


def test_scenario_verify_5057_cached_model_without_endpoint_stays_reusable(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5057: missing judge server does not erase SOTA cache readiness."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        model_resolver=_one_model_resolver,
        endpoint_probe=_blocked_endpoint_probe,
        endpoints=["http://127.0.0.1:8080"],
        now=lambda: 100.0,
        write=True,
    )

    assert artifact["honest_verdict"] == "complete_gate_state_preflight_partial_ready"
    assert artifact["sota_models_ready"] is True
    assert artifact["sota_judge_ready"] is False
    assert artifact["top_logprob_or_confidence_ready"] is False
    assert artifact["tool_first_verifier_ready"] is True
    assert artifact["legacy_models_smoke_only"] is True
    assert artifact["usable_sota_models"] == [
        {
            "role": "flagship_moe",
            "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
            "model_path": "/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        }
    ]
    assert artifact["model_specs"]["flagship_moe"]["resolved_path"].endswith(".gguf")
    assert artifact["model_specs"]["flagship_moe"]["missing_diagnostic"] is None
    assert artifact["model_specs"]["flagship_dense"]["resolved_path"] is None
    assert "endpoint_completion_unavailable" in artifact["skip_reasons"]
    assert "top_logprob_or_confidence_unavailable" in artifact["skip_reasons"]
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5057_telemetry_field_is_independent_of_model_cache(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-5057: telemetry readiness is not hidden by missing models."""

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_no_model_resolver,
        endpoint_probe=_telemetry_ready_endpoint_probe,
        endpoints=["http://127.0.0.1:8080"],
        now=lambda: 200.0,
        write=False,
    )

    assert artifact["honest_verdict"] == "complete_gate_state_preflight_partial_ready"
    assert artifact["sota_models_ready"] is False
    assert artifact["top_logprob_or_confidence_ready"] is True
    assert artifact["sota_judge_ready"] is False
    assert artifact["tool_first_verifier_ready"] is True
    assert artifact["usable_sota_models"] == []
    assert "sota_models_unavailable" in artifact["skip_reasons"]
    assert "sota_judge_unavailable" in artifact["skip_reasons"]
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5057_tool_first_smoke_and_schema_errors_are_deterministic(
    tmp_path: Path,
) -> None:
    """REQ-VERIFY-5057: JSON, constraint, and evidence checks gate tool readiness."""

    smoke = mod.run_tool_first_verifier_smoke()
    assert smoke["ready"] is True
    assert {check["name"] for check in smoke["checks"]} == {
        "json_parse_check",
        "constraint_check",
        "evidence_check",
    }

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_one_model_resolver,
        endpoint_probe=_telemetry_ready_endpoint_probe,
        endpoints=["http://127.0.0.1:8080"],
        now=lambda: 300.0,
        write=False,
    )
    assert artifact["honest_verdict"] == "complete_gate_state_preflight_all_ready"
    assert artifact["sota_judge_ready"] is True
    assert artifact["skip_reasons"] == []

    broken = dict(artifact)
    broken.pop("skip_reasons")
    broken["legacy_models_smoke_only"] = False
    broken["tool_first_verifier_ready"] = "true"
    broken["model_specs"] = {}
    errors = mod.artifact_schema_errors(broken)
    assert "missing field: skip_reasons" in errors
    assert "legacy_models_smoke_only must be true" in errors
    assert "tool_first_verifier_ready must be a bool" in errors
    assert "model_specs.flagship_moe missing" in errors


def test_req_verify_5057_validation_branches_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5057: malformed readiness artifacts expose exact schema errors."""

    empty_endpoint = {
        "candidate_endpoints": ["http://127.0.0.1:8080"],
        "selected_endpoint": None,
        "completion_ready": False,
        "top_logprob_ready": False,
        "confidence_ready": False,
        "telemetry_signal": None,
        "probes": [],
    }
    no_readiness = mod.build_artifact(
        model_specs={
            "flagship_moe": {
                "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
                "preferred_quant": "Q4_K_M",
                "resolved_path": None,
                "missing_diagnostic": "missing",
            },
            "flagship_dense": {
                "hf_id": "unsloth/gemma-4-31B-it-GGUF",
                "preferred_quant": "Q4_K_M",
                "resolved_path": None,
                "missing_diagnostic": "missing",
            },
            "middle_moe": {
                "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "preferred_quant": "Q4_K_M",
                "resolved_path": None,
                "missing_diagnostic": "missing",
            },
        },
        usable_sota_models=[],
        endpoint_summary=empty_endpoint,
        tool_first_verifier_summary={"ready": False, "checks": []},
        duration_s=-1.0,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH,
    )
    assert no_readiness["honest_verdict"] == "blocked_gate_state_preflight_no_ready_paths"
    assert "tool_first_verifier_unavailable" in no_readiness["skip_reasons"]
    assert no_readiness["duration_s"] == 0.0
    assert mod.artifact_schema_errors(no_readiness) == []

    wrong_hf = dict(no_readiness)
    wrong_hf["model_specs"] = dict(no_readiness["model_specs"])
    wrong_hf["model_specs"]["flagship_moe"] = dict(wrong_hf["model_specs"]["flagship_moe"])
    wrong_hf["model_specs"]["flagship_moe"]["hf_id"] = "wrong/model-GGUF"
    assert "model_specs.flagship_moe.hf_id mismatch" in mod.artifact_schema_errors(wrong_hf)

    no_path_or_diag = dict(no_readiness)
    no_path_or_diag["model_specs"] = dict(no_readiness["model_specs"])
    no_path_or_diag["model_specs"]["flagship_moe"] = dict(
        no_path_or_diag["model_specs"]["flagship_moe"]
    )
    no_path_or_diag["model_specs"]["flagship_moe"]["missing_diagnostic"] = None
    assert "model_specs.flagship_moe needs resolved_path or missing_diagnostic" in (
        mod.artifact_schema_errors(no_path_or_diag)
    )

    not_mapping = dict(no_readiness, model_specs=[])
    assert "model_specs must be a mapping" in mod.artifact_schema_errors(not_mapping)

    bad_endpoint_mapping = dict(no_readiness, endpoint_summary=[])
    assert "endpoint_summary must be a mapping" in mod.artifact_schema_errors(
        bad_endpoint_mapping
    )

    bad_endpoint_probes = dict(no_readiness, endpoint_summary={"probes": "not-list"})
    assert "endpoint_summary.probes must be a list" in mod.artifact_schema_errors(
        bad_endpoint_probes
    )

    bad_verdict = dict(no_readiness, honest_verdict="draft")
    assert "unexpected honest_verdict: 'draft'" in mod.artifact_schema_errors(bad_verdict)

    bad_judge_implication = dict(
        no_readiness,
        sota_judge_ready=True,
        top_logprob_or_confidence_ready=False,
    )
    assert "sota_judge_ready requires top_logprob_or_confidence_ready" in (
        mod.artifact_schema_errors(bad_judge_implication)
    )

    written_bad = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        model_resolver=_one_model_resolver,
        endpoint_probe=lambda _endpoints, _timeout_s: {"probes": "not-list"},
        endpoints=["http://127.0.0.1:8080"],
        now=lambda: 400.0,
        write=False,
    )
    assert written_bad["schema_errors"] == ["endpoint_summary.probes must be a list"]

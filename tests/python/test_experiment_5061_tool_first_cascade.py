"""Tests for Exp 5061 tool-first D6 cascade.

Spec refs: REQ-VERIFY-5061, SCENARIO-VERIFY-5061.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5061_tool_first_cascade as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _exp5057(*, tool_ready: bool = True, judge_ready: bool = False) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_5057_gate_state_preflight_v465.v1",
        "tool_first_verifier_ready": tool_ready,
        "sota_judge_ready": judge_ready,
        "legacy_models_smoke_only": True,
        "model_specs": {
            role: {"hf_id": hf_id, "resolved_path": f"/models/{role}.gguf"}
            for role, hf_id in mod.MANDATED_MODEL_SPECS.items()
        },
        "tool_first_verifier_summary": {
            "ready": tool_ready,
            "checks": [
                {"name": "json_parse_check", "ready": tool_ready},
                {"name": "constraint_check", "ready": tool_ready},
                {"name": "evidence_check", "ready": tool_ready},
            ],
        },
    }


def _exp5059(
    *,
    best_arm_available: bool = True,
    verifier: list[int] | None = None,
    tuned_sc: list[int] | None = None,
    cached_judge: list[int] | None = None,
    predictions: list[str | None] | None = None,
) -> dict[str, Any]:
    verifier_correct = verifier or [1, 1, 1, 0]
    tuned_correct = tuned_sc or [1, 0, 1, 0]
    paired_correct: dict[str, list[int]] = {
        "verifier": verifier_correct,
        "tuned_self_consistency": tuned_correct,
    }
    if cached_judge is not None:
        paired_correct["cached_sota_judge"] = cached_judge
    return {
        "schema": "carnot.experiment_5059_d1_sota_refresh_audit.v1",
        "best_arm_available": best_arm_available,
        "verifier_is_oracle": False,
        "legacy_models_smoke_only": True,
        "accuracy": sum(verifier_correct) / len(verifier_correct),
        "tuned_sc_accuracy": sum(tuned_correct) / len(tuned_correct),
        "n_questions": len(verifier_correct),
        "model_specs": {"mandated_sota": dict(mod.MANDATED_MODEL_SPECS)},
        "refreshed_candidate_metrics": {
            "predictions": predictions or ["A", "B", "C", "D"],
            "paired_correct": paired_correct,
            "tuned_self_consistency": {"candidate_pool_counts": [3] * len(verifier_correct)},
        },
        "scorer_source": {"method": "cached_exp5045_powered_d1_selection_projection"},
    }


def _write_gates(
    root: Path,
    *,
    exp5057: dict[str, Any] | None = None,
    exp5059: dict[str, Any] | None = None,
) -> None:
    _write_json(root / mod.EXP5057_RESULT_RELATIVE_PATH, exp5057 or _exp5057())
    _write_json(root / mod.EXP5059_RESULT_RELATIVE_PATH, exp5059 or _exp5059())


def test_req_verify_5061_spec_declares_tool_first_contract() -> None:
    """REQ-VERIFY-5061: OpenSpec anchors the tool-first cascade artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    module_text = (REPO / mod.MODULE_RELATIVE_PATH).read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5061",
        "SCENARIO-VERIFY-5061",
        "experiment_5061_tool_first_cascade.py",
        "results/experiment_5061_tool_first_cascade.json",
        "deterministic constraint checks",
        "SAFE-style evidence checks",
        "blocked_judge_server",
        "judge_call_fraction",
        "legacy_models_smoke_only",
    ):
        assert marker in spec
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec
    for hf_id in mod.MANDATED_MODEL_SPECS.values():
        assert hf_id in spec
        assert hf_id in module_text


def test_scenario_verify_5061_runs_without_sota_judge_server(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5061: missing SOTA judge still executes tool-first replay."""

    _write_gates(tmp_path)
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH

    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        bootstrap_samples=64,
        now=lambda: 100.0,
        write=True,
    )

    assert artifact["honest_verdict"].startswith("success_tool_first_cascade_parity")
    assert "blocked_judge_server" not in artifact["honest_verdict"]
    assert artifact["cascade_executed"] is True
    assert artifact["tool_first_path_used"] is True
    assert artifact["sota_judge_used"] is False
    assert artifact["baseline_source"] == "tuned_self_consistency"
    assert artifact["cascade_accuracy"] == pytest.approx(0.75)
    assert artifact["judge_only_accuracy"] == pytest.approx(0.5)
    assert artifact["delta_vs_judge_only"] == pytest.approx(0.25)
    assert artifact["judge_call_fraction"] == pytest.approx(0.0)
    assert artifact["judge_call_reduction"] == pytest.approx(1.0)
    assert artifact["tool_call_count"] == 8
    assert artifact["verifier_call_count"] == 4
    assert artifact["efficiency_win"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["legacy_models_smoke_only"] is True
    assert mod.artifact_schema_errors(artifact) == []
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact


def test_scenario_verify_5061_charges_cached_sota_judge_fallback(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5061: cached SOTA judge fallback is optional and charged."""

    _write_gates(
        tmp_path,
        exp5057=_exp5057(tool_ready=True, judge_ready=True),
        exp5059=_exp5059(
            verifier=[1, 0, 0, 0],
            tuned_sc=[0, 0, 0, 0],
            cached_judge=[1, 1, 0, 0],
            predictions=["A", None, "C", "D"],
        ),
    )

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / "out.json",
        bootstrap_samples=64,
        write=False,
    )

    assert artifact["cascade_executed"] is True
    assert artifact["sota_judge_used"] is True
    assert artifact["baseline_source"] == "cached_sota_judge"
    assert artifact["cascade_accuracy"] == pytest.approx(0.5)
    assert artifact["judge_only_accuracy"] == pytest.approx(0.5)
    assert artifact["judge_call_count"] == 1
    assert artifact["judge_call_fraction"] == pytest.approx(0.25)
    assert artifact["route_counts"]["sota_judge_fallback"] == 1
    assert artifact["route_counts"]["cheap_verifier"] == 3
    assert artifact["verifier_call_count"] == 4
    assert artifact["efficiency_win"] is True
    assert mod.artifact_schema_errors(artifact) == []


def test_req_verify_5061_blocks_only_failed_tool_first_preconditions(tmp_path: Path) -> None:
    """REQ-VERIFY-5061: failed tool or best-arm gates fail closed without judge blocking."""

    _write_gates(tmp_path, exp5057=_exp5057(tool_ready=False), exp5059=_exp5059())
    tool_blocked = mod.run(root=tmp_path, artifact_path=tmp_path / "tool.json", write=True)
    assert tool_blocked["honest_verdict"] == "blocked_tool_first_verifier_unavailable"
    assert tool_blocked["cascade_executed"] is False
    assert "blocked_judge_server" not in tool_blocked["honest_verdict"]
    assert mod.artifact_schema_errors(tool_blocked) == []

    _write_gates(
        tmp_path,
        exp5057=_exp5057(tool_ready=True),
        exp5059=_exp5059(best_arm_available=False),
    )
    arm_blocked = mod.run(root=tmp_path, artifact_path=tmp_path / "arm.json", write=False)
    assert arm_blocked["honest_verdict"] == "blocked_exp5059_best_arm_unavailable"
    assert arm_blocked["cascade_executed"] is False
    assert arm_blocked["tool_first_path_used"] is True
    assert "blocked_judge_server" not in arm_blocked["honest_verdict"]
    assert mod.artifact_schema_errors(arm_blocked) == []

    malformed = dict(tool_blocked)
    malformed["cascade_accuracy"] = "bad"
    malformed["legacy_models_smoke_only"] = False
    malformed["paired_ci95"] = [0.0]
    errors = mod.artifact_schema_errors(malformed)
    assert "cascade_accuracy" in errors
    assert "legacy_models_smoke_only" in errors
    assert "paired_ci95" in errors


def test_req_verify_5061_helper_edges_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-5061: malformed replay inputs stay non-executing and non-oracle."""

    assert mod._number(True) is None
    assert mod._number("bad") is None
    assert mod._as_binary_list("bad") == []
    assert mod._as_binary_list(["bad"]) == []
    assert mod._prediction_list({"predictions": "bad"}) == []
    assert mod._evidence_available({}, {"paired_correct": {"verifier": [1]}}) is True

    route_counts = {
        "cheap_verifier": 0,
        "tuned_sc_fallback": 0,
        "sota_judge_fallback": 0,
        "abstain_uncertain": 0,
    }
    correct, tool_calls, judge_calls = mod._fallback_correct(
        baseline_source="tuned_self_consistency",
        baseline_correct=[1],
        index=0,
        route_counts=route_counts,
    )
    assert (correct, tool_calls, judge_calls) == (1, 1, 0)
    assert route_counts["tuned_sc_fallback"] == 1

    _write_gates(
        tmp_path,
        exp5057=_exp5057(tool_ready=True),
        exp5059={
            **_exp5059(),
            "refreshed_candidate_metrics": {
                "paired_correct": {
                    "verifier": [],
                    "tuned_self_consistency": [],
                },
                "predictions": [],
            },
        },
    )
    blocked = mod.run(root=tmp_path, artifact_path=tmp_path / "empty.json", write=False)
    assert blocked["honest_verdict"] == "blocked_tool_first_execution_unavailable"
    assert blocked["cascade_executed"] is False
    assert mod.artifact_schema_errors(blocked) == []

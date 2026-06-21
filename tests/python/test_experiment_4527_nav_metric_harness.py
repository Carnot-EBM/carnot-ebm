"""Tests for Exp 4527 nav metric submission-gate hardening.

Spec refs: REQ-ARC-FCP-4527, SCENARIO-ARC-FCP-4527.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot import experiment_4527_nav_metric_harness as exp4527


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "gate_help_precondition": True,
        "baseline_file_present": True,
        "spec_has_req_4527": True,
        "ok": True,
    }


def _gate_result() -> dict[str, object]:
    return {
        "pass": True,
        "verdict": "PASS (non-inferior): CORE per-level efficiency 2.0074 vs baseline 2.0074",
        "baseline_guard": {"ok": True, "errors": []},
        "lever_dashboard_row": {
            "lever": "submitted_default",
            "verdict_pass": True,
            "nav_regression_warning": "WARN: reset_replay_steps increased for ['lp85']",
        },
        "current": {
            "policy": "e3",
            "games": list(exp4527.CANONICAL_GAME_SET),
            "solved_games": list(exp4527.CANONICAL_CORE_GAMES),
            "core_efficiency": exp4527.CORE_EFFICIENCY_BASELINE,
            "efficiency_by_game": {
                "lp85": 2.0069,
                "m0r0": 0.0003,
                "sp80": 0.0001,
                "vc33": 0.0001,
            },
            "per_level_efficiency_by_game": {
                "lp85": 2.0069,
                "m0r0": 0.0003,
                "sp80": 0.0001,
                "vc33": 0.0001,
            },
            "deepest_level_by_game": {
                "lp85": 1,
                "m0r0": 1,
                "sp80": 1,
                "vc33": 1,
            },
            "navigation_by_game": {
                "lp85": {"reset_replay_steps": 25, "forward_walk_hit_rate": 0.10},
                "m0r0": {"reset_replay_steps": 10, "forward_walk_hit_rate": 0.25},
                "sp80": {"reset_replay_steps": 10, "forward_walk_hit_rate": 0.25},
                "vc33": {"reset_replay_steps": 10, "forward_walk_hit_rate": 0.25},
            },
            "per_game": [
                {
                    "game": "lp85",
                    "solved": True,
                    "actions": 7792,
                    "efficiency": 2.0069,
                    "per_level_efficiency": 2.0069,
                    "levels": 1,
                    "deepest_level_reached": 1,
                    "reset_replay_steps": 25,
                    "forward_walk_hit_rate": 0.10,
                }
            ],
        },
    }


def test_req_arc_fcp_4527_spec_declares_nav_metric_harness_contract() -> None:
    """REQ-ARC-FCP-4527: OpenSpec anchors the nav metric harness fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4527" in spec
    assert "SCENARIO-ARC-FCP-4527" in spec
    assert exp4527.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4527.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_req_arc_fcp_4527_builds_terminal_artifact_with_required_fields() -> None:
    """REQ-ARC-FCP-4527: artifact records score fields, nav fields, and tests."""

    artifact = exp4527.build_artifact(
        preconditions_checked=_preconditions(),
        gate_result=_gate_result(),
        tests_added_pass={"command": "fixture", "passed": True},
        duration_s=0.25,
    )

    assert exp4527.artifact_schema_errors(artifact) == []
    assert artifact["honest_verdict"] == "shipped: nav_metric_first_class_ci_guarded"
    assert artifact["inference_substrate"] == exp4527.INFERENCE_SUBSTRATE
    assert artifact["nav_metric_added"]["per_game_fields"] == [
        "deepest_level_reached",
        "per_level_efficiency",
        "reset_replay_steps",
        "forward_walk_hit_rate",
    ]
    assert artifact["nav_metric_added"]["secondary_warning"] == (
        "WARN: reset_replay_steps increased for ['lp85']"
    )
    assert artifact["per_game_deepest_level_reached"]["lp85"] == 1
    assert artifact["per_game_per_level_efficiency"]["lp85"] == 2.0069
    assert artifact["per_game_nav_diagnostics"]["lp85"]["reset_replay_steps"] == 25
    assert artifact["tests_added_pass"]["passed"] is True


def test_scenario_arc_fcp_4527_run_writes_json_with_injected_gate(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4527: run() writes the stable deliverable JSON."""

    artifact = exp4527.run(
        root=tmp_path,
        write=True,
        preconditions_checked=_preconditions(),
        gate_runner=lambda _root: _gate_result(),
        tests_runner=lambda _root: {"command": "fixture", "passed": True},
        now=lambda: 1.0,
    )

    out = tmp_path / exp4527.RESULT_RELATIVE_PATH
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == artifact
    assert artifact["honest_verdict"].startswith("shipped:")


def test_req_arc_fcp_4527_fallback_extractors_read_per_game_rows() -> None:
    """REQ-ARC-FCP-4527: artifact helpers preserve fields when maps are absent."""

    current = {
        "deepest_level_by_game": [],
        "per_game": [
            [],
            {"game": None, "levels": 4},
            {
                "game": "lp85",
                "reached": 2,
                "efficiency": 1.5,
                "navigation_diagnostics": {
                    "reset_replay_steps": 8,
                    "forward_walk_hit_rate": 0.75,
                },
            },
            {
                "game": "m0r0",
                "levels": 1,
                "per_level_efficiency": 0.25,
                "reset_replay_steps": 3,
                "forward_walk_hit_rate": 0.5,
            },
            {"game": "sp80", "levels": 0},
        ],
    }

    assert exp4527._map_int(None) == {}
    assert exp4527._deepest_level_by_game(current) == {"lp85": 2, "m0r0": 1, "sp80": 0}
    assert exp4527._per_level_efficiency_by_game(current) == {"lp85": 1.5, "m0r0": 0.25}
    assert exp4527._nav_by_game(current) == {
        "lp85": {"reset_replay_steps": 8, "forward_walk_hit_rate": 0.75},
        "m0r0": {"reset_replay_steps": 3, "forward_walk_hit_rate": 0.5},
        "sp80": {"reset_replay_steps": 0, "forward_walk_hit_rate": 0.0},
    }


def test_req_arc_fcp_4527_honest_verdict_branches() -> None:
    """REQ-ARC-FCP-4527: terminal verdicts name the partial reason."""

    base_kwargs = {
        "preconditions_checked": _preconditions(),
        "gate_result": _gate_result(),
        "tests_added_pass": {"passed": True},
        "deepest": {"lp85": 1},
        "efficiency": {"lp85": 2.0},
        "nav": {"lp85": {"reset_replay_steps": 0, "forward_walk_hit_rate": 1.0}},
    }

    assert exp4527._honest_verdict(**{**base_kwargs, "preconditions_checked": {"ok": False}}) == (
        "blocked_nav_metric_preconditions"
    )
    assert exp4527._honest_verdict(**{**base_kwargs, "gate_result": {"baseline_guard": {"ok": False}}}) == (
        "complete: nav_metric_partial_baseline_guard"
    )
    assert exp4527._honest_verdict(**{**base_kwargs, "tests_added_pass": {"passed": False}}) == (
        "complete: nav_metric_partial_tests_not_green"
    )
    assert exp4527._honest_verdict(**{**base_kwargs, "deepest": {}}) == (
        "complete: nav_metric_partial_missing_per_game_fields"
    )
    assert exp4527._honest_verdict(**{**base_kwargs, "gate_result": {"pass": False}}) == (
        "complete: nav_metric_partial_gate_regression"
    )


def test_req_arc_fcp_4527_schema_errors_cover_guardrails() -> None:
    """REQ-ARC-FCP-4527: schema validation rejects malformed terminal artifacts."""

    valid = exp4527.build_artifact(
        preconditions_checked=_preconditions(),
        gate_result=_gate_result(),
        tests_added_pass={"command": "fixture", "passed": True},
        duration_s=0.25,
    )

    missing = dict(valid)
    missing.pop("honest_verdict")
    assert any("missing required field honest_verdict" in error for error in exp4527.artifact_schema_errors(missing))

    replacements = [
        ("honest_verdict", "maybe"),
        ("inference_substrate", "live_llm_inference"),
        ("field_principles", {}),
        ("canonical_game_set", []),
        ("canonical_core_games", []),
        ("leaderboard_submission", True),
        ("tests_added_pass", {"passed": False}),
        ("nav_metric_added", None),
        ("reproducibility_checksum", "bad"),
    ]
    for key, value in replacements:
        artifact = dict(valid)
        artifact[key] = value
        assert exp4527.artifact_schema_errors(artifact), key

    wrong_nav = dict(valid)
    wrong_nav["nav_metric_added"] = {"per_game_fields": ["deepest_level_reached"]}
    assert any("four per-game fields" in error for error in exp4527.artifact_schema_errors(wrong_nav))

    wrong_map = dict(valid)
    wrong_map["per_game_deepest_level_reached"] = []
    assert any("must be a mapping" in error for error in exp4527.artifact_schema_errors(wrong_map))

    wrong_checksum = dict(valid)
    wrong_checksum["reproducibility_checksum"] = "sha256:bad"
    assert any("checksum must match" in error for error in exp4527.artifact_schema_errors(wrong_checksum))


def test_req_arc_fcp_4527_run_rejects_invalid_generated_artifact(monkeypatch) -> None:
    """REQ-ARC-FCP-4527: run() refuses to write a schema-invalid artifact."""

    monkeypatch.setattr(exp4527, "artifact_schema_errors", lambda _artifact: ["boom"])

    try:
        exp4527.run(
            root=REPO,
            write=False,
            preconditions_checked=_preconditions(),
            gate_runner=lambda _root: _gate_result(),
            tests_runner=lambda _root: {"command": "fixture", "passed": True},
            now=lambda: 1.0,
        )
    except ValueError as exc:
        assert "boom" in str(exc)
    else:  # pragma: no cover - defensive assertion path.
        raise AssertionError("run() should reject schema errors")

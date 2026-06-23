"""Tests for Exp 4640 graded goal-energy generation measurement.

Spec refs: REQ-ARC-WMTE-4640, SCENARIO-ARC-WMTE-4640.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot import experiment_4640_goal_energy_generation_live as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _attempt(sig: str, *, solved: bool, actions: int | None = None) -> dict[str, Any]:
    return {
        "variant_signature": sig,
        "game": sig.split("~", 1)[0],
        "attempted": True,
        "solved": solved,
        "first_win": solved,
        "actions_to_first_levelup": actions if solved else None,
        "actions": actions if actions is not None else 200,
        "reachable_headroom": True,
        "cell_recall": 0.91,
        "reproduction_gate": {
            "reproduced": solved,
            "claimed_level": 1 if solved else 0,
            "reached_level": 1 if solved else 0,
        },
    }


def test_req_arc_wmte_4640_spec_declares_required_artifact_fields() -> None:
    """REQ-ARC-WMTE-4640: OpenSpec declares the goal-energy live schema."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4640" in spec
    assert "SCENARIO-ARC-WMTE-4640" in spec
    assert mod.RESULT_RELATIVE_PATH in spec
    for field in mod.FIELD_PRINCIPLES:
        assert field in spec


def test_scenario_arc_wmte_4640_success_requires_baseline_and_uniform_beaten(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-4640: a lift must beat baseline and uniform energy."""

    baseline = [
        _attempt("g1~color01", solved=True, actions=9),
        _attempt("g2~color01", solved=False),
        _attempt("g3~color01", solved=False),
    ]
    goal = [
        _attempt("g1~color01", solved=True, actions=5),
        _attempt("g2~color01", solved=True, actions=7),
        _attempt("g3~color01", solved=False),
    ]
    uniform = [
        _attempt("g1~color01", solved=True, actions=8),
        _attempt("g2~color01", solved=False),
        _attempt("g3~color01", solved=False),
    ]

    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        baseline_measurement=mod.measurement_from_attempts(baseline),
        goal_energy_measurement=mod.measurement_from_attempts(goal),
        uniform_measurement=mod.measurement_from_attempts(uniform),
        live_path_check={"passed": True},
        parity_test={"passed": True},
        duration_s=1.0,
        n_bootstrap=0,
    )

    assert artifact["honest_verdict"] == "success: goal_energy_live_generation_solverate_up_1"
    assert artifact["live_solve_rate_goal_energy"] == pytest.approx(2 / 3)
    assert artifact["live_solve_rate_baseline"] == pytest.approx(1 / 3)
    assert artifact["solve_rate_delta"] == pytest.approx(1 / 3)
    assert artifact["first_win_rate_delta"] == pytest.approx(1 / 3)
    assert artifact["median_actions_to_win_delta"] == pytest.approx(4.0)
    assert artifact["uniform_energy_ablation_passed"] is True
    assert artifact["chosen_submitted_config"]["goal_energy_enabled"] is True
    assert artifact["offline_reproduced"] is True
    assert mod.validate_artifact(artifact) == []


def test_scenario_arc_wmte_4640_honest_null_keeps_config_unchanged(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-4640: zero deltas are explicitly annotated as a null."""

    rows = [
        _attempt("g1~color01", solved=True, actions=9),
        _attempt("g2~color01", solved=False),
        _attempt("g3~color01", solved=False),
    ]
    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        baseline_measurement=mod.measurement_from_attempts(rows),
        goal_energy_measurement=mod.measurement_from_attempts(rows),
        uniform_measurement=mod.measurement_from_attempts(rows),
        live_path_check={"passed": True},
        parity_test={"passed": True},
        duration_s=1.0,
        n_bootstrap=0,
    )

    assert (
        artifact["honest_verdict"] == "complete: goal_energy_no_live_lift_honest_null_gap_sharpened"
    )
    assert artifact["solve_rate_delta"] == 0.0
    assert artifact["first_win_rate_delta"] == 0.0
    assert artifact["median_actions_to_win_delta"] == 0.0
    assert "honest no-value null" in artifact["null_delta_methodology_note"]
    assert artifact["chosen_submitted_config"] == "unchanged"
    assert artifact["residual_bridge_gaps"]
    assert mod.validate_artifact(artifact) == []


def test_req_arc_wmte_4640_validation_rejects_bad_schema(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-4640: schema validation catches unsafe field shapes."""

    rows = [_attempt("g1~color01", solved=True, actions=9)]
    artifact = mod.build_artifact(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        baseline_measurement=mod.measurement_from_attempts(rows),
        goal_energy_measurement=mod.measurement_from_attempts(rows),
        uniform_measurement=mod.measurement_from_attempts(rows),
        live_path_check={"passed": True},
        parity_test={"passed": True},
        duration_s=1.0,
        n_bootstrap=0,
    )
    bad: dict[str, Any] = dict(artifact)
    bad["honest_verdict"] = "not terminal"
    bad["verifier_is_oracle"] = True
    bad["field_principles"] = {}
    bad["uniform_energy_ablation_passed"] = "yes"

    errors = mod.validate_artifact(bad)

    assert "honest_verdict must be terminal-prefixed" in errors
    assert "verifier_is_oracle must be false" in errors
    assert "field_principles mismatch" in errors
    assert "uniform_energy_ablation_passed must be a bare bool" in errors


def test_req_arc_wmte_4640_run_writes_artifact_with_fixture_runner(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-4640: run() writes a stable artifact from matched arm attempts."""

    attempts: dict[str, list[Mapping[str, Any]]] = {
        "baseline": [_attempt("g1~color01", solved=True, actions=9)],
        "goal_energy": [_attempt("g1~color01", solved=True, actions=6)],
        "uniform_energy": [_attempt("g1~color01", solved=True, actions=9)],
    }

    artifact = mod.run(
        root=tmp_path,
        preconditions_checked=mod.ok_preconditions_for_tests(),
        arm_attempts=attempts,
        live_path_check=lambda _root: {"passed": True},
        parity_test=lambda _root: {"passed": True},
        write=True,
        now=lambda: 10.0,
        sleep_fn=lambda _seconds: None,
        n_bootstrap=0,
    )

    written = tmp_path / mod.RESULT_RELATIVE_PATH
    assert written.exists()
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert mod.validate_artifact(artifact) == []

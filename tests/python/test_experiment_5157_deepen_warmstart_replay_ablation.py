"""Tests for Exp 5157 deepen warm-start replay ablation.

Spec refs: REQ-ARC-WMTE-5157,
SCENARIO-ARC-WMTE-5157-TRACE-PRECONDITION,
SCENARIO-ARC-WMTE-5157-REDRAW-WARM-START,
SCENARIO-ARC-WMTE-5157-STABLE-ARTIFACT.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_5157_deepen_warmstart_replay_ablation_v473 as exp5157
from carnot.agentic.arc_executable_world_model import Transition


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"


def _transition(
    value: int,
    next_value: int,
    *,
    action: int = 1,
    level_before: int = 0,
    level_after: int | None = None,
) -> Transition:
    grid = np.zeros((3, 3), dtype=np.int16)
    grid[1, 1] = value
    next_grid = grid.copy()
    next_grid[1, 1] = next_value
    return Transition(
        grid=grid,
        action=action,
        data=None,
        next_grid=next_grid,
        level_before=level_before,
        level_after=level_before if level_after is None else level_after,
    )


def _breakdown(
    game: str,
    delta: float,
    *,
    actions_cold: int = 10,
    actions_warmstart: int = 10,
) -> dict[str, object]:
    cold = 0.25
    return {
        "game": game,
        "level_from": 0,
        "level_to": 1,
        "cold_accuracy": cold,
        "warmstart_accuracy": cold + delta,
        "accuracy_delta": delta,
        "actions_cold": actions_cold,
        "actions_warmstart": actions_warmstart,
        "actions_saved_pct": (
            (actions_cold - actions_warmstart) / actions_cold if actions_cold else 0.0
        ),
        "heldout_changed_cells": 1,
        "cold_diag": {"warm_start": False},
        "warmstart_diag": {"warm_start": True},
    }


def _checks() -> dict[str, object]:
    return {
        "AGENTS.md": True,
        "CODEX.md": True,
        "research_references_v473_read": True,
        "experiment_5155_read": True,
        "registry_trace_precondition": "passed",
    }


def test_req_arc_wmte_5157_spec_declares_ablation_contract() -> None:
    """REQ-ARC-WMTE-5157: OpenSpec anchors the Exp5157 artifact contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in exp5157.SPEC_REFS + (exp5157.RESULT_RELATIVE_PATH,):
        assert marker in spec
    for field, principle in exp5157.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_scenario_arc_wmte_5157_redraw_warm_start_beats_cold_low_data() -> None:
    """SCENARIO-ARC-WMTE-5157-REDRAW-WARM-START: prior transitions enter induction."""

    pre_boundary = tuple(_transition(0, 4, level_before=0) for _ in range(8))
    heldout = _transition(0, 4, level_before=1, level_after=1)
    case = exp5157.BoundaryCase(
        game="toy",
        level_from=0,
        level_to=1,
        pre_boundary=pre_boundary,
        post_boundary=(heldout,),
        source_artifact="synthetic",
    )

    row = exp5157.evaluate_boundary_case(case, dynamics_backend="dsl")

    assert row["cold_accuracy"] == 0.0
    assert row["warmstart_accuracy"] == 1.0
    assert row["accuracy_delta"] == 1.0
    assert row["cold_diag"]["skip"] == "too_few_transitions"
    assert row["warmstart_diag"]["warm_start"] is True
    assert row["warmstart_diag"]["prior_transition_count"] == 8


def test_scenario_arc_wmte_5157_trace_precondition_counts_recoverable_boundaries() -> None:
    """SCENARIO-ARC-WMTE-5157-TRACE-PRECONDITION: only usable boundaries count."""

    good = exp5157.BoundaryCase(
        game="g1",
        level_from=0,
        level_to=1,
        pre_boundary=(_transition(0, 1, level_before=0, level_after=1),),
        post_boundary=(_transition(1, 2, level_before=1),),
        source_artifact="a.json",
    )
    no_heldout = exp5157.BoundaryCase(
        game="g2",
        level_from=0,
        level_to=1,
        pre_boundary=(_transition(0, 1, level_before=0, level_after=1),),
        post_boundary=(_transition(1, 1, level_before=1),),
        source_artifact="b.json",
    )

    recoverable = exp5157.recoverable_boundaries([good, no_heldout])

    assert recoverable == [good]
    assert exp5157.boundary_has_changed_heldout(good) is True
    assert exp5157.boundary_has_changed_heldout(no_heldout) is False


def test_req_arc_wmte_5157_artifact_gate_and_checksum(tmp_path: Path) -> None:
    """SCENARIO-ARC-WMTE-5157-STABLE-ARTIFACT: gate and required fields are stable."""

    breakdown = [_breakdown(f"g{i}", 0.12, actions_cold=10, actions_warmstart=10) for i in range(6)]
    artifact = exp5157.build_artifact(breakdown, preconditions_checked=_checks())

    assert artifact["honest_verdict"].startswith("success:")
    assert artifact["gate_passed"] is True
    assert artifact["warmstart_vs_cold_delta_median"] == 0.12
    assert artifact["actions_saved_pct_median"] == 0.0
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["offline_reproduced"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["games_tested"] == [
        {"game": f"g{i}", "n_level_transitions_tested": 1} for i in range(6)
    ]
    assert artifact["reproducibility_checksum"] == exp5157.reproducibility_checksum(artifact)
    exp5157.validate_artifact(artifact)

    output = tmp_path / exp5157.RESULT_RELATIVE_PATH
    exp5157.write_artifact(artifact, output)
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_req_arc_wmte_5157_gate_can_pass_on_action_savings() -> None:
    """REQ-ARC-WMTE-5157: the exp5155 action-saving gate is applied verbatim."""

    breakdown = [_breakdown(f"g{i}", 0.0, actions_cold=10, actions_warmstart=7) for i in range(6)]
    artifact = exp5157.build_artifact(breakdown, preconditions_checked=_checks())

    assert artifact["warmstart_vs_cold_delta_median"] == 0.0
    assert artifact["actions_saved_pct_median"] == 0.3
    assert artifact["gate_passed"] is True


def test_req_arc_wmte_5157_blocked_when_fewer_than_six_boundaries() -> None:
    """SCENARIO-ARC-WMTE-5157-TRACE-PRECONDITION: fewer than six traces blocks."""

    artifact = exp5157.build_blocked_artifact(
        recoverable_games=[{"game": "g1", "n_level_transitions_tested": 1}],
        preconditions_checked=_checks(),
    )

    assert artifact["honest_verdict"].startswith("complete:")
    assert "blocked_insufficient_transition_traces" in artifact["honest_verdict"]
    assert artifact["gate_passed"] is False
    assert artifact["games_tested"] == [{"game": "g1", "n_level_transitions_tested": 1}]
    exp5157.validate_artifact(artifact)


def test_req_arc_wmte_5157_validation_fails_closed() -> None:
    """REQ-ARC-WMTE-5157: malformed artifacts do not validate."""

    artifact = exp5157.build_artifact(
        [_breakdown(f"g{i}", 0.0) for i in range(6)],
        preconditions_checked=_checks(),
    )

    missing = dict(artifact)
    missing.pop("gate_passed")
    missing["reproducibility_checksum"] = exp5157.reproducibility_checksum(missing)
    with pytest.raises(ValueError, match="artifact missing fields"):
        exp5157.validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="done")
    bad_verdict["reproducibility_checksum"] = exp5157.reproducibility_checksum(bad_verdict)
    with pytest.raises(ValueError, match="honest_verdict"):
        exp5157.validate_artifact(bad_verdict)

    bad_provenance = dict(artifact, solve_provenance="live_agent_self_discovery")
    bad_provenance["reproducibility_checksum"] = exp5157.reproducibility_checksum(bad_provenance)
    with pytest.raises(ValueError, match="solve_provenance"):
        exp5157.validate_artifact(bad_provenance)

    bad_checksum = dict(artifact, reproducibility_checksum="sha256:bad")
    with pytest.raises(ValueError, match="checksum"):
        exp5157.validate_artifact(bad_checksum)

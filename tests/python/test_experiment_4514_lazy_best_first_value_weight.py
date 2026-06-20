"""Tests for Exp 4514 lazy best-first value-weight remeasurement.

Spec refs: REQ-ARC-FCP-4514, SCENARIO-ARC-FCP-4514.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4514_lazy_best_first_value_weight as exp4514
from carnot.agentic.arc_competition_agent import StepwiseExplorer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


class CountingValueHead:
    def __init__(self, values: dict[object, float]) -> None:
        self.values = values
        self.calls: list[object] = []

    def __call__(self, frame: object) -> float:
        self.calls.append(frame)
        return float(self.values[frame])


def _node(depth: int, frame: object) -> dict[str, object]:
    return {
        "path": [{"action": 1, "data": None}] * depth,
        "untested": [{"action": 2, "data": None}],
        "value": None,
        "frame": frame,
    }


def _preconditions() -> dict[str, object]:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "offline_arcade_import_smoke": True,
        "torch_import": True,
        "torch_version": "fixture-torch",
        "control_value_weight_0_present": True,
    }


def _lazy_speedup() -> dict[str, object]:
    return {
        "confirmed": True,
        "source": exp4514.LAZY_VALUE_EVAL_SOURCE,
        "speedup_factor": 232.69,
        "routing_quality_preserved": True,
        "lazy_top_k": exp4514.DEFAULT_LAZY_VALUE_TOP_K,
        "cache_by_frame_hash": True,
    }


def _summary(
    weight: float,
    *,
    solve_rate: float,
    core_actions: float,
    wall_s: float = 2.0,
    solved_games: list[str] | None = None,
) -> dict[str, object]:
    solved = solved_games or list(exp4514.CORE_GAMES)
    actions = {game: int(core_actions) for game in solved}
    return {
        "value_weight": float(weight),
        "heldout_solve_rate": float(solve_rate),
        "median_actions_on_core": float(core_actions),
        "median_per_game_wall_s": float(wall_s),
        "core_solves_preserved": all(game in solved for game in exp4514.CORE_GAMES),
        "solved_games": solved,
        "actions_by_game": actions,
        "per_game": [],
    }


def test_req_arc_fcp_4514_spec_declares_lazy_best_first_contract() -> None:
    """REQ-ARC-FCP-4514: OpenSpec anchors the lazy best-first sweep."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4514" in spec
    assert "SCENARIO-ARC-FCP-4514" in spec
    assert exp4514.RESULT_RELATIVE_PATH in spec
    assert "value_weight=5.0" in spec
    assert "arc_solver_kit.reproduce()" in spec
    for field, principle in exp4514.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4514_frontier_lazy_scores_topk_without_filtering() -> None:
    """SCENARIO-ARC-FCP-4514: lazy value scoring never drops tail frontier nodes."""

    value_head = CountingValueHead({"root": 0.1, "near": 0.2, "tail": 0.0})
    explorer = StepwiseExplorer(
        value_head=value_head,
        value_weight=1.0,
        search_mode="best_first",
        lazy_value_top_k=2,
        online_discriminative=False,
    )
    explorer.graph = {
        "h-root": _node(0, "root"),
        "h-near": _node(1, "near"),
        "h-tail": _node(2, "tail"),
    }

    assert explorer._frontier() == "h-root"
    assert value_head.calls == ["root", "near"]
    assert explorer.graph["h-tail"]["untested"] == [{"action": 2, "data": None}]
    assert explorer.graph["h-tail"]["value"] is None

    explorer.graph["h-root"]["untested"] = []
    explorer.graph["h-near"]["untested"] = []
    assert explorer._frontier() == "h-tail"
    assert value_head.calls == ["root", "near", "tail"]

    assert explorer._value("tail", node_hash="h-tail") == pytest.approx(0.0)
    assert explorer._value("tail", node_hash="h-tail") == pytest.approx(0.0)
    diagnostics = explorer.lazy_value_diagnostics()
    assert diagnostics["lazy_top_k"] == 2
    assert diagnostics["value_head_evals"] == 3
    assert diagnostics["cache_hits"] == 2


def test_req_arc_fcp_4514_selects_positive_weight_only_when_core_actions_improve() -> None:
    """REQ-ARC-FCP-4514: a positive winner must beat the 0.0 control on the core gate."""

    per_weight = {
        "0.0": _summary(0.0, solve_rate=0.5, core_actions=100.0),
        "0.5": _summary(0.5, solve_rate=0.5, core_actions=90.0),
        "1.0": _summary(1.0, solve_rate=0.5, core_actions=130.0),
    }

    decision = exp4514.choose_submitted_value_weight(per_weight)

    assert decision["selected_value_weight"] == 0.5
    assert decision["should_raise_submitted_value_weight"] is True
    assert decision["selection_reason"] == "positive_weight_beats_control_on_core_actions"


def test_req_arc_fcp_4514_null_keeps_zero_without_control_beating_weight() -> None:
    """REQ-ARC-FCP-4514: nulls are valid only with an explicit 0.0 control."""

    per_weight = {
        "0.0": _summary(0.0, solve_rate=0.5, core_actions=100.0),
        "0.5": _summary(0.5, solve_rate=0.5, core_actions=100.0),
        "1.0": _summary(
            1.0,
            solve_rate=0.5,
            core_actions=80.0,
            solved_games=["lp85", "m0r0", "sp80"],
        ),
    }

    decision = exp4514.choose_submitted_value_weight(per_weight)

    assert decision["selected_value_weight"] == 0.0
    assert decision["should_raise_submitted_value_weight"] is False
    assert decision["selection_reason"] == "no_positive_weight_beat_control"

    with pytest.raises(ValueError, match="control"):
        exp4514.choose_submitted_value_weight({"0.5": per_weight["0.5"]})


def test_scenario_arc_fcp_4514_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4514: run writes the principle-annotated sweep artifact."""

    def measure_sweep(**_kwargs):
        return {
            "0.0": _summary(0.0, solve_rate=0.5, core_actions=100.0),
            "0.5": _summary(0.5, solve_rate=0.5, core_actions=90.0),
        }

    artifact = exp4514.run(
        root=tmp_path,
        write=True,
        measure_sweep=measure_sweep,
        preconditions_checked=_preconditions(),
        lazy_eval_speedup_confirmed=_lazy_speedup(),
        random_seed=4514,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "success: lazy_value_weight_0.5_beats_0"
    assert artifact["inference_substrate"] == exp4514.INFERENCE_SUBSTRATE
    assert artifact["chosen_submitted_value_weight"] == 0.5
    assert artifact["control_value_weight_0"]["value_weight"] == 0.0
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["random_seed"] == 4514
    assert artifact["lazy_eval_speedup_confirmed"]["confirmed"] is True
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert exp4514.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4514.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["chosen_submitted_value_weight"] == 0.5


def test_req_arc_fcp_4514_null_artifact_schema_accepts_chosen_zero() -> None:
    """REQ-ARC-FCP-4514: an honest null with chosen 0.0 is schema-valid."""

    per_weight = {
        "0.0": _summary(0.0, solve_rate=0.5, core_actions=100.0),
        "0.5": _summary(0.5, solve_rate=0.5, core_actions=100.0),
    }
    decision = exp4514.choose_submitted_value_weight(per_weight)
    artifact = exp4514.build_artifact(
        preconditions_checked=_preconditions(),
        per_weight_results=per_weight,
        lazy_eval_speedup_confirmed=_lazy_speedup(),
        decision=decision,
        random_seed=4514,
        duration_s=1.0,
    )

    assert artifact["honest_verdict"] == "complete: lazy_value_weight_null_keep_0"
    assert artifact["chosen_submitted_value_weight"] == 0.0
    assert exp4514.artifact_schema_errors(artifact) == []


def test_req_arc_fcp_4514_artifact_schema_rejects_bad_fields() -> None:
    """REQ-ARC-FCP-4514: schema rejects unprincipled or unsafe artifacts."""

    per_weight = {
        "0.0": _summary(0.0, solve_rate=0.5, core_actions=100.0),
        "0.5": _summary(0.5, solve_rate=0.5, core_actions=90.0),
    }
    decision = exp4514.choose_submitted_value_weight(per_weight)
    artifact = exp4514.build_artifact(
        preconditions_checked=_preconditions(),
        per_weight_results=per_weight,
        lazy_eval_speedup_confirmed=_lazy_speedup(),
        decision=decision,
        random_seed=4514,
        duration_s=1.0,
    )
    assert exp4514.artifact_schema_errors(artifact) == []

    bad = {
        **artifact,
        "honest_verdict": "done",
        "inference_substrate": "live_llm_inference",
        "field_principles": {},
        "false_negative_risk_checked": False,
        "lazy_eval_speedup_confirmed": {"confirmed": False},
        "chosen_submitted_value_weight": 2.0,
        "control_value_weight_0": {},
        "reproducibility_checksum": "bad",
    }

    errors = exp4514.artifact_schema_errors(bad)

    assert "honest_verdict must start with a terminal prefix" in errors
    assert "inference_substrate must match the required substrate" in errors
    assert "field_principles must match required field principles" in errors
    assert "false_negative_risk_checked must be true when control is present" in errors
    assert "lazy_eval_speedup_confirmed.confirmed must be true" in errors
    assert "chosen_submitted_value_weight must match the decision" in errors
    assert "control_value_weight_0 must contain the 0.0 control summary" in errors
    assert "reproducibility_checksum must be sha256-prefixed" in errors

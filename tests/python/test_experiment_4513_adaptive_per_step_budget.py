"""Tests for Exp 4513 ACT-style adaptive per-step budget.

Spec refs: REQ-ARC-FCP-4513, SCENARIO-ARC-FCP-4513.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_4513_adaptive_per_step_budget as exp4513
from carnot.agentic import arc_adaptive_budget as adaptive_budget
from carnot.agentic.arc_adaptive_budget import (
    adaptive_budget_decision,
    apply_adaptive_budget,
    predicted_noop_fraction,
    value_head_margin,
)
from carnot.agentic import arc_graph_explore
from carnot.agentic.arc_agi3_live_adapter import ArcAction
from carnot.agentic.arc_competition_agent import StepwiseExplorer


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"


def _frame(value: int = 0) -> SimpleNamespace:
    grid = np.zeros((8, 8), dtype=np.int16)
    grid[1, 1] = int(value)
    grid[6, 6] = int(value) + 1
    return SimpleNamespace(frame=grid, available_actions=[1, 2, 6])


class CandidateValueHead:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def candidate_values(self, _frame: object, _candidates: list[object]) -> list[float]:
        return list(self.values)


class ChangeScorer:
    def __init__(self, scores: dict[tuple[int, tuple[tuple[str, int], ...]], float]) -> None:
        self.scores = scores

    def candidate_score(self, _frame: object, candidate: object) -> float:
        data = tuple(sorted((getattr(candidate, "data", None) or {}).items()))
        return float(self.scores.get((int(candidate.action_id), data), 0.0))


def _gate_metrics(adaptive_median: float, adaptive_solved: int = 4) -> dict[str, object]:
    return {
        "baseline": {
            "solved_count": 4,
            "median_actions_on_solved": exp4513.BASELINE_MEDIAN_ACTIONS,
            "games": ["aa00", "bb00"],
            "per_game": [
                {"game": "aa00", "solved": True, "actions": 7792},
                {"game": "bb00", "solved": False, "actions": 8000},
            ],
        },
        "with_adaptive": {
            "solved_count": adaptive_solved,
            "median_actions_on_solved": adaptive_median,
            "games": ["aa00", "bb00"],
            "per_game": [
                {
                    "game": "aa00",
                    "solved": True,
                    "actions": int(adaptive_median),
                    "actions_to_first_levelup": int(adaptive_median),
                    "reproduced": True,
                },
                {
                    "game": "bb00",
                    "solved": False,
                    "actions": 8000,
                    "actions_to_first_levelup": None,
                    "reproduced": None,
                },
            ],
            "threshold": 0.55,
            "adaptive_budget_diagnostics": {
                "commit_count": 3,
                "expanded_count": 2,
                "candidates_skipped": 6,
            },
        },
        "threshold_sweep": [
            {
                "threshold": 0.35,
                "solved_count": 4,
                "median_actions_on_solved": 9000.0,
                "per_game": [],
            },
            {
                "threshold": 0.55,
                "solved_count": adaptive_solved,
                "median_actions_on_solved": adaptive_median,
                "per_game": [],
            },
        ],
        "measurement_script": exp4513.LOCAL_GATE_RELATIVE_PATH,
    }


def test_req_arc_fcp_4513_spec_declares_adaptive_budget_artifact_contract() -> None:
    """REQ-ARC-FCP-4513: OpenSpec anchors the adaptive-budget experiment."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-FCP-4513" in spec
    assert "SCENARIO-ARC-FCP-4513" in spec
    assert exp4513.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4513.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle in spec


def test_scenario_arc_fcp_4513_gate_commits_easy_frame_and_expands_ambiguous() -> None:
    """SCENARIO-ARC-FCP-4513: ambiguity below threshold keeps only the top candidate."""

    frame = _frame(3)
    candidates = [
        ArcAction(6, {"x": 6, "y": 6}, "top"),
        ArcAction(1, None, "noop_keyboard"),
        ArcAction(2, None, "noop_keyboard"),
    ]
    scorer = ChangeScorer({(6, (("x", 6), ("y", 6))): 0.95, (1, ()): 0.05, (2, ()): 0.05})
    value_head = CandidateValueHead([0.0, 4.0, 5.0])

    kept, decision = apply_adaptive_budget(
        frame,
        candidates,
        threshold=0.45,
        value_head=value_head,
        frame_change_scorer=scorer,
        frame_is_novel=False,
    )

    assert kept == [candidates[0]]
    assert decision.committed_single_candidate is True
    assert decision.budget == 1
    assert decision.normal_width == 3
    assert decision.components["value_head_margin"] == pytest.approx(4.0)
    assert decision.components["predicted_noop_fraction"] == pytest.approx(2 / 3)
    assert decision.components["frame_novelty"] is False

    ambiguous = adaptive_budget_decision(
        frame,
        candidates,
        threshold=0.45,
        value_head=value_head,
        frame_change_scorer=scorer,
        frame_is_novel=True,
    )
    assert ambiguous.committed_single_candidate is False
    assert ambiguous.budget == 3
    assert ambiguous.ambiguity_score >= 0.45

    expanded, expanded_decision = apply_adaptive_budget(
        frame,
        candidates,
        threshold=0.45,
        value_head=value_head,
        frame_change_scorer=scorer,
        frame_is_novel=True,
    )
    assert expanded == candidates
    assert expanded_decision.committed_single_candidate is False


def test_req_arc_fcp_4513_signal_helpers_cover_defensive_branches() -> None:
    """REQ-ARC-FCP-4513: unavailable or partial signals degrade to neutral ambiguity."""

    frame = _frame(8)
    candidates = [ArcAction(1, None, "a"), ArcAction(2, None, "b")]

    assert value_head_margin(frame, candidates, None) is None
    assert value_head_margin(frame, [candidates[0]], CandidateValueHead([7.0])) == 1.0

    class CandidateValue:
        def candidate_value(self, _frame: object, candidate: object) -> float:
            return 2.0 if candidate is candidates[0] else 5.0

    assert value_head_margin(frame, candidates, CandidateValue()) == 3.0
    assert value_head_margin(frame, candidates, lambda _frame, candidate: 1.0 if candidate is candidates[0] else 4.0) == 3.0
    assert value_head_margin(frame, candidates, CandidateValueHead(["bad", 1.0])) is None

    class BrokenCandidateValues:
        def candidate_values(self, _frame: object, _candidates: list[object]) -> list[float]:
            raise RuntimeError("broken")

    class BrokenCandidateValue:
        def candidate_value(self, _frame: object, _candidate: object) -> float:
            raise RuntimeError("broken")

    class NotCallable:
        pass

    assert value_head_margin(frame, candidates, BrokenCandidateValues()) is None
    assert value_head_margin(frame, candidates, BrokenCandidateValue()) is None
    assert value_head_margin(frame, candidates, lambda _frame: 0.0) is None

    def broken_callable(_frame: object, _candidate: object) -> float:
        raise RuntimeError("broken")

    assert value_head_margin(frame, candidates, broken_callable) is None
    assert value_head_margin(frame, candidates, NotCallable()) is None

    assert predicted_noop_fraction(frame, candidates, None) is None
    assert adaptive_budget._candidate_change_score(frame, candidates[0], None) is None
    assert predicted_noop_fraction(frame, [], ChangeScorer({})) is None
    assert predicted_noop_fraction(frame, candidates, lambda _frame, candidate: 0.1 if candidate is candidates[0] else 0.9) == 0.5
    assert predicted_noop_fraction(frame, candidates, lambda _frame: 0.1) is None

    class BrokenScorer:
        def candidate_score(self, _frame: object, _candidate: object) -> float:
            raise RuntimeError("broken")

    assert predicted_noop_fraction(frame, candidates, BrokenScorer()) is None
    assert predicted_noop_fraction(frame, candidates, object()) is None

    disabled = adaptive_budget_decision(frame, candidates, threshold=None, frame_is_novel=False)
    assert disabled.enabled is False
    assert disabled.budget == 2

    single = adaptive_budget_decision(frame, [candidates[0]], threshold=0.2, frame_is_novel=False)
    assert single.enabled is True
    assert single.budget == 1


def test_req_arc_fcp_4513_explorer_truncates_candidates_when_gate_commits(monkeypatch) -> None:
    """REQ-ARC-FCP-4513: StepwiseExplorer wires the adaptive budget gate."""

    frame = _frame(4)
    candidates = [
        ArcAction(6, {"x": 6, "y": 6}, "top"),
        ArcAction(1, None, "weak"),
        ArcAction(2, None, "weak"),
    ]
    monkeypatch.setattr(
        arc_graph_explore,
        "rich_action_candidates",
        lambda *_args, **_kwargs: list(candidates),
    )

    explorer = StepwiseExplorer(
        frame_change_scorer=ChangeScorer(
            {(6, (("x", 6), ("y", 6))): 0.95, (1, ()): 0.05, (2, ()): 0.05}
        ),
        adaptive_budget_threshold=0.45,
        adaptive_budget_value_head=CandidateValueHead([0.0, 4.0, 5.0]),
    )
    explorer.graph[explorer._hash(frame)] = {"path": [], "untested": []}

    rows = explorer._candidates(frame, path=[])
    diagnostics = explorer.adaptive_budget_diagnostics()

    # 2026-07-25: rows now carry an additive 'tier' annotation when the frontier tier barrier
    # is enabled (shipped ON -- see the flag block in arc_competition_agent.py). Assert the
    # MEANINGFUL fields rather than exact dict equality, which was brittle to any new annotation.
    assert len(rows) == 1
    assert rows[0]["action"] == 6
    assert rows[0]["data"] == {"x": 6, "y": 6}
    assert diagnostics["enabled"] is True
    assert diagnostics["commit_count"] == 1
    assert diagnostics["expanded_count"] == 0
    assert diagnostics["candidates_skipped"] == 2
    assert diagnostics["history"][-1]["budget"] == 1


def test_scenario_arc_fcp_4513_artifact_schema_and_write(tmp_path: Path) -> None:
    """SCENARIO-ARC-FCP-4513: run writes required JSON and solve-rate guard fields."""

    def measure_gate(**_kwargs):
        return _gate_metrics(120.0)

    artifact = exp4513.run(
        root=tmp_path,
        write=True,
        measure_gate=measure_gate,
        thresholds=(0.35, 0.55),
        random_seed=4513,
        now=lambda: 10.0,
    )

    assert artifact["honest_verdict"] == "success: adaptive_budget_median_actions_120_below_7760"
    assert artifact["median_actions_baseline"] == exp4513.BASELINE_MEDIAN_ACTIONS
    assert artifact["median_actions_with_adaptive"] == 120.0
    assert artifact["solve_rate_baseline"] == 4
    assert artifact["solve_rate_with_adaptive"] == 4
    assert artifact["ambiguity_signal_components"] == exp4513.AMBIGUITY_SIGNAL_COMPONENTS
    assert artifact["positive_control_passed"] is True
    assert artifact["false_negative_risk_checked"] is True
    assert artifact["random_seed"] == 4513
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    assert exp4513.artifact_schema_errors(artifact) == []

    written = json.loads((tmp_path / exp4513.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["honest_verdict"] == artifact["honest_verdict"]


def test_req_arc_fcp_4513_artifact_schema_rejects_bad_fields() -> None:
    """REQ-ARC-FCP-4513: schema rejects unsafe or non-terminal artifacts."""

    artifact = exp4513.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        gate_metrics=_gate_metrics(100.0),
        positive_control={"actions_reduced": True},
        thresholds=(0.35, 0.55),
        selected_threshold=0.55,
        random_seed=4513,
        reproducibility_checksum="sha256:" + "0" * 64,
        duration_s=1.0,
    )
    assert exp4513.artifact_schema_errors(artifact) == []

    bad_verdict = {**artifact, "honest_verdict": "blocked"}
    assert any("terminal prefix" in error for error in exp4513.artifact_schema_errors(bad_verdict))

    dropped_success = {
        **artifact,
        "honest_verdict": "success: adaptive_budget_median_actions_100_below_7760",
        "solve_rate_with_adaptive": 3,
    }
    assert any("solve-rate" in error for error in exp4513.artifact_schema_errors(dropped_success))

    bad_signals = {**artifact, "ambiguity_signal_components": ["novelty_only"]}
    assert any("ambiguity_signal_components" in error for error in exp4513.artifact_schema_errors(bad_signals))

    missing = dict(artifact)
    missing.pop("local_gate_metrics")
    assert any("missing required field" in error for error in exp4513.artifact_schema_errors(missing))

    bad_substrate = {**artifact, "inference_substrate": "live_llm_inference"}
    assert any("substrate" in error for error in exp4513.artifact_schema_errors(bad_substrate))

    bad_principles = {**artifact, "field_principles": {}}
    assert any("field_principles" in error for error in exp4513.artifact_schema_errors(bad_principles))

    bad_baseline = {**artifact, "median_actions_baseline": 1}
    assert any("7760" in error for error in exp4513.artifact_schema_errors(bad_baseline))

    no_control = {**artifact, "positive_control_passed": False}
    assert any("positive_control" in error for error in exp4513.artifact_schema_errors(no_control))

    no_guard = {**artifact, "false_negative_risk_checked": False}
    assert any("false_negative" in error for error in exp4513.artifact_schema_errors(no_guard))

    no_checksum = {**artifact, "reproducibility_checksum": "0" * 64}
    assert any("checksum" in error for error in exp4513.artifact_schema_errors(no_checksum))


def test_req_arc_fcp_4513_defensive_helper_paths(tmp_path: Path) -> None:
    """REQ-ARC-FCP-4513: helper paths stay deterministic and honest."""

    assert exp4513._gate_value({}, "missing", "field") is None
    assert exp4513.false_negative_risk_guard({"actions_reduced": False}, {}) == (
        "positive_control_failed_null_uninterpretable"
    )
    assert exp4513.false_negative_risk_guard(
        {"actions_reduced": True},
        _gate_metrics(8000.0),
    ) == "positive_control_passed_null_interpretable"

    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    (ops_dir / "arc-submission-baseline.json").write_text(
        json.dumps({"solved_count": 1, "median_actions_on_solved": 7760, "games": ["aa00"]}),
        encoding="utf-8",
    )
    assert exp4513.load_gate_baseline(tmp_path)["solved_count"] == 1
    assert exp4513._json_action_label(6, {"x": 1, "y": 2}) == (
        '{"action": 6, "data": {"x": 1, "y": 2}}'
    )

    summary = exp4513._summarize_rows(
        rows=[
            {
                "game": "aa00",
                "solved": True,
                "actions": 9,
                "actions_to_first_levelup": 5,
                "adaptive_budget_diagnostics": {
                    "commit_count": 2,
                    "expanded_count": 1,
                    "candidates_skipped": 4,
                },
            },
            {
                "game": "bb00",
                "solved": False,
                "actions": 11,
                "timed_out": True,
                "adaptive_budget_diagnostics": {},
            },
        ],
        games=["aa00", "bb00"],
        budget=20,
        threshold=0.55,
    )
    assert summary["median_actions_on_solved"] == 5.0
    assert summary["timed_out_count"] == 1
    assert summary["adaptive_budget_diagnostics"] == {
        "commit_count": 2,
        "expanded_count": 1,
        "candidates_skipped": 4,
    }
    assert exp4513._select_best_sweep_row([], baseline_solved=4) == {}
    assert exp4513._select_best_sweep_row(
        [
            {"threshold": 0.35, "solved_count": 3, "median_actions_on_solved": 50},
            {"threshold": 0.55, "solved_count": 2, "median_actions_on_solved": 10},
        ],
        baseline_solved=4,
    )["threshold"] == 0.55
    assert exp4513._select_best_sweep_row(
        [
            {"threshold": 0.35, "solved_count": 4, "median_actions_on_solved": None},
            {"threshold": 0.55, "solved_count": 4, "median_actions_on_solved": 80},
        ],
        baseline_solved=4,
    )["threshold"] == 0.55

    blocked = exp4513.build_artifact(
        preconditions_checked={"offline_arcade_import": False},
        gate_metrics={
            "baseline": {"solved_count": 4, "median_actions_on_solved": 7760},
            "with_adaptive": {"solved_count": 0, "median_actions_on_solved": None},
            "threshold_sweep": [],
        },
        positive_control={"actions_reduced": True},
        thresholds=(0.35,),
        selected_threshold=None,
        random_seed=4513,
        reproducibility_checksum="sha256:" + "1" * 64,
        duration_s=0.0,
    )
    assert blocked["honest_verdict"] == "complete: blocked_offline_arcade_import_failed"

    solve_drop = exp4513.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        gate_metrics=_gate_metrics(100.0, adaptive_solved=3),
        positive_control={"actions_reduced": True},
        thresholds=(0.35, 0.55),
        selected_threshold=0.55,
        random_seed=4513,
        reproducibility_checksum="sha256:" + "2" * 64,
        duration_s=0.0,
    )
    assert solve_drop["honest_verdict"] == "complete: adaptive_budget_solve_rate_guard_failed"

    null = exp4513.build_artifact(
        preconditions_checked={"offline_arcade_import": True},
        gate_metrics=_gate_metrics(8000.0),
        positive_control={"actions_reduced": True},
        thresholds=(0.35, 0.55),
        selected_threshold=0.55,
        random_seed=4513,
        reproducibility_checksum="sha256:" + "3" * 64,
        duration_s=0.0,
    )
    assert null["honest_verdict"] == "complete: adaptive_budget_no_reduction_honest_null"


def test_scenario_arc_fcp_4513_run_blocks_when_offline_arcade_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-4513: missing arcade exits with a blocked terminal artifact."""

    monkeypatch.setattr(
        exp4513,
        "check_preconditions",
        lambda _root: {"offline_arcade_import": False},
    )

    def measure_gate(**_kwargs):
        raise AssertionError("measurement should not run without offline arcade")

    artifact = exp4513.run(
        root=tmp_path,
        write=False,
        measure_gate=measure_gate,
        random_seed=4513,
        now=lambda: 5.0,
    )

    assert artifact["honest_verdict"] == "complete: blocked_offline_arcade_import_failed"


def test_scenario_arc_fcp_4513_run_raises_on_invalid_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-FCP-4513: schema errors prevent writing invalid results."""

    monkeypatch.setattr(
        exp4513,
        "positive_control",
        lambda: {"actions_reduced": False},
    )

    with pytest.raises(ValueError, match="positive_control"):
        exp4513.run(
            root=tmp_path,
            write=False,
            measure_gate=lambda **_kwargs: _gate_metrics(8000.0),
            random_seed=4513,
        )

import json
from pathlib import Path

from carnot.agentic import arc_paw_amortization_gate as gate


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_req_5215_remaining_distribution_uses_levelup_logs_and_reports_missing(
    tmp_path: Path,
) -> None:
    """REQ-ARC-WMTE-5215: compute remaining-action budgets from ARC action logs."""

    _write_json(
        tmp_path / "arc_loop_solve_aa00.json",
        {
            "game": "aa00",
            "moves": 10,
            "reproduced_levels": 2,
            "offline_reproduced": True,
            "solution_labels": ["a"] * 10,
            "level_up_action_indices": [4, 10],
        },
    )
    _write_json(
        tmp_path / "arc_loop_solve_bb00.json",
        {
            "game": "bb00",
            "moves": 20,
            "reproduced_levels": 1,
            "offline_reproduced": True,
            "solution_labels": ["b"] * 20,
            "level_up_actions": [7],
        },
    )
    _write_json(
        tmp_path / "arc_loop_solve_cc00.json",
        {
            "game": "cc00",
            "moves": 5,
            "reproduced_levels": 1,
            "offline_reproduced": True,
            "solution_labels": ["c"] * 5,
        },
    )

    records = gate.load_arc_loop_records(tmp_path)
    distribution = gate.remaining_action_distribution(records)

    assert distribution.values == [6.0, 13.0]
    assert distribution.median == 9.5
    assert distribution.p75 == 11.25
    assert distribution.missing[0]["game"] == "cc00"
    assert "missing_level_up_checkpoint" in distribution.missing[0]["reason"]


def test_scenario_5215_replay_recovers_first_levelup_checkpoint() -> None:
    """SCENARIO-ARC-WMTE-5215-AMORTIZATION-GATE: derive checkpoints by replay."""

    class Frame:
        def __init__(self, level: int) -> None:
            self.levels_completed = level

    class Env:
        def reset(self) -> Frame:
            return Frame(0)

    def apply(_env: Env, label: str, frame: Frame) -> Frame:
        if label == "warm":
            return frame
        return Frame(frame.levels_completed + (1 if label == "up" else 0))

    indices = gate.replay_level_up_action_indices(
        labels=["wait", "up", "wait", "up"],
        env=Env(),
        apply=apply,
        warmup_label="warm",
    )

    assert indices == (2, 4)

    record = gate.ArcEpisodeRecord("aa00", 4, 2, ("wait", "up", "wait", "up"))
    assert record.with_level_up_action_indices(indices).level_up_action_indices == (2, 4)


def test_req_5215_break_even_requires_median_and_p75_margin() -> None:
    """REQ-ARC-WMTE-5215: viability requires both budgets to clear margin."""

    break_even = gate.break_even_remaining_actions(
        compile_wall_clock_s=100.0,
        current_step_wall_clock_s=10.0,
        cheap_step_wall_clock_s=5.0,
    )

    assert break_even == 20.0
    assert gate.paw_amortization_viable(
        median_remaining_actions=30.0,
        p75_remaining_actions=40.0,
        break_even_remaining_actions=break_even,
        margin=1.25,
    )
    assert not gate.paw_amortization_viable(
        median_remaining_actions=24.0,
        p75_remaining_actions=40.0,
        break_even_remaining_actions=break_even,
        margin=1.25,
    )
    assert gate.break_even_remaining_actions(
        compile_wall_clock_s=100.0,
        current_step_wall_clock_s=5.0,
        cheap_step_wall_clock_s=5.0,
    ) == float("inf")


def test_scenario_5215_artifact_is_pure_analysis_and_does_not_claim_solve(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5215-NO-SOLVE-OR-REGISTRY-MUTATION."""

    records = (
        gate.ArcEpisodeRecord(
            game="aa00",
            total_actions=10,
            reached_level=2,
            solution_labels=("a",) * 10,
            level_up_action_indices=(4, 10),
            source_path="results/arc_loop_solve_aa00.json",
        ),
        gate.ArcEpisodeRecord(
            game="bb00",
            total_actions=20,
            reached_level=1,
            solution_labels=("b",) * 20,
            level_up_action_indices=(7,),
            source_path="results/arc_loop_solve_bb00.json",
        ),
    )
    timing = gate.TimingEstimate(
        compile_wall_clock_s=256.0,
        current_step_wall_clock_s=7.0,
        cheap_step_wall_clock_s=2.0,
        evidence={"unit_test": True},
    )

    artifact = gate.build_artifact(
        records=records,
        timing=timing,
        duration_s=1.23,
    )

    assert artifact["arc_registry_modified"]["value"] is False
    assert artifact["inference_substrate"]["value"] == "arc_log_analysis_plus_local_timing"
    assert artifact["honest_verdict"]["value"].startswith("complete_")
    assert "solve" not in artifact["honest_verdict"]["value"].replace("no_arc_solve_claim", "")
    assert artifact["paw_amortization_viable"]["value"] is False
    assert artifact["break_even_remaining_actions"]["value"] == 51.2
    assert artifact["checkpoint_analysis"]["stable_transition_model"]["status"] == "missing_data"

    result_path = tmp_path / "result.json"
    gate.write_artifact(result_path, artifact)
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    assert saved["arc_registry_modified"]["value"] is False

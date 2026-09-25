"""REQ-ARC-WMTE-7639 regressions for guarded planner HUD deduplication."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

from carnot import experiment_7639_v666_arc_goal_dedup as exp
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic import arc_executable_world_model as e3


def _candidate(action: int = 1) -> dict[str, object]:
    return {"action": action, "data": None}


def _transition(before: np.ndarray, after: np.ndarray) -> e3.Transition:
    return e3.Transition(before, 1, None, after, 0, 0)


def _bare_policy(mask, transitions):
    policy = object.__new__(agent.E3AgentPolicy)
    policy.two_sided_goal_contract = None
    policy.explorer = SimpleNamespace(hud_mask=mask)
    policy.cell = 1
    policy.transitions = list(transitions)
    policy._episode_transition_start = 0
    return policy


def test_terminal_masked_duplicate_checks_full_grid_goal_first(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7639-TERMINAL-BEFORE-DUPLICATE keeps a terminal state."""
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate()])
    root = np.zeros((1, 2), dtype=np.int16)
    seen_by_engine: list[np.ndarray] = []
    seen_by_goal: list[np.ndarray] = []

    def engine(grid, _action, _data):
        seen_by_engine.append(np.asarray(grid).copy())
        return np.array([[0, 7]], dtype=np.int16)

    def goal(grid):
        seen_by_goal.append(np.asarray(grid).copy())
        return int(grid[0, 1]) == 7

    diagnostics: dict[str, object] = {}
    plan = e3.plan_in_model(
        engine,
        goal,
        root,
        diagnostics=diagnostics,
        dedup_mask=np.array([[False, True]]),
    )
    assert plan == [_candidate()]
    assert diagnostics["termination_reason"] == "plan_found"
    assert seen_by_engine[0].tolist() == [[0, 0]]
    assert seen_by_goal[0].tolist() == [[0, 7]]


def test_intermediate_masked_counter_is_retained_by_wrapper_guard(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7639-INTERMEDIATE-GOAL-STATE refuses an observed alias."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [_candidate()])
    root = np.zeros((2, 3), dtype=np.int16)
    observed_after = root.copy()
    observed_after[0, :2] = 4
    observed_after[1, 0] = 9
    observed_after[1, 2] = 1
    mask = np.zeros_like(root, dtype=bool)
    mask[1, 2] = True
    policy = _bare_policy(mask, [_transition(root, observed_after)])

    def engine(grid, _action, _data):
        out = np.asarray(grid).copy()
        out[1, 2] += 1
        return out

    diagnostics: dict[str, object] = {}
    plan = policy._call_plan_in_model(
        e3.plan_in_model,
        engine,
        lambda grid: int(grid[1, 2]) == 3,
        root,
        diagnostics=diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert plan == [_candidate()] * 3
    assert diagnostics["planner_hud_dedup_mask_status"] == "refused"
    assert diagnostics["planner_hud_dedup_planner_reason"] == "observed_mask_cell_change"
    assert diagnostics["planner_hud_dedup_equivalence"]["changed_cells_inside_mask"] == 1
    assert "hud_dedup_mask_status" not in diagnostics


@pytest.mark.parametrize(
    ("transitions", "expected_reason"),
    [
        ([], "no_transitions"),
        (
            [
                _transition(
                    np.zeros((2, 3), dtype=np.int16),
                    np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int16),
                )
            ],
            "no_changed_cells_outside_mask_cannot_distinguish",
        ),
    ],
)
def test_unsafe_or_unmeasurable_mask_is_refused(monkeypatch, transitions, expected_reason) -> None:
    """SCENARIO-ARC-WMTE-7639-TRUTHFUL-TELEMETRY names unsafe evidence."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [])
    mask = np.zeros((2, 3), dtype=bool)
    mask[1, 2] = True
    policy = _bare_policy(mask, transitions)
    diagnostics: dict[str, object] = {}
    policy._call_plan_in_model(
        e3.plan_in_model,
        lambda grid, _action, _data: grid,
        lambda _grid: False,
        np.zeros((2, 3), dtype=np.int16),
        diagnostics=diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert diagnostics["planner_hud_dedup_mask_status"] == "refused"
    assert diagnostics["planner_hud_dedup_planner_reason"] == expected_reason


def test_no_mask_and_invalid_shape_are_not_reported_as_applied(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7639-TRUTHFUL-TELEMETRY distinguishes mask absence."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    monkeypatch.setattr(e3, "_model_candidates", lambda _grid: [])
    start = np.zeros((1, 3), dtype=np.int16)

    no_mask_diagnostics: dict[str, object] = {}
    _bare_policy(None, [])._call_plan_in_model(
        e3.plan_in_model,
        lambda grid, _action, _data: grid,
        lambda _grid: False,
        start,
        diagnostics=no_mask_diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert no_mask_diagnostics["planner_hud_dedup_mask_status"] == "unresolved"
    assert no_mask_diagnostics["planner_hud_dedup_planner_reason"] == "explorer_mask_unresolved"

    invalid_diagnostics: dict[str, object] = {}
    invalid_mask = np.zeros((2, 3), dtype=bool)
    invalid_mask[1, :] = True
    _bare_policy(invalid_mask, [])._call_plan_in_model(
        e3.plan_in_model,
        lambda grid, _action, _data: grid,
        lambda _grid: False,
        start,
        diagnostics=invalid_diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert invalid_diagnostics["planner_hud_dedup_mask_status"] == "not_used"
    assert invalid_diagnostics["planner_hud_dedup_planner_reason"] == "shape_mismatch"


def test_callable_signature_rejection_is_truthful(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7639-TRUTHFUL-TELEMETRY reports signature rejection."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    root = np.zeros((2, 3), dtype=np.int16)
    changed = root.copy()
    changed[0, 0] = 1
    mask = np.zeros_like(root, dtype=bool)
    mask[1, :] = True
    policy = _bare_policy(mask, [_transition(root, changed)])
    diagnostics: dict[str, object] = {}

    def planner(_engine, _goal, _grid):
        return []

    assert (
        policy._call_plan_in_model(
            planner,
            object(),
            lambda _grid: False,
            root,
            diagnostics=diagnostics,
            goal_energy_override=lambda _grid: 1.0,
        )
        == []
    )
    assert diagnostics["planner_hud_dedup_mask_status"] == "not_used"
    assert diagnostics["planner_hud_dedup_planner_reason"] == "callable_signature_rejected"


def test_restarted_call_clears_stale_applied_telemetry(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7639-TRUTHFUL-TELEMETRY clears prior call results."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "1")
    root = np.zeros((2, 3), dtype=np.int16)
    changed = root.copy()
    changed[0, 0] = 1
    mask = np.zeros_like(root, dtype=bool)
    mask[1, :] = True
    policy = _bare_policy(mask, [_transition(root, changed)])
    diagnostics: dict[str, object] = {}

    def reports_use(_engine, _goal, _grid, **kwargs):
        inner = kwargs["diagnostics"]
        inner["hud_dedup_mask_status"] = "applied"
        inner["hud_dedup_mask_reason"] = "accepted"
        return []

    policy._call_plan_in_model(
        reports_use,
        object(),
        lambda _grid: False,
        root,
        diagnostics=diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert diagnostics["planner_hud_dedup_mask_status"] == "applied"

    def ignores_mask(_engine, _goal, _grid, **_kwargs):
        return []

    policy._call_plan_in_model(
        ignores_mask,
        object(),
        lambda _grid: False,
        root,
        diagnostics=diagnostics,
        goal_energy_override=lambda _grid: 1.0,
    )
    assert diagnostics["planner_hud_dedup_mask_status"] == "not_used"
    assert diagnostics["planner_hud_dedup_planner_reason"] == "planner_did_not_report_use"


def test_flags_off_does_not_read_mask_or_add_diagnostics(monkeypatch) -> None:
    """SCENARIO-ARC-WMTE-7639-FLAGS-OFF-PARITY avoids mask work."""
    monkeypatch.delenv("CARNOT_ARC_PLAN_HUD_DEDUP", raising=False)

    class ExplodingExplorer:
        @property
        def hud_mask(self):
            raise AssertionError("flags-off path inspected the mask")

    policy = _bare_policy(None, [])
    policy.explorer = ExplodingExplorer()
    captured: dict[str, object] = {}
    diagnostics: dict[str, object] = {}

    def planner(_engine, _goal, _grid, **kwargs):
        captured.update(kwargs)
        return []

    assert (
        policy._call_plan_in_model(
            planner,
            object(),
            lambda _grid: False,
            np.zeros((1, 1), dtype=np.int16),
            diagnostics=diagnostics,
            goal_energy_override=lambda _grid: 1.0,
        )
        == []
    )
    assert "dedup_mask" not in captured
    assert not any(key.startswith("planner_hud_dedup") for key in diagnostics)


def _selection_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for game_index in range(10):
        game = f"g{game_index:02d}"
        rows.append(
            {
                "pair_id": f"{game}__expert",
                "game": game,
                "engine_family": "EXPERT",
                "variant": "positive-control",
                "provenance_cohort": "stall_window",
                "source_path": f"source/{game}/expert.py",
                "source_sha256": f"expert-{game}",
                "real_level_up": bool(game_index % 2),
            }
        )
        for attempt_index in range(3):
            rows.append(
                {
                    "pair_id": f"{game}__attempt_{attempt_index}",
                    "game": game,
                    "engine_family": "CODEONLY",
                    "variant": f"seed-{attempt_index}",
                    "provenance_cohort": "stall_window",
                    "source_path": f"source/{game}/{attempt_index}.py",
                    "source_sha256": f"attempt-{game}-{attempt_index}",
                    "real_level_up": bool((game_index + attempt_index) % 2),
                }
            )
    return rows


def test_manifest_selection_is_outcome_blind_and_meets_frozen_budget() -> None:
    """SCENARIO-ARC-WMTE-7639-MANIFEST-AND-ARTIFACT freezes identities only."""
    rows = _selection_rows()
    flipped = copy.deepcopy(rows)
    for row in flipped:
        row["real_level_up"] = not row["real_level_up"]
        row["planner_engine_calls"] = 999_999
    manifest = exp.build_window_manifest(rows)
    replay = exp.build_window_manifest(flipped)
    assert manifest == replay
    assert manifest["selection_uses_arm_outcomes"] is False
    assert len(manifest["exposed_expert_controls"]) == 10
    assert len(manifest["additional_attempt_windows"]) == 20
    assert manifest["additional_game_count"] == 10
    assert manifest["breadth_sufficient"] is True
    selected = manifest["exposed_expert_controls"] + manifest["additional_attempt_windows"]
    assert all("real_level_up" not in row for row in selected)
    assert all("planner_engine_calls" not in row for row in selected)


def test_manifest_retains_small_eligible_census() -> None:
    """SCENARIO-ARC-WMTE-7639-MANIFEST-AND-ARTIFACT predeclares low breadth."""
    rows = _selection_rows()[:8]
    manifest = exp.build_window_manifest(rows)
    eligible_additional = [row for row in rows if row["engine_family"] != "EXPERT"]
    assert len(manifest["additional_attempt_windows"]) == len(eligible_additional)
    assert manifest["breadth_sufficient"] is False
    assert manifest["insufficient_breadth_reason"]


def test_only_off_and_guarded_hud_arms_are_frozen() -> None:
    """REQ-ARC-WMTE-7639 keeps novelty tie-breaking out of both arms."""
    assert [arm.name for arm in exp.MEASUREMENT_ARMS] == ["OFF", "HUD_DEDUP"]
    assert exp.MEASUREMENT_ARMS[0].environment == {}
    assert exp.MEASUREMENT_ARMS[1].environment == {"CARNOT_ARC_PLAN_HUD_DEDUP": "1"}
    assert all(
        "CARNOT_ARC_PLAN_GOAL_TIEBREAK" not in arm.environment for arm in exp.MEASUREMENT_ARMS
    )


def test_artifact_exposes_required_terminal_fields(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7639-MANIFEST-AND-ARTIFACT retains audit fields."""
    manifest = exp.build_window_manifest(_selection_rows())
    rows = [
        exp.synthetic_regression_row("terminal_duplicate", arm.name, passed=True)
        for arm in exp.MEASUREMENT_ARMS
    ]
    artifact = exp.build_artifact(
        rows=rows,
        manifest=manifest,
        preconditions_checked=[{"check": "fixture", "available": True}],
        duration_s=1.0,
        phase_spans=[],
        validation_receipts=[],
    )
    output = tmp_path / "artifact.json"
    exp.atomic_json(output, artifact)
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] in {"positive", "circular_positive", "null"}
    assert artifact["MODEL_SPECS"] == []
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["execution_host"]
    assert artifact["execution_device_uuid"] is None
    assert artifact["execution_owned_pid"] > 0
    assert artifact["production_defaults_changed"] is False
    assert artifact["planner_goal_guard_ready_score"] == 1
    assert artifact["arc_window_manifest_path"] == str(exp.WINDOW_MANIFEST_REL)
    assert artifact["solve_provenance"] == "development_proxy"
    assert artifact["reproducibility_checksum"]
    assert set(artifact["acceptance_gate_results"]) == {
        "validity",
        "readiness",
        "probability_benefit",
        "utility",
        "retention",
        "freshness",
    }


def test_live_regression_measurement_covers_planner_and_wrapper() -> None:
    """REQ-ARC-WMTE-7639 runs both reviewed failures through real entrypoints."""
    rows, evidence = exp.measure_regression_evidence()
    assert len(rows) == 4
    assert all(row["passed"] for row in rows)
    assert evidence["flags_off_parity"] is True
    assert evidence["telemetry_truthful"] is True
    assert evidence["terminal_duplicate_passed"] is True
    assert evidence["intermediate_counter_passed"] is True
    assert all(evidence["telemetry_cases"].values())


def test_reduction_detects_off_winner_regression() -> None:
    """REQ-ARC-WMTE-7639 retains OFF winners as an independent gate."""
    rows = [
        {
            "unit_id": "window",
            "arm": "OFF",
            "real_level_up": True,
            "plan_found": True,
            "censored": False,
        },
        {
            "unit_id": "window",
            "arm": "HUD_DEDUP",
            "real_level_up": False,
            "plan_found": False,
            "censored": True,
        },
    ]
    reduced = exp.independent_reduce(rows)
    assert reduced["off_winner_regressions"] == ["window"]
    assert reduced["censored_row_count"] == 1


def test_artifact_readers_recompute_and_reject_mutations(tmp_path) -> None:
    """SCENARIO-ARC-WMTE-7639-MANIFEST-AND-ARTIFACT uses independent readers."""
    manifest = exp.build_window_manifest(_selection_rows())
    rows = [
        exp.synthetic_regression_row("terminal_duplicate", arm.name, passed=True)
        for arm in exp.MEASUREMENT_ARMS
    ]
    artifact = exp.build_artifact(
        rows=rows,
        manifest=manifest,
        preconditions_checked=[{"check": "fixture", "passed": True}],
        duration_s=1.0,
        phase_spans=[],
        validation_receipts=[{"name": "focused", "passed": True, "exit_code": 0}],
        source_artifact_hashes={"authenticated_sources": {"fixture": "sha256:test"}},
        regression_evidence={"flags_off_parity": True, "telemetry_truthful": True},
    )
    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.validate_artifact(artifact) == []
    assert exp.cold_replay(path) == []
    assert exp.independent_replay(path) == []

    invalid = copy.deepcopy(artifact)
    invalid.pop("phase_spans")
    invalid["honest_verdict"] = "not_terminal"
    invalid["verdict_class"] = "unknown"
    invalid["MODEL_SPECS"] = ["model"]
    invalid["execution_venue"] = {"host": "free-text-is-not-a-venue"}
    invalid["rows"][0]["arm"] = "NOVELTY"
    invalid["reproducibility_checksum"] = "bad"
    errors = exp.validate_artifact(invalid)
    assert "phase_spans" in errors
    assert "honest_verdict_terminal_prefix" in errors
    assert "verdict_class" in errors
    assert "no_model_load_declaration" in errors
    assert "execution_identity" in errors
    assert "measurement_arms" in errors
    assert "reproducibility_checksum" in errors

    mismatched = copy.deepcopy(artifact)
    mismatched["independent_reduction"] = {}
    mismatched_path = tmp_path / "mismatched.json"
    mismatched_path.write_text(json.dumps(mismatched), encoding="utf-8")
    assert exp.independent_replay(mismatched_path) == ["independent_reduction"]


def test_command_manifests_parser_progress_and_json_conversion(tmp_path, capsys) -> None:
    """REQ-ARC-WMTE-7639 freezes bounded commands and serializable diagnostics."""
    manifest = exp.affected_validation_manifest()
    assert manifest["coverage_required_percent"] == 100
    assert {row.name for row in exp.build_e2e_commands(Path.cwd(), tmp_path)} == {
        "e2e_009",
        "e2e_011",
        "e2e_013",
        "no_induction_e3_cpu_smoke",
    }
    candidate = tmp_path / "candidate.json"
    assert {row.name for row in exp.build_terminal_commands(Path.cwd(), candidate)} == {
        "declared_entrypoint_validate",
        "fresh_process_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    args = exp.parse_args(["--date", exp.RUN_DATE, "--output", "result.json"])
    assert args.output == Path("result.json")
    exp.progress(time.monotonic(), "test", "boundary", units=1)
    assert "phase=test" in capsys.readouterr().out
    converted = exp._jsonable(
        {
            "array": np.array([1]),
            "scalar": np.int64(2),
            "path": Path("x"),
            "items": (np.int64(3),),
        }
    )
    assert converted == {"array": [1], "scalar": 2, "path": "x", "items": [3]}
    assert exp._all_passed([{"passed": True}]) is True
    assert exp._all_passed([]) is False


def test_arm_environment_restores_values_and_candidate_identity(monkeypatch) -> None:
    """REQ-ARC-WMTE-7639 isolates arms and serializes only pre-outcome identity."""
    monkeypatch.setenv("CARNOT_ARC_PLAN_HUD_DEDUP", "prior")
    monkeypatch.setenv("CARNOT_ARC_PLAN_GOAL_TIEBREAK", "prior-novelty")
    with exp.arm_environment(exp.MEASUREMENT_ARMS[0]):
        assert "CARNOT_ARC_PLAN_HUD_DEDUP" not in exp.os.environ
        assert "CARNOT_ARC_PLAN_GOAL_TIEBREAK" not in exp.os.environ
    assert exp.os.environ["CARNOT_ARC_PLAN_HUD_DEDUP"] == "prior"
    assert exp.os.environ["CARNOT_ARC_PLAN_GOAL_TIEBREAK"] == "prior-novelty"

    identity = exp._candidate_identity(
        SimpleNamespace(
            pair_id="g__codeonly__seed",
            game="g",
            engine_family="CODEONLY",
            variant="seed",
            source_path="source.py",
            source_sha256="sha256:test",
            source_status="cached",
            model_family="historical",
            think_mode="off",
            token_budget=None,
            is_control=False,
        )
    )
    assert identity["pair_id"] == "g__codeonly__seed"
    assert "real_level_up" not in identity

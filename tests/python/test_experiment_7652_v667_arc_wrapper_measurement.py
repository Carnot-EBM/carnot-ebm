"""REQ-REPORT-7652: complete census and honest wrapper reduction."""

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from carnot import experiment_7652_v667_arc_wrapper_measurement as exp
from carnot import experiment_10013_planner_dedup_tiebreak as planner


ROOT = Path(__file__).resolve().parents[2]


def test_census_is_complete_and_outcome_blind() -> None:
    """SCENARIO-REPORT-7652-CENSUS keeps all source identities once."""

    source = json.loads((ROOT / "results/experiment_10013_planner_dedup_tiebreak.json").read_text())
    manifest = exp.freeze_census(source["per_pair_arm_rows"])
    assert len(manifest["induced_engine_windows"]) == 40
    assert len(manifest["expert_controls"]) == 10
    assert len(manifest["identity_controls"]) == 10
    assert len({r["pair_id"] for r in manifest["induced_engine_windows"]}) == 40
    altered = deepcopy(source["per_pair_arm_rows"])
    for row in altered:
        row["real_level_up"] = not bool(row.get("real_level_up"))
        row["planner_engine_calls"] = -1
    assert exp.freeze_census(altered) == manifest


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("honest_verdict", "partial"),
        ("planner_goal_guard_ready_score", 0),
        ("flagged_adversarial", True),
    ],
)
def test_guard_names_exact_failed_operand(field: str, value: object) -> None:
    """SCENARIO-REPORT-7652-GUARD blocks changed guard authority."""

    guard = {
        "honest_verdict": "complete_null_arc_goal_guard_ready_no_hidden_game_benefit",
        "planner_goal_guard_ready_score": 1,
        "flagged_adversarial": False,
        "verdict_class": "null",
    }
    guard[field] = value
    failed = exp.check_goal_guard(
        guard, "results/experiment_7645_v667_arc_validation_requalification.json"
    )
    assert failed and failed["field"] == field
    assert failed["observed"] == value
    assert failed["expected"] != value
    assert failed["operator"] == "=="


def _row(unit: str, game: str, arm: str, success: bool, calls: int = 3) -> dict:
    return {
        "unit_id": unit,
        "game": game,
        "arm": arm,
        "engine_family": "CODEONLY",
        "real_level_up": success,
        "censored": False,
        "planner_engine_calls": calls,
        "real_actions_used": 1,
        "planner_wall_s": 0.1,
        "plan_length": 1,
        "planner_diagnostics": {
            "goal_evaluations": 4,
            "hud_dedup_states_merged": int(arm == "HUD_DEDUP"),
        },
        "mask": {
            "stage2_status": "admitted",
            "status": "applied" if arm == "HUD_DEDUP" else "disabled",
        },
        "state_rebuild": {
            "expected_sha256": "same",
            "observed_sha256": "same",
            "actions_replayed": 2,
        },
        "executed_actions": [{"action": 1, "data": None}],
        "raw_provenance": {"source_sha256": "sha256:abc"},
    }


def test_pair_reduction_counts_windows_once_and_controls_separately() -> None:
    """SCENARIO-REPORT-7652-REPLAY requires an added induced success with no loss."""

    rows = [_row("a", "g1", "OFF", False), _row("a", "g1", "HUD_DEDUP", True)]
    rows += [_row("b", "g2", "OFF", True), _row("b", "g2", "HUD_DEDUP", True)]
    control = _row("expert", "g1", "OFF", True)
    control["engine_family"] = "EXPERT"
    rows.append(control)
    result = exp.reduce_pairs(rows)
    assert result["induced_windows"] == 2
    assert result["induced_new_successes"] == 1
    assert result["induced_lost_successes"] == 0
    assert result["expert_control_rows"] == 1
    assert result["development_proxy_improvement"] is True
    assert result["ci95_game_cluster"][0] <= result["ci95_game_cluster"][1]


def test_cold_mutations_reject_mask_mismatch_and_goal_skip() -> None:
    """SCENARIO-REPORT-7652-REPLAY rejects unsafe wrapper telemetry."""

    rows = [_row("a", "g1", "OFF", False), _row("a", "g1", "HUD_DEDUP", True)]
    assert exp.validate_rows(rows) == []
    wrong_mask = deepcopy(rows)
    wrong_mask[1]["mask"]["planner_reason"] = "shape_mismatch"
    assert "mask_mismatch_applied" in exp.validate_rows(wrong_mask)
    skipped = deepcopy(rows)
    skipped[1]["planner_diagnostics"]["goal_evaluations"] = 0
    assert "goal_check_skipped" in exp.validate_rows(skipped)


def test_direct_goal_counter_preserves_predicate() -> None:
    """REQ-ARC-WMTE-7652 counts direct checks without changing their result."""

    counted, counter = planner.count_direct_goal_checks(lambda grid: grid == 7)
    assert counted(2) is False
    assert counted(7) is True
    assert counter["goal_checks"] == 2


def test_source_preconditions_and_validation_scope_are_real() -> None:
    """SCENARIO-REPORT-7652-GUARD authenticates existing inputs, not output."""

    checks, hashes = exp.collect_preconditions(ROOT)
    assert all(check["passed"] for check in checks)
    assert str(exp.RESULT) in hashes["planned_outputs_not_inputs"]
    assert str(exp.RESULT) not in hashes["producer_files"]
    scope = exp.validation_manifest()
    assert scope["tests"] == [str(exp.TEST)]
    assert scope["coverage_required_percent"] == 100
    assert exp._check("x", "upstream", Path("input"), "field", 1, 2)["passed"] is False


def test_census_rejects_conflict_and_ignores_unknown_family() -> None:
    """SCENARIO-REPORT-7652-CENSUS fails on ambiguous source identity."""

    row = {
        "pair_id": "u",
        "game": "g",
        "engine_family": "CODEONLY",
        "provenance_cohort": "stall_window",
    }
    changed = {**row, "source_sha256": "different"}
    with pytest.raises(ValueError, match="source_identity_conflict"):
        exp.freeze_census([row, changed])
    assert exp.freeze_census([{**row, "engine_family": "UNKNOWN"}])["induced_engine_windows"] == []


def test_reducer_rejects_unpaired_duplicate_and_censors_missing_engine() -> None:
    """SCENARIO-REPORT-7652-REPLAY keeps absent source in the denominator."""

    off = _row("u", "g", "OFF", False)
    with pytest.raises(ValueError, match="unpaired_window"):
        exp.reduce_pairs([off])
    with pytest.raises(ValueError, match="duplicate_arm"):
        exp.reduce_pairs([off, off])
    hud = _row("u", "g", "HUD_DEDUP", False)
    off["censored"] = True
    result = exp.reduce_pairs([off, hud, {**off, "engine_family": "OTHER"}])
    assert result["induced_windows"] == 1
    assert result["usable_induced_windows"] == 0
    assert result["ci95_game_cluster"] == [0.0, 0.0]


def test_terminal_readers_recompute_and_reject_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-7652-TERMINAL checks checksum and cold reduction."""

    rows = [_row("u", "g", "OFF", False), _row("u", "g", "HUD_DEDUP", True)]
    value = {
        "honest_verdict": "complete_null_fixture",
        "verdict_class": "null",
        "MODEL_SPECS": [],
        "model_invoked": False,
        "execution_venue": "host",
        "rows": rows,
        "independent_reduction": exp.reduce_pairs(rows),
        "reproducibility_checksum": "",
    }
    value["reproducibility_checksum"] = exp.canonical_hash(value)
    assert exp.validate_artifact(value) == []
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(value))
    assert exp.independent_replay(path) == []
    damaged = deepcopy(value)
    damaged["rows"][1]["real_level_up"] = False
    path.write_text(json.dumps(damaged))
    assert "independent_reduction" in exp.independent_replay(path)
    assert "reproducibility_checksum" in exp.validate_artifact(damaged)
    damaged.update(
        {
            "honest_verdict": "partial",
            "verdict_class": "bogus",
            "MODEL_SPECS": ["x"],
            "execution_venue": "gpu",
        }
    )
    assert {"terminal_verdict", "verdict_class", "no_model_load", "execution_venue"} <= set(
        exp.validate_artifact(damaged)
    )


def test_progress_and_span_emit_owned_monotonic_time(capsys: pytest.CaptureFixture[str]) -> None:
    """SCENARIO-REPORT-7652-TERMINAL prints an immediate flushed boundary."""

    start = time.monotonic()
    exp.progress(start, "test", "before", unit=1)
    assert "phase=test event=before" in capsys.readouterr().out
    spans: list[dict] = []
    exp._span(spans, "test", start, start, 1)
    assert spans[0]["completed_units"] == 1
    assert exp._gate(True, "source truth", {"calls": 1})["measured_operands"] == {"calls": 1}


def test_terminal_reader_rejects_duplicate_arm() -> None:
    """SCENARIO-REPORT-7652-REPLAY refuses multiplied sample size."""

    row = _row("u", "g", "OFF", False)
    value = {
        "rows": [row, row],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "execution_venue": "host",
        "honest_verdict": "complete_null",
        "verdict_class": "null",
        "reproducibility_checksum": "",
    }
    value["reproducibility_checksum"] = exp.canonical_hash(value)
    assert "duplicate_arm:u:OFF" in exp.validate_artifact(value)


def test_complete_measurement_is_separate_from_suite_failure() -> None:
    """SCENARIO-REPORT-7652-TERMINAL keeps completed telemetry despite disqualification."""

    rows = [_row("u", "g", "OFF", False), _row("u", "g", "HUD_DEDUP", False)]
    manifest = {
        "induced_engine_windows": [{"pair_id": "u"}],
        "expert_controls": [],
        "identity_controls": [],
    }
    artifact = exp.build_artifact(
        rows=rows,
        manifest=manifest,
        checks=[],
        hashes={"producer_files": {"x": "hash"}},
        receipts=[{"name": "full_python_suite", "passed": False, "exit_code": 2}],
        spans=[],
        duration=1.0,
    )
    assert artifact["wrapper_measurement_complete_score"] == 1
    assert artifact["verdict_class"] == "disqualified"

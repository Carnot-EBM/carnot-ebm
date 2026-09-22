"""Tests for REQ-ARC-7526 live supervisor eligibility evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from carnot import experiment_7526_v658_arc_eligibility as exp
from carnot.agentic import arc_competition_agent as agent
from carnot.agentic.arc_competition_agent import E3AgentPolicy
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    ARM_TOOL_LOOP_REINDUCTION,
    Redirect,
    TrajectorySnapshot,
    TrajectorySupervisor,
)


ROOT = Path(__file__).resolve().parents[2]


def _snapshot(**updates: object) -> TrajectorySnapshot:
    values: dict[str, object] = {
        "level": 1,
        "goal_bias_installed": False,
        "induced": False,
        "induction_attempts": 0,
        "new_transitions_since_induction": 0,
        "diversity_active": False,
    }
    values.update(updates)
    return TrajectorySnapshot(**values)  # type: ignore[arg-type]


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_shared_predicates_cover_curated_spent_and_ineligible_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-PREDICATES use the selector's table."""

    monkeypatch.delenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", raising=False)
    supervisor = TrajectorySupervisor(window=1, reinduction_evidence_floor=2)
    snapshot = _snapshot(
        goal_bias_installed=True,
        induced=True,
        induction_attempts=1,
        new_transitions_since_induction=2,
    )
    before = supervisor.arm_eligibility(snapshot)
    selected = supervisor.observe(snapshot)
    after = supervisor.arm_eligibility(snapshot)

    assert [row["arm"] for row in before] == [
        ARM_DROP_GOAL_BIAS,
        ARM_ALLOW_REINDUCTION,
        ARM_TOOL_LOOP_REINDUCTION,
        ARM_FORCE_DIVERSITY,
    ]
    assert before[0]["eligible"] is True
    assert before[2] == {
        "arm": ARM_TOOL_LOOP_REINDUCTION,
        "enabled": False,
        "eligible": False,
        "diagnosis": "",
    }
    assert selected is not None and selected.arm == ARM_DROP_GOAL_BIAS
    assert after[0]["eligible"] is False


def test_tool_arm_requires_enablement_spent_plain_arm_and_attempt_headroom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-PREDICATES keep the real tool predicate."""

    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "1")
    supervisor = TrajectorySupervisor(window=1, reinduction_attempt_cap=3)
    supervisor._arms_used.add(ARM_ALLOW_REINDUCTION)
    rows = supervisor.arm_eligibility(_snapshot(induction_attempts=2, diversity_active=True))

    assert next(row for row in rows if row["arm"] == ARM_TOOL_LOOP_REINDUCTION)["eligible"] is True
    selected = supervisor.observe(_snapshot(induction_attempts=2, diversity_active=True))
    assert selected is not None and selected.arm == ARM_TOOL_LOOP_REINDUCTION


def test_default_off_recorder_and_bounded_shadow_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-APPLICATION separate shadow selection."""

    monkeypatch.delenv(exp.RECORDER_ENV_FLAG, raising=False)
    assert exp.maybe_make_eligibility_recorder("g") is exp.NOOP_ELIGIBILITY_RECORDER

    path = tmp_path / "eligibility.jsonl"
    recorder = exp.EligibilityReceiptRecorder("g", path=path, max_rows=1)
    supervisor = TrajectorySupervisor(window=1)
    snapshot = _snapshot(goal_bias_installed=True)
    predicates = supervisor.arm_eligibility(snapshot)
    redirect = supervisor.observe(snapshot)
    recorder.record_selection(
        supervisor=supervisor,
        snapshot=snapshot,
        predicates=predicates,
        redirect=redirect,
        mode="shadow",
        applied=False,
    )
    recorder.record_observation_failure(mode="shadow", error="unreadable_level")

    (row,) = _rows(path)
    assert row["action_id"] == 1
    assert row["mode"] == "shadow"
    assert row["selected_arm"] == ARM_DROP_GOAL_BIAS
    assert row["applied"] is False
    assert row["old_state_hash"] is None and row["new_state_hash"] is None
    assert row["arm_rows"][0]["selected"] is True
    assert recorder.dropped_rows == 1


def test_policy_records_applied_mutation_noop_and_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-APPLICATION keep application factual."""

    path = tmp_path / "applied.jsonl"
    monkeypatch.setenv(exp.RECORDER_ENV_FLAG, "1")
    monkeypatch.setenv(exp.RECORDER_PATH_ENV, str(path))
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", "1")
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", "1")
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    policy = E3AgentPolicy("g", proposer=object(), target_levels=2, value_head=None)
    policy.explorer.set_goal_bias(lambda _frame: 0.0)
    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=1))

    first = _rows(path)[0]
    assert first["selected_arm"] == ARM_DROP_GOAL_BIAS
    assert first["applied"] is True
    assert isinstance(first["old_state_hash"], str)
    assert isinstance(first["new_state_hash"], str)
    assert first["old_state_hash"] != first["new_state_hash"]

    policy._trajectory_supervisor = SimpleNamespace(
        _actions_total=1,
        arm_eligibility=lambda _snapshot: [
            {
                "arm": ARM_FORCE_DIVERSITY,
                "enabled": True,
                "eligible": True,
                "diagnosis": "fixture",
            }
        ],
        observe=lambda _snapshot: Redirect(ARM_FORCE_DIVERSITY, 2, 1, "fixture"),
    )
    policy.explorer = None
    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=1))
    assert _rows(path)[1]["applied"] is False
    assert _rows(path)[1]["application_disposition"] == "no_state_change"

    policy._trajectory_supervisor.observe = lambda _snapshot: (_ for _ in ()).throw(
        RuntimeError("selection failed")
    )
    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=1))
    failed = _rows(path)[2]
    assert failed["observation_status"] == "failed"
    assert failed["arm_rows"][0]["eligible"] is None
    assert policy._trajectory_supervisor_errors == 1


def _parity_run(monkeypatch: pytest.MonkeyPatch, path: Path | None) -> dict[str, object]:
    if path is None:
        monkeypatch.delenv(exp.RECORDER_ENV_FLAG, raising=False)
        monkeypatch.delenv(exp.RECORDER_PATH_ENV, raising=False)
    else:
        monkeypatch.setenv(exp.RECORDER_ENV_FLAG, "1")
        monkeypatch.setenv(exp.RECORDER_PATH_ENV, str(path))
    monkeypatch.delenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR", raising=False)
    monkeypatch.setenv("CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW", "1")
    monkeypatch.setattr(agent, "_level_of", lambda frame: int(frame.levels_completed))
    random.seed(658_026)
    policy = E3AgentPolicy("g", proposer=object(), target_levels=2, value_head=None)
    policy.explorer.set_goal_bias(lambda _frame: 0.0)
    before = exp.policy_redirect_state(policy)
    policy._maybe_supervise_trajectory(SimpleNamespace(levels_completed=1))
    return {
        "state": exp.policy_redirect_state(policy),
        "before": before,
        "rng": random.getstate(),
        "model_calls": 0,
        "environment_calls": 0,
        "action": ("RESET", None),
    }


def test_recorder_on_off_policy_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-PARITY preserve all behavior surfaces."""

    disabled = _parity_run(monkeypatch, None)
    enabled = _parity_run(monkeypatch, tmp_path / "parity.jsonl")

    assert enabled == disabled


def test_frozen_panel_manifest_and_outer_timing_import() -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-PANEL freeze roster before outcomes."""

    manifest = exp.build_panel_manifest()

    assert len(manifest["games"]) == 6
    assert set(manifest["games"]).issubset(set(exp.E6_PANEL_GAMES))
    assert manifest["selection_seed"] == 658_026
    assert manifest["episode_seeds"] == [658_027, 658_028]
    assert manifest["action_cap"] == max(840, 2 * manifest["supervisor_window"] + 40)
    assert manifest["episode_cap_s"] == 180
    assert manifest["collection_cap_s"] == 3000
    assert exp.EXCLUSIVE_TIMING_HELPER.endswith("E6TimedObserver")


def test_terminal_fixture_is_schema_complete_and_null_ready() -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-TERMINAL separate readiness and benefit."""

    artifact = exp.build_artifact_for_test(ROOT)

    assert exp.validate_artifact(artifact, ROOT, require_terminal=True) == []
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert not any(artifact["invocation_counts"].values())
    assert artifact["eligibility_receipt_ready_score"] == 1
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["verifier_is_oracle"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"


def test_missing_external_precondition_builds_exact_blocked_artifact(tmp_path: Path) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-TERMINAL name unchanged absence."""

    blocked = exp.build_blocked_artifact(
        tmp_path,
        path="ops/arc_solve_registry.yaml",
        field="exists",
        expected=True,
        observed=False,
    )

    assert blocked["honest_verdict"] == "complete_blocked_external_prerequisite_absent"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"] == {
        "path": "ops/arc_solve_registry.yaml",
        "field": "exists",
        "expected": True,
        "observed": False,
    }


def test_replay_rejects_mutated_comparative_rows(tmp_path: Path, capsys: object) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-TERMINAL independently reduce rows."""

    artifact = exp.build_artifact_for_test(ROOT)
    changed = deepcopy(artifact)
    changed["rows"][0]["eligible"] = not changed["rows"][0]["eligible"]
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(changed), encoding="utf-8")

    assert exp.main(["--replay", str(candidate), "--reduce-only"]) == 1
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert "independent_reduction_mismatch" in output["validation_errors"]
    errors = exp.validate_artifact(changed, ROOT, require_terminal=False)
    assert "independent_reduction_mismatch" in errors
    assert "raw_reduction_checksum_mismatch" in errors


def test_validation_and_terminal_plans_are_exact(tmp_path: Path) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-TERMINAL freeze every required command."""

    plan = exp.build_validation_plan(ROOT, tmp_path / "validation")

    assert exp.validate_validation_plan(ROOT, plan) == []
    names = [row.name for row in plan]
    assert {"e2e_009", "e2e_010", "e2e_011", "private_arc_smoke"}.issubset(names)
    assert "full_python_suite" not in names
    assert exp.REPOSITORY_HEALTH_OBSERVATION["command_argv"] == [
        ".venv/bin/pytest",
        "tests/python",
        "-q",
    ]
    candidate = tmp_path / "candidate.json"
    terminal = exp.terminal_command_specs(ROOT, candidate)
    assert [row.name for row in terminal] == [
        "declared_entrypoint_cold_replay",
        "independent_cold_reducer",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    assert "--strict" in terminal[-1].argv


def test_receipt_reducer_preserves_known_unknown_selected_and_applied(tmp_path: Path) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-APPLICATION reduce raw facts exactly."""

    receipt = tmp_path / "receipt.jsonl"
    complete = {
        "mode": "applied",
        "arm_rows": [
            {
                "arm": exp.ARM_SELECTION_ORDER[0],
                "enabled": True,
                "eligible": True,
                "selected": True,
                "applied": True,
                "mode": "applied",
            }
        ],
    }
    failed = deepcopy(complete)
    failed["arm_rows"][0].update(eligible=None, selected=False, applied=False)
    ineligible = deepcopy(complete)
    ineligible["arm_rows"] = [
        {
            "arm": exp.ARM_SELECTION_ORDER[1],
            "enabled": True,
            "eligible": False,
            "selected": False,
            "applied": False,
            "mode": "applied",
        },
        {"arm": "not_curated", "eligible": True},
        "malformed",
    ]
    receipt.write_text("\n".join(map(json.dumps, (complete, failed, ineligible))), encoding="utf-8")
    episodes = [
        {
            "episode_id": "g:seed-1",
            "game": "g",
            "seed": 1,
            "disposition": "complete",
            "eligibility_path": receipt.relative_to(tmp_path).as_posix(),
            "eligibility_sha256": "sha256:fixture",
        }
    ]

    rows = exp.reduce_panel_receipts(tmp_path, episodes)
    first = rows[0]
    assert first["eligible_count"] == 1
    assert first["unknown_eligibility_count"] == 1
    assert first["selected_count"] == 1
    assert first["applied_count"] == 1
    assert first["eligible"] is None
    assert rows[1]["ineligible_count"] == 1
    reduced = exp.independently_reduce_rows(rows)
    assert reduced["explicit_eligibility_count"] == 2
    assert reduced["unknown_eligibility_count"] == 1


def test_parity_and_receipt_helpers_fail_closed() -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-PARITY require both parity receipts."""

    good = exp.parity_rows_from_receipts(
        [{"name": "focused_pytest", "passed": True}, {"name": "e2e_011", "passed": True}]
    )
    bad = exp.parity_rows_from_receipts([{"name": "focused_pytest", "passed": True}])

    assert all(row["passed"] for row in good)
    assert not any(row["passed"] for row in bad)
    assert exp._receipts_pass([{"name": "a", "passed": True}], ["a"])
    assert not exp._receipts_pass([], ["a"])
    span = exp._phase("fixture", 1.0, 0.0, 2)
    assert span["phase"] == "fixture" and span["completed_units"] == 2


def test_defensive_readers_factories_and_gate_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-ARC-7526 fails closed for malformed local evidence and recorder setup."""

    assert exp.utc_now().endswith("Z")
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    assert exp._load_object(missing) == {}
    assert exp._load_object(malformed) == {}
    assert exp._load_object(array) == {}
    assert exp._read_jsonl(missing) == []
    malformed.write_text('{"ok": 1}\nnot-json\n[]\n', encoding="utf-8")
    assert exp._read_jsonl(malformed) == [{"ok": 1}]
    with pytest.raises(ValueError, match="unsupported_gate_operator"):
        exp._gate("bad", "validity", 1, 1, "!=", "fixture")

    monkeypatch.setenv(exp.RECORDER_ENV_FLAG, "1")
    monkeypatch.delenv(exp.RECORDER_PATH_ENV, raising=False)
    assert exp.maybe_make_eligibility_recorder("g") is exp.NOOP_ELIGIBILITY_RECORDER
    exp.NOOP_ELIGIBILITY_RECORDER.record_observation_failure(error="ignored")


def test_validation_reader_rejects_each_contract_mutation() -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-TERMINAL reject field drift."""

    base = exp.build_artifact_for_test(ROOT)
    mutations = {
        "missing_field:panel_manifest": lambda value: value.pop("panel_manifest"),
        "schema_mismatch": lambda value: value.__setitem__("schema", "wrong"),
        "experiment_id_mismatch": lambda value: value.__setitem__("experiment_id", "wrong"),
        "milestone_mismatch": lambda value: value.__setitem__("milestone", "wrong"),
        "model_specs_nonempty": lambda value: value.__setitem__("MODEL_SPECS", ["model"]),
        "model_invoked_not_false": lambda value: value.__setitem__("model_invoked", True),
        "substrate_class_mismatch": lambda value: value.__setitem__(
            "inference_substrate_class", "aggregation"
        ),
        "substrate_mismatch": lambda value: value.__setitem__("inference_substrate", "wrong"),
        "current_invocation_counts_nonzero": lambda value: value["invocation_counts"].update(
            generation_calls_attempted=1
        ),
        "readiness_not_bare_numeric": lambda value: value.__setitem__(
            "eligibility_receipt_ready_score", None
        ),
        "rows_not_sequence": lambda value: value.__setitem__("rows", {}),
        "field_principles_incomplete": lambda value: value["field_principles"].pop("schema"),
        "terminal_prefix_missing": lambda value: value.__setitem__("honest_verdict", "null"),
        "verdict_class_invalid": lambda value: value.__setitem__("verdict_class", "unknown"),
    }
    for expected, mutate in mutations.items():
        changed = deepcopy(base)
        mutate(changed)
        assert expected in exp.validate_artifact(changed, ROOT, require_terminal=True)


def test_validation_plan_drift_is_rejected(tmp_path: Path) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-TERMINAL reject command substitution."""

    plan = exp.build_validation_plan(ROOT, tmp_path)
    changed = [row for row in plan if row.name != "e2e_011"]
    errors = exp.validate_validation_plan(ROOT, changed)

    assert any(error.startswith("command_count:e2e_011") for error in errors)


def test_cli_requires_date_and_valid_replay_passes(tmp_path: Path, capsys: object) -> None:
    """REQ-ARC-7526 and SCENARIO-ARC-7526-TERMINAL expose only bounded CLI roles."""

    with pytest.raises(SystemExit, match="--date is required"):
        exp.main([])
    artifact = exp.build_artifact_for_test(ROOT)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--replay", str(candidate)]) == 0
    assert json.loads(capsys.readouterr().out)["matches_declared"] is True  # type: ignore[attr-defined]


def test_cli_date_dispatches_declared_producer(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-ARC-7526 routes the declared date to the reusable producer."""

    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(exp, "run_experiment", lambda root, date: calls.append((root, date)))

    assert exp.main(["--date", "20260922"]) == 0
    assert calls == [(exp.REPO_ROOT, "20260922")]


def test_terminal_classes_cover_invalid_and_supported_boundaries() -> None:
    """REQ-ARC-7526 keeps validity, readiness, support, and benefit independent."""

    base = exp.build_artifact_for_test(ROOT)
    common = {
        "root": ROOT,
        "episode_rows": base["episode_rows"],
        "preconditions": base["preconditions_checked"],
        "sources": base["source_artifact_hashes"],
        "phase_spans": [],
        "started_at_utc": base["started_at_utc"],
        "ended_at_utc": base["ended_at_utc"],
        "duration_s": 1.0,
    }
    invalid = exp.build_artifact(
        **common,
        rows=base["rows"],
        parity_rows=base["parity_rows"],
        validation_receipts=[{"name": "failed", "required": True, "passed": False}],
    )
    positive_rows = []
    for index in range(10):
        row = deepcopy(base["rows"][0])
        row.update(
            episode_id=f"fixture:{index}",
            eligible=True,
            selected_count=1,
            applied_count=1,
        )
        positive_rows.append(row)
    positive = exp.build_artifact(
        **common,
        rows=positive_rows,
        parity_rows=base["parity_rows"],
        validation_receipts=[{"name": "passed", "required": True, "passed": True}],
    )

    assert invalid["verdict_class"] == "disqualified"
    assert positive["verdict_class"] == "positive"

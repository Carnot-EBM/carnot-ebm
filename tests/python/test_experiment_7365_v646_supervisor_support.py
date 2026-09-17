"""Tests for REQ-ARC-WMTE-7365 and SCENARIO-ARC-WMTE-7365-*.

The tests use small receipt and registry fixtures. The E2E entrypoint later reads
the real durable ledger and registry.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7365_v646_supervisor_support as exp
from carnot.agentic.arc_supervisor_refinement import empty_ledger
from carnot.agentic.arc_trajectory_supervisor import (
    ARM_ALLOW_REINDUCTION,
    ARM_DROP_GOAL_BIAS,
    ARM_FORCE_DIVERSITY,
    ARM_ORDER,
    ARM_TOOL_LOOP_REINDUCTION,
    TrajectorySnapshot,
    TrajectorySupervisor,
)
from carnot.experiment_7365_v646_supervisor_support import (
    CANDIDATE_ORDERINGS,
    EXPERIMENT_CONFIG_SCHEMA,
    MIN_DEVELOPMENT_GAMES,
    MIN_SUPPORTED_PER_ARM,
    REQUIRED_EVALUATION_GAMES,
    ZERO_CURRENT_INVOCATIONS,
    OrderedTrajectorySupervisor,
    audit_ledger,
    build_artifact,
    build_blocked_artifact,
    check_upstream_contract,
    config_hash,
    freeze_game_split,
    install_experiment_supervisor,
    replay_trajectory,
    reproducibility_checksum,
    validate_artifact,
)


def _entry(game: str, receipt_id: str, arms: list[str], helped: bool = True) -> dict:
    return {
        "source": f"{game}.json",
        "game": game,
        "seed": int(receipt_id.rsplit("-", 1)[-1]),
        "harness_arm": "eval:e3:budget20000",
        "window": 120,
        "mode": "applied",
        "actions_observed": 500,
        "stagnations_unredirected": 0,
        "levels": 1 if helped else 0,
        "actions": 500,
        "arms_enabled": list(ARM_ORDER),
        "unredirected_windows": [],
        "unredirected_windows_dropped": 0,
        "redirects": [
            {
                "arm": arm,
                "action_index": 120 * (index + 1),
                "level": 0,
                "stretch_level": 0,
                "resolved_by_levelup": helped,
                "actions_to_levelup": 30 if helped else None,
                "co_credited_count": len(arms) if helped else None,
            }
            for index, arm in enumerate(arms)
        ],
        "receipt_id": receipt_id,
    }


def _registry() -> dict:
    return {
        "games": [
            {"game": "d1", "mechanic_class": "dev-a", "levels_reproduced": 1},
            {"game": "d2", "mechanic_class": "dev-b", "levels_reproduced": 1},
            {"game": "d3", "mechanic_class": "dev-c", "levels_reproduced": 1},
            {"game": "e1", "mechanic_class": "eval-a", "levels_reproduced": 2},
            {"game": "e2", "mechanic_class": "eval-b", "levels_reproduced": 2},
            {"game": "e3", "mechanic_class": "eval-c", "levels_reproduced": 2},
            {"game": "e4", "mechanic_class": "eval-d", "levels_reproduced": 2},
            {"game": "bad", "mechanic_class": "dev-a", "levels_reproduced": 8},
            {"game": "zero", "mechanic_class": "eval-z", "levels_reproduced": 0},
        ]
    }


def test_upstream_contract_checks_exact_yaml_gates() -> None:
    """REQ-ARC-WMTE-7365: dependent work starts only after exact producer gates."""

    producer = {
        "experiment_id": "exp7358-validation-contract",
        "milestone": "2026.09.646",
        "status": "complete_validation_contract_null_science",
        "validation_contract_ready_score": 1,
        "verdict_class": "null",
        "flagged_adversarial": False,
    }
    rows = check_upstream_contract(producer, available=True, excluded=False)
    assert all(row["passed"] for row in rows)
    assert {row["artifact_field"] for row in rows} >= {
        "validation_contract_ready_score",
        "verdict_class",
        "flagged_adversarial",
    }
    broken = dict(producer, verdict_class="partial")
    assert not all(
        row["passed"] for row in check_upstream_contract(broken, available=True, excluded=True)
    )
    missing = check_upstream_contract({}, available=False, excluded=False)
    assert missing == [
        {
            "check": "validation_contract_path",
            "upstream": "results/experiment_7358_v646_validation_contract.json",
            "artifact_field": "path",
            "expected": "readable_nonempty_json",
            "observed": "missing",
            "passed": False,
        }
    ]


def test_changed_action_censors_the_entire_suffix() -> None:
    """SCENARIO-ARC-WMTE-7365-PREFIX: no post-change outcome is counterfactual data."""

    entry = _entry(
        "d1",
        "run-1",
        [ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY],
    )
    baseline = replay_trajectory(entry, CANDIDATE_ORDERINGS[0], "development")
    assert all(row["supported"] for row in baseline)
    changed = replay_trajectory(entry, CANDIDATE_ORDERINGS[1], "development")
    assert changed[0]["supported"] is True
    assert [row["censoring_reason"] for row in changed[1:]] == [
        "candidate_changed_actual_action",
        "after_candidate_changed_action",
    ]
    assert all(row["observed_outcome"] is None for row in changed[1:])
    assert all(row["unsupported_counterfactual"] for row in changed[1:])


def test_replay_rejects_missing_outcome_and_preserves_credit() -> None:
    """SCENARIO-ARC-WMTE-7365-SUPPORT: missing outcomes fail closed; co-credit stays visible."""

    entry = _entry("d1", "run-1", [ARM_DROP_GOAL_BIAS])
    entry["redirects"][0].pop("resolved_by_levelup")
    (row,) = replay_trajectory(entry, CANDIDATE_ORDERINGS[0], "development")
    assert row["supported"] is False
    assert row["censoring_reason"] == "missing_observed_outcome"
    assert row["co_credited_count"] == 1
    assert row["observed_outcome"] is None


def test_game_split_is_frozen_reachable_and_family_disjoint() -> None:
    """REQ-ARC-WMTE-7365: four held games share no game or family with development."""

    split = freeze_game_split(_registry(), ["d3", "d1", "d2"])
    assert split["development_game_ids"] == ["d1", "d2", "d3"]
    assert split["evaluation_game_ids"] == ["e1", "e2", "e3", "e4"]
    assert split["development_family_ids"] == ["dev-a", "dev-b", "dev-c"]
    assert split["evaluation_family_ids"] == ["eval-a", "eval-b", "eval-c", "eval-d"]
    assert split["disjoint_games"] is True
    assert split["disjoint_families"] is True
    assert split["evaluation_reachable"] is True


def test_audit_groups_trajectories_and_keeps_controls() -> None:
    """SCENARIO-ARC-WMTE-7365-SUPPORT: redirects cannot inflate trajectory/game units."""

    ledger = empty_ledger()
    ledger["entries"] = {
        "r1": _entry("d1", "run-1", [ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION]),
        "r2": _entry("d2", "run-2", [ARM_DROP_GOAL_BIAS]),
        "r3": _entry("d3", "run-3", [ARM_DROP_GOAL_BIAS]),
    }
    ledger["controls"] = {"shadow-1": {"game": "d1", "would_have_redirects": []}}
    result = audit_ledger(ledger, _registry())
    assert result["receipt_counts"] == {
        "applied": 3,
        "shadow": 1,
        "error": 0,
        "duplicate": 0,
    }
    assert result["independent_trajectory_count"] == 3
    assert result["development_game_count"] == MIN_DEVELOPMENT_GAMES
    assert result["evaluation_game_count"] == REQUIRED_EVALUATION_GAMES
    assert result["supervisor_trial_ready_score"] == 0
    assert result["frozen_ordering"] is None
    assert result["unsupported_counterfactual_count"] > 0
    assert any(row.get("kind") == "shadow_control" for row in result["support_rows"])
    assert len(result["candidate_rows"]) == len(CANDIDATE_ORDERINGS)


def test_leave_one_game_out_floor_can_be_met_without_redirect_pooling() -> None:
    """REQ-ARC-WMTE-7365: every held-game fold independently keeps ten points per arm."""

    ledger = empty_ledger()
    arms = [ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY]
    for game_index, game in enumerate(("d1", "d2", "d3")):
        for run_index in range(MIN_SUPPORTED_PER_ARM):
            key = f"{game}-{run_index}"
            ledger["entries"][key] = _entry(game, f"run-{game_index * 100 + run_index}", arms)
    result = audit_ledger(ledger, _registry(), candidate_orderings=(CANDIDATE_ORDERINGS[0],))
    assert result["supervisor_trial_ready_score"] == 1
    assert result["frozen_ordering"] == list(CANDIDATE_ORDERINGS[0])
    assert all(row["passed"] for row in result["leave_one_game_out_rows"])
    assert result["frozen_supervisor_trial_manifest"]["config_hash"].startswith("sha256:")


class _Policy:
    pass


def _valid_config(order: tuple[str, ...]) -> dict:
    payload = {
        "schema": EXPERIMENT_CONFIG_SCHEMA,
        "enabled": True,
        "arm_order": list(order),
    }
    return {**payload, "config_hash": config_hash(payload)}


def test_experiment_route_requires_valid_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-ARC-WMTE-7365-ROUTING: absent or bad config cannot mutate the E3 seam."""

    policy = _Policy()
    assert install_experiment_supervisor(policy, None) is False
    assert not hasattr(policy, "_trajectory_supervisor")
    invalid = _valid_config(CANDIDATE_ORDERINGS[1])
    invalid["config_hash"] = "sha256:wrong"
    assert install_experiment_supervisor(policy, invalid) is False
    assert not hasattr(policy, "_trajectory_supervisor")

    monkeypatch.setenv("CARNOT_ARC_SUPERVISOR_TOOL_ARM", "1")
    valid = _valid_config(CANDIDATE_ORDERINGS[1])
    assert install_experiment_supervisor(policy, valid) is True
    assert isinstance(policy._trajectory_supervisor, OrderedTrajectorySupervisor)
    assert policy._trajectory_supervisor_applies is True
    assert tuple(policy._trajectory_supervisor.arm_order) == CANDIDATE_ORDERINGS[1]

    snapshot = TrajectorySnapshot(
        level=0,
        goal_bias_installed=True,
        induced=True,
        induction_attempts=1,
        new_transitions_since_induction=300,
        diversity_active=False,
    )
    assert policy._trajectory_supervisor._first_eligible_arm(snapshot)[0] == ARM_DROP_GOAL_BIAS
    policy._trajectory_supervisor._arms_used.add(ARM_DROP_GOAL_BIAS)
    assert policy._trajectory_supervisor._first_eligible_arm(snapshot)[0] == ARM_FORCE_DIVERSITY
    policy._trajectory_supervisor._arms_used.update({ARM_FORCE_DIVERSITY, ARM_ALLOW_REINDUCTION})
    assert (
        policy._trajectory_supervisor._first_eligible_arm(snapshot)[0] == ARM_TOOL_LOOP_REINDUCTION
    )


def test_ordered_supervisor_rejects_non_curated_order() -> None:
    """REQ-ARC-WMTE-7365: the wrapper cannot add, remove, or duplicate an arm."""

    with pytest.raises(ValueError, match="permutation"):
        OrderedTrajectorySupervisor(("invented",))
    assert isinstance(TrajectorySupervisor(), TrajectorySupervisor)


def test_artifact_null_and_mutations_validate() -> None:
    """SCENARIO-ARC-WMTE-7365-NULL: a complete support shortfall is null, not partial."""

    ledger = empty_ledger()
    ledger["entries"]["r1"] = _entry("d1", "run-1", [ARM_DROP_GOAL_BIAS])
    audit = audit_ledger(ledger, _registry())
    artifact = build_artifact(
        audit=audit,
        preconditions_checked=[{"passed": True}],
        source_artifact_hashes={"x": "sha256:x"},
        validation_receipts=[
            {"name": name, "passed": True}
            for name in (
                "worktree_imports",
                "focused_pytest",
                "changed_module_coverage",
                "changed_module_coverage_report",
                "ruff_check",
                "ruff_format",
                "changed_module_mypy",
                "scoped_spec_coverage",
                "full_python_suite",
                "independent_reducer",
                "adversarial_verify",
                "verdict_row_consistency_strict",
            )
        ],
        duration_s=1.25,
        phase_spans=[],
        started_at_utc="2026-09-17T00:00:00+00:00",
        completed_at_utc="2026-09-17T00:00:01+00:00",
    )
    assert artifact["status"] == "complete_null_insufficient_supported_outcomes"
    assert artifact["verdict_class"] == "null"
    assert artifact["supervisor_trial_ready_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["invocation_counts"]["current"] == ZERO_CURRENT_INVOCATIONS
    assert validate_artifact(artifact) == []
    assert artifact["reproducibility_checksum"] == reproducibility_checksum(artifact)

    forged = deepcopy(artifact)
    forged["unsupported_counterfactual_count"] += 1
    assert "reproducibility_checksum" in validate_artifact(forged)
    promoted = deepcopy(artifact)
    promoted["supervisor_trial_ready_score"] = 1
    promoted["reproducibility_checksum"] = reproducibility_checksum(promoted)
    assert "readiness" in validate_artifact(promoted)


def test_blocked_artifact_names_exact_failed_gate() -> None:
    """REQ-ARC-WMTE-7365: external missing input is blocked with exact gate evidence."""

    failed = {
        "check": "validation_contract_path",
        "upstream": "results/experiment_7358_v646_validation_contract.json",
        "artifact_field": "path",
        "expected": "readable_nonempty_json",
        "observed": "missing",
        "passed": False,
    }
    artifact = build_blocked_artifact(
        [failed], {}, duration_s=0.1, started_at_utc="a", completed_at_utc="b"
    )
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["first_failure"] == failed
    assert validate_artifact(artifact, require_validation=False) == []


def test_artifact_json_round_trip_is_plain_data(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7365: the terminal record reloads without custom object state."""

    split = freeze_game_split(_registry(), ["d1", "d2", "d3"])
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split, sort_keys=True), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == split


def _passing_receipts() -> list[dict]:
    names = (
        *exp.validation_scope.REQUIRED_CHECK_NAMES,
        "full_python_suite",
        *exp.TERMINAL_CHECK_NAMES,
    )
    return [{"name": name, "passed": True} for name in names]


def test_small_io_and_progress_helpers(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """REQ-ARC-WMTE-7365: boundaries flush, JSON fails closed, and registry type is strict."""

    assert "T" in exp.utc_now()
    exp.progress(0.0, "unit", "done", count=1)
    assert "phase=unit" in capsys.readouterr().out

    missing = tmp_path / "missing.json"
    assert exp._load_json_object(missing) == {}
    missing.write_text("[1]", encoding="utf-8")
    assert exp._load_json_object(missing) == {}
    missing.write_text("{", encoding="utf-8")
    assert exp._load_json_object(missing) == {}
    missing.write_text('{"x": 1}', encoding="utf-8")
    assert exp._load_json_object(missing) == {"x": 1}

    registry = tmp_path / "registry.yaml"
    registry.write_text("games: []\n", encoding="utf-8")
    assert exp._load_registry(registry) == {"games": []}
    registry.write_text("- not-a-map\n", encoding="utf-8")
    with pytest.raises(exp.SupervisorSupportError, match="registry_not_mapping"):
        exp._load_registry(registry)


def test_real_preconditions_authenticate_named_inputs() -> None:
    """REQ-ARC-WMTE-7365: the real gate is checked before dependent work."""

    checks, hashes, historical = exp.collect_preconditions(exp.REPO_ROOT)
    assert all(row["passed"] for row in checks)
    assert hashes[exp.VALIDATION_CONTRACT_PATH.as_posix()].startswith("sha256:")
    assert len(historical) == 3
    assert all(row["historical_only"] for row in historical)


def test_invalid_orderings_and_exhausted_order_fail_closed() -> None:
    """SCENARIO-ARC-WMTE-7365-ROUTING: invalid candidates and exhausted arms do not route."""

    with pytest.raises(ValueError, match="candidate_order"):
        replay_trajectory(_entry("d1", "run-1", []), ("bad",), "development")
    with pytest.raises(ValueError, match="at most three"):
        audit_ledger(
            empty_ledger(), _registry(), candidate_orderings=(*CANDIDATE_ORDERINGS, ARM_ORDER)
        )
    ledger = empty_ledger()
    ledger["entries"]["not-a-map"] = "bad"
    assert audit_ledger(ledger, _registry())["independent_trajectory_count"] == 0

    supervisor = OrderedTrajectorySupervisor(ARM_ORDER)
    supervisor._arms_used.update(ARM_ORDER)
    snapshot = TrajectorySnapshot(0, False, False, 0, 0, True)
    assert supervisor._first_eligible_arm(snapshot) == (None, "")
    tool_first = OrderedTrajectorySupervisor(
        (
            ARM_TOOL_LOOP_REINDUCTION,
            ARM_DROP_GOAL_BIAS,
            ARM_ALLOW_REINDUCTION,
            ARM_FORCE_DIVERSITY,
        )
    )
    eligible_drop = TrajectorySnapshot(0, True, False, 0, 0, True)
    assert tool_first._first_eligible_arm(eligible_drop)[0] == ARM_DROP_GOAL_BIAS


def test_ready_and_disqualified_artifact_branches() -> None:
    """REQ-ARC-WMTE-7365: readiness and failed validation stay separate terminal states."""

    ledger = empty_ledger()
    arms = [ARM_DROP_GOAL_BIAS, ARM_ALLOW_REINDUCTION, ARM_FORCE_DIVERSITY]
    for game_index, game in enumerate(("d1", "d2", "d3")):
        for run_index in range(MIN_SUPPORTED_PER_ARM):
            key = f"{game}-{run_index}"
            ledger["entries"][key] = _entry(game, f"run-{game_index * 100 + run_index}", arms)
    audit = audit_ledger(ledger, _registry(), candidate_orderings=(ARM_ORDER,))
    ready = build_artifact(
        audit=audit,
        preconditions_checked=[{"passed": True}],
        source_artifact_hashes={},
        validation_receipts=_passing_receipts(),
        duration_s=1.0,
        phase_spans=[],
        started_at_utc="a",
        completed_at_utc="b",
    )
    assert ready["supervisor_trial_ready_score"] == 1
    assert ready["status"] == "complete_supervisor_trial_manifest_ready_no_policy_benefit"
    assert validate_artifact(ready) == []

    failed = build_artifact(
        audit=audit,
        preconditions_checked=[{"passed": True}],
        source_artifact_hashes={},
        validation_receipts=[],
        duration_s=1.0,
        phase_spans=[],
        started_at_utc="a",
        completed_at_utc="b",
        flagged_adversarial=True,
    )
    assert failed["verdict_class"] == "disqualified"
    assert failed["supervisor_trial_ready_score"] == 0


def test_validator_reports_each_independent_contract_error() -> None:
    """REQ-ARC-WMTE-7365: cold validation detects field mutations, not only hash drift."""

    blocked = build_blocked_artifact(
        [{"passed": False}], {}, duration_s=0.0, started_at_utc="a", completed_at_utc="b"
    )
    assert validate_artifact(None) == ["artifact_object"]
    mutations = {
        "identity": {"schema": "bad"},
        "model_contract": {"model_invoked": True},
        "invocation_counts": {"invocation_counts": {}},
        "substrate_class": {"inference_substrate_class": "bad"},
        "execution_venue": {"execution_venue": "bad"},
        "verdict_class": {"verdict_class": "bad"},
        "promotion": {"promotion_score": 1},
        "production_defaults": {"production_defaults_changed": True},
        "support_rows": {"support_rows": None},
        "manifest": {
            "verdict_class": "null",
            "frozen_supervisor_trial_manifest": None,
        },
        "validation_receipts": {
            "verdict_class": "null",
            "validation_receipts": [],
        },
    }
    for expected, changes in mutations.items():
        value = deepcopy(blocked)
        value.update(changes)
        value["reproducibility_checksum"] = reproducibility_checksum(value)
        assert expected in validate_artifact(value)

    invalid_ready = deepcopy(blocked)
    invalid_ready.update(
        {
            "verdict_class": "null",
            "supervisor_trial_ready_score": 1,
            "frozen_supervisor_trial_manifest": {
                "arm_order": ["bad"],
                "config_hash": "bad",
                "evaluation_game_ids": [],
            },
        }
    )
    invalid_ready["reproducibility_checksum"] = reproducibility_checksum(invalid_ready)
    assert "readiness" in validate_artifact(invalid_ready, require_validation=False)
    false_ready = deepcopy(blocked)
    false_ready.update(
        {
            "verdict_class": "null",
            "supervisor_trial_ready_score": 0,
            "frozen_supervisor_trial_manifest": {"arm_order": list(ARM_ORDER)},
        }
    )
    false_ready["reproducibility_checksum"] = reproducibility_checksum(false_ready)
    assert "readiness" in validate_artifact(false_ready, require_validation=False)


def test_command_wrappers_keep_scoped_plans_and_exact_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7365: validation wrappers delegate to the shipped streaming runner."""

    command = exp.validation_scope.CommandSpec("x", ("python", "-V"), "scope")
    monkeypatch.setattr(exp.validation_contract, "build_command_plan", lambda *args: [command])
    monkeypatch.setattr(exp.validation_contract, "validate_command_plan", lambda *args: [])
    monkeypatch.setattr(
        exp.validation_contract,
        "run_categorized_commands",
        lambda *args, **kwargs: [{"name": "x", "passed": True}],
    )
    assert exp.run_affected_validation(tmp_path, tmp_path) == [{"name": "x", "passed": True}]

    monkeypatch.setattr(
        exp.validation_contract,
        "validate_command_plan",
        lambda *args: ["bad"],
    )
    with pytest.raises(exp.SupervisorSupportError, match="invalid_scoped_command_plan"):
        exp.run_affected_validation(tmp_path, tmp_path)

    captured: list[list[str]] = []

    def fake_run(_root: Path, commands: list, **_kwargs: object) -> list[dict]:
        captured.append([item.name for item in commands])
        return [{"name": item.name, "passed": True} for item in commands]

    monkeypatch.setattr(exp.validation_scope, "run_commands", fake_run)
    assert exp.run_full_python_suite(tmp_path, tmp_path)[0]["name"] == "full_python_suite"
    terminal = exp.run_terminal_validation(tmp_path, tmp_path / "candidate.json", tmp_path)
    assert [row["name"] for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert captured == [["full_python_suite"], list(exp.TERMINAL_CHECK_NAMES)]


def test_prior_full_suite_receipt_is_reused_only_once(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-7365: broad repository health runs once across metadata correction."""

    path = tmp_path / "prior.json"
    assert exp.prior_full_suite_receipt(path) == []
    path.write_text('{"validation_receipts": {}}', encoding="utf-8")
    assert exp.prior_full_suite_receipt(path) == []
    path.write_text('{"validation_receipts": []}', encoding="utf-8")
    assert exp.prior_full_suite_receipt(path) == []
    path.write_text(
        json.dumps(
            {
                "validation_receipts": [
                    {"name": "full_python_suite", "passed": False, "exit_code": -15}
                ]
            }
        ),
        encoding="utf-8",
    )
    (row,) = exp.prior_full_suite_receipt(path)
    assert row["reused_as_repository_health_diagnostic"] is True
    assert row["current_required_validation"] is False
    path.write_text(json.dumps({"validation_receipts": [row, row]}), encoding="utf-8")
    assert exp.prior_full_suite_receipt(path) == []


def test_run_experiment_success_and_blocked_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ARC-WMTE-7365: the orchestrator publishes terminal null or blocked records."""

    preconditions = [{"passed": True}]
    monkeypatch.setattr(
        exp,
        "collect_preconditions",
        lambda _root: (preconditions, {"x": "sha256:x"}, []),
    )
    ledger = empty_ledger()
    ledger["entries"]["r1"] = _entry("d1", "run-1", [ARM_DROP_GOAL_BIAS])
    monkeypatch.setattr(exp, "load_ledger", lambda _path: ledger)
    monkeypatch.setattr(exp, "_load_registry", lambda _path: _registry())
    affected = [
        {"name": name, "passed": True} for name in exp.validation_scope.REQUIRED_CHECK_NAMES
    ]
    monkeypatch.setattr(exp, "run_affected_validation", lambda *_args: affected)
    monkeypatch.setattr(
        exp,
        "run_full_python_suite",
        lambda *_args: [{"name": "full_python_suite", "passed": True}],
    )
    monkeypatch.setattr(
        exp,
        "run_terminal_validation",
        lambda *_args: [{"name": name, "passed": True} for name in exp.TERMINAL_CHECK_NAMES],
    )
    monkeypatch.setattr(exp, "RAW_DIR", tmp_path / "raw")
    output = tmp_path / "result.json"
    result = exp.run_experiment(exp.REPO_ROOT, exp.RUN_DATE, output_path=output)
    assert exp.RUN_DATE == "20260917"
    assert result["status"] == "complete_null_insufficient_supported_outcomes"
    assert (
        json.loads(output.read_text())["reproducibility_checksum"]
        == result["reproducibility_checksum"]
    )
    assert [row["phase"] for row in result["phase_spans"]] == [
        "load",
        "generation",
        "evaluation",
        "validation",
        "write",
    ]
    with pytest.raises(SystemExit, match="--date"):
        exp.run_experiment(exp.REPO_ROOT, "wrong", output_path=output)

    monkeypatch.setattr(exp, "validate_artifact", lambda *_args, **_kwargs: ["forged"])
    with pytest.raises(exp.SupervisorSupportError, match="terminal_artifact_invalid"):
        exp.run_experiment(exp.REPO_ROOT, exp.RUN_DATE, output_path=output)
    monkeypatch.undo()

    monkeypatch.setattr(exp, "RAW_DIR", tmp_path / "raw-blocked")
    monkeypatch.setattr(
        exp,
        "collect_preconditions",
        lambda _root: ([{"passed": False}], {}, []),
    )
    blocked_path = tmp_path / "blocked.json"
    blocked = exp.run_experiment(exp.REPO_ROOT, exp.RUN_DATE, output_path=blocked_path)
    assert blocked["verdict_class"] == "blocked"


def test_main_validate_and_run_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """REQ-ARC-WMTE-7365: the thin CLI supports cold validation and execution."""

    artifact = build_blocked_artifact(
        [{"passed": False}], {}, duration_s=0.0, started_at_utc="a", completed_at_utc="b"
    )
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--date", exp.RUN_DATE, "--validate", str(path)]) == 0
    assert '"errors": []' in capsys.readouterr().out

    called: list[tuple[Path, str, Path]] = []
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, *, output_path: called.append((root, date, output_path)) or {},
    )
    output = tmp_path / "out.json"
    assert exp.main(["--date", exp.RUN_DATE, "--output", str(output)]) == 0
    assert called == [(exp.REPO_ROOT, exp.RUN_DATE, output)]

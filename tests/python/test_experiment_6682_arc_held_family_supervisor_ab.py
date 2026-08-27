"""Tests for REQ-ARC-WMTE-6682 and its held-family supervisor A/B."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_6682_arc_held_family_supervisor_ab as exp


def _observation(*, level: int = 0, state: str = "NOT_FINISHED") -> dict:
    return {
        "game_id": "held-a-runtime-id",
        "guid": "receipt-only-guid",
        "action_input": {"id": 1, "data": {"game_id": "held-a-runtime-id"}},
        "frame": [[[0, 1], [2, 3]]],
        "state": state,
        "levels_completed": level,
        "win_levels": 3,
        "full_reset": False,
        "available_actions": [1, 6],
    }


def _raw_row(
    arm: str,
    *,
    index: int = 0,
    proposed: dict | None = None,
    applied: dict | None = None,
    before_level: int = 0,
    after_level: int = 0,
    state: str = "NOT_FINISHED",
    matched_unit_id: str = "held-a:0",
) -> dict:
    proposed = proposed or {"kind": 1, "data": None}
    applied = applied or proposed
    suffix = f"{matched_unit_id}:{arm}:{index}"
    return {
        "matched_unit_id": matched_unit_id,
        "arm": arm,
        "family": "held-a",
        "episode_seed": 6682001,
        "episode_id": f"episode:{suffix}",
        "action_index": index,
        "proposal_id": f"proposal:{suffix}",
        "application_id": f"application:{suffix}",
        "environment_step_id": f"step:{suffix}",
        "outcome_id": f"outcome:{suffix}",
        "lineage": {
            "proposal_id": f"proposal:{suffix}",
            "application_id": f"application:{suffix}",
            "environment_step_id": f"step:{suffix}",
            "outcome_id": f"outcome:{suffix}",
        },
        "proposed_action": proposed,
        "policy_selected_action": applied,
        "applied_action": applied,
        "supervisor_decision": {
            "fired": arm == "on" and applied != proposed,
            "arm": "reset_after_stagnant_repeat" if arm == "on" and applied != proposed else None,
            "state": "stagnant_repeat" if arm == "on" and applied != proposed else "observing",
            "inputs": list(exp.SUPERVISOR_INPUT_FIELDS),
        },
        "observation_before": _observation(level=before_level),
        "observation_after": _observation(level=after_level, state=state),
        "reward": {
            "present": False,
            "value": None,
            "source": "arc_agi.FrameDataRaw.step_return_schema",
            "synthetic": False,
        },
        "termination": {
            "terminated": state in {"WIN", "GAME_OVER"},
            "truncated": False,
            "state": state,
            "source": "arc_agi.FrameDataRaw.state",
        },
        "action_cost": 1,
        "fully_joined": True,
        "live_return": True,
        "outcome_status": "returned",
        "decision_sealed_before_outcome": True,
        "supervisor_rule_scope": "game_agnostic_frozen_fsm",
        "evidence_source": "canonical_live_environment_return",
        "action_budget": 4,
        "stopping_rules_hash": _manifest()["stopping_rules_hash"],
        "policy_hash": "sha256:policy",
        "supervisor_hash": "sha256:supervisor",
        "initial_policy_state_hash": "sha256:initial-policy",
        "initial_observation_hash": "sha256:initial-observation",
        "episode_status": "complete",
    }


def _rows(*, on_after: int = 0, off_after: int = 0) -> list[dict]:
    return [
        exp.enrich_action_row(_raw_row("off", after_level=off_after)),
        exp.enrich_action_row(
            _raw_row(
                "on",
                proposed={"kind": 1, "data": None},
                applied={"kind": "RESET", "data": None},
                after_level=on_after,
            )
        ),
    ]


def _manifest() -> dict:
    return exp.freeze_run_manifest(
        held_families=("held-a",),
        episode_seeds=(6682001,),
        action_budget=4,
        frozen_fsm={
            "schema": "carnot.arc.trace_fsm.v1",
            "fsm_hash": "sha256:supervisor",
        },
        policy_hash="sha256:policy",
        arm_order_seed=6682991,
    )


def _passing_preconditions() -> dict:
    return {
        "passed": True,
        "failed_checks": [],
        "checks": [
            {
                "check": "exp6681.arc_outcome_transport_ready",
                "expected": True,
                "observed": True,
                "passed": True,
            },
            {
                "check": "exp6681.eligible_redirect_outcome_rows",
                "expected": ">=30",
                "observed": 30,
                "passed": True,
            },
        ],
        "registry_precheck": {
            "path": "ops/arc_solve_registry.yaml",
            "registry_sha256": "sha256:registry",
            "duplicate_solve_exclusion_result": "pass_no_game_or_level_target",
            "declared_target_solve": False,
        },
        "hashes": {
            "canonical_policy": "sha256:policy",
            "frozen_supervisor_source": "sha256:supervisor-source",
            "frozen_fsm": "sha256:supervisor",
            "active_roadmap": "sha256:roadmap",
            "conductor": "sha256:conductor",
            "solve_registry": "sha256:registry",
        },
        "sdk": {"package": "arc-agi", "version": "0.9.8", "installed": True},
        "access": {
            "network_reachable": True,
            "anonymous_access_available": True,
            "held_families_present": True,
        },
        "resources": {
            "cpu_count": 24,
            "ram_total_bytes": 128 * 1024**3,
            "disk_free_bytes": 1024**3,
        },
        "run_date": "20260827",
    }


def test_scenario_6682_preconditions_fail_closed_before_live_runner(tmp_path: Path):
    """SCENARIO-ARC-WMTE-6682-PRECONDITIONS-AND-FREEZE."""

    preconditions = _passing_preconditions()
    preconditions["passed"] = False
    preconditions["failed_checks"] = [
        {"check": "live_access", "expected": True, "observed": False, "passed": False}
    ]
    called = False

    def forbidden_runner(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("live runner called after a failed gate")

    output = tmp_path / "blocked.json"
    artifact = exp.build_artifact(
        result_path=output,
        preconditions=preconditions,
        run_manifest=_manifest(),
        live_runner=forbidden_runner,
        duration_s=0.02,
    )

    assert called is False
    assert artifact["status"] == "blocked_preconditions"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["arc_supervisor_ab_ready"] is False
    assert artifact["gate_check_summary"]["failed_check"] == "live_access"
    assert not list(tmp_path.glob(".*.tmp"))
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert exp.validate_artifact(artifact) == []


def test_scenario_6682_freeze_is_deterministic_and_game_blind():
    """SCENARIO-ARC-WMTE-6682-PRECONDITIONS-AND-FREEZE."""

    first = _manifest()
    second = _manifest()

    assert first == second
    assert first["expected_matched_unit_count"] == 1
    assert first["arms"] == ["off", "on"]
    assert first["missing_row_policy"] == "fail_closed_no_imputation"
    assert first["online_environment_seed_effective"] is False
    assert first["supervisor_input_fields"] == list(exp.SUPERVISOR_INPUT_FIELDS)
    assert "family" not in first["supervisor_input_fields"]
    assert first["manifest_hash"] == exp.hash_without_field(first, "manifest_hash")


def test_scenario_6682_exact_validity_and_transition_utility():
    """SCENARIO-ARC-WMTE-6682-EXACT-ROW-AND-UTILITY."""

    before = _observation()
    assert exp.action_validity({"kind": "RESET", "data": None}, before)["valid"] is True
    assert exp.action_validity({"kind": 1, "data": None}, before)["valid"] is True
    assert exp.action_validity({"kind": 2, "data": None}, before)["forbidden"] is True
    assert exp.action_validity({"kind": 6, "data": {"x": 1, "y": 0}}, before)["valid"] is True
    assert exp.action_validity({"kind": 6, "data": {"x": 2, "y": 0}}, before)["valid"] is False
    assert exp.action_validity({"kind": 6, "data": {"x": 1}}, before)["valid"] is False

    level_up = exp.enrich_action_row(_raw_row("off", after_level=1))
    game_over = exp.enrich_action_row(_raw_row("off", state="GAME_OVER"))
    rewarded = _raw_row("off")
    rewarded["reward"] = {
        "present": True,
        "value": 2.5,
        "source": "environment_step_return[1]",
        "synthetic": False,
    }

    assert level_up["transition_utility"] == 1.0
    assert level_up["transition_utility_source"] == "exact_public_level_and_state_transition"
    assert game_over["transition_utility"] == -1.0
    assert exp.enrich_action_row(rewarded)["transition_utility"] == 2.5
    with pytest.raises(ValueError, match="numeric"):
        invalid_reward = copy.deepcopy(rewarded)
        invalid_reward["reward"]["value"] = "not-numeric"
        exp.enrich_action_row(invalid_reward)


def test_scenario_6682_pairs_interventions_and_family_intervals():
    """SCENARIO-ARC-WMTE-6682-MATCHED-PAIRS-AND-INTERVENTIONS."""

    analysis = exp.recompute_analysis(_rows(on_after=0, off_after=1), _manifest())

    assert analysis["ready"] is True
    assert len(analysis["paired_episode_rows"]) == 1
    pair = analysis["paired_episode_rows"][0]
    assert pair["transition_utility_delta"] == -1.0
    assert pair["valid_action_block_delta"] == 1
    assert pair["forbidden_action_delta"] == 0
    assert pair["actions_spent_delta"] == 0
    assert len(analysis["false_intervention_rows"]) == 1
    assert analysis["false_intervention_rows"][0]["benefit_observed"] is False
    assert analysis["transition_utility_summary"]["losses"] == 1
    assert analysis["transition_utility_summary"]["interval_95"]["sample_size"] == 1
    assert analysis["held_family_rows"][0]["family"] == "held-a"


def test_scenario_6682_forbidden_benefit_and_no_headroom_recompute():
    """REQ-ARC-WMTE-6682 keeps rejection and exact benefit separate."""

    invalid = {"kind": 2, "data": None}
    off = exp.enrich_action_row(_raw_row("off", proposed=invalid, applied=invalid))
    on = exp.enrich_action_row(_raw_row("on"))
    analysis = exp.recompute_analysis([off, on], _manifest())

    assert analysis["forbidden_action_summary"]["off_count"] == 1
    assert analysis["forbidden_action_summary"]["on_count"] == 0
    assert analysis["forbidden_action_summary"]["benefit_delta"] == 1.0
    assert analysis["forbidden_action_summary"]["wins"] == 1
    assert analysis["forbidden_action_summary"]["no_headroom_rows"] == 0

    no_headroom = exp.recompute_analysis(_rows(), _manifest())
    assert no_headroom["forbidden_action_summary"]["no_headroom_rows"] == 1


def test_scenario_6682_attacks_reject_every_contamination():
    """SCENARIO-ARC-WMTE-6682-ATTACKS-FAIL-CLOSED."""

    rows = _rows()
    assert exp.validate_unit_rows(rows, _manifest()) == []
    attacks = exp.run_attack_matrix(rows, _manifest())

    assert {row["attack_id"] for row in attacks} == set(exp.ATTACK_IDS)
    assert all(row["passed"] is True and row["rejected"] is True for row in attacks)

    duplicate = rows + [copy.deepcopy(rows[0])]
    assert "duplicate_action" in exp.validate_unit_rows(duplicate, _manifest())
    missing = copy.deepcopy(rows)
    missing[0]["outcome_id"] = ""
    assert "missing_outcome" in exp.validate_unit_rows(missing, _manifest())
    unequal = copy.deepcopy(rows)
    unequal[1]["action_budget"] = 5
    assert "unequal_budget" in exp.validate_unit_rows(unequal, _manifest())


def test_scenario_6682_ready_artifact_verdict_no_solve_and_checksum(tmp_path: Path):
    """SCENARIO-ARC-WMTE-6682-VERDICT-AND-NO-SOLVE and ATOMIC-RECOMPUTATION."""

    output = tmp_path / "ready.json"
    artifact = exp.build_artifact(
        result_path=output,
        preconditions=_passing_preconditions(),
        run_manifest=_manifest(),
        per_unit_rows=_rows(),
        live_metadata={"scorecard": {"submitted_to_leaderboard": False}},
        duration_s=0.02,
    )

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_arc_supervisor_ab_null"
    assert artifact["verdict_class"] == "null"
    assert artifact["arc_supervisor_ab_ready"] is True
    assert artifact["solve_claim_scope"] == "none"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["aggregate_row_recomputation"]["all_headlines_match"] is True
    assert exp.validate_artifact(artifact) == []

    changed = copy.deepcopy(artifact)
    changed["action_efficiency_summary"]["on_action_cost"] += 1
    assert "aggregate recomputation mismatch" in exp.validate_artifact(changed)
    assert "reproducibility checksum mismatch" in exp.validate_artifact(changed)


def test_scenario_6682_positive_requires_exact_benefit_not_rejection():
    """SCENARIO-ARC-WMTE-6682-VERDICT-AND-NO-SOLVE."""

    rejective = exp.classify_ready_verdict(
        transition_summary={"delta": 0.0, "interval_95": {"lower": 0.0}},
        forbidden_summary={"benefit_delta": 0.0, "interval_95": {"lower": 0.0}},
        valid_action_block_count=4,
    )
    beneficial = exp.classify_ready_verdict(
        transition_summary={"delta": 1.0, "interval_95": {"lower": 1.0}},
        forbidden_summary={"benefit_delta": 0.0, "interval_95": {"lower": 0.0}},
        valid_action_block_count=4,
    )

    assert rejective == ("complete_arc_supervisor_ab_null", "null")
    assert beneficial == ("complete_arc_supervisor_ab_positive", "circular_positive")


def test_scenario_6682_normalized_match_ignores_receipt_ids_not_state():
    """REQ-ARC-WMTE-6682 does not mistake a requested seed for a matched start."""

    left = _observation()
    right = copy.deepcopy(left)
    right["guid"] = "different"
    right["game_id"] = "different-runtime-id"
    right["action_input"] = {"id": 0}
    assert exp.normalized_initial_observation(left) == exp.normalized_initial_observation(right)

    right["frame"][0][0][0] = 9
    assert exp.normalized_initial_observation(left) != exp.normalized_initial_observation(right)


def test_scenario_6682_cli_validate_and_missing_output(tmp_path: Path, capsys):
    """SCENARIO-ARC-WMTE-6682-ATOMIC-RECOMPUTATION."""

    output = tmp_path / "artifact.json"
    artifact = exp.build_artifact(
        result_path=output,
        preconditions=_passing_preconditions(),
        run_manifest=_manifest(),
        per_unit_rows=_rows(),
        duration_s=0.02,
    )
    assert artifact["arc_supervisor_ab_ready"] is True
    assert exp.main(["--validate", "--result-path", str(output)]) == 0
    assert capsys.readouterr().out.strip() == "OK"
    assert exp.main(["--validate", "--result-path", str(tmp_path / "missing.json")]) == 1


def test_req_6682_deterministic_edge_contracts(tmp_path: Path):
    """REQ-ARC-WMTE-6682 covers public checker and statistic edge cases."""

    assert exp.sha256_file(tmp_path / "missing") == "missing"
    assert exp._load_json(tmp_path / "missing") == {}
    assert exp._action_kind({"kind": "ACTION6"}) == 6
    assert exp._action_kind({"kind": True}) is None
    observation = _observation()
    observation["available_actions"] = [{"value": "ACTION6"}, "junk", 0, 6]
    assert exp.action_validity({"kind": "ACTION6", "data": {"x": 1, "y": 1}}, observation)["valid"]
    assert exp.action_validity({"kind": 6, "data": {"x": 0, "y": 0}}, {})["valid"] is False
    assert (
        exp.action_validity({"kind": 6, "data": {"x": 0, "y": 0}}, {"available_actions": [6]})[
            "valid"
        ]
        is False
    )
    assert (
        exp.action_validity(
            {"kind": 6, "data": {"x": 0, "y": 0}}, {"frame": [[]], "available_actions": [6]}
        )["valid"]
        is False
    )
    interval = exp.paired_interval([1, -1], seed=7, resamples=20)
    assert interval["sample_size"] == 2
    assert interval["lower"] <= interval["mean"] <= interval["upper"]

    malformed = _raw_row("off")
    malformed["observation_before"]["levels_completed"] = True
    with pytest.raises(ValueError, match="level fields"):
        exp.enrich_action_row(malformed)
    malformed["observation_before"]["levels_completed"] = None
    with pytest.raises(ValueError, match="level fields"):
        exp.enrich_action_row(malformed)


def test_scenario_6682_row_validator_localizes_all_contract_failures():
    """SCENARIO-ARC-WMTE-6682-ATTACKS-FAIL-CLOSED localizes row failures."""

    rows = _rows()
    rows[0].update(
        {
            "family": "wrong",
            "stopping_rules_hash": "wrong",
            "policy_hash": "wrong",
            "supervisor_hash": "wrong",
            "fully_joined": False,
            "episode_status": "environment_error",
            "initial_observation_hash": "different",
        }
    )
    unknown = copy.deepcopy(rows[0])
    unknown.update(
        {
            "matched_unit_id": "unknown:0",
            "arm": "bogus",
            "action_index": 9,
            "proposal_id": "unknown-proposal",
            "application_id": "unknown-application",
            "environment_step_id": "unknown-step",
            "outcome_id": "unknown-outcome",
        }
    )
    issues = exp.validate_unit_rows([*rows, unknown], _manifest())

    assert {
        "unmatched_episode",
        "unequal_budget",
        "game_specific_rule",
        "missing_outcome",
    }.issubset(issues)


def test_req_6682_build_failure_paths_and_cli_dispatch(tmp_path: Path, monkeypatch, capsys):
    """REQ-ARC-WMTE-6682 fails closed for runner, attack, and partial paths."""

    manifest = _manifest()

    def successful_runner(_manifest, _preconditions):
        return _rows(), {"scorecard": {"submitted_to_leaderboard": False}}

    ready = exp.build_artifact(
        repo_root=tmp_path,
        result_path=Path("ready.json"),
        preconditions=_passing_preconditions(),
        run_manifest=manifest,
        live_runner=successful_runner,
        duration_s=0.01,
    )
    assert ready["arc_supervisor_ab_ready"] is True

    def failing_runner(_manifest, _preconditions):
        raise RuntimeError("live failed")

    partial = exp.build_artifact(
        repo_root=tmp_path,
        result_path=tmp_path / "partial.json",
        preconditions=_passing_preconditions(),
        run_manifest=manifest,
        live_runner=failing_runner,
        duration_s=0.01,
    )
    assert partial["verdict_class"] == "partial"
    assert "live failed" in partial["canonical_path_receipts"]["live_metadata"]["error"]

    bad_rows = _rows()
    bad_rows[0]["reward"] = {"present": True, "value": "not-numeric"}
    partial = exp.build_artifact(
        repo_root=tmp_path,
        result_path=tmp_path / "bad-row.json",
        preconditions=_passing_preconditions(),
        run_manifest=manifest,
        per_unit_rows=bad_rows,
        duration_s=0.01,
    )
    assert partial["verdict_class"] == "partial"
    assert "row_enrichment_error" in partial["per_unit_rows"][0]

    with monkeypatch.context() as protected_patch:
        protected_patch.setattr(
            exp,
            "_protected_receipt",
            lambda _root, _before: {
                "rows": [],
                "all_protected_files_unchanged": False,
            },
        )
        protected_failure = exp.build_artifact(
            repo_root=tmp_path,
            result_path=tmp_path / "protected-failure.json",
            preconditions=_passing_preconditions(),
            run_manifest=manifest,
            per_unit_rows=_rows(),
            duration_s=0.01,
            write=False,
        )
    assert protected_failure["gate_check_summary"]["failed_check"] == "protected_file_change"

    monkeypatch.setattr(
        exp,
        "run_attack_matrix",
        lambda _rows, _manifest: [
            {"attack_id": "label_leakage", "passed": False, "rejected": False}
        ],
    )
    disqualified = exp.build_artifact(
        repo_root=tmp_path,
        result_path=tmp_path / "disqualified.json",
        preconditions=_passing_preconditions(),
        run_manifest=manifest,
        per_unit_rows=_rows(),
        duration_s=0.01,
        write=False,
    )
    assert disqualified["verdict_class"] == "disqualified"

    monkeypatch.setattr(exp, "REPO_ROOT", tmp_path)
    called = []
    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: called.append(kwargs) or {})
    assert exp.main(["--result-path", "cli.json"]) == 0
    assert called[0]["result_path"] == tmp_path / "cli.json"

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    assert exp.main(["--validate", "--result-path", "invalid.json"]) == 1
    assert "required fields mismatch" in capsys.readouterr().out


def test_req_6682_failed_mandatory_verification_keeps_readiness_false(tmp_path: Path):
    """REQ-ARC-WMTE-6682 records a failed mandatory suite without claiming readiness."""

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        result_path=tmp_path / "verification-failed.json",
        preconditions=_passing_preconditions(),
        run_manifest=_manifest(),
        per_unit_rows=_rows(),
        test_results={
            exp.FULL_TEST_COMMAND: {
                "exit_code": 3,
                "summary": "pytest xdist internal error after global failures",
            }
        },
        duration_s=0.01,
    )

    assert artifact["arc_supervisor_ab_ready"] is False
    assert artifact["verdict_class"] == "partial"
    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["gate_check_summary"]["failed_check"] == "verification_failure"
    full = next(row for row in artifact["tests_run"] if row["command"] == exp.FULL_TEST_COMMAND)
    assert full["exit_code"] == 3
    assert exp.validate_artifact(artifact) == []


def test_req_6682_artifact_validator_rejects_each_claim_axis(tmp_path: Path):
    """REQ-ARC-WMTE-6682 validates schema, claims, hashes, attacks, and readiness."""

    artifact = exp.build_artifact(
        repo_root=tmp_path,
        result_path=tmp_path / "base.json",
        preconditions=_passing_preconditions(),
        run_manifest=_manifest(),
        per_unit_rows=_rows(),
        duration_s=0.01,
    )
    broken = copy.deepcopy(artifact)
    broken["extra"] = True
    broken.update(
        {
            "status": "running",
            "verdict_class": "invalid",
            "solve_claim_scope": "level",
            "inference_substrate": "archive",
            "verifier_is_oracle": False,
            "arc_supervisor_ab_ready": False,
        }
    )
    broken["frozen_run_manifest"]["manifest_hash"] = "wrong"
    broken["canonical_path_receipts"]["live_metadata"]["attack_rows"] = []
    issues = exp.validate_artifact(broken)
    assert {
        "required fields mismatch",
        "status lacks terminal prefix",
        "verdict class invalid",
        "solve scope mismatch",
        "inference substrate mismatch",
        "oracle flag mismatch",
        "frozen manifest hash mismatch",
        "attack rows mismatch",
        "readiness mismatch",
    }.issubset(issues)

    wrong_ready_verdict = copy.deepcopy(artifact)
    wrong_ready_verdict["status"] = "complete_arc_supervisor_ab_positive"
    wrong_ready_verdict["verdict_class"] = "circular_positive"
    wrong_ready_verdict["reproducibility_checksum"] = exp.reproducibility_checksum(
        wrong_ready_verdict
    )
    assert "ready verdict mismatch" in exp.validate_artifact(wrong_ready_verdict)

    blocked = copy.deepcopy(artifact)
    blocked["preconditions_checked"]["passed"] = False
    blocked["verdict_class"] = "partial"
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    assert "blocked precondition verdict mismatch" in exp.validate_artifact(blocked)

    default = exp._default_manifest(
        Path(exp.__file__).resolve().parents[2], _passing_preconditions()
    )
    assert default["expected_matched_unit_count"] == len(exp.HELD_FAMILIES) * len(exp.EPISODE_SEEDS)

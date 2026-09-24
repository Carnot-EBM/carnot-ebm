"""REQ-ARC-7625 supervisor outcome-ledger transfer tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from carnot import experiment_7625_v665_arc_supervisor_transfer as exp


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _redirect(
    arm: str,
    *,
    resolved: bool,
    action_index: int,
    actions_to_levelup: int | None,
) -> dict:
    return {
        "arm": arm,
        "action_index": action_index,
        "level": 0,
        "stretch_level": 0,
        "diagnosis": "general stagnation",
        "resolved_by_levelup": resolved,
        "actions_to_levelup": actions_to_levelup,
        "co_credited_count": 1 if resolved else None,
    }


def _episode(
    game: str,
    seed: int,
    *,
    mode: str = "shadow",
    redirects: list[dict] | None = None,
    termination: str = "action_limit",
) -> dict:
    redirects = redirects or []
    base = {
        "schema": "fixture.episode.v1",
        "episode_id": f"{game}:{seed}",
        "game": game,
        "seed": seed,
        "policy": "E3AgentPolicy",
        "adapter_withheld": True,
        "stored_solutions_withheld": True,
        "game_source_read": False,
        "hidden_state_read": False,
        "offline_ground_truth_bfs": False,
        "steps": [{"action_index": 0, "action": {"kind": 1}}],
        "termination": {"reason": termination},
    }
    outcomes: dict[str, dict[str, int]] = {}
    for row in redirects:
        arm = row["arm"]
        outcome = outcomes.setdefault(arm, {"fired": 0, "helped": 0})
        outcome["fired"] += 1
        outcome["helped"] += int(row["resolved_by_levelup"])
    if mode == "applied":
        supervisor = {
            "enabled": True,
            "mode": "applied",
            "window": 120,
            "arms_enabled": ["drop_goal_bias", "allow_reinduction"],
            "arms_used": sorted(outcomes),
            "redirects": redirects,
            "arm_outcomes": outcomes,
            "stagnations_unredirected": 0,
            "unredirected_windows": [],
            "unredirected_windows_dropped": 0,
        }
    else:
        would_have = []
        for row in redirects:
            changed = deepcopy(row)
            changed["levelup_followed_without_redirect"] = changed.pop("resolved_by_levelup")
            changed["actions_to_levelup_without_redirect"] = changed.pop("actions_to_levelup")
            would_have.append(changed)
        supervisor = {
            "enabled": False,
            "mode": "shadow",
            "window": 120,
            "arms_enabled": ["drop_goal_bias", "allow_reinduction"],
            "arms_used": sorted(outcomes),
            "would_have_redirects": would_have,
            "would_have_arm_outcomes": outcomes,
            "stagnations_unredirected": 1,
            "unredirected_windows": [],
            "unredirected_windows_dropped": 0,
        }
    base["trajectory_supervisor"] = supervisor
    return base


def _write_producer(root: Path, name: str, episodes: list[dict]) -> exp.ProducerSpec:
    raw_dir = root / "raw" / name
    raw_dir.mkdir(parents=True)
    hashes = {}
    for index, episode in enumerate(episodes):
        path = raw_dir / f"{index:02d}.json"
        path.write_text(json.dumps(episode), encoding="utf-8")
        hashes[path.relative_to(root).as_posix()] = _sha(path)
    result = root / f"{name}.json"
    result.write_text(
        json.dumps(
            {
                "honest_verdict": "complete_null_fixture",
                "verdict_class": "null",
                "flagged_adversarial": False,
                "source_artifact_hashes": {"authenticated_sources": hashes},
            }
        ),
        encoding="utf-8",
    )
    return exp.ProducerSpec(name, result.relative_to(root), raw_dir.relative_to(root))


# REQ-ARC-7625; SCENARIO-ARC-7625-AUTH.
def test_authenticated_receipts_are_deduplicated_by_logical_episode(tmp_path: Path) -> None:
    first = _episode("su15", 1)
    replay = _episode("su15", 2)
    producer = _write_producer(tmp_path, "v664", [first, replay])

    selected = exp.select_authenticated_receipts(tmp_path, [producer])

    assert selected["schema_errors"] == []
    assert selected["observed_receipts"] == 2
    assert selected["deduplicated_receipts"] == 1
    assert selected["duplicate_receipts"] == 1
    assert selected["receipts"][0]["payload"]["game"] == "su15"
    assert len(selected["receipts"][0]["duplicate_sources"]) == 1
    assert all(value.startswith("sha256:") for value in selected["source_hashes"].values())


# REQ-ARC-7625; SCENARIO-ARC-7625-ZERO.
def test_shadow_outcomes_are_not_counted_as_actual_firings() -> None:
    redirect = _redirect("drop_goal_bias", resolved=True, action_index=120, actions_to_levelup=8)
    episode = _episode("su15", 1, redirects=[redirect])
    reduction = exp.reduce_receipts(
        [{"payload": episode, "path": "receipt.json", "sha256": "sha256:raw"}]
    )

    assert reduction["schema_errors"] == []
    assert reduction["proposed_redirect_count"] == 1
    assert reduction["would_have_redirect_count"] == 1
    assert reduction["applied_redirect_count"] == 0
    assert reduction["actual_firing_count"] == 0
    assert reduction["helped_count"] == 0
    assert reduction["supervisor_outcome_ledger_ready_score"] == 1
    assert reduction["per_game_results"][0]["stagnations_unredirected"] == 1


# REQ-ARC-7625; SCENARIO-ARC-7625-JOIN.
def test_applied_redirects_join_outcomes_and_censoring() -> None:
    helped = _episode(
        "su15",
        1,
        mode="applied",
        redirects=[
            _redirect("drop_goal_bias", resolved=True, action_index=120, actions_to_levelup=7)
        ],
        termination="policy_done",
    )
    unresolved = _episode(
        "sp80",
        2,
        mode="applied",
        redirects=[
            _redirect("drop_goal_bias", resolved=False, action_index=120, actions_to_levelup=None)
        ],
        termination="action_limit",
    )
    reduction = exp.reduce_receipts(
        [
            {"payload": helped, "path": "helped.json", "sha256": "sha256:helped"},
            {
                "payload": unresolved,
                "path": "unresolved.json",
                "sha256": "sha256:unresolved",
            },
        ]
    )

    assert reduction["schema_errors"] == []
    assert reduction["actual_firing_count"] == 2
    assert reduction["helped_count"] == 1
    assert reduction["censored_firing_count"] == 1
    helped_row = next(row for row in reduction["actual_redirects"] if row["game"] == "su15")
    censored_row = next(row for row in reduction["actual_redirects"] if row["game"] == "sp80")
    assert helped_row["resolved_by_levelup"] is True
    assert helped_row["actions_to_levelup"] == 7
    assert helped_row["censoring"]["censored"] is False
    assert censored_row["direction"] == "censored"
    assert censored_row["denominator"] == 0


# REQ-ARC-7625; SCENARIO-ARC-7625-JOIN.
def test_missing_or_inconsistent_outcome_schema_fails_closed() -> None:
    missing = _episode("su15", 1)
    del missing["trajectory_supervisor"]["would_have_arm_outcomes"]
    missing_result = exp.reduce_receipts(
        [{"payload": missing, "path": "missing.json", "sha256": "sha256:missing"}]
    )
    assert missing_result["supervisor_outcome_ledger_ready_score"] == 0
    assert "missing_shadow_outcome_schema:missing.json" in missing_result["schema_errors"]

    inconsistent = _episode(
        "su15",
        1,
        mode="applied",
        redirects=[
            _redirect("drop_goal_bias", resolved=True, action_index=120, actions_to_levelup=5)
        ],
        termination="policy_done",
    )
    inconsistent["trajectory_supervisor"]["arm_outcomes"]["drop_goal_bias"]["fired"] = 2
    result = exp.reduce_receipts(
        [
            {
                "payload": inconsistent,
                "path": "inconsistent.json",
                "sha256": "sha256:bad",
            }
        ]
    )
    assert any(error.startswith("arm_outcome_join_mismatch:") for error in result["schema_errors"])

    shadow_inconsistent = _episode(
        "su15",
        1,
        redirects=[
            _redirect("drop_goal_bias", resolved=True, action_index=120, actions_to_levelup=5)
        ],
    )
    shadow_inconsistent["trajectory_supervisor"]["would_have_arm_outcomes"]["drop_goal_bias"][
        "helped"
    ] = 0
    shadow_result = exp.reduce_receipts(
        [
            {
                "payload": shadow_inconsistent,
                "path": "shadow-inconsistent.json",
                "sha256": "sha256:bad-shadow",
            }
        ]
    )
    assert shadow_result["schema_errors"] == [
        "would_have_outcome_join_mismatch:shadow-inconsistent.json"
    ]


# REQ-ARC-7625; SCENARIO-ARC-7625-SUPPORT.
def test_supported_never_helped_arm_gets_loo_and_binomial_upper_bound() -> None:
    receipts = []
    counts = {"su15": 7, "sp80": 7, "ft09": 6}
    for game, count in counts.items():
        redirects = [
            _redirect(
                "drop_goal_bias",
                resolved=False,
                action_index=120 + index,
                actions_to_levelup=None,
            )
            for index in range(count)
        ]
        receipts.append(
            {
                "payload": _episode(
                    game, count, mode="applied", redirects=redirects, termination="game_over"
                ),
                "path": f"{game}.json",
                "sha256": f"sha256:{game}",
            }
        )

    reduction = exp.reduce_receipts(receipts)
    stats = reduction["arm_statistics"]["drop_goal_bias"]

    assert stats["eligible"] is True
    assert stats["uncensored_actual_firings"] == 20
    assert stats["game_count"] == 3
    assert stats["help_rate"] == 0.0
    assert set(stats["per_game_rates"]) == set(counts)
    assert len(stats["leave_one_game_out"]) == 3
    assert stats["leave_one_game_out_stable"] is True
    assert 0.0 < stats["never_helped_binomial_upper_95"] < 0.2


# REQ-ARC-7625; SCENARIO-ARC-7625-EXHAUSTION.
def test_exhausted_applied_receipt_yields_only_general_new_arm_specification() -> None:
    episode = _episode("su15", 1, mode="applied")
    supervisor = episode["trajectory_supervisor"]
    supervisor["arms_used"] = list(supervisor["arms_enabled"])
    supervisor["stagnations_unredirected"] = 1
    supervisor["unredirected_windows"] = [
        {
            "arms_enabled": list(supervisor["arms_enabled"]),
            "arms_used": list(supervisor["arms_enabled"]),
            "attempt_cap_reached": True,
            "diversity_active": True,
            "goal_bias_installed": False,
            "evidence_floor_met": True,
        }
    ]

    reduction = exp.reduce_receipts(
        [{"payload": episode, "path": "exhausted.json", "sha256": "sha256:exhausted"}]
    )

    spec = reduction["new_arm_specification"]
    assert spec["status"] == "specified_from_receipts"
    assert spec["game_specific_route"] is False
    assert spec["model_generated"] is False
    assert "su15" not in json.dumps(spec)


# REQ-ARC-7625; SCENARIO-ARC-7625-AUTH.
def test_authentication_mismatch_is_reported_before_reduction(tmp_path: Path) -> None:
    producer = _write_producer(tmp_path, "v664", [_episode("su15", 1)])
    raw = next((tmp_path / producer.episode_dir).glob("*.json"))
    raw.write_text(json.dumps(_episode("su15", 99)), encoding="utf-8")

    selected = exp.select_authenticated_receipts(tmp_path, [producer])

    assert selected["receipts"] == []
    assert selected["schema_errors"] == [f"source_hash_mismatch:{raw.relative_to(tmp_path)}"]


def _zero_artifact() -> dict:
    episode = _episode("su15", 1)
    reduction = exp.reduce_receipts(
        [{"payload": episode, "path": "receipt.json", "sha256": "sha256:receipt"}]
    )
    artifact = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        reduction=reduction,
        preconditions_checked=[
            exp.check_row(
                "source_custody",
                upstream="fixture",
                path="receipt.json",
                field="is_file",
                operator="==",
                expected=True,
                observed=True,
            )
        ],
        source_hashes={
            "producer_artifacts": {"receipt.json": "sha256:receipt"},
            "pre_gate_receipts": {
                "results/experiment_7611_v664_arc_matched_support.json": "sha256:7611",
                "results/experiment_7612_v664_arc_history_measurement.json": "sha256:7612",
            },
            "missing_artifacts": [],
        },
        validation_receipts=[],
        phase_spans=[],
        sample_counts={
            "intended_independent_units": 6,
            "observed_independent_units": 1,
            "excluded_independent_units": 5,
            "censored_independent_units": 0,
            "observed_receipts": 1,
            "duplicate_receipts": 0,
        },
    )
    return artifact


# REQ-ARC-7625; SCENARIO-ARC-7625-ZERO; SCENARIO-ARC-7625-TERMINAL.
def test_zero_firing_artifact_is_complete_null_and_self_validating(tmp_path: Path) -> None:
    artifact = _zero_artifact()

    assert artifact["honest_verdict"] == "complete_null_no_firings_nothing_to_refine"
    assert artifact["verdict_class"] == "null"
    assert artifact["flagged_adversarial"] is False
    assert artifact["supervisor_outcome_ledger_ready_score"] == 1
    assert artifact["selection_recommendation"]["action"] == "no_change"
    assert artifact["selection_recommendation"]["live_defaults_changed"] is False
    assert artifact["acceptance_gate_results"]["benefit"]["result"] is False
    assert artifact["solve_provenance"] == "live_agent_self_discovery"
    assert artifact["solve_claim"] is False
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["field_principles"]) == set(artifact)
    assert exp.validate_artifact(artifact) == []

    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_reduction(path) == []
    assert exp.independent_reduction(path) == []


# REQ-ARC-7625; SCENARIO-ARC-7625-JOIN.
def test_missing_schema_builds_complete_blocked_artifact() -> None:
    reduction = exp.reduce_receipts(
        [
            {
                "payload": {
                    "game": "su15",
                    "seed": 1,
                    "policy": "E3AgentPolicy",
                    "trajectory_supervisor": {"enabled": False},
                },
                "path": "bad.json",
                "sha256": "sha256:bad",
            }
        ]
    )
    artifact = exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=0.5,
        reduction=reduction,
        preconditions_checked=[],
        source_hashes={
            "producer_artifacts": {"bad.json": "sha256:bad"},
            "pre_gate_receipts": {},
            "missing_artifacts": [],
        },
        validation_receipts=[],
        phase_spans=[],
        sample_counts={
            "intended_independent_units": 6,
            "observed_independent_units": 1,
            "excluded_independent_units": 5,
            "censored_independent_units": 0,
            "observed_receipts": 1,
            "duplicate_receipts": 0,
        },
    )

    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    failed = artifact["gate_check_summary"][0]
    assert failed["check"] == "supervisor_outcome_schema"
    assert failed["upstream"] == "selected_trajectory_supervisor_receipts"
    assert failed["path"] == "bad.json"
    assert failed["field"] == "trajectory_supervisor"
    assert failed["operator"] == "has_complete_outcome_schema"
    assert failed["expected"] is True
    assert failed["observed"] is False
    assert exp.validate_artifact(artifact) == []


def test_validator_rejects_mutated_metrics_and_checksum(tmp_path: Path) -> None:
    artifact = _zero_artifact()
    artifact["rows"][0]["rate"] = 1.0
    assert "row_rate_mismatch:0" in exp.validate_artifact(artifact)
    assert "reproducibility_checksum" in exp.validate_artifact(artifact)

    path = tmp_path / "bad.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert "row_rate_mismatch:0" in exp.independent_reduction(path)


def test_source_selector_rejects_missing_flagged_and_non_live_inputs(tmp_path: Path) -> None:
    missing = exp.ProducerSpec("missing", Path("missing.json"), Path("missing"))
    flagged_dir = tmp_path / "flagged-raw"
    flagged_dir.mkdir()
    flagged_result = tmp_path / "flagged.json"
    flagged_result.write_text(json.dumps({"flagged_adversarial": True}), encoding="utf-8")
    flagged = exp.ProducerSpec("flagged", Path("flagged.json"), Path("flagged-raw"))
    no_dir_result = tmp_path / "no-dir.json"
    no_dir_result.write_text(json.dumps({"source_artifact_hashes": {}}), encoding="utf-8")
    no_dir = exp.ProducerSpec("no-dir", Path("no-dir.json"), Path("absent-dir"))

    result = exp.select_authenticated_receipts(tmp_path, [missing, flagged, no_dir])
    assert "producer_missing:missing.json" in result["schema_errors"]
    assert "producer_flagged:flagged.json" in result["schema_errors"]
    assert "episode_directory_missing:absent-dir" in result["schema_errors"]

    producer = _write_producer(
        tmp_path,
        "invalid-live",
        [_episode("su15", 1), _episode("sp80", 2)],
    )
    paths = sorted((tmp_path / producer.episode_dir).glob("*.json"))
    wrong_policy = json.loads(paths[0].read_text())
    wrong_policy["policy"] = "OfflineSolver"
    paths[0].write_text(json.dumps(wrong_policy), encoding="utf-8")
    adapter_used = json.loads(paths[1].read_text())
    adapter_used["adapter_withheld"] = False
    paths[1].write_text(json.dumps(adapter_used), encoding="utf-8")
    parent = json.loads((tmp_path / producer.result_path).read_text())
    for path in paths:
        parent["source_artifact_hashes"]["authenticated_sources"][
            path.relative_to(tmp_path).as_posix()
        ] = _sha(path)
    (tmp_path / producer.result_path).write_text(json.dumps(parent), encoding="utf-8")

    rejected = exp.select_authenticated_receipts(tmp_path, [producer])
    assert any(error.startswith("policy_not_live_path:") for error in rejected["schema_errors"])
    assert any(error.startswith("adapter_not_withheld:") for error in rejected["schema_errors"])
    assert exp._authenticated_hashes({}) == {}
    assert exp._relative(tmp_path, tmp_path / "child") == str(tmp_path.resolve())


def test_reducer_rejects_bad_payload_and_applied_schema() -> None:
    invalid = exp.reduce_receipts([{"payload": "bad", "path": "payload.json"}])
    assert invalid["schema_errors"] == ["episode_payload_invalid:payload.json"]
    missing = exp.reduce_receipts(
        [{"payload": {"game": "su15"}, "path": "missing.json", "sha256": "sha256:x"}]
    )
    assert missing["schema_errors"] == ["missing_outcome_schema:missing.json"]
    malformed = _episode("su15", 1, mode="applied")
    del malformed["trajectory_supervisor"]["arm_outcomes"]
    result = exp.reduce_receipts(
        [{"payload": malformed, "path": "applied.json", "sha256": "sha256:x"}]
    )
    assert result["schema_errors"] == ["missing_applied_outcome_schema:applied.json"]
    assert exp._join_matches_outcomes({"arm_outcomes": []}, []) is False
    assert exp._join_matches_outcomes({"arm_outcomes": {"x": "bad"}}, []) is False


def test_small_actual_sample_and_stable_sample_choose_different_recommendations() -> None:
    one = _episode(
        "su15",
        1,
        mode="applied",
        redirects=[
            _redirect("drop_goal_bias", resolved=False, action_index=120, actions_to_levelup=None)
        ],
        termination="game_over",
    )
    small = exp.reduce_receipts([{"payload": one, "path": "one.json", "sha256": "sha256:one"}])
    assert small["arm_statistics"]["drop_goal_bias"]["leave_one_game_out"][0]["help_rate"] is None
    assert exp._recommendation(small)["reason"] == "insufficient_support"

    receipts = []
    for game, count in (("su15", 7), ("sp80", 7), ("ft09", 6)):
        rows = [
            _redirect(
                "drop_goal_bias", resolved=False, action_index=120 + i, actions_to_levelup=None
            )
            for i in range(count)
        ]
        receipts.append(
            {
                "payload": _episode(
                    game, count, mode="applied", redirects=rows, termination="game_over"
                ),
                "path": f"{game}.json",
                "sha256": f"sha256:{game}",
            }
        )
    stable = exp.reduce_receipts(receipts)
    recommendation = exp._recommendation(stable)
    assert recommendation["action"] == "curated_priority_review"
    assert recommendation["candidate_arms"] == ["drop_goal_bias"]


def _artifact_from_reduction(reduction: dict, **kwargs) -> dict:
    return exp.build_artifact(
        run_date=exp.RUN_DATE,
        duration_s=1.0,
        reduction=reduction,
        preconditions_checked=[],
        source_hashes={
            "producer_artifacts": {"receipt.json": "sha256:receipt"},
            "pre_gate_receipts": {},
            "missing_artifacts": [],
        },
        validation_receipts=[],
        phase_spans=[],
        sample_counts={
            "intended_independent_units": 6,
            "observed_independent_units": 1,
            "excluded_independent_units": 5,
            "censored_independent_units": 0,
            "observed_receipts": 1,
            "duplicate_receipts": 0,
        },
        **kwargs,
    )


def test_artifact_verdict_branches_remain_nonpositive() -> None:
    redirect = _redirect("drop_goal_bias", resolved=True, action_index=120, actions_to_levelup=3)
    small = exp.reduce_receipts(
        [
            {
                "payload": _episode(
                    "su15", 1, mode="applied", redirects=[redirect], termination="policy_done"
                ),
                "path": "small.json",
                "sha256": "sha256:small",
            }
        ]
    )
    small_artifact = _artifact_from_reduction(small)
    assert small_artifact["honest_verdict"] == "complete_null_insufficient_actual_firing_support"
    assert exp.validate_artifact(small_artifact) == []

    invalid_validation = _artifact_from_reduction(small, validation_passed=False)
    assert invalid_validation["verdict_class"] == "disqualified"
    flagged = _artifact_from_reduction(small, flagged_adversarial=True)
    assert flagged["honest_verdict"] == "complete_disqualified_adversarial_reader"

    block = exp.check_row(
        "missing",
        upstream="fixture",
        path="fixture.json",
        field="exists",
        operator="==",
        expected=True,
        observed=False,
    )
    blocked = _artifact_from_reduction(
        {**small, "supervisor_outcome_ledger_ready_score": 0}, blocking_checks=[block]
    )
    assert blocked["honest_verdict"] == "complete_blocked_precondition"
    assert blocked["gate_check_summary"] == [block]

    supported_receipts = []
    for game, count in (("su15", 7), ("sp80", 7), ("ft09", 6)):
        redirects = [
            _redirect(
                "drop_goal_bias", resolved=True, action_index=120 + index, actions_to_levelup=2
            )
            for index in range(count)
        ]
        supported_receipts.append(
            {
                "payload": _episode(
                    game, count, mode="applied", redirects=redirects, termination="policy_done"
                ),
                "path": f"{game}.json",
                "sha256": f"sha256:{game}",
            }
        )
    supported = _artifact_from_reduction(exp.reduce_receipts(supported_receipts))
    assert supported["honest_verdict"] == "complete_null_observational_supervisor_association"
    assert supported["verdict_class"] == "null"


def test_validator_names_each_mutated_contract() -> None:
    base = _zero_artifact()
    cases = [
        (lambda row: row.__setitem__("schema", "wrong"), "schema"),
        (lambda row: row.__setitem__("field_principles", {}), "field_principles"),
        (lambda row: row.__setitem__("verdict_class", "other"), "verdict_class"),
        (lambda row: row.__setitem__("honest_verdict", "null"), "honest_verdict"),
        (lambda row: row.__setitem__("MODEL_SPECS", ["model"]), "model_declaration"),
        (lambda row: row.__setitem__("solve_claim", True), "solve_claim"),
        (
            lambda row: row.__setitem__("production_defaults_changed", True),
            "production_defaults_changed",
        ),
        (
            lambda row: row.__setitem__("inference_substrate_class", "wrong"),
            "inference_substrate_class",
        ),
        (lambda row: row.__setitem__("current_work_receipt", {}), "current_work_receipt"),
        (lambda row: row.__setitem__("acceptance_gate_results", {}), "acceptance_gate_results"),
        (
            lambda row: row["acceptance_gate_results"]["validity"].pop("principle"),
            "acceptance_gate_principles",
        ),
        (
            lambda row: row.__setitem__("supervisor_outcome_ledger_ready_score", 2),
            "supervisor_outcome_ledger_ready_score",
        ),
        (
            lambda row: row.__setitem__("honest_verdict", "complete_null_other"),
            "zero_firing_verdict",
        ),
    ]
    for mutate, expected in cases:
        changed = deepcopy(base)
        mutate(changed)
        assert expected in exp.validate_artifact(changed)

    blocked = deepcopy(
        exp.build_artifact(
            run_date=exp.RUN_DATE,
            duration_s=1.0,
            reduction={
                "schema_errors": ["missing_outcome_schema:bad.json"],
                "supervisor_outcome_ledger_ready_score": 0,
            },
            preconditions_checked=[],
            source_hashes={"producer_artifacts": {}, "missing_artifacts": []},
            validation_receipts=[],
            phase_spans=[],
            sample_counts={},
        )
    )
    blocked["gate_check_summary"] = {}
    assert "blocked_gate_check_summary" in exp.validate_artifact(blocked)


def test_validator_rejects_bad_rows_redirects_and_counts(tmp_path: Path) -> None:
    redirect = _redirect("drop_goal_bias", resolved=True, action_index=120, actions_to_levelup=2)
    reduction = exp.reduce_receipts(
        [
            {
                "payload": _episode(
                    "su15", 1, mode="applied", redirects=[redirect], termination="policy_done"
                ),
                "path": "actual.json",
                "sha256": "sha256:actual",
            }
        ]
    )
    base = _artifact_from_reduction(reduction)
    assert exp.validate_artifact(base) == []

    row_invalid = deepcopy(base)
    row_invalid["rows"] = ["bad"]
    assert "row_invalid:0" in exp.validate_artifact(row_invalid)
    bad_operand = deepcopy(base)
    bad_operand["rows"][0]["operand_checksum"] = "sha256:bad"
    assert "row_operand_checksum:0" in exp.validate_artifact(bad_operand)
    actual_invalid = deepcopy(base)
    actual_invalid["actual_redirects"] = ["bad"]
    assert "actual_redirect_invalid:0" in exp.validate_artifact(actual_invalid)
    bad_rate = deepcopy(base)
    bad_rate["actual_redirects"][0]["rate"] = 0.0
    assert "actual_redirect_rate_mismatch:0" in exp.validate_artifact(bad_rate)
    bad_hash = deepcopy(base)
    bad_hash["actual_redirects"][0]["row_sha256"] = "sha256:bad"
    assert "actual_redirect_checksum:0" in exp.validate_artifact(bad_hash)
    bad_count = deepcopy(base)
    bad_count["redirect_counts"]["helped_count"] = 2
    assert "redirect_count_mismatch:helped_count" in exp.validate_artifact(bad_count)

    path = tmp_path / "artifact.json"
    independent_bad = deepcopy(base)
    independent_bad["redirect_counts"]["proposed_redirect_count"] = 99
    path.write_text(json.dumps(independent_bad), encoding="utf-8")
    assert "independent_count_mismatch:proposed_redirect_count" in exp.independent_reduction(path)


def test_command_manifests_preconditions_and_helpers(tmp_path: Path, capsys) -> None:
    validation_root = tmp_path / "validation"
    commands = exp.build_validation_commands(exp.repo_root(), validation_root)
    assert [row.name for row in commands] == [
        "worktree_imports",
        "focused_pytest",
        "changed_module_coverage",
        "changed_module_coverage_report",
        "ruff_check",
        "ruff_format",
        "changed_module_mypy",
        "scoped_spec_coverage",
    ]
    assert any(row.argv[0] == "/usr/bin/env" for row in commands)
    assert (validation_root / "coverage").is_dir()
    assert (validation_root / "pytest").is_dir()
    e2e_root = tmp_path / "e2e"
    assert [row.name for row in exp.build_e2e_commands(exp.repo_root(), e2e_root)] == [
        "e2e_011",
        "e2e_013",
    ]
    assert e2e_root.is_dir()
    terminal = exp.build_terminal_commands(exp.repo_root(), tmp_path / "candidate.json")
    assert [row.name for row in terminal] == [
        "declared_entrypoint",
        "fresh_process_cold_reduction",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]

    assert exp.repo_root().name == "carnot"
    available, version = exp._tool_version((str(exp.repo_root() / ".venv/bin/python"), "--version"))
    assert available is True and "Python" in version
    missing, error = exp._tool_version((str(tmp_path / "missing-tool"), "--version"))
    assert missing is False and error == "FileNotFoundError"
    checks, context = exp.collect_preconditions(exp.repo_root())
    assert checks and all(row["passed"] for row in checks)
    assert context["registry_games"] == sorted(exp.GAMES)

    started = 10.0
    span = exp._phase_span("fixture", 10.0, started, completed_units=2)
    assert span["phase"] == "fixture" and span["completed_units"] == 2
    assert (
        exp._receipt_summary([{"name": "x", "exit_code": 0, "passed": True}])["x"]["passed"] is True
    )
    assert exp._all_passed([]) is False
    assert exp._all_passed([{"passed": True}]) is True
    exp.progress(0.0, "fixture", "done", units=1)
    assert "phase=fixture" in capsys.readouterr().out

    selection = {
        "receipts": [{"payload": {"game": "su15"}, "path": "raw.json", "sha256": "sha256:raw"}],
        "observed_receipts": 2,
        "deduplicated_receipts": 1,
        "duplicate_receipts": 1,
        "source_hashes": {"raw.json": "sha256:raw"},
        "schema_errors": ["producer_missing:x"],
    }
    reduction = {
        "per_game_results": [
            {"game": "su15", "actual_firings": 1, "censoring": {"actual_firings_censored": 1}}
        ]
    }
    budget = exp._sample_counts(selection, reduction)
    assert budget["observed_independent_units"] == 1
    assert budget["censored_independent_units"] == 1
    groups = exp._source_hash_groups(tmp_path, {"hashes": {"a": "sha256:a"}}, selection)
    assert groups["actual_producers"] == [{"path": "raw.json", "sha256": "sha256:raw"}]
    assert groups["missing_artifacts"] == ["producer_missing:x"]


def test_cli_reader_modes_and_argument_validation(tmp_path: Path, monkeypatch) -> None:
    artifact = _zero_artifact()
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.main(["--validate", str(path)]) == 0
    assert exp.main(["--cold-reduce", str(path)]) == 0
    assert exp.main(["--independent-reduce", str(path)]) == 0
    with pytest.raises(SystemExit):
        exp.parse_args(["--date", "20260101"])

    called = {}
    monkeypatch.setattr(
        exp,
        "run_experiment",
        lambda root, date, output: called.update(root=root, date=date, output=output),
    )
    assert exp.main(["--output", str(tmp_path / "out.json")]) == 0
    assert called["date"] == exp.RUN_DATE

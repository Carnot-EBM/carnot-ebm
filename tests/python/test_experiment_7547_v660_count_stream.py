"""Tests for REQ-CL-7547 and SCENARIO-CL-7547-*.

The private rows test failure boundaries. Real cached rows test the registered
counts and identities without replacing the empirical deliverable.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7547_v660_count_stream as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _feature(group: int, role: str, probability: float) -> dict[str, Any]:
    """Build one label-free row whose first feature exactly maps to its forecast."""

    source_hash = f"sha256:{group:064x}"
    response_hash = f"sha256:{group + 1000:064x}"
    return {
        "group_id": f"group-{group:04d}",
        "role": role,
        "source_hash": source_hash,
        "response_hash": response_hash,
        "raw_whole_expectation": probability,
        "features": [exp.logit(probability), *([0.0] * 9)],
    }


def _predictor(row: dict[str, Any]) -> dict[str, Any]:
    """Mirror the original public identity fields for one private feature row."""

    return {
        "group_id": row["group_id"],
        "role": row["role"],
        "source_hash": row["source_hash"],
        "response_hash": row["response_hash"],
        "source_family": f"family-{int(row['group_id'][-1]) % 3}",
    }


def _public_rows() -> list[dict[str, Any]]:
    """Create enough role-separated rows for a release and a censored tail."""

    rows = [_feature(index, "training", (index + 0.5) / 16.0) for index in range(16)]
    rows.extend(_feature(100 + index, "online", (index + 1) / 22.0) for index in range(20))
    rows.extend(_feature(200 + index, "test", (index + 1) / 10.0) for index in range(8))
    return rows


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Represent one bounded command without running it in a unit test."""

    return {
        "name": name,
        "command": f"private {name}",
        "command_argv": ["private", name],
        "scope": "private_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "a" * 64,
        "output_tail": "private fixture",
        "passed": passed,
        "timed_out": False,
    }


def _receipts(*, terminal: bool = True) -> list[dict[str, Any]]:
    """Supply every fixed validation category to the pure artifact builder."""

    names = list(validation_scope.REQUIRED_CHECK_NAMES)
    if terminal:
        names.extend(exp.TERMINAL_CHECK_NAMES)
    return [_receipt(name) for name in names]


def test_protocol_and_label_free_bin_means_are_frozen() -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-PROTOCOL."""

    protocol = exp.frozen_protocol()
    assert protocol["order_seeds"] == [7549001, 7549002, 7549003, 7549004, 7549005]
    assert protocol["block_size"] == protocol["feedback_delay"] == 8
    assert protocol["prior_mass"] == 8
    assert protocol["clip"] == 1e-4
    assert protocol["arms"] == ["frozen", "global", "local", "shuffled_local"]
    assert protocol["prediction_precedes_release_and_update"] is True

    training = [
        _feature(1, "training", 0.02),
        _feature(2, "training", 0.10),
        _feature(3, "training", 0.90),
    ]
    config = exp.compute_count_config(training)
    assert config.bin_means[0] == pytest.approx(0.06)
    assert config.bin_means[1] == pytest.approx(3 / 16)
    assert config.bin_means[4] == pytest.approx(9 / 16)
    assert config.bin_means[7] == pytest.approx(0.90)
    assert config.global_mean == pytest.approx(0.34)


def test_join_preserves_raw_orientation_roles_and_hashes() -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-CUSTODY."""

    features = _public_rows()
    predictors = [_predictor(row) for row in features]
    joined = exp.assemble_public_rows(features, predictors)
    assert len(joined["training"]) == 16
    assert len(joined["online"]) == 20
    assert len(joined["test"]) == 8
    assert joined["online"][0]["base_probability"] == features[16]["raw_whole_expectation"]
    assert joined["online"][0]["label_orientation"] == "one_means_contains_unsupported"

    changed = deepcopy(predictors)
    changed[0]["source_hash"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        exp.assemble_public_rows(features, changed)
    changed = deepcopy(predictors)
    changed[0]["role"] = "online"
    with pytest.raises(ValueError, match="role_mismatch"):
        exp.assemble_public_rows(features, changed)
    changed_features = deepcopy(features)
    changed_features[0]["raw_whole_expectation"] = 0.0
    with pytest.raises(ValueError, match="raw_probability_invalid"):
        exp.assemble_public_rows(changed_features, predictors)


def test_orders_are_complete_and_do_not_accept_labels() -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-PROTOCOL."""

    joined = exp.assemble_public_rows(_public_rows(), [_predictor(row) for row in _public_rows()])
    frozen = exp.freeze_public_protocol(joined)
    expected = {row["group_id"] for row in joined["online"]}
    assert set(frozen["orders"]) == {str(seed) for seed in exp.ORDER_SEEDS}
    assert all(
        set(order) == expected and len(order) == len(expected)
        for order in frozen["orders"].values()
    )
    assert frozen["labels_read"] is False
    assert '"label":' not in json.dumps(frozen)
    assert frozen["protocol_hash"] == exp.canonical_hash(frozen["hash_payload"])


def test_label_reader_uses_exp7509_hallucination_orientation() -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-CUSTODY."""

    joined = exp.assemble_public_rows(_public_rows(), [_predictor(row) for row in _public_rows()])
    evaluators = [
        {"group_id": row["group_id"], "role": role, "label": index % 2}
        for role in ("online", "test")
        for index, row in enumerate(joined[role])
    ]
    labels = exp.read_oriented_labels(evaluators, joined, roles=("online", "test"))
    assert labels["online"][joined["online"][0]["group_id"]] == 1
    assert labels["online"][joined["online"][1]["group_id"]] == 0

    wrong = deepcopy(evaluators)
    wrong[0]["role"] = "test"
    with pytest.raises(ValueError, match="label_identity_mismatch"):
        exp.read_oriented_labels(wrong, joined, roles=("online", "test"))
    with pytest.raises(ValueError, match="label_orientation_invalid"):
        exp.read_oriented_labels(
            evaluators,
            joined,
            roles=("online", "test"),
            orientation="one_means_supported",
        )


def test_predict_release_update_persist_reload_with_censored_tail(tmp_path: Path) -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-CHRONOLOGY and -RESTART."""

    public = _public_rows()
    joined = exp.assemble_public_rows(public, [_predictor(row) for row in public])
    frozen = exp.freeze_public_protocol(joined)
    evaluators = [
        {"group_id": row["group_id"], "role": role, "label": index % 2}
        for role in ("online", "test")
        for index, row in enumerate(joined[role])
    ]
    labels = exp.read_oriented_labels(evaluators, joined, roles=("online", "test"))
    seed = exp.ORDER_SEEDS[0]
    replay = exp.run_order_lifecycle(
        joined["online"],
        labels["online"],
        frozen["orders"][str(seed)],
        frozen["count_config"],
        seed=seed,
        checkpoint_dir=tmp_path,
    )

    assert replay["prediction_count"] == 20 * 4
    assert replay["released_event_count"] == 8
    assert replay["censored_event_count"] == 12
    assert replay["chronology_violation_count"] == 0
    assert replay["restart_mismatch_count"] == 0
    assert replay["exact_normalization_mismatch_count"] == 0
    assert replay["release_rows"][0]["release_time"] == 15
    assert replay["release_rows"][0]["prediction_completed_through"] == 15
    assert replay["release_rows"][-1]["disposition"] == "censored_end_of_stream"
    assert replay["release_rows"][-1]["short_final_block"] is True
    released = replay["release_rows"][0]
    assert sorted(released["labels"]) == sorted(released["shuffled_labels"])
    assert set(released["shuffled_label_origins"]) == set(released["event_ids"])
    assert replay["persistence_receipts"][0]["payload_equal_after_reload"] is True
    assert replay["persistence_receipts"][0]["next_prediction_equal_after_reload"] is True


def test_real_cached_rows_keep_registered_counts_and_sources() -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-CUSTODY uses real authenticated rows."""

    joined = exp.load_public_role_rows(ROOT)
    assert {role: len(joined[role]) for role in joined} == {
        "training": 176,
        "calibration_tuning": 60,
        "online": 159,
        "test": 116,
    }
    assert len({row["source_hash"] for rows in joined.values() for row in rows}) == 511
    assert all(
        row["base_probability"] == row["raw_whole_expectation"]
        for rows in joined.values()
        for row in rows
    )
    frozen = exp.freeze_public_protocol(joined)
    assert frozen["role_counts"] == {
        "training": 176,
        "calibration_tuning": 60,
        "online": 159,
        "test": 116,
    }
    labels = exp.load_private_labels(ROOT, joined)
    assert len(labels["online"]) == 159
    assert len(labels["test"]) == 116


def test_mutation_controls_and_preconditions_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-MUTATION."""

    controls = exp.run_private_mutation_controls()
    assert {row["mutation"] for row in controls} == {
        "source_hash",
        "role_membership",
        "label_orientation",
        "raw_normalization",
    }
    assert all(row["qualified"] and row["rejected"] for row in controls)

    checks, hashes = exp.collect_preconditions(ROOT)
    assert all(row["passed"] for row in checks if row["required"])
    assert exp.FEATURE_PATH.as_posix() in hashes
    assert any(row["field_path"] == "evidence_ready_score" for row in checks)
    assert any(row["field_path"] == "causal_audit_complete_score" for row in checks)
    assert any(row["field_path"] == "count_memory_ready_score" for row in checks)

    absent = tmp_path / "absent"
    blocked_checks, _ = exp.collect_preconditions(absent)
    failed = next(row for row in blocked_checks if row["required"] and not row["passed"])
    blocked = exp.build_blocked_artifact(failed, blocked_checks, duration_s=0.01)
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["cached_stream_ready_score"] == 0
    assert blocked["gate_check_summary"]["first_failure"]["field_path"]
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["planned_inference_substrate_class"] == "no_model_load"


def test_artifact_reduction_keeps_readiness_separate_from_benefit(tmp_path: Path) -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-ARTIFACT."""

    public = _public_rows()
    joined = exp.assemble_public_rows(public, [_predictor(row) for row in public])
    frozen = exp.freeze_public_protocol(joined)
    evaluators = [
        {"group_id": row["group_id"], "role": role, "label": index % 2}
        for role in ("online", "test")
        for index, row in enumerate(joined[role])
    ]
    labels = exp.read_oriented_labels(evaluators, joined, roles=("online", "test"))
    replays = [
        exp.run_order_lifecycle(
            joined["online"],
            labels["online"],
            frozen["orders"][str(seed)],
            frozen["count_config"],
            seed=seed,
            checkpoint_dir=tmp_path / str(seed),
        )
        for seed in exp.ORDER_SEEDS
    ]
    artifact = exp.build_artifact(
        frozen=frozen,
        replays=replays,
        retention_rows=joined["test"],
        labels=labels,
        validation_receipts=_receipts(),
        preconditions_checked=[],
        source_hashes={},
        sidecars={},
        duration_s=1.0,
        phase_spans=[],
    )
    reduction = exp.independent_reduce(artifact)
    assert reduction["cached_stream_ready_score"] == 1
    assert reduction["validity_passed"] is True
    assert reduction["benefit_measured"] is False
    assert artifact["verdict_class"] == "null"
    assert artifact["positive_claim"] is False
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["exposure_scope"]["fresh_confirmatory_claim_allowed"] is False
    assert artifact["sample_size_budget"]["independent_online_groups"]["completed"] == 20
    exp.validate_artifact(artifact, verify_sidecars=False)

    changed = deepcopy(artifact)
    changed["cached_stream_ready_score"] = 0
    with pytest.raises(ValueError, match="cached_stream_ready_score_mismatch"):
        exp.validate_artifact(changed, verify_sidecars=False)
    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        exp.validate_artifact(changed, verify_sidecars=False)

    artifact_path = tmp_path / "artifact.json"
    exp.atomic_json(artifact_path, artifact)
    assert exp.cold_replay(artifact_path, verify_sidecars=False)["cached_stream_ready_score"] == 1
    sidecars = exp.write_stream_sidecars(tmp_path, frozen, replays, joined["test"], labels)
    with_sidecars = deepcopy(artifact)
    with_sidecars["raw_sidecars"] = sidecars
    with_sidecars["reproducibility_checksum"] = exp.reproducibility_checksum(with_sidecars)
    exp.validate_artifact(with_sidecars, root=tmp_path)

    early_mutations = (
        ("schema", "changed", "artifact_identity_mismatch"),
        ("run_date", "19000101", "artifact_date_or_milestone_mismatch"),
        ("MODEL_SPECS", ["forbidden"], "model_specs_must_be_empty"),
        ("model_invoked", True, "current_model_invocation_mismatch"),
        (
            "inference_substrate_class",
            "model_full_generation",
            "inference_substrate_class_mismatch",
        ),
        ("inference_substrate", "changed", "inference_substrate_mismatch"),
        ("execution_venue", "host_cpu", "execution_venue_mismatch"),
        ("positive_claim", True, "unmeasured_benefit_claim"),
        ("verdict_class", "positive", "verdict_class_mismatch"),
    )
    for field, replacement, error in early_mutations:
        changed = deepcopy(artifact)
        changed[field] = replacement
        with pytest.raises(ValueError, match=error):
            exp.validate_artifact(changed, verify_sidecars=False)
    changed = deepcopy(artifact)
    changed["independent_reduction"]["protocol_valid"] = False
    with pytest.raises(ValueError, match="independent_reduction_mismatch"):
        exp.validate_artifact(changed, verify_sidecars=False)
    changed = deepcopy(artifact)
    changed["field_principles"].pop("rows")
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="field_principles_incomplete"):
        exp.validate_artifact(changed, verify_sidecars=False)

    without_terminal = exp.build_artifact(
        frozen=frozen,
        replays=replays,
        retention_rows=joined["test"],
        labels=labels,
        validation_receipts=_receipts(terminal=False),
        preconditions_checked=[],
        source_hashes={},
        sidecars={},
        duration_s=1.0,
        phase_spans=[],
    )
    with pytest.raises(ValueError, match="terminal_validation_missing_or_failed"):
        exp.validate_artifact(without_terminal, verify_sidecars=False)


def test_validation_manifest_and_terminal_commands_are_exact(tmp_path: Path) -> None:
    """REQ-CL-7547 freezes scoped validation and exact terminal readers."""

    commands = exp.build_validation_commands(ROOT, tmp_path)
    assert [command.name for command in commands] == list(validation_scope.REQUIRED_CHECK_NAMES)
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert focused.argv[1:6] == ("-n", "0", "-o", "addopts=", "--no-cov")
    assert exp.TEST_PATH.as_posix() in focused.argv
    assert "--fail-under=100" in next(
        command.argv for command in commands if command.name == "changed_module_coverage_report"
    )
    terminal = exp.terminal_commands(tmp_path / "candidate.json")
    assert [command.name for command in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert terminal[-1].argv[-2:] == ("--strict", str(tmp_path / "candidate.json"))


def test_private_input_guards_reject_each_reachable_drift(tmp_path: Path) -> None:
    """REQ-CL-7547 rejects malformed private inputs without repairing them."""

    features = _public_rows()
    predictors = [_predictor(row) for row in features]
    with pytest.raises(ValueError, match="raw_probability_invalid"):
        exp.logit(0.0)
    with pytest.raises(ValueError, match="predictor_group_duplicate"):
        exp.assemble_public_rows(features, [*predictors, predictors[0]])
    with pytest.raises(ValueError, match="public_group_identity_invalid"):
        exp.assemble_public_rows(features, predictors[1:])

    changed = deepcopy(features)
    changed[1]["source_hash"] = changed[0]["source_hash"]
    changed_predictors = deepcopy(predictors)
    changed_predictors[1]["source_hash"] = changed[0]["source_hash"]
    with pytest.raises(ValueError, match="normalized_source_duplicate"):
        exp.assemble_public_rows(changed, changed_predictors)
    changed = deepcopy(features)
    changed[0]["features"] = [0.0]
    with pytest.raises(ValueError, match="feature_shape_invalid"):
        exp.assemble_public_rows(changed, predictors)
    changed = deepcopy(features)
    changed[0]["features"][1] = float("nan")
    with pytest.raises(ValueError, match="feature_value_invalid"):
        exp.assemble_public_rows(changed, predictors)
    changed = deepcopy(predictors)
    changed[0]["response_hash"] = "sha256:" + "b" * 64
    with pytest.raises(ValueError, match="response_hash_mismatch"):
        exp.assemble_public_rows(features, changed)
    with pytest.raises(ValueError, match="role_counts_invalid"):
        exp.assemble_public_rows(features, predictors, expected_counts=exp.EXPECTED_ROLE_COUNTS)
    with pytest.raises(ValueError, match="training_forecasts_required"):
        exp.compute_count_config([])
    with pytest.raises(ValueError, match="training_probability_invalid"):
        exp.compute_count_config([{"raw_whole_expectation": float("nan")}])

    joined = exp.assemble_public_rows(features, predictors)
    missing_role = dict(joined)
    missing_role.pop("test")
    with pytest.raises(ValueError, match="public_role_missing"):
        exp.freeze_public_protocol(missing_role)
    with pytest.raises(ValueError, match="label_roles_invalid"):
        exp.read_oriented_labels([], joined, roles=("training",))
    evaluators = [
        {"group_id": row["group_id"], "role": role, "label": index % 2}
        for role in ("online", "test")
        for index, row in enumerate(joined[role])
    ]
    changed_eval = deepcopy(evaluators)
    changed_eval[0]["label"] = 2
    with pytest.raises(ValueError, match="private_label_invalid"):
        exp.read_oriented_labels(changed_eval, joined, roles=("online", "test"))
    with pytest.raises(ValueError, match="incomplete_roster"):
        exp.read_oriented_labels(evaluators[1:], joined, roles=("online", "test"))
    frozen = exp.freeze_public_protocol(joined)
    labels = exp.read_oriented_labels(evaluators, joined, roles=("online", "test"))
    with pytest.raises(ValueError, match="lifecycle_roster_mismatch"):
        exp.run_order_lifecycle(
            joined["online"],
            labels["online"],
            frozen["orders"][str(exp.ORDER_SEEDS[0])][1:],
            frozen["count_config"],
            seed=exp.ORDER_SEEDS[0],
            checkpoint_dir=tmp_path,
        )


def test_sidecars_and_fresh_reader_bind_exact_bytes(tmp_path: Path) -> None:
    """REQ-CL-7547; SCENARIO-CL-7547-ARTIFACT binds sidecar bytes."""

    public_rows = _public_rows()
    public = exp.assemble_public_rows(public_rows, [_predictor(row) for row in public_rows])
    frozen = exp.freeze_public_protocol(public)
    evaluators = [
        {"group_id": row["group_id"], "role": role, "label": index % 2}
        for role in ("online", "test")
        for index, row in enumerate(public[role])
    ]
    labels = exp.read_oriented_labels(evaluators, public, roles=("online", "test"))
    replay = exp.run_order_lifecycle(
        public["online"],
        labels["online"],
        frozen["orders"][str(exp.ORDER_SEEDS[0])],
        frozen["count_config"],
        seed=exp.ORDER_SEEDS[0],
        checkpoint_dir=tmp_path / "checkpoints",
    )
    sidecars = exp.write_stream_sidecars(tmp_path, frozen, [replay], public["test"], labels)
    exp._verify_sidecars(tmp_path, sidecars)
    with pytest.raises(ValueError, match="raw_sidecar_set_mismatch"):
        exp._verify_sidecars(tmp_path, {})
    changed = deepcopy(sidecars)
    changed["frozen_protocol"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="raw_sidecar_hash_mismatch"):
        exp._verify_sidecars(tmp_path, changed)
    changed = deepcopy(sidecars)
    changed["frozen_protocol"]["bytes"] += 1
    with pytest.raises(ValueError, match="raw_sidecar_size_mismatch"):
        exp._verify_sidecars(tmp_path, changed)

    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="artifact_unreadable"):
        exp.cold_replay(missing, verify_sidecars=False)

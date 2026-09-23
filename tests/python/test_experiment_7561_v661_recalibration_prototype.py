"""Tests for REQ-CL-7561 and SCENARIO-CL-7561-*.

All labels in this file are constructed. They test arithmetic and lifecycle
rules. They do not stand in for the untouched empirical groups.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot import experiment_7561_v661_recalibration_prototype as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Represent one bounded command without starting a child process."""

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
    """Supply each required receipt for pure artifact construction tests."""

    names = list(validation_scope.REQUIRED_CHECK_NAMES)
    if terminal:
        names.extend(exp.TERMINAL_CHECK_NAMES)
    return [_receipt(name) for name in names]


@pytest.fixture(scope="module")
def complete_artifact(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict[str, Any]]:
    """Build the expensive private fixture evidence once for mutation tests."""

    root = tmp_path_factory.mktemp("exp7561-artifact")
    return root, exp.build_test_artifact(root, validation_receipts=_receipts())


def _feedback(start: int = 0, count: int = 8) -> list[tuple[str, int]]:
    """Build one legal release block with two probability regimes."""

    return [(f"event-{index:03d}", int(index % 3 != 0)) for index in range(start, start + count)]


def _predict_block(machine: exp.RecalibrationEventMachine, start: int, release_index: int) -> None:
    """Seal one block before its labels become visible."""

    for index in range(start, start + 8):
        machine.predict(f"event-{index:03d}", 0.1 + 0.1 * (index % 8), release_index)


def test_piecewise_map_energy_and_typed_policy() -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-NUMERICAL."""

    assert exp.piecewise_design(0.0).tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert exp.piecewise_design(1.0).tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert exp.map_probability(0.4375, exp.KNOTS) == pytest.approx(0.4375)
    assert exp.map_probability(-1.0, exp.KNOTS) == pytest.approx(0.0)
    assert exp.map_probability(2.0, exp.KNOTS) == pytest.approx(1.0)

    for probability in (0.0, 1e-8, 0.2, 0.5, 0.8, 1.0 - 1e-8, 1.0):
        energies = exp.binary_energies(probability)
        expected = min(1.0 - exp.LOG_CLIP, max(exp.LOG_CLIP, probability))
        assert exp.normalized_probability(energies) == pytest.approx(expected)

    assert exp.typed_decision(0.01)["action"] == "accept"
    assert exp.typed_decision(0.5)["action"] == "escalate"
    assert exp.typed_decision(0.9)["action"] == "reject"
    assert exp.typed_decision(0.04)["action"] == "escalate"
    assert exp.typed_decision(0.8)["action"] == "escalate"


@pytest.mark.parametrize(
    ("probabilities", "labels"),
    [
        ([0.0] * 32, [0] * 32),
        ([0.0] * 32, [1] * 32),
        ([1.0] * 32, [0] * 32),
        ([1.0] * 32, [1] * 32),
        ([index / 40 for index in range(41)], [index % 2 for index in range(41)]),
    ],
)
def test_primary_and_independent_solvers_agree(
    probabilities: list[float], labels: list[int]
) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-NUMERICAL uses two solvers."""

    gram, target = exp.statistics_from_examples(probabilities, labels)
    primary, receipt = exp.solve_constrained_map(gram, target)
    independent, independent_receipt = exp.solve_constrained_map(
        gram, target, method=exp.INDEPENDENT_SOLVER
    )
    assert receipt["converged"] is True
    assert independent_receipt["converged"] is True
    assert exp.constraint_errors(primary) == []
    assert exp.constraint_errors(independent) == []
    assert exp.quadratic_objective(primary, gram, target) == pytest.approx(
        exp.quadratic_objective(independent, gram, target), abs=1e-7
    )
    assert np.max(np.abs(primary - independent)) < 2e-5


def test_sufficient_statistics_change_without_raw_examples() -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-RESTART limits durable state."""

    learner = exp.SufficientStatisticMap.create()
    before = learner.theta.copy()
    receipt = learner.update_batch([(f"sample-{index}", 0.2, 1) for index in range(8)])
    payload = learner.to_payload()
    assert receipt["sample_count"] == 8
    assert learner.sample_count == 8
    assert not np.array_equal(learner.theta, before)
    assert payload["gram_shape"] == [9, 9]
    assert len(payload["gram"]) == 9
    assert len(payload["target"]) == 9
    assert "examples" not in payload
    assert "probabilities" not in payload
    assert "labels" not in payload
    assert max(abs(value - knot) for value, knot in zip(learner.theta, exp.KNOTS)) <= 0.1 + 1e-9
    assert exp.SufficientStatisticMap.from_payload(payload).state_hash() == learner.state_hash()

    state = learner.state_hash()
    with pytest.raises(ValueError, match="duplicate_feedback"):
        learner.update_batch([("sample-0", 0.2, 1)])
    assert learner.state_hash() == state
    with pytest.raises(ValueError, match="binary_label_required"):
        learner.update_batch([("new", 0.2, 2)])
    with pytest.raises(ValueError, match="probability_not_finite"):
        learner.predict(float("nan"))


def test_unconstrained_ablation_can_break_registered_bound() -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-FIXTURES keeps ablation separate."""

    probabilities = [0.0] * 160
    labels = [1] * 160
    gram, target = exp.statistics_from_examples(probabilities, labels)
    constrained, _receipt = exp.solve_constrained_map(gram, target)
    unconstrained, unconstrained_receipt = exp.solve_unconstrained_map(gram, target)
    assert constrained[0] == pytest.approx(0.1, abs=1e-7)
    assert unconstrained_receipt["converged"] is True
    assert unconstrained[0] > 0.5
    assert exp.constraint_errors(constrained) == []
    assert "movement_bound" in exp.constraint_errors(unconstrained)


def test_predict_release_update_persist_reload(tmp_path: Path) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-LIFECYCLE and -RESTART."""

    machine = exp.RecalibrationEventMachine.create(exp.fixture_count_config())
    _predict_block(machine, 0, 0)
    predictions = machine.predictions
    assert set(predictions["event-000"]["arms"]) == set(exp.ARMS)
    assert predictions["event-000"]["label_available_at_prediction"] is False
    initial_hash = machine.state_hash()

    receipt = machine.release(0, _feedback())
    assert receipt["update_count"] == 8
    assert receipt["shuffled_update_count"] == 8
    assert set(receipt["shuffled_label_origins"]) == {event_id for event_id, _label in _feedback()}
    assert machine.state_hash() != initial_hash
    machine.acknowledge(0)
    checkpoint = tmp_path / "state.json"
    machine.save(checkpoint)
    reloaded = exp.RecalibrationEventMachine.load(checkpoint)
    assert reloaded.to_payload() == machine.to_payload()
    assert reloaded.state_hash() == machine.state_hash()
    assert reloaded.numerical_state_bytes() > 0

    _predict_block(machine, 8, 1)
    _predict_block(reloaded, 8, 1)
    assert reloaded.predictions == machine.predictions
    receipt_a = machine.release(1, _feedback(8))
    receipt_b = reloaded.release(1, _feedback(8))
    assert receipt_a == receipt_b
    assert reloaded.state_hash() == machine.state_hash()


def test_duplicate_future_and_unknown_feedback_do_not_mutate() -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-LIFECYCLE fails closed."""

    machine = exp.RecalibrationEventMachine.create(exp.fixture_count_config())
    _predict_block(machine, 0, 0)
    with pytest.raises(ValueError, match="release_out_of_order"):
        machine.release(1, _feedback())
    assert machine.next_release_index == 0
    with pytest.raises(ValueError, match="feedback_event_unknown"):
        machine.release(0, [("future-event", 1)])
    assert machine.next_release_index == 0

    machine.release(0, _feedback())
    state = machine.state_hash()
    with pytest.raises(ValueError, match="duplicate_feedback_release"):
        machine.release(0, _feedback())
    assert machine.state_hash() == state
    with pytest.raises(ValueError, match="prediction_event_duplicate"):
        machine.predict("event-000", 0.1, 1)
    with pytest.raises(ValueError, match="prediction_release_already_closed"):
        machine.predict("late-event", 0.1, 0)
    with pytest.raises(ValueError, match="acknowledgment_without_release"):
        machine.acknowledge(2)
    with pytest.raises(ValueError, match="feedback_event_duplicate_in_release"):
        exp.RecalibrationEventMachine._canonical_feedback([("x", 0), ("x", 1)])
    with pytest.raises(ValueError, match="binary_label_required"):
        exp.RecalibrationEventMachine._canonical_feedback([("x", 2)])

    prerelease = exp.RecalibrationEventMachine.create(exp.fixture_count_config())
    prerelease.predict("future", 0.2, 1)
    with pytest.raises(ValueError, match="feedback_prerelease"):
        prerelease.release(0, [("future", 1)])


def test_state_payload_mutations_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-RESTART authenticates durable state."""

    machine = exp.RecalibrationEventMachine.create(exp.fixture_count_config())
    _predict_block(machine, 0, 0)
    machine.release(0, _feedback())
    machine.acknowledge(0)
    payload = machine.to_payload()

    changed = deepcopy(payload)
    changed["schema"] = "wrong"
    with pytest.raises(ValueError, match="state_schema_mismatch"):
        exp.RecalibrationEventMachine.from_payload(changed)
    changed = deepcopy(payload)
    changed["journal"][0]["entry_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="journal_hash_mismatch"):
        exp.RecalibrationEventMachine.from_payload(changed)
    changed = deepcopy(payload)
    changed["journal"][0]["sequence"] = 2
    with pytest.raises(ValueError, match="journal_chain_mismatch"):
        exp.RecalibrationEventMachine.from_payload(changed)
    changed = deepcopy(payload)
    changed["arms"].pop("local_count")
    with pytest.raises(ValueError, match="state_arm_set_mismatch"):
        exp.RecalibrationEventMachine.from_payload(changed)

    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="state_payload_not_object"):
        exp.RecalibrationEventMachine.load(path)


def test_fixture_panel_qualifies_only_circular_evidence(tmp_path: Path) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-FIXTURES."""

    panel = exp.run_fixture_panel(tmp_path)
    assert {row["fixture"] for row in panel["rows"]} == {
        "calibration_shift",
        "no_shift",
        "recurrence",
    }
    assert panel["fixture_gates_passed"] is True
    assert panel["parameter_change_count"] >= 2
    assert panel["monotonicity_violation_count"] == 0
    assert panel["movement_violation_count"] == 0
    assert panel["normalization_failure_count"] == 0
    assert panel["duplicate_feedback_rejection_count"] == 3
    assert panel["future_label_rejection_count"] == 3
    assert panel["restart_mismatch_count"] == 0
    shift = next(row for row in panel["rows"] if row["fixture"] == "calibration_shift")
    assert shift["constrained_brier"] < shift["raw_brier"]
    assert shift["brier_improvement"] > 0.0
    assert shift["unconstrained_maximum_movement"] > exp.MOVEMENT_BOUND
    assert panel["verifier_is_oracle"] is True


def test_numerical_qualification_has_dense_scalar_witnesses() -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-NUMERICAL."""

    qualification = exp.run_numerical_qualification()
    assert qualification["passed"] is True
    assert qualification["solver_pair_count"] >= 5
    assert qualification["dense_scalar_case_count"] >= 101
    assert qualification["convergence_failure_count"] == 0
    assert qualification["constraint_failure_count"] == 0
    assert qualification["solver_parity_failure_count"] == 0
    assert qualification["maximum_objective_delta"] <= exp.SOLVER_OBJECTIVE_TOLERANCE


def test_numerical_input_guards_and_unconstrained_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-NUMERICAL fails closed."""

    with pytest.raises(ValueError, match="theta_shape"):
        exp.map_probability(0.5, [0.0])
    with pytest.raises(ValueError, match="binary_energy_requires_two"):
        exp.normalized_probability([0.0])
    with pytest.raises(ValueError, match="probability_label_length"):
        exp.statistics_from_examples([0.2], [])
    with pytest.raises(ValueError, match="binary_label_required"):
        exp.statistics_from_examples([0.2], [2])
    assert exp.constraint_errors([0.0]) == ["theta_shape_or_finiteness"]
    assert "probability_range" in exp.constraint_errors([-0.2, *exp.KNOTS[1:]])
    assert "monotonicity" in exp.constraint_errors([0.0, 0.2, 0.1, *exp.KNOTS[3:]])
    assert "movement_bound" in exp.constraint_errors([0.2, *exp.KNOTS[1:]])

    gram = np.zeros((9, 9))
    target = np.zeros(9)
    with pytest.raises(ValueError, match="shape_invalid"):
        exp.solve_constrained_map(np.zeros((2, 2)), np.zeros(2))
    changed = gram.copy()
    changed[0, 0] = np.nan
    with pytest.raises(ValueError, match="not_finite"):
        exp.solve_constrained_map(changed, target)
    changed = gram.copy()
    changed[0, 1] = 1.0
    with pytest.raises(ValueError, match="gram_not_symmetric"):
        exp.solve_constrained_map(changed, target)
    with pytest.raises(ValueError, match="solver_method_not_frozen"):
        exp.solve_constrained_map(gram, target, method="wrong")
    with pytest.raises(ValueError, match="solver_configuration_not_frozen"):
        exp.SolverConfig(ridge_mass=7.0)
    with pytest.raises(ValueError, match="theta_shape"):
        exp.SufficientStatisticMap(gram, target, np.zeros(2), 0, set(), exp.SolverConfig())

    unconstrained = exp.SufficientStatisticMap.create(constrained=False)
    assert unconstrained.update_batch([("u", 0.2, 1)])["converged"] is True
    with pytest.raises(ValueError, match="sufficient_statistic_schema"):
        exp.SufficientStatisticMap.from_payload({"schema": "wrong"})

    learner = exp.SufficientStatisticMap.create()
    monkeypatch.setattr(
        exp,
        "solve_constrained_map",
        lambda _gram, _target: (exp.KNOTS.copy(), {"converged": False}),
    )
    with pytest.raises(RuntimeError, match="did_not_converge"):
        learner.update_batch([("failed", 0.2, 1)])


def test_frozen_protocol_is_label_blind_and_complete() -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-BENCHMARK."""

    protocol = exp.freeze_learning_protocol()
    assert protocol["online_group_count"] == 160
    assert protocol["retention_group_count"] == 80
    assert protocol["order_seeds"] == list(exp.ORDER_SEEDS)
    assert protocol["feedback_delay"] == protocol["release_block_size"] == 8
    assert protocol["retention_checkpoints"] == [0, 40, 80, 120, 160]
    assert protocol["bootstrap_replicates"] == 1000
    assert protocol["count_prior_mass"] == 8.0
    assert protocol["label_blind_freeze"] is True
    assert "labels" not in json.dumps(protocol).lower()
    roster = set(protocol["online_group_ids"])
    assert len(roster) == 160
    assert all(set(order) == roster for order in protocol["orders"].values())
    assert protocol["objective"]["ridge_mass"] == 8.0
    assert protocol["solver"]["tolerance"] == 1e-8


def test_complete_replay_benchmark_keeps_registered_units() -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-BENCHMARK keeps all units."""

    protocol = exp.freeze_learning_protocol()
    benchmark = exp.run_replay_benchmark(protocol, replicates=3)
    assert benchmark["bootstrap_replicates_planned"] == 3
    assert benchmark["bootstrap_replicates_completed"] == 3
    assert benchmark["order_replays_completed"] == 15
    assert benchmark["events_completed"] == 3 * 5 * 160
    assert benchmark["failed"] == 0
    assert benchmark["censored"] == 0
    assert benchmark["unstarted"] == 0
    assert benchmark["measured_duration_s"] >= 0.0
    assert benchmark["projected_duration_s"] >= 0.0
    assert benchmark["fits_2400_seconds_with_reserve"] is True

    completed: list[int] = []
    exp.run_replay_benchmark(protocol, replicates=1, progress_hook=completed.append)
    assert completed == [1]
    with pytest.raises(ValueError, match="replicates_must_be_positive"):
        exp.run_replay_benchmark(protocol, replicates=0)


def test_raw_sidecars_are_hash_bound(tmp_path: Path) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-ARTIFACT protects raw custody."""

    protocol = exp.freeze_learning_protocol()
    state_schema = exp.frozen_state_schema()
    receipts = exp.write_raw_sidecars(tmp_path, protocol, state_schema)
    assert set(receipts) == {"frozen_learning_protocol", "numerical_state_schema"}
    assert all(str(row["sha256"]).startswith("sha256:") for row in receipts.values())
    loaded = exp.read_raw_sidecar(tmp_path, receipts["frozen_learning_protocol"])
    assert loaded == protocol
    path = tmp_path / receipts["frozen_learning_protocol"]["path"]
    changed_receipt = deepcopy(receipts["frozen_learning_protocol"])
    changed_receipt["bytes"] += 1
    with pytest.raises(ValueError, match="raw_sidecar_size_mismatch"):
        exp.read_raw_sidecar(tmp_path, changed_receipt)

    path.write_text("[]\n", encoding="utf-8")
    changed_receipt["bytes"] = path.stat().st_size
    changed_receipt["sha256"] = exp.sha256_file(path)
    with pytest.raises(ValueError, match="raw_sidecar_not_object"):
        exp.read_raw_sidecar(tmp_path, changed_receipt)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="raw_sidecar_hash_mismatch"):
        exp.read_raw_sidecar(tmp_path, receipts["frozen_learning_protocol"])


def test_artifact_is_complete_circular_and_independently_reduced(
    complete_artifact: tuple[Path, dict[str, Any]],
) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-ARTIFACT separates benefit."""

    root, artifact = complete_artifact
    reduction = exp.validate_artifact(artifact, root=root, require_terminal=True)
    assert reduction["valid"] is True
    assert reduction["recalibration_ready"] is True
    assert reduction["learning_compute_feasible"] is True
    assert artifact["recalibration_ready_score"] == 1
    assert artifact["learning_compute_feasible_score"] == 1
    assert type(artifact["recalibration_ready_score"]) is int
    assert type(artifact["learning_compute_feasible_score"]) is int
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["verifier_is_oracle"] is True
    assert artifact["positive_claim"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["execution_venue"] == "host"
    assert artifact["maximum_prediction_movement"]["bound"] == 0.1
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["state_bytes"] > 0
    assert artifact["flagged_adversarial"] is False
    assert artifact["sample_size_budget"]["benchmark_order_replays"]["completed"] == 5000
    assert artifact["sample_size_budget"]["future_empirical_online_groups"]["unstarted"] == 160
    assert set(artifact) <= set(artifact["field_principles"])
    assert all(gate["principle"] for gate in artifact["acceptance_gate_results"])


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("run_date", "20260922", "identity_mismatch"),
        ("recalibration_ready_score", True, "score_not_bare_numeric"),
        ("verdict_class", "positive", "verdict_mismatch"),
        ("positive_claim", True, "positive_claim_for_fixture"),
        ("verifier_is_oracle", False, "oracle_declaration_missing"),
        ("MODEL_SPECS", [{}], "model_specs_not_empty"),
        ("model_invoked", True, "current_invocation_claim_invalid"),
        ("inference_substrate_class", "model_load_no_generation", "substrate_class_invalid"),
        ("execution_venue", "host_cpu", "execution_venue_invalid"),
        ("recalibration_ready_score", 0, "recalibration_ready_score_mismatch"),
        ("learning_compute_feasible_score", 0, "learning_compute_feasible_score_mismatch"),
        ("reproducibility_checksum", "sha256:changed", "checksum_mismatch"),
    ],
)
def test_artifact_mutations_fail_closed(
    complete_artifact: tuple[Path, dict[str, Any]], field: str, value: Any, error: str
) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-ARTIFACT detects changed claims."""

    root, baseline = complete_artifact
    artifact = deepcopy(baseline)
    artifact[field] = value
    with pytest.raises(ValueError, match=error):
        exp.validate_artifact(artifact, root=root, require_terminal=True)


def test_required_validation_failure_is_disqualified(tmp_path: Path) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-ARTIFACT makes invalid evidence unusable."""

    receipts = _receipts()
    receipts[0] = _receipt(receipts[0]["name"], passed=False)
    artifact = exp.build_test_artifact(tmp_path, validation_receipts=receipts)
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["honest_verdict"] == "complete_disqualified_required_validation"
    assert artifact["recalibration_ready_score"] == 0
    assert artifact["learning_compute_feasible_score"] == 1
    assert exp.validate_artifact(artifact, root=tmp_path, require_terminal=True)["valid"] is False


def test_valid_compute_null_and_artifact_annotation_guards(
    complete_artifact: tuple[Path, dict[str, Any]],
) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-ARTIFACT keeps a valid null reusable."""

    root, baseline = complete_artifact
    panel = {"rows": deepcopy(baseline["rows"]), **deepcopy(baseline["fixture_summary"])}
    benchmark = deepcopy(baseline["benchmark_receipt"])
    benchmark["fits_2400_seconds_with_reserve"] = False
    artifact = exp.build_artifact(
        panel=panel,
        numerical=baseline["numerical_qualification"],
        benchmark=benchmark,
        raw_sidecars=baseline["raw_sidecars"],
        validation_receipts=baseline["validation_receipts"],
        preconditions_checked=baseline["preconditions_checked"],
        source_hashes=baseline["source_artifact_hashes"],
        duration_s=0.2,
        phase_spans=[],
    )
    assert artifact["verdict_class"] == "null"
    assert artifact["recalibration_ready_score"] == 1
    assert artifact["learning_compute_feasible_score"] == 0
    assert exp.validate_artifact(artifact, root=root)["valid"] is True

    changed = deepcopy(baseline)
    changed["field_principles"].pop("rows")
    with pytest.raises(ValueError, match="field_principles_incomplete"):
        exp.validate_artifact(changed, root=root)
    changed = deepcopy(baseline)
    changed["acceptance_gate_results"][0]["principle"] = ""
    with pytest.raises(ValueError, match="gate_principle_missing"):
        exp.validate_artifact(changed, root=root)


def test_independent_reducer_rejects_raw_and_row_mutations(
    complete_artifact: tuple[Path, dict[str, Any]],
) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-ARTIFACT replays raw structure."""

    root, baseline = complete_artifact
    changed = deepcopy(baseline)
    changed["raw_sidecars"] = {}
    with pytest.raises(ValueError, match="raw_sidecar_set_mismatch"):
        exp.independent_reduce(changed, root=root)
    changed = deepcopy(baseline)
    changed["frozen_learning_protocol"]["protocol_hash"] = "sha256:changed"
    with pytest.raises(ValueError, match="protocol_hash_mismatch"):
        exp.independent_reduce(changed, root=root)
    changed = deepcopy(baseline)
    changed["raw_sidecars"]["numerical_state_schema"] = deepcopy(
        changed["raw_sidecars"]["frozen_learning_protocol"]
    )
    with pytest.raises(ValueError, match="numerical_state_schema_mismatch"):
        exp.independent_reduce(changed, root=root)
    changed = deepcopy(baseline)
    changed["rows"] = {}
    with pytest.raises(ValueError, match="fixture_rows_not_list"):
        exp.independent_reduce(changed, root=root)
    changed = deepcopy(baseline)
    changed["rows"].append("malformed")
    reduction = exp.independent_reduce(changed, root=root)
    assert reduction["fixture_lifecycle_passed"] is False


def test_cold_replay_rejects_unreadable_object(tmp_path: Path) -> None:
    """REQ-CL-7561 rejects malformed fresh-reader input."""

    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_unreadable_or_not_object"):
        exp.cold_replay(path, root=tmp_path)


def test_preconditions_name_exact_missing_upstream(tmp_path: Path) -> None:
    """REQ-CL-7561 preconditions block instead of fabricating data."""

    checks = exp.collect_preconditions(tmp_path)
    failed = next(row for row in checks if row["required"] and not row["passed"])
    assert failed["upstream"]
    assert failed["path_or_field"]
    assert failed["expected"] == "present_file"
    assert failed["observed"] == "missing"
    blocked = exp.build_blocked_artifact(failed, checks, duration_s=0.01)
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"] == failed
    assert blocked["rows"] == []
    assert blocked["sample_size_budget"]["benchmark_order_replays"]["unstarted"] == 5000
    assert blocked["recalibration_ready_score"] == 0
    assert blocked["learning_compute_feasible_score"] == 0


def test_repository_preconditions_include_spec_and_resources() -> None:
    """REQ-CL-7561 checks specifications and resources before measurement."""

    checks = exp.collect_preconditions(ROOT)
    assert checks
    assert all(row["passed"] for row in checks if row["required"])
    by_check = {row["check"]: row for row in checks}
    assert by_check["spec_requirement_present"]["observed"] is True
    assert by_check["scipy_available"]["observed"] is True
    assert by_check["logical_cpu_available"]["passed"] is True
    assert by_check["logical_cpu_available"]["observed"] >= 1
    assert (
        by_check["v660_historical_verdict_preserved"]["observed"]
        == "complete_null_count_claims_qualified_benefit_gate_failed"
    )
    assert exp.MODULE_PATH.as_posix() in exp._source_hashes(ROOT)


def test_validation_plan_is_scoped_and_private(tmp_path: Path) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-ARTIFACT freezes affected files."""

    commands = exp.build_validation_commands(ROOT, tmp_path)
    assert [command.name for command in commands] == list(validation_scope.REQUIRED_CHECK_NAMES)
    focused = next(command for command in commands if command.name == "focused_pytest")
    assert "-n" in focused.argv and "0" in focused.argv
    assert "-o" in focused.argv and "addopts=" in focused.argv
    assert "--no-cov" in focused.argv
    assert exp.TEST_PATH.as_posix() in focused.argv
    assert "tests/python" not in focused.argv
    coverage = next(command for command in commands if command.name == "changed_module_coverage")
    assert any(str(tmp_path / ".coverage.exp7561") in arg for arg in coverage.argv)

    terminal = exp.terminal_commands(tmp_path / "candidate.json", ROOT)
    assert [command.name for command in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert terminal[0].argv[-2:] == ("--cold-replay", str(tmp_path / "candidate.json"))
    assert terminal[1].argv[-2:] == ("--independent-reduce", str(tmp_path / "candidate.json"))
    assert terminal[2].argv[-1] == str(tmp_path / "candidate.json")
    assert terminal[3].argv[-1] == str(tmp_path / "candidate.json")
    manifest = tmp_path / "affected.json"
    exp._write_affected_manifest(manifest)
    assert json.loads(manifest.read_text())["experiment_id"] == exp.EXPERIMENT_ID


def test_cold_replay_rehashes_serialized_evidence(
    tmp_path: Path, complete_artifact: tuple[Path, dict[str, Any]]
) -> None:
    """REQ-CL-7561; SCENARIO-CL-7561-ARTIFACT uses a fresh-reader API."""

    root, baseline = complete_artifact
    artifact = deepcopy(baseline)
    path = tmp_path / "artifact.json"
    exp.atomic_json(path, artifact)
    reduction = exp.cold_replay(path, root=root, require_terminal=True)
    assert reduction["valid"] is True
    assert reduction == exp.independent_reduce(artifact, root=root, require_terminal=True)

    artifact["raw_sidecars"]["frozen_learning_protocol"]["sha256"] = "sha256:" + "0" * 64
    exp.atomic_json(path, artifact)
    with pytest.raises(ValueError, match="raw_sidecar_hash_mismatch"):
        exp.cold_replay(path, root=root, require_terminal=True)


def test_arguments_reject_wrong_date() -> None:
    """REQ-CL-7561 binds the exact run date."""

    assert exp.parse_args(["--date", exp.RUN_DATE]).date == exp.RUN_DATE
    with pytest.raises(ValueError, match="run_date_must_equal"):
        exp.main(["--date", "20260922"])

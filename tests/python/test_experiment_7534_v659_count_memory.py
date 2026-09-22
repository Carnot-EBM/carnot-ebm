"""Focused tests for REQ-CL-7534 frozen-bin count memory."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path

import pytest

import carnot.experiment_7534_v659_count_memory as exp
from carnot.experiment_7534_v659_count_memory import (
    AFFECTED_MANIFEST,
    CountArm,
    CountConfig,
    CountEventMachine,
    analytical_streams,
    bin_index,
    build_artifact,
    build_test_artifact,
    clip_probability,
    cold_replay,
    independent_reduce,
    main,
    normalized_probability,
    run_fixture_panel,
    run_restart_panel,
    validate_artifact,
)


MEANS = (0.05, 0.18, 0.30, 0.43, 0.57, 0.70, 0.82, 0.95)


def _config() -> CountConfig:
    return CountConfig(bin_means=MEANS, global_mean=0.5)


def _machine() -> CountEventMachine:
    return CountEventMachine.create(_config())


def _predict_release(machine: CountEventMachine, *, acknowledge: bool = True) -> None:
    machine.predict("r0-a", 0.30, release_index=0)
    machine.predict("r0-b", 0.70, release_index=0)
    machine.release(0, (("r0-a", 1), ("r0-b", 0)))
    if acknowledge:
        machine.acknowledge(0)


def test_exact_arithmetic_and_binary_energy() -> None:
    """REQ-CL-7534 and SCENARIO-CL-7534-ARITHMETIC use exact equations."""

    assert clip_probability(0.0) == 1e-4
    assert clip_probability(1.0) == 1.0 - 1e-4
    assert bin_index(0.0) == 0
    assert bin_index(1.0) == 7

    arm = CountArm.create("local", _config())
    before = arm.predict(0.30)
    assert before.probability == pytest.approx(0.30)
    assert before.energies == pytest.approx((0.0, -math.log(0.30 / 0.70)))
    assert normalized_probability(before.energies) == pytest.approx(before.probability)

    arm.update("event-1", 0.30, 1)
    posterior = (8.0 * MEANS[2] + 1.0) / 9.0
    expected_logit = math.log(0.30 / 0.70) + math.log(posterior / (1 - posterior))
    expected_logit -= math.log(MEANS[2] / (1 - MEANS[2]))
    assert arm.counts[2] == pytest.approx((3.4, 5.6))
    assert arm.predict(0.30).probability == pytest.approx(1 / (1 + math.exp(-expected_logit)))


def test_configuration_and_predictions_are_frozen() -> None:
    """SCENARIO-CL-7534-CHRONOLOGY freezes configuration and prediction rows."""

    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.kappa = 9.0  # type: ignore[misc]
    machine = CountEventMachine.create(config)
    row = machine.predict("sealed", 0.42, release_index=0)
    original = machine.predictions["sealed"]
    with pytest.raises(TypeError):
        row["local"]["probability"] = 0.99
    machine.release(0, (("sealed", 1),))
    assert machine.predictions["sealed"] == original


def test_global_and_local_states_have_distinct_support() -> None:
    """REQ-CL-7534 keeps one global pair and eight independent local pairs."""

    config = _config()
    global_arm = CountArm.create("global", config)
    local_arm = CountArm.create("local", config)
    high_global_before = global_arm.predict(0.90).probability
    high_local_before = local_arm.predict(0.90).probability
    global_arm.update("low", 0.10, 1)
    local_arm.update("low", 0.10, 1)
    assert global_arm.predict(0.90).probability > high_global_before
    assert local_arm.predict(0.90).probability == pytest.approx(high_local_before)
    assert global_arm.counts == [(5.0, 4.0)]
    assert len(local_arm.counts) == 8


def test_chronology_uses_release_order_not_adversarial_ids() -> None:
    """SCENARIO-CL-7534-CHRONOLOGY rejects future and prerelease feedback."""

    machine = _machine()
    machine.predict("z-future-name", 0.2, release_index=0)
    machine.predict("a-past-name", 0.8, release_index=1)
    with pytest.raises(ValueError, match="release_out_of_order"):
        machine.release(1, (("a-past-name", 1),))
    with pytest.raises(ValueError, match="feedback_event_unknown"):
        machine.release(0, (("never-predicted", 0),))
    machine.release(0, (("z-future-name", 0),))
    with pytest.raises(ValueError, match="feedback_prerelease"):
        machine.release(1, (("z-future-name", 0),))
    machine.release(1, (("a-past-name", 1),))
    assert machine.next_release_index == 2


def test_permuted_arm_uses_same_released_multiset() -> None:
    """SCENARIO-CL-7534-PERMUTATION preserves evidence and changes origins."""

    machine = _machine()
    for event_id, p0 in (("a", 0.1), ("b", 0.4), ("c", 0.9)):
        machine.predict(event_id, p0, release_index=0)
    receipt = machine.release(0, (("a", 1), ("b", 0), ("c", 1)))
    assert sorted(receipt["labels"]) == sorted(receipt["permuted_labels"])
    assert receipt["update_count"] == receipt["permuted_update_count"] == 3
    assert all(row["event_id"] != row["label_origin"] for row in receipt["permutation"])


def test_duplicate_feedback_is_idempotent_and_conflicts_fail() -> None:
    """SCENARIO-CL-7534-RESTART applies every released label exactly once."""

    machine = _machine()
    machine.predict("once", 0.3, release_index=0)
    first = machine.release(0, (("once", 1),))
    state_hash = machine.state_hash()
    assert machine.release(0, (("once", 1),)) == first
    assert machine.state_hash() == state_hash
    with pytest.raises(ValueError, match="duplicate_release_conflict"):
        machine.release(0, (("once", 0),))


def test_hash_chain_and_checkpoint_reject_corruption(tmp_path: Path) -> None:
    """SCENARIO-CL-7534-RESTART binds state and journal bytes across reload."""

    machine = _machine()
    _predict_release(machine)
    checkpoint = tmp_path / "state.json"
    machine.save(checkpoint)
    loaded = CountEventMachine.load(checkpoint)
    assert loaded.to_payload() == machine.to_payload()
    assert loaded.predict("later", 0.6, release_index=1) == machine.predict(
        "later", 0.6, release_index=1
    )

    corrupt = json.loads(checkpoint.read_text(encoding="utf-8"))
    corrupt["journal"][0]["entry_hash"] = "sha256:" + "0" * 64
    checkpoint.write_text(json.dumps(corrupt), encoding="utf-8")
    with pytest.raises(ValueError, match="journal_hash_mismatch"):
        CountEventMachine.load(checkpoint)


def test_all_restart_points_match_uninterrupted(tmp_path: Path) -> None:
    """SCENARIO-CL-7534-RESTART covers all four registered crash points."""

    rows = run_restart_panel(tmp_path)
    assert [row["crash_point"] for row in rows] == [
        "after_prediction",
        "after_release",
        "before_durable_acknowledgment",
        "after_durable_acknowledgment",
    ]
    assert all(row["prediction_parity"] for row in rows)
    assert all(row["exactly_once"] and row["journal_valid"] for row in rows)


def test_analytical_streams_are_complete_and_honest() -> None:
    """SCENARIO-CL-7534-CONTROLS limits analytical evidence to qualification."""

    streams = analytical_streams()
    assert {row["name"] for row in streams} == {
        "unchanged_information",
        "conditional_drift_stable_global_prevalence",
        "global_only_drift",
        "alternating_recurrence",
        "no_feedback",
        "all_one_class_labels",
        "adversarial_delayed_ids",
    }
    fixture_rows = run_fixture_panel()
    no_feedback = next(row for row in fixture_rows if row["stream"] == "no_feedback")
    assert no_feedback["update_count"] == 0
    assert no_feedback["local_state_changed"] is False
    assert all(row["complete"] for row in fixture_rows)
    assert all(row["evidence_class"] == "constructed_control" for row in fixture_rows)


def test_independent_reduction_separates_readiness_and_benefit(tmp_path: Path) -> None:
    """SCENARIO-CL-7534-ARTIFACT makes readiness independent of benefit."""

    artifact = build_test_artifact(tmp_path)
    reduced = independent_reduce(artifact)
    assert reduced == {
        "arithmetic_passed": True,
        "chronology_passed": True,
        "restart_passed": True,
        "fixtures_complete": True,
        "oracle_fixture": True,
        "count_memory_ready_score": 1,
        "verdict_class": "circular_positive",
        "positive_claim": False,
    }
    assert artifact["verifier_is_oracle"] is True
    assert artifact["positive_claim"] is False


def test_artifact_validation_fails_closed_on_raw_mutation(tmp_path: Path) -> None:
    """SCENARIO-CL-7534-ARTIFACT binds raw rows and readiness to checksum."""

    artifact = build_test_artifact(tmp_path)
    validate_artifact(artifact, require_validation=False)
    changed = json.loads(json.dumps(artifact))
    changed["fixture_rows"][0]["complete"] = False
    with pytest.raises(ValueError, match="independent_reduction_mismatch"):
        validate_artifact(changed, require_validation=False)
    changed = json.loads(json.dumps(artifact))
    changed["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        validate_artifact(changed, require_validation=False)


def test_build_artifact_declares_no_model_and_required_fields(tmp_path: Path) -> None:
    """REQ-CL-7534 declares truthful substrate, budgets, gates, and principles."""

    artifact = build_artifact(
        fixture_rows=run_fixture_panel(),
        restart_rows=run_restart_panel(tmp_path),
        validation_receipts=[],
        required_checks_passed=False,
        duration_s=0.25,
        phase_spans=[],
        source_hashes={"spec": "sha256:test"},
        preconditions_checked=[],
    )
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["inference_substrate"] == "verifier_ensemble_against_cached_candidates"
    assert artifact["count_memory_ready_score"] == 0
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["sample_size_budget"]["planned"] == 7
    assert set(artifact["field_principles"]) >= {
        "schema",
        "rows",
        "count_memory_ready_score",
        "hardware_path",
    }


def test_cold_readers_and_manifest_are_scoped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CL-7534-ARTIFACT uses exact scoped files and fresh readers."""

    artifact = build_test_artifact(tmp_path)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert cold_replay(candidate, require_validation=False)["count_memory_ready_score"] == 1
    assert main(["--cold-replay", str(candidate), "--allow-test-validation"]) == 0
    assert main(["--independent-reduce", str(candidate), "--allow-test-validation"]) == 0
    printed = capsys.readouterr().out
    assert '"count_memory_ready_score": 1' in printed
    assert AFFECTED_MANIFEST.test_paths == (
        "tests/python/test_experiment_7534_v659_count_memory.py",
    )
    assert AFFECTED_MANIFEST.changed_modules == (
        "python/carnot/experiment_7534_v659_count_memory.py",
    )


def test_input_and_count_arm_guards() -> None:
    """REQ-CL-7534 rejects invalid frozen settings, energies, and feedback."""

    with pytest.raises(ValueError, match="probability_not_finite"):
        clip_probability(math.nan)
    with pytest.raises(ValueError, match="binary_energy_requires_two_values"):
        normalized_probability((0.0,))
    with pytest.raises(ValueError, match="exactly_eight"):
        CountConfig((0.5,), 0.5)
    with pytest.raises(ValueError, match="prior_mass"):
        CountConfig(MEANS, 0.5, 9.0)
    with pytest.raises(ValueError, match="bin_mean"):
        CountConfig((0.0, *MEANS[1:]), 0.5)
    with pytest.raises(ValueError, match="global_mean"):
        CountConfig(MEANS, 1.0)
    with pytest.raises(ValueError, match="kind_invalid"):
        CountArm.create("other", _config())

    arm = CountArm.create("local", _config())
    with pytest.raises(ValueError, match="binary_label"):
        arm.update("bad", 0.2, 2)
    arm.update("once", 0.2, 1)
    with pytest.raises(ValueError, match="duplicate_feedback"):
        arm.update("once", 0.2, 1)
    with pytest.raises(ValueError, match="frozen_arm"):
        CountArm.create("frozen", _config()).update("x", 0.2, 1)


def test_event_machine_rejects_duplicate_and_malformed_operations() -> None:
    """SCENARIO-CL-7534-CHRONOLOGY rejects ambiguous event operations."""

    machine = _machine()
    machine.predict("x", 0.2, release_index=0)
    with pytest.raises(ValueError, match="prediction_event_duplicate"):
        machine.predict("x", 0.2, release_index=0)
    with pytest.raises(ValueError, match="acknowledgment_without_release"):
        machine.acknowledge(0)
    with pytest.raises(ValueError, match="feedback_event_duplicate_in_release"):
        machine.release(0, (("x", 0), ("x", 1)))
    with pytest.raises(ValueError, match="binary_label"):
        machine.release(0, (("x", 3),))
    machine.release(0, (("x", 0),))
    machine.acknowledge(0)
    machine.acknowledge(0)
    with pytest.raises(ValueError, match="release_already_closed"):
        machine.predict("past", 0.2, release_index=0)


def test_serialized_state_shape_guards(tmp_path: Path) -> None:
    """SCENARIO-CL-7534-RESTART rejects malformed schema, chain, and arm sets."""

    payload = _machine().to_payload()
    changed = json.loads(json.dumps(payload))
    changed["schema"] = "wrong"
    with pytest.raises(ValueError, match="state_schema"):
        CountEventMachine.from_payload(changed)

    machine = _machine()
    machine.predict("x", 0.2, release_index=0)
    changed = machine.to_payload()
    changed["journal"][0]["sequence"] = 2
    with pytest.raises(ValueError, match="journal_chain"):
        CountEventMachine.from_payload(changed)

    changed = _machine().to_payload()
    del changed["arms"]["global"]
    with pytest.raises(ValueError, match="state_arm_set"):
        CountEventMachine.from_payload(changed)

    invalid = tmp_path / "list.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="state_payload_not_object"):
        CountEventMachine.load(invalid)


def test_artifact_validation_guard_branches(tmp_path: Path) -> None:
    """SCENARIO-CL-7534-ARTIFACT rejects every changed terminal contract layer."""

    artifact = build_test_artifact(tmp_path)

    changed = json.loads(json.dumps(artifact))
    changed["model_invoked"] = True
    with pytest.raises(ValueError, match="field_invalid:model_invoked"):
        validate_artifact(changed, require_validation=False)

    changed = json.loads(json.dumps(artifact))
    changed["honest_verdict"] = "unfinished"
    changed["reproducibility_checksum"] = exp.canonical_hash(
        {key: value for key, value in changed.items() if key != "reproducibility_checksum"}
    )
    with pytest.raises(ValueError, match="terminal_verdict_prefix"):
        validate_artifact(changed, require_validation=False)

    changed = json.loads(json.dumps(artifact))
    changed["acceptance_gate_results"] = []
    with pytest.raises(ValueError, match="acceptance_gate_reduction"):
        validate_artifact(changed, require_validation=False)

    changed = json.loads(json.dumps(artifact))
    changed["acceptance_gate_results"][0]["principle"] = ""
    changed["reproducibility_checksum"] = exp.canonical_hash(
        {key: value for key, value in changed.items() if key != "reproducibility_checksum"}
    )
    with pytest.raises(ValueError, match="acceptance_gate_reduction"):
        validate_artifact(changed, require_validation=False)

    changed = json.loads(json.dumps(artifact))
    del changed["field_principles"]["rows"]
    with pytest.raises(ValueError, match="field_principles_incomplete"):
        validate_artifact(changed, require_validation=False)

    changed = json.loads(json.dumps(artifact))
    changed["validation_receipts"] = []
    changed["reproducibility_checksum"] = exp.canonical_hash(
        {key: value for key, value in changed.items() if key != "reproducibility_checksum"}
    )
    with pytest.raises(ValueError, match="required_validation_receipts_failed"):
        validate_artifact(changed)

    changed = json.loads(json.dumps(artifact))
    changed["preconditions_checked"] = [{"passed": False}]
    changed["reproducibility_checksum"] = exp.canonical_hash(
        {key: value for key, value in changed.items() if key != "reproducibility_checksum"}
    )
    with pytest.raises(ValueError, match="preconditions_failed"):
        validate_artifact(changed)

    with pytest.raises(ValueError, match="terminal_validation_receipts_failed"):
        validate_artifact(artifact, require_terminal=True)
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        validate_artifact(artifact, require_validation=False, verify_sources=True)


def test_blocked_artifact_names_exact_missing_operand() -> None:
    """REQ-CL-7534 publishes exact expected and observed blocked operands."""

    failed = exp._precondition(
        "missing_input",
        "upstream.json",
        "ready_score",
        1,
        None,
        False,
    )
    blocked = exp.build_blocked_artifact(failed, [failed], 0.01)
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == "complete_blocked_missing_input"
    assert blocked["gate_check_summary"]["first_failure"]["observed"] is None

    changed = json.loads(json.dumps(blocked))
    changed["inference_substrate_class"] = "no_model_load"
    with pytest.raises(ValueError, match="blocked_substrate_class"):
        validate_artifact(changed)
    changed = json.loads(json.dumps(blocked))
    changed["gate_check_summary"]["first_failure"] = {}
    with pytest.raises(ValueError, match="blocked_gate_summary"):
        validate_artifact(changed)
    changed = json.loads(json.dumps(blocked))
    changed["honest_verdict"] = "blocked"
    with pytest.raises(ValueError, match="blocked_verdict_prefix"):
        validate_artifact(changed)
    changed = json.loads(json.dumps(blocked))
    del changed["field_principles"]["rows"]
    with pytest.raises(ValueError, match="field_principles_incomplete"):
        validate_artifact(changed)
    changed = json.loads(json.dumps(blocked))
    changed["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        validate_artifact(changed)


def test_preconditions_manifest_and_terminal_helpers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7534 authenticates sources and freezes exact affected files."""

    checks, hashes = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert exp.SPEC_PATH.as_posix() in hashes
    assert any(row["required"] is False for row in checks)

    manifest = tmp_path / "manifest.json"
    exp._write_frozen_manifest(manifest)
    first = manifest.read_bytes()
    exp._write_frozen_manifest(manifest)
    assert manifest.read_bytes() == first
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest_changed"):
        exp._write_frozen_manifest(manifest)

    commands = exp._terminal_commands(tmp_path / "candidate.json")
    assert [row.name for row in commands] == [
        "fresh_process_cold_replay",
        "independent_raw_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    ]
    receipts = [
        {"name": row.name, "passed": True, "exit_code": 0, "timed_out": False} for row in commands
    ]
    assert exp._terminal_receipts_pass(receipts)
    receipts[0]["timed_out"] = True
    assert not exp._terminal_receipts_pass(receipts)

    started = exp.time.monotonic()
    span = exp._span("unit", started, started, 1)
    exp.progress(started, "unit", "complete", completed_units=1)
    assert span["completed_units"] == 1
    assert "phase=unit" in capsys.readouterr().out


def test_small_sanity_panel_and_file_helpers(tmp_path: Path) -> None:
    """SCENARIO-CL-7534-CONTROLS runs a bounded real update-cost panel."""

    panel = exp.run_numerical_sanity_panel(8)
    assert panel["operation_count"] == 8
    assert panel["pathological_update_cost"] is False
    sample = tmp_path / "sample.txt"
    sample.write_text("count-memory", encoding="utf-8")
    assert exp.sha256_file(sample).startswith("sha256:")
    assert exp._load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    assert exp._load_object(non_object) == {}
    with pytest.raises(ValueError, match="artifact_unreadable"):
        cold_replay(non_object)


def test_cli_errors_and_producer_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """SCENARIO-CL-7534-ARTIFACT keeps reader and producer exit states explicit."""

    with pytest.raises(SystemExit, match="--date must"):
        main(["--date", "20260101"])
    with pytest.raises(ValueError, match="artifact_unreadable"):
        main(["--independent-reduce", str(tmp_path / "missing.json")])

    artifact = build_test_artifact(tmp_path)
    monkeypatch.setattr(exp, "run_experiment", lambda _root, _date: artifact)
    assert main([]) == 0
    assert '"count_memory_ready_score": 1' in capsys.readouterr().out

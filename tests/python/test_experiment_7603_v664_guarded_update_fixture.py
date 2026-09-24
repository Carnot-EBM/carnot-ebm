"""Tests for the bounded guarded-update lifecycle.

Spec refs: REQ-CL-7603 and SCENARIO-CL-7603-*.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from carnot import experiment_7603_v664_guarded_update_fixture as exp


def _write_inputs(root: Path) -> None:
    """Create a small hash-bound copy of the two allowed historical roles."""

    role_path = root / exp.CACHED_ROLES_PATH
    role_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(16):
        rows.append(
            {
                "role": "fit",
                "source_id": f"fit-{index:03d}",
                "probability": 0.35 + 0.02 * (index % 8),
                "label": index % 2,
            }
        )
    order = []
    for index in range(80):
        source_id = f"online-{index:03d}"
        order.append(source_id)
        rows.append(
            {
                "role": "online",
                "source_id": source_id,
                "probability": 0.15 + 0.7 * ((index % 10) / 9),
                "label": int(index % 4 in (1, 2)),
            }
        )
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    role_path.write_text(encoded, encoding="utf-8")
    protocol_path = root / exp.FROZEN_PROTOCOL_PATH
    protocol_path.write_text(
        json.dumps(
            {
                "feedback_delay": 8,
                "orders": {str(exp.STREAM_SEED): order},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    upstream = {
        "honest_verdict": "complete_null_cached_protocol",
        "verdict_class": "null",
        "frozen_protocol": {"feedback_delay": 8, "orders": {str(exp.STREAM_SEED): order}},
        "raw_sidecars": {
            "cached_roles": {
                "path": exp.CACHED_ROLES_PATH.as_posix(),
                "rows": len(rows),
                "sha256": exp.sha256_file(role_path),
            },
            "frozen_protocol": {
                "path": exp.FROZEN_PROTOCOL_PATH.as_posix(),
                "sha256": exp.sha256_file(protocol_path),
            },
        },
    }
    artifact_path = root / exp.UPSTREAM_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(upstream, sort_keys=True), encoding="utf-8")


def _example(label: int, value: float = 1.0) -> dict[str, object]:
    return {
        "event_id": f"example-{label}-{value}",
        "base_probability": 0.5,
        "features": [value, -value],
        "label": label,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"feature_count": 0},
        {"feature_count": 9},
        {"hidden_units": 0},
        {"hidden_units": 9},
        {"step_size": -0.1},
        {"parameter_bound": 0.0},
        {"residual_bound": 0.0},
        {"residual_bound": math.inf},
        {"anchor_tolerance": -0.1},
        {"lag": 0},
    ],
)
def test_config_rejects_values_outside_req_cl_7603(change: dict[str, object]) -> None:
    """REQ-CL-7603 bounds capacity, update size, parameters, and lag."""

    with pytest.raises(ValueError):
        exp.UpdateConfig(**change)


def test_normalized_energy_head_is_finite_and_pure() -> None:
    """REQ-CL-7603 uses a normalized binary head without generator mutation."""

    config = exp.UpdateConfig()
    state = exp.initial_parameters(config)
    original = state.to_payload()
    head = exp.ResidualEnergyHead(config, state)
    probability = head.predict(0.4, [0.2, -0.3])
    energies = head.energies(0.4, [0.2, -0.3])
    normalized = math.exp(-energies[1]) / sum(math.exp(-item) for item in energies)
    assert probability == pytest.approx(normalized)
    assert 0.0 < probability < 1.0
    assert state.to_payload() == original
    with pytest.raises(ValueError, match="feature_count"):
        head.predict(0.4, [0.2])
    with pytest.raises(ValueError, match="base_probability"):
        head.predict(1.0, [0.2, -0.3])


def test_proposal_gradient_clipping_and_zero_step_are_distinct() -> None:
    """SCENARIO-CL-7603-CONTROLS keeps clipping and no-update controls distinct."""

    fit = [_example(1, 0.2 + index) for index in range(4)]
    config = exp.UpdateConfig(step_size=0.2)
    state = exp.initial_parameters(config)
    proposal = exp.propose_update(config, state, fit)
    assert proposal.changed is True
    assert proposal.clipped is False
    assert state == exp.initial_parameters(config)
    assert exp.parameters_are_finite_and_bounded(config, proposal.candidate)

    clipped_config = exp.UpdateConfig(step_size=100.0, parameter_bound=0.01)
    clipped = exp.propose_update(clipped_config, exp.initial_parameters(clipped_config), fit)
    assert clipped.changed is True
    assert clipped.clipped is True

    zero_config = exp.UpdateConfig(step_size=0.0)
    zero = exp.propose_update(zero_config, exp.initial_parameters(zero_config), fit)
    assert zero.changed is False
    assert zero.clipped is False
    with pytest.raises(ValueError, match="four_fit_examples"):
        exp.propose_update(config, state, fit[:3])
    bad = [*fit[:3], {**fit[3], "label": 2}]
    with pytest.raises(ValueError, match="binary_label"):
        exp.propose_update(config, state, bad)


def test_guard_accepts_helpful_and_rejects_harmful_proposals() -> None:
    """SCENARIO-CL-7603-UPDATE admits only held-out-safe Brier proposals."""

    config = exp.UpdateConfig(step_size=0.2)
    prior = exp.initial_parameters(config)
    fit = [_example(1, 0.2 + index) for index in range(4)]
    proposal = exp.propose_update(config, prior, fit)
    anchors = [_example(index % 2, index / 16) for index in range(16)]
    helpful = exp.evaluate_proposal(config, proposal, fit, anchors)
    harmful = exp.evaluate_proposal(
        config, proposal, [_example(0, 0.2 + index) for index in range(4)], anchors
    )
    assert helpful.accepted is True
    assert all(helpful.checks.values())
    assert harmful.accepted is False
    assert harmful.checks["lower_admission_brier"] is False
    assert harmful.candidate_hash != harmful.prior_hash


def _released_lifecycle(
    tmp_path: Path,
    *,
    arm: str = "guarded",
    fit_label: int = 1,
    admission_label: int = 1,
) -> tuple[exp.UpdateLifecycle, list[str]]:
    config = exp.UpdateConfig(step_size=0.2)
    anchors = [_example(index % 2, index / 16) for index in range(16)]
    machine = exp.UpdateLifecycle(
        config=config,
        arm=arm,
        anchors=anchors,
        state_path=tmp_path / f"{arm}.json",
    )
    event_ids = []
    suffixes = ("z", "a", "y", "b", "x", "c", "w", "d")
    for index in range(8):
        event_id = f"block-0-event-{suffixes[index]}"
        event_ids.append(event_id)
        receipt = machine.predict(event_id, 0.5, [0.2 + index, -0.1], index)
        if index == 0:
            receipt["probability"] = -1.0
    assert machine.prediction_receipts[event_ids[0]]["probability"] > 0.0
    with pytest.raises(ValueError, match="feedback_too_early"):
        machine.release(event_ids[0], fit_label, 7)
    with pytest.raises(PermissionError, match="evaluator_read_denied"):
        machine.release(event_ids[0], fit_label, 8, role="evaluation")
    for index, event_id in enumerate(event_ids):
        label = fit_label if index < 4 else admission_label
        machine.release(event_id, label, index + config.lag)
    return machine, event_ids


def test_lifecycle_is_exactly_once_atomic_and_restart_safe(tmp_path: Path) -> None:
    """SCENARIO-CL-7603-LIFECYCLE covers causal release and durable parity."""

    machine, event_ids = _released_lifecycle(tmp_path)
    with pytest.raises(ValueError, match="duplicate_prediction"):
        machine.predict(event_ids[0], 0.5, [0.0, 0.0], 99)
    with pytest.raises(ValueError, match="duplicate_release"):
        machine.release(event_ids[0], 1, 99)
    with pytest.raises(ValueError, match="unknown_event"):
        machine.release("missing", 1, 99)
    before = machine.parameter_hash
    proposal = machine.propose("block-0", event_ids)
    outcome = machine.admit(proposal, event_ids)
    assert outcome["accepted"] is True
    assert machine.parameter_hash != before
    machine.persist()
    reloaded = exp.UpdateLifecycle.reload(
        state_path=machine.state_path,
        config=machine.config,
        arm="guarded",
        anchors=machine.anchors,
    )
    assert reloaded.parameter_hash == machine.parameter_hash
    assert reloaded.prediction_receipts == machine.prediction_receipts
    assert reloaded.released_event_ids == machine.released_event_ids
    assert reloaded.accepted_update_count == 1


def test_rejection_frozen_and_unguarded_share_one_interface(tmp_path: Path) -> None:
    """REQ-CL-7603 exposes frozen, unguarded, and guarded arms through one API."""

    guarded, events = _released_lifecycle(tmp_path / "guarded", fit_label=1, admission_label=0)
    before = guarded.parameter_hash
    rejected = guarded.admit(guarded.propose("harmful", events), events)
    assert rejected["accepted"] is False
    assert guarded.parameter_hash == before
    guarded.persist()
    assert (
        exp.UpdateLifecycle.reload(
            state_path=guarded.state_path,
            config=guarded.config,
            arm="guarded",
            anchors=guarded.anchors,
        ).parameter_hash
        == before
    )

    frozen, frozen_events = _released_lifecycle(tmp_path / "frozen", arm="frozen")
    frozen_before = frozen.parameter_hash
    frozen_result = frozen.admit(frozen.propose("frozen", frozen_events), frozen_events)
    assert frozen_result["reason"] == "frozen_predictor"
    assert frozen.parameter_hash == frozen_before

    unguarded, unguarded_events = _released_lifecycle(
        tmp_path / "unguarded", arm="unguarded", fit_label=1, admission_label=0
    )
    unsafe = unguarded.admit(unguarded.propose("unguarded", unguarded_events), unguarded_events)
    assert unsafe["accepted"] is True
    assert unsafe["guard_passed"] is False


def test_lifecycle_rejects_invalid_or_stale_operations(tmp_path: Path) -> None:
    """SCENARIO-CL-7603-LIFECYCLE fails closed on malformed lifecycle calls."""

    machine, event_ids = _released_lifecycle(tmp_path)
    with pytest.raises(ValueError, match="eight_event_block"):
        machine.propose("short", event_ids[:7])
    proposal = machine.propose("block", event_ids)
    machine.admit(proposal, event_ids)
    with pytest.raises(ValueError, match="stale_proposal"):
        machine.admit(proposal, event_ids)
    with pytest.raises(ValueError, match="event_order_mismatch"):
        machine.propose("reordered", list(reversed(event_ids)))
    with pytest.raises(ValueError, match="finite_features"):
        machine.predict("nan", 0.5, [math.nan, 0.0], 20)


def test_fixture_runs_ten_delayed_blocks_and_all_controls(tmp_path: Path) -> None:
    """SCENARIO-CL-7603-CAUSAL and CONTROLS run the complete finite fixture."""

    root = tmp_path / "repo"
    _write_inputs(root)
    checks, sources = exp.collect_preconditions(root)
    assert all(row["passed"] for row in checks)
    stream, anchors = exp.load_authenticated_inputs(root)
    evidence = exp.run_fixture(stream, anchors, tmp_path / "states")
    assert len(stream) == 80
    assert len(anchors) == 16
    assert len(evidence["rows"]) == 80 * len(exp.ARMS)
    assert len(evidence["block_results"]) == 10 * len(exp.ARMS)
    assert evidence["causal_timing_passed"] is True
    assert evidence["restart_parity"] is True
    assert evidence["accepted_control_passed"] is True
    assert evidence["rejected_control_passed"] is True
    assert evidence["harmful_update_rejected"] is True
    assert evidence["duplicate_rejected"] is True
    assert evidence["evaluator_read_denied"] is True
    assert evidence["deranged_control"]["within_block_only"] is True
    assert evidence["deranged_control"]["release_times_preserved"] is True
    assert evidence["deranged_control"]["label_counts_preserved"] is True
    assert evidence["control_results"]["clipped"]["clipped"] is True
    assert evidence["control_results"]["no_update"]["changed"] is False
    assert evidence["timing_ns"]["inference"] > 0
    assert evidence["timing_ns"]["update"] > 0
    assert sources


def test_preconditions_fail_closed_with_exact_gate_summary(tmp_path: Path) -> None:
    """REQ-CL-7603 publishes a complete blocked record for missing inputs."""

    checks, sources = exp.collect_preconditions(tmp_path)
    failed = next(row for row in checks if not row["passed"])
    blocked = exp.build_blocked_artifact(failed, checks, sources, duration_s=0.01)
    assert blocked["honest_verdict"].startswith("complete_blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"]["first_failure"] == {
        key: failed[key]
        for key in ("check", "upstream", "path", "field", "operator", "expected", "observed")
    }
    assert exp.validate_artifact(blocked, root=tmp_path)["blocked"] is True

    root = tmp_path / "changed"
    _write_inputs(root)
    role_path = root / exp.CACHED_ROLES_PATH
    role_path.write_text(role_path.read_text() + "{}\n", encoding="utf-8")
    changed_checks, _sources = exp.collect_preconditions(root)
    assert any(
        row["check"] == "cached_roles_sha256" and not row["passed"] for row in changed_checks
    )


def test_artifact_reduction_readers_and_mutations(tmp_path: Path) -> None:
    """SCENARIO-CL-7603-TERMINAL makes raw rows and readers authoritative."""

    root = tmp_path / "repo"
    _write_inputs(root)
    artifact = exp.build_test_artifact(root, tmp_path / "work")
    reduction = exp.validate_artifact(artifact, root=root, require_terminal=True)
    assert reduction["row_count"] == len(artifact["rows"])
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["guarded_update_ready_score"] == 1
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["planned_inference_substrate_class"] == "no_model_load"
    assert artifact["acceptance_gate_results"]["benefit"]["result"] is False
    assert artifact["acceptance_gate_results"]["retention"]["result"] is False
    assert artifact["verifier_is_oracle"] is True
    assert artifact["hardware_path"]["measured_speedup"] is None
    assert set(exp.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert artifact["sample_size_budget"]["observed_independent_units"] == 80
    assert artifact["generator_weights_immutable"] is True

    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path, root=root)["row_count"] == len(artifact["rows"])
    assert exp.independent_reduce_artifact(path, root=root) == artifact["independent_reduction"]

    row_mutation = deepcopy(artifact)
    row_mutation["rows"][0]["raw_squared_error_numerator"] += 0.1
    row_mutation["reproducibility_checksum"] = exp.reproducibility_checksum(row_mutation)
    with pytest.raises(ValueError, match="row_arithmetic"):
        exp.validate_artifact(row_mutation, root=root)

    readiness_mutation = deepcopy(artifact)
    readiness_mutation["guarded_update_ready_score"] = 0
    readiness_mutation["reproducibility_checksum"] = exp.reproducibility_checksum(
        readiness_mutation
    )
    with pytest.raises(ValueError, match="readiness_score"):
        exp.validate_artifact(readiness_mutation, root=root)

    generator_mutation = deepcopy(artifact)
    generator_mutation["generator_weights_immutable"] = False
    generator_mutation["reproducibility_checksum"] = exp.reproducibility_checksum(
        generator_mutation
    )
    with pytest.raises(ValueError, match="generator_weights"):
        exp.validate_artifact(generator_mutation, root=root)

    receipt_mutation = deepcopy(artifact)
    receipt_mutation["validation_receipts"] = []
    receipt_mutation["reproducibility_checksum"] = exp.reproducibility_checksum(receipt_mutation)
    with pytest.raises(ValueError, match="terminal_validation"):
        exp.validate_artifact(receipt_mutation, root=root, require_terminal=True)


def test_manifest_commands_and_cli_read_modes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7603 freezes affected scope and exposes fresh-process readers."""

    manifest = tmp_path / "manifest.json"
    exp.write_affected_manifest(manifest)
    value = json.loads(manifest.read_text(encoding="utf-8"))
    assert value["test_paths"] == [exp.TEST_PATH.as_posix()]
    assert value["changed_modules"] == [exp.MODULE_PATH.as_posix()]
    commands = exp.build_validation_commands(Path.cwd(), tmp_path / "private")
    assert [row.name for row in commands] == list(exp.validation_scope.REQUIRED_CHECK_NAMES)
    terminal = exp.terminal_commands(tmp_path / "candidate.json", Path.cwd())
    assert [row.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)

    root = tmp_path / "repo"
    _write_inputs(root)
    artifact = exp.build_test_artifact(root, tmp_path / "fixture")
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    assert exp.main(["--root", str(root), "--cold-replay", str(candidate)]) == 0
    assert "row_count" in capsys.readouterr().out
    assert exp.main(["--root", str(root), "--independent-reduce", str(candidate)]) == 0
    assert "arm_summaries" in capsys.readouterr().out
    args = exp.parse_args(["--date", exp.RUN_DATE])
    assert args.date == exp.RUN_DATE


def test_metric_reducer_rejects_incomplete_or_duplicate_rows() -> None:
    """SCENARIO-CL-7603-TERMINAL rejects aggregate-only evidence."""

    row = exp.metric_row("unit", "guarded", 0.25, 0, 7)
    reduction = exp.reduce_rows([row])
    assert reduction["arm_summaries"]["guarded"]["mean_brier"] == pytest.approx(0.0625)
    with pytest.raises(ValueError, match="duplicate_unit_arm"):
        exp.reduce_rows([row, row])
    with pytest.raises(ValueError, match="row_required_field"):
        exp.reduce_rows([{key: value for key, value in row.items() if key != "seed"}])
    with pytest.raises(ValueError, match="row_arithmetic"):
        exp.reduce_rows([{**row, "brier": 0.5}])


def test_defensive_math_and_lifecycle_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-CL-7603 names malformed math and lifecycle inputs before mutation."""

    config = exp.UpdateConfig()
    prior = exp.initial_parameters(config)
    with pytest.raises(ValueError, match="nonempty_examples"):
        exp._mean_metrics(config, prior, [])
    proposal = exp.propose_update(config, prior, [_example(1) for _ in range(4)])
    anchors = [_example(index % 2, index / 16) for index in range(16)]
    with pytest.raises(ValueError, match="four_admission"):
        exp.evaluate_proposal(config, proposal, [_example(1)] * 3, anchors)
    with pytest.raises(ValueError, match="sixteen_anchor"):
        exp.evaluate_proposal(config, proposal, [_example(1)] * 4, anchors[:15])
    with pytest.raises(ValueError, match="arm_invalid"):
        exp.UpdateLifecycle(
            config=config, arm="bad", anchors=anchors, state_path=tmp_path / "bad.json"
        )
    with pytest.raises(ValueError, match="sixteen_anchor"):
        exp.UpdateLifecycle(
            config=config,
            arm="guarded",
            anchors=anchors[:15],
            state_path=tmp_path / "bad-anchor.json",
        )
    machine = exp.UpdateLifecycle(
        config=config,
        arm="guarded",
        anchors=anchors,
        state_path=tmp_path / "state.json",
    )
    event_ids = []
    for index in range(8):
        event_id = f"event-{index}"
        event_ids.append(event_id)
        machine.predict(event_id, 0.5, [0.1, 0.2], index)
    with pytest.raises(ValueError, match="binary_label"):
        machine.release(event_ids[0], 2, 8)
    for index, event_id in enumerate(event_ids[:7]):
        machine.release(event_id, 1, index + 8)
    with pytest.raises(ValueError, match="not_fully_released"):
        machine.propose("incomplete", event_ids)
    machine.release(event_ids[7], 1, 15)
    ready = machine.propose("complete", event_ids)
    with pytest.raises(ValueError, match="eight_event_block"):
        machine.admit(ready, event_ids[:7])


def test_snapshot_reload_detects_every_integrity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7603-LIFECYCLE makes corrupt snapshots and receipts fail closed."""

    machine, _events = _released_lifecycle(tmp_path)
    machine.persist()
    original = json.loads(machine.state_path.read_text(encoding="utf-8"))

    def expect_reload_error(change: dict[str, object], message: str) -> None:
        payload = deepcopy(original)
        payload.update(change)
        machine.state_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            exp.UpdateLifecycle.reload(
                state_path=machine.state_path,
                config=machine.config,
                arm="guarded",
                anchors=machine.anchors,
            )

    expect_reload_error({"schema": "bad"}, "state_schema")
    expect_reload_error({"arm": "frozen"}, "state_contract")
    expect_reload_error({"parameter_hash": "bad"}, "parameter_hash")
    receipt_payload = deepcopy(original)
    first = next(iter(receipt_payload["predictions"].values()))
    first["receipt_hash"] = "bad"
    machine.state_path.write_text(json.dumps(receipt_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt_hash"):
        exp.UpdateLifecycle.reload(
            state_path=machine.state_path,
            config=machine.config,
            arm="guarded",
            anchors=machine.anchors,
        )

    real_atomic = exp.atomic_json

    def corrupt_atomic(path: Path, value: Mapping[str, object]) -> None:
        real_atomic(path, {**value, "corrupt": True})

    monkeypatch.setattr(exp, "atomic_json", corrupt_atomic)
    with pytest.raises(OSError, match="snapshot_reload"):
        machine.persist()


def test_input_and_fixture_shape_errors_fail_before_updates(tmp_path: Path) -> None:
    """SCENARIO-CL-7603-CAUSAL rejects unauthenticated or incomplete streams."""

    with pytest.raises(ValueError, match="external_precondition"):
        exp.load_authenticated_inputs(tmp_path)
    root = tmp_path / "repo"
    _write_inputs(root)
    upstream_path = root / exp.UPSTREAM_PATH
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream["frozen_protocol"]["orders"][str(exp.STREAM_SEED)][0] = "absent"
    upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
    with pytest.raises(ValueError, match="role_membership"):
        exp.load_authenticated_inputs(root)

    _write_inputs(root)
    role_path = root / exp.CACHED_ROLES_PATH
    rows = [json.loads(line) for line in role_path.read_text().splitlines()]
    rows[-1]["probability"] = 1.0
    role_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    upstream["raw_sidecars"]["cached_roles"]["sha256"] = exp.sha256_file(role_path)
    upstream_path.write_text(json.dumps(upstream), encoding="utf-8")
    with pytest.raises(ValueError, match="probability_or_label"):
        exp.load_authenticated_inputs(root)
    with pytest.raises(ValueError, match="80_stream"):
        exp.run_fixture([], [], tmp_path / "states")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema": "bad"}, "identity"),
        ({"honest_verdict": "unfinished"}, "terminal_prefix"),
        ({"verdict_class": "positive"}, "verdict_class"),
        ({"MODEL_SPECS": [{"name": "forbidden"}]}, "model_specs"),
        ({"no_model_load": False}, "no_model_contract"),
        ({"field_principles": {}}, "field_principles"),
        ({"rows": None}, "rows_missing"),
        ({"independent_reduction": {}}, "independent_reduction"),
        ({"validation_receipts": None}, "validation_receipts"),
        (
            {
                "sample_size_budget": {
                    "observed_independent_units": 79,
                }
            },
            "sample_size_budget",
        ),
        ({"current_work_receipt": None}, "current_work_receipt"),
        ({"acceptance_gate_results": {}}, "acceptance_gates"),
    ],
)
def test_artifact_validator_rejects_schema_mutations(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    """SCENARIO-CL-7603-TERMINAL rejects each governed-field mutation."""

    root = tmp_path / "repo"
    _write_inputs(root)
    artifact = exp.build_test_artifact(root, tmp_path / "work")
    artifact.update(change)
    artifact["reproducibility_checksum"] = exp.reproducibility_checksum(artifact)
    with pytest.raises(ValueError, match=message):
        exp.validate_artifact(artifact, root=root)


def test_artifact_readers_reject_bad_bytes_and_reductions(tmp_path: Path) -> None:
    """SCENARIO-CL-7603-TERMINAL does not trust unreadable or stale candidates."""

    root = tmp_path / "repo"
    _write_inputs(root)
    artifact = exp.build_test_artifact(root, tmp_path / "work")
    wrong_checksum = deepcopy(artifact)
    wrong_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        exp.validate_artifact(wrong_checksum, root=root)

    blocked = exp.build_blocked_artifact({"check": "x"}, [], [], duration_s=0.1)
    blocked["honest_verdict"] = "bad"
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    with pytest.raises(ValueError, match="blocked_gate"):
        exp.validate_artifact(blocked, root=root)

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        exp.cold_replay(empty, root=root)
    with pytest.raises(ValueError, match="unreadable"):
        exp.independent_reduce_artifact(empty, root=root)

    candidate = tmp_path / "candidate.json"
    changed = deepcopy(artifact)
    changed["independent_reduction"] = {}
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    exp.atomic_json(candidate, changed)
    with pytest.raises(ValueError, match="independent_reduction"):
        exp.independent_reduce_artifact(candidate, root=root)


def test_preterminal_command_span_and_producer_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7603 covers the producer dispatch without running subprocesses in unit tests."""

    command = exp.terminal_commands(tmp_path / "candidate.json", Path.cwd(), allow_preterminal=True)
    assert "--allow-preterminal" in command[0].argv
    span = exp._span("unit", 1.0, 0.0, 2)
    assert span["completed_units"] == 2
    called: dict[str, object] = {}

    def fake_run(root: Path, date: str, *, output_path: Path | None = None) -> dict[str, object]:
        called.update(root=root, date=date, output_path=output_path)
        return {}

    monkeypatch.setattr(exp, "run_experiment", fake_run)
    relative = Path("custom.json")
    assert exp.main(["--root", str(tmp_path), "--output", str(relative)]) == 0
    assert called["root"] == tmp_path.resolve()
    assert called["date"] == exp.RUN_DATE
    assert called["output_path"] == tmp_path.resolve() / relative

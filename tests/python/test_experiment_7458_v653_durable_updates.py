"""Tests for REQ-CL-7458 and SCENARIO-CL-7458-*.

The tests use private local directories. They do not change production storage
or claim performance from an accelerator.
"""

from __future__ import annotations

from copy import deepcopy
import errno
import json
from pathlib import Path

import pytest

from carnot import experiment_7458_v653_durable_updates as exp


def _apply(events: list[dict]) -> dict:
    state = exp.initial_state()
    for event in events:
        state = exp.apply_event(state, event)
    return state


def test_fixed_numeric_stream_has_four_experts_and_exact_predictions() -> None:
    """REQ-CL-7458: one fixed state and event stream control both stores."""

    events = exp.fixed_events(128)
    assert tuple(exp.initial_state()["experts"]) == exp.EXPERT_NAMES
    assert [row["sequence"] for row in events] == list(range(1, 129))
    first = _apply(events[:1])
    again = _apply(events[:1])
    assert first == again
    assert exp.predict_state(first, events[1]["features"]) == exp.predict_state(
        again, events[1]["features"]
    )
    with pytest.raises(ValueError, match="sequence_gap"):
        exp.apply_event(exp.initial_state(), events[1])


@pytest.mark.parametrize("store_type", [exp.DeltaJournalStore, exp.WholeStateStore])
def test_acknowledged_updates_restart_once_with_snapshot_parity(
    tmp_path: Path, store_type: type
) -> None:
    """SCENARIO-CL-7458-ACK: success is durable and replay is exactly once."""

    directory = tmp_path / store_type.__name__
    store = store_type(directory)
    events = exp.fixed_events(64)
    receipts = [store.acknowledge(event) for event in events]
    assert all(row["acknowledged"] for row in receipts)
    recovered = store_type(directory).recover()
    expected = _apply(events)
    assert recovered["state"] == expected
    assert recovered["state"]["sequence"] == 64
    assert recovered["evidence"]["acknowledged_loss_count"] == 0
    assert exp.predict_state(recovered["state"], events[-1]["features"]) == exp.predict_state(
        expected, events[-1]["features"]
    )
    if store_type is exp.DeltaJournalStore:
        assert store.snapshot_path.is_file()
        assert receipts[-1]["snapshot_written"] is True
        assert receipts[-1]["journal_fsync_count"] == 1
    else:
        assert receipts[-1]["file_fsync_count"] == 1
        assert receipts[-1]["directory_fsync_count"] == 1


def test_torn_tail_is_truncated_with_evidence(tmp_path: Path) -> None:
    """SCENARIO-CL-7458-FAULTS: a torn unacknowledged tail cannot enter state."""

    store = exp.DeltaJournalStore(tmp_path / "torn")
    event = exp.fixed_events(1)[0]
    assert store.acknowledge(event)["acknowledged"] is True
    with store.journal_path.open("ab") as stream:
        stream.write(b'{"schema":"torn"')
        stream.flush()
    recovered = exp.DeltaJournalStore(store.directory).recover()
    assert recovered["state"] == _apply([event])
    assert recovered["evidence"]["tail_action"] == "truncated"
    assert recovered["evidence"]["truncated_bytes"] > 0
    assert store.journal_path.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("store_type", [exp.DeltaJournalStore, exp.WholeStateStore])
def test_disk_full_and_duplicate_delivery_preserve_state(tmp_path: Path, store_type: type) -> None:
    """SCENARIO-CL-7458-DISK-DUPLICATE: failure and retry do not double apply."""

    directory = tmp_path / f"capacity-{store_type.__name__}"
    event = exp.fixed_events(1)[0]
    store = store_type(directory)
    acknowledged = store.acknowledge(event)
    before = store_type(directory).recover()["state"]
    duplicate = store_type(directory).acknowledge(event)
    assert acknowledged["acknowledged"] is True
    assert duplicate["acknowledged"] is True
    assert duplicate["disposition"] == "duplicate"
    assert store_type(directory).recover()["state"] == before

    full_directory = tmp_path / f"full-{store_type.__name__}"
    full = store_type(full_directory, capacity_bytes=1)
    with pytest.raises(OSError) as caught:
        full.acknowledge(event)
    assert caught.value.errno == errno.ENOSPC
    recovered = store_type(full_directory).recover()
    assert recovered["state"] == exp.initial_state()
    assert recovered["evidence"]["acknowledged_loss_count"] == 0


def test_owned_child_fault_matrix_recovers_each_acknowledged_prefix(tmp_path: Path) -> None:
    """SCENARIO-CL-7458-FAULTS: all five child-kill points retain exact parity."""

    rows = exp.run_fault_matrix(tmp_path / "faults")
    assert {row["crash_point"] for row in rows} == set(exp.CRASH_POINTS)
    assert {row["store"] for row in rows} == {"delta_journal", "whole_state"}
    assert len(rows) == len(exp.CRASH_POINTS) * 2
    assert all(row["child_terminated"] for row in rows)
    assert all(row["acknowledged_prefix_preserved"] for row in rows)
    assert all(row["recovered_exactly_once"] for row in rows)
    assert all(row["numeric_state_parity"] and row["prediction_parity"] for row in rows)
    partial = next(
        row
        for row in rows
        if row["store"] == "delta_journal" and row["crash_point"] == "after_partial_append"
    )
    assert partial["tail_action"] == "truncated"


def test_paired_benchmark_records_complete_service_and_alternates(tmp_path: Path) -> None:
    """SCENARIO-CL-7458-BENCHMARK: rows retain every durable update cost."""

    rows = exp.benchmark_protocol(tmp_path / "benchmark", blocks=4, updates=8, ceiling_s=30.0)
    measured = [row for row in rows if row["row_type"] == "service_timing"]
    warmup = [row for row in rows if row["row_type"] == "warmup"]
    assert len(warmup) == 2
    assert len(measured) == 8
    assert [row["order"] for row in measured[::2]] == [
        ["delta_journal", "whole_state"],
        ["whole_state", "delta_journal"],
        ["delta_journal", "whole_state"],
        ["whole_state", "delta_journal"],
    ]
    assert all(len(row["update_latencies_ns"]) == 8 for row in measured)
    assert all(row["total_service_ns"] >= sum(row["update_latencies_ns"]) for row in measured)
    assert all(row["durable_bytes"] > 0 for row in measured)
    assert all(row["write_amplification"] > 0 for row in measured)
    assert all(row["hash_cost_ns"] >= 0 and row["fsync_cost_ns"] >= 0 for row in measured)
    assert all(row["p95_update_latency_ns"] >= row["p50_update_latency_ns"] for row in measured)
    assert all(row["state_parity"] and row["prediction_parity"] for row in measured)
    summary = exp.summarize_timing(rows, draws=500, seed=exp.RESAMPLING_SEED)
    assert summary["paired_block_count"] == 4
    assert summary["ratio_ci95_upper"] >= summary["ratio_ci95_lower"]
    assert 0.0 < summary["residual_host_fraction"] <= 1.0
    assert summary["amdahl_infinite_numeric_speedup_x"] == pytest.approx(
        1.0 / summary["residual_host_fraction"]
    )


def test_ceiling_censors_unstarted_blocks_without_changing_budget(tmp_path: Path) -> None:
    """REQ-CL-7458: the 900-second rule keeps censored units explicit."""

    rows = exp.benchmark_protocol(tmp_path / "censored", blocks=3, updates=2, ceiling_s=0.0)
    measured = [row for row in rows if row["row_type"] == "service_timing"]
    assert len(measured) == 6
    assert all(row["censored"] and row["disposition"] == "unstarted" for row in measured)
    summary = exp.summarize_timing(rows, draws=10, seed=exp.RESAMPLING_SEED)
    assert summary["paired_block_count"] == 0
    assert summary["ratio_ci95_upper"] is None


def test_artifact_reduction_separates_completion_value_and_hardware_claims(
    tmp_path: Path,
) -> None:
    """SCENARIO-CL-7458-GATES/HARDWARE: readiness is not a device claim."""

    artifact = exp.build_fixture_artifact(tmp_path)
    assert exp.validate_artifact(artifact, root=tmp_path) == []
    reduced = exp.independent_reduce(artifact)
    assert reduced["matches_declared"] is True
    assert artifact["durable_update_complete_score"] == 1
    assert artifact["promotion_score"] == 0
    assert artifact["hardware_acceleration_path"]["hardware_performance_claimed"] is False
    assert artifact["hardware_acceleration_path"]["durable_journal_placement"] == "CPU/storage"
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["small_ebm_training"]["performed"] is False
    assert artifact["verdict_class"] in {"positive", "null"}

    changed = deepcopy(artifact)
    changed["service_timing_rows"][0]["durable_bytes"] += 1
    assert "raw_rows_mismatch" in exp.validate_artifact(changed, root=tmp_path)
    changed = deepcopy(artifact)
    changed["durable_update_value_score"] = 1 - artifact["durable_update_value_score"]
    assert "independent_reduction_mismatch" in exp.validate_artifact(changed, root=tmp_path)
    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["unsloth/Qwen3.8-27B-GGUF"]
    assert "current_model_boundary_invalid" in exp.validate_artifact(changed, root=tmp_path)


def test_source_gates_and_validation_plan_are_narrow(tmp_path: Path) -> None:
    """SCENARIO-CL-7458-ARTIFACT: source and command scopes fail closed."""

    checks, hashes, context = exp.collect_preconditions(exp.REPO_ROOT)
    assert checks and all(row["passed"] for row in checks)
    assert context["exp7432"]["verdict_class"] == "null"
    assert context["exp7445"]["flagged_adversarial"] is False
    assert context["exp7438"]["mixture_prototype_ready_score"] == 1
    assert hashes[exp.EXP7445_PATH.as_posix()]["original_verdict_class"] == "null"

    commands = exp.build_validation_plan(exp.REPO_ROOT, tmp_path)
    assert exp.validate_validation_plan(exp.REPO_ROOT, commands) == []
    assert [row.name for row in commands] == list(exp.AFFECTED_CHECK_NAMES)
    focused = next(row for row in commands if row.name == "focused_pytest")
    assert "-n" in focused.argv and "--no-cov" in focused.argv
    assert "tests/python" not in focused.argv


def test_cli_modes_and_date_validation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-CL-7458: the thin command surface supports strict fresh readers."""

    artifact = exp.build_fixture_artifact(tmp_path)
    candidate = tmp_path / "candidate.json"
    exp.atomic_json(candidate, artifact)
    assert (
        exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path), "--cold-replay", str(candidate)])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["errors"] == []
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--root",
                str(tmp_path),
                "--independent-reduce",
                str(candidate),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["reduction"]["matches_declared"] is True
    with pytest.raises(SystemExit, match="--date must be"):
        exp.parse_args(["--date", "20260919"])


def test_invalid_numeric_events_and_envelopes_fail_closed(tmp_path: Path) -> None:
    """REQ-CL-7458: changed numeric, identity, shape, and state bytes are rejected."""

    assert exp._load_object(tmp_path / "absent.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    with pytest.raises(ValueError, match="event_count_negative"):
        exp.fixed_events(-1)

    base = exp.fixed_events(1)[0]
    mutations = []
    for field, value in (
        ("schema", "wrong"),
        ("event_checksum", "sha256:wrong"),
        ("sequence", True),
        ("event_id", "wrong"),
    ):
        changed = deepcopy(base)
        changed[field] = value
        if field not in {"schema", "event_checksum"}:
            payload = {key: item for key, item in changed.items() if key != "event_checksum"}
            changed["event_checksum"] = exp._checksum_bytes(payload)
        mutations.append(changed)
    missing_expert = deepcopy(base)
    missing_expert["deltas"].pop(exp.EXPERT_NAMES[0])
    missing_expert["event_checksum"] = exp._checksum_bytes(
        {key: item for key, item in missing_expert.items() if key != "event_checksum"}
    )
    mutations.append(missing_expert)
    for changed in mutations:
        with pytest.raises(ValueError):
            exp.apply_event(exp.initial_state(), changed)

    bad_shape = deepcopy(base)
    bad_shape["deltas"][exp.EXPERT_NAMES[0]]["coefficients_q16"] = [1]
    bad_shape["event_checksum"] = exp._checksum_bytes(
        {key: item for key, item in bad_shape.items() if key != "event_checksum"}
    )
    with pytest.raises(ValueError, match="coefficient_shape_invalid"):
        exp.apply_event(exp.initial_state(), bad_shape)
    with pytest.raises(ValueError, match="feature_shape_invalid"):
        exp.predict_state(exp.initial_state(), [1])

    envelope = exp._state_envelope(exp.CONTROL_SCHEMA, exp.initial_state())
    with pytest.raises(ValueError, match="state_envelope_invalid"):
        exp._decode_state_envelope({}, exp.CONTROL_SCHEMA)
    changed_envelope = deepcopy(envelope)
    changed_envelope["checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="state_envelope_checksum_invalid"):
        exp._decode_state_envelope(changed_envelope, exp.CONTROL_SCHEMA)


def test_corrupt_journal_rows_and_sequence_gaps_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7458-FAULTS: only a single invalid tail can be truncated."""

    for name, tail in (
        ("invalid-json", b"{bad}\n"),
        ("non-object", b"[]\n"),
        ("bad-checksum", b'{"schema":"wrong","checksum":"bad"}\n'),
    ):
        store = exp.DeltaJournalStore(tmp_path / name)
        store.acknowledge(exp.fixed_events(1)[0])
        with store.journal_path.open("ab") as stream:
            stream.write(tail)
        assert exp.DeltaJournalStore(store.directory).recover()["state"]["sequence"] == 1

    middle = exp.DeltaJournalStore(tmp_path / "middle")
    middle.acknowledge(exp.fixed_events(1)[0])
    with middle.journal_path.open("ab") as stream:
        stream.write(b"{bad}\n{}\n")
    with pytest.raises(ValueError, match="journal_corruption_before_tail"):
        exp.DeltaJournalStore(middle.directory)

    gap = exp.DeltaJournalStore(tmp_path / "gap")
    first, _second, third = exp.fixed_events(3)
    first_receipt = gap.acknowledge(first)
    assert first_receipt["acknowledged"] is True
    payload = {
        "schema": exp.JOURNAL_SCHEMA,
        "sequence": 3,
        "event": third,
        "previous_checksum": gap.last_journal_checksum,
    }
    record = {**payload, "checksum": exp._checksum_bytes(payload)}
    with gap.journal_path.open("ab") as stream:
        stream.write(exp._json_bytes(record) + b"\n")
    with pytest.raises(ValueError, match="journal_sequence_gap"):
        exp.DeltaJournalStore(gap.directory)

    for store_type in (exp.DeltaJournalStore, exp.WholeStateStore):
        with pytest.raises(ValueError, match="sequence_gap"):
            store_type(tmp_path / f"future-{store_type.__name__}").acknowledge(
                exp.fixed_events(2)[1]
            )
    with pytest.raises(ValueError, match="store_unknown"):
        exp._store("unknown", tmp_path / "unknown")


def test_benchmark_guards_and_mid_block_ceiling_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7458-BENCHMARK: invalid, remote, and censored work stays named."""

    with pytest.raises(ValueError, match="benchmark_budget_invalid"):
        exp.benchmark_protocol(tmp_path / "invalid", blocks=0)
    monkeypatch.setattr(exp, "_filesystem_type", lambda _path: "nfs")
    with pytest.raises(RuntimeError, match="network_filesystem_forbidden"):
        exp.benchmark_protocol(tmp_path / "network", blocks=1, updates=1)

    monkeypatch.setattr(exp, "_filesystem_type", lambda _path: "ext4")
    ticks = iter([0.0, 0.0])

    def elapsed() -> float:
        return next(ticks, 1.0)

    monkeypatch.setattr(exp.time, "monotonic", elapsed)
    rows = exp.benchmark_protocol(tmp_path / "mid", blocks=1, updates=2, ceiling_s=0.5)
    measured = [row for row in rows if row["row_type"] == "service_timing"]
    assert all(row["disposition"] == "censored_ceiling" for row in measured)
    assert all(row["censored"] is True for row in measured)


def test_small_reducer_boundaries_and_mount_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-CL-7458: reducers reject empty inputs and retain uncommon budget rows."""

    with pytest.raises(ValueError, match="percentile_needs_values"):
        exp._percentile([], 0.5)
    assert exp._percentile([7], 0.95) == 7.0
    with pytest.raises(ValueError, match="paired_ratio_inputs_invalid"):
        exp._paired_ratio_interval([], [], draws=1, seed=1)
    assert exp._required_receipts([], []) is True

    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, **_kwargs: "malformed\n/dev/root / ext4 rw 0 0\n",
    )
    assert exp._filesystem_type(tmp_path) == "ext4"
    monkeypatch.undo()

    rows = exp._fixture_timing_rows()
    rows.append({"row_type": "warmup", "block": -1})
    rows[0]["censored"] = True
    rows[1]["failed"] = True
    budget = exp._sample_budget(rows, 5)
    assert budget["censored_independent_units"] == 1
    assert budget["failed_independent_units"] == 1

    monkeypatch.setattr(exp, "progress", lambda *_args, **_kwargs: print("progress"))
    exp.benchmark_protocol(tmp_path / "progress", blocks=1, updates=1, emit_progress=True)
    assert "progress" in capsys.readouterr().out

    outside = tmp_path / "outside.json"
    exp.atomic_json(outside, {"value": 1})
    assert exp._raw_reference(outside, tmp_path / "different-root")["path"] == str(outside)


def test_artifact_reader_rejects_reference_schema_and_hardware_mutations(tmp_path: Path) -> None:
    """SCENARIO-CL-7458-ARTIFACT: every terminal authority fails closed."""

    artifact = exp.build_fixture_artifact(tmp_path)
    changed = deepcopy(artifact)
    changed["raw_evidence_reference"] = None
    assert "raw_evidence_reference_invalid" in exp.validate_artifact(changed, root=tmp_path)

    changed = deepcopy(artifact)
    changed["raw_evidence_reference"]["sha256"] = "sha256:wrong"
    assert "raw_evidence_hash_mismatch" in exp.validate_artifact(changed, root=tmp_path)

    changed = deepcopy(artifact)
    changed["field_principles"].pop("schema")
    assert "field_principles_mismatch" in exp.validate_artifact(changed, root=tmp_path)

    changed = deepcopy(artifact)
    changed["verdict_class"] = "unknown"
    assert "verdict_class_invalid" in exp.validate_artifact(changed, root=tmp_path)

    changed = deepcopy(artifact)
    changed["hardware_acceleration_path"]["hardware_performance_claimed"] = True
    assert "hardware_boundary_invalid" in exp.validate_artifact(changed, root=tmp_path)

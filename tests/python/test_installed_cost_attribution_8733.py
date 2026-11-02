"""REQ-REPORT-8733 / REQ-VERIFY-8733: authenticated installed diagnostic controls."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import installed_cost_attribution_8733 as a
from carnot.reporting.v709_execution import child


def test_inputs_and_roster(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8733-ATTRIBUTION: current contract and fixed scientific operands."""
    inputs = a.inputs(a.e.ROOT)
    assert inputs["passed"], inputs["gate_check_summary"]
    assert len(inputs["contract"]["rows"]) == 14
    assert len(a.roster({})) == 40
    assert len({r["unit_id"] for r in a.roster({})}) == 40
    assert not a.inputs(tmp_path)["passed"]
    monkeypatch.setattr(a, "COST_PIN", "sha256:changed")
    assert not a.inputs(a.e.ROOT)["passed"]


def test_intervals_and_overhead():
    """SCENARIO-REPORT-8733-ATTRIBUTION: whole paired processes determine uncertainty."""
    rows = [
        dict(
            r,
            status="completed",
            passed=True,
            duration_s=1.0,
            components={"kernel": 0.2},
            child_duration_s=0.5,
            parent_components={"process_startup": 0.1},
            request_digest="same",
            output_digest="same",
        )
        for r in a.roster({})
    ]
    summary = a.aggregate(rows)
    assert summary["ready"] and summary["overhead"]["upper95"] == 0
    assert len(summary["fractions"]) == 4
    assert {f["clock_domain"] for f in summary["fractions"]} == {"child_serve", "parent_process"}
    for r in rows:
        if r["instrumented"]:
            r["duration_s"] = 1.1
    assert not a.aggregate(rows)["ready"]
    rows[0]["status"] = "failed"
    assert not a.aggregate(rows)["complete"]


def test_profile_spans(monkeypatch):
    """REQ-VERIFY-8733: nested operation timing yields disjoint local segments."""
    from carnot.inference.installed_profile_8733 import Recorder

    recorder = Recorder()
    assert recorder.call("outer", lambda: recorder.call("inner", lambda: 7)) == 7
    assert all(s["end_ns"] >= s["start_ns"] for s in recorder.spans)
    assert all(x["end_ns"] <= y["start_ns"] for x, y in zip(recorder.spans, recorder.spans[1:]))
    with pytest.raises(ValueError):
        recorder.call("error", lambda: (_ for _ in ()).throw(ValueError("control")))


@pytest.fixture
def measured(tmp_path):
    """REQ-REPORT-8733: real installed process pairs keep complete frame primitives."""
    inputs = a.inputs(a.e.ROOT)
    raw = tmp_path / "measurement"
    raw.mkdir()
    command, config, receipts = a.prepare(raw, tmp_path, inputs)
    assert all(r["passed"] for r in receipts)
    values = []
    for slot in a.roster({})[:2]:
        slot = dict(slot, panels=1, item_evaluations=64)
        values.append(
            a.session(
                command, config, slot, inputs["requests"], tmp_path, raw / slot["unit_id"], 2**63
            )
        )
    return inputs, values


def test_real_spans_and_parity(measured):
    """REQ-REPORT-8733: exact same-arm output and request bytes, bounded local spans."""
    inputs, values = measured
    reduced = [a.reduce(v, inputs) for v in values]
    assert all(r["process_row"]["passed"] for r in reduced)
    assert (
        reduced[0]["process_row"]["request_digest"] == reduced[1]["process_row"]["request_digest"]
    )
    assert reduced[0]["process_row"]["output_digest"] == reduced[1]["process_row"]["output_digest"]
    assert any(r["process_row"]["components"] for r in reduced)
    for mode in ["exit", "span", "output", "request"]:
        value = deepcopy(values[0])
        if mode == "exit":
            value["actual_exit"] = 1
        elif mode == "span":
            value["profile"]["spans"] = [dict(operation="forged", start_ns=0, end_ns=2**63)]
        elif mode == "output":
            value["observations"][0]["response"]["responses"][0]["decision"] = "forged"
        else:
            value["observations"][0]["request"]["rows"] = []
        with pytest.raises(ValueError):
            a.reduce(value, inputs)


def test_blocked_cli_and_replay(tmp_path):
    """SCENARIO-VERIFY-8733-CUSTODY: real CLI publication and sealed cold reads."""
    output = tmp_path / (a.e.NAME + ".json")
    args = [sys.executable, "-u", a.e.CLI]
    receipt = child(
        "blocked",
        args + ["--root", str(tmp_path), "--private-output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 40
    authority = value["primitive_reference"]
    assert a.replay(output, authority)
    assert not a.replay(output, None)
    changed = deepcopy(value)
    changed["completed_count"] += 1
    a.e.atomic_json(output, changed)
    assert not a.replay(output, authority)
    assert not a.replay(tmp_path / "missing", authority)
    assert a.main(["--date", "wrong"]) == 2
    assert a.main(["--private-output", str(a.e.ROOT / "unsafe.json")]) == 2
    a.e.atomic_json(output, value)
    assert (
        a.main(
            ["--cold-replay", str(output), "--authority", authority["path"], authority["sha256"]]
        )
        == 0
    )


def test_profile_opt_in_main(tmp_path, monkeypatch):
    """REQ-VERIFY-8733: an opt-in wrapper retains service errors and original imports."""
    from types import SimpleNamespace
    from carnot.inference import installed_profile_8733 as p

    inputs = a.inputs(a.e.ROOT)
    config = tmp_path / "head.json"
    a.e.atomic_json(
        config,
        dict(
            inputs["workload"]["configuration_template"], extension=inputs["workload"]["extension"]
        ),
    )
    # Register restoration before install changes the shared in-process module.
    for owner, names in [
        (p.wire, ["read_frame", "write_frame", "sha256_file", "configuration", "decide", "json"]),
        (p.batch, ["evaluate", "json"]),
        (p.wire.numeric, ["batch", "extract", "json"]),
    ]:
        for name in names:
            monkeypatch.setattr(owner, name, getattr(owner, name))
    monkeypatch.setattr(p, "sys", SimpleNamespace(path=["unchanged"], modules={}))

    def serve(path, digest, arm):
        cfg, native, loaded = p.wire.configuration(path, digest, arm)
        frame = dict(
            p.wire.envelope("one", digest, {}), operation="batch", rows=inputs["requests"][:16]
        )
        assert p.wire.decide(frame, cfg, digest, native, set())["status"] == "ok"
        return 0

    monkeypatch.setattr(p.wire, "serve", serve)
    args = [
        str(config),
        a.e.reference(config)["sha256"],
        "python",
        "--profile-output",
        str(tmp_path / "profile"),
    ]
    assert p.main(args) == 0
    assert p.main(args + ["--instrument"]) == 0
    value = json.loads((tmp_path / "profile").read_text())
    assert value["spans"] and value["actual_exit"] == 0


def test_paired_tamper_and_report_states(tmp_path):
    """REQ-REPORT-8733: overhead, paired byte drift and owned failures cannot certify cost."""
    rows = [
        dict(
            r,
            status="completed",
            passed=True,
            duration_s=1.0,
            child_duration_s=0.5,
            parent_components={"process_startup": 0.1},
            request_digest="same",
            output_digest="same",
            components={"kernel": 0.1},
        )
        for r in a.roster({})
    ]
    rows[0]["output_digest"] = "changed"
    with pytest.raises(ValueError, match="paired_exact_parity"):
        a.aggregate(rows)
    rows[0]["output_digest"] = "same"
    raw = tmp_path / "raw"
    raw.mkdir()
    a.e.atomic_json(raw / "work.json", {})
    work = dict(
        raw=str(raw),
        invocation_id="control",
        measurement=dict(process_rows=rows, receipts=[], complete_cost_rows=[]),
        validation=dict(
            receipts=[dict(passed=True)],
            coverage=dict(complete=True),
            custody=dict(passed=True),
            raw_shard_hashes=[],
        ),
        private_control=False,
        owned_error=None,
        inputs=dict(gate_check_summary=[], references=[], contract=dict(rows=[])),
        preconditions={},
        duration_s=1.0,
        started_at_utc="now",
        ended_at_utc="now",
        phase_spans=[],
        sources=[],
        raw_references=[],
    )
    result = a.e.build(work, tmp_path / "output")
    assert result["installed_attribution_ready_score"] == 1
    for row in rows:
        if row["instrumented"]:
            row["duration_s"] = 1.2
    assert a.e.build(work, tmp_path / "output")["verdict_class"] == "disqualified"
    work["owned_error"] = "failure"
    assert a.e.build(work, tmp_path / "output")["installed_attribution_ready_score"] == 0
    work["validation"]["coverage"]["complete"] = False
    assert a.e.build(work, tmp_path / "output")["required_checks_passed"] is False


def test_replay_primitives_and_rehashed_rejections(measured, tmp_path):
    """SCENARIO-VERIFY-8733-CUSTODY: replay reconstructs component rows from sealed frames."""
    inputs, values = measured
    raw = Path(values[0]["profile_reference"]["path"]).parents[2]
    reduced = [a.reduce(v, inputs) for v in values]
    measurement = dict(process_rows=[r["process_row"] for r in reduced], receipts=[])
    for key in ["request_rows", "parity_rows", "complete_cost_rows"]:
        measurement[key] = sum((r[key] for r in reduced), [])
    output = tmp_path / (a.e.NAME + ".json")
    work = dict(
        raw=str(raw),
        output=str(output),
        input_root=str(a.e.ROOT),
        invocation_id="control",
        inputs=inputs,
        measurement=measurement,
        sources=[a.e.reference(Path(a.__file__))],
        raw_references=[],
        validation=dict(
            receipts=[],
            coverage=dict(complete=False),
            custody=dict(passed=False),
            raw_shard_hashes=[],
        ),
        private_control=True,
        owned_error=None,
        preconditions={},
        duration_s=1.0,
        started_at_utc="now",
        ended_at_utc="now",
        phase_spans=[],
    )

    def seal(value):
        value = {k: v for k, v in value.items() if k != "seal_reference"}
        a.e.atomic_json(raw / "seal.json", value)
        value["seal_reference"] = a.e.reference(raw / "seal.json")
        a.e.atomic_json(raw / "work.json", value)
        result = a.e.build(value, output)
        a.e.atomic_json(output, result)
        return result["primitive_reference"]

    authority = seal(work)
    assert a.replay(output, authority)
    for mode in ["inputs", "sources", "measurement", "seal"]:
        changed = deepcopy(work)
        if mode == "inputs":
            changed["inputs"]["passed"] = False
        elif mode == "sources":
            changed["sources"][0]["sha256"] = "changed"
        elif mode == "measurement":
            changed["measurement"]["complete_cost_rows"][0]["duration_s"] += 1
        ref = seal(changed)
        if mode == "seal":
            changed_seal = a.e.load(a.e.reference(raw / "seal.json"))
            changed_seal["duration_s"] += 1
            a.e.atomic_json(raw / "seal.json", changed_seal)
            current = a.e.load(ref)
            current["seal_reference"] = a.e.reference(raw / "seal.json")
            a.e.atomic_json(raw / "work.json", current)
            ref = a.e.reference(raw / "work.json")
            updated = a.e.build(current, output)
            a.e.atomic_json(output, updated)
        assert not a.replay(output, ref), mode
    receipt = child(
        "source_profile_help",
        [sys.executable, "-m", "carnot.inference.installed_profile_8733", "--help"],
        tmp_path / "help",
    )
    assert receipt["passed"]


def test_remaining_clock_and_receipt_rejections(measured, tmp_path, monkeypatch):
    """REQ-VERIFY-8733: malformed local clocks, parent clocks and receipts fail closed."""
    inputs, values = measured
    for mode in ["local", "sum", "count", "parent", "residual"]:
        value = deepcopy(values[0])
        if mode == "local":
            value["profile"]["spans"] = [dict(operation="bad", start_ns=0, end_ns=2**63)]
        elif mode == "sum":
            value["profile"].update(spans=[], start_ns=2, end_ns=1)
        elif mode == "count":
            value["observations"].pop()
        elif mode == "parent":
            value["observations"][0]["packed_ns"] = 0
        else:
            value["end_ns"] = value["start_ns"]
        profile = tmp_path / (mode + ".json")
        a.e.atomic_json(profile, value["profile"])
        value["profile_reference"] = a.e.reference(profile)
        with pytest.raises(ValueError):
            a.reduce(value, inputs)
    monkeypatch.setattr(a, "read_bound_sidecar", lambda *args: dict(report=dict(passed=False)))
    assert not a.inputs(a.e.ROOT)["passed"]

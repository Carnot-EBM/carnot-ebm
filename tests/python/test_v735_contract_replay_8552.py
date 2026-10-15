"""REQ-VERIFY-8552: real child paths and independent gate reductions."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import v735_contract_methods as e
from carnot.reporting import v735_contract_runner as r
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from test_v735_contract_methods_8552 import full_root  # noqa: F401


def fake_history(absent=None):
    """Vary one original family without inventing scientific measurements."""
    return dict(
        families=[
            dict(experiment_id=n, qualified=n != absent, gate_check_summary=[]) for n in e.FIELDS
        ],
        references=[],
        receipts=[],
    )


@pytest.mark.parametrize("absent", [None, 8430, 8433, 8529, 8536])
def test_family_boundaries(full_root, tmp_path, monkeypatch, absent):
    """SCENARIO-REPORT-8552-METHODS: absence never couples independent branches."""
    monkeypatch.setattr(e, "historical", lambda *a: fake_history(absent))
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["methods_ready_score"] == value["current_contract_ready_score"] == 1
    assert value["source_inputs_ready_score"] == int(absent not in {8430, 8529})
    assert value["learning_inputs_ready_score"] == int(absent not in {8430, 8433})
    assert value["nine_input_inputs_ready_score"] == int(absent != 8430)
    assert value["canary_inputs_ready_score"] == int(absent not in {8430, 8529})
    assert value["runtime_repair_ready_score"] == 0
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    assert value["future_dependencies"][0]["experiment_id"] == 8553
    assert e.build(work, [], raw, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_missing_and_killed(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8552-CONTROLS: missing authority blocks; killed work fails."""
    history = fake_history()
    monkeypatch.setattr(e, "historical", lambda *a: history)
    raw = tmp_path / "raw"
    work = e.measure(tmp_path / "missing", raw)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "blocked" and value["methods_ready_score"] == 0
    receipt = child("real", [sys.executable, "-c", "print('real')"], raw / "logs")
    history["receipts"] = [dict(receipt, passed=False, normal_exit=False, actual_exit=-9)]
    work = e.measure(tmp_path / "missing", tmp_path / "killed")
    value = e.build(work, [receipt], tmp_path / "killed", tmp_path / (e.NAME + ".json"))
    assert value["verdict_class"] == "disqualified" and not value["required_checks_passed"]
    clock = iter([0, 3601_000_000_000])
    monkeypatch.setattr(e, "measurement_clock", lambda: next(clock))
    assert any(
        g["artifact_field"] == "measurement_cap_s"
        for g in e.measure(tmp_path / "missing", tmp_path / "late")["failures"]
    )


@pytest.mark.parametrize(
    "mode", ["valid", "protocol", "authority", "family", "source", "receipt", "upstream"]
)
def test_rehashed_primitives(full_root, tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8552-CONTROLS: independent operands defeat repaired hashes."""
    monkeypatch.setattr(e, "historical", lambda *a: fake_history())
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    receipt = child("real", [sys.executable, "-c", "print('real')"], raw / "logs")
    if mode == "protocol":
        work["protocol"]["partition"]["reseeding"] = True
    if mode == "authority":
        work["contract"]["contract_rows"][0]["checks"]["full_object"] = False
    if mode == "family":
        work["history"]["families"][0]["qualified"] = False
    if mode == "upstream":
        work["v734_reader_receipts"][0]["summary"] = "forged current reader selection"
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, [receipt], raw, tmp_path / (e.NAME + ".json"))
    if mode == "source":
        value["code_config_hashes"][0]["sha256"] = "changed"
    if mode == "receipt":
        value["validation_receipts"][0]["duration_s"] += 1
    value["reproducibility_checksum"] = e.checksum(value)
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert e.replay(candidate) == (mode == "valid")


def test_private_cli(tmp_path, full_root):
    """SCENARIO-VERIFY-8552-CONTROLS: actual CLI publishes and replays private bytes."""
    output = tmp_path / (e.NAME + ".json")
    base = [sys.executable, "-u", str(e.ROOT / e.CLI), "--root", str(full_root)]
    cases = [
        (["--date", "20261015", "--private-control", "--output", str(output)], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261014"], 2),
        (["--private-control"], 2),
        (["--history-request", str(tmp_path / "missing")], 2),
        (
            [
                "--private-control",
                "--root",
                str(tmp_path / "missing"),
                "--output",
                str(tmp_path / "absent" / (e.NAME + ".json")),
            ],
            0,
        ),
    ]
    for i, (args, expected) in enumerate(cases):
        receipt = child(f"cli-{i}", base + args, tmp_path / "logs", expected=expected, deadline=600)
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "circular_positive" and value["methods_ready_score"] == 1
    assert value["source_inputs_ready_score"] == value["learning_inputs_ready_score"] == 0
    assert value["nine_input_inputs_ready_score"] == 0
    value["nine_input_inputs_ready_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)


def test_runner_boundaries(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8552-CONTROLS: all worker and timer restoration paths run."""
    monkeypatch.setattr(r.history, "worker", lambda *a: 3)
    assert (
        r.main(
            ["--history-request", str(tmp_path / "in"), "--history-output", str(tmp_path / "out")]
        )
        == 3
    )
    monkeypatch.setattr(r.shipped.shipped, "run", lambda *a, **kw: 0)
    assert r.main(["--private-control", "--output", str(tmp_path / (e.NAME + ".json"))]) == 0
    monkeypatch.setattr(
        r.shipped.shipped, "run", lambda *a, **kw: (_ for _ in ()).throw(ValueError("intentional"))
    )
    with pytest.raises(ValueError, match="intentional"):
        r.main(["--output", str(tmp_path / (e.NAME + ".json"))])
    calls = []
    monkeypatch.setattr(r, "child", lambda *a, **kw: calls.append(kw) or {})
    r.bounded_child("plain", [], tmp_path)
    r.bounded_child("cold", ["--cold-replay"], tmp_path)
    assert calls == [{}, {"deadline": 600}]


def test_manifest(tmp_path):
    """SCENARIO-VERIFY-8552-CONTROLS: shards keep exactly the collected original union."""
    plan = r.manifest(tmp_path / "plan")
    for check in (p for p in plan if p["name"] in {"ruff_check", "ruff_format", "spec_coverage"}):
        assert all(t in check["argv"] for t in e.TESTS)
    for collection in (
        p for p in plan if p["name"].startswith("collect_") and p["argv"][-1] not in e.TESTS
    ):
        nodes = [
            n
            for p in plan
            if p["name"].startswith("historical_")
            for n in p["collected_node_ids"]
            if n.startswith(collection["argv"][-1] + "::")
        ]
        assert nodes == collection["collected_node_ids"] and len(nodes) == len(set(nodes))


def test_reader_selected_inputs(full_root, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8552-CONTROLS: pre-gate paths come from actual reader receipts."""
    monkeypatch.setattr(e, "historical", lambda *a: fake_history())
    source = tmp_path / "actual_pregate.json"
    atomic_json(source, dict(required_checks_passed=False))
    monkeypatch.setattr(
        e,
        "reader_receipt",
        lambda *a, **kw: dict(gate_path=str(source), gate_sha256=e.sha256_file(source)),
    )
    raw = tmp_path / "raw"
    work = e.measure(full_root, raw)
    value = e.build(work, [dict(passed=True)], raw, tmp_path / (e.NAME + ".json"))
    assert e.reference(source) in value["source_artifact_hashes"]


def test_preflight_and_counts(full_root, tmp_path, monkeypatch, capsys):
    """SCENARIO-VERIFY-8552-CONTROLS: writable scratch and thirteen real slots precede work."""
    monkeypatch.setattr(e, "historical", lambda *a: fake_history())
    work = e.measure(full_root, tmp_path / "raw")
    assert work["preconditions"]["scratch_write_read_passed"] is True
    output = capsys.readouterr().out
    assert "phase=authority_after completed=13 pending=0" in output
    assert "phase=methods_frozen_after completed=13 pending=0" in output

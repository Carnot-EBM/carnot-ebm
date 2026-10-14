"""REQ-REPORT-8520, REQ-VERIFY-8520: fresh clocks need exact current authority."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot.reporting import append_service_cost_8520 as r
from carnot.reporting import append_service_runner_8520 as runner
from carnot.reporting.current_work_receipt import atomic_json


def test_exact_admission(tmp_path):
    """SCENARIO-REPORT-8520-GATES: qualified imports grant fresh work, never old clocks."""
    inputs = r.admission(r.ROOT, tmp_path)
    assert inputs["authority"]["activated"]
    assert [t["cell"]["batch"] for t in inputs["tapes"]] == [1, 8]
    assert inputs["receipt"]["passed"]
    assert inputs["upstream"]["run_date"] == "20261013"


def test_bound_preflight(tmp_path):
    """SCENARIO-REPORT-8520-GATES: temporary worker identities cannot relabel historical inputs."""
    with r.binding(), patch.multiple(r.store, TASK=r.TASK, NAME=r.NAME):
        checks = r.preflight(r.ROOT, tmp_path)
    assert not checks["gate_check_summary"]


def test_exact_strict_typing():
    """SCENARIO-VERIFY-8520-VALIDATION: imported shipped modules must be checked with the adapter."""
    result = subprocess.run(
        [
            str(r.ROOT / ".venv/bin/mypy"),
            "--config-file=/dev/null",
            "--strict",
            "--follow-imports=skip",
            "--ignore-missing-imports",
            *r.OWNED,
            *r.shipped.OWNED,
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "hash",
        "score",
        "identity",
        "side",
        "consumer",
        "cold",
        "protocol",
        "date",
        "authority",
        "task",
    ],
)
def test_admission_rejection(tmp_path, monkeypatch, failure):
    """SCENARIO-REPORT-8520-GATES: altered qualification or protocol fails before timing."""
    original = r.read(r.reference(r.ROOT / r.UPSTREAM))
    read = r.read
    monkeypatch.setattr(
        r, "preflight", lambda *a: dict(gate_check_summary=[], source_artifact_hashes=[])
    )
    if failure == "missing":
        with pytest.raises(r.GateFailure) as caught:
            r.admission(tmp_path, tmp_path / "raw")
        assert caught.value.row["observed"] is None
        return
    if failure == "hash":
        original["append_runner_ready_score"] = 0
        atomic_json(tmp_path / Path(r.UPSTREAM).name, original)
        monkeypatch.setattr(r, "UPSTREAM", Path(r.UPSTREAM).name)
        with pytest.raises(r.GateFailure):
            r.admission(tmp_path, tmp_path / "raw")
        return
    if failure == "score":
        original["append_runner_ready_score"] = 0
    if failure == "identity":
        original["task_id"] = "exp8506-pre-gate"
    if failure == "side":
        monkeypatch.setattr(r, "read_bound_sidecar", lambda *a: dict(report=dict(passed=False)))
    if failure == "consumer":
        monkeypatch.setattr(r, "reader_receipt", lambda *a, **kw: dict(passed=False))
    if failure == "cold":
        monkeypatch.setattr(r, "child", lambda *a, **kw: dict(passed=False))
    if failure == "protocol":
        protocol = r.methods.protocol()
        protocol["append_protocol"] = {}
        monkeypatch.setattr(r.methods, "protocol", lambda: protocol)
    if failure == "authority":
        monkeypatch.setattr(r.methods, "authority", lambda *a: dict(activated=False, tasks=[]))
    if failure == "task":
        authority = r.methods.authority(r.ROOT, tmp_path / "authority")
        next(t for t in authority["tasks"] if t["id"] == r.TASK)["deliverable"] = "wrong"
        monkeypatch.setattr(r.methods, "authority", lambda *a: authority)

    def altered(ref):
        if ref["path"] == str(r.ROOT / r.UPSTREAM):
            return deepcopy(original)
        value = read(ref)
        if failure == "date" and ref["path"] == str(r.ROOT / r.store.UPSTREAM):
            value["run_date"] = "20261014"
        return value

    monkeypatch.setattr(r, "read", altered)
    with pytest.raises(r.GateFailure):
        r.admission(r.ROOT, tmp_path / "raw")


def test_private_cli_and_tamper(tmp_path):
    """SCENARIO-REPORT-8520-TERMINAL: absence blocks; repaired aggregate hashes reject."""
    output = tmp_path / (r.NAME + ".json")
    proc = subprocess.run(
        [
            sys.executable,
            "-u",
            str(r.ROOT / r.CLI),
            "--date",
            "20261014",
            "--private-control",
            "--root",
            str(tmp_path / "missing"),
            "--output",
            str(output),
        ],
        capture_output=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["censored_count"] == 24 and value["primary_speedup_ci95"] is None
    assert value["service_cost_ready_score"] == value["append_speed_benefit_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["run_date"] == "20261014"
    assert r.replay(output)
    value["service_cost_ready_score"] = 1
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(output, value)
    assert not r.replay(output)


def test_cli_controls(tmp_path):
    """SCENARIO-VERIFY-8520-VALIDATION: real dates and worker argument errors are checked."""
    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--private-control"]) == 2
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261013"])
    with pytest.raises(SystemExit):
        runner.main(["--worker", str(tmp_path / "absent")])


def test_preflight_rejection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8520-GATES: resource and authority failures retain their original row."""
    failed = r.qualified.gate(tmp_path / "missing", "tools_memory_disk", True, None)
    monkeypatch.setattr(r, "preflight", lambda *a: dict(gate_check_summary=[failed]))
    with pytest.raises(r.GateFailure) as caught:
        r.admission(r.ROOT, tmp_path)
    assert caught.value.row == failed


@pytest.fixture(scope="module")
def primitives(tmp_path_factory):
    """SCENARIO-VERIFY-8520-PAIRS: fresh actual children preserve the complete five-input service."""
    private = tmp_path_factory.mktemp("service8520")
    original = r.read(r.reference(r.ROOT / r.store.UPSTREAM))
    tape = r.read(original["transition_fixture"]["tape_references"][1])
    frozen = private / "frozen.json"
    atomic_json(frozen, tape)
    rows = []
    for arm in ("snapshot", "append"):
        child = subprocess.run(
            [
                sys.executable,
                "-u",
                str(r.ROOT / r.CLI),
                "--worker",
                str(frozen),
                "--store",
                str(private / arm),
                "--arm",
                arm,
            ],
            capture_output=True,
            timeout=120,
        )
        assert child.returncode == 0, child.stderr.decode()
        primitive = r.read(r.reference(private / arm / "primitive.json"))
        metrics = r.k.reduce(tape, primitive)
        assert metrics["decision_attempts"] == 64
        assert metrics["applied_updates"] == 7 and metrics["nonapplicable_feedback"] == 1
        assert metrics["lost"] == metrics["missing"] == metrics["pending"] == 0
        rows.append(primitive)
    assert r.k.pair(tape, *rows)["accounting_equality"]
    return tape, rows


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "identity",
        "length",
        "authority",
        "tapes",
        "hash",
        "roster",
        "frozen",
        "metrics",
        "store_hash",
        "disk_state",
        "summary",
    ],
)
def test_cold_primitive_rejection(primitives, tmp_path, monkeypatch, mutation):
    """SCENARIO-REPORT-8520-TERMINAL: self-consistent hashes cannot repair altered operands."""
    tape, primitives = primitives
    work = r.measure(tmp_path / "missing", tmp_path / "raw")
    work.update(tapes=[deepcopy(tape), deepcopy(tape)], gate_check_summary=[])
    for index, primitive in zip((15, 14), primitives, strict=True):
        work["rows"][index].update(
            status="completed",
            missing_reason=None,
            primitive_reference=r.reference(Path(primitive["directory"]) / "primitive.json"),
            metrics=r.k.reduce(tape, primitive),
        )
    sealed = deepcopy(work["tapes"])
    monkeypatch.setattr(r, "admission", lambda *a: dict(tapes=sealed))
    output = tmp_path / "candidate.json"
    value = r.build(work, [], Path(work["raw"]), output)
    for share in value["complete_cost_shares"]:
        if share["wall_s"]:
            assert sum(share["seconds"].values()) == pytest.approx(share["wall_s"])
            assert sum(share["fractions"].values()) == pytest.approx(1)
    if mutation == "identity":
        value["experiment_id"] = 1
    elif mutation == "length":
        work["rows"].pop()
    elif mutation == "authority":
        work["authority"].update(activated=True, tasks=[])
    elif mutation == "tapes":
        work["tapes"][0]["cell"]["batch"] = 32
    elif mutation == "hash":
        value["code_config_hashes"][0]["sha256"] = "forged"
    elif mutation == "roster":
        work["rows"][0]["arm"] = "append"
    elif mutation == "metrics":
        work["rows"][14]["metrics"]["wall_s"] += 1
    elif mutation == "summary":
        value["complete_cost_shares"][14]["fractions"]["startup_s"] += 1
    elif mutation in ("frozen", "store_hash", "disk_state"):
        row = work["rows"][15]
        primitive = deepcopy(r.read(row["primitive_reference"]))
        if mutation == "frozen":
            frozen = r.read(primitive["frozen_reference"])
            frozen["cell"]["batch"] = 1
            atomic_json(tmp_path / "changed_frozen.json", frozen)
            primitive["frozen_reference"] = r.reference(tmp_path / "changed_frozen.json")
        elif mutation == "store_hash":
            primitive["store_references"][0]["sha256"] = "forged"
        else:
            import shutil

            target = tmp_path / "older" / "store"
            shutil.copytree(Path(primitive["directory"]) / "store", target)
            primitive["directory"] = str(target.parent)
            initial = r.k.base.oracle(sealed[1], 0)
            r.k.base.Snapshot(target, sealed[1]).store.commit(initial["state"])
            primitive["store_references"] = [
                r.reference(p) for p in sorted(target.iterdir()) if p.is_file()
            ]
        atomic_json(tmp_path / "changed_primitive.json", primitive)
        row["primitive_reference"] = r.reference(tmp_path / "changed_primitive.json")
    atomic_json(Path(work["raw"]) / "measurement.json", work)
    value["work_reference"] = r.reference(Path(work["raw"]) / "measurement.json")
    value["reproducibility_checksum"] = r.checksum(value)
    atomic_json(output, value)
    assert r.replay(output) is (mutation == "none")


def test_manifest_union(tmp_path):
    """SCENARIO-VERIFY-8520-VALIDATION: exact original node IDs and private paths survive adaptation."""
    plan = runner.manifest(tmp_path)
    original = json.loads((tmp_path / "collections.json").read_bytes())
    nodes = [n for c in original["collections"] for n in c["nodes"]]
    retained = [n for p in plan if p["name"].startswith("historical_") for n in p.get("nodes", [])]
    assert nodes == retained and len(set(nodes)) == len(nodes)
    assert all(p["deadline"] <= 300 for p in plan)
    assert all(
        "-n" in p["argv"] and p["argv"][p["argv"].index("-n") + 1] == "0"
        for p in plan
        if p.get("nodes")
    )

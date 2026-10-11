"""REQ-VERIFY-8410 and REQ-REPORT-8410: reading never implies a CUDA remedy."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.verify import cuda_reader_8410 as q
from carnot.verify import cuda_reader_runner_8410 as r


def test_cached_reduction(tmp_path):
    """SCENARIO-REPORT-8410-REPLAY: sealed primitive bytes preserve the failed diagnosis."""
    result = q.cached(q.ROOT / q.CACHED, tmp_path)
    assert result["diagnosis"]["api_returns"] == {"cuInit": 999}
    assert result["diagnosis"]["root_cause_status"] == "cause_unproved"
    assert result["diagnosis"]["failure_layer"] == "driver_initialization"
    assert result["diagnosis"]["correction"] is None
    assert result["trace_manifest"]["device_returns"]
    assert result["previous"]["boot_identity"] is None
    assert result["previous"]["namespaces"] is None
    bad = tmp_path / "bad.json"
    atomic_json(bad, {"diagnostic": {}})
    with pytest.raises(ValueError):
        q.cached(bad, tmp_path / "bad")


def test_inventory_and_comparison(tmp_path):
    """SCENARIO-VERIFY-8410-DELTA: absent past operands cannot manufacture execution change."""
    previous = q.cached(q.ROOT / q.CACHED, tmp_path)["previous"]
    observed = q.inventory(previous)
    assert observed["boot_identity"] and observed["namespaces"]
    rows = q.compare(previous, observed)
    assert not any(row["changed"] for row in rows)
    assert next(row for row in rows if row["field"] == "boot_identity")["missing_reason"]
    observed["kernel_driver"] += "changed"
    assert any(row["changed"] for row in q.compare(previous, observed))
    observed["kernel_driver"] = None
    assert not next(
        row for row in q.compare(previous, observed) if row["field"] == "kernel_driver"
    )["changed"]


def test_no_delta_no_probe(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8410-DELTA: qualifying cached evidence never repeats retired inspection."""
    probe = MagicMock(side_effect=AssertionError("no new probe authorized"))
    monkeypatch.setattr(q, "probe", probe)
    work = q.measure(q.ROOT, tmp_path)
    assert not work["failures"]
    receipt = child("owned", [r.PY, "-c", "print('actual owned check')"], tmp_path / "checks")
    value = q.build(work, tmp_path, [receipt])
    assert value["runtime_reader_ready_score"] == 1
    assert value["runtime_changed_score"] == value["cuda_context_ready_score"] == 0
    assert value["current_device_probe_count"] == 0
    assert value["verdict_class"] == "blocked"
    assert not any(value["model_invocation_counts"].values())
    assert set(value) <= set(value["field_principles"])
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert q.replay(candidate)
    for field in [
        "runtime_reader_ready_score",
        "runtime_changed_score",
        "cuda_context_ready_score",
        "current_device_probe_count",
        "completed_count",
    ]:
        bad = deepcopy(value)
        bad[field] += 1
        bad["reproducibility_checksum"] = q.checksum(bad)
        atomic_json(candidate, bad)
        assert not q.replay(candidate)
    receipt["passed"] = False
    assert q.build(work, tmp_path, [receipt])["verdict_class"] == "disqualified"
    assert not q.replay(tmp_path / "absent")


@pytest.mark.parametrize("mode", ["success", "error", "busy", "bad_json", "timeout"])
def test_probe_paths(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8410-DELTA: one owned child retains errors and cleans up its lease."""
    lease = MagicMock()
    lease.owner_receipt.return_value = {"device_uuid": "GPU-test"}
    lease.release.return_value = {"released": True}
    monkeypatch.setattr(
        q.GpuLease,
        "acquire",
        MagicMock(return_value=lease, side_effect=ValueError("busy") if mode == "busy" else None),
    )
    calls = []

    def run(name, argv, raw, **kwargs):
        calls.append(argv)
        assert kwargs["deadline"] == 120
        primitive = {
            "api_returns": {"cuInit": 0 if mode == "success" else 999},
            "context_copy_ready": mode == "success",
            "byte_copy_parity": mode == "success",
            "cleanup_passed": True,
            "allocation_bytes": 32,
        }
        receipt = child(
            name,
            [
                r.PY,
                "-c",
                "print(" + repr("bad" if mode == "bad_json" else json.dumps(primitive)) + ")",
            ],
            raw,
        )
        if mode == "timeout":
            receipt.update(passed=False, timed_out=True)
        return receipt

    monkeypatch.setattr(q, "child", run)
    result = q.probe(
        {"permitted_uuid": "GPU-test", "libraries": [], "driver_library": "/missing"}, tmp_path
    )
    assert result["probe_count"] == int(mode != "busy")
    assert result["ready"] == (mode == "success")
    assert bool(result["cleanup_receipts"]) == (mode != "busy")
    assert len(calls) <= 1


def test_missing_authority_and_receipt(tmp_path):
    """SCENARIO-REPORT-8410-REPLAY: missing upstream is blocked, never retryable partial."""
    work = q.measure(tmp_path / "absent", tmp_path / "raw")
    receipt = child("owned", [r.PY, "-c", "print('ok')"], tmp_path / "checks")
    value = q.build(work, tmp_path / "raw", [receipt])
    assert value["verdict_class"] == "blocked"
    assert value["runtime_reader_ready_score"] == 0
    assert any(g["observed"] is None for g in value["gate_check_summary"])
    assert value["current_device_probe_count"] == 0
    repair = tmp_path / "repair.json"
    atomic_json(repair, {})
    work = q.measure(q.ROOT, tmp_path / "invalid", repair, "sha256:" + "0" * 64)
    assert any(g["artifact_field"] == "explicit_operator_repair" for g in work["failures"])


def test_changed_inventory_admission(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8410-DELTA: an actual content delta admits exactly one owned probe."""
    original = q.inventory

    def changed(previous):
        observed = original(previous)
        observed["kernel_driver"] += "changed"
        return observed

    monkeypatch.setattr(q, "inventory", changed)
    probe = MagicMock(
        return_value={
            "probe_count": 1,
            "ready": True,
            "rows": [],
            "checks": [],
            "lease_receipt": {"owned": True},
            "cleanup_receipts": [{"released": True}],
        }
    )
    monkeypatch.setattr(q, "probe", probe)
    work = q.measure(q.ROOT, tmp_path)
    assert probe.call_count == 1
    receipt = child("owned", [r.PY, "-c", "print('ok')"], tmp_path / "checks")
    value = q.build(work, tmp_path, [receipt])
    assert value["runtime_changed_score"] == value["cuda_context_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert value["generalized_learning_benefit_score"] == 0


def test_real_cli_and_manifest(tmp_path):
    """SCENARIO-VERIFY-8410-VALIDATION: actual CLI failures and child shards remain auditable."""
    private = tmp_path / "validation"
    private.mkdir()
    plan = r.manifest(private)
    assert not any(s["scope"] == "global" for s in plan)
    assert any(s["name"] == "private_E2E018" for s in plan)
    output = tmp_path / (q.NAME + ".json")
    receipt = child(
        "private_cli",
        [
            r.PY,
            "-u",
            str(q.ROOT / q.CLI),
            "--date",
            "20261011",
            "--private-run",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert q.replay(output)
    assert json.loads(output.read_bytes())["current_device_probe_count"] == 0
    for args, expected in [
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--deliberate-error"], 7),
        (["--date", "20261010"], 2),
        (["--private-run"], 2),
        (["--cuda-probe", "driver", "--library", "/missing"], 0),
    ]:
        receipt = child(
            "cli_" + str(expected) + "_" + str(len(args)),
            [r.PY, "-u", str(q.ROOT / q.CLI), *args],
            tmp_path / "log-controls",
            expected=expected,
            deadline=60,
        )
        assert receipt["passed"]


def test_real_owned_child(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8410-VALIDATION: an actual child records unavailable-driver failure."""
    lease = MagicMock()
    lease.owner_receipt.return_value = {"owned": True}
    lease.release.return_value = {"released": True}
    monkeypatch.setattr(q.GpuLease, "acquire", lambda **kwargs: lease)
    result = q.probe({"permitted_uuid": "GPU-test", "driver_library": "/missing"}, tmp_path)
    assert result["probe_count"] == 1 and not result["ready"]
    assert result["rows"][0]["primitive"]["error"]
    assert result["cleanup_receipts"] == [{"released": True}]


def test_repair_authority(tmp_path):
    """SCENARIO-VERIFY-8410-DELTA: a receipt must bind actual current bytes and time."""
    current = {"boot_identity": "current"}
    current_path = tmp_path / "current.json"
    atomic_json(current_path, current)
    document = {
        "authority": "operator",
        "previous_primary_sha256": q.CACHED_PIN,
        "executed_at_ns": 20,
        "action": "reboot",
        "description": "operator supplied",
        "current_reference": q.reference(current_path),
    }
    path = tmp_path / "repair.json"
    atomic_json(path, document)
    assert q.repair(path, q.sha256_file(path), current, 30, 10) == document
    with pytest.raises(ValueError, match="inventory_drift"):
        q.repair(path, q.sha256_file(path), {}, 30, 10)
    with pytest.raises(ValueError, match="authority"):
        q.repair(path, q.sha256_file(path), current, 30, 25)


def test_rehashed_primitive_controls(tmp_path):
    """SCENARIO-REPORT-8410-REPLAY: changing primitive operands and headline hashes is rejected."""
    raw = tmp_path / "raw"
    work = q.measure(q.ROOT, raw)
    receipt = child("owned", [r.PY, "-c", "print('ok')"], raw / "checks")
    value = q.build(work, raw, [receipt])
    candidate = tmp_path / "candidate.json"
    original_bytes = (raw / "measurement.json").read_bytes()
    for mutate in [
        lambda w: w["cached"]["previous"].update(kernel_driver="forged"),
        lambda w: w["cached"]["diagnosis"].update(root_cause_status="forged"),
        lambda w: w["family"].update(ready=False),
        lambda w: w["current"].update(kernel_driver="forged"),
        lambda w: w["authority"]["tasks"][0].update(title="forged"),
        lambda w: w["authority"].update(activated=False),
    ]:
        changed = deepcopy(work)
        mutate(changed)
        atomic_json(raw / "measurement.json", changed)
        bad = q.build(changed, raw, [receipt])
        atomic_json(candidate, bad)
        assert not q.replay(candidate)
        (raw / "measurement.json").write_bytes(original_bytes)
    value["run_date"] = "20261010"
    value["reproducibility_checksum"] = q.checksum(value)
    atomic_json(candidate, value)
    assert not q.replay(candidate)


def test_resource_and_cache_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8410-VALIDATION: unavailable resources stop owned measurement."""
    resources = q.base.resources
    monkeypatch.setattr(
        q.base, "resources", lambda raw: dict(resources(raw), private_disk_backed_scratch=False)
    )
    work = q.measure(q.ROOT, tmp_path / "resources")
    assert any(g["artifact_field"] == "private_disk_memory_budget" for g in work["failures"])
    analyze = q.old.analyze
    monkeypatch.setattr(
        q.old, "analyze", lambda text, observed: dict(analyze(text, observed), correction="forged")
    )
    with pytest.raises(ValueError, match="cached_primitive_drift"):
        q.cached(q.ROOT / q.CACHED, tmp_path / "cached")


def test_actual_private_lease(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8410-VALIDATION: real lease cleanup cannot signal unrelated processes."""
    acquire = q.GpuLease.acquire

    def private_lease(**kwargs):
        kwargs["runtime_dir"] = str(tmp_path / "leases")
        return acquire(**kwargs)

    monkeypatch.setattr(q.GpuLease, "acquire", private_lease)
    result = q.probe(
        {"permitted_uuid": "GPU-fixture", "driver_library": "/missing"}, tmp_path / "child"
    )
    assert result["probe_count"] == 1 and not result["ready"]
    assert result["cleanup_receipts"][0]["released"]
    assert result["cleanup_receipts"][0]["signals_sent"] == []


def test_successful_explicit_receipt(tmp_path):
    """SCENARIO-VERIFY-8410-DELTA: even an authentic receipt cannot replace missing old operands."""
    observed = q.inventory(q.cached(q.ROOT / q.CACHED, tmp_path / "cached")["previous"])
    current = tmp_path / "current.json"
    atomic_json(current, observed)
    receipt = tmp_path / "repair.json"
    atomic_json(
        receipt,
        {
            "authority": "operator",
            "previous_primary_sha256": q.CACHED_PIN,
            "executed_at_ns": q.time.time_ns() - 1,
            "action": "reboot",
            "description": "private receipt control only",
            "current_reference": q.reference(current),
        },
    )
    work = q.measure(q.ROOT, tmp_path / "measure", receipt, q.sha256_file(receipt), fixture=True)
    assert work["repair_receipt"] and not work["failures"]
    assert work["probe"]["probe_count"] == 0


def test_probe_cold_reduction(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8410-REPLAY: cold readers check real child output, readiness and pins."""
    raw = tmp_path / "raw"
    work = q.measure(tmp_path / "absent", raw)
    lease = MagicMock()
    lease.owner_receipt.return_value = {"owned": True}
    lease.release.return_value = {"released": True}
    monkeypatch.setattr(q.GpuLease, "acquire", lambda **kwargs: lease)
    work["probe"] = q.probe(
        {"permitted_uuid": "GPU-fixture", "driver_library": "/missing"}, raw / "probe"
    )
    receipt = child("owned", [r.PY, "-c", "print('ok')"], raw / "check")
    candidate = tmp_path / "candidate.json"
    for mutation in ["match", "primitive", "ready", "pin"]:
        changed = deepcopy(work)
        if mutation == "primitive":
            changed["probe"]["rows"][0]["primitive"]["error"] = "forged"
        if mutation == "ready":
            changed["probe"]["ready"] = True
        if mutation == "pin":
            primitive = raw / "wrong_protocol.json"
            atomic_json(primitive, {"wrong": True})
            changed["refs"][0].update(
                exists=True, snapshot_path=str(primitive), sha256=q.sha256_file(primitive)
            )
        atomic_json(raw / "measurement.json", changed)
        atomic_json(candidate, q.build(changed, raw, [receipt]))
        assert q.replay(candidate) == (mutation == "match")


def test_normal_runner_coverage_path(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8410-VALIDATION: real Coverage.py bytes survive a normal runner path."""
    import sys

    output = tmp_path / (q.NAME + ".json")

    def plan(private):
        private.mkdir(exist_ok=True)
        code = (
            "import coverage;from carnot.verify import cuda_reader_8410 as q;c=coverage.Coverage(config_file=False,data_file="
            + repr(str(private / ".coverage.control"))
            + ');c.start();q.progress("coverage_control");c.stop();c.save();c.json_report(outfile='
            + repr(str(private / "coverage.json"))
            + ")"
        )
        return [
            {
                "name": "actual_coverage",
                "argv": [r.PY, "-u", "-c", code],
                "deadline": 60,
                "expected": 0,
                "scope": "owned",
            }
        ]

    monkeypatch.setattr(r, "manifest", plan)
    monkeypatch.setattr(
        sys, "argv", [q.CLI, "--root", str(tmp_path / "absent"), "--output", str(output)]
    )
    assert r.main() == 0
    assert q.replay(output)
    raw = Path(json.loads(output.read_bytes())["primitive_reference"]["path"]).parent
    assert list((raw / "coverage_shards").iterdir())


def test_upstream_readiness_gate(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8410-VALIDATION: a zero family field fails even if custody is mocked."""
    primary = tmp_path / "zero.json"
    atomic_json(primary, {"historical_runtime_consumers_ready_score": 0})
    checked = q.h.checked
    monkeypatch.setattr(
        q.h,
        "checked",
        lambda path, digest: primary if str(path).endswith(q.UPSTREAM) else checked(path, digest),
    )
    work = q.measure(q.ROOT, tmp_path / "raw")
    assert any(
        g["observed"] == "historical_runtime_consumers_ready_score" for g in work["failures"]
    )


def test_independent_full_task_pin(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8410-REPLAY: an erroneous authority reader cannot bless a changed task."""
    raw = tmp_path / "raw"
    work = q.measure(q.ROOT, raw)
    work["authority"]["tasks"][8]["prompt"] += " forged"
    monkeypatch.setattr(q.h, "authority", lambda root, receipts: work["authority"])
    atomic_json(raw / "measurement.json", work)
    receipt = child("owned", [r.PY, "-c", "print('ok')"], raw / "check")
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, q.build(work, raw, [receipt]))
    assert not q.replay(candidate)

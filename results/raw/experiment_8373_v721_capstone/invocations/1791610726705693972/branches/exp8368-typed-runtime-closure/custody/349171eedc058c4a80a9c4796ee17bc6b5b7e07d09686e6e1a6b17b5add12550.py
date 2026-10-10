"""REQ-VERIFY-8307 / REQ-REPORT-8307: private evidence grants no live readiness."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest
import yaml

from carnot.verify import runtime_change_boundary_8307 as q
from carnot.verify import runtime_change_execution_8307 as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.evidence_features_custody_7980 import reference


@pytest.fixture
def historical_authority(tmp_path):
    """SCENARIO-VERIFY-8353-AUTHORITY: frozen tasks avoid mutable current activation."""
    from carnot.reporting.roadmap_contract import parse_design

    root = tmp_path / "historical_authority"
    design = root / "openspec/change-proposals/research-roadmap-vNEXT.md"
    design.parent.mkdir(parents=True)
    design.write_bytes(
        (
            q.ROOT / "openspec/change-proposals/research-roadmap-v717-preserved-20261008.md"
        ).read_bytes()
    )
    tasks = parse_design(design.read_text(), milestone="2026.10.717")[1]
    (root / "research-roadmap.yaml").write_text(
        yaml.safe_dump(dict(milestone="2026.10.717", tasks=tasks))
    )
    return root


def fixture(tmp_path):
    """Freeze private identities so changes cannot be confused with GPU observations."""
    path = tmp_path / "operand"
    path.write_bytes(b"private fixture bytes")
    identity = dict(
        driver="driver-a",
        kernel="kernel-a",
        devices=[dict(uuid="GPU-abc", index="0")],
        masks={},
        libraries=[dict(path=str(path), sha256=sha256_file(path))],
        nodes=[],
        permitted_uuid="GPU-abc",
        binary=str(path),
    )
    return dict(
        previous=identity,
        current=deepcopy(identity),
        checks=[],
        refs=[reference(path)],
        historical=dict(failure_layer=[dict(layer="driver", failed_api=dict(cuInit=101))]),
        diagnostic=dict(rows=[], lease_receipt={}, cleanup_receipt={}),
        receipt_rows=[],
        fixture=True,
        observation_receipts=[],
    )


def cli(tmp_path, *args):
    """Execute the standalone CLI in scratch; preserve subprocess coverage when requested."""
    command = [str(q.ROOT / ".venv/bin/python")]
    config = os.environ.get("CARNOT8307_COVERAGE_CONFIG")
    if config:
        command += ["-m", "coverage", "run", "-p", "--rcfile", config]
    command += [str(q.ROOT / q.CLI), *map(str, args)]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )


def test_unchanged_and_missing(tmp_path):
    """SCENARIO-VERIFY-8307-BOUNDARY: no delta or missing evidence cannot schedule a probe."""
    data = fixture(tmp_path)
    with patch.object(q, "probe_changed", side_effect=AssertionError("forbidden probe")):
        value = q.reduce(data, True)
    assert value["honest_verdict"] == "complete_blocked_cuda_runtime"
    assert value["current_probe_count"] == 0
    assert value["runtime_changed_score"] == value["cuda_context_ready_score"] == 0
    data["checks"] = [q.gate(tmp_path / "absent", "exists", True, False)]
    assert q.reduce(data, True)["honest_verdict"] == "complete_blocked_upstream_evidence"
    assert q.reduce(data, False)["verdict_class"] == "disqualified"


def test_delta_normalization(tmp_path):
    """SCENARIO-VERIFY-8307-BOUNDARY: UUID spelling and clocks cannot become repair evidence."""
    data = fixture(tmp_path)
    data["current"]["permitted_uuid"] = "gpu-ABC"
    data["current"]["devices"][0]["uuid"] = "gpu-ABC"
    assert not any(r["potentially_causal"] for r in q.delta_rows(data))
    data["current"]["kernel"] = "kernel-b"
    assert q.reduce(data, True)["runtime_changed_score"] == 1
    data["current"]["kernel"] = None
    assert not any(r["potentially_causal"] for r in q.delta_rows(data))


def test_ready_requires_all_layers(tmp_path):
    """REQ-VERIFY-8307: native enumeration cannot replace context and byte-copy parity."""
    data = fixture(tmp_path)
    data["current"]["driver"] = "driver-b"
    rows = [
        dict(
            layer=layer,
            binding="explicit_uuid",
            receipt=dict(passed=True, normal_exit=True),
            primitive=dict(
                context_copy_ready=True,
                api_returns=dict(
                    cuInit=0,
                    cudaGetDeviceCount=0,
                    cuCtxCreate_v2=0,
                    cuMemAlloc_v2=0,
                    cuMemcpyHtoD_v2=0,
                    cuMemcpyDtoH_v2=0,
                ),
                device_count=1,
                native_compatible=True,
                byte_copy_parity=True,
                cleanup_passed=True,
                allocation_bytes=32,
                copy_source_hex=bytes(range(32)).hex(),
                copy_target_hex=bytes(range(32)).hex(),
                environment=dict(CUDA_VISIBLE_DEVICES="GPU-abc"),
            ),
        )
        for layer in ["driver", "runtime", "native"]
    ]
    data["diagnostic"].update(
        rows=rows, cleanup_receipt=dict(released=True), lease_receipt=dict(device_uuid="GPU-abc")
    )
    assert q.reduce(data, True)["cuda_context_ready_score"] == 1
    rows[0]["primitive"]["context_copy_ready"] = False
    assert q.reduce(data, True)["cuda_context_ready_score"] == 0
    rows.pop()
    assert q.reduce(data, True)["cuda_context_ready_score"] == 0


def test_private_cli_and_tamper(tmp_path):
    """SCENARIO-REPORT-8307-REPLAY: cold children reject even a rehashed headline edit."""
    data = fixture(tmp_path)
    source = tmp_path / "fixture.json"
    atomic_json(source, data)
    output = tmp_path / (q.NAME + ".json")
    run = cli(tmp_path, "--date", "20261008", "--input", source, "--output", output)
    assert run.returncode == 0, run.stdout + run.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["fixture_mode"]
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["runtime_changed_score"] = 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--date", "20261007").returncode == 2
    assert cli(tmp_path, "--input", source).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing.json").returncode == 1


def test_authentication_and_inventory(tmp_path):
    """REQ-VERIFY-8307: byte-bound historic failure rows stay independent of inventory."""
    absent = q.load(tmp_path, tmp_path / "raw")
    assert absent["checks"] and not absent["historical"]
    with patch.object(q, "inventory", return_value=(fixture(tmp_path)["current"], [])):
        actual = q.load(q.ROOT, tmp_path / "live_inputs")
    assert all(r["passed"] for r in actual["checks"]), actual["checks"]
    assert actual["historical"]["root_cause_status"] == "unproved"
    assert actual["previous"]["driver"] == "615.71.09"
    assert q.verify_primitives(actual) is None
    bad = deepcopy(actual)
    bad["previous"]["driver"] = "invented"
    with pytest.raises(ValueError):
        q.verify_primitives(bad)


def test_changed_probe_and_lease_failure(tmp_path):
    """SCENARIO-VERIFY-8307-BOUNDARY: changed conditions run each child once under one lease."""
    data = fixture(tmp_path)
    data["current"].update(runtime_library="fixture-runtime", driver_library="fixture-driver")
    stream = tmp_path / "child.stdout"
    stream.write_text(json.dumps(dict(context_copy_ready=False)))
    receipt = dict(passed=True, normal_exit=True, stdout_path=str(stream))
    lease = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    lease.owner_receipt.return_value = dict(device_uuid="GPU-abc")
    lease.release.return_value = dict(released=True)
    lease.document = dict(phase="terminal_blocked")
    with (
        patch.object(q.GpuLease, "acquire", return_value=lease),
        patch.object(q, "child", return_value=receipt) as run,
    ):
        diag = q.probe_changed(data, tmp_path / "probe")
    assert run.call_count == 3 and len(diag["rows"]) == 3
    assert diag["cleanup_receipt"]["released"]
    with patch.object(q.GpuLease, "acquire", side_effect=q.LeaseError("occupied")):
        failed = q.probe_changed(data, tmp_path / "occupied")
    assert failed["checks"] and not failed["rows"]


def test_inventory_read_only(tmp_path):
    """REQ-VERIFY-8307: inventory measures identities without invoking CUDA or native contexts."""
    data = fixture(tmp_path)
    stream = tmp_path / "inventory.stdout"
    stream.write_text("0, GPU-abc, private GPU, 615.71.09\n")
    receipt = dict(passed=True, stdout_path=str(stream))
    with patch.object(q, "child", return_value=receipt):
        current, receipts = q.inventory(data["previous"], tmp_path / "inv")
    assert current["driver"] == "615.71.09" and receipts == [receipt]
    stream.write_text("not an inventory\n")
    with patch.object(q, "child", return_value=receipt):
        assert q.inventory(data["previous"], tmp_path)[0]["driver"] is None


def test_failed_validation_and_recovery(tmp_path):
    """SCENARIO-VERIFY-8307-VALIDATION: failed owned checks clear readiness and can recover."""
    data = fixture(tmp_path)
    source = tmp_path / "fixture.json"
    atomic_json(source, data)
    output = tmp_path / (q.NAME + ".json")
    with patch.object(e, "validate", return_value=dict(passed=False, checks=[])):
        assert e.main(["--input", str(source), "--output", str(output)]) == 1
    assert not output.exists()
    assert e.main(["--input", str(source), "--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    value["reproducibility_checksum"] = "sha256:bad"
    atomic_json(output, value)
    with pytest.raises(ValueError, match="checksum"):
        e.replay(output)


def test_authority_and_frozen_plan(tmp_path, historical_authority):
    """REQ-REPORT-8307: activation and all exact validation paths freeze before measurement."""
    auth = e.authority(historical_authority, tmp_path / "authority")
    assert auth["activated"] and auth["task"]["deliverable"] == "results/" + q.NAME + ".json"
    assert not e.authority(tmp_path, tmp_path / "missing")["activated"]
    commands = e.plan(tmp_path / "checks")
    assert len(commands) == 10
    assert all("repository_health" not in c["name"] for c in commands)
    assert "--fail-under=100" in commands[3]["argv"]


def test_probe_cli_and_cleanup(tmp_path):
    """SCENARIO-VERIFY-8307-VALIDATION: actual child dispatch and exceptional cleanup are owned."""
    native = tmp_path / "native"
    native.write_text("#!/bin/sh\nprintf 'CUDA0 private fixture\\n'\n")
    native.chmod(0o700)
    run = cli(tmp_path, "--cuda-probe", "native", "--binary", native)
    assert run.returncode == 0 and json.loads(run.stdout.splitlines()[-1])["native_compatible"]
    data = fixture(tmp_path)
    data["current"]["devices"] = []
    assert q.probe_changed(data, tmp_path / "absent")["checks"]
    data = fixture(tmp_path)
    data["current"].update(driver_library="libcuda.so.1", runtime_library="libcudart.so.12")
    lease = __import__("unittest.mock", fromlist=["MagicMock"]).MagicMock()
    lease.owner_receipt.return_value = {}
    lease.release.return_value = dict(released=True)
    lease.document = dict(phase="admitted")
    with (
        patch.object(q.GpuLease, "acquire", return_value=lease),
        patch.object(q, "child", side_effect=OSError("child failed")),
    ):
        diag = q.probe_changed(data, tmp_path / "cleanup")
    assert diag["cleanup_receipt"]["released"]
    lease.transition.assert_called_with("terminal_blocked")


def test_authentication_negative_receipts(tmp_path):
    """SCENARIO-VERIFY-8307-BOUNDARY: sidecar, terminal and primitive contradictions block."""
    with patch.object(q, "read_bound_sidecar", return_value=dict(report=dict(passed=False))):
        assert any(not c["passed"] for c in q.load(q.ROOT, tmp_path / "auditor")["checks"])
    with patch.object(
        q,
        "read_bound_sidecar",
        return_value=dict(report=dict(passed=True, checks=[dict(passed=False, normal_exit=True)])),
    ):
        assert any(not c["passed"] for c in q.load(q.ROOT, tmp_path / "auditor_exit")["checks"])
    original = json.loads
    for case in ("owned", "terminal", "probe"):

        def altered(blob):
            value = original(blob)
            if case == "owned" and isinstance(value, dict) and value.get("experiment_id") == 8290:
                value["required_checks_passed"] = False
            if case == "terminal" and isinstance(value, dict) and "normal_process_exit" in value:
                value["normal_process_exit"] = False
            if case == "probe" and isinstance(value, dict) and value.get("layer") == "driver":
                value["context_copy_ready"] = True
            return value

        with patch.object(q.json, "loads", side_effect=altered):
            data = q.load(q.ROOT, tmp_path / case)
        assert any(not c["passed"] for c in data["checks"])


def test_operator_receipt_and_identity_replay(tmp_path):
    """REQ-VERIFY-8307: operator actions need exact baseline bytes and a bounded execution date."""
    repair = tmp_path / "operator.json"
    source = json.loads((q.ROOT / q.UPSTREAM).read_bytes())
    receipt = dict(
        authority="operator",
        previous_primary_sha256=q.PIN,
        action="driver_repair",
        executed_at_ns=q.frontier_wall_ns(source) + 1,
        description="private receipt control",
    )
    atomic_json(repair, receipt)
    with (
        patch.object(q, "REPAIR", str(repair)),
        patch.object(q, "inventory", return_value=(fixture(tmp_path)["current"], [])),
    ):
        data = q.load(q.ROOT, tmp_path / "repair_inputs")
    assert data["receipt_rows"][0]["eligible"]
    assert not q.eligible_repair(
        dict(receipt, executed_at_ns=0), source, q.frontier_wall_ns(source) + 10
    )
    assert not q.eligible_repair(
        dict(receipt, executed_at_ns=q.frontier_wall_ns(source) + 11),
        source,
        q.frontier_wall_ns(source) + 10,
    )
    assert not q.eligible_repair(
        dict(receipt, authority="task"), source, q.frontier_wall_ns(source) + 10
    )
    bad = deepcopy(data)
    bad["receipt_rows"][0]["eligible"] = False
    with pytest.raises(ValueError, match="operator_repair"):
        q.verify_primitives(bad)
    current = tmp_path / "current.json"
    atomic_json(current, data["current"])
    data["current_reference"] = reference(current)
    q.verify_primitives(data)
    data["current"]["driver"] = "tamper"
    with pytest.raises(ValueError, match="current_identity"):
        q.verify_primitives(data)


def test_owned_main_receipts_coverage_and_rehashed_controls(tmp_path, historical_authority):
    """SCENARIO-REPORT-8307-REPLAY: real supervised checks and private custody survive cold replay."""
    data = fixture(tmp_path)
    data["current"]["kernel"] = "kernel-b"
    output = tmp_path / (q.NAME + ".json")
    original_plan = e.plan

    def private_plan(private):
        original_plan(private)
        atomic_json(private / "coverage.json", dict(totals=dict(percent_covered=100), files={}))
        (private / ".coverage.fixture").write_bytes(b"constructed private coverage custody control")
        return [
            dict(
                name="private_owned_child",
                argv=[str(q.ROOT / ".venv/bin/python"), "-c", "print('private owned child')"],
                deadline=15,
                expected=0,
                scope="private_control",
            )
        ]

    with (
        patch.object(e, "plan", side_effect=private_plan),
        patch.object(q, "load", return_value=data),
        patch.object(q, "probe_changed", return_value=data["diagnostic"]) as probe,
    ):
        assert e.main(["--root", str(historical_authority), "--output", str(output)]) == 0
    assert probe.call_count == 1
    value = json.loads(output.read_bytes())
    assert value["validation_receipts"][0]["passed"]
    assert e.replay(output)["passed"]
    for field, replacement, message in [
        ("experiment_id", 9999, "identity"),
        ("required_checks_passed", False, "owned_check"),
        ("MODEL_SPECS", [dict(model_id="fixture")], "model_provenance"),
    ]:
        bad = deepcopy(value)
        bad[field] = replacement
        bad["reproducibility_checksum"] = e.checksum(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError, match=message):
            e.replay(output)


def test_probe_stream_replay(tmp_path):
    """SCENARIO-REPORT-8307-REPLAY: rehashed primitive edits still disagree with actual child bytes."""
    data = fixture(tmp_path)
    stdout, stderr = tmp_path / "stdout", tmp_path / "stderr"
    p = dict(layer="driver", context_copy_ready=False)
    stdout.write_text(json.dumps(p))
    stderr.write_bytes(b"")
    data["diagnostic"]["rows"] = [
        dict(
            layer="driver",
            primitive=p,
            receipt=dict(
                passed=True,
                stdout_path=str(stdout),
                stdout_sha256=sha256_file(stdout),
                stderr_path=str(stderr),
                stderr_sha256=sha256_file(stderr),
            ),
        )
    ]
    q.verify_primitives(data)
    data["diagnostic"]["rows"][0]["primitive"] = dict(p, context_copy_ready=True)
    with pytest.raises(ValueError, match="current_probe"):
        q.verify_primitives(data)


def test_missing_library_is_not_change(tmp_path):
    """SCENARIO-VERIFY-8307-BOUNDARY: missing binaries or failed inventory stay external blocks."""
    data = fixture(tmp_path)
    data["current"]["libraries"][0]["sha256"] = None
    assert not any(r["potentially_causal"] for r in q.delta_rows(data))
    data["observation_receipts"] = [
        dict(passed=False, stdout_path=str(tmp_path / "missing.stdout"))
    ]
    assert q.reduce(data, True)["honest_verdict"] == "complete_blocked_upstream_evidence"


def test_library_resolution_labels(tmp_path):
    """REQ-VERIFY-8307: symlink spelling grants no change without a byte delta."""
    data = fixture(tmp_path)
    library = tmp_path / "libcuda.so.123"
    library.write_bytes(b"fixture library")
    (tmp_path / "libcuda.so.1").symlink_to(library)
    data["previous"]["libraries"] = [reference(library)]
    stream = tmp_path / "inventory.stdout"
    stream.write_text("0, GPU-abc, private GPU, driver-a\n")
    with patch.object(q, "child", return_value=dict(passed=True, stdout_path=str(stream))):
        current, _ = q.inventory(data["previous"], tmp_path / "inventory")
    assert current["libraries"] == data["previous"]["libraries"]


def test_restricted_mask_and_absent_observation(tmp_path):
    """REQ-VERIFY-8307: intentional restrictions and failed preconditions cannot be bypassed."""
    data = fixture(tmp_path)
    data["current"]["masks"] = dict(CUDA_VISIBLE_DEVICES="")
    with patch.object(q.GpuLease, "acquire", side_effect=AssertionError("must not lease")):
        assert q.probe_changed(data, tmp_path / "restricted")["checks"]
    with patch.object(q, "inventory", side_effect=AssertionError("must not measure")):
        observed = q.load(q.ROOT, tmp_path / "no_measure", observe=False)
    assert observed["current"] == {} and not observed["observation_receipts"]

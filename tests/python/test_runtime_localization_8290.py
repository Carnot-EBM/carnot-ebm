"""REQ-REPORT-8290 / REQ-VERIFY-8290: private controls cannot qualify live CUDA."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import shutil
from unittest.mock import patch

import pytest
import yaml

from carnot.verify import runtime_localization_8290 as q
from carnot.verify import cuda_primitive_8290 as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """Compile a private API fixture so the actual child executes every C boundary."""
    raw = tmp_path_factory.mktemp("cuda8290")
    source = raw / "fixture.c"
    source.write_text("""
#include <stdlib.h>
#include <string.h>
static unsigned char mem[32];
int code(const char *s) { const char *v=getenv("TEST_CUDA_CASE");
 return v && strcmp(v,s)==0 ? 101 : 0; }
int cuInit(unsigned x) { const char *m=getenv("CUDA_VISIBLE_DEVICES");
 if(m && (!strcmp(m,"999") || !strcmp(m,"-1"))) return 101;
 return code("init"); }
int cuDriverGetVersion(int *v) { *v=12080; return 0; }
int cuDeviceGetCount(int *v) { *v=1; return code("count"); }
int cuDeviceGet(int *v, int i) { *v=i; return code("device"); }
int cuDeviceGetUuid(unsigned char *v, int i) { memset(v,0,16); return code("uuid"); }
int cuCtxCreate_v2(void **v,unsigned x,int d) { *v=(void*)1; return code("context"); }
int cuMemAlloc_v2(unsigned long long *v,size_t n) { *v=1; return code("allocate"); }
int cuMemcpyHtoD_v2(unsigned long long v,void *s,size_t n) {
 memcpy(mem,s,n); return code("copy"); }
int cuMemcpyDtoH_v2(void *d,unsigned long long v,size_t n) {
 memcpy(d,mem,n); if(code("parity")) memset(d,0,n); return code("copyback"); }
int cuMemFree_v2(unsigned long long v) { return code("free"); }
int cuCtxDestroy_v2(void *v) { return code("destroy"); }
int cudaRuntimeGetVersion(int *v) { *v=12080; return 0; }
int cudaGetDeviceCount(int *v) { *v=1; return code("runtime"); }
""")
    output = raw / "libfixture.so"
    subprocess.run(["cc", "-shared", "-fPIC", str(source), "-o", str(output)], check=True)
    return output


def cli(tmp_path, *args, env=None):
    """Use the direct script from private scratch without an ambient import path."""
    variables = dict(os.environ if env is None else env)
    variables.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(q.ROOT / ".venv/bin/python"), "-u", str(q.ROOT / q.CLI), *map(str, args)],
        cwd=tmp_path,
        env=variables,
        capture_output=True,
        text=True,
        timeout=65,
    )


@pytest.mark.parametrize(
    "case",
    [
        "",
        "init",
        "count",
        "device",
        "uuid",
        "context",
        "allocate",
        "copy",
        "copyback",
        "parity",
        "free",
        "destroy",
    ],
)
def test_driver_child(tmp_path, library, case):
    """SCENARIO-VERIFY-8290-LAYERS: actual child records each API failure and cleanup."""
    env = dict(os.environ, TEST_CUDA_CASE=case, CUDA_VISIBLE_DEVICES="all")
    out = cli(
        tmp_path,
        "--cuda-probe",
        "driver",
        "--library",
        library,
        "--uuid",
        "GPU-00000000-0000-0000-0000-000000000000",
        env=env,
    )
    assert out.returncode == 0, out.stderr
    value = json.loads(out.stdout.splitlines()[-1])
    assert value["context_copy_ready"] == (case == "")
    assert value["libraries"][0]["sha256"] == sha256_file(library)
    assert value["pid"] > 0 and value["pid_start_ticks"] > 0


@pytest.mark.parametrize("mask", ["999", "-1"])
def test_masks(tmp_path, library, mask):
    """SCENARIO-VERIFY-8290-LAYERS: masks fail before any permitted context exists."""
    out = cli(
        tmp_path,
        "--cuda-probe",
        "driver",
        "--library",
        library,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=mask),
    )
    assert json.loads(out.stdout.splitlines()[-1])["api_returns"]["cuInit"] == 101


@pytest.mark.parametrize("case", ["", "runtime"])
def test_runtime_child(tmp_path, library, case):
    """SCENARIO-VERIFY-8290-LAYERS: runtime enumeration is separate from a context."""
    out = cli(
        tmp_path,
        "--cuda-probe",
        "runtime",
        "--library",
        library,
        env=dict(os.environ, TEST_CUDA_CASE=case),
    )
    v = json.loads(out.stdout.splitlines()[-1])
    assert v["api_returns"]["cudaGetDeviceCount"] == (101 if case else 0)
    assert not v["context_copy_ready"]


def test_stub_node_and_native(tmp_path, library):
    """SCENARIO-VERIFY-8290-LAYERS: missing nodes, loader errors and CPU output fail."""
    assert not p.node_access([tmp_path / "missing"])[0]["read_write"]
    for layer in ("driver", "runtime"):
        assert p.probe(layer, "/missing-library", "", "")["error"]
    assert not p.probe("driver", str(library), "GPU-wrong", "")["context_copy_ready"]
    for text in ("Available devices:\n", "CUDA0: fixture"):
        binary = tmp_path / "native"
        binary.write_text("#!/bin/sh\nprintf '%s\\n' '" + text + "'\n")
        binary.chmod(0o700)
        out = cli(tmp_path, "--cuda-probe", "native", "--binary", binary)
        v = json.loads(out.stdout.splitlines()[-1])
        assert v["native_compatible"] == ("CUDA0" in text)
    assert p.probe("native", "", "", "/missing-native")["error"]


def authorities(tmp_path):
    """Keep full current task fixtures private and preserve the science bytes."""
    root = tmp_path / "repo"
    (root / Path(q.DESIGN).parent).mkdir(parents=True)
    (root / q.DESIGN).write_bytes((q.ROOT / q.DESIGN).read_bytes())
    tasks = q.parse_design((root / q.DESIGN).read_text(), milestone=q.MILESTONE)[1]
    for name in (q.ACTIVE, q.STAGED):
        (root / name).write_text(yaml.safe_dump(dict(milestone=q.MILESTONE, tasks=tasks)))
    (root / q.PROTOCOL).write_bytes((q.ROOT / q.PROTOCOL).read_bytes())
    return root


def test_authority(tmp_path):
    """SCENARIO-REPORT-8290-AUTHORITY: activation requires all fourteen full tasks."""
    root = authorities(tmp_path)
    w = q.authority_work(root, tmp_path / "raw")
    assert w["contract"]["activated"]
    assert len(w["tasks"]) == 14 and w["tasks"][0]["id"] == q.TASK
    assert w["tasks"][-1]["id"] == "exp8303-capstone"
    paths = {t["id"]: t["deliverable"] for t in w["tasks"]}
    assert all(
        g["artifact_path"] == paths[g["upstream"]]
        for t in w["execution_contract"]["producers"]
        for g in t["current_gate_fields"]
    )
    (root / q.ACTIVE).unlink()
    assert not q.authority_work(root, tmp_path / "staged")["contract"]["activated"]
    assert q.authority_work(root, tmp_path / "staged2")["contract"]["planning_matched"]
    assert q.authority_work(tmp_path / "absent", tmp_path / "missing")["failures"]


def checks():
    """Private successful command rows test reduction only, never live readiness."""
    return [
        dict(
            name=n,
            passed=True,
            exit_code=0,
            duration_s=0,
            started_monotonic_ns=1,
            ended_monotonic_ns=2,
            started_wall_ns=1,
        )
        for n in (
            "owned_unit_and_private_CLI",
            "current_contract_tests",
            "coverage_custody_tests",
            "view_component",
            "admission_component",
        )
    ]


@pytest.fixture
def work(tmp_path):
    """Use authenticated historical mechanics as a private adapter fixture."""
    old = json.loads(
        (q.ROOT / "results/experiment_8276_v715_current_contract_readiness.json").read_bytes()
    )
    value = json.loads(Path(old["work_reference"]["path"]).read_bytes())
    historical_refs = value["refs"][4:]
    value.update(q.authority_work(authorities(tmp_path), tmp_path / "authority"))
    value["refs"].extend(historical_refs)
    value.update(
        diagnostic=dict(
            rows=[],
            checks=[],
            lease_receipt={},
            cleanup_receipt={},
            device_inventory=[],
            runtime_binding={},
            phase_spans=[],
        )
    )
    return value


def test_independent_scores(work):
    """SCENARIO-VERIFY-8290-LAYERS: CUDA never lends or removes component readiness."""
    good = q.reduce(work, checks(), True, True)
    assert [good[k] for k in q.SCORES] == [1, 1, 1, 1]
    assert good["cuda_context_ready_score"] == 0
    w = deepcopy(work)
    w["diagnostic"]["checks"] = [
        dict(component="cuda", artifact_field="cuInit", expected=0, observed=101, passed=False)
    ]
    v = q.reduce(w, checks(), True, True)
    assert v["verdict_class"] == "blocked" and [v[k] for k in q.SCORES] == [1, 1, 1, 1]
    bad = checks()
    bad[-1].update(passed=False, exit_code=1)
    v = q.reduce(w, bad, True, True)
    assert v["verdict_class"] == "disqualified" and v[q.SCORES[-1]] == 0
    w["diagnostic"]["rows"] = [
        dict(
            binding=b,
            layer=l,
            primitive=dict(
                context_copy_ready=True,
                native_compatible=True,
                api_returns={"cudaGetDeviceCount": 0},
                device_count=1,
            ),
            receipt=dict(passed=True, normal_exit=True),
        )
        for b in ("inherited", "explicit_uuid")
        for l in ("driver", "runtime", "native")
    ]
    w["diagnostic"]["checks"] = []
    w["diagnostic"]["cleanup_receipt"] = dict(released=True)
    assert q.reduce(w, checks(), True, True)["cuda_context_ready_score"] == 1
    for row in w["diagnostic"]["rows"]:
        row["primitive"].update(
            context_copy_ready=False,
            native_compatible=False,
            api_returns={"cudaGetDeviceCount": 101},
        )
        row["receipt"]["stdout_path"] = work["refs"][0]["snapshot_path"]
    assert q.reduce(w, checks(), True, True)["cuda_context_ready_score"] == 0


def test_manifest_cli_and_current_controls(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8290-VALIDATION: current CLI retains genuine crash controls."""
    plan = q.manifest(tmp_path, tmp_path / "candidate.json")
    assert plan["owned"] == q.OWNED
    assert plan["commands"][0]["argv"][-1] == q.TEST
    assert cli(tmp_path, "--date", "wrong").returncode == 2
    assert cli(tmp_path, "--cold-replay", tmp_path / "absent").returncode == 1
    assert cli(tmp_path, "--coverage-replay", tmp_path / "absent").returncode == 1
    from test_protocol_conformance_8263 import test_hard_exit_pending_resume, test_capture_real_peer

    monkeypatch.setattr(q.prior.protocol, "CLI", q.CLI)
    test_hard_exit_pending_resume(tmp_path)
    test_capture_real_peer(tmp_path)
    with patch.object(q.runner, "main", return_value=0):
        assert q.main([]) == 0


def test_probe_exception_and_native_timeout(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8290-LAYERS: timeout is bounded and only the owned child dies."""
    binary = tmp_path / "slow"
    binary.write_text("#!/bin/sh\nexec sleep 30\n")
    binary.chmod(0o700)
    monkeypatch.setattr(p, "NATIVE_DEADLINE", 0.05)
    assert p.probe("native", "", "", str(binary))["timed_out"]
    with patch.object(q.runner, "main", return_value=1):
        assert q.main(["--coverage-replay", "missing"]) == 1


def test_diagnostic_private_matrix(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8290-LAYERS: fixture children test lease and matrix supervision."""
    original = q.GpuLease.acquire
    uuid = "GPU-00000000-0000-0000-0000-000000000000"

    def acquire(**kw):
        return original(**dict(kw, runtime_dir=tmp_path / "leases"))

    monkeypatch.setattr(q.GpuLease, "acquire", acquire)

    def child(name, argv, raw, **kw):
        out, err = raw / (name + ".stdout"), raw / (name + ".stderr")
        out.write_text(
            "0, " + uuid + ", fixture, 12080, 0, 24000\n"
            if name == "inventory"
            else json.dumps(dict(error=None, context_copy_ready=False))
        )
        err.write_text("")
        return dict(
            passed=True,
            normal_exit=True,
            stdout_path=str(out),
            stderr_path=str(err),
            stdout_sha256=sha256_file(out),
            stderr_sha256=sha256_file(err),
            exit_code=0,
        )

    monkeypatch.setattr(q, "child", child)
    binary = tmp_path / "native"
    binary.write_text("fixture")
    result = q.diagnostic(tmp_path / "raw", uuid, binary)
    assert len(result["rows"]) == 6 and result["cleanup_receipt"]["released"]
    assert q.diagnostic(tmp_path / "missing", "", binary)["checks"]
    lease = acquire(
        task_id="private", device_uuid=uuid, expected_model="no_model_load", vram_before_mb=0
    )
    try:
        assert q.diagnostic(tmp_path / "occupied", uuid, binary)["checks"]
    finally:
        lease.transition("terminal_blocked")
        lease.release()

    def failed_child(name, argv, raw, **kw):
        receipt = child(name, argv, raw, **kw)
        if name != "inventory":
            receipt.update(passed=False, normal_exit=False)
        return receipt

    def malformed(name, argv, raw, **kw):
        if name == "inherited_driver":
            raise ValueError("private malformed output")
        return child(name, argv, raw, **kw)

    monkeypatch.setattr(q, "child", malformed)
    assert q.diagnostic(tmp_path / "malformed", uuid, binary)["cleanup_receipt"]["released"]
    monkeypatch.setattr(q, "child", failed_child)
    assert all(
        r["primitive"]["error"] == "child_exit"
        for r in q.diagnostic(tmp_path / "exits", uuid, binary)["rows"]
    )


def test_measure_adapter_private(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8290-REPLAY: imports and current measurements stay separate."""
    monkeypatch.setattr(q.prior, "measure", lambda root, raw: deepcopy(work))
    diag = dict(work["diagnostic"], phase_spans=[dict(duration_s=0)])
    monkeypatch.setattr(q, "diagnostic", lambda *args: deepcopy(diag))
    measured = q.measure(q.ROOT, tmp_path / "measure")
    assert measured["historical_runtime"]["8277"]["verdict_class"] == "blocked"
    assert any(r["disposition"] == "cascade_skip_absent_primary" for r in measured["history"])
    original = q.snapshot

    def wrong_hash(path, raw, role):
        ref = original(path, raw, role)
        return dict(ref, sha256="wrong") if role == "stderr" else ref

    monkeypatch.setattr(q, "snapshot", wrong_hash)
    assert any(
        r["artifact_field"] == "historical_stream_hash"
        for r in q.measure(q.ROOT, tmp_path / "badstreams")["failures"]
    )


def test_build_and_replay_private(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8290-REPLAY: frozen snapshots reject rehashed primitive edits."""
    raw = tmp_path / "raw"
    raw.mkdir()
    old = json.loads(
        (q.ROOT / "results/experiment_8276_v715_current_contract_readiness.json").read_bytes()
    )
    shutil.copytree(Path(old["component_primitive_reference"]["path"]).parent, raw / "protocol")
    work.update(runtime_checks=[], invocation_argv=[q.CLI])
    atomic_json(raw / "measurement.json", work)
    for name in ["validation_commands.json", "validation_receipts.json"]:
        atomic_json(raw / name, {})
    stream = raw / "private.stdout"
    stream.write_text("private fixture")
    receipts = [
        dict(
            r,
            stdout_path=str(stream),
            stderr_path=str(stream),
            stdout_sha256=sha256_file(stream),
            stderr_sha256=sha256_file(stream),
        )
        for r in checks()
    ]
    with patch.object(q.runner, "q", q):
        value = q.build(work, raw, receipts, {}, [], tmp_path / "removed", {})
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert q.replay(candidate)["passed"]

    def check(v, message):
        v.pop("reproducibility_checksum", None)
        v["reproducibility_checksum"] = canonical_hash(v)
        atomic_json(candidate, v)
        with pytest.raises(ValueError, match=message):
            q.replay(candidate)

    bad = deepcopy(value)
    bad["experiment_id"] = 0
    check(bad, "invocation_identity")
    bad = deepcopy(value)
    bad["scratch_removed"] = False
    check(bad, "durable_coverage_reduction")
    bad = deepcopy(value)
    bad["cuda_context_ready_score"] = 1
    check(bad, "primitive_readiness_drift")
    bad = deepcopy(value)
    bad["MODEL_SPECS"] = [{}]
    check(bad, "current_model_provenance")
    atomic_json(candidate, dict(value, reproducibility_checksum="bad"))
    with pytest.raises(ValueError, match="candidate_checksum"):
        q.replay(candidate)
    with patch.object(q.prior.protocol, "replay", return_value=False):
        check(deepcopy(value), "protocol_primitive_drift")
    private_stream = tmp_path / "stream"
    private_stream.write_text("actual")
    receipt = dict(
        stdout_path=str(private_stream),
        stderr_path=str(private_stream),
        stdout_sha256="bad",
        stderr_sha256="bad",
    )
    bad = deepcopy(value)
    bad["cold_replay_rows"] = [receipt]
    check(bad, "validation_stream_hash")

    def work_tamper(mutate, message):
        changed = deepcopy(work)
        mutate(changed)
        atomic_json(raw / "measurement.json", changed)
        bad = deepcopy(value)
        derived = q.reduce(changed, receipts, False, False)
        derived.pop("component_primitive_reference")
        bad.update(derived)
        bad["work_reference"]["sha256"] = sha256_file(raw / "measurement.json")
        check(bad, message)
        atomic_json(raw / "measurement.json", work)

    work_tamper(
        lambda w: w["contract"].update(canonical_tasks_sha256="wrong"), "authority_reduction_drift"
    )
    work_tamper(lambda w: w["tasks"][0].update(prompt="changed"), "full_task_primitive_drift")
    work_tamper(
        lambda w: w["history"][0]["source_counts"].update(intended_count=-1),
        "historical_primitive_drift",
    )
    stream = tmp_path / "cuda.stdout"
    stream.write_text(json.dumps(dict(context_copy_ready=False)))
    row = dict(
        binding="inherited",
        layer="driver",
        primitive=dict(context_copy_ready=False),
        receipt=dict(
            passed=True,
            normal_exit=True,
            stdout_path=str(stream),
            stderr_path=str(stream),
            stdout_sha256="wrong",
            stderr_sha256=sha256_file(stream),
        ),
    )
    work_tamper(lambda w: w["diagnostic"].update(rows=[row]), "cuda_stream_hash")
    row["receipt"]["stdout_sha256"] = sha256_file(stream)
    row["primitive"] = dict(context_copy_ready=False, error="contradiction")
    work_tamper(lambda w: w["diagnostic"].update(rows=[row]), "cuda_primitive_drift")
    row["primitive"] = dict(context_copy_ready=False)
    changed = deepcopy(work)
    changed["diagnostic"]["rows"] = [row]
    atomic_json(raw / "measurement.json", changed)
    valid = deepcopy(value)
    derived = q.reduce(changed, receipts, False, False)
    derived.pop("component_primitive_reference")
    valid.update(derived)
    valid["work_reference"]["sha256"] = sha256_file(raw / "measurement.json")
    valid.pop("reproducibility_checksum")
    valid["reproducibility_checksum"] = canonical_hash(valid)
    atomic_json(candidate, valid)
    assert q.replay(candidate)["passed"]
    assert cli(tmp_path, "--cold-replay", candidate).returncode == 0

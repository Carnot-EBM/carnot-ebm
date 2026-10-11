"""REQ-VERIFY-8396 and REQ-REPORT-8396: cause claims need owned child evidence."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.verify import cuda_failure_cause_8396 as q
from carnot.verify import cuda_failure_runner_8396 as e


@pytest.mark.parametrize(
    "text,code,cause",
    [
        (
            'openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR) = -1 EACCES (Permission denied)',
            101,
            "device_open_denied",
        ),
        (
            'openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR) = -1 ENOENT (No such file)',
            101,
            "device_namespace_missing",
        ),
        (
            "ioctl(3</dev/nvidiactl>, 0xc020462a, 0x123) = -1 ENODEV (No such device)",
            101,
            "device_ioctl_rejected",
        ),
        ('openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR) = 3', 101, "cause_unproved"),
        ("ioctl(3</dev/nvidiactl>, 0x1) = -1 EINVAL", 0, "cause_unproved"),
    ],
)
def test_trace(text, code, cause):
    """SCENARIO-VERIFY-8396-TRACE: only a failed initialization permits causal attribution."""
    result = q.analyze(text, dict(api_returns=dict(cuInit=code), libraries=[], environment={}))
    assert result["root_cause_status"] == cause
    assert result["runtime_changed_score"] == 0


def test_binding_and_library_causes():
    """SCENARIO-VERIFY-8396-TRACE: a distinct loaded stub or mask gets one correction."""
    p = dict(
        api_returns=dict(cuInit=101),
        libraries=[dict(path="/opt/cuda/stubs/libcuda.so")],
        environment={},
    )
    assert q.analyze("", p)["correction"] == "installed_library"
    p.update(libraries=[], environment=dict(CUDA_VISIBLE_DEVICES="-1"))
    assert q.analyze("", p)["correction"] == "process_binding"
    assert q.analyze("", dict(p, environment=dict(CUDA_VISIBLE_DEVICES="0")))["correction"] is None


@pytest.mark.parametrize(
    "mode", ["normal", "busy", "missing_tool", "parse", "timeout", "corrected", "correction_failed"]
)
def test_diagnose(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8396-TRACE: supervision preserves all exits and releases owned leases."""
    lease = MagicMock()
    monkeypatch.setattr(
        q.GpuLease,
        "acquire",
        MagicMock(return_value=lease, side_effect=ValueError("busy") if mode == "busy" else None),
    )
    monkeypatch.setattr(
        q.shutil, "which", lambda name: None if mode == "missing_tool" else "/usr/bin/" + name
    )
    calls = []

    def run(name, argv, raw, **kwargs):
        calls.append(name)
        p = dict(api_returns=dict(cuInit=101), libraries=[], environment={})
        if mode in {"corrected", "correction_failed"}:
            p["environment"] = dict(CUDA_VISIBLE_DEVICES="-1")
            if name == "corrected_init" and mode == "corrected":
                p["api_returns"]["cuInit"] = 0
            if name == "context_copy":
                p.update(
                    context_copy_ready=True,
                    byte_copy_parity=True,
                    cleanup_passed=True,
                    allocation_bytes=32,
                )
        if name == "inherited_init":
            (tmp_path / "driver.strace").write_text(
                'openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR) = 3\n'
            )
        raw.mkdir(parents=True, exist_ok=True)
        receipt = child(
            name,
            [e.PY, "-c", "print(" + repr("bad" if mode == "parse" else json.dumps(p)) + ")"],
            raw,
        )
        if mode == "timeout":
            receipt.update(passed=False, timed_out=True)
        return receipt

    monkeypatch.setattr(q, "child", run)
    result = q.diagnose(
        dict(permitted_uuid="GPU-test", driver_library="/usr/lib/libcuda.so.1"), tmp_path
    )
    assert bool(result["cleanup_receipts"]) == (mode not in {"busy", "missing_tool"})
    assert len(calls) <= 3
    assert result["runtime_changed_score"] == int(mode == "corrected")
    assert result["cuda_context_ready_score"] == int(mode == "corrected")


def test_initialization_child(monkeypatch):
    """SCENARIO-VERIFY-8396-TRACE: actual loader failure and mocked driver success remain distinct."""
    assert q.initialize("/definitely/missing")["error"]
    library = MagicMock()
    library.cuInit.return_value = 0
    monkeypatch.setattr(q.c, "CDLL", lambda path: library)
    assert q.initialize("libcuda.so.1")["api_returns"] == dict(cuInit=0)


def test_replay_and_tamper(tmp_path):
    """SCENARIO-REPORT-8396-REPLAY: rehashed headlines cannot override sealed reductions."""
    raw = tmp_path / "raw"
    work = q.measure(tmp_path / "missing", raw, fixture=True)
    receipt = child("control", [e.PY, "-c", "print('real child')"], raw / "logs")
    value = q.build(work, raw, [receipt])
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert q.replay(path)
    assert not q.replay(tmp_path / "missing.json")
    for field in (
        "runtime_changed_score",
        "cuda_context_ready_score",
        "runtime_reader_ready_score",
        "root_cause_status",
    ):
        bad = deepcopy(value)
        bad[field] = "forged"
        bad["reproducibility_checksum"] = q.checksum(bad)
        atomic_json(path, bad)
        assert not q.replay(path)
    receipt["passed"] = False
    assert q.build(work, raw, [receipt])["verdict_class"] == "disqualified"


def test_real_cli(tmp_path):
    """SCENARIO-VERIFY-8396-VALIDATION: private real CLI and child have no model load."""
    out = tmp_path / (q.NAME + ".json")
    receipt = child(
        "private_cli",
        [
            e.PY,
            "-u",
            str(q.ROOT / q.CLI),
            "--private-run",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(out),
        ],
        tmp_path / "cli",
        deadline=90,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert q.replay(out)
    receipt = child(
        "init_cli",
        [e.PY, "-u", str(q.ROOT / q.CLI), "--driver-init", "--library", "/missing"],
        tmp_path / "cli",
    )
    assert receipt["passed"] and '"cuInit"' not in Path(receipt["stdout_path"]).read_text()


def test_natural_custody_and_primitive_tamper(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8396-REPLAY: actual typed history and current authority qualify separately."""
    raw = tmp_path / "natural"
    work = q.measure(q.ROOT, raw, fixture=True)
    assert not work["failures"]
    receipt = child("owned", [e.PY, "-c", "print('ok')"], raw / "logs")
    path = tmp_path / "valid.json"
    atomic_json(path, q.build(work, raw, [receipt]))
    assert q.replay(path)
    assert q.reduction(dict(work, fixture=False), True)["runtime_reader_ready_score"] == 1
    from carnot.reporting.primary_publication import reader_receipt

    public = tmp_path / "results" / (q.NAME + ".json")
    atomic_json(public, q.build(dict(work, fixture=False), raw, [receipt]))
    fields = ("runtime_reader_ready_score", "runtime_changed_score", "cuda_context_ready_score")
    assert [reader_receipt(q.TASK, public.parent, field=f)["passed"] for f in fields] == [
        True,
        False,
        False,
    ]
    for kind in (
        "historical",
        "baseline",
        "pin",
        "task",
        "task_without_snapshots",
        "identity",
        "checksum",
        "missing_ref",
    ):
        altered = deepcopy(work)
        value = q.build(work, raw, [receipt])
        if kind == "identity":
            value["experiment_id"] = 0
        elif kind == "checksum":
            value["reproducibility_checksum"] = "invalid"
        else:
            if kind in {"historical", "baseline"}:
                altered[kind]["tampered"] = True
            elif kind in {"task", "task_without_snapshots"}:
                next(t for t in altered["authority"]["tasks"] if t["id"] == q.TASK)["title"] = (
                    "edited"
                )
                if kind == "task_without_snapshots":
                    altered["authority"]["authority_snapshots"] = {}
            elif kind == "pin":
                substituted = tmp_path / "substituted.json"
                atomic_json(substituted, dict(edited=True))
                altered["refs"][0].update(
                    q.reference(substituted),
                    path=work["refs"][0]["path"],
                    snapshot_path=str(substituted),
                )
            else:
                altered["refs"] = altered["refs"][1:]
            primitive_path = tmp_path / (kind + ".json")
            atomic_json(primitive_path, altered)
            value["primitive_reference"] = q.reference(primitive_path)
            value["raw_shard_hashes"] = [q.reference(primitive_path)]
        if kind != "checksum":
            value["reproducibility_checksum"] = q.checksum(value)
        atomic_json(path, value)
        assert not q.replay(path), kind
    monkeypatch.setattr(q, "diagnose", lambda b, r: deepcopy(work["diagnostic"]))
    assert q.measure(q.ROOT, tmp_path / "real-path")["authority"]["activated"]


@pytest.mark.parametrize(
    "mode",
    ["library", "library_resolved", "library_absent", "no_trace", "copy_failed", "fixed_exit"],
)
def test_additional_child_branches(tmp_path, monkeypatch, mode):
    """SCENARIO-VERIFY-8396-TRACE: missing trace and failed correction never grant copy readiness."""
    lease = MagicMock()
    lease.owner_receipt.return_value = dict(exclusive=True, device_uuid="GPU-test")
    lease.release.return_value = dict(released=True)
    monkeypatch.setattr(q.GpuLease, "acquire", MagicMock(return_value=lease))
    monkeypatch.setattr(q.shutil, "which", lambda n: "/usr/bin/" + n)
    stub = tmp_path / "stubs/libcuda.so"
    stub.parent.mkdir()
    stub.write_bytes(b"mocked library control")

    def run(name, argv, raw, **kwargs):
        p = dict(
            api_returns=dict(cuInit=101 if name == "inherited_init" else 0),
            libraries=[],
            environment=dict(CUDA_VISIBLE_DEVICES="-1"),
        )
        if mode.startswith("library") and name == "inherited_init":
            p.update(libraries=[q.reference(stub)], environment={})
        if mode.startswith("library") and name == "context_copy":
            p.update(
                context_copy_ready=True,
                byte_copy_parity=True,
                cleanup_passed=True,
                allocation_bytes=32,
            )
        if name == "inherited_init" and mode != "no_trace":
            (tmp_path / "driver.strace").write_text("device observation\n")
        code = (
            "raise SystemExit(1)"
            if (name == "context_copy" and mode == "copy_failed")
            or (name == "corrected_init" and mode == "fixed_exit")
            else "print(" + repr(json.dumps(p)) + ")"
        )
        return child(name, [e.PY, "-c", code], raw)

    monkeypatch.setattr(q, "child", run)
    installed = tmp_path / "libcuda.host.so"
    installed.write_bytes(b"mocked installed library")
    diagnostic = q.diagnose(
        dict(
            permitted_uuid="GPU-test",
            driver_library="libcuda.so.1"
            if mode in {"library_resolved", "library_absent"}
            else "/usr/lib/libcuda.so.1",
            libraries=[q.reference(installed)] if mode == "library_resolved" else [],
        ),
        tmp_path,
    )
    work = q.measure(tmp_path / "absent", tmp_path / "raw", fixture=True)
    work["diagnostic"] = diagnostic
    atomic_json(tmp_path / "raw/measurement.json", work)
    receipt = child("validation", [e.PY, "-c", "print('ok')"], tmp_path / "logs")
    value = q.build(work, tmp_path / "raw", [receipt])
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, value)
    assert q.replay(candidate)
    if mode in {"library", "library_resolved"}:
        positive = dict(
            work,
            fixture=False,
            failures=[],
            authority=dict(activated=True),
            historical=dict(runtime_reader_ready_score=1),
        )
        reduced = q.reduction(positive, True)
        assert reduced["verdict_class"] == "null"
        assert [
            reduced[f]
            for f in (
                "runtime_reader_ready_score",
                "runtime_changed_score",
                "cuda_context_ready_score",
            )
        ] == [1, 1, 1]
    for kind in ("primitive", "cause", "changed"):
        altered = deepcopy(work)
        if kind == "primitive":
            altered["diagnostic"]["rows"][0]["primitive"]["forged"] = True
        elif kind == "cause":
            altered["diagnostic"]["root_cause_status"] = "forged"
        else:
            altered["diagnostic"]["runtime_changed_score"] = 1 - diagnostic["runtime_changed_score"]
        primitive_path = tmp_path / (kind + ".json")
        atomic_json(primitive_path, altered)
        bad = deepcopy(value)
        bad["primitive_reference"] = q.reference(primitive_path)
        bad["reproducibility_checksum"] = q.checksum(bad)
        atomic_json(candidate, bad)
        assert not q.replay(candidate)


def test_runner_and_resource_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8396-VALIDATION: resource absence blocks before diagnosis."""
    from carnot.verify import runtime_evidence_execution_8382 as base

    monkeypatch.setattr(
        base,
        "resources",
        lambda p: dict(
            private_disk_backed_scratch=False, available_memory_bytes=0, minimum_memory_bytes=1
        ),
    )
    assert q.measure(tmp_path / "absent", tmp_path / "raw")["failures"]
    assert e.main(["--driver-init", "--library", "/absent"]) == 0
    monkeypatch.setattr(base, "main", lambda a: 0)
    assert e.main([]) == 0
    assert e.manifest(tmp_path)


def test_successful_retry_is_not_causal():
    """SCENARIO-VERIFY-8396-TRACE: optional paths and successful retries cannot explain a failure."""
    p = dict(api_returns=dict(cuInit=101), libraries=[], environment={})
    assert (
        q.analyze('openat(AT_FDCWD, "/dev/nvidia-caps", O_RDWR) = -1 EACCES', p)[
            "root_cause_status"
        ]
        == "cause_unproved"
    )
    text = 'openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR) = -1 EACCES\nopenat(AT_FDCWD, "/dev/nvidiactl", O_RDWR) = 3'
    assert q.analyze(text, p)["root_cause_status"] == "cause_unproved"


def test_original_failure_authentication(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8396-VALIDATION: an invented historical explanation is rejected."""
    substitute = tmp_path / "substitute.json"
    atomic_json(substitute, dict(root_cause_status="invented", failure_layer=[]))
    original = q.typed.authenticate
    monkeypatch.setattr(q.prior, "historical_aliases", lambda *a: [])
    monkeypatch.setattr(
        q.typed,
        "authenticate",
        lambda r: (
            substitute
            if r["path"].endswith("experiment_8290_v716_runtime_localization.json")
            else original(r)
        ),
    )
    work = q.measure(q.ROOT, tmp_path / "raw", fixture=True)
    assert any(g.get("observed") == "original_cuInit101_controls" for g in work["failures"])

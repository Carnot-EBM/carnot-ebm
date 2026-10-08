"""REQ-VERIFY-8277 / REQ-REPORT-8277: deterministic private backend controls."""

from pathlib import Path

import pytest

from carnot.verify import lease_backend_8277 as m


def device(index=0, uuid="GPU-a"):
    return dict(index=index, uuid=uuid, memory_used_mb=4, memory_free_mb=24000)


@pytest.mark.parametrize(
    "mask,expected",
    [(None, ["GPU-a"]), ("0", ["GPU-a"]), ("1", []), ("", []), ("-1", [])],
)
def test_device_masks(mask, expected):
    """SCENARIO-VERIFY-8277-DEVICE: stale ordinals never authorize a GPU."""
    env = {} if mask is None else {"CUDA_VISIBLE_DEVICES": mask}
    assert [d["uuid"] for d in m.permitted([device()], env)] == expected


def test_uuid_nonidentity_mapping():
    """SCENARIO-VERIFY-8277-DEVICE: UUID selection maps locally to zero."""
    rows = [device(0, "GPU-a"), device(3, "GPU-b")]
    assert m.permitted(rows, {"CUDA_VISIBLE_DEVICES": "GPU-b,GPU-a"}) == rows
    assert m.permitted(rows, {"NVIDIA_VISIBLE_DEVICES": "GPU-b"}) == [rows[1]]
    flags = m.device_flags("--device --main-gpu --no-warmup", "CUDA0: test", 0)
    assert flags == ["--device", "CUDA0", "--main-gpu", "0", "--no-warmup"]
    with pytest.raises(ValueError, match="native_cuda_device"):
        m.device_flags("--device --main-gpu --no-warmup", "CPU: test", 0)
    with pytest.raises(ValueError, match="no_warmup"):
        m.device_flags("--device", "CUDA0: test", 0)


def test_model_and_residency_fail_closed():
    """SCENARIO-VERIFY-8277-LOAD: health cannot replace GPU/model evidence."""
    assert m.load_errors({"model_path": "/wrong"}, Path("/right"), [0, 0])
    assert (
        m.load_errors(
            {"model_path": "/right", "chat_template": "template"}, Path("/right"), [65, 65]
        )
        == []
    )
    owner = dict(pid=12, start_time_ticks=9)
    assert m.residency("GPU-a, 12, 16000", "GPU-a", owner)["resident_memory_mb"] == 16000
    with pytest.raises(ValueError, match="pid_bound_residency"):
        m.residency("GPU-b, 12, 16000", "GPU-a", owner)


def test_owned_failure_overrides_external_block(tmp_path):
    """REQ-REPORT-8277: owned check failure is disqualified, never ready."""
    work = m.empty_work()
    m.gate(work, tmp_path / "missing", "exists", True, False, "external")
    v = m.build(work, tmp_path, [dict(name="unit", passed=True)])
    assert v["verdict_class"] == "blocked"
    assert v["honest_verdict"] == "complete_blocked_exists"
    assert v["inference_substrate_class"] == "blocked_no_run"
    assert v["load_only_contract"]["class"] == "model_load_no_generation"
    assert v["load_only_contract"]["minimum_duration_s"] == 2
    assert v["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert (
        m.build(work, tmp_path, [dict(name="unit", passed=False)])["verdict_class"]
        == "disqualified"
    )


def test_occupied_lease(tmp_path):
    """SCENARIO-VERIFY-8277-DEVICE: existing custody cannot be stolen."""
    from carnot.gpu_lease_phase_journal import GpuLease, LeaseError

    lease = GpuLease.acquire(
        runtime_dir=tmp_path,
        task_id="test",
        device_uuid="GPU-a",
        expected_model="model",
        vram_before_mb=4,
    )
    try:
        with pytest.raises(LeaseError):
            GpuLease.acquire(
                runtime_dir=tmp_path,
                task_id="other",
                device_uuid="GPU-a",
                expected_model="model",
                vram_before_mb=4,
            )
    finally:
        lease.transition("terminal_blocked")
        lease.release()


@pytest.fixture
def fake_load(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8277-LOAD: private transports never load real weights."""
    import json
    import os

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent-mask")
    monkeypatch.setenv("CARNOT_FORCE_LIVE", "0")
    model = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    model.write_bytes(b"private model fixture")
    binary = tmp_path / "server"
    binary.write_bytes(b"private backend fixture")
    lease = dict(pid=os.getpid(), pid_start_ticks=9, device_uuid="GPU-a", lease_id="lease-a")
    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            dict(
                released=False,
                lease_id="lease-a",
                expected_model=str(model),
                expires_monotonic_ns=10**30,
            )
        )
    )
    plan = dict(
        model=str(model),
        binary=str(binary),
        binary_sha256=m.MODEL_PIN,
        lease=lease,
        journal=str(journal),
    )
    owner = dict(pid=os.getpid(), start_time_ticks=9)
    state = dict(
        health=True,
        props=dict(model_path=str(model), chat_template="embedded"),
        offload="offloaded 65/65 layers to GPU",
        help="--device --main-gpu --no-warmup",
        enum="CUDA0: fixture",
        query=f"GPU-a, {os.getpid()}, 16000",
        exit=0,
    )

    class Server:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            state["server"] = self

        def launch(self):
            self.kwargs["log_path"].write_text(state["offload"])
            return owner

        def wait_for_health(self, timeout):
            if state.get("timeout"):
                raise TimeoutError("load_timeout")
            return dict(ok=state["health"])

        def cleanup(self):
            return dict(leak_free=state.get("cleanup", True), unrelated_process_kill_count_delta=0)

    def command(name, argv, raw, **kwargs):
        raw.mkdir(parents=True, exist_ok=True)
        out, err = raw / (name + ".stdout"), raw / (name + ".stderr")
        out.write_text(
            state["help"]
            if name == "native_help"
            else state["enum"]
            if name == "native_devices"
            else state["query"]
        )
        err.write_text("")
        return dict(
            name=name,
            passed=state["exit"] == 0,
            exit_code=state["exit"],
            stdout_path=str(out),
            stderr_path=str(err),
        )

    monkeypatch.setattr(m, "child", command)
    monkeypatch.setattr(m, "OwnedLlamaCppProcess", Server)
    monkeypatch.setattr(m, "sha256_file", lambda p: m.MODEL_PIN)
    monkeypatch.setattr(m, "proc_start_ticks", lambda p: 9)
    monkeypatch.setattr(m, "get_json", lambda url: state["props"])
    monkeypatch.setattr(m, "loaded_libraries", lambda owner, raw: dict(libraries=["private-CUDA"]))
    return plan, state, tmp_path


@pytest.mark.parametrize(
    "mutation",
    [
        "success",
        "cpu",
        "timeout",
        "wrong_model",
        "zero_offload",
        "unhealthy",
        "bad_exit",
        "resident_failure",
        "cleanup_failure",
        "owner_mismatch",
        "released",
        "bad_hash",
        "missing_file",
    ],
)
def test_load_controls(fake_load, monkeypatch, mutation):
    """SCENARIO-VERIFY-8277-LOAD: every failed operand retains owned cleanup."""
    import json

    plan, state, raw = fake_load
    if mutation == "cpu":
        state["enum"] = "CPU: fixture"
    if mutation == "timeout":
        state["timeout"] = True
    if mutation == "wrong_model":
        state["props"]["model_path"] = "/wrong-model"
    if mutation == "zero_offload":
        state["offload"] = "no offload"
    if mutation == "unhealthy":
        state["health"] = False
    if mutation == "bad_exit":
        state["exit"] = 1
    if mutation == "resident_failure":
        state["query"] = "GPU-b, 0, 0"
    if mutation == "cleanup_failure":
        state["cleanup"] = False
    if mutation == "owner_mismatch":
        monkeypatch.setattr(m, "proc_start_ticks", lambda p: 8)
    if mutation == "released":
        value = json.loads(Path(plan["journal"]).read_text())
        value["released"] = True
        Path(plan["journal"]).write_text(json.dumps(value))
    if mutation == "bad_hash":
        plan["binary_sha256"] = "wrong"
    if mutation == "missing_file":
        Path(plan["journal"]).unlink()
    result = m.load_child(plan, raw, raw)
    if mutation == "cpu":
        assert result["checks"][0]["path"] == str(raw / "native_devices.stderr")
    assert result["counts"]["generation_calls_attempted"] == 0
    assert result["counts"].get("gpu_model_loads_completed", 0) == int(
        mutation in {"success", "cleanup_failure"}
    )
    assert all(c["passed"] for c in result["checks"]) == (mutation == "success")
    assert (raw / "load_result.json").is_file()


def test_more_device_controls():
    """SCENARIO-VERIFY-8277-DEVICE: backend capabilities and capacity matter."""
    with pytest.raises(ValueError, match="native_device_flags"):
        m.device_flags("--no-warmup", "CUDA0: test", 0)
    assert m.permitted([device()], {"NVIDIA_VISIBLE_DEVICES": "all"})
    assert not m.permitted([dict(device(), memory_free_mb=100)], {})
    assert not m.permitted([dict(device(), memory_used_mb=1000)], {})


def historical(root):
    """REQ-REPORT-8277: private source sidecars bind the precise primary bytes."""
    import json

    results = root / "results"
    results.mkdir(exist_ok=True)
    for identity, suffix in [
        (8264, "v714_evidence_view_canary"),
        (8276, "v715_current_contract_readiness"),
    ]:
        primary = results / f"experiment_{identity}_{suffix}.json"
        raw = results / "raw" / primary.stem
        raw.mkdir(parents=True)
        out, err = raw / "old.stdout", raw / "old.stderr"
        out.write_text("cuda_runtime_available=False\n")
        err.write_text("Error 101: invalid device ordinal\n")
        receipt = dict(
            argv=["private-pytorch-probe"],
            actual_exit=1,
            stdout_path=str(out),
            stderr_path=str(err),
            stdout_sha256=m.sha256_file(out),
            stderr_sha256=m.sha256_file(err),
        )
        terminal = raw / "terminal.json"
        m.atomic_json(
            primary,
            dict(
                experiment_id=identity,
                task_id=f"exp{identity}-private",
                terminal_validation_sidecar_path=str(terminal),
                cuda_preflight_receipt=receipt,
            ),
        )
        digest = m.sha256_file(primary)
        sidecar = raw / "sidecar.json"
        m.atomic_json(sidecar, dict(primary_sha256=digest, report=dict(passed=True)))
        m.atomic_json(
            terminal, dict(publication=dict(primary_sha256=digest, sidecar_path=str(sidecar)))
        )
    return results


def test_authenticate_sources(tmp_path):
    """REQ-REPORT-8277: verbatim Error101 is preserved, not causally explained."""
    root, raw = tmp_path / "sources", tmp_path / "raw"
    root.mkdir()
    raw.mkdir()
    historical(root)
    work = m.empty_work()
    m.authenticate(root, raw, work)
    assert all(c["passed"] for c in work["checks"])
    assert (
        work["cuda_failure_comparison"]["stderr_verbatim"] == "Error 101: invalid device ordinal\n"
    )
    assert work["cuda_failure_comparison"]["causality_proved"] is False
    m.authenticate(tmp_path / "missing", raw, work)
    assert not work["checks"][-1]["passed"]


@pytest.mark.parametrize("mode", ["blocked", "success", "busy", "bad_source", "no_result"])
def test_measure_controls(tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8277: real lease rules compose with a private child transport."""
    raw = tmp_path / "raw"
    raw.mkdir()
    model = tmp_path / "Q4_K_M.gguf"
    model.write_bytes(b"fixture")
    monkeypatch.setattr(m, "authenticate", lambda root, raw, work: None)
    monkeypatch.setattr(m, "cached_current_model", lambda: dict(model_path=str(model)))
    monkeypatch.setattr(m, "read_gguf_metadata", lambda path: dict(quantization="Q4_K_M"))
    monkeypatch.setattr(m, "MODEL_PIN", m.sha256_file(model))
    monkeypatch.setattr(m.Path, "home", lambda: tmp_path)
    binary = tmp_path / ".cache/llama.cpp-master/build/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"fixture-binary")
    real_lease = m.GpuLease

    class Lease:
        @classmethod
        def acquire(cls, **kwargs):
            if mode == "busy":
                raise m.LeaseError("occupied")
            return real_lease.acquire(**dict(kwargs, runtime_dir=tmp_path / "leases"))

    monkeypatch.setattr(m, "GpuLease", Lease)
    if mode == "bad_source":
        monkeypatch.setattr(
            m, "authenticate", lambda *args: (_ for _ in ()).throw(ValueError("source"))
        )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    def command(name, argv, logs, **kwargs):
        out = raw / (name + ".stdout")
        out.write_text("0, GPU-a, fixture, 4, 24000\n" if mode != "blocked" else "")
        if name == "after_gpu":
            out.write_text("GPU-a, 4\n")
        if name == "leased_load_child" and mode != "no_result":
            result = m.empty_work()
            result["counts"].update(
                model_loads_attempted=1, model_loads_completed=1, gpu_model_loads_completed=1
            )
            result["cleanup_receipt"] = dict(leak_free=True)
            result["resident_gpu_receipt"] = dict(resident_memory_mb=16000)
            m.atomic_json(raw / "load_result.json", result)
        return dict(name=name, passed=True, exit_code=0, stdout_path=str(out))

    monkeypatch.setattr(m, "child", command)
    parent_mask = dict(m.os.environ)
    result = m.measure(tmp_path, raw)
    assert dict(m.os.environ) == parent_mask
    assert (raw / "measurement.json").is_file()
    assert (raw / "backend_binding.json").is_file()
    if mode in {"success", "no_result"}:
        assert result["gpu_lease_receipt"]["release"]["released"]
    else:
        assert not all(c["passed"] for c in result["checks"])


def private_candidate(tmp_path, receipts=None):
    """SCENARIO-REPORT-8277-REPLAY: a private block is valid evidence, not success."""
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    work = m.empty_work()
    m.gate(work, tmp_path / "absent", "exists", True, False, "external")
    m.atomic_json(raw / "measurement.json", work)
    m.atomic_json(raw / "backend_binding.json", dict(generation_permitted=False))
    value = m.build(work, raw, receipts or [dict(name="private-check", passed=True)])
    path = tmp_path / (m.NAME + ".json")
    m.atomic_json(path, value)
    return path, value, work, raw


def test_replay_and_real_cli(tmp_path):
    """SCENARIO-REPORT-8277-REPLAY: actual fresh CLI rejects rehashed headlines."""
    import subprocess
    import sys

    path, value, work, raw = private_candidate(tmp_path)
    assert m.replay(path)
    cli = subprocess.run(
        [sys.executable, str(m.ROOT / m.CLI), "--cold-replay", str(path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert cli.returncode == 0, cli.stderr
    assert "replay_passed" in cli.stdout
    value["gguf_backend_ready_score"] = 1
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = m.canonical_hash(value)
    m.atomic_json(path, value)
    assert not m.replay(path)
    assert m.main(["--cold-replay", str(path)]) == 1
    assert not m.replay(tmp_path / "absent.json")
    with pytest.raises(SystemExit):
        m.main(["--date", "20261009"])


def test_replay_hash_controls(tmp_path):
    """SCENARIO-REPORT-8277-REPLAY: stream and source bytes retain custody."""
    source = tmp_path / "source"
    out, err = tmp_path / "stdout", tmp_path / "stderr"
    source.write_bytes(b"source")
    out.write_bytes(b"stdout")
    err.write_bytes(b"stderr")
    receipt = dict(
        name="private",
        passed=True,
        stdout_path=str(out),
        stderr_path=str(err),
        stdout_sha256=m.sha256_file(out),
        stderr_sha256=m.sha256_file(err),
    )
    path, value, work, raw = private_candidate(tmp_path, [receipt])
    work["refs"] = [m.reference(source)]
    m.atomic_json(raw / "measurement.json", work)
    value = m.build(work, raw, [receipt])
    m.atomic_json(path, value)
    assert m.replay(path)
    source.write_bytes(b"tamper")
    assert not m.replay(path)
    source.write_bytes(b"source")
    out.write_bytes(b"tamper")
    assert not m.replay(path)


def test_terminal_controls(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8277-REPLAY: freeze unchanged validators and negatives."""
    path, value, work, raw = private_candidate(tmp_path)
    seen = []

    def command(name, argv, logs, **kwargs):
        seen.append((name, argv, kwargs))
        return dict(name=name, passed=True)

    monkeypatch.setattr(m, "child", command)
    assert m.terminal(path, raw, tmp_path)["passed"]
    assert [row[0] for row in seen] == [
        "adversarial_verify",
        "strict_rows",
        "cold_replay",
        "rehashed_tamper",
        "negative_control",
    ]
    assert seen[-1][2]["expected"] == 1
    commands = m.manifest(tmp_path, path, raw)
    assert commands[0]["name"] == "owned_unit_coverage"
    assert "--fail-under=100" in commands[3]["argv"]


@pytest.mark.parametrize("failure", [False, True])
def test_main_private_publication(tmp_path, monkeypatch, failure):
    """REQ-REPORT-8277: normal exit publishes atomically; failed terminal stays private."""

    def measure(root, raw):
        work = m.empty_work()
        m.gate(work, root / "absent", "exists", True, False, "external")
        m.atomic_json(raw / "measurement.json", work)
        m.atomic_json(raw / "backend_binding.json", {})
        return work

    monkeypatch.setattr(m, "measure", measure)
    monkeypatch.setattr(m, "child", lambda name, *args, **kwargs: dict(name=name, passed=True))
    monkeypatch.setattr(
        m,
        "terminal",
        lambda path, *args: dict(passed=not failure, candidate_sha256=m.sha256_file(path)),
    )
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--root", str(tmp_path), "--output", str(output)]) == int(failure)
    assert output.exists() != failure


def test_real_child_cli_no_backend(tmp_path):
    """SCENARIO-VERIFY-8277-LOAD: fresh script-path child has no GPU generation."""
    import os
    import subprocess
    import sys

    plan = dict(
        model="/missing-model",
        binary="/missing-binary",
        lease=dict(pid=os.getpid(), pid_start_ticks=-1, device_uuid="GPU-a", lease_id="private"),
        journal="/missing-journal",
    )
    path = tmp_path / "load_plan.json"
    m.atomic_json(path, plan)
    result = subprocess.run(
        [
            sys.executable,
            str(m.ROOT / m.CLI),
            "--load-child",
            str(path),
            "--private",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "load_result.json").is_file()


def test_resident_start_identity(fake_load, monkeypatch):
    """SCENARIO-VERIFY-8277-LOAD: a reused PID cannot claim GPU residence."""
    plan, state, raw = fake_load
    ticks = iter([9, 8])
    monkeypatch.setattr(m, "proc_start_ticks", lambda pid: next(ticks))
    result = m.load_child(plan, raw, raw)
    assert "resident_identity_or_query" in result["checks"][0]["observed"]


def test_loaded_primitive_replay(fake_load):
    """SCENARIO-REPORT-8277-REPLAY: rehashed primitive offload edits fail."""
    plan, state, raw = fake_load
    loaded = m.load_child(plan, raw, raw)
    work = dict(loaded)
    work["duration_s"] = 3
    work["gpu_lease_receipt"] = dict(loaded["gpu_lease_receipt"], release=dict(released=True))
    m.atomic_json(raw / "measurement.json", work)
    m.atomic_json(raw / "backend_binding.json", {})
    receipts = [dict(name="private", passed=True)]
    value = m.build(work, raw, receipts)
    path = raw.parent / "private-loaded.json"
    m.atomic_json(path, value)
    assert value["gguf_backend_ready_score"] == 1
    assert m.replay(path)
    (raw / "server.log").write_text("offloaded 0/65 layers to GPU")
    assert not m.replay(path)
    (raw / "server.log").write_text(state["offload"])
    work["cleanup_receipt"] = dict(work["cleanup_receipt"], leak_free=False)
    m.atomic_json(raw / "measurement.json", work)
    assert not m.replay(path)
    work["cleanup_receipt"] = dict(work["cleanup_receipt"], leak_free=True)
    work["counts"] = dict(work["counts"], model_loads_attempted=2)
    m.atomic_json(raw / "measurement.json", work)
    assert not m.replay(path)


def test_reuse_global_diagnostic(tmp_path, monkeypatch):
    """REQ-REPORT-8277: retain failed global health without running it twice."""
    health = tmp_path / "repository_health.json"
    m.atomic_json(health, dict(name="full_python_suite", passed=False, actual_exit=2))
    seen = []

    def command(name, *args, **kwargs):
        seen.append(name)
        return dict(name=name, passed=True)

    def measure(root, raw):
        work = m.empty_work()
        m.gate(work, root / "absent", "exists", True, False, "external")
        m.atomic_json(raw / "measurement.json", work)
        m.atomic_json(raw / "backend_binding.json", {})
        return work

    monkeypatch.setattr(m, "child", command)
    monkeypatch.setattr(m, "measure", measure)
    monkeypatch.setattr(
        m, "terminal", lambda path, *args: dict(passed=True, candidate_sha256=m.sha256_file(path))
    )
    output = tmp_path / (m.NAME + ".json")
    assert m.main(["--output", str(output), "--repository-health-receipt", str(health)]) == 0
    assert "full_python_suite" not in seen
    assert m.main(["--output", str(output), "--repository-health-receipt", str(health)]) == 0

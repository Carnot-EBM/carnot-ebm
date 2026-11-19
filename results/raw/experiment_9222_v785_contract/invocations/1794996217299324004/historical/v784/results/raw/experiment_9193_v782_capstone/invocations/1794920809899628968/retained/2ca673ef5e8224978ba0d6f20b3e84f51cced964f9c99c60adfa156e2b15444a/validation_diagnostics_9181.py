"""REQ-REPORT-9181-V782: bind measurements to exited children and unique logs.

The startup hook is explicit so private CLIs keep their normal import isolation.
Only completed streams are sealed; a killed child leaves an honest partial trace.
"""

from __future__ import annotations

import atexit
from collections import Counter
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import CodeType
from typing import Any
import uuid

import psutil

from carnot.reporting.current_work_receipt import sha256_file

Json = dict[str, Any]


def immutable(path: Path, value: Json) -> Json:
    """Exclusive creation prevents a later attempt from replacing earlier evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return dict(path=str(path), sha256=sha256_file(path))


def run(
    name: str,
    argv: list[str],
    raw: Path,
    stop: float,
    *,
    expected: int = 0,
    extra_env: dict[str, str] | None = None,
) -> Json:
    """One absolute deadline includes termination and reaping of this process group."""
    directory = raw / (name + "-" + uuid.uuid4().hex)
    directory.mkdir(parents=True, mode=0o700)
    stdout, stderr = directory / "stdout", directory / "stderr"
    start = time.monotonic()
    stop = min(stop, float(os.environ.get("CARNOT_9181_DEADLINE", str(stop))))
    print(f"[exp9181] before subprocess={name}", flush=True)
    with stdout.open("xb") as out, stderr.open("xb") as err:
        process = subprocess.Popen(
            argv,
            stdout=out,
            stderr=err,
            start_new_session=True,
            env=dict(os.environ, **(extra_env or {}), CARNOT_9181_DEADLINE=str(stop)),
        )
        timed_out = False
        while process.poll() is None:
            remaining = stop - time.monotonic() - 0.25
            if remaining <= 0:
                timed_out = True
                try:
                    for descendant in psutil.Process(process.pid).children(recursive=True):
                        descendant.kill()
                except psutil.NoSuchProcess:
                    pass
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                break
            try:
                process.wait(timeout=min(60, remaining))
            except subprocess.TimeoutExpired:
                print(f"[exp9181] heartbeat subprocess={name} pid={process.pid}", flush=True)
        # Descendants share the attempt lifetime, including after a parent exit.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    end = time.monotonic()
    stdout.chmod(0o444)
    stderr.chmod(0o444)
    row = dict(
        name=name,
        argv=argv,
        pid=process.pid,
        ppid=os.getpid(),
        exit_code=process.returncode,
        expected_exit=expected,
        passed=process.returncode == expected and not timed_out,
        timed_out=timed_out,
        normal_exit=process.returncode >= 0 and not timed_out,
        started_monotonic_s=start,
        ended_monotonic_s=end,
        duration_s=end - start,
        deadline_monotonic_s=stop,
        deadline_s=max(0, stop - start),
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        stdout_sha256=sha256_file(stdout),
        stderr_sha256=sha256_file(stderr),
    )
    row["receipt_reference"] = immutable(directory / "receipt.json", row)
    print(f"[exp9181] after subprocess={name} exit={process.returncode}", flush=True)
    return row


def startup() -> None:
    """Record child operations before application import without changing its answers."""
    directory = os.environ.get("CARNOT_9181_TRACE")
    if directory is None:
        return
    raw = Path(directory)
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"trace-{os.getpid()}-{uuid.uuid4().hex}.jsonl"
    stream = path.open("x", buffering=1)
    counts: Counter[str] = Counter()
    elapsed: Counter[str] = Counter()
    active: dict[int, int] = {}
    overhead = 0
    start = time.monotonic_ns()
    watched = {
        "check_tautology",
        "_is_arc_generation_or_exploration_artifact",
        "canonical_hash",
        "sha256_file",
        "_verify_artifact_impl",
    }
    stream.write(
        json.dumps(
            dict(
                event="start",
                pid=os.getpid(),
                ppid=os.getppid(),
                argv=sys.argv,
                pythonpath=os.environ.get("PYTHONPATH"),
            )
        )
        + "\n"
    )

    monitor = sys.monitoring
    tool = 4

    def begin(code: CodeType, offset: int) -> object:
        nonlocal overhead
        before = time.monotonic_ns()
        if code.co_name not in watched:
            return monitor.DISABLE
        frame = sys._getframe(1)
        counts[code.co_name] += 1
        active[id(frame)] = before
        monitor.set_local_events(tool, code, monitor.events.PY_RETURN)
        overhead += time.monotonic_ns() - before
        return None

    def end(code: CodeType, offset: int, value: Any) -> None:
        nonlocal overhead
        before = time.monotonic_ns()
        elapsed[code.co_name] += before - active.pop(id(sys._getframe(1)), before)
        overhead += time.monotonic_ns() - before

    def finish() -> None:
        monitor.set_events(tool, 0)
        monitor.free_tool_id(tool)
        stream.write(
            json.dumps(
                dict(
                    event="summary",
                    pid=os.getpid(),
                    ppid=os.getppid(),
                    counts=dict(counts),
                    elapsed_ns=dict(elapsed),
                    hook_callback_ns=overhead,
                    duration_ns=time.monotonic_ns() - start,
                )
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        path.chmod(0o444)

    monitor.use_tool_id(tool, "carnot-9181-diagnostic")
    monitor.register_callback(tool, monitor.events.PY_START, begin)
    monitor.register_callback(tool, monitor.events.PY_RETURN, end)
    monitor.set_events(tool, monitor.events.PY_START)
    atexit.register(finish)


def probe(arm: str, baseline: Path) -> Json:
    """Compare actual work on a fixed local artifact, retaining exact ordered flags."""
    import runpy
    from unittest.mock import patch
    from scripts import adversarial_verify as av
    from carnot.reporting import validation_reuse_9167 as reuse

    value = dict(
        schema="arc_go_explore",
        arms_non_degenerate=True,
        null_delta_methodology_note="Fixed input with passing control",
        positive_control_passed=True,
        **{f"metric_{i}": (i + 1) / 17 for i in range(12)},
        first_win_left=0.123456789,
        first_win_right=0.123456789,
        nested=[dict(field="structural") for _ in range(100)],
    )
    original = runpy.run_path(str(baseline))["check_tautology"]
    function = original if arm == "original" else av.check_tautology
    namespace = function.__globals__
    classify = namespace["_is_arc_generation_or_exploration_artifact"]
    counts = dict(full_scans=0, hashes=0)

    def counted(d: Json) -> bool:
        counts["full_scans"] += 1
        return bool(classify(d))

    def hashed(v: Any) -> str:
        counts["hashes"] += 1
        return str(hash_function(v))

    flags: list[Any] = []
    start = time.monotonic_ns()
    with patch.dict(namespace, _is_arc_generation_or_exploration_artifact=counted):
        if arm == "v781_wrapper":
            hash_function = reuse.canonical_hash
            with patch.object(reuse, "canonical_hash", hashed), reuse.preprocessing():
                original.__globals__["_is_arc_generation_or_exploration_artifact"] = (
                    av._is_arc_generation_or_exploration_artifact
                )
                try:
                    original(value, flags)
                finally:
                    original.__globals__["_is_arc_generation_or_exploration_artifact"] = classify
        else:
            function(value, flags)
    return dict(
        invocation=os.getpid(),
        arm=arm,
        artifact_size=len(json.dumps(value).encode()),
        numeric_pairs=len(list(av._numeric_pairs(value))),
        **counts,
        flags=[f.to_dict() for f in flags],
        measured_duration_s=(time.monotonic_ns() - start) / 1e9,
        input=value,
        status="completed",
        censor_reason=None,
    )

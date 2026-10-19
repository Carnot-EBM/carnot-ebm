"""REQ-VERIFY-8205: retain private parents and exact bounded child evidence.

Separate streams keep failed commands auditable. Each deadline owns only the
new process group, so cleanup cannot kill unrelated research processes.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[3]
Json = dict[str, Any]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual completed work so a silent child is still visible."""
    print(f"[exp8205] phase={phase} completed={completed} pending={pending}", flush=True)


def child(
    name: str,
    argv: list[str],
    logs: Path,
    *,
    deadline: float = 180,
    expected: int = 0,
    heartbeat: float = 30,
    scope: str = "owned",
) -> Json:
    """Keep both streams durable while the invocation parent outlives the child."""
    logs.mkdir(parents=True, exist_ok=True)
    stdout, stderr = logs / (name + ".stdout"), logs / (name + ".stderr")
    start = time.monotonic_ns()
    wall = time.time_ns()
    env = dict(
        os.environ,
        PYTHONPATH=f"{ROOT / 'python'}:{ROOT}",
        PYTHONUNBUFFERED="1",
        JAX_PLATFORMS="cpu",
    )
    progress("before_subprocess_" + name, 0, 1)
    timed_out = False
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            argv, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True
        )
        while process.poll() is None:
            remaining = deadline - (time.monotonic_ns() - start) / 1e9
            try:
                process.wait(timeout=max(0.001, min(heartbeat, remaining)))
            except subprocess.TimeoutExpired:
                progress("waiting_" + name, 0, 1)
                if (time.monotonic_ns() - start) / 1e9 >= deadline:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    end = time.monotonic_ns()
    progress("after_subprocess_" + name, 1, 0)
    receipt = dict(
        name=name,
        argv=argv,
        command=shlex.join(argv),
        scope=scope,
        exit_code=process.returncode,
        actual_exit=process.returncode,
        expected_exit=expected,
        passed=process.returncode == expected and not timed_out,
        normal_exit=process.returncode >= 0 and not timed_out,
        timed_out=timed_out,
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        started_wall_ns=wall,
        duration_s=(end - start) / 1e9,
        stdout_path=str(stdout),
        stderr_path=str(stderr),
        stdout_sha256=sha256_file(stdout),
        stderr_sha256=sha256_file(stderr),
        log_path=str(stdout),
        log_sha256=sha256_file(stdout),
        deadline_s=deadline,
    )
    atomic_json(logs / (name + ".receipt.json"), receipt)
    return receipt


def execute(plan: list[Json], logs: Path) -> list[Json]:
    """Run frozen argument vectors with honest counts between bounded children."""
    rows = []
    for index, spec in enumerate(plan):
        progress("validation", index, len(plan) - index)
        rows.append(
            child(
                spec["name"],
                spec["argv"],
                logs,
                deadline=spec["deadline"],
                expected=spec["expected"],
                scope=spec["scope"],
            )
        )
    progress("validation_complete", len(rows), 0)
    return rows


def pytest_plan(parent: Path) -> list[Json]:
    """Create controls before measurement so exact child argv can be frozen."""
    parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, expression, expected in [("tiny_pass", "True", 0), ("tiny_fail", "False", 1)]:
        source = parent / (name + ".py")
        source.write_text(
            "def test_private(tmp_path):\n"
            "    (tmp_path / 'proof').write_text('actual child')\n"
            f"    assert {expression}\n"
        )
        rows.append(
            dict(
                name=name,
                argv=[
                    str(ROOT / ".venv/bin/pytest"),
                    "-c",
                    "/dev/null",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    str(source),
                    "--basetemp=" + str(parent / (name + "_temp")),
                ],
                expected=expected,
                deadline=180,
                scope="owned",
            )
        )
    return rows


def pytest_controls(parent: Path, logs: Path) -> list[Json]:
    """Exercise tmp_path creation and an assertion failure in actual pytest."""
    return execute(pytest_plan(parent), logs)

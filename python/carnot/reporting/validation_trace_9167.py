"""REQ-REPORT-9167-V781: observe Python descendants before application imports.

The hook records executed operations without replacing their results. Periodic
Python stack snapshots avoid competing native traceback timers in pytest.
"""

from __future__ import annotations

import atexit
from collections import Counter
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from types import CodeType, FrameType
from typing import Any

Json = dict[str, Any]


def proc(pid: int) -> Json:
    """Keep permission failures visible instead of inventing a process identity."""
    try:
        root = Path("/proc") / str(pid)
        return dict(
            pid=pid,
            status=(root / "status").read_text(),
            argv=(root / "cmdline").read_bytes().replace(b"\0", b" ").decode(),
            wait=(root / "wchan").read_text(),
        )
    except OSError as error:
        return dict(pid=pid, error=type(error).__name__ + ":" + str(error))


def bootstrap(raw: Path) -> Path:
    """An inherited sitecustomize instruments grandchildren before their imports."""
    directory = raw / "bootstrap"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sitecustomize.py").write_text(
        "import runpy\nrunpy.run_path(" + repr(str(Path(__file__).resolve())) + ')["install"]()\n'
    )
    return directory


def install() -> None:
    """Flush identities, operation timings, counters and wait stacks until exit."""
    raw = Path(os.environ["CARNOT_9167_TRACE"])
    raw.mkdir(parents=True, exist_ok=True)
    stream = (raw / f"trace-{os.getpid()}.jsonl").open("a", buffering=1)
    counts: Counter[str] = Counter()
    elapsed: Counter[str] = Counter()
    active: dict[int, tuple[str, int]] = {}
    overhead = [0]
    started = time.monotonic_ns()
    closed = threading.Event()
    own = __file__

    def emit(event: str, **fields: Any) -> None:
        stream.write(
            json.dumps(
                dict(
                    event=event,
                    pid=os.getpid(),
                    ppid=os.getppid(),
                    monotonic_ns=time.monotonic_ns(),
                    **fields,
                )
            )
            + "\n"
        )

    def profile(frame: FrameType, event: str, arg: Any) -> None:
        filename = frame.f_code.co_filename
        before = time.monotonic_ns()
        if event in {"call", "return"}:
            key = filename + ":" + frame.f_code.co_name
            if event == "call":
                counts[key] += 1
                active[id(frame)] = (key, before)
                if counts[key] & (counts[key] - 1) == 0:
                    emit("call", operation=key, count=counts[key], line=frame.f_lineno)
            elif id(frame) in active:
                _, start = active.pop(id(frame))
                elapsed[key] += before - start
                if frame.f_code.co_name in {
                    "measure",
                    "authenticate",
                    "historical_report",
                    "replay",
                    "_verify_artifact_impl",
                    "measured_work",
                }:
                    emit("return", operation=key, elapsed_ns=before - start)
                if (
                    filename.endswith("subprocess.py")
                    and frame.f_code.co_name in {"wait", "poll"}
                    and arg is not None
                ):
                    emit("child_exit", child_pid=frame.f_locals["self"].pid, exit_code=arg)
        overhead[0] += time.monotonic_ns() - before

    monitor = sys.monitoring
    tool = 3

    def begin(code: CodeType, offset: int) -> object:
        filename = code.co_filename
        if filename == own or not (
            "/reporting/" in filename
            or "/testing/" in filename
            or "adversarial_verify.py" in filename
            or filename.endswith("subprocess.py")
            or code.co_name == "measured_work"
        ):
            return monitor.DISABLE
        monitor.set_local_events(tool, code, monitor.events.PY_RETURN)
        profile(sys._getframe(1), "call", None)
        return None

    def end(code: CodeType, offset: int, value: Any) -> None:
        profile(sys._getframe(1), "return", value)

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event in {"open", "os.scandir", "os.listdir"}:
            counts["audit:" + event] += 1
        if event == "subprocess.Popen":
            emit("spawn", executable=str(args[0]), argv=args[1], cwd=str(args[2]))

    def snapshot(event: str) -> None:
        emit(
            event,
            counters=dict(counts),
            elapsed_ns=dict(elapsed),
            overhead_ns=overhead[0],
            duration_ns=time.monotonic_ns() - started,
            process=proc(os.getpid()),
            stacks={
                str(tid): traceback.format_stack(frame)
                for tid, frame in sys._current_frames().items()
            },
        )

    def heartbeat() -> None:
        while not closed.wait(60):
            snapshot("heartbeat")
            print(
                f"[exp9167] heartbeat pid={os.getpid()} operations={sum(counts.values())}",
                flush=True,
            )

    def finish() -> None:
        monitor.set_events(tool, 0)
        monitor.free_tool_id(tool)
        closed.set()
        snapshot("summary")
        stream.close()

    emit("start", argv=sys.argv, process=proc(os.getpid()))
    sys.addaudithook(audit)
    monitor.use_tool_id(tool, "carnot-validation-trace")
    monitor.register_callback(tool, monitor.events.PY_START, begin)
    monitor.register_callback(tool, monitor.events.PY_RETURN, end)
    monitor.set_events(tool, monitor.events.PY_START)
    threading.Thread(target=heartbeat, daemon=True).start()
    atexit.register(finish)

"""Run one induced engine on ONE held-out input in a new, restricted process.

REQ-ARC-WMTE-10010 (scoring isolation). The Experiment 10010 harness starts this file
once per held-out row, as a new interpreter (exec, not fork). The job holds the engine
source and that one row's input: grid, action, and data. It holds no other row and no
answer. The held-out rows are consecutive transitions, so row k+1's input grid IS row
k's answer: a process that holds two rows leaks an answer through the stack or the heap.

How one row runs:

1. The parent writes one JSON job to stdin and starts this file with `python -s -P`.
2. This process keeps a private copy of stdout, reads the job, and refuses any job that
   is not exactly one row. It then points fds 0, 1, and 2 at /dev/null.
3. It sets the memory limit and the alarm, installs the audit hook, loads a fresh
   module, calls `engine` once, and checks the output type.
4. It writes one JSON line to its private stdout and exits.

The audit hook refuses file opens outside the Python install, any write open, and
process, socket, and ctypes use. It is a second wall: no other row is in this process
at all. This file imports only the standard library and numpy.
"""

from __future__ import annotations

import json
import os
import random
import signal
import sys
from typing import Any, Optional

import numpy as np

# The expected output is one ARC frame. Anything far larger is refused before JSON
# encoding, so a runaway engine cannot flood the pipe.
MAX_OUTPUT_CELLS_FACTOR = 4
MIN_OUTPUT_CELLS_CAP = 4096
MAX_RESULT_BYTES = 8 * 1024 * 1024
# Used when the parent disables the per-call timeout: the row process is still killed.
HARD_CAP_S = 60.0
# The per-row kill deadline is load + call time plus this margin.
KILL_MARGIN_S = 5.0
# Extra room for a new interpreter to start and import numpy before the load + call.
STARTUP_S = 30.0
REFIRE_S = 0.05
# A job carries exactly these keys, and its row exactly these. Anything else is refused,
# so a second row (or an answer) cannot ride along by mistake.
JOB_KEYS = frozenset({"source", "name", "row", "timeout_s", "rss_delta_mb", "seed"})
ROW_KEYS = frozenset({"grid", "action", "data"})

# Audit events that could reach data or another process. The engine needs none of them.
_DENIED_EVENTS = frozenset(
    {
        "subprocess.Popen",
        "os.system",
        "os.exec",
        "os.posix_spawn",
        "os.spawn",
        "os.fork",
        "os.forkpty",
        "os.kill",
        "os.killpg",
        "os.putenv",
        "os.unsetenv",
        "os.symlink",
        "os.link",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.mkdir",
        "os.chmod",
        "os.chown",
        "os.truncate",
        "os.chdir",
        "socket.__new__",
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "ctypes.dlopen",
        "ctypes.dlsym",
        "ctypes.addressof",
        "ctypes.cdata",
        "ctypes.cdata/buffer",
        "ctypes.call_function",
        "ctypes.string_at",
        "ctypes.wstring_at",
        "resource.setrlimit",
        "resource.prlimit",
        "shutil.copyfile",
        "shutil.copymode",
        "shutil.copystat",
        "shutil.copytree",
        "shutil.move",
        "shutil.rmtree",
        "shutil.chown",
        "shutil.make_archive",
        "shutil.unpack_archive",
        "urllib.Request",
        "http.client.connect",
        "webbrowser.open",
        "mmap.__new__",
        "pty.spawn",
    }
)
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


class EngineCallTimeout(Exception):
    """Same name as the live guard's timeout, so row errors read the same."""


class OutputTypeError(Exception):
    """The engine returned something that is not a grid of integers."""


def allowed_read_roots() -> tuple[str, ...]:
    """Directories an engine may read from: the Python install and its libraries only."""
    import sysconfig

    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    roots.update(str(v) for v in sysconfig.get_paths().values() if v)
    roots.add(os.path.dirname(os.path.dirname(np.__file__)))
    return tuple(sorted({os.path.realpath(r) for r in roots if r}))


def _under(path: str, roots: tuple[str, ...]) -> bool:
    return any(path == r or path.startswith(r.rstrip(os.sep) + os.sep) for r in roots)


def make_audit_hook(roots: tuple[str, ...]) -> Any:
    """The hook the per-row process installs. A hook cannot be removed once added."""

    def hook(event: str, args: tuple) -> None:
        if event in _DENIED_EVENTS or event.startswith("ctypes."):
            raise PermissionError(f"engine sandbox denies {event}")
        if event != "open" or not args:
            return
        target = args[0]
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            raise PermissionError("engine sandbox denies write opens")
        if isinstance(flags, int) and flags & _WRITE_FLAGS:
            raise PermissionError("engine sandbox denies write opens")
        if isinstance(target, int):
            return  # an fd this process already holds
        try:
            path = os.path.realpath(os.fsdecode(os.fspath(target)))
        except Exception as exc:
            raise PermissionError(f"engine sandbox cannot resolve {target!r}") from exc
        if not _under(path, roots):
            raise PermissionError(f"engine sandbox denies reading {path}")

    return hook


def check_output(out: Any, n_cells_in: int) -> np.ndarray:
    """Accept only an integer (or integral float) array; return it as int64.

    An object array could hold elements whose `==` always says True, and a structured
    array cannot be compared at all. Both are refused here, so the parent only ever
    compares plain integers.
    """
    arr = np.asarray(out)
    cap = max(MIN_OUTPUT_CELLS_CAP, MAX_OUTPUT_CELLS_FACTOR * int(n_cells_in))
    if arr.size > cap:
        raise OutputTypeError(f"output_too_large shape={list(arr.shape)}")
    kind = arr.dtype.kind
    if kind in "iub":
        return arr.astype(np.int64)
    if kind == "f" and bool(np.all(np.isfinite(arr))) and bool(np.all(arr == np.round(arr))):
        return arr.astype(np.int64)
    raise OutputTypeError(f"output dtype {arr.dtype.str!r} is not an integer grid")


def _on_alarm(signum: int, frame: Any) -> None:
    raise EngineCallTimeout("engine call exceeded its time budget")


def _timed(timeout_s: Optional[float], fn: Any, *args: Any) -> Any:
    # The timer re-fires every REFIRE_S, so an engine that swallows the first raise
    # is raised again, like the live guard's persistent re-fire.
    if timeout_s is None:
        return fn(*args)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s), REFIRE_S)
    try:
        return fn(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)


def run_one_row(
    source: str, name: str, row: dict[str, Any], timeout_s: Optional[float], seed: int
) -> dict[str, Any]:
    """Load a fresh module and call `engine` once. Runs in the per-row process."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    grid = np.asarray(row["grid"], dtype=np.int64)
    try:
        code = compile(source, name, "exec")
        namespace: dict[str, Any] = {"__name__": name}
        _timed(timeout_s, exec, code, namespace)
        engine = namespace.get("engine")
        if not callable(engine):
            return {"ok": False, "error": "EngineLoadError: module defines no callable engine"}
        out = _timed(timeout_s, engine, grid.copy(), int(row["action"]), row.get("data"))
        arr = check_output(out, grid.size)
        return {"ok": True, "grid": arr.tolist(), "shape": list(arr.shape)}
    except BaseException as exc:  # SystemExit and KeyboardInterrupt are row errors here
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}


def _set_memory_limit(rss_delta_mb: Optional[float]) -> None:
    # Address-space room above what this process already maps. It stands in for the
    # live guard's RSS-growth bound, and it can stop one giant allocation, which the
    # in-process guard cannot.
    if rss_delta_mb is None:
        return
    import resource

    with open("/proc/self/status") as fh:
        vm_kb = next(int(line.split()[1]) for line in fh if line.startswith("VmSize:"))
    limit = vm_kb * 1024 + int(float(rss_delta_mb) * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def parse_one_row_job(text: str) -> dict[str, Any]:
    """Read a job and refuse it unless it holds exactly one row's input.

    The parent never sends more, but this process checks anyway: a second row here
    would put an answer within the engine's reach.
    """
    job = json.loads(text)
    if not isinstance(job, dict) or set(job) - JOB_KEYS or "source" not in job:
        raise ValueError(f"job keys refused: {sorted(job) if isinstance(job, dict) else job!r}")
    row = job.get("row")
    if not isinstance(row, dict) or set(row) - ROW_KEYS or not {"grid", "action"} <= set(row):
        raise ValueError("job must hold exactly one row with grid, action, and data")
    return job


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def main() -> int:  # pragma: no cover - runs as the per-row process; tested through it
    # Keep a private copy of stdout for the one result line. After the job is read,
    # fds 0-2 go to /dev/null so nothing an engine prints can reach the parent's parser.
    private_out = os.dup(1)
    job = parse_one_row_job(sys.stdin.read())
    roots = allowed_read_roots()
    info = {
        "process_per_row": True,
        "audit_hook": True,
        "cwd": os.getcwd(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "allowed_read_roots": list(roots),
    }
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    _set_memory_limit(job.get("rss_delta_mb"))
    signal.signal(signal.SIGALRM, _on_alarm)
    sys.addaudithook(make_audit_hook(roots))
    source, name, row = str(job["source"]), str(job.get("name", "<engine>")), job["row"]
    timeout_s, seed = job.get("timeout_s"), int(job.get("seed", 0))
    del job
    try:
        result = run_one_row(source, name, row, timeout_s, seed)
    except BaseException as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    _write_all(private_out, (json.dumps({"result": result, "child": info}) + "\n").encode())
    return 0


if __name__ == "__main__":
    # Exit at once, as the fork-based scorer did: a thread the engine left running
    # must not hold the process open past its result. A job refusal still raises.
    os._exit(main())

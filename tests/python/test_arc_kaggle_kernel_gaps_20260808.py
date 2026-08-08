"""Spec: REQ-ARC-WMTE-6228.

Regression tests for four Kaggle submission-kernel gaps from the 2026-08-08 adversarial
live-agent review, "Gaps" section:

  4. Swarm subprocess timeout equalled the full 12h Kaggle cap, `TimeoutExpired` uncaught,
     the return code never inspected, terminal log line unconditional.
  5. `MAX_ACTIONS = 2000` shipped on an unconfirmed host-RAM assumption with no `/proc/meminfo`
     preflight print next to the GPU one.
  6. The agent-code self-locate is a bare `next()` over an rglob -- a missing dataset raises a
     message-free StopIteration at import time.
  7. The agent's llama-server stderr log lands in `/tmp` inside the ephemeral Kaggle container
     because the kernel never sets `CARNOT_ARC_SERVER_LOG_DIR`.

THE FIXES (all in scripts/kaggle/submission_kernel/main.py):

  4. A `try`/`except subprocess.TimeoutExpired` around the swarm `subprocess.run`, timeout
     lowered to 41400s (30 min margin under the 43200s cap), and an explicit
     `SWARM EXITED rc=N after Ns` / `SWARM TIMED OUT after Ns` print either way.
  5. A `/proc/meminfo` `MemTotal` print alongside the existing `nvidia-smi` preflight line,
     inside AGENT_SRC (the agent runs on the scored host, so it reads its own /proc).
  6. `list()` the rglob hits; on empty, print a loud "attach the dataset" line and
     `raise SystemExit` instead of letting a bare `next()` raise message-free StopIteration.
  7. `os.environ.setdefault("CARNOT_ARC_SERVER_LOG_DIR", "/kaggle/working")` at the top of the
     OUTER kernel script, before `run_env = os.environ.copy()` -- child processes inherit it.

TESTING APPROACH. Mirrors tests/python/test_arc_kernel_agent_src_executable.py's own pattern:
slice the relevant statements out of the shipped source BY LINE (never re-typed), and `exec()`
them against a controlled namespace -- so these tests exercise the actual shipped bytes, and a
future edit that silently reverts or renames the fix fails loudly here instead of testing a
stale copy.
"""

from __future__ import annotations

import ast
import os
import subprocess
import textwrap
from pathlib import Path

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_KERNEL = Path(_REPO) / "scripts" / "kaggle" / "submission_kernel" / "main.py"


def _kernel_src() -> str:
    src = _KERNEL.read_text()
    ast.parse(src)
    return src


def _agent_src() -> str:
    """The `my_agent.py` source the kernel authors at runtime, as a string."""
    src = _kernel_src()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "AGENT_SRC"
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    pytest.fail("AGENT_SRC not found in the submission kernel")
    raise AssertionError("unreachable")  # pragma: no cover


def _slice_between(
    src: str, start_marker: str, end_marker: str, *, from_index: int = 0, extra_lines: int = 0
) -> str:
    """Dedented so the slice compiles standalone even when the source is indented in context.

    `extra_lines` grabs N more lines after the end-marker match, for a marker that lands
    mid-statement (e.g. inside a multi-line print(...) call) rather than on the closing line.
    """
    lines = src.splitlines()
    start = next((i for i in range(from_index, len(lines)) if start_marker in lines[i]), None)
    if start is None:
        pytest.fail(f"start marker not found: {start_marker!r}")
    end = next((i for i, ln in enumerate(lines[start:], start) if end_marker in ln), None)
    if end is None:
        pytest.fail(f"end marker not found: {end_marker!r}")
    return textwrap.dedent("\n".join(lines[start : end + 1 + extra_lines]))


# --------------------------------------------------------------------------------------- #
# Gap 6: the self-locate bare next()                                                       #
# --------------------------------------------------------------------------------------- #
def _self_locate_block() -> str:
    # Starts AFTER `inp = Path("/kaggle/input")` deliberately: that assignment is excluded so the
    # test's own `inp` (a fake tree) survives in the exec namespace instead of being clobbered by
    # the real hardcoded path.
    return _slice_between(
        _agent_src(),
        "# self-locate the bundled carnot package",
        "sys.path.insert(0, str(carnot))",
    )


def test_self_locate_missing_dataset_fails_loudly_not_with_bare_stopiteration(
    tmp_path: Path,
) -> None:
    """The defect this fix pins: a missing/re-laid-out dataset must raise a message that names
    the problem, not a bare StopIteration a reader of the tail of a 12h log cannot diagnose."""
    inp = tmp_path / "kaggle_input"
    inp.mkdir()  # exists, but contains no carnot package at all
    printed: list[str] = []
    ns = {
        "inp": inp,
        "Path": Path,
        "sys": __import__("sys"),
        "print": lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
    }
    block = _self_locate_block()
    with pytest.raises(SystemExit) as excinfo:
        exec(compile(block, "<AGENT_SRC self-locate>", "exec"), ns)  # noqa: S102
    assert "not attached" in str(excinfo.value) or "not found" in str(excinfo.value)
    assert any(
        "FATAL" in line and "attach the carnot package dataset" in line for line in printed
    ), f"missing dataset must print a loud, actionable line before raising; got: {printed}"


def test_self_locate_present_dataset_still_resolves_correctly(tmp_path: Path) -> None:
    """The happy path must be byte-identical in outcome to the pre-fix bare next(): resolves to
    the directory two levels above arc_competition_agent.py's mount point."""
    inp = tmp_path / "kaggle_input"
    nested = inp / "datasets" / "iancblenke" / "carnot-agent-code" / "python"
    (nested / "carnot" / "agentic").mkdir(parents=True)
    (nested / "carnot" / "agentic" / "arc_competition_agent.py").write_text("")
    ns = {"inp": inp, "Path": Path, "sys": __import__("sys"), "print": lambda *a, **k: None}
    block = _self_locate_block()
    exec(compile(block, "<AGENT_SRC self-locate>", "exec"), ns)  # noqa: S102
    assert ns["carnot"] == nested


# --------------------------------------------------------------------------------------- #
# Gap 5: the /proc/meminfo preflight print                                                 #
# --------------------------------------------------------------------------------------- #
def test_host_ram_preflight_print_reads_proc_meminfo() -> None:
    """The agent's GPU preflight ('LLM GPU HARDWARE:') must now be paired with a 'HOST RAM:'
    line reading /proc/meminfo -- the review's exact fix for the unconfirmed 16 GiB assumption
    behind MAX_ACTIONS=2000."""
    agent_src = _agent_src()
    assert "HOST RAM:" in agent_src
    assert "/proc/meminfo" in agent_src
    block = _slice_between(agent_src, "# HOST RAM,", 'print(f"HOST RAM: unavailable')
    printed: list[str] = []
    ns = {"print": lambda *a, **k: printed.append(" ".join(str(x) for x in a))}
    exec(compile(block, "<AGENT_SRC host-ram-preflight>", "exec"), ns)  # noqa: S102
    assert len(printed) == 1
    assert printed[0].startswith("HOST RAM: MemTotal:"), printed[0]


def test_host_ram_preflight_degrades_gracefully_when_meminfo_unreadable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matches the existing GPU preflight's own 'REPORTS, DOES NOT ABORT' contract -- a read
    failure must print a diagnosable line, not raise and kill the run before any game."""
    agent_src = _agent_src()
    block = _slice_between(agent_src, "# HOST RAM,", 'print(f"HOST RAM: unavailable')

    def _raise_open(*_a, **_k):
        raise OSError("no such file")

    printed: list[str] = []
    ns = {"open": _raise_open, "print": lambda *a, **k: printed.append(" ".join(str(x) for x in a))}
    exec(compile(block, "<AGENT_SRC host-ram-preflight>", "exec"), ns)  # noqa: S102
    assert len(printed) == 1
    assert printed[0].startswith("HOST RAM: unavailable"), printed[0]


# --------------------------------------------------------------------------------------- #
# Gap 7: CARNOT_ARC_SERVER_LOG_DIR setdefault                                              #
# --------------------------------------------------------------------------------------- #
def test_server_log_dir_env_setdefault_present_before_run_env_copy() -> None:
    """The setdefault must appear BEFORE `run_env = os.environ.copy()` (the swarm subprocess's
    env source), or a child process would not inherit it."""
    src = _kernel_src()
    setdefault_idx = src.index('os.environ.setdefault("CARNOT_ARC_SERVER_LOG_DIR"')
    # The 4-space-indented, no-backtick form: excludes this fix's own explanatory comment a few
    # lines above the setdefault, which mentions the same phrase inside backticks as prose.
    run_env_idx = src.index("    run_env = os.environ.copy()")
    assert setdefault_idx < run_env_idx, (
        "CARNOT_ARC_SERVER_LOG_DIR must be set before the swarm's run_env is copied from "
        "os.environ, or the child process (and whatever it spawns per game) will not inherit it"
    )


def test_server_log_dir_setdefault_never_overrides_an_operator_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CARNOT_ARC_SERVER_LOG_DIR", "/kaggle/working/custom-logs")
    os.environ.setdefault("CARNOT_ARC_SERVER_LOG_DIR", "/kaggle/working")
    assert os.environ["CARNOT_ARC_SERVER_LOG_DIR"] == "/kaggle/working/custom-logs"


def test_server_log_dir_setdefault_applies_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CARNOT_ARC_SERVER_LOG_DIR", raising=False)
    os.environ.setdefault("CARNOT_ARC_SERVER_LOG_DIR", "/kaggle/working")
    assert os.environ["CARNOT_ARC_SERVER_LOG_DIR"] == "/kaggle/working"


# --------------------------------------------------------------------------------------- #
# Gap 4: swarm subprocess timeout margin + visible exit status                             #
# --------------------------------------------------------------------------------------- #
def _swarm_block() -> str:
    # The 4-space-indented marker excludes the identical phrase inside this fix's own
    # explanatory comment a few lines above (`` `run_env = os.environ.copy()` below ``).
    return _slice_between(
        _kernel_src(),
        "    run_env = os.environ.copy()",
        "(41400s budget) -- killed before the swarm process exited on its own",
        extra_lines=2,  # closes the multi-line print(...) call: `flush=True,` then `)`
    )


def test_swarm_timeout_has_real_margin_under_the_12h_kaggle_cap() -> None:
    src = _kernel_src()
    assert "timeout=41400," in src, "the swarm timeout must be a named, sub-12h-cap constant"
    KAGGLE_CAP_S = 43200
    assert 41400 < KAGGLE_CAP_S, "sanity: the fixture constant itself must be under the cap"
    margin = KAGGLE_CAP_S - 41400
    assert margin >= 1200, (
        f"only {margin}s of margin under the 12h Kaggle cap -- too tight to reliably observe "
        "and print the outcome before Kaggle's own harness kills the kernel"
    )


def test_swarm_normal_exit_prints_the_return_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import time as _time

    class _FakeResult:
        returncode = 7

    def _fake_run(*_a, **_k):
        return _FakeResult()

    printed: list[str] = []
    ns = {
        "os": os,
        "sys": __import__("sys"),
        "subprocess": subprocess,
        "time": _time,
        "fw": "/fake/fw",
        "print": lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
    }
    monkeypatch.setattr(subprocess, "run", _fake_run)
    exec(compile(_swarm_block(), "<kernel swarm-run>", "exec"), ns)  # noqa: S102
    assert any(line.startswith("SWARM EXITED rc=7 after") for line in printed), printed


def test_swarm_timeout_is_caught_and_named_not_an_uncaught_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect this fix pins: a run that dies AT the deadline used to raise
    subprocess.TimeoutExpired uncaught -- a hard kernel crash indistinguishable in the log from
    any other traceback. It must now be caught and named."""
    import time as _time

    def _fake_run(*_a, **kwargs):
        raise subprocess.TimeoutExpired(cmd="main.py", timeout=kwargs.get("timeout", 41400))

    printed: list[str] = []
    ns = {
        "os": os,
        "sys": __import__("sys"),
        "subprocess": subprocess,
        "time": _time,
        "fw": "/fake/fw",
        "print": lambda *a, **k: printed.append(" ".join(str(x) for x in a)),
    }
    monkeypatch.setattr(subprocess, "run", _fake_run)
    exec(compile(_swarm_block(), "<kernel swarm-run>", "exec"), ns)  # noqa: S102
    assert any("SWARM TIMED OUT" in line for line in printed), (
        f"a subprocess.TimeoutExpired must be caught and reported, not left to propagate; "
        f"printed: {printed}"
    )

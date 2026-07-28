"""The generator server's stderr is CAPTURED, not discarded.

REQ-ARC-WMTE-5996 / SCENARIO-ARC-WMTE-5996-A-500-IS-A-RETURN-CODE-NOT-AN-EXCEPTION-AND-ITS-DISCRIMINATOR-IS-KEPT

REQ-ARC-WMTE-5995 / SCENARIO: generator-server-stderr-is-captured-to-a-file

WHY THIS EXISTS (2026-07-27, from an operator question: "wouldn't a 500 error mean an exception
of some kind?").

The answer turned out to be NO, and chasing it exposed the reason the K>=2 concurrency fault
stayed invisible for months. llama.cpp's decode-failure handler
(``tools/server/server-context.cpp:3200-3230``) does not throw. It checks the RETURN CODE of
``llama_decode()`` and logs::

    SRV_ERR("%s i = %d, n_batch = %d, ret = %d\\n", err.c_str(), i, n_batch, ret);

That ``ret`` is the ONLY discriminator between our failure modes:

===========  ==========================================================================
``ret``      meaning
===========  ==========================================================================
``1``        "Context size has been exceeded."  -- mode A, pool exhaustion, SURVIVABLE
``-1``       "Invalid input batch."
``< -1``     "Compute error."
``2``        explicitly UNHANDLED upstream (``// TODO: handle ret == 2 (abort)``)
===========  ==========================================================================

A hard ``GGML_ASSERT`` abort -- mode B, where the server DIES permanently -- also prints only to
stderr. So while ``_ensure_server`` launched with ``stderr=subprocess.DEVNULL``, mode A and mode B
were **indistinguishable from the client**: both surface as a failed request, and the one integer
that separates them was written to /dev/null. The 2026-07-27 adversarial review closed with "Mode B
is UNRESOLVED at 81920; needs the server's stderr captured" -- this is that capture.

Note the graceful path is a DIFFERENT site. The per-request admission check at ``:2704-2712`` sends
a clean 400 ("try increasing it") BEFORE decoding. It is per-request, so it cannot catch the
aggregate case where K requests each fit individually but jointly exhaust the shared ``kv_unified``
pool -- that fails later, inside ``llama_decode``, as a 500. Concurrency escapes the graceful path
by construction, and the 500 handler then errors EVERY processing slot
(``for (auto & slot : slots) ... send_error``), which is why the measured failures are 2/2 at K=2
and 4/4 at K=4 rather than a single victim.

A FILE AND NOT A PIPE, deliberately. ``subprocess.PIPE`` with no reader deadlocks the server as
soon as the OS pipe buffer fills (~64KB); llama-server is chatty enough to reach that in a long
run, and the resulting hang would look exactly like the fault being diagnosed. A file has no
backpressure. Capture is also best-effort: if the log cannot be opened the launch still proceeds on
DEVNULL, because losing diagnostics must never cost us the generator itself.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_executable_world_model as m  # noqa: E402


class _FakePopen:
    """Captures the kwargs ``_ensure_server`` passes, without starting a process."""

    last_kwargs: dict = {}

    def __init__(self, args, **kwargs):
        _FakePopen.last_kwargs = dict(kwargs)
        _FakePopen.last_args = list(args)
        self.pid = 424242

    def poll(self):
        return None

    def kill(self):
        return None


@pytest.fixture
def launched(tmp_path, monkeypatch):
    """Drive the REAL _ensure_server far enough to build the Popen call, with no real server.

    The binary and model are STUBBED rather than required. An earlier draft guarded on a real
    GGUF being present and skipped when it was not -- so all four tests passed as `s` and proved
    nothing. CLAUDE.md forbids skipped tests for exactly this reason, and a test that silently
    declines to run is the same dead-channel defect this file exists to document.
    """
    monkeypatch.setenv("CARNOT_ARC_SERVER_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(m.subprocess, "Popen", _FakePopen)
    fake_bin = tmp_path / "llama-server"
    fake_bin.write_text("#!/bin/true\n")
    fake_bin.chmod(0o755)
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"GGUF")
    # Accepts the `ffn_cpu_layers` argument the real function gained on 2026-07-28:
    # `_ensure_server()` threads the proposer's ACTUAL offload count through so the free-VRAM
    # guard budgets for the offload the server will really launch with, rather than for whatever
    # `_default_ffn_cpu_layers()` happens to return on a second, later read. A 0-arg stub here
    # does not just fail -- it would, if written as `*_a, **_k`, silently stop exercising that
    # wiring, so the parameter is named rather than swallowed.
    monkeypatch.setattr(
        m, "_generator_server_and_env", lambda _ffn_cpu_layers=None, _mtp=None: (fake_bin, None)
    )
    p = m.LocalGGUFProposer()
    p.model_path = str(fake_gguf)
    p.port = 45001

    # _healthy must be False on the FIRST call and True afterwards. `_ensure_server` opens with
    # `if self._healthy(): return True` ("reuse an already-running server"), so a stub that is
    # always True short-circuits before Popen is ever reached -- and then
    # `_FakePopen.last_kwargs` stays empty, `None is not subprocess.DEVNULL` is trivially true,
    # and test_stderr_is_not_devnull PASSES WITHOUT TESTING ANYTHING. That false pass actually
    # happened while writing this file; the three sibling tests failing is what exposed it. It is
    # the forced-gate shape CLAUDE.md warns about: a pass whose region cannot fail.
    state = {"calls": 0}

    def _healthy(self):
        state["calls"] += 1
        return state["calls"] > 1

    monkeypatch.setattr(type(p), "_healthy", _healthy, raising=False)
    return p


def _assert_popen_actually_ran():
    """Guard every assertion below against the vacuous-pass described in the fixture."""
    assert _FakePopen.last_kwargs, (
        "Popen was never called -- _ensure_server short-circuited, so any assertion about its "
        "kwargs would pass vacuously"
    )


def test_stderr_is_not_devnull(launched):
    """THE REGRESSION THIS PINS: reverting to stderr=subprocess.DEVNULL must fail here."""
    _FakePopen.last_kwargs = {}
    launched._ensure_server()
    _assert_popen_actually_ran()
    assert _FakePopen.last_kwargs.get("stderr") is not subprocess.DEVNULL, (
        "the server's stderr is the ONLY place the llama_decode ret= discriminator appears; "
        "discarding it makes mode A and mode B indistinguishable"
    )


def test_stderr_is_a_file_not_a_pipe(launched):
    """A PIPE with no reader deadlocks the server once the ~64KB buffer fills."""
    _FakePopen.last_kwargs = {}
    launched._ensure_server()
    _assert_popen_actually_ran()
    sink = _FakePopen.last_kwargs.get("stderr")
    assert sink is not subprocess.PIPE, "a pipe with no reader would hang the server"
    assert hasattr(sink, "write"), "expected a writable file object"


def test_log_path_is_recorded_so_a_human_can_find_it(launched, tmp_path):
    """Capturing diagnostics nobody can locate is the same dead channel in a new place."""
    _FakePopen.last_kwargs = {}
    launched._ensure_server()
    _assert_popen_actually_ran()
    lp = getattr(launched, "_stderr_log_path", None)
    assert lp is not None, "the launch must record WHERE it wrote the stderr log"
    assert str(tmp_path) in str(lp), "must honour CARNOT_ARC_SERVER_LOG_DIR"
    assert str(launched.port) in str(lp), "port in the name so concurrent arms do not collide"


def test_capture_failure_falls_back_to_devnull_and_never_blocks_the_launch(launched, monkeypatch):
    """Losing diagnostics must never cost us the generator itself."""

    def _boom(*a, **k):
        raise OSError("read-only filesystem")

    _FakePopen.last_kwargs = {}
    monkeypatch.setattr(Path, "mkdir", _boom)
    assert launched._ensure_server() is True, "a failed log open must not fail the launch"
    _assert_popen_actually_ran()
    assert getattr(launched, "_stderr_log_path", "sentinel") is None
    assert _FakePopen.last_kwargs.get("stderr") is subprocess.DEVNULL

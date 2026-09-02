"""The opt-in single-stream slot pin (`CARNOT_ARC_LLAMA_SERVER_PARALLEL`).

REQ-ARC-WMTE-6870 / SCENARIO-ARC-WMTE-6870-PARALLEL-REACHES-THE-SERVER

WHY THIS FILE EXISTS. Measured 2026-09-02: with no `--parallel`, llama-server auto-picks 4
kv_unified slots that SHARE one `-c` pool. A single fresh induce stream owns the whole pool
(a 15,000-token ignore_eos generation completed in full at -c 49152). But the live agent reuses
the server across many turns with `cache_prompt=True`, so sibling slots accumulate cached prompts
and hoard the shared pool; an induce firing after several turns then truncates far below its
budget (the r11l run: 18431/2996/4066 of a 26800 budget). The offline eval runs games strictly
sequentially, so it never needs the 4 slots. Setting `CARNOT_ARC_LLAMA_SERVER_PARALLEL=1` launches
ONE slot, so the lone stream gets the whole pool every turn and no sibling can hoard it.

The dangerous outcome, exactly as for the sibling FFN-offload knob, is the flag being accepted and
SILENTLY DOING NOTHING: built but never appended to the Popen argv is, from outside the process,
indistinguishable from working. So these tests assert the two things that can actually be wrong:
the default path gains NO `--parallel` argument (the scored 4-slot path is byte-identical), and
the env knob's value reaches the exact argv handed to subprocess.Popen.
"""

from __future__ import annotations

import importlib

import pytest

MOD = "carnot.agentic.arc_executable_world_model"


@pytest.fixture()
def wm(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_LLAMA_SERVER_PARALLEL", raising=False)
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    monkeypatch.delenv("CARNOT_ARC_INDUCE_N_CTX", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_CUDA_GPU", raising=False)
    return importlib.import_module(MOD)


# --------------------------------------------------------------------------------------------
# The resolver.
# --------------------------------------------------------------------------------------------


def test_unset_returns_none_so_the_flag_is_omitted(wm, monkeypatch) -> None:
    """Unset MUST mean byte-identical behaviour to before this knob existed: no flag, so the
    server auto-picks its 4 kv_unified slots. None is the caller's signal to append nothing."""
    monkeypatch.delenv("CARNOT_ARC_LLAMA_SERVER_PARALLEL", raising=False)
    assert wm._llama_server_parallel_launch() is None


def test_one_is_read(wm, monkeypatch) -> None:
    monkeypatch.setenv("CARNOT_ARC_LLAMA_SERVER_PARALLEL", "1")
    assert wm._llama_server_parallel_launch() == 1


@pytest.mark.parametrize("bad", ["zero", "", "  ", "1.5", "0", "65", "-1"])
def test_malformed_or_out_of_range_degrades_to_none(wm, monkeypatch, bad) -> None:
    """A typo or a nonsense count must NOT launch a broken server. It degrades to None (flag
    omitted) rather than crashing the live path. 0 and 65 are out of the 1..64 bound."""
    monkeypatch.setenv("CARNOT_ARC_LLAMA_SERVER_PARALLEL", bad)
    assert wm._llama_server_parallel_launch() is None


# --------------------------------------------------------------------------------------------
# The part that actually matters: does the flag REACH the server process?
# --------------------------------------------------------------------------------------------


def _launch_and_capture_argv(wm, monkeypatch, tmp_path) -> list[str]:
    """Drive the real `_ensure_server()` with a fake Popen and return the argv it built. Only the
    external dependencies (binary on disk, GGUF on disk, subprocess, health poll) are faked; the
    argv construction under test is the real code path. Mirrors
    tests/python/test_arc_ffn_cpu_offload.py:_launch_and_capture_argv."""
    fake_server = tmp_path / "llama-server"
    fake_server.write_text("#!/bin/sh\nexit 0\n")
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"\0")

    monkeypatch.setattr(
        wm,
        "_generator_server_and_env",
        lambda _ffn_cpu_layers=None, _mtp=None: (fake_server, None),
    )
    monkeypatch.setattr(wm, "_resolve_gguf", lambda _s: str(fake_gguf))

    captured: dict[str, list[str]] = {}

    class _FakeProc:
        pid = 1234

    def _fake_popen(args, **_kw):
        captured["argv"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(wm.subprocess, "Popen", _fake_popen)

    prop = wm.LocalGGUFProposer(
        repo_substr="gemma-4-31B-it", n_ctx=32768, kv_quant="q8_0", ffn_cpu_layers=0
    )
    calls = {"n": 0}

    def _healthy():
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(prop, "_healthy", _healthy)

    assert prop._ensure_server() is True
    assert "argv" in captured, "_ensure_server never reached subprocess.Popen"
    return captured["argv"], prop


def test_default_launch_argv_has_no_parallel_flag(wm, monkeypatch, tmp_path):
    """The opt-in contract: knob off -> argv is exactly what it was before, so the scored 4-slot
    path is untouched."""
    argv, prop = _launch_and_capture_argv(wm, monkeypatch, tmp_path)
    assert "--parallel" not in argv, argv
    assert prop.last_parallel_launch is None


def test_env_knob_alone_puts_parallel_1_on_the_argv(wm, monkeypatch, tmp_path):
    """THE test. The operator sets ONLY an env var (not a constructor arg). Prove `--parallel 1`
    reaches the exact argv handed to subprocess.Popen, and is recorded on the instance."""
    monkeypatch.setenv("CARNOT_ARC_LLAMA_SERVER_PARALLEL", "1")
    argv, prop = _launch_and_capture_argv(wm, monkeypatch, tmp_path)
    assert "--parallel" in argv, argv
    assert argv[argv.index("--parallel") + 1] == "1"
    assert prop.last_parallel_launch == 1

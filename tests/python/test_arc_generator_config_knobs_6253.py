"""Tests for REQ-ARC-WMTE-6253: env knobs for the two 96GB-relevant generator constants.

The scored card moved from a 16-24 GB class device to a single 96 GB RTX PRO 6000 on
2026-08-11. Two constants that were correct for the old card had NO env override, so the
sizing could not be tested on the real hardware without editing source. These tests pin
BOTH the new knobs AND, more importantly, that the shipped defaults did not move.

Covers SCENARIO-ARC-WMTE-6253-DEFAULTS-UNCHANGED,
SCENARIO-ARC-WMTE-6253-SLOTS-KNOB-RESIZES-CONTEXT, and
SCENARIO-ARC-WMTE-6253-KV-QUANT-KNOB-REACHES-ARGV.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from carnot.agentic.arc_executable_world_model import (
    _default_induce_n_ctx,
    _LLAMA_SERVER_DEFAULT_SLOTS,
    _llama_server_slots,
)


@contextmanager
def _env(**kw):
    """Set env vars for one test and always restore. A leaked knob would silently
    reconfigure every later test in the same process."""
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# SCENARIO-ARC-WMTE-6253-DEFAULTS-UNCHANGED


def test_unset_slots_env_returns_the_shipped_default() -> None:
    with _env(CARNOT_ARC_LLAMA_SERVER_SLOTS=None):
        assert _llama_server_slots() == _LLAMA_SERVER_DEFAULT_SLOTS == 4


def test_empty_slots_env_returns_the_shipped_default() -> None:
    with _env(CARNOT_ARC_LLAMA_SERVER_SLOTS=""):
        assert _llama_server_slots() == 4


def test_unset_slots_env_leaves_context_size_byte_identical() -> None:
    # The whole safety argument for this change is "the default does not move".
    # 4 * (22352 + 4096) = 105792, rounded up to a 4096 multiple = 106496.
    with _env(CARNOT_ARC_LLAMA_SERVER_SLOTS=None, CARNOT_ARC_INDUCE_N_CTX=None):
        assert _default_induce_n_ctx() == 106496


# SCENARIO-ARC-WMTE-6253-SLOTS-KNOB-RESIZES-CONTEXT


def test_slots_knob_scales_the_context_pool() -> None:
    with _env(CARNOT_ARC_LLAMA_SERVER_SLOTS="8", CARNOT_ARC_INDUCE_N_CTX=None):
        assert _llama_server_slots() == 8
        assert _default_induce_n_ctx() == 212992  # exactly twice the K=4 pool


def test_garbage_slots_value_falls_back_rather_than_raising() -> None:
    # A bad env value must never propagate into the VRAM arithmetic, and must never
    # crash the generator at launch time either.
    with _env(CARNOT_ARC_LLAMA_SERVER_SLOTS="not-a-number"):
        assert _llama_server_slots() == 4


def test_out_of_range_slots_value_falls_back() -> None:
    for bad in ("0", "-3", "999"):
        with _env(CARNOT_ARC_LLAMA_SERVER_SLOTS=bad):
            assert _llama_server_slots() == 4, bad


def test_explicit_n_ctx_override_still_wins_over_the_slots_knob() -> None:
    # CARNOT_ARC_INDUCE_N_CTX is the pre-existing hard override. The new knob must not
    # quietly outrank it, or an operator pinning a pool would silently not get it.
    with _env(CARNOT_ARC_LLAMA_SERVER_SLOTS="8", CARNOT_ARC_INDUCE_N_CTX="65536"):
        assert _default_induce_n_ctx() == 65536


# ---------------------------------------------------------------------------
# SCENARIO-ARC-WMTE-6253-KV-QUANT-KNOB-REACHES-ARGV
#
# The knob is only real if it reaches the argv handed to subprocess.Popen. A value that
# is read but never appended is indistinguishable, from outside, from one that is
# ignored. This reuses the fake-Popen harness pattern from
# tests/python/test_arc_ffn_cpu_offload.py: every faked thing is an EXTERNAL dependency
# (binary on disk, GGUF on disk, the subprocess, the health poll); the argv construction
# under test is the real code path.
# ---------------------------------------------------------------------------

import importlib  # noqa: E402

import pytest  # noqa: E402

MOD = "carnot.agentic.arc_executable_world_model"


@pytest.fixture
def wm(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    monkeypatch.delenv("CARNOT_ARC_INDUCE_N_CTX", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_CUDA_GPU", raising=False)
    monkeypatch.delenv("CARNOT_ARC_KV_QUANT", raising=False)
    monkeypatch.delenv("CARNOT_ARC_LLAMA_SERVER_SLOTS", raising=False)
    return importlib.import_module(MOD)


def _capture_argv(wm, monkeypatch, tmp_path, *, kv_quant="q8_0"):
    fake_server = tmp_path / "llama-server"
    fake_server.write_text("#!/bin/sh\nexit 0\n")
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"\0")
    monkeypatch.setattr(
        wm, "_generator_server_and_env", lambda _ffn_cpu_layers=None, _mtp=None: (fake_server, None)
    )
    monkeypatch.setattr(wm, "_resolve_gguf", lambda _s: str(fake_gguf))
    captured: dict[str, list[str]] = {}

    class _FakeProc:
        pid = 1234

    def _fake_popen(args, **_kw):
        captured["argv"] = list(args)
        return _FakeProc()

    monkeypatch.setattr(wm.subprocess, "Popen", _fake_popen)
    prop = wm.LocalGGUFProposer(repo_substr="gemma-4-31B-it", n_ctx=32768, kv_quant=kv_quant)
    calls = {"n": 0}

    def _healthy():
        calls["n"] += 1
        return calls["n"] > 1

    monkeypatch.setattr(prop, "_healthy", _healthy)
    assert prop._ensure_server() is True
    assert "argv" in captured, "_ensure_server never reached subprocess.Popen"
    return captured["argv"], prop


def test_default_kv_quant_argv_is_unchanged_by_this_change(wm, monkeypatch, tmp_path) -> None:
    argv, prop = _capture_argv(wm, monkeypatch, tmp_path)
    assert "--cache-type-k" in argv
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v") + 1] == "q8_0"
    assert prop.last_kv_quant_used == "q8_0"


def test_kv_quant_env_knob_reaches_the_launch_argv(wm, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CARNOT_ARC_KV_QUANT", "f16")
    argv, prop = _capture_argv(wm, monkeypatch, tmp_path)
    assert argv[argv.index("--cache-type-k") + 1] == "f16"
    assert argv[argv.index("--cache-type-v") + 1] == "f16"
    assert prop.last_kv_quant_used == "f16"


def test_kv_quant_none_drops_the_flags_entirely(wm, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CARNOT_ARC_KV_QUANT", "none")
    argv, prop = _capture_argv(wm, monkeypatch, tmp_path)
    assert "--cache-type-k" not in argv
    assert "--cache-type-v" not in argv
    assert prop.last_kv_quant_used is None


def test_kv_quant_env_overrides_the_constructor_argument(wm, monkeypatch, tmp_path) -> None:
    # An operator setting the env var must win over a call site that hardcoded q8_0 --
    # both live construction sites pass it explicitly, so an env var that lost to the
    # argument would be unreachable in production and therefore useless.
    monkeypatch.setenv("CARNOT_ARC_KV_QUANT", "f16")
    argv, _ = _capture_argv(wm, monkeypatch, tmp_path, kv_quant="q8_0")
    assert argv[argv.index("--cache-type-k") + 1] == "f16"

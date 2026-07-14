"""Tests for LocalGGUFProposer.extra_server_args (task 14 -- the -fit off fix, REQ-ARC-WMTE-5599-2).

Spec refs: REQ-ARC-WMTE-5599-2.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from carnot.agentic.arc_executable_world_model import LocalGGUFProposer


def _make_proposer(**kwargs) -> LocalGGUFProposer:
    return LocalGGUFProposer(model_path="/fake/model.gguf", port=9999, **kwargs)


def _launch_args_for(monkeypatch, proposer: LocalGGUFProposer) -> list[str]:
    """Drives _ensure_server() far enough to capture the constructed subprocess args,
    without actually launching anything or waiting for a real health check."""

    import carnot.agentic.arc_executable_world_model as mod

    monkeypatch.setattr(proposer, "_healthy", lambda: False)
    monkeypatch.setattr(
        mod, "_generator_server_and_env", lambda: (Path("/fake/llama-server"), None)
    )
    monkeypatch.setattr(mod.Path, "exists", lambda self: True)

    captured: dict[str, list[str]] = {}

    def _fake_popen(args, **_kwargs):
        captured["args"] = args
        return MagicMock()

    monkeypatch.setattr(mod.subprocess, "Popen", _fake_popen)
    # The post-launch health-poll loop always sees _healthy()=False and runs
    # max(90, timeout/2) iterations; patch sleep to a no-op so this returns fast --
    # we only need the args captured at the subprocess.Popen call above.
    monkeypatch.setattr(mod.time, "sleep", lambda *_a: None)
    proposer._ensure_server()
    return captured["args"]


def test_extra_server_args_defaults_to_empty_tuple() -> None:
    proposer = _make_proposer()
    assert proposer.extra_server_args == ()


def test_extra_server_args_omitted_from_launch_when_unset(monkeypatch) -> None:
    proposer = _make_proposer()
    args = _launch_args_for(monkeypatch, proposer)
    assert "-fit" not in args


def test_extra_server_args_appended_to_launch_when_set(monkeypatch) -> None:
    proposer = _make_proposer(extra_server_args=("-fit", "off"))
    args = _launch_args_for(monkeypatch, proposer)
    assert args[-2:] == ["-fit", "off"]


def test_extra_server_args_appended_after_kv_quant_and_mtp_flags(monkeypatch) -> None:
    proposer = _make_proposer(
        mtp=True, kv_quant="q8_0", extra_server_args=("-fit", "off", "--parallel", "1")
    )
    args = _launch_args_for(monkeypatch, proposer)
    assert "--spec-type" in args
    assert "--cache-type-k" in args
    assert args[-4:] == ["-fit", "off", "--parallel", "1"]

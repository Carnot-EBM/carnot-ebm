"""Tests for _meaningful_error_tail in scripts/research_conductor.py.

Origin: 2026-05-29. codex (and any echoing agent) prints the full prompt to
stdout before working. On failure the conductor logged full_output[-N:], which
for a long prompt is just the END OF THE ECHOED PROMPT, not the agent's error.
Every "Codex CLI error: ...you finish the real work inside 10 minutes, that is
correct" log this session (the verbatim last line of the stop_postamble) was
undiagnosable for this reason. _meaningful_error_tail strips the echoed prompt.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load():
    root = Path(__file__).resolve().parents[2]
    p = root / "scripts" / "research_conductor.py"
    spec = importlib.util.spec_from_file_location("research_conductor", p)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["research_conductor"] = m
    spec.loader.exec_module(m)
    return m


_RC = _load()
_tail = _RC._meaningful_error_tail


def test_strips_echoed_prompt_returns_real_error():
    prompt = "Do the task. " + ("X" * 300) + " If you finish inside 10 minutes that is correct."
    # codex echoes the prompt, then prints a real error AFTER it
    full = "OpenAI Codex\nuser\n" + prompt + "\nerror: model gpt-5.5 rate_limited (429)\n"
    out = _tail(full, prompt, 500)
    assert "rate_limited" in out
    assert "If you finish inside 10 minutes" not in out  # echoed prompt stripped


def test_no_output_after_prompt_reports_clearly():
    prompt = "Do the task. " + ("Y" * 300) + " exit promptly."
    # codex echoed the prompt and produced NOTHING after (failed during ingestion)
    full = "OpenAI Codex\nuser\n" + prompt
    out = _tail(full, prompt, 500)
    assert "NO generated output after prompt ingestion" in out
    # the confusing prompt fragment must NOT be presented as the error
    assert out.count("exit promptly") <= 1  # only in the raw-tail diagnostic, labeled


def test_the_real_world_postamble_case():
    # the exact failure signature from .307 / .299
    postamble = "If you finish the real work inside 10 minutes, that is correct and expected — exit promptly."
    prompt = "EXPERIMENT SPEC ... " + ("Z" * 400) + " " + postamble
    full = "OpenAI Codex\nuser\n" + prompt  # nothing after = the masked failure
    out = _tail(full, prompt, 500)
    # the old behavior returned the postamble as if it were the error; now it's labeled
    assert "NO generated output after prompt ingestion" in out


def test_empty_output():
    assert _tail("", "some prompt", 500) == "(no output captured)"


def test_short_prompt_no_strip_needed():
    # short prompt (< 40 chars) — no echo-stripping, just tail
    out = _tail("some agent output here that is the real error", "hi", 500)
    assert "real error" in out


def test_passthrough_when_prompt_absent_from_output():
    # if the prompt was never echoed, return the tail unchanged
    full = "agent ran and produced this genuine error trace line"
    out = _tail(full, "a totally different prompt string not present in output xxxxxxxxxxxxxxxxxxxx", 500)
    assert "genuine error trace" in out

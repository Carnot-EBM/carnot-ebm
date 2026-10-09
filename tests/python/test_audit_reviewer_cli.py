"""REQ-OPS-AUDIT-REVIEWER-1: audit reviewer calls honor CODEX_BIN and keep the real error.

Incident (2026-10-08): six hostile-reviewer audits launched plain `codex`. Under the conductor
service that is the old system copy, which rejects newer model names with HTTP 400. They also kept
only the FIRST 200 characters of stderr, which is codex's startup banner, so the error was never
seen. The old tests mocked the whole `_call` function, so the subprocess wrapper that failed was
never exercised. These tests run the real wrappers against a fake `subprocess.run`.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_reviewer_cli as arc  # noqa: E402

# A realistic codex failure: banner, then an echo of the whole prompt, then the real error.
BANNER = (
    "OpenAI Codex v0.156.1\n--------\nworkdir: /repo\nmodel: gpt-6.1-sol\nprovider: openai\n"
    "approval: never\nsandbox: workspace-write\n--------\nuser\n"
)
ERROR = (
    'ERROR: {"status":400,"error":{"message":"The \'gpt-6.1-sol\' model is not supported '
    'when using Codex with a ChatGPT account."}}'
)
CODEX_STDERR = (
    BANNER
    + ("prompt line\n" * 6000)
    + "warning: Model metadata not found.\n"
    + ERROR
    + "\n"
    + ERROR
)


class FakeRun:
    """Stand-in for subprocess.run that records each command and replays canned results."""

    def __init__(self, results):
        self.results = list(results)
        self.cmds: list[list[str]] = []
        self.kwargs: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.cmds.append([str(c) for c in cmd])
        self.kwargs.append(kwargs)
        rc, out, err = self.results.pop(0)
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)


# ---------------------------------------------------------------- the helper itself


def test_codex_bin_defaults_to_plain_codex(monkeypatch):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-BIN: with no override, the command is `codex`."""
    monkeypatch.delenv("CODEX_BIN", raising=False)
    assert arc.codex_bin() == "codex"


def test_codex_bin_honors_override_and_ignores_empty(monkeypatch):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-BIN: CODEX_BIN wins; an empty value falls back."""
    monkeypatch.setenv("CODEX_BIN", "/opt/newer/codex")
    assert arc.codex_bin() == "/opt/newer/codex"
    monkeypatch.setenv("CODEX_BIN", "")
    assert arc.codex_bin() == "codex"


def test_failure_text_surfaces_the_error_not_the_banner():
    """SCENARIO-OPS-AUDIT-REVIEWER-1-TAIL: the error is visible even in a short prefix."""
    text = arc.failure_text("codex", 1, CODEX_STDERR, binary="/usr/bin/codex")
    assert "not supported" in text[:200]  # the old code kept only the first 200 chars: the banner
    assert "OpenAI Codex v0.156.1" in text  # version is still named, after the error
    assert "/usr/bin/codex" in text
    assert len(text) < 900  # bounded: a 65k echo must not flood the report
    body = text.split(":", 1)[1].lstrip()
    assert not body.startswith("OpenAI Codex")  # the banner must not lead the message


def test_failure_text_handles_missing_and_short_stderr():
    assert "<no stderr>" in arc.failure_text("agy", 1, "")
    assert "<no stderr>" in arc.failure_text("agy", 1, None)
    short = arc.failure_text("agy", 0, "jetski: no output produced")
    assert "jetski: no output produced" in short and "agy exit 0" in short


def test_chain_failure_keeps_both_reasons():
    both = arc.chain_failure("agy exit 0: jetski denied", "codex exit 1: not supported")
    assert "jetski denied" in both and "not supported" in both


# ---------------------------------------------------------------- the six real wrappers

# (module, callable, positional args after the prompt-body shape, keyword)
SIX = [
    ("experiment_claim_audit", lambda m: m._call("codex", "m", "p", "b")),
    ("artifact_convention_audit", lambda m: m._call("codex", "m", "p", "b")),
    ("verifier_authenticity_audit", lambda m: m.call_codex("p", "b", model="m")),
    ("qa_layer_authenticity_audit", lambda m: m.call_codex("p", "b", model="m")),
    ("pages_adversarial_audit", lambda m: m.call_codex("p", model="m")),
    ("arc_self_solve_audit", lambda m: m.call_codex("p", "b", model="m")),
]


@pytest.mark.parametrize("name,call", SIX, ids=[s[0] for s in SIX])
def test_codex_fallback_uses_codex_bin_and_reports_the_real_error(name, call, monkeypatch):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-BIN and -TAIL, on each of the six real wrappers."""
    mod = importlib.import_module(name)
    monkeypatch.setenv("CODEX_BIN", "/opt/newer/codex")
    fake = FakeRun([(1, "", CODEX_STDERR)])
    monkeypatch.setattr(mod.subprocess, "run", fake)
    ok, text = call(mod)
    assert ok is False
    assert fake.cmds and fake.cmds[0][0] == "/opt/newer/codex"
    assert "not supported" in text[:200]
    assert len(text) < 900


@pytest.mark.parametrize("name,call", SIX, ids=[s[0] for s in SIX])
def test_codex_wrapper_without_override_runs_plain_codex(name, call, monkeypatch):
    mod = importlib.import_module(name)
    monkeypatch.delenv("CODEX_BIN", raising=False)
    fake = FakeRun([(0, "## VERDICT\nOK\n", BANNER)])
    monkeypatch.setattr(mod.subprocess, "run", fake)
    ok, text = call(mod)
    assert ok is True and "VERDICT" in text
    assert fake.cmds[0][0] == "codex"


AGY_DENIED = (0, "", 'jetski: no output produced - a tool required the "command" permission')


@pytest.mark.parametrize("name", ["experiment_claim_audit", "artifact_convention_audit"])
def test_agy_failure_then_codex_failure_reports_both(name, monkeypatch):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-CHAIN: when agy and the fallback both fail, both reasons show."""
    mod = importlib.import_module(name)
    monkeypatch.setenv("CODEX_BIN", "/opt/newer/codex")
    fake = FakeRun([AGY_DENIED, (1, "", CODEX_STDERR)])
    monkeypatch.setattr(mod.subprocess, "run", fake)
    ok, text = mod._call("agy", "gemini-x", "p", "b")
    assert ok is False
    assert "no output produced" in text and "not supported" in text
    assert fake.cmds[1][0] == "/opt/newer/codex"  # the fallback used CODEX_BIN, not /usr/bin


@pytest.mark.parametrize("name", ["experiment_claim_audit", "artifact_convention_audit"])
def test_agy_failure_then_codex_success_returns_the_answer(name, monkeypatch):
    mod = importlib.import_module(name)
    fake = FakeRun([AGY_DENIED, (0, "## VERDICT\nNO_CLAIM\n", BANNER)])
    monkeypatch.setattr(mod.subprocess, "run", fake)
    ok, text = mod._call("agy", "gemini-x", "p", "b")
    assert ok is True and "NO_CLAIM" in text


def test_claim_audit_report_row_keeps_the_error_visible(monkeypatch, tmp_path):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-TAIL: the report row is not cut back to the banner."""
    import json

    mod = importlib.import_module("experiment_claim_audit")
    art = tmp_path / "experiment_9301_x.json"
    art.write_text(json.dumps({"honest_verdict": "complete: ok"}))
    report = tmp_path / "report.md"
    monkeypatch.setattr(mod, "REPORT", report)
    monkeypatch.setenv("CODEX_BIN", "/opt/newer/codex")
    monkeypatch.setattr(mod.subprocess, "run", FakeRun([AGY_DENIED, (1, "", CODEX_STDERR)]))
    # Drive the real main() and read the real report: this fails if the production row is cut back.
    code = mod.main(
        [
            "--artifact",
            str(art),
            "--budget-seconds",
            "0",
            "--agent-type",
            "agy",
            "--model-name",
            "gemini-x",
        ]
    )
    text = report.read_text()
    assert code == 0 and "CANNOT_DETERMINE" in text
    assert "not supported" in text and "no output produced" in text


# ---------------------------------------------------------------- regression guard


def test_no_audit_script_launches_plain_codex_again():
    """SCENARIO-OPS-AUDIT-REVIEWER-1-GUARD: a literal `codex` command in these six scripts fails.

    The bug was a missing call in six places, so the guard reads the source of every audit.
    The two daily watch scripts are excluded on purpose: their systemd units put ~/.local/bin
    ahead of /usr/bin on PATH, so plain `codex` already resolves to the current binary there.
    """
    pattern = re.compile(r'\[\s*"codex"\s*,\s*"exec"|^\s*"codex",\s*\n\s*"exec"', re.M)
    offenders = [name for name, _ in SIX if pattern.search((SCRIPTS / f"{name}.py").read_text())]
    assert offenders == []
    for name, _ in SIX:
        assert "codex_bin()" in (SCRIPTS / f"{name}.py").read_text(), name


# ---------------------------------------------------------------- chain path in the other four audits

FOUR_AGY = [
    ("verifier_authenticity_audit", lambda m: m.call_agy("p", "b", model="gemini-x")),
    ("qa_layer_authenticity_audit", lambda m: m.call_agy("p", "b", model="gemini-x")),
    ("pages_adversarial_audit", lambda m: m.call_agy("p", model="gemini-x")),
    ("arc_self_solve_audit", lambda m: m.call_agy("p", "b", model="gemini-x")),
]


@pytest.mark.parametrize("name,call", FOUR_AGY, ids=[s[0] for s in FOUR_AGY])
def test_agy_then_codex_failure_reports_both_in_the_other_four(name, call, monkeypatch):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-CHAIN: all six audits keep both reasons, not just two."""
    mod = importlib.import_module(name)
    monkeypatch.setenv("CODEX_BIN", "/opt/newer/codex")
    fake = FakeRun([AGY_DENIED, (1, "", CODEX_STDERR)])
    monkeypatch.setattr(mod.subprocess, "run", fake)
    ok, text = call(mod)
    assert ok is False
    assert "no output produced" in text
    assert "not supported" in text[:400]  # the fallback error stays near the front of the chain
    assert fake.cmds[1][0] == "/opt/newer/codex"


@pytest.mark.parametrize("name,call", FOUR_AGY, ids=[s[0] for s in FOUR_AGY])
def test_agy_then_codex_success_in_the_other_four(name, call, monkeypatch):
    mod = importlib.import_module(name)
    monkeypatch.setattr(
        mod.subprocess, "run", FakeRun([AGY_DENIED, (0, "## VERDICT\nOK\n", BANNER)])
    )
    ok, text = call(mod)
    assert ok is True and "VERDICT" in text


def test_chain_failure_cuts_the_first_reason_short():
    long_first = "agy exit 0: " + ("y" * 400)
    chain = arc.chain_failure(long_first, "codex exit 1: not supported")
    assert chain.index("not supported") < 220
    assert "fallback:" in chain


def test_failure_text_has_no_stray_space_when_there_is_only_a_binary():
    text = arc.failure_text("codex", 1, "", binary="/opt/newer/codex")
    assert "[ at" not in text and "[at /opt/newer/codex]" in text


def test_report_cuts_in_the_two_audits_that_build_their_own_rows_keep_600_chars():
    """SCENARIO-OPS-AUDIT-REVIEWER-1-TAIL: verifier and qa-layer rows were cut to 200 characters."""
    for name in ("verifier_authenticity_audit", "qa_layer_authenticity_audit"):
        src = (SCRIPTS / f"{name}.py").read_text()
        cuts = [int(n) for n in re.findall(r"audit call failed: \{report\[:(\d+)\]\}", src)]
        assert cuts and min(cuts) >= 600, (name, cuts)


def test_failure_text_keeps_a_near_limit_message_whole():
    """A 301-character agy message must not lose its first word to the 300-character tail."""
    err = "jetski: no output produced - " + ("x" * 272)
    assert len(err) == 301
    text = arc.failure_text("agy", 0, err)
    assert "jetski: no output produced" in text


# ---------------------------------------------------------------- agy runs without tools, outside the repo

SIX_AGY = [
    ("experiment_claim_audit", lambda m: m._call("agy", "gemini-x", "p", "b")),
    ("artifact_convention_audit", lambda m: m._call("agy", "gemini-x", "p", "b")),
] + FOUR_AGY


@pytest.mark.parametrize("name,call", SIX_AGY, ids=[s[0] for s in SIX_AGY])
def test_agy_gets_the_no_tools_note_and_an_empty_directory(name, call, monkeypatch):
    """SCENARIO-OPS-AUDIT-REVIEWER-1-AGY: headless agy denied its own tool calls inside the repo.

    On the real claim-audit packet, 0 of 3 runs in the repo gave output and 4 of 4 runs gave output
    with this note, run from an empty directory. Each of the six audits must do both.
    """
    mod = importlib.import_module(name)
    fake = FakeRun([(0, "## VERDICT\nOK\n", "")])
    monkeypatch.setattr(mod.subprocess, "run", fake)
    ok, _ = call(mod)
    assert ok is True
    cmd, kw = fake.cmds[0], fake.kwargs[0]
    assert cmd[1:3] == ["--model", "gemini-x"] and cmd[3] == "--print"
    assert cmd[4].startswith(arc.AGY_NO_TOOLS_NOTE)
    cwd = Path(kw["cwd"])
    assert cwd == Path(arc.agy_cwd())
    assert cwd.is_dir() and list(cwd.iterdir()) == []
    assert REPO not in cwd.parents and cwd != REPO


def test_agy_cwd_is_private_and_stable():
    first = Path(arc.agy_cwd())
    assert arc.agy_cwd() == str(first)  # same path every call, no new directory per run
    assert (first.stat().st_mode & 0o077) == 0  # not readable by other users

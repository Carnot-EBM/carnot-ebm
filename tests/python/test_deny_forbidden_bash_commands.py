"""Tests for the forbidden-command deny-hook and the submission-verb WARN scan.

REQ: REQ-CONDUCTOR-DENYHOOK-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-CONDUCTOR-DENYHOOK-1,
SCENARIO-CONDUCTOR-DENYHOOK-2,
SCENARIO-CONDUCTOR-DENYHOOK-3,
SCENARIO-CONDUCTOR-DENYHOOK-4.

Layer 1 is the PreToolUse Bash hook (deny at the harness); Layer 2 is
the WARN-only prompt scan at roadmap activation. The forbidden strings
below appear as test DATA only. No test writes tracked state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import deny_forbidden_bash_commands as deny  # noqa: E402
import research_conductor as rc  # noqa: E402

HOOK = REPO_ROOT / "scripts" / "deny_forbidden_bash_commands.py"


def _run_hook(payload: object) -> subprocess.CompletedProcess:
    raw = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=raw,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_stash_denied_and_readonly_forms_allowed() -> None:
    # SCENARIO-CONDUCTOR-DENYHOOK-1
    msg = deny.forbidden("git stash")
    assert msg is not None and "safe-pull.sh" in msg
    assert deny.forbidden("git stash pop") is not None
    assert deny.forbidden("git stash list") is None
    assert deny.forbidden("git stash show -p") is None


def test_no_verify_denied_only_on_git_commands() -> None:
    # SCENARIO-CONDUCTOR-DENYHOOK-2
    assert deny.forbidden("git commit --no-verify -m x") is not None
    assert deny.forbidden("git push --no-verify") is not None
    assert deny.forbidden("git commit -m x") is None
    # A non-git tool's --no-verify flag is not this rule's business, and
    # the single-command guard keeps a pipe from leaking the match.
    assert deny.forbidden("sometool --no-verify") is None
    assert deny.forbidden("git log | sometool --no-verify") is None


def test_publication_commands_denied() -> None:
    # SCENARIO-CONDUCTOR-DENYHOOK-3
    for cmd in (
        "arxiv submit paper.tar.gz",
        "python -m openreview.api upload",
        "gh release create v1.0 --notes x",
        "twine upload dist/*",
    ):
        assert deny.forbidden(cmd) is not None, cmd


def test_benign_commands_allowed() -> None:
    # REQ-CONDUCTOR-DENYHOOK-1: the boundary is six command shapes, not
    # a general gate — ordinary work must pass untouched.
    for cmd in (
        "git status --short",
        "git add -A && git commit -m '[outer-loop] preserve transient state'",
        "scripts/safe-pull.sh",
        ".venv/bin/pytest tests/python --no-cov -q",
        "gh release list",
        "gh pr create --title x",
        "pip install twine",
    ):
        assert deny.forbidden(cmd) is None, cmd


def test_hook_end_to_end_denies_with_exit_2() -> None:
    # SCENARIO-CONDUCTOR-DENYHOOK-1 at the process boundary: exit 2 +
    # the rule named on stderr is the harness deny contract.
    proc = _run_hook({"tool_name": "Bash", "tool_input": {"command": "git stash"}})
    assert proc.returncode == 2
    assert "DENIED" in proc.stderr and "safe-pull.sh" in proc.stderr
    ok = _run_hook({"tool_name": "Bash", "tool_input": {"command": "git status"}})
    assert ok.returncode == 0


def test_hook_fails_open_on_malformed_input() -> None:
    # REQ-CONDUCTOR-DENYHOOK-1: deliberate fail-open — a harness glitch
    # must not block every Bash call (that trains bypassing the hook).
    proc = _run_hook("this is not json {")
    assert proc.returncode == 0
    assert "allowing" in proc.stderr


def test_submission_verb_warning_fires_and_override_clears() -> None:
    # SCENARIO-CONDUCTOR-DENYHOOK-4
    task = {"id": "exp-x", "prompt": "step 7: run arxiv submit on the package"}
    warns = rc._submission_verb_warnings({"tasks": [task]})
    assert len(warns) == 1 and "exp-x" in warns[0]
    task_overridden = dict(task, operator_override="2026-08-21 operator authorized")
    assert rc._submission_verb_warnings({"tasks": [task_overridden]}) == []


def test_submission_scan_ignores_bare_openreview_mention() -> None:
    # Layer 2 keys openreview on a nearby submit/upload verb: the bare
    # name appears legitimately in literature-discussion prompts.
    clean = {"id": "exp-lit", "prompt": "read the OpenReview discussion of paper X"}
    assert rc._submission_verb_warnings({"tasks": [clean]}) == []
    hot = {"id": "exp-sub", "prompt": "use openreview API to submit the draft"}
    assert len(rc._submission_verb_warnings({"tasks": [hot]})) == 1

"""REQ-CONDUCTOR-DENYHOOK-1: the deny-hook wiring survives a drifted shell cwd.

INCIDENT 2026-09-30. `.claude/settings.json` wired the hook as
`python3 scripts/deny_forbidden_bash_commands.py`, a path relative to the shell's cwd. A Bash call
that left the tool shell in `python/` made that path resolve to a file that does not exist.
`python3` exits 2 for a missing script, and exit 2 is how a PreToolUse hook says "deny", so EVERY
later Bash call was denied, including the `cd` that would have fixed it. The script's own docstring
says the hook fails OPEN; the wiring made it fail CLOSED.

These tests run the command that settings.json actually configures, from a cwd that is NOT the repo
root, so they bite the wiring and not just the script (test_deny_forbidden_bash_commands.py already
covers the script).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SETTINGS = REPO / ".claude" / "settings.json"
DENY_PAYLOAD = json.dumps({"tool_input": {"command": "git stash"}})
BENIGN_PAYLOAD = json.dumps({"tool_input": {"command": "ls -la"}})


def _configured_command() -> str:
    """The exact command string settings.json runs for the Bash PreToolUse deny hook."""
    hooks = json.loads(SETTINGS.read_text(encoding="utf-8"))["hooks"]["PreToolUse"]
    for group in hooks:
        if group.get("matcher") == "Bash":
            for hook in group["hooks"]:
                if "deny_forbidden_bash_commands" in hook["command"]:
                    return str(hook["command"])
    raise AssertionError("no Bash PreToolUse deny hook is configured in .claude/settings.json")


def _run(payload: str, cwd: Path, project_dir: Path | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PROJECT_DIR"}
    if project_dir is not None:
        env["CLAUDE_PROJECT_DIR"] = str(project_dir)
    return subprocess.run(
        ["bash", "-c", _configured_command()],
        input=payload,
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        timeout=30,
    )


def test_the_wiring_is_not_a_bare_relative_path() -> None:
    command = _configured_command()
    assert "CLAUDE_PROJECT_DIR" in command
    assert not command.startswith("python3 scripts/")


def test_a_forbidden_command_is_still_denied_from_a_drifted_cwd(tmp_path: Path) -> None:
    """The guard must keep working when the shell is somewhere other than the repo root."""
    result = _run(DENY_PAYLOAD, cwd=tmp_path, project_dir=REPO)
    assert result.returncode == 2
    assert "DENIED" in result.stderr


def test_a_benign_command_is_allowed_from_a_drifted_cwd(tmp_path: Path) -> None:
    result = _run(BENIGN_PAYLOAD, cwd=tmp_path, project_dir=REPO)
    assert result.returncode == 0


def test_a_missing_script_fails_open_not_closed(tmp_path: Path) -> None:
    """The incident itself: no script at the resolved path must ALLOW, never exit 2."""
    result = _run(BENIGN_PAYLOAD, cwd=tmp_path, project_dir=tmp_path)
    assert result.returncode == 0
    assert "not found" in result.stderr


def test_unset_project_dir_from_the_repo_root_still_denies() -> None:
    """Back-compat: with no CLAUDE_PROJECT_DIR the old cwd-relative behavior is unchanged."""
    result = _run(DENY_PAYLOAD, cwd=REPO, project_dir=None)
    assert result.returncode == 2
    assert "DENIED" in result.stderr


def test_unset_project_dir_from_a_drifted_cwd_fails_open(tmp_path: Path) -> None:
    """Before the fix this exited 2 and blocked every Bash call."""
    result = _run(BENIGN_PAYLOAD, cwd=tmp_path, project_dir=None)
    assert result.returncode == 0


@pytest.mark.parametrize("payload", [DENY_PAYLOAD, BENIGN_PAYLOAD])
def test_exit_status_of_the_script_is_propagated(payload: str) -> None:
    """The `if` wrapper must not swallow a deny: exit 2 from the script reaches the harness."""
    expected = 2 if payload == DENY_PAYLOAD else 0
    assert _run(payload, cwd=REPO, project_dir=REPO).returncode == expected

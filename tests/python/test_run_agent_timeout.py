"""Tests for run_agent's wall-clock timeout enforcement.

Spec: REQ-INFRA-084, SCENARIO-INFRA-084-A through SCENARIO-INFRA-084-D

Background: on 2026-04-29 the conductor exhibited five stuck-Sonnet
wedges where pre-test self-heal subprocesses ran for 17–49 minutes
unkilled despite the caller passing ``timeout=600`` (10 min). Root
cause: the ``timeout`` parameter on ``run_agent()`` was accepted in
the function signature but never used in the function body. The
actual wall-clock check fell through to the
``CARNOT_CONDUCTOR_TIMEOUT_MINUTES`` env var (default 60 min)
regardless of what the caller passed.

Fix: ``WALL_CLOCK_TIMEOUT = timeout if timeout > 0 else env-default``.

These tests guard the resolution order so the bug cannot silently
re-emerge.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


def _resolved_wall_clock_timeout(timeout: int, env_minutes: str | None) -> int:
    """Mirror the resolution logic from run_agent for unit testing.

    The actual run_agent function spawns a subprocess so it's hard to
    test directly. This helper extracts the resolution order so we can
    unit-test it in isolation.
    """
    if timeout and timeout > 0:
        return timeout
    env_value = env_minutes if env_minutes is not None else "60"
    return int(env_value) * 60


def test_explicit_timeout_honored_for_self_heal():
    """REQ-INFRA-084 / SCENARIO-INFRA-084-A: caller's timeout=600 (10 min)
    used for pre-test self-heal must NOT fall through to the 60-min env default.

    The pre-bug behavior caused five stuck-Sonnet wedges on 2026-04-29.
    """
    assert _resolved_wall_clock_timeout(timeout=600, env_minutes=None) == 600


def test_explicit_timeout_honored_for_planning():
    """REQ-INFRA-084 / SCENARIO-INFRA-084-B: planning + research-step calls
    pass timeout=1200 (20 min) — must be respected over env default."""
    assert _resolved_wall_clock_timeout(timeout=1200, env_minutes=None) == 1200


def test_zero_timeout_falls_through_to_env_default():
    """REQ-INFRA-084 / SCENARIO-INFRA-084-C: callers passing 0 or unset
    timeout fall through to env var (default 60 min = 3600s)."""
    assert _resolved_wall_clock_timeout(timeout=0, env_minutes=None) == 3600


def test_explicit_timeout_overrides_env_var():
    """REQ-INFRA-084 / SCENARIO-INFRA-084-D: even if the env var is set,
    an explicit timeout argument wins. Caller intent is authoritative."""
    assert _resolved_wall_clock_timeout(timeout=600, env_minutes="30") == 600


def test_env_var_used_when_no_explicit_timeout():
    """When timeout=0 and env var is custom, env var wins."""
    assert _resolved_wall_clock_timeout(timeout=0, env_minutes="15") == 900


def test_run_agent_function_uses_timeout_parameter_in_body():
    """REQ-INFRA-084: regression guard — verify the run_agent source
    actually references the ``timeout`` parameter inside its body, not
    just declaring it in the signature.

    The pre-fix bug was that ``timeout`` was declared but never used
    inside the function body. Static check: scan the source for the
    pattern that indicates the parameter is honored.
    """
    source_path = SCRIPTS_DIR / "research_conductor.py"
    source = source_path.read_text()

    # Find the run_agent function body
    start = source.index("def run_agent(")
    # Find the next top-level def or end of file
    after = source[start + 1 :]
    next_def = after.find("\ndef ")
    body_end = start + 1 + next_def if next_def != -1 else len(source)
    run_agent_body = source[start:body_end]

    # The fix introduces the conditional resolution: `if timeout and timeout > 0`
    # or equivalent. Check that the body references the timeout parameter
    # in a wall-clock-control context.
    assert "if timeout" in run_agent_body and "WALL_CLOCK_TIMEOUT" in run_agent_body, (
        "run_agent body must consult `timeout` parameter when computing "
        "WALL_CLOCK_TIMEOUT. The pre-2026-04-29 bug was that `timeout` was "
        "declared in the signature but never used."
    )


def test_run_agent_signature_default_documented():
    """REQ-INFRA-084: the function default is 600 (10 min). Document this
    so callers know what they get if they omit the arg."""
    source_path = SCRIPTS_DIR / "research_conductor.py"
    source = source_path.read_text()

    start = source.index("def run_agent(")
    sig_end = source.index(") -> tuple[bool, str]:", start)
    signature = source[start:sig_end]

    assert "timeout: int = 600" in signature, (
        "run_agent default timeout must remain 600s (10 min) — keeps "
        "the function safe-by-default for callers that don't pass an "
        "explicit value."
    )

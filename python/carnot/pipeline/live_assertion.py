"""live_assertion — Import-time CARNOT_FORCE_LIVE enforcement.

**Why this module exists (RETRO-062):**
    Three consecutive milestone failures (.42, .43, .44) blocked the Live 50q A benchmark
    because ``CARNOT_FORCE_LIVE`` was not set at session start.  The existing
    ``EnvironmentAutoFix`` (env_autofix.py) injects the variable when GPU hardware is
    detected at experiment startup, but there was a gap: if a live GPU experiment script
    imported models or loaded pipelines before calling ``apply_env_autofix()``, or if
    the autofix call was simply forgotten, the experiment could silently fall back to
    synthetic mode and produce misleading results without any error.

    This module provides a **hard import-time assertion** that raises ``RuntimeError``
    immediately when a live GPU is present but ``CARNOT_FORCE_LIVE`` is not ``'1'``.
    The error happens before any model is loaded, making silent fallback structurally
    impossible.

**What this module provides:**
    ``assert_live_gpu_available()`` — raises ``RuntimeError`` when CUDA is available
    but ``CARNOT_FORCE_LIVE != '1'``.  Returns silently when CUDA is absent or the
    var is correctly set.  Safe to call at module import time.

    ``assert_live_or_ci_skip()`` — softer variant for test suites.  Skips the check
    entirely when ``CARNOT_IS_CI=1`` (CI environments don't have live GPUs).  Otherwise
    delegates to ``assert_live_gpu_available()``.

**Relationship to env_autofix:**
    ``apply_env_autofix()`` is a belt-and-suspenders *repair* mechanism — it injects the
    var if it's missing.  ``assert_live_gpu_available()`` is a *hard gate* that runs
    AFTER the repair window has passed.  Call order should be:

        1. ``apply_env_autofix()``           # try to self-heal
        2. ``assert_live_gpu_available()``   # fail loudly if healing didn't work

Spec: REQ-INFRA-082, SCENARIO-INFRA-087, SCENARIO-INFRA-088
"""

from __future__ import annotations

import os

# The error message shown when a live GPU is present but the env var is missing.
# It includes remediation steps so operators can fix it immediately.
_MISSING_VAR_MESSAGE = (
    "CARNOT_FORCE_LIVE must be set to 1 for live GPU experiments.\n"
    "Run: source scripts/session_startup.sh\n"
    "Or: export CARNOT_FORCE_LIVE=1"
)


def assert_live_gpu_available() -> None:
    """Raise RuntimeError when CUDA is available but CARNOT_FORCE_LIVE is not '1'.

    This function is the hard gate that makes silent fallback to synthetic mode
    impossible when a live GPU is present.  It should be called after
    ``apply_env_autofix()`` so the repair window has already passed.

    Algorithm
    ---------
    1. Try to import ``torch``.  If ``ImportError``: return (no GPU possible, assertion
       does not apply — environments without torch cannot have a CUDA GPU).
    2. Call ``torch.cuda.is_available()``.  If ``False``: return (no live GPU present,
       the env var requirement does not apply).
    3. If CUDA is available AND ``os.environ.get('CARNOT_FORCE_LIVE') != '1'``:
       raise ``RuntimeError`` with a clear message and remediation instructions.

    Returns
    -------
    None
        When CUDA is not available or ``CARNOT_FORCE_LIVE='1'`` is already set.

    Raises
    ------
    RuntimeError
        When CUDA is available AND ``CARNOT_FORCE_LIVE`` is absent or not '1'.
        The message includes the exact commands needed to fix the problem.

    Spec: REQ-INFRA-082, SCENARIO-INFRA-087
    """
    # Step 1: probe for torch — if it's not installed there is no CUDA GPU to worry about.
    try:
        import torch  # noqa: PLC0415 — intentional late import (CI may lack torch)
    except ImportError:
        # No torch installed: GPU detection impossible, assertion does not apply.
        return

    # Step 2: check whether CUDA hardware is actually present and available.
    if not torch.cuda.is_available():
        # No live GPU: the env var requirement does not apply.
        return

    # Step 3: CUDA is available — the env var MUST be '1'.
    # Any other value ('0', '', 'false', absent) means the experiment is about to run
    # without the live gate, which caused three consecutive milestone failures (RETRO-062).
    if os.environ.get("CARNOT_FORCE_LIVE") != "1":
        raise RuntimeError(_MISSING_VAR_MESSAGE)


def assert_live_or_ci_skip() -> None:
    """Softer variant of assert_live_gpu_available() for test suites.

    CI environments typically don't have live GPUs but must still be able to run
    the test suite.  This function skips the assertion entirely when ``CARNOT_IS_CI=1``
    is set in the environment, allowing CI to pass without a GPU.

    Algorithm
    ---------
    1. If ``CARNOT_IS_CI=1`` in ``os.environ``: return without checking (CI skip).
    2. Otherwise: call ``assert_live_gpu_available()`` and propagate any exception.

    Returns
    -------
    None
        When CARNOT_IS_CI=1, or when CUDA is not available, or when
        CARNOT_FORCE_LIVE='1' is correctly set.

    Raises
    ------
    RuntimeError
        When CARNOT_IS_CI is not '1' AND CUDA is available AND
        CARNOT_FORCE_LIVE is not '1'.  Same message as assert_live_gpu_available().

    Spec: REQ-INFRA-082, SCENARIO-INFRA-088
    """
    # CI environments skip the GPU assertion entirely — they don't have live hardware.
    if os.environ.get("CARNOT_IS_CI") == "1":
        return

    # For non-CI environments, delegate to the hard gate.
    assert_live_gpu_available()

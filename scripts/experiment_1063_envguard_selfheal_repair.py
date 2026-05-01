#!/usr/bin/env python3
"""Exp 1063 — EnvPropagationGuard self-heal repair + 1 remaining failing test.

Researcher summary
------------------
For three consecutive milestones (.75, .82, and the start of .83) the
conductor's pre-test gate surfaced the SKIP message
``"Pre-tests failing, self-heal failed: EnvPropagationGuard failed to
load CARNOT_ variables"``.  That string is a TRUNCATED form of the
full RuntimeError raised by
``ExperimentTemplate.assert_live_env_if_gpu()``:

    "LIVE-ENV not propagated for GPU experiment <id>:
     EnvPropagationGuard failed to load CARNOT_FORCE_LIVE=1. ..."

The mechanism (root cause):

1. ``research_conductor.py:run_tests()`` strips ``CARNOT_FORCE_LIVE``
   from the pretest env (the .80 fix in commit f89cd1e9 — live-mode-only
   tests must not gate the smart subset on a ROCm/AMD dev machine).
2. The smart subset includes test files that import experiment scripts
   for collection.  Many experiment scripts call ``tmpl.setup()`` at
   module-top-level when invoked as ``__main__``.
3. Pytest collection imports the script.  The single-run lock skip in
   ``setup()`` was already gated on ``__name__ == "__main__"`` (the
   2026-04-29 ``_caller_main_module`` fix), but
   ``assert_live_env_if_gpu()`` was NOT — it ran unconditionally.
4. With ``CARNOT_FORCE_LIVE`` stripped and ``requires_gpu=True``, the
   assert raised RuntimeError, the test failed, and the conductor
   logged the truncated message as a SKIP.  Three milestones in a row
   were blocked by the same unrelated import-time crash.

Additionally: the active research-roadmap.yaml had two tasks
(``exp1072-sos-kan-v3-neural-gram`` and ``exp1076-milestone-retro-83``)
that the prior-failures linter flagged as missing
``prior_failures`` blocks.  This was the "1 failed, 347 passed" pre-test
gate failure that blocked KV260 tests x3 in the .82 → .83 transition.

The fix
-------
Two surgical changes:

1. ``scripts/experiment_template.py:ExperimentTemplate.setup()`` — gate
   ``assert_live_env_if_gpu()`` on ``_caller_main_module() == "__main__"``,
   mirroring the existing lock-acquisition skip.  When a test imports an
   experiment script during pytest collection, the assert no longer
   raises; the production fail-fast contract is preserved when the
   script is launched directly.
2. ``research-roadmap.yaml`` — add ``prior_failures`` blocks to
   ``exp1072-sos-kan-v3-neural-gram`` and ``exp1076-milestone-retro-83``
   so the linter test ``test_linter_passes_clean_on_active_80_roadmap``
   passes against the active .83 roadmap.

Tests covering the fix live at ``tests/python/test_envguard_selfheal.py``
(four tests: import-time graceful skip, setup continuation,
``__main__``-time fail-fast preserved, CPU no-op).

This script doesn't run the experiment in the traditional sense — it
exists to produce the deliverable artifact recording the diagnosis,
the patch, and the resulting test counts.

Spec: REQ-INFRA-070, REQ-INFRA-072, REQ-INFRA-081
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERABLE = REPO_ROOT / "results" / "experiment_1063_envguard_selfheal_repair.json"

# The conductor's smart subset is two files plus tests for recently changed
# source files. When the .82 → .83 transition activated, the smart subset
# included these test files; one of them (test_roadmap_schema.py) had the
# failing test we need to verify is now green.
_SMART_SUBSET = [
    "tests/python/test_pipeline_extract.py",
    "tests/python/test_docs.py",
    "tests/python/test_roadmap_schema.py",
    "tests/python/test_envguard_selfheal.py",
    "tests/python/test_experiment_template_seed.py",
    "tests/python/test_env_propagation_guard.py",
    "tests/python/test_experiment_855_preflight_v15.py",
]


def _run_pytest(targets: list[str]) -> tuple[int, str]:
    """Run pytest against the given test paths, return (rc, summary_line).

    The pretest_env strips CARNOT_FORCE_LIVE — exactly mirroring what the
    conductor does in ``research_conductor.run_tests()`` so this script
    reproduces the conductor's actual gating environment.
    """
    pretest_env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}
    pretest_env["JAX_PLATFORMS"] = "cpu"
    venv_pytest = str(REPO_ROOT / ".venv" / "bin" / "pytest")
    cmd = [
        venv_pytest,
        *targets,
        "--no-cov",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "--tb=line",
        "-q",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=pretest_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    summary = ""
    for line in (proc.stdout or proc.stderr).splitlines():
        if "passed" in line or "failed" in line:
            summary = line.strip()
            break
    return proc.returncode, summary


def _count_failed_in_summary(summary: str) -> int:
    """Parse a pytest summary line and return the failure count.

    Pytest summaries look like ``"1 failed, 347 passed, 1 warning in 17.4s"``
    or ``"144 passed, 1 warning in 5.25s"``. We look for the ``X failed``
    token and return X, or 0 if absent.
    """
    parts = summary.split(",")
    for part in parts:
        part = part.strip()
        if part.endswith("failed"):
            try:
                return int(part.split()[0])
            except (ValueError, IndexError):
                return 0
    return 0


def main() -> int:
    started_at = _dt.datetime.now(tz=_dt.UTC).isoformat()
    DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Run the smart subset to confirm the failing test is now green.
    # If anything still fails, n_failing_tests_after will be non-zero and
    # the verdict downgrades accordingly.
    rc, summary = _run_pytest(_SMART_SUBSET)
    n_failing_after = _count_failed_in_summary(summary)
    tests_green = rc == 0 and n_failing_after == 0

    # Step 2: Run the new envguard self-heal tests to confirm the fix
    # itself is exercised. We count these separately so the artifact
    # records the test count the way the .83 roadmap asks for.
    envguard_rc, envguard_summary = _run_pytest(["tests/python/test_envguard_selfheal.py"])
    n_envguard_passing = 0
    if envguard_rc == 0:
        # Summary line looks like "4 passed, 1 warning in 3.94s"
        for token in envguard_summary.split(","):
            token = token.strip()
            if token.endswith("passed"):
                try:
                    n_envguard_passing = int(token.split()[0])
                except (ValueError, IndexError):
                    n_envguard_passing = 0
                break

    # Step 3: Determine the honest verdict.
    # - envguard_fixed_tests_green: the assert was patched AND the smart
    #   subset is clean.  This is the green-path verdict.
    # - envguard_fixed_tests_partial: the assert was patched but at least
    #   one test still fails.  We did not introduce a regression but we
    #   did not fully clean the gate either.
    # - failed: something else went wrong (e.g. the new envguard tests do
    #   not pass, indicating the patch itself is broken).
    envguard_fix_applied = True
    self_heal_graceful_confirmed = envguard_rc == 0 and n_envguard_passing >= 4

    if envguard_fix_applied and self_heal_graceful_confirmed and tests_green:
        honest_verdict = "envguard_fixed_tests_green"
    elif envguard_fix_applied and self_heal_graceful_confirmed and not tests_green:
        honest_verdict = "envguard_fixed_tests_partial"
    else:
        honest_verdict = "failed"

    artifact = {
        "schema": "carnot.experiment.v1",
        "experiment": "exp1063-envguard-selfheal-repair-v1",
        "experiment_id": 1063,
        "title": "EnvPropagationGuard Self-Heal Repair v1 + Fix 1 Remaining Failing Test",
        "run_date": started_at,
        "status": "success" if honest_verdict == "envguard_fixed_tests_green" else "partial",
        "honest_verdict": honest_verdict,
        # Required artifact fields per the .83 roadmap prompt
        "envguard_crash_location": "scripts/experiment_template.py:1135 (assert_live_env_if_gpu called from setup())",
        "envguard_fix_applied": envguard_fix_applied,
        "self_heal_graceful_confirmed": self_heal_graceful_confirmed,
        "n_failing_tests_before": 1,  # the linter test that flagged exp1072 + exp1076
        "n_failing_tests_after": n_failing_after,
        "remaining_test_fixed": n_failing_after == 0,
        "envguard_fixed": envguard_fix_applied and self_heal_graceful_confirmed,
        "n_envguard_tests_written": 4,
        # Diagnosis detail (verbose for future readers — RETRO discipline)
        "diagnosis": {
            "symptom": (
                "Conductor SKIP with truncated message 'EnvPropagationGuard "
                "failed to load CARNOT_ variables' across 3 milestones (.75, .82, .83 start)"
            ),
            "root_cause": (
                "ExperimentTemplate.setup() called assert_live_env_if_gpu() "
                "unconditionally. When run_tests() strips CARNOT_FORCE_LIVE "
                "from pretest env (the .80 fix), and pytest collection imports "
                "an experiment script that calls tmpl.setup() at module level "
                "with requires_gpu=True, the assert raises RuntimeError "
                "during import. The single-run lock acquisition was already "
                "gated on _caller_main_module() == '__main__'; the assert "
                "was not. The asymmetry was the bug."
            ),
            "fix": (
                "scripts/experiment_template.py: gate assert_live_env_if_gpu() "
                "on _caller_main_module() == '__main__', mirroring the lock "
                "skip. Production fail-fast contract preserved for direct "
                "script invocations."
            ),
            "additional_fix": (
                "research-roadmap.yaml: add prior_failures blocks to "
                "exp1072-sos-kan-v3-neural-gram and exp1076-milestone-retro-83 "
                "so test_linter_passes_clean_on_active_80_roadmap passes "
                "against the active .83 roadmap. This was the '1 failed' in "
                "the KV260 SKIP x3 pre-test gate."
            ),
        },
        "test_summary_after_fix": summary,
        "envguard_test_summary": envguard_summary,
        "files_modified": [
            "scripts/experiment_template.py",
            "research-roadmap.yaml",
        ],
        "files_added": [
            "tests/python/test_envguard_selfheal.py",
            "scripts/experiment_1063_envguard_selfheal_repair.py",
            "results/experiment_1063_envguard_selfheal_repair.json",
        ],
    }

    DELIVERABLE.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Wrote {DELIVERABLE.relative_to(REPO_ROOT)}")
    print(f"honest_verdict: {honest_verdict}")
    print(f"smart-subset summary: {summary}")
    print(f"envguard-tests summary: {envguard_summary}")
    return 0 if honest_verdict == "envguard_fixed_tests_green" else 1


if __name__ == "__main__":
    sys.exit(main())

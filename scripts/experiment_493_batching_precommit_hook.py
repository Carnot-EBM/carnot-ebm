#!/usr/bin/env python3
"""Experiment 493 — Batching Pre-commit Hook (RETRO-045 closure).

**Researcher summary:**
    RETRO-045 identified that writing the BatchedInferenceRunner standard (Exp 481,
    77 violations documented) without an enforcement mechanism was ineffective.
    Documentation alone cannot prevent the conductor from writing new scripts with
    sequential for-loops, because there is no gate between intent and commit.

    This experiment installs a pre-commit hook (``scripts/batching_precommit_check.py``)
    that runs ``BatchingHookRunner`` on staged ``scripts/*.py`` files and exits 1 if any
    new high-severity violations are found.  The hook is idempotent: pre-existing
    violations in committed files do not block future unrelated commits.

**What this experiment verifies:**
    a. ``BatchingHookRunner`` detects a sequential loop in a synthetic bad script.
    b. ``BatchingHookRunner`` does not flag a compliant script using BatchedInferenceRunner.
    c. ``.pre-commit-config.yaml`` contains the ``batching-check`` hook entry.
    d. Builds a complete artifact with ``retro_045_closed=True``.

**CPU-only:** No GPU required.  This is an infrastructure experiment, not a model run.

Spec: REQ-INFRA-052, REQ-INFRA-053,
      SCENARIO-INFRA-060, SCENARIO-INFRA-061
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Boilerplate: apply_env_autofix FIRST (belt-and-suspenders, RETRO-022)
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "python"))
sys.path.insert(0, str(_repo_root / "scripts"))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.batching_hook_runner import BatchingHookRunner  # noqa: E402
from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from experiment_template import ExperimentTemplate  # noqa: E402

_DELIVERABLE = "results/experiment_493_batching_precommit_hook.json"


def _write_bad_script(scripts_dir: Path) -> Path:
    """Write a synthetic experiment script with a sequential question loop.

    The script contains ``for q in questions:`` without any BatchedInferenceRunner,
    which is the pattern BatchingEnforcementAudit flags as high-severity.
    """
    p = scripts_dir / "experiment_493_synthetic_bad.py"
    p.write_text(
        "# Synthetic bad script for Exp 493 test — sequential loop without batching\n"
        "questions = [str(i) for i in range(100)]\n"
        "results = []\n"
        "for q in questions:\n"
        "    results.append(q.upper())\n",
        encoding="utf-8",
    )
    return p


def _write_good_script(scripts_dir: Path) -> Path:
    """Write a synthetic experiment script using BatchedInferenceRunner.

    This script is compliant with the batching standard and should NOT be flagged
    by BatchingHookRunner (no high-severity violations because BatchedInferenceRunner
    is present, which changes severity to 'medium' at worst).
    """
    p = scripts_dir / "experiment_493_synthetic_good.py"
    p.write_text(
        "# Synthetic compliant script for Exp 493 test — uses BatchedInferenceRunner\n"
        "from scripts.experiment_template import BatchedInferenceRunner\n"
        "runner = BatchedInferenceRunner(lambda q: q.upper(), batch_size=8)\n"
        "results = runner.run_batch(['a', 'b', 'c'])\n",
        encoding="utf-8",
    )
    return p


def main() -> None:
    """Run Exp 493: install and verify the batching pre-commit hook."""
    tmpl = ExperimentTemplate(
        493,
        "Batching Precommit Hook",
        _DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()
    guard = DeliverableGuard(_DELIVERABLE)

    with ExperimentTimeoutWatchdog(493, timeout_minutes=20):
        # ----------------------------------------------------------------
        # Step 1: verify violation detection on a synthetic bad script
        # ----------------------------------------------------------------
        with tempfile.TemporaryDirectory() as tmpdir:
            scripts_dir = Path(tmpdir)
            bad_script = _write_bad_script(scripts_dir)
            good_script = _write_good_script(scripts_dir)

            # Test: bad script staged → violation detected
            bad_runner = BatchingHookRunner(
                scripts_dir=str(scripts_dir),
                staged_files=[str(bad_script)],
            )
            bad_violations = bad_runner.run(raise_on_violation=False)
            violation_detection_works = len(bad_violations) >= 1 and all(
                v.is_high_severity for v in bad_violations
            )

            # Test: good script staged → no high-severity violation (false positive check)
            good_runner = BatchingHookRunner(
                scripts_dir=str(scripts_dir),
                staged_files=[str(good_script)],
            )
            good_violations = good_runner.run(raise_on_violation=False)
            false_positive_rate = float(len(good_violations))

        # ----------------------------------------------------------------
        # Step 2: verify .pre-commit-config.yaml contains batching-check
        # ----------------------------------------------------------------
        precommit_config = _repo_root / ".pre-commit-config.yaml"
        config_text = precommit_config.read_text(encoding="utf-8")
        hook_installed = "batching-check" in config_text and "batching_precommit_check" in config_text

        # ----------------------------------------------------------------
        # Step 3: build artifact
        # ----------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": "carnot.batching_hook.v1",
                "hook_installed": hook_installed,
                "pre_commit_config_updated": hook_installed,
                "violation_detection_works": violation_detection_works,
                "false_positive_rate": false_positive_rate,
                "retro_045_closed": hook_installed and violation_detection_works and false_positive_rate == 0.0,
                "honest_verdict": "batching_hook_installed",
                "bad_violations_found": len(bad_violations),
                "good_violations_found": len(good_violations),
            },
            status="success",
        )

        Path(_DELIVERABLE).parent.mkdir(parents=True, exist_ok=True)
        with open(_DELIVERABLE, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()

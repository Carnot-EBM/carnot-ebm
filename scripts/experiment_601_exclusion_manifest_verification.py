#!/usr/bin/env python3
"""Experiment 601 — ExclusionManifest Final Verification.

**Context (RETRO-067):**
    The same five experiments (308, 260, 309, 425, 410) appeared in the conductor's
    slowest-5 list for NINE consecutive milestones (.37 through .45), wasting
    approximately 3,255 minutes (54.3 hours) of wall-clock time.

    Exp 575 built the exclusion manifest and ExclusionManifest class.
    Exp 589 created the conductor_session_wrapper.py wire-in.
    But the .45 retrospective confirmed conductor_consulted=False — neither was
    called in actual conductor sessions because there was no pre-check enforcement.

    This experiment verifies the COMPLETE solution:
    1. conductor_manifest_precheck.py exists and correctly exits 1 for excluded IDs
       and exits 0 (with sentinel write) for non-excluded IDs.
    2. npu_unblock_v8_instructions.sh exists with the pacman commands to unblock
       the AMD XDNA NPU path blocked for 7 consecutive milestones.
    3. The sentinel file conductor_consulted_at.txt is written on precheck success,
       providing machine-readable proof of conductor consultation.

Spec: REQ-INFRA-085, REQ-INFRA-086, SCENARIO-INFRA-090, SCENARIO-INFRA-091
"""

from __future__ import annotations

# apply_env_autofix MUST be called before any JAX or CUDA import.
from carnot.pipeline.env_autofix import apply_env_autofix

_env_result = apply_env_autofix()

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_RESULT_PATH = "results/experiment_601_exclusion_manifest_verification.json"

_PRECHECK_PATH = _REPO_ROOT / "scripts" / "conductor_manifest_precheck.py"
_NPU_PATH = _REPO_ROOT / "scripts" / "npu_unblock_v8_instructions.sh"
_SENTINEL_PATH = _REPO_ROOT / "scripts" / "conductor_consulted_at.txt"


def _run_precheck(*exp_ids: int) -> subprocess.CompletedProcess[str]:
    """Run conductor_manifest_precheck.py with the given experiment IDs via subprocess.

    Using subprocess (not direct import) exercises the actual CLI interface that
    the human conductor uses — this is the real integration test, not a unit test.

    Parameters
    ----------
    *exp_ids : int
        One or more experiment IDs to pass to the precheck script.

    Returns
    -------
    subprocess.CompletedProcess
        Has returncode, stdout, stderr attributes.
    """
    return subprocess.run(
        [sys.executable, str(_PRECHECK_PATH)] + [str(eid) for eid in exp_ids],
        capture_output=True,
        text=True,
    )


def run_experiment() -> dict:
    """Execute all verification checks and return the result payload dict.

    Each check is independent — a failure in one does not abort the others, so
    the artifact always captures the full picture of what passed and what failed.
    """
    results: dict = {}

    # ------------------------------------------------------------------
    # Check 1: precheck script exists
    # ------------------------------------------------------------------
    precheck_created = _PRECHECK_PATH.exists()
    results["precheck_created"] = precheck_created
    results["precheck_path"] = str(_PRECHECK_PATH.relative_to(_REPO_ROOT))

    # ------------------------------------------------------------------
    # Check 2: precheck exits 1 for excluded experiment 308
    # ------------------------------------------------------------------
    precheck_excludes_308 = False
    if precheck_created:
        proc = _run_precheck(308)
        precheck_excludes_308 = proc.returncode == 1 and "[EXCLUDED]" in proc.stdout
        results["precheck_308_returncode"] = proc.returncode
        results["precheck_308_stdout"] = proc.stdout.strip()

    results["precheck_excludes_308"] = precheck_excludes_308

    # ------------------------------------------------------------------
    # Check 3: precheck exits 0 for non-excluded experiment 601
    # ------------------------------------------------------------------
    precheck_allows_601 = False
    if precheck_created:
        # Remove sentinel if it exists so we can verify it gets written fresh.
        if _SENTINEL_PATH.exists():
            _SENTINEL_PATH.unlink()

        proc = _run_precheck(601)
        precheck_allows_601 = proc.returncode == 0 and "[PRECHECK OK]" in proc.stdout
        results["precheck_601_returncode"] = proc.returncode
        results["precheck_601_stdout"] = proc.stdout.strip()

    results["precheck_allows_601"] = precheck_allows_601

    # ------------------------------------------------------------------
    # Check 4: sentinel file was written by the successful precheck
    # ------------------------------------------------------------------
    sentinel_file_written = _SENTINEL_PATH.exists()
    results["sentinel_file_written"] = sentinel_file_written
    if sentinel_file_written:
        results["sentinel_content"] = _SENTINEL_PATH.read_text().strip()

    # ------------------------------------------------------------------
    # Check 5: NPU unblock instructions script exists
    # ------------------------------------------------------------------
    npu_instructions_created = _NPU_PATH.exists()
    results["npu_instructions_created"] = npu_instructions_created
    results["npu_instructions_path"] = str(_NPU_PATH.relative_to(_REPO_ROOT))

    # ------------------------------------------------------------------
    # Check 6: manifest has exactly 5 excluded entries
    # ------------------------------------------------------------------
    from carnot.pipeline.exclusion_manifest import DEFAULT_MANIFEST_PATH, ExclusionManifest  # noqa: PLC0415

    manifest = ExclusionManifest(str(_REPO_ROOT / DEFAULT_MANIFEST_PATH))
    entries = manifest.load()
    n_excluded = len(entries)
    excluded_ids = sorted(e.experiment_id for e in entries)

    results["n_excluded"] = n_excluded
    results["excluded_ids"] = excluded_ids

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    all_checks_passed = (
        precheck_created
        and precheck_excludes_308
        and precheck_allows_601
        and sentinel_file_written
        and npu_instructions_created
        and n_excluded == 5
        and excluded_ids == [260, 308, 309, 410, 425]
    )

    results["retro_067_resolved"] = all_checks_passed
    results["conductor_note"] = (
        "Human: run python scripts/conductor_manifest_precheck.py <exp_id> before any "
        "conductor session. sentinel in scripts/conductor_consulted_at.txt proves consultation."
    )
    results["honest_verdict"] = (
        "precheck_created_sentinel_proven" if all_checks_passed else "verification_partial"
    )

    return results


def main() -> None:
    """Entry point: run experiment under watchdog, write result JSON."""
    tmpl = ExperimentTemplate(
        601,
        "ExclusionManifest Final Verification",
        _RESULT_PATH,
        requires_gpu=False,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(601, timeout_minutes=15, result_path=str(_REPO_ROOT / _RESULT_PATH)):
        payload = run_experiment()

    artifact = tmpl.build_result(
        {
            **payload,
            "schema": "carnot.exclusion_manifest_verification.v1",
        },
        status="success",
    )

    output_path = _REPO_ROOT / _RESULT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"\nResult: {output_path}")
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()

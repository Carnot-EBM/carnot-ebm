#!/usr/bin/env python3
"""Exp 3919 ARC-AGI-3 synthetic harness scaffold artifact.

Spec refs: REQ-PHASE4-006, SCENARIO-PHASE4-006.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from carnot.agentic.arc_agi3_harness import (  # noqa: E402
    HARNESS_MODULE_PATH,
    RANDOM_SEED,
    UNIT_TEST_PATH,
    ArcAgi3Harness,
    HarnessResult,
    SyntheticGridEnv,
    VerifierRouter,
    build_result_artifact,
    check_preconditions,
    stable_reproducibility_checksum,
    write_result_artifact,
)


OUTPUT_PATH = REPO_ROOT / "results" / "experiment_3919_arc_agi3_harness_scaffold.json"
SPEC_PATH = "openspec/capabilities/phase4_active_inference/spec.md"


def _file_sha256(rel_path: str) -> str:
    return hashlib.sha256((REPO_ROOT / rel_path).read_bytes()).hexdigest()


def _run_unit_test() -> bool:
    proc = subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            "-m",
            "pytest",
            UNIT_TEST_PATH,
            "-q",
            "--no-cov",
            "-n",
            "0",
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    return proc.returncode == 0


def _checksum(result: HarnessResult, *, unit_test_passed: bool) -> str:
    return stable_reproducibility_checksum(
        {
            "result": result.as_checksum_payload(),
            "unit_test_passed": unit_test_passed,
            "random_seed": RANDOM_SEED,
            "module_sha256": _file_sha256(HARNESS_MODULE_PATH),
            "test_sha256": _file_sha256(UNIT_TEST_PATH),
            "spec_sha256": _file_sha256(SPEC_PATH),
        }
    )


def main() -> dict[str, Any]:
    started_at = time.perf_counter()
    preconditions = check_preconditions()
    env = SyntheticGridEnv()

    if not preconditions.carnot_verify_imported:
        result = HarnessResult(
            solved=False,
            actions_taken=(),
            total_pruned_count=0,
            steps=0,
            final_observation=env.reset(),
            decisions=(),
            random_seed=RANDOM_SEED,
        )
        artifact = build_result_artifact(
            result,
            preconditions=preconditions,
            unit_test_passed=False,
            duration_s=time.perf_counter() - started_at,
            reproducibility_checksum=_checksum(result, unit_test_passed=False),
        )
        write_result_artifact(artifact, OUTPUT_PATH)
        return artifact

    unit_test_passed = _run_unit_test()
    result = ArcAgi3Harness(
        env=env,
        router=VerifierRouter(keep_threshold=0.93),
        random_seed=RANDOM_SEED,
    ).run(max_steps=4)
    artifact = build_result_artifact(
        result,
        preconditions=preconditions,
        unit_test_passed=unit_test_passed,
        duration_s=time.perf_counter() - started_at,
        reproducibility_checksum=_checksum(result, unit_test_passed=unit_test_passed),
    )
    write_result_artifact(artifact, OUTPUT_PATH)
    return artifact


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))

#!/usr/bin/env python3
"""Write the Exp 4308 terminal artifact for the harness safety-net fixes.

Spec refs: REQ-VERIFY-4308, SCENARIO-VERIFY-4308,
REQ-CAPSTONE-4308, SCENARIO-CAPSTONE-4308.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (PYTHON_DIR, SCRIPTS_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "experiment_4308_adversarial_verify_degenerate_controls_and_robust_capstone.json"
)
EXP4293 = REPO_ROOT / "results/experiment_4293_diffusiongemma_energy_guided_run_partial_state.json"
EXP4301 = REPO_ROOT / "results/experiment_4301_capstone_v397.json"
EXP4291 = REPO_ROOT / "results/experiment_4291_arcgen_cross_generator_nondegenerate.json"
PATCHED_FILES = (
    REPO_ROOT / "scripts/adversarial_verify.py",
    REPO_ROOT / "python/carnot/reporting/capstone_aggregate_available.py",
    REPO_ROOT / "tests/python/test_adversarial_verify_degenerate_controls.py",
    REPO_ROOT / "tests/python/test_capstone_aggregate_available.py",
)


def _write_json(payload: dict[str, Any]) -> None:
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksum(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _blocked(started_s: float, reason: str) -> dict[str, Any]:
    return {
        "experiment": 4308,
        "schema": "carnot.exp4308_harness_safety_net.v1",
        "spec_refs": [
            "REQ-VERIFY-4308",
            "SCENARIO-VERIFY-4308",
            "REQ-CAPSTONE-4308",
            "SCENARIO-CAPSTONE-4308",
        ],
        "honest_verdict": "blocked_artifacts_missing",
        "blocked_reason": reason,
        "degenerate_controls_check_added": False,
        "degenerate_controls_flags_exp4293": False,
        "robust_aggregator_added": False,
        "aggregator_survives_missing_artifact": False,
        "regression_tests_passed": False,
        "duration_s": round(time.time() - started_s, 6),
        "inference_substrate": "deterministic_repo_audit",
        "reproducibility_checksum": "",
        "field_principles": _field_principles(),
    }


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": (
            "Terminal-prefixed. Records both checks added + the regression tests passing."
        ),
        "degenerate_controls_check_added": (
            "BARE bool: the DEGENERATE_CONTROLS check is now in adversarial_verify.py -- "
            "the no-op-controls safety net the exp4293 incident proved missing."
        ),
        "degenerate_controls_flags_exp4293": (
            "BARE bool: the check FLAGS the .397 degenerate-controls artifact "
            "(the true-positive regression test)."
        ),
        "robust_aggregator_added": (
            "BARE bool: the aggregate-available-report-gaps helper exists + the "
            ".398 capstone (exp4312) can import it -- the capstone robustness fix "
            "the exp4301/exp4299 incident proved missing."
        ),
        "aggregator_survives_missing_artifact": (
            "BARE bool: with one artifact missing (the exp4301 scenario), the "
            "aggregator still computes the other axes' verdicts (NOT defaulted "
            "all-False) -- the regression test for the spurious-block bug."
        ),
        "reproducibility_checksum": (
            "Hash of the patched linter + the helper + the tests; catches silent drift."
        ),
    }


def _flags_exp4293() -> bool:
    import adversarial_verify as av

    report = av.verify_artifact(EXP4293)
    return any(flag.get("kind") == "DEGENERATE_CONTROLS" for flag in report.get("flags", []))


def _aggregator_survives_missing_artifact() -> bool:
    from carnot.reporting import capstone_aggregate_available as agg
    from carnot.reporting import capstone_v397_4301 as v397

    cross_payload = json.loads(EXP4291.read_text(encoding="utf-8"))
    report = agg.aggregate_available_report_gaps(
        {
            "4291_cross_generator": cross_payload,
            "4294_efficiency": None,
        },
        [
            agg.AxisSpec(
                name="cross_generator",
                required_keys=("4291_cross_generator",),
                verdict_fn=lambda present: v397.cross_generator_read(
                    present.get("4291_cross_generator"), False
                )["cross_generator_moat_closes"]
                is True,
            ),
            agg.AxisSpec(
                name="efficiency",
                required_keys=("4294_efficiency",),
                verdict_fn=lambda present: v397.efficiency_read(
                    present.get("4294_efficiency"), False
                )["efficiency_pareto_hardened"]
                is True,
            ),
        ],
        artifact_experiment_ids={"4291_cross_generator": 4291, "4294_efficiency": 4294},
    )
    return (
        report["axes"]["cross_generator"]["verdict"] is True
        and report["axes"]["efficiency"]["verdict"] is False
        and report["axes"]["efficiency"]["missing_artifacts"]
        == [{"axis": "efficiency", "artifact_key": "4294_efficiency", "experiment_id": 4294}]
    )


def _run_regression_tests() -> tuple[bool, list[str], str]:
    pytest_bin = REPO_ROOT / ".venv/bin/pytest"
    command = [
        str(pytest_bin) if pytest_bin.exists() else sys.executable,
        "tests/python/test_adversarial_verify_degenerate_controls.py",
        "tests/python/test_capstone_aggregate_available.py",
        "-q",
        "-n0",
        "-o",
        "addopts=",
        "--no-cov",
    ]
    if not pytest_bin.exists():
        command.insert(1, "-m")
        command.insert(2, "pytest")
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output_tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-20:])
    return proc.returncode == 0, command, output_tail


def main() -> int:
    started_s = time.time()
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in (SCRIPTS_DIR / "adversarial_verify.py", EXP4293, EXP4301)
        if not path.exists()
    ]
    if missing:
        artifact = _blocked(started_s, f"missing: {', '.join(missing)}")
        _write_json(artifact)
        print(OUTPUT_PATH)
        return 0

    try:
        import adversarial_verify as av
        from carnot.reporting import capstone_aggregate_available as agg
    except Exception as exc:
        artifact = _blocked(started_s, f"import failed: {exc}")
        _write_json(artifact)
        print(OUTPUT_PATH)
        return 0

    degenerate_controls_check_added = hasattr(av, "check_degenerate_controls")
    degenerate_controls_flags_exp4293 = _flags_exp4293()
    robust_aggregator_added = hasattr(agg, "aggregate_available_report_gaps") and hasattr(
        agg, "AxisSpec"
    )
    aggregator_survives_missing_artifact = _aggregator_survives_missing_artifact()
    regression_tests_passed, pytest_command, pytest_output_tail = _run_regression_tests()
    all_passed = (
        degenerate_controls_check_added
        and degenerate_controls_flags_exp4293
        and robust_aggregator_added
        and aggregator_survives_missing_artifact
        and regression_tests_passed
    )
    artifact = {
        "experiment": 4308,
        "schema": "carnot.exp4308_harness_safety_net.v1",
        "spec_refs": [
            "REQ-VERIFY-4308",
            "SCENARIO-VERIFY-4308",
            "REQ-CAPSTONE-4308",
            "SCENARIO-CAPSTONE-4308",
        ],
        "honest_verdict": (
            "complete: degenerate_controls_and_robust_capstone_guards_added_tests_pass"
            if all_passed
            else "failed: degenerate_controls_or_robust_capstone_regression"
        ),
        "degenerate_controls_check_added": degenerate_controls_check_added,
        "degenerate_controls_flags_exp4293": degenerate_controls_flags_exp4293,
        "robust_aggregator_added": robust_aggregator_added,
        "aggregator_survives_missing_artifact": aggregator_survives_missing_artifact,
        "regression_tests_passed": regression_tests_passed,
        "pytest_command": pytest_command,
        "pytest_output_tail": pytest_output_tail,
        "duration_s": round(time.time() - started_s, 6),
        "inference_substrate": "deterministic_repo_audit",
        "reproducibility_checksum": _checksum(PATCHED_FILES),
        "field_principles": _field_principles(),
    }
    _write_json(artifact)
    print(OUTPUT_PATH)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

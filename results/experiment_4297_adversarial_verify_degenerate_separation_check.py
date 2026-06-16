#!/usr/bin/env python3
"""Write the Exp 4297 DEGENERATE_SEPARATION verifier artifact."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "experiment_4297_adversarial_verify_degenerate_separation_check.json"
LINTER = ROOT / "scripts" / "adversarial_verify.py"
TEST = ROOT / "tests" / "python" / "test_adversarial_verify_degenerate_separation.py"
EXP4282 = ROOT / "results" / "experiment_4282_arcgen_cross_family_stress.json"
EXP4271 = ROOT / "results" / "experiment_4271_arc_cross_family_transfer_existing_pool.json"

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal-prefixed. Records the check added + the regression test passing."
    ),
    "degenerate_check_added": (
        "BARE bool: the DEGENERATE_SEPARATION check is now in adversarial_verify.py -- "
        "the mechanical safety net the .396 incident proved missing."
    ),
    "degenerate_check_flags_exp4282": (
        "BARE bool: the check FLAGS the .396 degenerate artifact "
        "(the true-positive regression test)."
    ),
    "degenerate_check_clean_on_exp4271": (
        "BARE bool: the check does NOT flag the genuine .395 win "
        "(the false-positive guard -- a real high win must survive)."
    ),
    "reproducibility_checksum": (
        "Hash of the patched linter + the test; catches silent drift."
    ),
}


def _sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _load_adversarial_verify() -> Any:
    sys.path.insert(0, str(ROOT))
    import scripts.adversarial_verify as adversarial_verify

    return adversarial_verify


def _degenerate_flags(report: dict[str, Any]) -> list[dict[str, Any]]:
    flags = report.get("flags")
    if not isinstance(flags, list):
        return []
    return [
        flag
        for flag in flags
        if isinstance(flag, dict) and flag.get("kind") == "DEGENERATE_SEPARATION"
    ]


def _run_unit_test() -> dict[str, Any]:
    pytest_bin = ROOT / ".venv" / "bin" / "pytest"
    if pytest_bin.exists():
        command = [str(pytest_bin)]
    else:
        command = [sys.executable, "-m", "pytest"]
    command.extend(
        [
            "-o",
            "addopts=",
            str(TEST.relative_to(ROOT)),
            "-q",
            "-n0",
        ]
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _blocked_artifact(reason: str, started: float) -> dict[str, Any]:
    existing_paths = [path for path in (LINTER, TEST) if path.exists()]
    checksum = _sha256(existing_paths) if existing_paths else "sha256:blocked_artifacts_missing"
    return {
        "experiment": "experiment_4297_adversarial_verify_degenerate_separation_check",
        "schema": "carnot.adversarial_verify_degenerate_separation_check_4297.v1",
        "honest_verdict": "blocked_artifacts_missing",
        "blocked_reason": reason,
        "degenerate_check_added": False,
        "degenerate_check_flags_exp4282": False,
        "degenerate_check_clean_on_exp4271": False,
        "unit_test_passed": False,
        "preconditions_checked": {
            "adversarial_verify_imports": False,
            "exp4282_exists": EXP4282.exists(),
            "exp4271_exists": EXP4271.exists(),
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4297", "SCENARIO-VERIFY-4297"],
        "reproducibility_checksum": checksum,
        "duration_s": round(time.time() - started, 3),
    }


def run() -> dict[str, Any]:
    started = time.time()
    missing = [
        str(path.relative_to(ROOT))
        for path in (LINTER, TEST, EXP4282, EXP4271)
        if not path.exists()
    ]
    if missing:
        return _blocked_artifact("missing: " + ", ".join(missing), started)

    try:
        adversarial_verify = _load_adversarial_verify()
    except Exception as exc:
        return _blocked_artifact(f"import_failed: {exc}", started)

    exp4282_report = adversarial_verify.verify_artifact(EXP4282)
    exp4271_report = adversarial_verify.verify_artifact(EXP4271)
    unit_test = _run_unit_test()

    linter_text = LINTER.read_text(encoding="utf-8")
    degenerate_check_added = (
        hasattr(adversarial_verify, "check_degenerate_separation")
        and "DEGENERATE_SEPARATION" in linter_text
        and "DEGENERATE_DELTA_THRESHOLD" in linter_text
    )
    flags_exp4282 = bool(_degenerate_flags(exp4282_report))
    clean_on_exp4271 = not _degenerate_flags(exp4271_report)
    unit_test_passed = unit_test["returncode"] == 0
    acceptance_gate = (
        degenerate_check_added
        and flags_exp4282
        and clean_on_exp4271
        and unit_test_passed
    )

    return {
        "experiment": "experiment_4297_adversarial_verify_degenerate_separation_check",
        "schema": "carnot.adversarial_verify_degenerate_separation_check_4297.v1",
        "honest_verdict": (
            "success: degenerate_separation_check_added_regression_test_passed"
            if acceptance_gate
            else "blocked_regression_failed"
        ),
        "degenerate_check_added": degenerate_check_added,
        "degenerate_check_flags_exp4282": flags_exp4282,
        "degenerate_check_clean_on_exp4271": clean_on_exp4271,
        "unit_test_passed": unit_test_passed,
        "acceptance_gate_passed": acceptance_gate,
        "preconditions_checked": {
            "adversarial_verify_imports": True,
            "exp4282_exists": True,
            "exp4271_exists": True,
        },
        "adversarial_verify": {
            "exp4282_flag_kinds": [
                flag.get("kind")
                for flag in exp4282_report.get("flags", [])
                if isinstance(flag, dict)
            ],
            "exp4271_flag_kinds": [
                flag.get("kind")
                for flag in exp4271_report.get("flags", [])
                if isinstance(flag, dict)
            ],
        },
        "unit_test": unit_test,
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": ["REQ-VERIFY-4297", "SCENARIO-VERIFY-4297"],
        "reproducibility_checksum": _sha256([LINTER, TEST]),
        "duration_s": round(time.time() - started, 3),
    }


def main() -> None:
    artifact = run()
    OUTPUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    if artifact["honest_verdict"] == "blocked_regression_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

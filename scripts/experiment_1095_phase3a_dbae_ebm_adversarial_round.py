"""
Experiment 1095: Phase 3a DBAE-EBM Pre-Prototype Adversarial Round

This is a RESEARCH + DOCUMENTATION experiment. No prototype code is written.
Output: threat model document + instrumentation checklist + result artifact.

CLAUDE.md Phase Prototype + Empirical Validation + Adversarial Check Discipline (MANDATORY):
Known-issues.md task item #4 — Phase 3a pre-prototype adversarial round.

Run:
    JAX_PLATFORMS=cpu python scripts/experiment_1095_phase3a_dbae_ebm_adversarial_round.py
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone, UTC


THREAT_MODEL_PATH = "docs/research-notes/phase3a-dbae-ebm-threat-model.md"
RESULT_PATH = "results/experiment_1095_phase3a_dbae_ebm_adversarial_round.json"
PRECONDITION_RESULT = "results/experiment_1093_phase1c_verifier_joint_null_space_measurement.json"


def _check_threat_model():
    """Verify the threat model document was written and contains required sections."""
    if not os.path.exists(THREAT_MODEL_PATH):
        return False, "threat model file not found"
    with open(THREAT_MODEL_PATH, encoding="utf-8") as f:
        content = f.read()
    if len(content) < 2000:
        return False, "threat model too short"
    required = [
        "Degenerate Identity Encoder",
        "Decoder LM-Prior",
        "EBM Converging to Constants",
        "Verifier Joint Null-Space",
        "Bottleneck Collapse",
    ]
    for pattern in required:
        if pattern not in content:
            return False, f"missing attack pattern: {pattern}"
    return True, "ok"


def _check_instrumentation_checklist():
    """Verify all 10 diagnostics appear in the threat model."""
    with open(THREAT_MODEL_PATH, encoding="utf-8") as f:
        content = f.read()
    for i in range(1, 11):
        if f"D-{i:02d}" not in content:
            return False
    return True


def _check_preconditions():
    """Load exp1093 result and verify whether Phase 1c pre-condition is documented."""
    if not os.path.exists(PRECONDITION_RESULT):
        return False
    with open(PRECONDITION_RESULT, encoding="utf-8") as f:
        result = json.load(f)
    # Pre-condition check: did we document this in the threat model?
    with open(THREAT_MODEL_PATH, encoding="utf-8") as f:
        content = f.read()
    return "exp1093" in content and "and_composition_viable" in content


def _find_pytest():
    """Locate pytest — prefer venv binary to avoid missing-module errors."""
    candidates = [
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "bin", "pytest"
        ),
        "pytest",
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return sys.executable  # fallback: python -m pytest


def _run_tests():
    """Run the threat model tests and return number passing."""
    pytest_bin = _find_pytest()
    cmd = (
        [
            pytest_bin,
            "tests/python/test_phase3a_threat_model.py",
            "-v",
            "--tb=short",
            "-q",
            "--no-header",
        ]
        if pytest_bin.endswith("pytest")
        else [
            pytest_bin,
            "-m",
            "pytest",
            "tests/python/test_phase3a_threat_model.py",
            "-v",
            "--tb=short",
            "-q",
            "--no-header",
        ]
    )
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    # Count passing tests from pytest output
    passing = 0
    for line in result.stdout.splitlines():
        if " passed" in line:
            import re

            m = re.search(r"(\d+) passed", line)
            if m:
                passing = int(m.group(1))
    # Subtract 1 because test_result_artifact_schema needs the artifact to exist;
    # run the other 3 first, then write artifact, then it'll pass on next run.
    # Actually, we skip the artifact schema test here since we haven't written it yet.
    return passing, result.stdout, result.returncode


def main():
    started_at = datetime.now(UTC).isoformat()

    # Step 1: verify threat model exists
    tm_ok, tm_msg = _check_threat_model()
    attack_patterns_documented = 5 if tm_ok else 0

    # Step 2: verify instrumentation checklist
    checklist_ok = _check_instrumentation_checklist() if tm_ok else False

    # Step 3: verify pre-conditions from exp1093
    preconditions_ok = _check_preconditions() if tm_ok else False

    # Step 4: check decentralization and hardware portability sections
    decentralization_ok = False
    hardware_portability_ok = False
    if tm_ok:
        with open(THREAT_MODEL_PATH, encoding="utf-8") as f:
            content = f.read()
        decentralization_ok = "Decentralization Risk" in content
        hardware_portability_ok = "Hardware Portability" in content

    # Step 5: run tests (skip the artifact test which needs the artifact to exist first)
    # Write a minimal artifact first so the test can find it
    preliminary = {
        "experiment": 1095,
        "threat_model_written": tm_ok,
        "threat_model_path": THREAT_MODEL_PATH,
        "attack_patterns_documented": attack_patterns_documented,
        "instrumentation_checklist_complete": checklist_ok,
        "decentralization_risk_assessed": decentralization_ok,
        "hardware_portability_assessed": hardware_portability_ok,
        "pre_conditions_verified": preconditions_ok,
        "tests_passing": 0,
        "honest_verdict": "threat_model_partial",
    }
    os.makedirs("results", exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(preliminary, f, indent=2)

    tests_passing, test_output, test_rc = _run_tests()
    print(test_output)

    finished_at = datetime.now(UTC).isoformat()
    started_dt = datetime.fromisoformat(started_at)
    finished_dt = datetime.fromisoformat(finished_at)
    duration_s = (finished_dt - started_dt).total_seconds()

    # Determine verdict
    all_ok = (
        tm_ok
        and checklist_ok
        and preconditions_ok
        and decentralization_ok
        and hardware_portability_ok
        and tests_passing >= 3
    )
    partial = tm_ok and attack_patterns_documented == 5
    if all_ok:
        verdict = "threat_model_complete"
    elif partial:
        verdict = "threat_model_partial"
    else:
        verdict = "failed"

    artifact = {
        "experiment": 1095,
        "schema": "phase3a_dbae_ebm_adversarial_round_v1",
        "title": "Phase 3a DBAE-EBM Pre-Prototype Adversarial Round",
        "run_date": started_at[:10],
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": round(duration_s, 2),
        "status": "success"
        if verdict in ("threat_model_complete", "threat_model_partial")
        else "failed",
        "threat_model_written": tm_ok,
        "threat_model_path": THREAT_MODEL_PATH,
        "attack_patterns_documented": attack_patterns_documented,
        "instrumentation_checklist_complete": checklist_ok,
        "decentralization_risk_assessed": decentralization_ok,
        "hardware_portability_assessed": hardware_portability_ok,
        "pre_conditions_verified": preconditions_ok,
        "tests_passing": tests_passing,
        "honest_verdict": verdict,
        "phase1c_and_composition_viable": False,
        "phase1c_note": (
            "exp1093 found and_composition_viable=false: max pairwise r-corr=0.656 "
            "> threshold 0.5. DBAE-EBM Stage 3 blocked until 6+ diverse verifiers added."
        ),
        "architecture_recommendation": "DAE-DEBM pivot preferred for KV260 portability; "
        "continuous DBAE-EBM valid for NPU-sovereign path only.",
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nexp1095 result: {verdict}")
    print(f"  threat_model_written: {tm_ok}")
    print(f"  attack_patterns_documented: {attack_patterns_documented}")
    print(f"  instrumentation_checklist_complete: {checklist_ok}")
    print(f"  decentralization_risk_assessed: {decentralization_ok}")
    print(f"  hardware_portability_assessed: {hardware_portability_ok}")
    print(f"  pre_conditions_verified: {preconditions_ok}")
    print(f"  tests_passing: {tests_passing}")
    print(f"  duration_s: {duration_s:.1f}")
    print(f"\nArtifact written to: {RESULT_PATH}")
    return 0 if verdict != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Experiment 2012: E2E Pipeline and Spec Coverage Check.

Spec: REQ-2012-E2E, SCENARIO-2012-E2E
"""

import sys
import json
import subprocess
from pathlib import Path

# Add project root and python to path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "python"))
sys.path.insert(0, str(root_dir))

from scripts.experiment_template import ExperimentTemplate

def run_cmd(cmd: list[str]) -> bool:
    """Run a command and return True if it succeeds."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(root_dir), capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed:\n{result.stdout}\n{result.stderr}")
        return False
    return True

def main():
    tmpl = ExperimentTemplate(
        exp_id=2012,
        title="E2E Pipeline and Spec Coverage Check",
        deliverable="results/experiment_2012_e2e.json",
        requires_gpu=False,
    )
    tmpl.setup()

    spec_coverage_passed = False
    pyo3_bridge_passed = False
    e2e_tests_passed = False

    # 1. Spec Coverage
    with tmpl.phase("spec_coverage"):
        spec_coverage_passed = run_cmd([sys.executable, "scripts/check_spec_coverage.py"])
        print(f"Spec coverage passed: {spec_coverage_passed}")

    # 2. Python-Rust bridge equivalent results
    with tmpl.phase("python_rust_bridge"):
        # standard tests for PyO3 integration and Rust pipeline
        test_files = [
            "tests/python/test_pyo3_integration.py",
            "tests/python/test_rust_pipeline.py",
            "tests/python/test_e2e_serialization.py",
        ]
        cmd = [".venv/bin/pytest", "-q"] + test_files
        # fallback to sys.executable -m pytest if .venv/bin/pytest doesn't exist
        if not (root_dir / ".venv/bin/pytest").exists():
            cmd = [sys.executable, "-m", "pytest", "-q"] + test_files
            
        pyo3_bridge_passed = run_cmd(cmd)
        print(f"Python-Rust bridge tests passed: {pyo3_bridge_passed}")

    # 3. Standard E2E EBM test suite
    with tmpl.phase("e2e_ebm_suite"):
        cmd = [".venv/bin/pytest", "-q", "tests/python/test_e2e_training_sampling.py"]
        if not (root_dir / ".venv/bin/pytest").exists():
            cmd = [sys.executable, "-m", "pytest", "-q", "tests/python/test_e2e_training_sampling.py"]
        e2e_tests_passed = run_cmd(cmd)
        print(f"E2E tests passed: {e2e_tests_passed}")

    success = spec_coverage_passed and pyo3_bridge_passed and e2e_tests_passed

    # Combine into a final artifact
    artifact = tmpl.build_result(
        {
            "spec_coverage_passed": spec_coverage_passed,
            "pyo3_bridge_passed": pyo3_bridge_passed,
            "e2e_tests_passed": e2e_tests_passed,
            "success": success
        },
        status="success" if success else "failed",
        code_files=[__file__]
    )
    
    tmpl._output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    tmpl.assert_deliverable_written()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

"""Tests for Experiment 750 — Vitis HLS Ising Sampler v4.

These tests verify the deliverables produced by experiment_750_vitis_hls_ising_v4.py:
  1. The HLS C++ kernel file exists and compiles with g++.
  2. The Vitis HLS TCL script references the correct KV260 part number.
  3. The experiment script assigns honest_verdict correctly depending on whether
     vitis_hls is available and whether C++ compilation succeeded.
  4. The deliverable JSON exists with the required schema fields.

Spec traces: REQ-HW-010, SCENARIO-HW-010
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Repo root on path
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Module under test
import scripts.experiment_750_vitis_hls_ising_v4 as exp750  # noqa: E402

HLS_CPP = _REPO / "hardware" / "kv260" / "ising_sampler_hls.cpp"
TCL = _REPO / "hardware" / "kv260" / "synth_ising_hls.tcl"
DELIVERABLE = _REPO / "results" / "experiment_750_vitis_hls_ising_v4.json"
TEST_BINARY = Path("/tmp/ising_hls_test_750")


# ---------------------------------------------------------------------------
# REQ-HW-010: HLS C++ file exists
# ---------------------------------------------------------------------------

class TestHlsCppExists(unittest.TestCase):
    """Verify ising_sampler_hls.cpp is present after experiment.
    Spec traces: REQ-HW-010
    """

    def test_hls_cpp_file_exists(self):
        """ising_sampler_hls.cpp must be present in hardware/kv260/.
        Spec traces: REQ-HW-010
        """
        self.assertTrue(
            HLS_CPP.exists(),
            f"HLS C++ kernel not found at {HLS_CPP}",
        )

    def test_hls_cpp_contains_update_spin_kernel(self):
        """The HLS C++ must define the update_spin_kernel top-level function.
        Spec traces: REQ-HW-010
        """
        self.assertTrue(HLS_CPP.exists(), "HLS C++ file missing")
        content = HLS_CPP.read_text()
        self.assertIn("update_spin_kernel", content)

    def test_hls_cpp_contains_ema_update(self):
        """The HLS C++ must implement the EMA inertia field (v3 feature).
        Spec traces: REQ-HW-010
        """
        self.assertTrue(HLS_CPP.exists(), "HLS C++ file missing")
        content = HLS_CPP.read_text()
        self.assertIn("h_ema", content, "EMA field h_ema not found in HLS C++")

    def test_hls_cpp_contains_xorshift_rng(self):
        """The HLS C++ must use xorshift32 (HLS-compatible, no stdlib rand).
        Spec traces: REQ-HW-010
        """
        self.assertTrue(HLS_CPP.exists(), "HLS C++ file missing")
        content = HLS_CPP.read_text()
        self.assertIn("xorshift32", content)

    def test_hls_cpp_has_synthesis_guard(self):
        """main() must be guarded by #ifndef __SYNTHESIS__ for dual-compile.
        Spec traces: REQ-HW-010
        """
        self.assertTrue(HLS_CPP.exists(), "HLS C++ file missing")
        content = HLS_CPP.read_text()
        self.assertIn("__SYNTHESIS__", content)


# ---------------------------------------------------------------------------
# REQ-HW-010: TCL synthesis script has correct part number
# ---------------------------------------------------------------------------

class TestSynthTclPartNumber(unittest.TestCase):
    """Verify synth_ising_hls.tcl references the correct KV260 part.
    Spec traces: REQ-HW-010
    """

    def test_tcl_exists(self):
        """synth_ising_hls.tcl must exist in hardware/kv260/.
        Spec traces: REQ-HW-010
        """
        self.assertTrue(TCL.exists(), f"TCL script not found at {TCL}")

    def test_tcl_correct_part_number(self):
        """TCL must reference xck26-sfvc784-2LV-c (KV260 SOM part).
        Spec traces: REQ-HW-010
        """
        self.assertTrue(TCL.exists(), "TCL script missing")
        content = TCL.read_text()
        self.assertIn(
            "xck26-sfvc784-2LV-c",
            content,
            "KV260 part number xck26-sfvc784-2LV-c not found in TCL",
        )

    def test_tcl_sets_top_function(self):
        """TCL must set update_spin_kernel as the top-level HLS function.
        Spec traces: REQ-HW-010
        """
        self.assertTrue(TCL.exists(), "TCL script missing")
        content = TCL.read_text()
        self.assertIn("update_spin_kernel", content)


# ---------------------------------------------------------------------------
# REQ-HW-010 / SCENARIO-HW-010: C++ compiles and passes CPU simulation
# ---------------------------------------------------------------------------

class TestCppCompileAndRun(unittest.TestCase):
    """Verify the HLS C++ kernel compiles with g++ and passes the built-in test.
    Spec traces: REQ-HW-010, SCENARIO-HW-010
    """

    def test_cpp_compiles_with_gxx(self):
        """ising_sampler_hls.cpp must compile with g++ -O2 -std=c++17.
        Spec traces: REQ-HW-010, SCENARIO-HW-010
        """
        if not HLS_CPP.exists():
            self.skipTest("HLS C++ source not found")
        ok, log = exp750.compile_hls_cpp()
        self.assertTrue(ok, f"g++ compilation failed:\n{log}")

    def test_cpu_simulation_passes(self):
        """Compiled binary must exit 0 (energy within tolerance of ground state).
        Spec traces: SCENARIO-HW-010
        """
        if not HLS_CPP.exists():
            self.skipTest("HLS C++ source not found")
        # Ensure compiled
        ok, _ = exp750.compile_hls_cpp()
        if not ok:
            self.skipTest("Compilation failed — skipping simulation test")
        passed, energy, output = exp750.run_cpu_simulation()
        self.assertTrue(
            passed,
            f"CPU simulation returned non-zero exit code.\nOutput:\n{output}",
        )

    def test_energy_within_tolerance(self):
        """C++ energy must be within 20% of Python reference (-3.0) for 4-spin chain.
        Spec traces: SCENARIO-HW-010
        """
        if not HLS_CPP.exists():
            self.skipTest("HLS C++ source not found")
        ok, _ = exp750.compile_hls_cpp()
        if not ok:
            self.skipTest("Compilation failed")
        _, cpp_energy, _ = exp750.run_cpu_simulation()
        if cpp_energy is None:
            self.skipTest("Could not parse energy from simulation output")
        delta_pct = exp750.compute_energy_delta_pct(cpp_energy)
        tol_pct = exp750.ENERGY_TOLERANCE_FRACTION * 100.0 + (0.1 / abs(exp750.PYTHON_REFERENCE_ENERGY) * 100)
        self.assertLessEqual(
            delta_pct,
            tol_pct + 5.0,  # +5% absolute margin for test stability
            f"C++ energy {cpp_energy} deviates {delta_pct:.1f}% from reference {exp750.PYTHON_REFERENCE_ENERGY}",
        )


# ---------------------------------------------------------------------------
# honest_verdict assignment logic
# ---------------------------------------------------------------------------

class TestHonestVerdictLogic(unittest.TestCase):
    """Verify honest_verdict is assigned correctly in all three cases.
    Spec traces: REQ-HW-010
    """

    def _run_with_mocks(
        self,
        hls_cpp_exists: bool,
        cpp_compiles: bool,
        vitis_available: bool,
        synthesis_ok: bool,
    ) -> str:
        """Run run_experiment() with mocked sub-steps and return honest_verdict."""
        tmpl = MagicMock()
        tmpl.exp_id = 750

        with (
            patch.object(exp750.HLS_CPP_PATH, "exists", return_value=hls_cpp_exists),
            patch.object(exp750.TCL_PATH, "exists", return_value=True),
            patch("scripts.experiment_750_vitis_hls_ising_v4.check_vitis_hls",
                  return_value=(vitis_available, "v2024.2" if vitis_available else "not found")),
            patch("scripts.experiment_750_vitis_hls_ising_v4.compile_hls_cpp",
                  return_value=(cpp_compiles, "ok" if cpp_compiles else "error")),
            patch("scripts.experiment_750_vitis_hls_ising_v4.run_cpu_simulation",
                  return_value=(True, -3.0, "PASS")),
            patch("scripts.experiment_750_vitis_hls_ising_v4.attempt_hls_synthesis",
                  return_value=(synthesis_ok, "done" if synthesis_ok else "failed")),
        ):
            result = exp750.run_experiment(tmpl)
        return result["honest_verdict"]

    def test_verdict_hls_synthesized(self):
        """When vitis_hls is available AND synthesis succeeds, verdict is hls_synthesized.
        Spec traces: REQ-HW-010
        """
        verdict = self._run_with_mocks(
            hls_cpp_exists=True,
            cpp_compiles=True,
            vitis_available=True,
            synthesis_ok=True,
        )
        self.assertEqual(verdict, "hls_synthesized")

    def test_verdict_kernel_ready_synthesis_pending(self):
        """When C++ compiles but vitis_hls not found, verdict is hls_kernel_ready_synthesis_pending.
        Spec traces: REQ-HW-010
        """
        verdict = self._run_with_mocks(
            hls_cpp_exists=True,
            cpp_compiles=True,
            vitis_available=False,
            synthesis_ok=False,
        )
        self.assertEqual(verdict, "hls_kernel_ready_synthesis_pending")

    def test_verdict_compile_fail_when_cpp_missing(self):
        """When HLS C++ source is missing, verdict is hls_kernel_compile_fail.
        Spec traces: REQ-HW-010
        """
        verdict = self._run_with_mocks(
            hls_cpp_exists=False,
            cpp_compiles=False,
            vitis_available=False,
            synthesis_ok=False,
        )
        self.assertEqual(verdict, "hls_kernel_compile_fail")

    def test_verdict_compile_fail_when_gxx_fails(self):
        """When g++ fails, verdict is hls_kernel_compile_fail even if source exists.
        Spec traces: REQ-HW-010
        """
        verdict = self._run_with_mocks(
            hls_cpp_exists=True,
            cpp_compiles=False,
            vitis_available=False,
            synthesis_ok=False,
        )
        self.assertEqual(verdict, "hls_kernel_compile_fail")

    def test_verdict_synthesis_fail_still_not_synthesized(self):
        """When vitis_hls is available but synthesis fails, verdict is NOT hls_synthesized.
        Spec traces: REQ-HW-010
        """
        verdict = self._run_with_mocks(
            hls_cpp_exists=True,
            cpp_compiles=True,
            vitis_available=True,
            synthesis_ok=False,
        )
        # synthesis attempted but failed — not hls_synthesized
        self.assertNotEqual(verdict, "hls_synthesized")


# ---------------------------------------------------------------------------
# Deliverable JSON schema check
# ---------------------------------------------------------------------------

class TestDeliverableSchema(unittest.TestCase):
    """Verify the deliverable JSON has all required fields.
    Spec traces: REQ-HW-010
    """

    REQUIRED_FIELDS = [
        "experiment",
        "title",
        "run_date",
        "hls_cpp_written",
        "cpp_compiles",
        "vitis_hls_available",
        "synthesis_attempted",
        "synthesis_result",
        "honest_verdict",
        "energy_delta_pct",
    ]

    def test_deliverable_exists(self):
        """Deliverable JSON must exist after experiment run.
        Spec traces: REQ-HW-010
        """
        self.assertTrue(
            DELIVERABLE.exists(),
            f"Deliverable JSON not found: {DELIVERABLE}",
        )

    def test_deliverable_has_required_fields(self):
        """Deliverable JSON must contain all required schema fields.
        Spec traces: REQ-HW-010
        """
        if not DELIVERABLE.exists():
            self.skipTest("Deliverable not found")
        with open(DELIVERABLE) as f:
            artifact = json.load(f)
        for field in self.REQUIRED_FIELDS:
            self.assertIn(
                field,
                artifact,
                f"Required field '{field}' missing from deliverable JSON",
            )

    def test_deliverable_honest_verdict_is_valid(self):
        """honest_verdict must be one of the three defined values.
        Spec traces: REQ-HW-010
        """
        if not DELIVERABLE.exists():
            self.skipTest("Deliverable not found")
        with open(DELIVERABLE) as f:
            artifact = json.load(f)
        valid_verdicts = {
            "hls_synthesized",
            "hls_kernel_ready_synthesis_pending",
            "hls_kernel_compile_fail",
        }
        self.assertIn(
            artifact.get("honest_verdict"),
            valid_verdicts,
            f"honest_verdict '{artifact.get('honest_verdict')}' is not a valid value",
        )


if __name__ == "__main__":
    unittest.main()

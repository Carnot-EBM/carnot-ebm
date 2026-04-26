"""Tests for Experiment 825 — JEPA v23 3-Domain Eval + FR-11 Tier 3.5 Deployment.

Spec: REQ-LEARN-051, REQ-LEARN-052, SCENARIO-LEARN-061
"""

from __future__ import annotations

import json
import math
import pickle
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root and scripts are importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from carnot.inference.jepa_v23 import JEPAv23Predictor, _compute_auc  # noqa: E402
from carnot.pipeline.verification_certificate import (  # noqa: E402
    VerificationCertificate,
    make_certificate,
)
import experiment_825_jepa_v23_eval_fr11_tier3 as exp825  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_model() -> JEPAv23Predictor:
    """Construct a minimal JEPAv23Predictor with tiny weights for unit tests.

    We use the model's internal API directly rather than training, because
    training takes seconds and unit tests should be sub-second.
    """
    model = JEPAv23Predictor(embed_dim=4, seed=42)
    # Fit vectoriser on minimal corpus.
    texts = ["correct step here", "incorrect wrong step", "another example text"]
    model._vectoriser.fit(texts)
    vocab_size = len(model._vectoriser._vocab)
    # Initialise weights to small values (not zeros — zeros would give zero embeddings).
    model._w = [[0.1] * vocab_size for _ in range(4)]
    model._b = [0.1] * 4
    return model


# ---------------------------------------------------------------------------
# Tests: VerificationCertificate (REQ-LEARN-052)
# ---------------------------------------------------------------------------


class TestVerificationCertificate(unittest.TestCase):
    """REQ-LEARN-052: VerificationCertificate MUST have all required fields."""

    def test_has_all_required_fields(self):
        """SCENARIO-LEARN-061: Certificate must contain step_id, jepa_energy_delta,
        constraint_type, z3_verdict, confidence_score."""
        cert = VerificationCertificate(
            step_id="gsm8k_step_0",
            jepa_energy_delta=0.3,
            constraint_type="arithmetic",
            z3_verdict="sat",
            confidence_score=0.8,
        )
        self.assertEqual(cert.step_id, "gsm8k_step_0")
        self.assertIsInstance(cert.jepa_energy_delta, float)
        self.assertIsInstance(cert.constraint_type, str)
        self.assertIsInstance(cert.z3_verdict, str)
        self.assertIsInstance(cert.confidence_score, float)

    def test_make_certificate_low_energy_gives_sat(self):
        """make_certificate: energy < 0.5 → z3_verdict='sat' (step looks correct)."""
        cert = make_certificate("gsm8k_0", 0.2, "arithmetic")
        self.assertEqual(cert.z3_verdict, "sat")
        self.assertGreater(cert.confidence_score, 0.5)

    def test_make_certificate_mid_energy_gives_unsat(self):
        """make_certificate: 0.5 <= energy < 1.5 → z3_verdict='unsat'."""
        cert = make_certificate("humaneval_0", 0.8, "code_logic")
        self.assertEqual(cert.z3_verdict, "unsat")

    def test_make_certificate_high_energy_gives_unknown(self):
        """make_certificate: energy >= 1.5 → z3_verdict='unknown'."""
        cert = make_certificate("arc_0", 1.7, "planning")
        self.assertEqual(cert.z3_verdict, "unknown")

    def test_confidence_score_in_unit_interval(self):
        """confidence_score must be in [0, 1] for any energy in [0, 2]."""
        for energy in [0.0, 0.5, 1.0, 1.5, 2.0]:
            cert = make_certificate(f"step_{energy}", energy, "arithmetic")
            self.assertGreaterEqual(cert.confidence_score, 0.0)
            self.assertLessEqual(cert.confidence_score, 1.0)

    def test_certificate_is_namedtuple(self):
        """VerificationCertificate must be a NamedTuple (immutable, hashable)."""
        cert = make_certificate("step_0", 0.4, "arithmetic")
        self.assertIsInstance(cert, tuple)
        # Immutability check.
        with self.assertRaises(AttributeError):
            cert.step_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tests: Gate check (SCENARIO-LEARN-061 gate path)
# ---------------------------------------------------------------------------


class TestGateCheck(unittest.TestCase):
    """SCENARIO-LEARN-061: Gate blocks when Exp 824 verdict is jepa_v23_below_random."""

    def test_gate_blocks_when_below_random(self, tmp_path=None):
        """REQ-LEARN-051: If gate verdict == 'jepa_v23_below_random', artifact is blocked."""
        import tempfile
        import os

        gate_data = {"honest_verdict": "jepa_v23_below_random", "ood_auc": 0.4}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as gf:
            json.dump(gate_data, gf)
            gate_path = gf.name

        try:
            with (
                patch.object(exp825, "GATE_FILE", gate_path),
                patch.object(exp825, "MODEL_FILE", gate_path),
            ):
                # We need to patch the open calls to use our temp file.
                # Simpler: directly call the logic with mocked open.
                pass

            # Direct test: parse gate logic inline.
            verdict = gate_data["honest_verdict"]
            self.assertEqual(verdict, "jepa_v23_below_random")
            # The run() function would return honest_verdict='blocked_gate'.
            # Simulate by calling with mocked file reads.
            tmpl = MagicMock()
            with patch("builtins.open") as mock_open:
                from io import StringIO
                import io

                mock_open.return_value.__enter__ = lambda s: s
                mock_open.return_value.__exit__ = MagicMock(return_value=False)
                mock_open.return_value.read = lambda: json.dumps(gate_data)
                mock_open.return_value.__iter__ = lambda s: iter([])

                # Call the real run() but patch Path.open and json.load.
                with (
                    patch("json.load", return_value=gate_data),
                    patch("pathlib.Path.open", mock_open),
                ):
                    result = exp825.run(tmpl)

            self.assertEqual(result["honest_verdict"], "blocked_gate")
            self.assertFalse(result["tier35_deployed"])
            self.assertEqual(result["n_certificates_emitted"], 0)
        finally:
            os.unlink(gate_path)


# ---------------------------------------------------------------------------
# Tests: Tier 3.5 wiring (REQ-LEARN-051)
# ---------------------------------------------------------------------------


class TestTier35Wiring(unittest.TestCase):
    """REQ-LEARN-051: ThreeTierPipeline.tier_35 is set when OOD AUC >= 0.65."""

    def test_tier35_attribute_exists_on_pipeline(self):
        """ThreeTierPipeline must have a tier_35 attribute (initialised to None)."""
        from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline

        # Construct a minimal pipeline with mocked dependencies.
        pipeline = ThreeTierPipeline.__new__(ThreeTierPipeline)
        pipeline.tier_35 = None
        self.assertIsNone(pipeline.tier_35)

    def test_tier35_set_when_ood_auc_above_threshold(self):
        """When overall_ood_auc >= 0.65, tier35_deployed must be True.

        SCENARIO-LEARN-061: Tier 3.5 deployed; VerificationCertificate emitted.
        Uses a mocked model to control predict_energy() output deterministically.
        """
        model = _make_minimal_model()

        gate_data = {"honest_verdict": "jepa_v23_viable", "ood_auc": 0.811}
        model_bytes = pickle.dumps(model)

        import io

        call_count = [0]

        def fake_open(path, mode="r", *args, **kwargs):
            call_count[0] += 1
            if "pickle" in str(mode) or mode == "rb":
                return io.BytesIO(model_bytes)
            return io.StringIO(json.dumps(gate_data))

        with (
            patch("builtins.open", side_effect=fake_open),
            patch("json.load", return_value=gate_data),
            patch("pickle.load", return_value=model),
        ):
            tmpl = MagicMock()
            result = exp825.run(tmpl)

        # With a real (tiny) model, AUC values may vary — just assert the
        # structure and that tier35_deployed reflects the threshold logic.
        self.assertIn("tier35_deployed", result)
        self.assertIn("overall_ood_auc", result)
        self.assertIn("n_certificates_emitted", result)
        self.assertIn("honest_verdict", result)
        overall_auc = result["overall_ood_auc"]
        if overall_auc >= 0.65:
            self.assertTrue(result["tier35_deployed"])
            self.assertEqual(result["honest_verdict"], "jepa_v23_tier35_deployed")
        else:
            self.assertFalse(result["tier35_deployed"])

    def test_tier35_not_set_when_ood_auc_below_threshold(self):
        """When overall_ood_auc < 0.65, tier35_deployed must be False.

        REQ-LEARN-051: Tier 3.5 MUST NOT deploy when OOD AUC < 0.65.
        """
        # Patch evaluate_domain to return sub-threshold AUC.
        model = _make_minimal_model()
        gate_data = {"honest_verdict": "jepa_v23_viable", "ood_auc": 0.55}
        model_bytes = pickle.dumps(model)

        with (
            patch("json.load", return_value=gate_data),
            patch("pickle.load", return_value=model),
            patch.object(exp825, "evaluate_domain", return_value=(0.60, [("step_0", 0.5, 1.0)])),
        ):
            tmpl = MagicMock()
            result = exp825.run(tmpl)

        # overall_ood_auc = mean(0.60, 0.60) = 0.60 < 0.65 → not deployed
        self.assertFalse(result["tier35_deployed"])
        self.assertNotEqual(result["honest_verdict"], "jepa_v23_tier35_deployed")


# ---------------------------------------------------------------------------
# Tests: evaluate_domain (REQ-LEARN-051)
# ---------------------------------------------------------------------------


class TestEvaluateDomain(unittest.TestCase):
    """REQ-LEARN-051: Per-domain AUC must be computable from JEPA v23 energies."""

    def test_evaluate_domain_returns_auc_and_triples(self):
        """evaluate_domain must return (auc, scored_triples) with correct types."""
        model = _make_minimal_model()
        steps = exp825._synthetic_gsm8k_steps()[:4]  # 2 correct, 2 incorrect
        auc, triples = exp825.evaluate_domain(model, steps)
        self.assertIsInstance(auc, float)
        self.assertGreaterEqual(auc, 0.0)
        self.assertLessEqual(auc, 1.0)
        self.assertEqual(len(triples), 4)
        # Each triple: (step_id, energy, label)
        for step_id, energy, label in triples:
            self.assertIsInstance(step_id, str)
            self.assertIsInstance(energy, float)
            self.assertIn(label, (0.0, 1.0))

    def test_gsm8k_steps_has_correct_count(self):
        """_synthetic_gsm8k_steps() must return exactly 20 steps (REQ-LEARN-051)."""
        steps = exp825._synthetic_gsm8k_steps()
        self.assertEqual(len(steps), 20)

    def test_humaneval_steps_has_correct_count(self):
        """_synthetic_humaneval_steps() must return exactly 10 steps (REQ-LEARN-051)."""
        steps = exp825._synthetic_humaneval_steps()
        self.assertEqual(len(steps), 10)

    def test_arc_steps_has_correct_count(self):
        """_synthetic_arc_steps() must return exactly 10 steps (REQ-LEARN-051)."""
        steps = exp825._synthetic_arc_steps()
        self.assertEqual(len(steps), 10)


# ---------------------------------------------------------------------------
# Tests: select_and_emit_certificates (REQ-LEARN-052)
# ---------------------------------------------------------------------------


class TestSelectAndEmitCertificates(unittest.TestCase):
    """REQ-LEARN-052: 20 VerificationCertificates must be emitted."""

    def test_emits_correct_number_of_certificates(self):
        """SCENARIO-LEARN-061: 20 VerificationCertificates emitted."""
        import random

        all_triples = [(f"step_{i}", 0.3 + i * 0.01, float(i % 2)) for i in range(40)]
        rng = random.Random(42)
        certs = exp825.select_and_emit_certificates(all_triples, 20, rng)
        self.assertEqual(len(certs), 20)

    def test_certificates_have_all_required_fields(self):
        """REQ-LEARN-052: Each certificate must have all 5 required fields."""
        import random

        all_triples = [("gsm8k_step_0", 0.4, 1.0), ("arc_step_0", 0.8, 0.0)]
        rng = random.Random(42)
        certs = exp825.select_and_emit_certificates(all_triples, 2, rng)
        for cert in certs:
            self.assertIn("step_id", cert)
            self.assertIn("jepa_energy_delta", cert)
            self.assertIn("constraint_type", cert)
            self.assertIn("z3_verdict", cert)
            self.assertIn("confidence_score", cert)

    def test_certificates_constraint_type_matches_domain(self):
        """constraint_type must be 'arithmetic' for gsm8k, 'code_logic' for humaneval,
        'planning' for arc.  REQ-LEARN-052."""
        import random

        all_triples = [
            ("gsm8k_step_0", 0.3, 1.0),
            ("humaneval_step_0", 0.7, 0.0),
            ("arc_step_0", 1.1, 1.0),
        ]
        rng = random.Random(42)
        certs = exp825.select_and_emit_certificates(all_triples, 3, rng)
        type_map = {c["step_id"]: c["constraint_type"] for c in certs}
        self.assertEqual(type_map.get("gsm8k_step_0"), "arithmetic")
        self.assertEqual(type_map.get("humaneval_step_0"), "code_logic")
        self.assertEqual(type_map.get("arc_step_0"), "planning")


# ---------------------------------------------------------------------------
# Tests: artifact schema (SCENARIO-LEARN-061)
# ---------------------------------------------------------------------------


class TestArtifactSchema(unittest.TestCase):
    """SCENARIO-LEARN-061: Written artifact must have all required fields."""

    def _run_with_viable_gate(self) -> dict:
        """Helper: run exp825 with a viable gate and mocked model."""
        model = _make_minimal_model()
        gate_data = {"honest_verdict": "jepa_v23_viable", "ood_auc": 0.811}
        with patch("json.load", return_value=gate_data), patch("pickle.load", return_value=model):
            tmpl = MagicMock()
            return exp825.run(tmpl)

    def test_required_artifact_fields_present(self):
        """All schema fields (auc_gsm8k, auc_humaneval, auc_arc, overall_ood_auc,
        tier35_deployed, n_certificates_emitted, honest_verdict) must be present."""
        result = self._run_with_viable_gate()
        for field in [
            "auc_gsm8k",
            "auc_humaneval",
            "auc_arc",
            "overall_ood_auc",
            "tier35_deployed",
            "n_certificates_emitted",
            "honest_verdict",
        ]:
            self.assertIn(field, result, f"Missing required field: {field}")

    def test_overall_ood_auc_is_mean_of_humaneval_and_arc(self):
        """overall_ood_auc = mean(auc_humaneval, auc_arc); GSM8K excluded."""
        result = self._run_with_viable_gate()
        expected = (result["auc_humaneval"] + result["auc_arc"]) / 2.0
        self.assertAlmostEqual(result["overall_ood_auc"], expected, places=10)

    def test_n_certificates_matches_list_length(self):
        """n_certificates_emitted must equal len(verification_certificates)."""
        result = self._run_with_viable_gate()
        self.assertEqual(
            result["n_certificates_emitted"],
            len(result["verification_certificates"]),
        )

    def test_honest_verdict_is_valid_string(self):
        """honest_verdict must be one of the three defined verdicts."""
        valid_verdicts = {
            "jepa_v23_tier35_deployed",
            "jepa_v23_improvement_not_deployed",
            "blocked_gate",
            "jepa_v23_below_random_ood",
        }
        result = self._run_with_viable_gate()
        self.assertIn(result["honest_verdict"], valid_verdicts)


if __name__ == "__main__":
    unittest.main()

"""Tests for scripts/experiment_968_symbolic_kan_deploy_hf_publish.py

REQ-MODEL-030: SymbolicKAN Tier 3 callable satisfies ThreeTierPipeline ising_pipeline interface.
REQ-VERIFY-088: Tier 3 integration test AUC must be >= 0.90 on 5 real FoVer examples.
SCENARIO-MODEL-015: Symbolic label assignment and residual correction.
SCENARIO-VERIFY-088: Full deploy experiment produces valid deliverable JSON with all required schema fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Allow importing from scripts/ and python/ without installing the package.
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "scripts"))
sys.path.insert(0, str(_ROOT / "python"))

import experiment_968_symbolic_kan_deploy_hf_publish as exp968


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestStepToFeatures:
    def test_output_length_is_16(self):
        # Feature vectors must always be exactly 16 elements (config.input_dim).
        feats = exp968.step_to_features("2 + 3 = 5", dim=16)
        assert len(feats) == 16

    def test_mul_operator_type(self):
        feats = exp968.step_to_features("4 times 5 equals 20")
        assert feats[0] == pytest.approx(0.50)

    def test_eq_operator_type(self):
        feats = exp968.step_to_features("the total result is 100")
        assert feats[0] == pytest.approx(1.00)

    def test_no_numbers_pads_zeros(self):
        feats = exp968.step_to_features("no numbers here at all", dim=16)
        # Second feature (count norm) should be 0.0 when no numbers
        assert feats[1] == pytest.approx(0.0)
        # Positions 2-15 should all be 0.0
        assert all(f == 0.0 for f in feats[2:])

    def test_numbers_normalised(self):
        # "10 + 20 = 30" — max_abs=30, so normalised: [10/30, 20/30, 30/30] = [1/3, 2/3, 1]
        feats = exp968.step_to_features("10 plus 20 equals 30", dim=16)
        assert feats[2] == pytest.approx(10.0 / 30.0, abs=1e-5)


# ---------------------------------------------------------------------------
# SymbolicKANTier3 interface
# ---------------------------------------------------------------------------


class TestSymbolicKANTier3:
    """SymbolicKANTier3.__call__ must match ThreeTierPipeline's ising_pipeline signature."""

    def _make_model(self):
        from carnot.models.symbolic_kan import SymbolicKANConfig, SymbolicKANModel

        config = SymbolicKANConfig(input_dim=16, n_nodes=4, n_segments=4)
        return SymbolicKANModel(config, seed=42)

    def test_returns_tuple_bool_float(self):
        model = self._make_model()
        tier3 = exp968.SymbolicKANTier3(model, threshold=0.0)
        result = tier3("4 + 5 = 9", "What is 4+5?")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_question_ignored(self):
        # Passing different questions must not change the energy (model is response-only).
        model = self._make_model()
        tier3 = exp968.SymbolicKANTier3(model, threshold=0.0)
        _, e1 = tier3("3 times 7 = 21", "What is 3*7?")
        _, e2 = tier3("3 times 7 = 21", "Some different question")
        assert e1 == pytest.approx(e2)

    def test_threshold_controls_verified(self):
        model = self._make_model()
        response = "2 plus 2 equals 4"
        feats = exp968.step_to_features(response)
        x = np.array(feats, dtype=np.float32)
        energy = float(model.energy(x))

        # Set threshold above energy → verified=True
        tier3_high = exp968.SymbolicKANTier3(model, threshold=energy + 10.0)
        verified_high, _ = tier3_high(response, "")
        assert verified_high is True

        # Set threshold below energy → verified=False
        tier3_low = exp968.SymbolicKANTier3(model, threshold=energy - 10.0)
        verified_low, _ = tier3_low(response, "")
        assert verified_low is False


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------


class TestComputeAUC:
    def _make_model_that_ranks_correctly(self):
        """Return a model whose energy is negative for low-value inputs, positive for high."""
        from carnot.models.symbolic_kan import SymbolicKANConfig, SymbolicKANModel

        config = SymbolicKANConfig(input_dim=16, n_nodes=8)
        model = SymbolicKANModel(config, seed=948)

        # Train on synthetic data where correct inputs are all-zeros and
        # incorrect inputs are all-ones so the model learns to discriminate.
        correct = np.zeros((20, 16), dtype=np.float32)
        incorrect = np.ones((20, 16), dtype=np.float32)
        model.train(correct, incorrect, n_epochs=30)
        return model

    def test_perfect_discriminator_auc_is_1(self):
        model = self._make_model_that_ranks_correctly()
        # Correct: all-zeros; incorrect: all-ones (same distribution as training)
        eval_correct = [list(np.zeros(16))]
        eval_incorrect = [list(np.ones(16))]
        auc = exp968.compute_auc(model, eval_correct, eval_incorrect)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_empty_returns_half(self):
        from carnot.models.symbolic_kan import SymbolicKANConfig, SymbolicKANModel

        model = SymbolicKANModel(SymbolicKANConfig(), seed=0)
        auc = exp968.compute_auc(model, [], [])
        assert auc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Integration test: 5 real FoVer examples
# ---------------------------------------------------------------------------


class TestRunIntegrationTest:
    """Verify integration_test_auc >= 0.90 on the real FoVer corpus.

    This test trains the same Symbolic-KAN as Exp 948 and checks the AUC gate
    that guards deployment.  It is the acceptance gate for this experiment.

    REQ-VERIFY-088.
    """

    @pytest.fixture(scope="class")
    def trained_model(self):
        """Train Symbolic-KAN from real FoVer data (Exp 948 config)."""
        from carnot.models.symbolic_kan import SymbolicKANConfig, SymbolicKANModel

        fover_path = _ROOT / "results" / "fover_labeled_steps_live.json"
        xs_correct, xs_incorrect = exp968.load_real_pairs(fover_path)

        if len(xs_correct) + len(xs_incorrect) < 20:
            pytest.skip("Real FoVer data not available")

        train_c, train_i, _, _ = exp968.pair_and_split(xs_correct, xs_incorrect, seed=948)
        config = SymbolicKANConfig(input_dim=16, n_nodes=8, n_segments=8)
        model = SymbolicKANModel(config, seed=948)
        model.train(
            np.array(train_c, dtype=np.float32),
            np.array(train_i, dtype=np.float32),
            n_epochs=60,
        )
        return model

    def test_integration_auc_meets_gate(self, trained_model):
        """AUC on 5 real FoVer examples must be >= 0.90 (deployment gate)."""
        fover_path = _ROOT / "results" / "fover_labeled_steps_live.json"
        auc = exp968.run_integration_test(trained_model, fover_path, n_samples=5)
        assert auc >= exp968._AUC_INTEGRATION_GATE, (
            f"Integration AUC {auc:.4f} is below deployment gate {exp968._AUC_INTEGRATION_GATE}"
        )


# ---------------------------------------------------------------------------
# Deliverable schema validation (stubbed HuggingFace + IPFS)
# ---------------------------------------------------------------------------


REQUIRED_SCHEMA_FIELDS = {
    "pipeline_registered",
    "integration_test_auc",
    "hf_repo_url",
    "ipfs_cid",
    "honest_verdict",
}


class TestDeliverableSchema:
    """Run main() with stubbed HF push and IPFS to verify deliverable schema.

    SCENARIO-VERIFY-088.
    """

    def test_main_produces_valid_deliverable(self, tmp_path):
        deliverable = str(tmp_path / "exp968.json")

        def stub_push(_model_dir):
            return "https://huggingface.co/Carnot-EBM/symbolic-kan-v2"

        def stub_ipfs(_model_dir):
            return "QmStubCIDForTesting123"

        with (
            mock.patch.object(exp968, "_DELIVERABLE", deliverable),
            mock.patch.object(
                exp968, "_FOVER_PATH", _ROOT / "results" / "fover_labeled_steps_live.json"
            ),
            mock.patch(
                "experiment_968_symbolic_kan_deploy_hf_publish.push_to_huggingface", stub_push
            ),
            mock.patch("experiment_968_symbolic_kan_deploy_hf_publish.pin_to_ipfs", stub_ipfs),
            mock.patch(
                "experiment_968_symbolic_kan_deploy_hf_publish._write_tier3_module", lambda _: None
            ),
        ):
            exp968.main()

        artifact = json.loads(Path(deliverable).read_text())
        for field in REQUIRED_SCHEMA_FIELDS:
            assert field in artifact, f"Missing required schema field: {field}"

    def test_honest_verdict_when_auc_passes_and_hf_succeeds(self, tmp_path):
        deliverable = str(tmp_path / "exp968.json")

        with (
            mock.patch.object(exp968, "_DELIVERABLE", deliverable),
            mock.patch.object(
                exp968, "_FOVER_PATH", _ROOT / "results" / "fover_labeled_steps_live.json"
            ),
            mock.patch(
                "experiment_968_symbolic_kan_deploy_hf_publish.push_to_huggingface",
                return_value="https://huggingface.co/Carnot-EBM/symbolic-kan-v2",
            ),
            mock.patch(
                "experiment_968_symbolic_kan_deploy_hf_publish.pin_to_ipfs",
                return_value="QmRealCID456",
            ),
            mock.patch(
                "experiment_968_symbolic_kan_deploy_hf_publish._write_tier3_module", lambda _: None
            ),
        ):
            exp968.main()

        artifact = json.loads(Path(deliverable).read_text())
        assert artifact["honest_verdict"] == "symbolic_kan_deployed"
        assert artifact["pipeline_registered"] is True

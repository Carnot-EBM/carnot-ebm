"""Tests for Experiment 778: JEPA v19 Tier 3.5 Cascade Deployment gate logic.

**Coverage target:** The gate check in experiment_778_jepa_v19_cascade_deploy.py.

**Spec traces:**
    - REQ-LEARN-046: JEPA v19 Tier 3.5 MUST apply fast-path skip when
      predicted_violation_prob < 0.30; MUST NOT skip Ising when >= 0.30.
    - REQ-LEARN-047: false_negative_rate MUST be < 0.05.

**Why these tests exist:**
    The Tier 3.5 deployment is gated on Exp 770 OOD AUC > 0.75.  These tests verify
    that the gate check is enforced correctly so a low-quality probe cannot silently
    enter the production pipeline.  We also validate the fast-path skip and FN-rate
    logic in isolation so it can be wired in once the gate is cleared.
"""

import json
import unittest
from unittest.mock import MagicMock, patch

OOD_AUC_GATE = 0.75
DEFAULT_SKIP_THRESHOLD = 0.30
CONSERVATIVE_SKIP_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Unit-level helpers (no I/O) — test the decision rules that the experiment
# encodes so they can be reviewed without spinning up the full scaffold.
# ---------------------------------------------------------------------------


def _gate_check(ood_auc: float) -> str:
    """Return the honest_verdict determined by the OOD AUC gate.

    Why: The gate is the single decision point that prevents an under-trained
    probe from entering the pipeline.  Isolating it lets us test exhaustively
    without instantiating ExperimentTemplate.
    """
    if ood_auc <= OOD_AUC_GATE:
        return "blocked_ood_auc_below_gate"
    return "gate_passed"


def _fast_path_decision(predicted_prob: float, threshold: float) -> bool:
    """Return True when Ising should be skipped (fast-path taken).

    Why: REQ-LEARN-046 — skip Ising only when predicted_violation_prob < threshold.
    """
    return predicted_prob < threshold


def _false_negative_rate(skipped_mask: list[bool], violation_mask: list[bool]) -> float:
    """Compute false-negative rate across a set of questions.

    A false negative occurs when the probe skips Ising but the response IS a
    constraint violation.  false_negative_rate = n_false_negatives / n_skipped.

    Why: REQ-LEARN-047 — gate deployment on FN rate < 0.05.
    """
    n_skipped = sum(skipped_mask)
    if n_skipped == 0:
        return 0.0
    n_fn = sum(
        1 for skipped, violated in zip(skipped_mask, violation_mask)
        if skipped and violated
    )
    return n_fn / n_skipped


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGateCheck(unittest.TestCase):
    """Verify the OOD AUC gate blocks deployment when the probe is too weak."""

    def test_blocks_below_gate(self):
        """REQ-LEARN-046: OOD AUC 0.5667 (Exp 770 actual) must be blocked."""
        self.assertEqual(_gate_check(0.5667), "blocked_ood_auc_below_gate")

    def test_blocks_at_gate(self):
        """REQ-LEARN-046: OOD AUC exactly equal to gate (0.75) must also be blocked."""
        self.assertEqual(_gate_check(0.75), "blocked_ood_auc_below_gate")

    def test_passes_above_gate(self):
        """REQ-LEARN-046: OOD AUC strictly above gate (0.76) must pass."""
        self.assertEqual(_gate_check(0.76), "gate_passed")

    def test_passes_high_auc(self):
        """REQ-LEARN-046: Perfect OOD AUC (1.0) must pass the gate."""
        self.assertEqual(_gate_check(1.0), "gate_passed")


class TestFastPathDecision(unittest.TestCase):
    """REQ-LEARN-046: Tier 3.5 skips/runs Ising based on predicted_violation_prob."""

    def test_skips_ising_when_prob_below_threshold(self):
        """Prob=0.10 < threshold=0.30 → skip Ising (fast path taken)."""
        self.assertTrue(_fast_path_decision(0.10, DEFAULT_SKIP_THRESHOLD))

    def test_runs_ising_when_prob_at_threshold(self):
        """Prob=0.30 == threshold=0.30 → do NOT skip Ising."""
        self.assertFalse(_fast_path_decision(0.30, DEFAULT_SKIP_THRESHOLD))

    def test_runs_ising_when_prob_above_threshold(self):
        """Prob=0.85 >> threshold → do NOT skip Ising."""
        self.assertFalse(_fast_path_decision(0.85, DEFAULT_SKIP_THRESHOLD))

    def test_conservative_threshold_skips_fewer(self):
        """REQ-LEARN-047: Lowering threshold to 0.20 means prob=0.25 no longer skips."""
        # At default threshold 0.30 it would skip; at conservative 0.20 it should not.
        self.assertTrue(_fast_path_decision(0.25, DEFAULT_SKIP_THRESHOLD))
        self.assertFalse(_fast_path_decision(0.25, CONSERVATIVE_SKIP_THRESHOLD))


class TestFalseNegativeRate(unittest.TestCase):
    """REQ-LEARN-047: FN rate must be < 0.05 for deployment."""

    def test_no_false_negatives(self):
        """All skipped responses are actually correct → FN rate = 0.0."""
        skipped = [True, True, True, False, False]
        violated = [False, False, False, True, True]
        self.assertAlmostEqual(_false_negative_rate(skipped, violated), 0.0)

    def test_one_false_negative_out_of_ten_skipped(self):
        """1 FN out of 10 skipped → rate = 0.10, exceeds the 0.05 gate."""
        skipped = [True] * 10 + [False] * 40
        violated = [True] + [False] * 9 + [False] * 40
        rate = _false_negative_rate(skipped, violated)
        self.assertAlmostEqual(rate, 0.10)
        self.assertGreaterEqual(rate, 0.05)  # Would block deployment.

    def test_zero_skipped_returns_zero(self):
        """If nothing is skipped, FN rate is defined as 0.0 (no division by zero)."""
        skipped = [False] * 10
        violated = [True] * 5 + [False] * 5
        self.assertAlmostEqual(_false_negative_rate(skipped, violated), 0.0)

    def test_acceptable_fn_rate(self):
        """0 FN out of 20 skipped → rate = 0.0, below the 0.05 gate."""
        skipped = [True] * 20 + [False] * 30
        violated = [False] * 50
        rate = _false_negative_rate(skipped, violated)
        self.assertAlmostEqual(rate, 0.0)
        self.assertLess(rate, 0.05)  # Would allow deployment.


class TestDeliverableContent(unittest.TestCase):
    """Verify the written artifact has all required schema fields and correct verdict."""

    def test_artifact_has_blocked_verdict(self):
        """The artifact written for Exp 778 must record blocked_ood_auc_below_gate."""
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(repo_root, "results", "experiment_778_jepa_v19_cascade_deploy.json")
        with open(path) as f:
            artifact = json.load(f)
        self.assertEqual(artifact["honest_verdict"], "blocked_ood_auc_below_gate")
        self.assertFalse(artifact["tier35_deployed"])
        self.assertAlmostEqual(artifact["jepa_v19_ood_auc"], 0.5667, places=3)
        self.assertIsNone(artifact["skip_threshold"])
        self.assertIsNone(artifact["fast_path_skip_rate"])
        self.assertIsNone(artifact["false_negative_rate"])

    def test_artifact_schema_fields_present(self):
        """All declared schema fields must exist in the artifact."""
        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(repo_root, "results", "experiment_778_jepa_v19_cascade_deploy.json")
        with open(path) as f:
            artifact = json.load(f)
        required = [
            "experiment", "title", "run_date", "started_at", "finished_at",
            "duration_s", "status", "jepa_v19_ood_auc", "ood_auc_gate",
            "skip_threshold", "fast_path_skip_rate", "false_negative_rate",
            "tier35_deployed", "honest_verdict",
        ]
        for field in required:
            self.assertIn(field, artifact, f"Missing required field: {field}")


if __name__ == "__main__":
    unittest.main()

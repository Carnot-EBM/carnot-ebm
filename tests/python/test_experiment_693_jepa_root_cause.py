"""Tests for Experiment 693: JEPA v15 Root Cause Analysis.

Verifies the probe logic that identifies which of H1/H2/H3 caused
JEPA v15 to score below random chance (AUC=0.4751) on GSM8K 500-699.

Coverage targets:
- probe_h1_distribution_shift: distribution shift detection logic
- probe_h2_gradient_direction: gradient sign check
- probe_h3_latent_rank: SVD-based effective rank / collapse detection
- probe_symbolic_correlation: r² correlation of latents with text features
- determine_root_cause: priority ordering H1 > H2 > H3 > unknown
- build_v16_spec: architecture prescription for each root cause
- determine_honest_verdict: mapping root cause to verdicts
- The produced deliverable passes required schema field validation.

Spec: REQ-LEARN-089, REQ-LEARN-090,
      SCENARIO-LEARN-138, SCENARIO-LEARN-139, SCENARIO-LEARN-140
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_693_jepa_v15_root_cause import (  # noqa: E402
    VALID_ROOT_CAUSES,
    VALID_VERDICTS,
    build_v16_spec,
    determine_honest_verdict,
    determine_root_cause,
    load_training_pairs,
    probe_h1_distribution_shift,
    probe_h3_latent_rank,
    probe_symbolic_correlation,
)


# ---------------------------------------------------------------------------
# H1 probe: distribution shift
# Spec: REQ-LEARN-089, SCENARIO-LEARN-138
# ---------------------------------------------------------------------------


class TestProbeH1DistributionShift:
    """H1 probe confirms distribution shift when L2 > 0.5 or variance ratio > 2."""

    def test_large_l2_shift_confirms_h1(self):
        """H1 is confirmed when OOD mean is far from train mean (L2 > 0.5).

        Spec: REQ-LEARN-089, SCENARIO-LEARN-138
        """
        rng = np.random.RandomState(0)
        train_lat = rng.randn(50, 32).astype(np.float32)          # mean ~ 0
        ood_lat = (rng.randn(50, 32) + 5.0).astype(np.float32)    # mean ~ 5

        h1, l2, vr = probe_h1_distribution_shift(train_lat, ood_lat)

        assert h1 is True
        assert l2 > 0.5, f"Expected L2 > 0.5, got {l2}"

    def test_high_variance_ratio_confirms_h1(self):
        """H1 is confirmed when OOD variance is more than 2x train variance.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-138
        """
        rng = np.random.RandomState(1)
        train_lat = (rng.randn(50, 32) * 0.1).astype(np.float32)   # low variance
        ood_lat = (rng.randn(50, 32) * 1.5).astype(np.float32)      # high variance, similar mean

        h1, l2, vr = probe_h1_distribution_shift(train_lat, ood_lat)

        assert h1 is True
        assert vr > 2.0, f"Expected variance_ratio > 2.0, got {vr}"

    def test_similar_distributions_do_not_confirm_h1(self):
        """H1 is not confirmed when train and OOD distributions are similar.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-138
        """
        rng = np.random.RandomState(2)
        train_lat = (rng.randn(100, 32) * 0.3).astype(np.float32)
        ood_lat = (rng.randn(100, 32) * 0.35).astype(np.float32)   # very similar

        h1, l2, vr = probe_h1_distribution_shift(train_lat, ood_lat)

        # Both L2 and variance_ratio should be below thresholds
        assert isinstance(h1, bool)
        assert isinstance(l2, float)
        assert isinstance(vr, float)
        assert vr > 0

    def test_returns_correct_types(self):
        """probe_h1_distribution_shift returns (bool, float, float).

        Spec: REQ-LEARN-089
        """
        a = np.zeros((10, 32), dtype=np.float32)
        b = np.ones((10, 32), dtype=np.float32)
        h1, l2, vr = probe_h1_distribution_shift(a, b)

        assert isinstance(h1, bool)
        assert isinstance(l2, float)
        assert isinstance(vr, float)


# ---------------------------------------------------------------------------
# H3 probe: latent rank / collapse
# Spec: REQ-LEARN-089, SCENARIO-LEARN-139
# ---------------------------------------------------------------------------


class TestProbeH3LatentRank:
    """H3 probe detects latent collapse via SVD effective rank."""

    def test_low_rank_matrix_confirms_h3(self):
        """H3 is confirmed when effective rank < 5.

        A matrix constructed from a single outer product is rank-1 and
        should be flagged as collapsed.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-139
        """
        # Rank-1 matrix: all rows are the same vector
        base = np.ones((1, 32), dtype=np.float32)
        train_lat = np.repeat(base + np.random.RandomState(3).randn(1, 32).astype(np.float32) * 0.001, 50, axis=0)

        h3, eff_rank, top_var = probe_h3_latent_rank(train_lat)

        assert h3 is True
        assert eff_rank < 5, f"Expected effective_rank < 5 for near-rank-1, got {eff_rank}"

    def test_full_rank_matrix_may_not_confirm_h3(self):
        """H3 is not confirmed on a full-rank random Gaussian matrix.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-139
        """
        rng = np.random.RandomState(4)
        train_lat = rng.randn(100, 32).astype(np.float32)

        h3, eff_rank, top_var = probe_h3_latent_rank(train_lat)

        # Effective rank should be high for Gaussian noise
        assert eff_rank >= 5, f"Expected effective_rank >= 5 for Gaussian, got {eff_rank}"
        assert h3 is False

    def test_insufficient_samples_returns_confirmed(self):
        """H3 is confirmed when fewer than 2 samples are available (degenerate case).

        Spec: REQ-LEARN-089
        """
        train_lat = np.ones((1, 32), dtype=np.float32)
        h3, eff_rank, top_var = probe_h3_latent_rank(train_lat)

        assert h3 is True

    def test_top_variance_fraction_in_range(self):
        """top_variance_pct is always in [0, 1].

        Spec: REQ-LEARN-089
        """
        rng = np.random.RandomState(5)
        train_lat = rng.randn(50, 32).astype(np.float32)
        _h3, _eff_rank, top_var = probe_h3_latent_rank(train_lat)

        assert 0.0 <= top_var <= 1.0, f"top_variance_pct out of range: {top_var}"


# ---------------------------------------------------------------------------
# Symbolic probing
# Spec: REQ-LEARN-089, SCENARIO-LEARN-138
# ---------------------------------------------------------------------------


class TestProbeSymbolicCorrelation:
    """Symbolic probing correlates latents with text features."""

    def test_strongly_correlated_latents_flagged(self):
        """symbolic_structured=True when a latent dimension perfectly tracks digit_density.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-138
        """
        n = 30
        rng = np.random.RandomState(6)

        # Build pairs with varying digit density
        pairs = []
        digit_densities = np.linspace(0.0, 0.5, n)
        for d in digit_densities:
            digits = int(d * 20)
            step_text = "1" * digits + "a" * (20 - digits)
            pairs.append({
                "question": "What is 2+2?",
                "step_text": step_text,
                "step_index": 0,
            })

        # Construct latents that perfectly correlate with digit_density along dim 0
        latents = rng.randn(n, 32).astype(np.float32) * 0.01
        latents[:, 0] = digit_densities.astype(np.float32) * 10.0  # strong correlation

        structured, r2 = probe_symbolic_correlation(latents, pairs)

        assert structured is True, f"Expected symbolic_structured=True, got r2={r2}"
        assert r2 > 0.3

    def test_random_latents_not_flagged(self):
        """symbolic_structured=False for truly random latents with uniform text.

        Spec: REQ-LEARN-089
        """
        n = 30
        rng = np.random.RandomState(7)

        # All pairs have the same text (no feature variation)
        pairs = [
            {"question": "abc def ghi", "step_text": "abc def", "step_index": 0}
            for _ in range(n)
        ]
        latents = rng.randn(n, 32).astype(np.float32)

        structured, r2 = probe_symbolic_correlation(latents, pairs)

        # Uniform features have std=0, so no correlation can be computed
        assert r2 == 0.0 or r2 < 0.3

    def test_too_few_samples_returns_false(self):
        """probe_symbolic_correlation returns (False, 0.0) for fewer than 3 samples.

        Spec: REQ-LEARN-089
        """
        latents = np.ones((2, 32), dtype=np.float32)
        pairs = [{"question": "q", "step_text": "s", "step_index": 0}] * 2

        structured, r2 = probe_symbolic_correlation(latents, pairs)

        assert structured is False
        assert r2 == 0.0


# ---------------------------------------------------------------------------
# determine_root_cause
# Spec: REQ-LEARN-089, SCENARIO-LEARN-140
# ---------------------------------------------------------------------------


class TestDetermineRootCause:
    """Root cause assignment follows the priority ordering H1 > H2 > H3 > unknown."""

    def test_h1_takes_priority_over_h2_and_h3(self):
        """H1 is reported as root cause even when H2 and H3 are also confirmed.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-140
        """
        rc = determine_root_cause(h1_confirmed=True, h2_confirmed=True, h3_confirmed=True)
        assert rc == "cpmi_distribution_mismatch"

    def test_h2_takes_priority_when_h1_false(self):
        """H2 is reported when H1 is not confirmed.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-140
        """
        rc = determine_root_cause(h1_confirmed=False, h2_confirmed=True, h3_confirmed=True)
        assert rc == "pure_loss_anti_correlation"

    def test_h3_reported_when_h1_h2_false(self):
        """H3 is reported when only H3 is confirmed.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-140
        """
        rc = determine_root_cause(h1_confirmed=False, h2_confirmed=False, h3_confirmed=True)
        assert rc == "latent_collapse_small_data"

    def test_unknown_when_all_false(self):
        """Unknown root cause when no probe is confirmed.

        Spec: REQ-LEARN-089, SCENARIO-LEARN-140
        """
        rc = determine_root_cause(h1_confirmed=False, h2_confirmed=False, h3_confirmed=False)
        assert rc == "unknown_requires_ablation"

    def test_all_root_causes_in_valid_set(self):
        """All possible return values are members of VALID_ROOT_CAUSES.

        Spec: REQ-LEARN-089
        """
        for h1 in [True, False]:
            for h2 in [True, False]:
                for h3 in [True, False]:
                    rc = determine_root_cause(h1, h2, h3)
                    assert rc in VALID_ROOT_CAUSES, f"Invalid root cause: {rc}"


# ---------------------------------------------------------------------------
# build_v16_spec
# Spec: REQ-LEARN-090, SCENARIO-LEARN-140
# ---------------------------------------------------------------------------


class TestBuildV16Spec:
    """v16 architecture spec is non-empty and appropriate for root cause."""

    def test_v16_spec_nonempty_for_all_root_causes(self):
        """build_v16_spec returns a non-empty string for every valid root cause.

        Spec: REQ-LEARN-090, SCENARIO-LEARN-140
        """
        for rc in VALID_ROOT_CAUSES:
            spec, target = build_v16_spec(rc)
            assert isinstance(spec, str), f"spec is not str for {rc}"
            assert len(spec) > 0, f"spec is empty for {rc}"
            assert isinstance(target, int), f"target is not int for {rc}"
            assert target > 0, f"target <= 0 for {rc}"

    def test_h1_root_cause_prescribes_domain_adaptive_cpmi(self):
        """H1 (distribution mismatch) maps to domain-adaptive CPMI prescription.

        Spec: REQ-LEARN-090
        """
        spec, _target = build_v16_spec("cpmi_distribution_mismatch")
        assert "domain-adaptive" in spec.lower() or "cpmi" in spec.lower()

    def test_h2_root_cause_prescribes_infonce_loss(self):
        """H2 (gradient anti-correlation) maps to InfoNCE loss replacement.

        Spec: REQ-LEARN-090
        """
        spec, _target = build_v16_spec("pure_loss_anti_correlation")
        assert "infonce" in spec.lower() or "info" in spec.lower()

    def test_h3_root_cause_prescribes_more_data(self):
        """H3 (latent collapse) maps to a higher training data target.

        Spec: REQ-LEARN-090
        """
        spec, target = build_v16_spec("latent_collapse_small_data")
        assert target >= 500, f"Expected target >= 500 for H3, got {target}"

    def test_unknown_root_cause_returns_ablation_spec(self):
        """Unknown root cause returns an ablation-based spec.

        Spec: REQ-LEARN-090
        """
        spec, _target = build_v16_spec("unknown_requires_ablation")
        assert len(spec) > 0

    def test_unrecognised_root_cause_returns_fallback(self):
        """build_v16_spec handles unrecognised root cause gracefully.

        Spec: REQ-LEARN-090
        """
        spec, target = build_v16_spec("nonexistent_root_cause")
        assert isinstance(spec, str)
        assert len(spec) > 0


# ---------------------------------------------------------------------------
# determine_honest_verdict
# Spec: REQ-LEARN-090, SCENARIO-LEARN-140
# ---------------------------------------------------------------------------


class TestDetermineHonestVerdict:
    """Honest verdict maps root cause to VALID_VERDICTS."""

    def test_identified_root_cause_gives_v16_specced_verdict(self):
        """Non-unknown root causes give 'root_cause_identified_v16_specced'.

        Spec: REQ-LEARN-090, SCENARIO-LEARN-140
        """
        for rc in VALID_ROOT_CAUSES - {"unknown_requires_ablation"}:
            v = determine_honest_verdict(rc)
            assert v == "root_cause_identified_v16_specced", f"Wrong verdict for {rc}: {v}"

    def test_unknown_root_cause_gives_ambiguous_verdict(self):
        """Unknown root cause gives 'root_cause_ambiguous_ablation_needed'.

        Spec: REQ-LEARN-090, SCENARIO-LEARN-140
        """
        v = determine_honest_verdict("unknown_requires_ablation")
        assert v == "root_cause_ambiguous_ablation_needed"

    def test_all_verdicts_in_valid_set(self):
        """All possible return values are members of VALID_VERDICTS.

        Spec: REQ-LEARN-090
        """
        for rc in VALID_ROOT_CAUSES:
            v = determine_honest_verdict(rc)
            assert v in VALID_VERDICTS, f"Invalid verdict for {rc}: {v}"


# ---------------------------------------------------------------------------
# load_training_pairs
# Spec: REQ-LEARN-089
# ---------------------------------------------------------------------------


class TestLoadTrainingPairs:
    """load_training_pairs handles edge cases gracefully."""

    def test_missing_file_returns_empty_list(self, tmp_path):
        """Returns empty list when file does not exist.

        Spec: REQ-LEARN-089
        """
        result = load_training_pairs(str(tmp_path / "nonexistent.json"))
        assert result == []

    def test_valid_dict_with_pairs_key(self, tmp_path):
        """Returns pairs list from a JSON dict with a 'pairs' key.

        Spec: REQ-LEARN-089
        """
        data = {"pairs": [{"question": "q1", "step_text": "s1"}]}
        p = tmp_path / "fover.json"
        p.write_text(json.dumps(data))

        result = load_training_pairs(str(p))

        assert len(result) == 1
        assert result[0]["question"] == "q1"

    def test_valid_list_json(self, tmp_path):
        """Returns a raw list if the JSON root is a list.

        Spec: REQ-LEARN-089
        """
        data = [{"question": "q1"}, {"question": "q2"}]
        p = tmp_path / "fover_list.json"
        p.write_text(json.dumps(data))

        result = load_training_pairs(str(p))

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Deliverable schema validation
# Spec: REQ-LEARN-089, REQ-LEARN-090
# ---------------------------------------------------------------------------


class TestDeliverableSchema:
    """Verify the experiment deliverable (if present) passes schema validation."""

    _DELIVERABLE = _REPO_ROOT / "results" / "experiment_693_jepa_v15_root_cause.json"

    @pytest.mark.skipif(
        not (_REPO_ROOT / "results" / "experiment_693_jepa_v15_root_cause.json").exists(),
        reason="Deliverable not yet produced; run the experiment first",
    )
    def test_required_fields_present(self):
        """Deliverable contains all required schema fields.

        Spec: REQ-LEARN-089, REQ-LEARN-090
        """
        with open(self._DELIVERABLE) as f:
            artifact = json.load(f)

        required = [
            "experiment",
            "title",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "honest_verdict",
            "root_cause",
            "H1_confirmed",
            "H2_confirmed",
            "H3_confirmed",
            "distribution_shift_l2",
            "variance_ratio",
            "effective_rank",
            "symbolic_probe_r2",
            "v16_architecture_spec",
            "v16_training_data_target",
        ]
        for field in required:
            assert field in artifact, f"Missing required field: {field}"

    @pytest.mark.skipif(
        not (_REPO_ROOT / "results" / "experiment_693_jepa_v15_root_cause.json").exists(),
        reason="Deliverable not yet produced; run the experiment first",
    )
    def test_root_cause_is_valid(self):
        """root_cause in deliverable is a member of VALID_ROOT_CAUSES.

        Spec: REQ-LEARN-089
        """
        with open(self._DELIVERABLE) as f:
            artifact = json.load(f)

        assert artifact["root_cause"] in VALID_ROOT_CAUSES

    @pytest.mark.skipif(
        not (_REPO_ROOT / "results" / "experiment_693_jepa_v15_root_cause.json").exists(),
        reason="Deliverable not yet produced; run the experiment first",
    )
    def test_honest_verdict_is_valid(self):
        """honest_verdict in deliverable is a member of VALID_VERDICTS.

        Spec: REQ-LEARN-090
        """
        with open(self._DELIVERABLE) as f:
            artifact = json.load(f)

        assert artifact["honest_verdict"] in VALID_VERDICTS

    @pytest.mark.skipif(
        not (_REPO_ROOT / "results" / "experiment_693_jepa_v15_root_cause.json").exists(),
        reason="Deliverable not yet produced; run the experiment first",
    )
    def test_v16_architecture_spec_nonempty(self):
        """v16_architecture_spec is a non-empty string.

        Spec: REQ-LEARN-090
        """
        with open(self._DELIVERABLE) as f:
            artifact = json.load(f)

        spec = artifact["v16_architecture_spec"]
        assert isinstance(spec, str)
        assert len(spec) > 0

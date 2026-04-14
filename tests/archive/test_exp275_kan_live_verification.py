"""Exp 275: KAN adaptive verification on live Exp 219-221 traces.

Trains KANConstraintModel on constraint satisfaction patterns extracted from
real model responses (Exp 221 constraint_ir benchmark), then measures AUROC
improvement from adaptive mesh refinement (AMR) via AdaptiveKAN.

Spec: REQ-CORE-001, REQ-CORE-002, REQ-TIER-001
Scenario: SCENARIO-CORE-001, SCENARIO-TIER-004
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from carnot.models.adaptive_kan import AdaptiveKAN, KANConstraintModel
from carnot.models.trace_features import (
    FEATURE_DIM,
    TraceRecord,
    _majority,
    auroc_score,
    extract_constraint_ir_features,
    load_constraint_ir_traces,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent.parent
_EXP221_PATH = _REPO_ROOT / "results" / "experiment_221_results.json"
_RESULTS_OUT = _REPO_ROOT / "results" / "experiment_275_results.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(
    *,
    families: dict[str, list[str]] | None = None,
    coverage: float = 1.0,
    partial: float = 1.0,
    violations: int = 0,
    mode: str = "baseline",
    style: str = "code_only",
    exact: bool = True,
) -> dict:
    """Build a minimal constraint_ir case dict for unit tests."""
    constraint_results = []
    if families:
        cid = 0
        for fam, statuses in families.items():
            for status in statuses:
                constraint_results.append(
                    {
                        "constraint_id": f"c{cid}",
                        "type": "function_name",
                        "family": fam,
                        "status": status,
                        "judge": "deterministic",
                        "details": {},
                    }
                )
                cid += 1

    return {
        "case_id": "test-0",
        "mode": mode,
        "output_style": style,
        "constraint_extraction_coverage": coverage,
        "partial_satisfaction": partial,
        "semantic_violation_count": violations,
        "exact_satisfaction": exact,
        "evaluation": {
            "constraint_results": constraint_results,
        },
    }


# ---------------------------------------------------------------------------
# Unit tests: _majority
# ---------------------------------------------------------------------------


class TestMajority:
    """REQ-CORE-001: _majority returns correct boolean aggregates."""

    def test_empty_returns_zero(self) -> None:
        """SCENARIO-CORE-001: Empty list → 0.0 (no evidence of satisfaction)."""
        assert _majority([]) == 0.0

    def test_all_true_returns_one(self) -> None:
        """REQ-CORE-001: All-True list → 1.0."""
        assert _majority([True, True, True]) == 1.0

    def test_all_false_returns_zero(self) -> None:
        """REQ-CORE-001: All-False list → 0.0."""
        assert _majority([False, False, False]) == 0.0

    def test_half_tie_returns_zero(self) -> None:
        """REQ-CORE-001: Exactly 50% True → 0.0 (strict majority required)."""
        assert _majority([True, False]) == 0.0

    def test_single_true(self) -> None:
        """REQ-CORE-001: Single True → 1.0."""
        assert _majority([True]) == 1.0

    def test_single_false(self) -> None:
        """REQ-CORE-001: Single False → 0.0."""
        assert _majority([False]) == 0.0

    def test_two_thirds_true(self) -> None:
        """REQ-CORE-001: 2/3 True → 1.0 (strict majority)."""
        assert _majority([True, True, False]) == 1.0


# ---------------------------------------------------------------------------
# Unit tests: extract_constraint_ir_features
# ---------------------------------------------------------------------------


class TestExtractConstraintIrFeatures:
    """REQ-CORE-001: Feature extraction from constraint_ir cases."""

    def test_feature_dim(self) -> None:
        """SCENARIO-CORE-001: Output shape matches FEATURE_DIM."""
        case = _make_case()
        feat = extract_constraint_ir_features(case)
        assert feat.shape == (FEATURE_DIM,)
        assert feat.dtype == np.float32

    def test_all_features_binary(self) -> None:
        """REQ-CORE-001: All features are 0.0 or 1.0."""
        case = _make_case(
            families={"literal": ["satisfied", "violated"], "semantic": ["satisfied"]},
            coverage=0.8,
            partial=0.6,
            violations=1,
            mode="verify_repair",
            style="text_prose",
        )
        feat = extract_constraint_ir_features(case, model_is_gemma=True)
        for val in feat:
            assert val in (0.0, 1.0), f"Non-binary feature value: {val}"

    def test_literal_majority_satisfied(self) -> None:
        """REQ-CORE-001: literal_majority_sat=1 when majority literal satisfied."""
        case = _make_case(
            families={"literal": ["satisfied", "satisfied", "violated"]}
        )
        feat = extract_constraint_ir_features(case)
        assert feat[0] == 1.0  # literal_family_majority_sat

    def test_literal_majority_not_satisfied(self) -> None:
        """REQ-CORE-001: literal_majority_sat=0 when minority literal satisfied."""
        case = _make_case(
            families={"literal": ["violated", "violated", "satisfied"]}
        )
        feat = extract_constraint_ir_features(case)
        assert feat[0] == 0.0

    def test_no_literal_constraints(self) -> None:
        """REQ-CORE-001: Missing family → 0.0 (empty list fallback)."""
        case = _make_case(families={"semantic": ["satisfied"]})
        feat = extract_constraint_ir_features(case)
        assert feat[0] == 0.0  # no literal constraints

    def test_search_opt_majority(self) -> None:
        """REQ-CORE-001: search_opt_majority_sat reflects search_optimization_limited."""
        case = _make_case(
            families={"search_optimization_limited": ["satisfied", "satisfied"]}
        )
        feat = extract_constraint_ir_features(case)
        assert feat[1] == 1.0

    def test_semantic_majority(self) -> None:
        """REQ-CORE-001: semantic_family_majority_sat reflects semantic constraints."""
        case = _make_case(families={"semantic": ["violated", "violated"]})
        feat = extract_constraint_ir_features(case)
        assert feat[2] == 0.0

    def test_coverage_above_75(self) -> None:
        """REQ-CORE-001: coverage>0.75 feature is 1 when coverage=0.8."""
        case = _make_case(coverage=0.8)
        feat = extract_constraint_ir_features(case)
        assert feat[3] == 1.0
        assert feat[4] == 0.0  # not perfect

    def test_coverage_perfect(self) -> None:
        """REQ-CORE-001: coverage==1.0 sets both coverage features."""
        case = _make_case(coverage=1.0)
        feat = extract_constraint_ir_features(case)
        assert feat[3] == 1.0
        assert feat[4] == 1.0

    def test_coverage_low(self) -> None:
        """REQ-CORE-001: Low coverage → both coverage features 0."""
        case = _make_case(coverage=0.5)
        feat = extract_constraint_ir_features(case)
        assert feat[3] == 0.0
        assert feat[4] == 0.0

    def test_partial_above_50(self) -> None:
        """REQ-CORE-001: partial>0.5 feature is 1 when partial=0.6."""
        case = _make_case(partial=0.6)
        feat = extract_constraint_ir_features(case)
        assert feat[5] == 1.0
        assert feat[6] == 0.0  # not above 0.75

    def test_partial_above_75(self) -> None:
        """REQ-CORE-001: partial>0.75 when partial=0.9."""
        case = _make_case(partial=0.9)
        feat = extract_constraint_ir_features(case)
        assert feat[5] == 1.0
        assert feat[6] == 1.0

    def test_partial_low(self) -> None:
        """REQ-CORE-001: Low partial sat → both partial features 0."""
        case = _make_case(partial=0.3)
        feat = extract_constraint_ir_features(case)
        assert feat[5] == 0.0
        assert feat[6] == 0.0

    def test_no_violations(self) -> None:
        """REQ-CORE-001: no_violations=1 when semantic_violation_count==0."""
        case = _make_case(violations=0)
        feat = extract_constraint_ir_features(case)
        assert feat[7] == 1.0

    def test_has_violations(self) -> None:
        """REQ-CORE-001: no_violations=0 when violations > 0."""
        case = _make_case(violations=2)
        feat = extract_constraint_ir_features(case)
        assert feat[7] == 0.0

    def test_mode_verify_only(self) -> None:
        """REQ-CORE-001: verify_only mode sets feat[8]=1, feat[9]=0."""
        case = _make_case(mode="verify_only")
        feat = extract_constraint_ir_features(case)
        assert feat[8] == 1.0
        assert feat[9] == 0.0

    def test_mode_verify_repair(self) -> None:
        """REQ-CORE-001: verify_repair mode sets feat[8]=0, feat[9]=1."""
        case = _make_case(mode="verify_repair")
        feat = extract_constraint_ir_features(case)
        assert feat[8] == 0.0
        assert feat[9] == 1.0

    def test_mode_baseline(self) -> None:
        """REQ-CORE-001: baseline mode → both mode features 0."""
        case = _make_case(mode="baseline")
        feat = extract_constraint_ir_features(case)
        assert feat[8] == 0.0
        assert feat[9] == 0.0

    def test_style_code_only(self) -> None:
        """REQ-CORE-001: code_only style sets feat[10]=1."""
        case = _make_case(style="code_only")
        feat = extract_constraint_ir_features(case)
        assert feat[10] == 1.0
        assert feat[11] == 0.0

    def test_style_text_prose(self) -> None:
        """REQ-CORE-001: text_prose style sets feat[11]=1."""
        case = _make_case(style="text_prose")
        feat = extract_constraint_ir_features(case)
        assert feat[10] == 0.0
        assert feat[11] == 1.0

    def test_style_other(self) -> None:
        """REQ-CORE-001: Unknown style → both style features 0."""
        case = _make_case(style="mixed_format")
        feat = extract_constraint_ir_features(case)
        assert feat[10] == 0.0
        assert feat[11] == 0.0

    def test_model_gemma(self) -> None:
        """REQ-CORE-001: model_is_gemma=True → feat[12]=1.0."""
        case = _make_case()
        feat = extract_constraint_ir_features(case, model_is_gemma=True)
        assert feat[12] == 1.0

    def test_model_not_gemma(self) -> None:
        """REQ-CORE-001: model_is_gemma=False → feat[12]=0.0."""
        case = _make_case()
        feat = extract_constraint_ir_features(case, model_is_gemma=False)
        assert feat[12] == 0.0

    def test_missing_keys_use_defaults(self) -> None:
        """REQ-CORE-001: Missing case fields fall back to neutral defaults."""
        # Minimal case with only 'evaluation'.
        case: dict = {"evaluation": {"constraint_results": []}}
        feat = extract_constraint_ir_features(case)
        assert feat.shape == (FEATURE_DIM,)
        # Coverage=0 → both coverage bits 0.
        assert feat[3] == 0.0
        assert feat[4] == 0.0
        # violations default=1 → no_violations=0.
        assert feat[7] == 0.0
        # mode default='baseline' → both mode bits 0.
        assert feat[8] == 0.0
        assert feat[9] == 0.0

    def test_unknown_family_ignored(self) -> None:
        """REQ-CORE-001: Constraint results with unrecognised family are ignored."""
        case: dict = {
            "evaluation": {
                "constraint_results": [
                    {"family": "novel_unknown_family", "status": "satisfied"},
                    {"family": "literal", "status": "satisfied"},
                ]
            },
            "constraint_extraction_coverage": 1.0,
            "partial_satisfaction": 1.0,
            "semantic_violation_count": 0,
            "mode": "baseline",
            "output_style": "code_only",
            "exact_satisfaction": True,
        }
        feat = extract_constraint_ir_features(case)
        # literal satisfied → feat[0]=1.
        assert feat[0] == 1.0


# ---------------------------------------------------------------------------
# Unit tests: load_constraint_ir_traces
# ---------------------------------------------------------------------------


class TestLoadConstraintIrTraces:
    """REQ-CORE-001: load_constraint_ir_traces produces valid TraceRecords."""

    def test_empty_paired_runs(self, tmp_path: Path) -> None:
        """REQ-CORE-001: JSON with no paired_runs returns empty list."""
        data = {"paired_runs": []}
        p = tmp_path / "exp_empty.json"
        p.write_text(json.dumps(data))
        records = load_constraint_ir_traces(p)
        assert records == []

    def test_single_run_single_case(self, tmp_path: Path) -> None:
        """SCENARIO-CORE-001: One run, one case → one TraceRecord."""
        case = _make_case(exact=True, coverage=1.0, partial=1.0, violations=0)
        data = {
            "paired_runs": [
                {
                    "model_name": "Qwen3.5-0.8B",
                    "cases": [case],
                }
            ]
        }
        p = tmp_path / "exp_one.json"
        p.write_text(json.dumps(data))
        records = load_constraint_ir_traces(p)
        assert len(records) == 1
        assert records[0].label == 1.0
        assert records[0].features.shape == (FEATURE_DIM,)

    def test_gemma_model_detected(self, tmp_path: Path) -> None:
        """REQ-CORE-001: Gemma model name → model_is_gemma feature set."""
        case = _make_case(exact=False)
        data = {
            "paired_runs": [
                {
                    "model_name": "Gemma4-E4B-it",
                    "cases": [case],
                }
            ]
        }
        p = tmp_path / "exp_gemma.json"
        p.write_text(json.dumps(data))
        records = load_constraint_ir_traces(p)
        assert records[0].features[12] == 1.0  # model_is_gemma

    def test_qwen_model_not_gemma(self, tmp_path: Path) -> None:
        """REQ-CORE-001: Qwen model name → model_is_gemma=0."""
        case = _make_case(exact=True)
        data = {
            "paired_runs": [
                {
                    "model_name": "Qwen3.5-0.8B",
                    "cases": [case],
                }
            ]
        }
        p = tmp_path / "exp_qwen.json"
        p.write_text(json.dumps(data))
        records = load_constraint_ir_traces(p)
        assert records[0].features[12] == 0.0

    def test_multiple_runs_accumulated(self, tmp_path: Path) -> None:
        """REQ-CORE-001: Multiple runs → all cases accumulated in order."""
        case_a = _make_case(exact=True)
        case_b = _make_case(exact=False)
        data = {
            "paired_runs": [
                {"model_name": "Qwen3.5-0.8B", "cases": [case_a, case_b]},
                {"model_name": "Gemma4-E4B-it", "cases": [case_a]},
            ]
        }
        p = tmp_path / "exp_multi.json"
        p.write_text(json.dumps(data))
        records = load_constraint_ir_traces(p)
        assert len(records) == 3
        assert records[0].label == 1.0
        assert records[1].label == 0.0
        assert records[2].label == 1.0

    def test_label_false_for_unsatisfied(self, tmp_path: Path) -> None:
        """REQ-CORE-001: exact_satisfaction=False → label=0.0."""
        case = _make_case(exact=False)
        data = {"paired_runs": [{"model_name": "Qwen", "cases": [case]}]}
        p = tmp_path / "exp_false.json"
        p.write_text(json.dumps(data))
        records = load_constraint_ir_traces(p)
        assert records[0].label == 0.0


# ---------------------------------------------------------------------------
# Unit tests: auroc_score
# ---------------------------------------------------------------------------


class TestAurocScore:
    """REQ-CORE-001: auroc_score computes correct AUROC values."""

    def test_empty_correct_returns_half(self) -> None:
        """REQ-CORE-001: Empty correct energies → 0.5 (undefined)."""
        assert auroc_score(np.array([]), np.array([1.0, 2.0])) == pytest.approx(0.5)

    def test_empty_wrong_returns_half(self) -> None:
        """REQ-CORE-001: Empty wrong energies → 0.5 (undefined)."""
        assert auroc_score(np.array([1.0, 2.0]), np.array([])) == pytest.approx(0.5)

    def test_both_empty_returns_half(self) -> None:
        """REQ-CORE-001: Both empty → 0.5."""
        assert auroc_score(np.array([]), np.array([])) == pytest.approx(0.5)

    def test_perfect_separation(self) -> None:
        """REQ-CORE-001: All correct < all wrong → AUROC=1.0."""
        correct = np.array([-1.0, -2.0, -3.0])
        wrong = np.array([1.0, 2.0, 3.0])
        assert auroc_score(correct, wrong) == pytest.approx(1.0)

    def test_inverted_separation(self) -> None:
        """REQ-CORE-001: All correct > all wrong → AUROC=0.0."""
        correct = np.array([5.0, 6.0])
        wrong = np.array([1.0, 2.0])
        assert auroc_score(correct, wrong) == pytest.approx(0.0)

    def test_chance_level(self) -> None:
        """REQ-CORE-001: Identical energies → AUROC=0.5 (all ties)."""
        correct = np.array([1.0, 1.0])
        wrong = np.array([1.0, 1.0])
        assert auroc_score(correct, wrong) == pytest.approx(0.5)

    def test_partial_separation(self) -> None:
        """REQ-CORE-001: 3/4 pairs correct → AUROC=0.75."""
        correct = np.array([1.0, 3.0])
        wrong = np.array([2.0, 4.0])
        # Pairs: (1<2)✓, (1<4)✓, (3<2)✗, (3<4)✓ → 3/4 = 0.75.
        assert auroc_score(correct, wrong) == pytest.approx(0.75)

    def test_single_pair(self) -> None:
        """REQ-CORE-001: Single correct/wrong pair."""
        assert auroc_score(np.array([0.5]), np.array([1.0])) == pytest.approx(1.0)
        assert auroc_score(np.array([1.0]), np.array([0.5])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration tests: load actual Exp 221 traces and train KAN
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_traces() -> list[TraceRecord]:
    """Load Exp 221 live traces (module-scoped for speed)."""
    if not _EXP221_PATH.exists():
        pytest.skip(f"Exp 221 results not found: {_EXP221_PATH}")
    return load_constraint_ir_traces(_EXP221_PATH)


@pytest.fixture(scope="module")
def split_traces(live_traces: list[TraceRecord]) -> dict:
    """Split live traces into train/test sets (80/20) and separate correct/wrong."""
    rng = np.random.default_rng(275)
    indices = rng.permutation(len(live_traces))
    n_train = int(0.8 * len(live_traces))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train = [live_traces[i] for i in train_idx]
    test = [live_traces[i] for i in test_idx]

    train_correct = np.array(
        [r.features for r in train if r.label == 1.0], dtype=np.float32
    )
    train_wrong = np.array(
        [r.features for r in train if r.label == 0.0], dtype=np.float32
    )
    test_correct = np.array(
        [r.features for r in test if r.label == 1.0], dtype=np.float32
    )
    test_wrong = np.array(
        [r.features for r in test if r.label == 0.0], dtype=np.float32
    )

    # Balance training: subsample wrong to match correct count.
    n_min = min(len(train_correct), len(train_wrong))
    rng2 = np.random.default_rng(275)
    train_correct = train_correct[rng2.choice(len(train_correct), n_min, replace=False)]
    train_wrong = train_wrong[rng2.choice(len(train_wrong), n_min, replace=False)]

    return {
        "train_correct": train_correct,
        "train_wrong": train_wrong,
        "test_correct": test_correct,
        "test_wrong": test_wrong,
        "n_total": len(live_traces),
        "n_train": len(train),
        "n_test": len(test),
    }


class TestLiveTraceLoading:
    """REQ-CORE-001: Live Exp 221 traces load with correct structure."""

    def test_trace_count(self, live_traces: list[TraceRecord]) -> None:
        """SCENARIO-CORE-001: 486 traces loaded (6 runs × 81 cases)."""
        assert len(live_traces) == 486

    def test_feature_dim_correct(self, live_traces: list[TraceRecord]) -> None:
        """REQ-CORE-001: All features have shape (FEATURE_DIM,)."""
        for rec in live_traces[:10]:
            assert rec.features.shape == (FEATURE_DIM,)

    def test_labels_binary(self, live_traces: list[TraceRecord]) -> None:
        """REQ-CORE-001: All labels are 0.0 or 1.0."""
        for rec in live_traces:
            assert rec.label in (0.0, 1.0)

    def test_correct_wrong_counts(self, live_traces: list[TraceRecord]) -> None:
        """REQ-CORE-001: Both correct and wrong examples present."""
        n_correct = sum(1 for r in live_traces if r.label == 1.0)
        n_wrong = sum(1 for r in live_traces if r.label == 0.0)
        assert n_correct > 0
        assert n_wrong > 0


class TestStaticKANOnLiveTraces:
    """REQ-CORE-001: Static KAN learns to discriminate constraint satisfaction."""

    def test_auroc_above_chance(self, split_traces: dict) -> None:
        """SCENARIO-CORE-001: Static KAN AUROC > 0.5 after training on live traces."""
        tc = split_traces["train_correct"]
        tw = split_traces["train_wrong"]
        ec = split_traces["test_correct"]
        ew = split_traces["test_wrong"]

        model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
        )
        model.train_discriminative_cd(
            tc, tw, n_epochs=80, lr=0.02, verbose=False
        )

        e_correct = model.energy_batch(ec)
        e_wrong = model.energy_batch(ew)
        auroc = auroc_score(e_correct, e_wrong)

        # Static KAN should beat random chance on real data.
        assert auroc > 0.5, f"Static KAN AUROC={auroc:.3f} failed to beat chance"

    def test_energy_gap_positive(self, split_traces: dict) -> None:
        """REQ-CORE-001: Mean energy gap (E_wrong - E_correct) > 0 after training."""
        tc = split_traces["train_correct"]
        tw = split_traces["train_wrong"]
        ec = split_traces["test_correct"]
        ew = split_traces["test_wrong"]

        model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
        )
        model.train_discriminative_cd(tc, tw, n_epochs=80, lr=0.02)

        e_correct = model.energy_batch(ec)
        e_wrong = model.energy_batch(ew)
        gap = float(np.mean(e_wrong) - np.mean(e_correct))
        assert gap > 0, f"Energy gap {gap:.4f} not positive"


class TestAdaptiveKANOnLiveTraces:
    """REQ-TIER-001: AdaptiveKAN with AMR maintains AUROC on live traces."""

    def test_adaptive_auroc_does_not_regress(self, split_traces: dict) -> None:
        """SCENARIO-TIER-004: AMR does not regress AUROC by more than 0.05."""
        tc = split_traces["train_correct"]
        tw = split_traces["train_wrong"]
        ec = split_traces["test_correct"]
        ew = split_traces["test_wrong"]

        # Train static model for baseline AUROC.
        static_model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
        )
        static_model.train_discriminative_cd(tc, tw, n_epochs=80, lr=0.02)
        e_c_static = static_model.energy_batch(ec)
        e_w_static = static_model.energy_batch(ew)
        auroc_static = auroc_score(e_c_static, e_w_static)

        # Train adaptive model with same init, then trigger AMR via test traces.
        adaptive_model = AdaptiveKAN(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
            restructure_every=50,  # triggers at 50 verifications
        )
        adaptive_model.train_discriminative_cd(tc, tw, n_epochs=80, lr=0.02)

        # Feed test traces through verify_and_maybe_restructure — this triggers AMR.
        all_test = np.vstack([ec, ew])
        for i in range(len(all_test)):
            adaptive_model.verify_and_maybe_restructure(all_test[i])

        # Fine-tune for 5 epochs post-AMR.
        adaptive_model.train_discriminative_cd(tc, tw, n_epochs=5, lr=0.005)

        e_c_adaptive = adaptive_model.energy_batch(ec)
        e_w_adaptive = adaptive_model.energy_batch(ew)
        auroc_adaptive = auroc_score(e_c_adaptive, e_w_adaptive)

        # AMR should not catastrophically regress.
        assert auroc_adaptive >= auroc_static - 0.05, (
            f"AUROC regressed: static={auroc_static:.3f} → adaptive={auroc_adaptive:.3f}"
        )

    def test_amr_triggers_at_least_once(self, split_traces: dict) -> None:
        """REQ-TIER-001: AMR fires at least once when fed enough test traces."""
        tc = split_traces["train_correct"]
        tw = split_traces["train_wrong"]
        ec = split_traces["test_correct"]
        ew = split_traces["test_wrong"]

        adaptive_model = AdaptiveKAN(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
            restructure_every=50,
        )
        adaptive_model.train_discriminative_cd(tc, tw, n_epochs=20, lr=0.02)

        # Feed enough test traces to trigger at least one AMR cycle.
        all_test = np.vstack([ec, ew])
        for i in range(len(all_test)):
            adaptive_model.verify_and_maybe_restructure(all_test[i])

        # Should have fired AMR at least once (n_test ≈ 97 > restructure_every=50).
        assert len(adaptive_model._curvature_history) >= 1


# ---------------------------------------------------------------------------
# End-to-end: write experiment_275_results.json
# ---------------------------------------------------------------------------


class TestWriteExp275Results:
    """REQ-CORE-001: Produce experiment_275_results.json with full metrics."""

    def test_write_results_json(self, split_traces: dict) -> None:
        """SCENARIO-CORE-001: Full pipeline writes valid results JSON."""
        tc = split_traces["train_correct"]
        tw = split_traces["train_wrong"]
        ec = split_traces["test_correct"]
        ew = split_traces["test_wrong"]

        # --- Static KAN ---
        static_model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
        )
        losses_static = static_model.train_discriminative_cd(
            tc, tw, n_epochs=80, lr=0.02
        )
        e_c_static = static_model.energy_batch(ec)
        e_w_static = static_model.energy_batch(ew)
        auroc_static = auroc_score(e_c_static, e_w_static)

        # --- Adaptive KAN ---
        adaptive_model = AdaptiveKAN(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
            restructure_every=50,
        )
        losses_adaptive = adaptive_model.train_discriminative_cd(
            tc, tw, n_epochs=80, lr=0.02
        )
        n_params_before_amr = adaptive_model.n_params

        # Feed all test vectors through verify loop (triggers AMR).
        all_test = np.vstack([ec, ew])
        restructure_events = 0
        for i in range(len(all_test)):
            _, restructured = adaptive_model.verify_and_maybe_restructure(all_test[i])
            if restructured:
                restructure_events += 1

        # Fine-tune post-AMR.
        adaptive_model.train_discriminative_cd(tc, tw, n_epochs=5, lr=0.005)
        n_params_after_amr = adaptive_model.n_params

        e_c_adaptive = adaptive_model.energy_batch(ec)
        e_w_adaptive = adaptive_model.energy_batch(ew)
        auroc_adaptive = auroc_score(e_c_adaptive, e_w_adaptive)

        # Build results dict.
        results = {
            "experiment": 275,
            "title": "KAN adaptive verification on live Exp 219-221 traces",
            "source_experiments": [219, 220, 221],
            "data": {
                "total_traces": split_traces["n_total"],
                "n_train": split_traces["n_train"],
                "n_test": split_traces["n_test"],
                "n_train_correct": len(tc),
                "n_train_wrong": len(tw),
                "n_test_correct": len(ec),
                "n_test_wrong": len(ew),
            },
            "static_kan": {
                "auroc": float(auroc_static),
                "n_params": static_model.n_params,
                "n_epochs": 80,
                "final_loss": float(losses_static[-1]) if losses_static else None,
            },
            "adaptive_kan": {
                "auroc_after_amr": float(auroc_adaptive),
                "auroc_improvement": float(auroc_adaptive - auroc_static),
                "n_params_before_amr": n_params_before_amr,
                "n_params_after_amr": n_params_after_amr,
                "param_delta": n_params_after_amr - n_params_before_amr,
                "restructure_events": restructure_events,
                "curvature_history": adaptive_model._curvature_history,
                "n_epochs_pretrain": 80,
                "n_epochs_finetune": 5,
                "final_loss": float(losses_adaptive[-1]) if losses_adaptive else None,
            },
            "feature_dim": FEATURE_DIM,
        }

        # Write JSON.
        _RESULTS_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(_RESULTS_OUT, "w") as fh:
            json.dump(results, fh, indent=2)

        # Assertions on the written file.
        assert _RESULTS_OUT.exists()
        with open(_RESULTS_OUT) as fh:
            loaded = json.load(fh)

        assert loaded["experiment"] == 275
        assert loaded["static_kan"]["auroc"] > 0.5
        assert loaded["adaptive_kan"]["restructure_events"] >= 1
        assert (
            loaded["adaptive_kan"]["auroc_after_amr"]
            >= loaded["static_kan"]["auroc"] - 0.05
        ), (
            f"AUROC regression in results JSON: "
            f"{loaded['static_kan']['auroc']:.3f} → "
            f"{loaded['adaptive_kan']['auroc_after_amr']:.3f}"
        )


# ---------------------------------------------------------------------------
# Tests: Checkpoint/restore functionality
# ---------------------------------------------------------------------------


class TestCheckpointRestore:
    """REQ-CORE-001: Checkpoint and restore preserve model state exactly."""

    def test_checkpoint_creates_files(self, tmp_path: Path) -> None:
        """REQ-CORE-001: checkpoint() writes .safetensors and .json files."""
        model = AdaptiveKAN(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
        )
        checkpoint_path = tmp_path / "test_model.safetensors"
        model.checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()
        json_path = tmp_path / "test_model.json"
        assert json_path.exists()

    def test_adaptive_kan_checkpoint_restore_roundtrip(
        self, split_traces: dict, tmp_path: Path
    ) -> None:
        """REQ-CORE-001: Checkpoint → restore → verify yields identical model."""
        tc = split_traces["train_correct"]
        tw = split_traces["train_wrong"]

        # Train original model.
        model = AdaptiveKAN(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
            restructure_every=50,
        )
        model.train_discriminative_cd(tc, tw, n_epochs=10, lr=0.02)

        # Feed test data to trigger AMR.
        for i in range(len(tc)):
            model.verify_and_maybe_restructure(tc[i])

        # Save checkpoint.
        checkpoint_path = tmp_path / "model.safetensors"
        model.checkpoint(str(checkpoint_path))

        # Restore.
        restored = AdaptiveKAN.from_checkpoint(str(checkpoint_path))

        # Verify state matches.
        assert restored.input_dim == model.input_dim
        assert restored.num_knots == model.num_knots
        assert restored.degree == model.degree
        assert restored.restructure_every == model.restructure_every
        assert restored._verification_count == model._verification_count
        assert len(restored._curvature_history) == len(model._curvature_history)

    def test_checkpoint_restore_energy_deterministic(
        self, split_traces: dict, tmp_path: Path
    ) -> None:
        """REQ-CORE-001: Restored model produces identical energies."""
        tc = split_traces["train_correct"]
        tw = split_traces["train_wrong"]

        # Train and checkpoint.
        model = AdaptiveKAN(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
            restructure_every=50,
        )
        model.train_discriminative_cd(tc, tw, n_epochs=10, lr=0.02)

        # Test energies on original.
        test_vec = tc[0]
        e_orig = model.energy_single(test_vec)

        # Checkpoint and restore.
        checkpoint_path = tmp_path / "model.safetensors"
        model.checkpoint(str(checkpoint_path))
        restored = AdaptiveKAN.from_checkpoint(str(checkpoint_path))

        # Test energies on restored.
        e_restored = restored.energy_single(test_vec)

        # Must match exactly (within floating-point tolerance).
        assert e_orig == pytest.approx(e_restored, abs=1e-6)

    def test_checkpoint_with_recent_inputs(
        self, split_traces: dict, tmp_path: Path
    ) -> None:
        """REQ-CORE-001: Recent inputs buffer survives checkpoint/restore."""
        tc = split_traces["train_correct"]
        tw = split_traces["train_wrong"]

        model = AdaptiveKAN(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
            restructure_every=50,
        )
        model.train_discriminative_cd(tc, tw, n_epochs=10, lr=0.02)

        # Feed some data to fill the buffer.
        for i in range(30):
            model.verify_and_maybe_restructure(tc[i])

        n_inputs_before = len(model._recent_inputs)
        assert n_inputs_before > 0

        # Checkpoint and restore.
        checkpoint_path = tmp_path / "model.safetensors"
        model.checkpoint(str(checkpoint_path))
        restored = AdaptiveKAN.from_checkpoint(str(checkpoint_path))

        # Buffer should be restored.
        assert len(restored._recent_inputs) == n_inputs_before
        for i in range(n_inputs_before):
            np.testing.assert_array_equal(
                restored._recent_inputs[i], model._recent_inputs[i]
            )


# ---------------------------------------------------------------------------
# Tests: Curvature computation with and without sample points
# ---------------------------------------------------------------------------


class TestCurvatureComputation:
    """REQ-CORE-001: Curvature computation works with and without sample points."""

    def test_compute_edge_curvature_uniform_grid(self) -> None:
        """REQ-CORE-001: compute_edge_curvature uses uniform grid when no sample_pts."""
        model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
        )
        # Call without sample_pts — triggers uniform grid path (line 269).
        curvatures = model.compute_edge_curvature(n_sample=50, h=0.01)

        assert len(curvatures) == len(model.edges)
        for edge, curv in curvatures.items():
            assert isinstance(curv, float)
            assert curv >= 0.0

    def test_compute_edge_curvature_with_sample_pts(self) -> None:
        """REQ-CORE-001: compute_edge_curvature uses provided sample_pts."""
        model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
        )
        # Provide custom sample points.
        sample_pts = {edge: np.linspace(-0.8, 0.8, 20) for edge in model.edges}
        curvatures = model.compute_edge_curvature(sample_pts=sample_pts)

        assert len(curvatures) == len(model.edges)
        for edge, curv in curvatures.items():
            assert isinstance(curv, float)
            assert curv >= 0.0


# ---------------------------------------------------------------------------
# Tests: Knot removal behavior
# ---------------------------------------------------------------------------


class TestKnotRemoval:
    """REQ-CORE-001: _remove_knot respects minimum control-point guard."""

    def test_remove_knot_respects_minimum(self) -> None:
        """REQ-CORE-001: Cannot remove knot if already at min (degree + 2)."""
        model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=4,  # num_knots + degree = 4 + 2 = 6 (minimum)
            degree=2,
            seed=275,
        )
        # Pick an edge and trim its control points to minimum.
        edge = model.edges[0]
        min_ctrl = model.degree + 2
        model.edge_control_pts[edge] = np.ones(min_ctrl, dtype=np.float32)
        model._edge_n_ctrl[edge] = min_ctrl

        # Try to remove — should fail (return False).
        removed = model._remove_knot(edge)
        assert removed is False

    def test_remove_knot_removes_linear_segment(self) -> None:
        """REQ-CORE-001: _remove_knot removes most-linear adjacent pair."""
        model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=8,
            degree=2,
            seed=275,
        )
        edge = model.edges[0]
        # Create a control point array with one nearly-linear segment.
        # [1.0, 1.0, 1.5, 5.0, 5.2] — the first two are most linear (diff=0.0).
        test_ctrl = np.array([1.0, 1.0, 1.5, 5.0, 5.2], dtype=np.float32)
        model.edge_control_pts[edge] = test_ctrl
        model._edge_n_ctrl[edge] = len(test_ctrl)

        # Remove one knot.
        removed = model._remove_knot(edge)
        assert removed is True
        # The segment [1.0, 1.0] should be merged to 1.0.
        assert len(model.edge_control_pts[edge]) == 4
        np.testing.assert_array_almost_equal(model.edge_control_pts[edge], [1.0, 1.5, 5.0, 5.2])


# ---------------------------------------------------------------------------
# Tests: Verbose training output
# ---------------------------------------------------------------------------


class TestVerboseTraining:
    """REQ-CORE-001: train_discriminative_cd logs when verbose=True."""

    def test_train_with_verbose_output(self, split_traces: dict, capsys) -> None:
        """REQ-CORE-001: Verbose training outputs epoch logs."""
        tc = split_traces["train_correct"][:50]
        tw = split_traces["train_wrong"][:50]

        model = KANConstraintModel(
            input_dim=FEATURE_DIM,
            num_knots=6,
            degree=2,
            seed=275,
        )
        # Train with verbose output.
        losses = model.train_discriminative_cd(
            tc, tw, n_epochs=30, lr=0.02, verbose=True
        )

        # Losses should be produced (every 25 epochs + last epoch).
        assert len(losses) == 30
        assert all(isinstance(l, float) for l in losses)

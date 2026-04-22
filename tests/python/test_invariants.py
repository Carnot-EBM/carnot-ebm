"""Tests for carnot.invariants — the machine-checkable invariant system.

Each invariant has at least one passing case (should not flag) and one
failing case (should flag with a substitute verdict).  Failing-case reasons
are spot-checked for substring content so future prose tweaks don't break
the test.

Spec: REQ-SAFE-011 (distillation real-teacher invariant), plus three
derived invariants for the .52 retraction patterns (publishable-has-TP,
ood-vs-indist, vr-baseline-plausibility).
"""

from __future__ import annotations

from carnot.invariants import (
    InvariantResult,
    check_distillation_has_real_teacher_time,
    check_labeler_agreement_nonzero,
    check_ood_not_dramatically_better_than_indist,
    check_publishable_has_nonzero_tp,
    check_vr_positive_has_plausible_baseline,
    run_invariants,
)


# ---------------------------------------------------------------------------
# Invariant 1: distillation requires real teacher time
# ---------------------------------------------------------------------------


class TestDistillationHasRealTeacherTime:
    def test_non_distillation_verdict_passes(self) -> None:
        """REQ-SAFE-011: invariant does not fire on non-distillation verdicts."""
        artifact = {
            "honest_verdict": "vr_positive",
            "teacher_inference_duration_s": 0.0,  # 0s is fine here — not claimed
        }
        result = check_distillation_has_real_teacher_time(artifact)
        assert result.passed is True

    def test_blocked_distillation_verdict_passes(self) -> None:
        """REQ-SAFE-011: 'blocked_on_*' sub-verdicts do not retrigger the check."""
        artifact = {
            "honest_verdict": "blocked_on_dependency",
            # No teacher_inference_duration_s — fine because run was blocked.
        }
        result = check_distillation_has_real_teacher_time(artifact)
        assert result.passed is True

    def test_real_distillation_passes_invariant(self) -> None:
        """REQ-SAFE-011: Exp 690's actual numbers pass the invariant.

        Exp 690: 200 prompts (n_train=160, n_test=40), teacher ran 6256.2 s.
        Floor = max(200 * 0.5, 100) = 100.  6256 >> 100, so this passes.
        """
        artifact = {
            "honest_verdict": "distillation_corpus_built_classifier_trained_auroc_below_threshold",
            "n_train": 160,
            "n_test": 40,
            "teacher_inference_duration_s": 6256.2,
        }
        result = check_distillation_has_real_teacher_time(artifact)
        assert result.passed is True

    def test_fake_distillation_fails_invariant(self) -> None:
        """REQ-SAFE-011: Exp 669's actual numbers FAIL the invariant.

        Exp 669: claimed distillation, duration_s=16.84, n_pairs=200.
        Floor = max(200 * 0.5, 100) = 100.  16.84 << 100, so this fails.
        """
        artifact = {
            "honest_verdict": "distillation_corpus_built_classifier_trained_auroc_below_threshold",
            "n_pairs": 200,
            "teacher_inference_duration_s": 16.84,  # physically impossible
        }
        result = check_distillation_has_real_teacher_time(artifact)
        assert result.passed is False
        assert result.suggested_verdict is not None
        assert "invariant_violated" in result.suggested_verdict
        assert "teacher_too_fast" in result.suggested_verdict
        assert result.reason is not None
        assert "16.8" in result.reason
        assert "100" in result.reason

    def test_missing_teacher_time_field_fails_invariant(self) -> None:
        """REQ-SAFE-011: a distillation verdict without the duration field fails."""
        artifact = {
            "honest_verdict": "distillation_corpus_built_classifier_trained_auroc_met",
            "n_train": 160,
            "n_test": 40,
            # teacher_inference_duration_s missing entirely
        }
        result = check_distillation_has_real_teacher_time(artifact)
        assert result.passed is False
        assert result.suggested_verdict is not None
        assert "no_teacher_time_field" in result.suggested_verdict

    def test_corpus_size_fallback_to_n_pairs(self) -> None:
        """REQ-SAFE-011: when n_train/n_test missing, n_pairs is used."""
        artifact = {
            "honest_verdict": "distillation_corpus_built_classifier_trained_auroc_met",
            "n_pairs": 500,
            "teacher_inference_duration_s": 50.0,  # < max(500*0.5, 100) = 250 floor
        }
        result = check_distillation_has_real_teacher_time(artifact)
        assert result.passed is False
        assert result.evidence.get("corpus_size") == 500
        assert result.evidence.get("floor_s") == 250.0


# ---------------------------------------------------------------------------
# Invariant 2: publishable requires non-zero TP
# ---------------------------------------------------------------------------


class TestPublishableHasNonzeroTp:
    def test_non_publishable_verdict_passes(self) -> None:
        """Invariant does not fire on verdicts that don't claim publishability."""
        artifact = {"honest_verdict": "generalization_partial_shareable_with_caveat"}
        result = check_publishable_has_nonzero_tp(artifact)
        assert result.passed is True

    def test_publishable_with_tp_passes(self) -> None:
        """A publishable verdict with at least one TP on one dataset passes."""
        artifact = {
            "honest_verdict": "generalization_verified_publishable",
            "per_dataset_cm": {
                "dataset_a": {"tp": 10, "fp": 2, "tn": 50, "fn": 3},
                "dataset_b": {"tp": 0, "fp": 0, "tn": 20, "fn": 20},
            },
        }
        result = check_publishable_has_nonzero_tp(artifact)
        assert result.passed is True

    def test_publishable_with_zero_tp_everywhere_fails(self) -> None:
        """Exp 691 exact pattern: AUROC high but zero TPs anywhere."""
        artifact = {
            "honest_verdict": "generalization_verified_publishable",
            "mean_auroc": 0.9585,
            "per_dataset_cm": {
                "hackaprompt": {"tp": 0, "fp": 0, "tn": 250, "fn": 250},
                "bipia":       {"tp": 0, "fp": 0, "tn": 200, "fn": 200},
                "synthetic":   {"tp": 0, "fp": 0, "tn": 100, "fn": 100},
            },
        }
        result = check_publishable_has_nonzero_tp(artifact)
        assert result.passed is False
        assert result.suggested_verdict is not None
        assert "zero_true_positives" in result.suggested_verdict
        assert result.evidence.get("total_tp") == 0
        assert result.reason is not None
        assert "zero true positives" in result.reason

    def test_publishable_missing_cm_fails(self) -> None:
        """Can't evaluate publishability without confusion matrices."""
        artifact = {"honest_verdict": "generalization_verified_publishable"}
        result = check_publishable_has_nonzero_tp(artifact)
        assert result.passed is False
        assert "no_confusion_matrix" in (result.suggested_verdict or "")


# ---------------------------------------------------------------------------
# Invariant 3: OOD AUROC cannot dramatically beat in-distribution AUROC
# ---------------------------------------------------------------------------


class TestOodNotDramaticallyBetterThanIndist:
    def test_no_indist_field_passes(self) -> None:
        """Invariant does not apply when training-distribution AUROC is missing."""
        artifact = {"mean_cross_dataset_auroc": 0.95}
        result = check_ood_not_dramatically_better_than_indist(artifact)
        assert result.passed is True

    def test_ood_matches_indist_passes(self) -> None:
        """Equal AUROCs are fine — no generalization claim is made."""
        artifact = {
            "training_distribution_auroc": 0.85,
            "mean_cross_dataset_auroc": 0.83,
        }
        result = check_ood_not_dramatically_better_than_indist(artifact)
        assert result.passed is True

    def test_ood_slightly_better_within_tolerance_passes(self) -> None:
        """Up to 0.05 excess is allowed — corpus-selection noise."""
        artifact = {
            "training_distribution_auroc": 0.80,
            "mean_cross_dataset_auroc": 0.84,  # +0.04, within tolerance
        }
        result = check_ood_not_dramatically_better_than_indist(artifact)
        assert result.passed is True

    def test_ood_dramatically_better_fails(self) -> None:
        """Exp 691's exact shape: in-dist 0.80, OOD 0.96 = +0.16 excess."""
        artifact = {
            "honest_verdict": "generalization_verified_publishable",
            "training_distribution_auroc": 0.7995,
            "mean_cross_dataset_auroc": 0.9585,
        }
        result = check_ood_not_dramatically_better_than_indist(artifact)
        assert result.passed is False
        assert result.suggested_verdict is not None
        assert "ood_exceeds_indist" in result.suggested_verdict
        assert result.reason is not None
        assert "physically implausible" in result.reason


# ---------------------------------------------------------------------------
# Invariant 4: VR positive requires plausible baseline
# ---------------------------------------------------------------------------


class TestVrPositiveHasPlausibleBaseline:
    def test_non_vr_verdict_passes(self) -> None:
        """Not a VR verdict — invariant doesn't fire."""
        artifact = {
            "honest_verdict": "dualgpu_confirmed",
            "baseline_accuracy": 0.0,  # fine, not being claimed as VR
        }
        result = check_vr_positive_has_plausible_baseline(artifact)
        assert result.passed is True

    def test_vr_positive_with_plausible_baseline_passes(self) -> None:
        """Exp 668 initial: baseline 0.36 (9/25). Passes plausibility floor."""
        artifact = {
            "honest_verdict": "vr_positive",
            "baseline_accuracy": 0.36,
            "n_questions": 25,
        }
        result = check_vr_positive_has_plausible_baseline(artifact)
        assert result.passed is True

    def test_vr_positive_with_zero_baseline_fails(self) -> None:
        """Exp 679 exact shape: baseline 0.0 on 200 GSM8K questions."""
        artifact = {
            "honest_verdict": "vr_200q_positive",
            "baseline_accuracy": 0.0,
            "post_accuracy": 1.0,
            "n_questions": 200,
        }
        result = check_vr_positive_has_plausible_baseline(artifact)
        assert result.passed is False
        assert result.suggested_verdict is not None
        assert "baseline_implausibly_low" in result.suggested_verdict
        assert result.reason is not None
        assert "0.000" in result.reason or "broken" in result.reason

    def test_vr_verdict_without_baseline_field_passes(self) -> None:
        """Invariant is permissive when baseline_accuracy is missing.

        Another check (not this one) should require the field — we don't
        want to chain false positives from a missing field.
        """
        artifact = {"honest_verdict": "vr_positive"}
        result = check_vr_positive_has_plausible_baseline(artifact)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Runner: run_invariants aggregates failures across all registered checks
# ---------------------------------------------------------------------------


class TestLabelerAgreementNonzero:
    """Tests for the fifth invariant: two labelers measuring the same thing
    must agree on at least 5% of samples or the combined corpus is noise."""

    def test_no_agreement_field_passes(self) -> None:
        """Invariant does not apply when no agreement field is present."""
        artifact = {"honest_verdict": "fover_v1_baseline_ok"}
        result = check_labeler_agreement_nonzero(artifact)
        assert result.passed is True

    def test_high_agreement_passes(self) -> None:
        """Exp 690's actual teacher_vs_source_agreement=0.965 passes."""
        artifact = {
            "honest_verdict": "distillation_corpus_built_classifier_trained_auroc_below_threshold",
            "teacher_vs_source_agreement_rate": 0.965,
        }
        result = check_labeler_agreement_nonzero(artifact)
        assert result.passed is True

    def test_exp712_zero_agreement_fails(self) -> None:
        """Exp 712's exact shape: pddl_z3_agreement_rate = 0.0 on 1400 pairs.
        The verdict claimed "fover_v2_target_met" but the corpus was junk."""
        artifact = {
            "honest_verdict": "fover_v2_target_met",
            "n_z3_pairs": 200,
            "n_pddl_pairs": 1200,
            "pddl_z3_agreement_rate": 0.0,
        }
        result = check_labeler_agreement_nonzero(artifact)
        assert result.passed is False
        assert result.suggested_verdict is not None
        assert "corpus_is_noise" in result.suggested_verdict
        assert result.reason is not None
        assert "PDDL" in result.reason and "Z3" in result.reason

    def test_below_threshold_verdict_is_not_reflagged(self) -> None:
        """If the verdict already names a negative outcome, don't double-flag."""
        artifact = {
            "honest_verdict": "sc_energy_v2_below_threshold",
            "labeler_agreement_rate": 0.0,
        }
        result = check_labeler_agreement_nonzero(artifact)
        assert result.passed is True

    def test_non_numeric_agreement_is_ignored(self) -> None:
        """If the field has an unparseable value, invariant does not fire."""
        artifact = {"teacher_vs_source_agreement_rate": "unknown"}
        result = check_labeler_agreement_nonzero(artifact)
        assert result.passed is True


class TestRunInvariants:
    def test_clean_artifact_returns_no_violations(self) -> None:
        """Exp 690 actual numbers — all invariants pass."""
        artifact = {
            "honest_verdict": "distillation_corpus_built_classifier_trained_auroc_below_threshold",
            "n_train": 160, "n_test": 40,
            "teacher_inference_duration_s": 6256.2,
        }
        assert run_invariants(artifact) == []

    def test_exp691_artifact_triggers_two_invariants(self) -> None:
        """Exp 691 fails both publishable-has-TP and ood-vs-indist."""
        artifact = {
            "honest_verdict": "generalization_verified_publishable",
            "training_distribution_auroc": 0.7995,
            "mean_cross_dataset_auroc": 0.9585,
            "per_dataset_cm": {
                "hackaprompt": {"tp": 0, "fp": 0, "tn": 250, "fn": 250},
                "bipia":       {"tp": 0, "fp": 0, "tn": 200, "fn": 200},
                "synthetic":   {"tp": 0, "fp": 0, "tn": 100, "fn": 100},
            },
        }
        violations = run_invariants(artifact)
        assert len(violations) == 2
        names = {v.invariant_name for v in violations}
        assert "publishable_has_nonzero_tp" in names
        assert "ood_not_dramatically_better_than_indist" in names

    def test_exp679_artifact_triggers_one_invariant(self) -> None:
        """Exp 679 shape fails the VR-baseline-plausibility invariant only."""
        artifact = {
            "honest_verdict": "vr_200q_positive",
            "baseline_accuracy": 0.0,
            "post_accuracy": 1.0,
            "n_questions": 200,
        }
        violations = run_invariants(artifact)
        assert len(violations) == 1
        assert violations[0].invariant_name == "vr_positive_has_plausible_baseline"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


class TestInvariantResultAsDict:
    def test_as_dict_roundtrips(self) -> None:
        result = InvariantResult(
            passed=False,
            invariant_name="test_check",
            reason="something",
            suggested_verdict="foo_invariant_violated_test",
            evidence={"a": 1},
        )
        d = result.as_dict()
        assert d["passed"] is False
        assert d["invariant_name"] == "test_check"
        assert d["reason"] == "something"
        assert d["suggested_verdict"] == "foo_invariant_violated_test"
        assert d["evidence"] == {"a": 1}

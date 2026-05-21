"""Tests for the Exp 2826 milestone .267 multi-corpus capstone synthesis.

WHY a dedicated test file: the .267 capstone must degrade gracefully under
two distinct conditions — (a) most upstream experiments failed (Gemini CLI
crash storm) and (b) some experiments ran but were adversarially flagged.
Tests verify both SCENARIO-PUBLISH-032 (nominal) and SCENARIO-PUBLISH-032B
(degraded) as well as the individual utility functions.

Spec refs: REQ-PUBLISH-032, SCENARIO-PUBLISH-032, SCENARIO-PUBLISH-032B.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v267_2826 as exp2826


# ---------------------------------------------------------------------------
# Helper: write a JSON file into a tmp_path tree
# ---------------------------------------------------------------------------

def _write(root: Path, rel: str, payload: dict) -> None:
    """Serialise *payload* as JSON at root/rel, creating parent dirs."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture helpers: canonical artifact shapes for each upstream experiment
# ---------------------------------------------------------------------------

def _exp2820_clean(learning_contribution: float = 0.12) -> dict:
    """A clean, adversarially-verified exp2820 (FoVer leakage isolation) artifact."""
    return {
        "honest_verdict": "complete: FoVer leakage isolation measured",
        "condition_a_production_auroc_mean": 0.9857,
        "condition_a_production_auroc_std": 0.0175,
        "condition_b_architecture_only_auroc_mean": 0.87,
        "condition_b_architecture_only_auroc_std": 0.02,
        "learning_contribution": learning_contribution,
        "flagged_adversarial": False,
        "duration_s": 620.3,
    }


def _exp2821_clean(arch_only: float = 0.72) -> dict:
    """A clean exp2821 (MBPP) artifact."""
    return {
        "honest_verdict": "complete: MBPP dual-condition measured",
        "condition_a_production_auroc_mean": 0.70,
        "condition_a_production_auroc_std": 0.03,
        "condition_b_architecture_only_auroc_mean": arch_only,
        "condition_b_architecture_only_auroc_std": 0.03,
        "learning_contribution": -0.02,
        "flagged_adversarial": False,
        "duration_s": 740.1,
    }


def _exp2822_clean(arch_only: float = 0.68) -> dict:
    """A clean exp2822 (HumanEval) artifact."""
    return {
        "honest_verdict": "complete: HumanEval dual-condition measured",
        "condition_a_production_auroc_mean": 0.67,
        "condition_a_production_auroc_std": 0.03,
        "condition_b_architecture_only_auroc_mean": arch_only,
        "condition_b_architecture_only_auroc_std": 0.03,
        "learning_contribution": -0.01,
        "flagged_adversarial": False,
        "duration_s": 810.2,
    }


def _exp2823_clean() -> dict:
    """A clean exp2823 (TruthfulQA) artifact."""
    return {
        "honest_verdict": "complete: TruthfulQA dual-condition measured",
        "condition_a_production_auroc_mean": 0.68,
        "condition_a_production_auroc_std": 0.02,
        "condition_b_architecture_only_auroc_mean": 0.69,
        "condition_b_architecture_only_auroc_std": 0.02,
        "learning_contribution": -0.01,
        "flagged_adversarial": False,
        "duration_s": 610.5,
    }


def _exp2823_flagged() -> dict:
    """The adversarially-flagged exp2823 that exists in .267."""
    return {
        "honest_verdict": "complete: TruthfulQA measured",
        "condition_a_production_auroc_mean": 0.68,
        "condition_b_architecture_only_auroc_mean": 0.69,
        "learning_contribution": -0.01,
        "flagged_adversarial": True,
        "duration_s": 9.58e-05,
        "corrigendum_pending": [
            {"kind": "DURATION_TOO_SHORT", "severity": "critical"},
        ],
    }


def _exp2824_real() -> dict:
    """A plausible, non-placeholder exp2824 (cross-corpus matrix) artifact."""
    return {
        "honest_verdict": "complete: Matrix generated successfully",
        "verifier_corpus_dual_matrix": {
            "tier_transfer": {
                "FoVer":       {"production": 0.81, "architecture_only": 0.80, "delta": 0.01},
                "MBPP":        {"production": 0.79, "architecture_only": 0.78, "delta": 0.01},
                "HumanEval":   {"production": 0.76, "architecture_only": 0.75, "delta": 0.01},
                "TruthfulQA":  {"production": 0.63, "architecture_only": 0.64, "delta": -0.01},
            },
            "tier_memory": {
                "FoVer":       {"production": 0.85, "architecture_only": 0.60, "delta": 0.25},
                "MBPP":        {"production": 0.55, "architecture_only": 0.54, "delta": 0.01},
                "HumanEval":   {"production": 0.53, "architecture_only": 0.52, "delta": 0.01},
                "TruthfulQA":  {"production": 0.51, "architecture_only": 0.51, "delta": 0.00},
            },
        },
        "architecture_transfer_verifiers": ["tier_transfer"],
        "memory_augmented_verifiers": ["tier_memory"],
        "corpus_specific_verifiers": ["tier_specific", "tier0r"],
        "low_signal_verifiers": ["tier_low"],
        "flagged_adversarial": False,
        "duration_s": 45.3,
    }


def _exp2824_placeholder() -> dict:
    """The placeholder-like exp2824 that exists in .267 — mostly 0.0/0.5 values.

    This fixture mirrors the real exp2824_cross_corpus_verifier_matrix.json from
    the .267 milestone: tier0r and tier0s show 0.0 for FoVer/MBPP/HumanEval
    because those corpora were never actually run against those verifiers.
    The tier_low verifier has uniform 0.5 across all corpora (chance baseline).
    These patterns indicate placeholder / absent data.
    """
    return {
        "honest_verdict": "complete: Matrix generated successfully",
        "verifier_corpus_dual_matrix": {
            "tier_specific": {
                "FoVer":      {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
                "MBPP":       {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
                "HumanEval":  {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
                "TruthfulQA": {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
            },
            "tier_transfer": {
                "FoVer":      {"production": 0.8,  "architecture_only": 0.8,  "delta": 0.0},
                "MBPP":       {"production": 0.8,  "architecture_only": 0.8,  "delta": 0.0},
                "HumanEval":  {"production": 0.8,  "architecture_only": 0.8,  "delta": 0.0},
                "TruthfulQA": {"production": 0.8,  "architecture_only": 0.8,  "delta": 0.0},
            },
            "tier_memory": {
                "FoVer":      {"production": 0.85, "architecture_only": 0.60, "delta": 0.25},
                "MBPP":       {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
                "HumanEval":  {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
                "TruthfulQA": {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
            },
            "tier0r": {
                "FoVer":      {"production": 0.0,  "architecture_only": 0.0,  "delta": 0.0},
                "MBPP":       {"production": 0.0,  "architecture_only": 0.0,  "delta": 0.0},
                "HumanEval":  {"production": 0.0,  "architecture_only": 0.0,  "delta": 0.0},
                "TruthfulQA": {"production": 0.68, "architecture_only": 0.69, "delta": -0.01},
            },
            "tier0s": {
                "FoVer":      {"production": 0.0,  "architecture_only": 0.0,  "delta": 0.0},
                "MBPP":       {"production": 0.0,  "architecture_only": 0.0,  "delta": 0.0},
                "HumanEval":  {"production": 0.0,  "architecture_only": 0.0,  "delta": 0.0},
                "TruthfulQA": {"production": 0.65, "architecture_only": 0.64, "delta": 0.01},
            },
            "tier_low": {
                "FoVer":      {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
                "MBPP":       {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
                "HumanEval":  {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
                "TruthfulQA": {"production": 0.5,  "architecture_only": 0.5,  "delta": 0.0},
            },
        },
        "architecture_transfer_verifiers": ["tier_transfer"],
        "memory_augmented_verifiers": ["tier_memory"],
        "corpus_specific_verifiers": ["tier_specific"],
        "low_signal_verifiers": ["tier_low"],
        "flagged_adversarial": False,
        "duration_s": 35.5,
    }


def _exp2825_clean() -> dict:
    return {
        "honest_verdict": "complete: Multicorpus table integrated",
        "paper_v6_compile_success": True,
        "submission_package_ready": True,
        "duration_s": 35.0,
    }


def _write_full_happy_path(root: Path) -> None:
    """Write all 7 .267 task artifacts in a nominal (clean) state."""
    _write(root, "results/experiment_2819_archive_v266.json",
           {"honest_verdict": "complete: archived"})
    _write(root, "results/experiment_2820_fover_memory_leakage_isolation.json",
           _exp2820_clean())
    _write(root, "results/experiment_2821_mbpp_ensemble_eval.json",
           _exp2821_clean())
    _write(root, "results/experiment_2822_humaneval_full_ensemble_eval.json",
           _exp2822_clean())
    _write(root, "results/experiment_2823_truthfulqa_ensemble_eval.json",
           _exp2823_clean())
    _write(root, "results/experiment_2824_cross_corpus_verifier_matrix.json",
           _exp2824_real())
    _write(root, "results/experiment_2825_paper_v6_multicorpus_table.json",
           _exp2825_clean())


def _write_degraded_inputs(root: Path) -> None:
    """Write only the partial .267 artifacts that actually exist (the .267 reality).

    exp2819-2822 are absent (Gemini crash storm).
    exp2823 is adversarially flagged.
    exp2824 is present but has placeholder-like data.
    exp2825 is present and clean.
    """
    _write(root, "results/experiment_2823_truthfulqa_ensemble_eval.json",
           _exp2823_flagged())
    _write(root, "results/experiment_2824_cross_corpus_verifier_matrix.json",
           _exp2824_placeholder())
    _write(root, "results/experiment_2825_paper_v6_multicorpus_table.json",
           _exp2825_clean())
    _write(root, "results/experiment_2818_capstone_v266.json",
           {"honest_verdict": "complete: capstone v266 written"})


# ===========================================================================
# Tests for utility functions
# ===========================================================================


class TestIsTerminalVerdict:
    """REQ-PUBLISH-032: terminal-prefix discipline."""

    def test_complete_colon_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("complete: foo") is True

    def test_complete_underscore_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("complete_under_score") is True

    def test_success_colon_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("success: all good") is True

    def test_success_underscore_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("success_all_good") is True

    def test_passed_colon_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("passed: tests green") is True

    def test_passed_underscore_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("passed_tests_green") is True

    def test_shipped_colon_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("shipped: artifact") is True

    def test_shipped_underscore_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("shipped_artifact") is True

    def test_leading_whitespace_still_accepted(self) -> None:
        assert exp2826.is_terminal_verdict("  complete: ws") is True

    def test_blocked_prefix_rejected(self) -> None:
        assert exp2826.is_terminal_verdict("blocked_precondition") is False

    def test_none_rejected(self) -> None:
        assert exp2826.is_terminal_verdict(None) is False  # type: ignore[arg-type]

    def test_int_rejected(self) -> None:
        assert exp2826.is_terminal_verdict(123) is False  # type: ignore[arg-type]

    def test_empty_string_rejected(self) -> None:
        assert exp2826.is_terminal_verdict("") is False

    def test_non_prefix_string_rejected(self) -> None:
        assert exp2826.is_terminal_verdict("gate_passed_without_data") is False


class TestReadJson:
    """REQ-PUBLISH-032: read_json degrades gracefully on missing/malformed files."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert exp2826.read_json(tmp_path / "nonexistent.json") == {}

    def test_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("NOT JSON {{", encoding="utf-8")
        assert exp2826.read_json(p) == {}

    def test_json_list_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        assert exp2826.read_json(p) == {}

    def test_valid_dict_is_returned(self, tmp_path: Path) -> None:
        p = tmp_path / "good.json"
        p.write_text('{"k": "v"}', encoding="utf-8")
        assert exp2826.read_json(p) == {"k": "v"}


# ===========================================================================
# Tests for thesis determination helpers
# ===========================================================================


class TestFoverOverfitDetermination:
    """REQ-PUBLISH-032: fover_shape_overfit_confirmed logic."""

    def test_missing_fover_artifact_gives_false(self) -> None:
        """SCENARIO-PUBLISH-032B: exp2820 absent → overfit unconfirmable."""
        confirmed, rationale = exp2826._determine_fover_overfit({}, [_exp2821_clean(), _exp2822_clean()])
        assert confirmed is False
        assert "missing" in rationale.lower() or "flagged" in rationale.lower()

    def test_no_non_fover_artifacts_gives_false(self) -> None:
        confirmed, rationale = exp2826._determine_fover_overfit(_exp2820_clean(), [])
        assert confirmed is False
        assert "non-fover" in rationale.lower()

    def test_overfit_confirmed_when_delta_exceeds_threshold(self) -> None:
        """FoVer arch-only=0.87, non-FoVer max=0.72 → delta=0.15 > 0.10."""
        fover = _exp2820_clean()  # condition_b=0.87
        non_fover = [_exp2821_clean(arch_only=0.72), _exp2822_clean(arch_only=0.68)]
        confirmed, rationale = exp2826._determine_fover_overfit(fover, non_fover)
        assert confirmed is True
        assert "0.10" in rationale or "0.15" in rationale

    def test_overfit_not_confirmed_when_delta_at_threshold(self) -> None:
        """FoVer arch-only=0.82, non-FoVer max=0.72 → delta=0.10, not > threshold."""
        fover = dict(_exp2820_clean(), condition_b_architecture_only_auroc_mean=0.82)
        non_fover = [_exp2821_clean(arch_only=0.72)]
        confirmed, _ = exp2826._determine_fover_overfit(fover, non_fover)
        assert confirmed is False

    def test_adversarially_flagged_non_fover_is_excluded(self) -> None:
        """Flagged exp2823 should not contribute a non-FoVer AUROC value."""
        fover = _exp2820_clean()  # arch-only=0.87
        flagged_truthfulqa = _exp2823_flagged()  # arch-only=0.69 but flagged
        confirmed, rationale = exp2826._determine_fover_overfit(fover, [flagged_truthfulqa])
        # The only non-FoVer is flagged → excluded → no valid non-FoVer data
        assert confirmed is False
        assert "non-fover" in rationale.lower()

    def test_adversarially_flagged_fover_gives_false(self) -> None:
        """Flagged exp2820 cannot contribute FoVer arch-only AUROC."""
        flagged_fover = dict(_exp2820_clean(), flagged_adversarial=True)
        confirmed, rationale = exp2826._determine_fover_overfit(flagged_fover, [_exp2821_clean()])
        assert confirmed is False


class TestSelfLearningContribution:
    """REQ-PUBLISH-032: self_learning_contribution_confirmed logic."""

    def test_missing_fover_artifact_gives_false(self) -> None:
        confirmed, rationale = exp2826._determine_self_learning_contribution({})
        assert confirmed is False
        assert "missing" in rationale.lower() or "flagged" in rationale.lower()

    def test_contribution_below_threshold_gives_false(self) -> None:
        fover = _exp2820_clean(learning_contribution=0.03)
        confirmed, _ = exp2826._determine_self_learning_contribution(fover)
        assert confirmed is False

    def test_contribution_above_threshold_gives_true(self) -> None:
        fover = _exp2820_clean(learning_contribution=0.12)
        confirmed, rationale = exp2826._determine_self_learning_contribution(fover)
        assert confirmed is True
        assert "0.12" in rationale or "0.05" in rationale

    def test_flagged_fover_gives_false(self) -> None:
        flagged_fover = dict(_exp2820_clean(), flagged_adversarial=True)
        confirmed, _ = exp2826._determine_self_learning_contribution(flagged_fover)
        assert confirmed is False

    def test_negative_contribution_gives_false(self) -> None:
        fover = _exp2820_clean(learning_contribution=-0.01)
        confirmed, _ = exp2826._determine_self_learning_contribution(fover)
        assert confirmed is False


class TestHeadlineRepin:
    """REQ-PUBLISH-032: recommended_headline_repin logic."""

    def test_no_clean_non_fover_gives_false(self) -> None:
        repin, _ = exp2826._determine_headline_repin([{}, {}])  # empty dicts
        assert repin is False

    def test_one_clean_non_fover_gives_false(self) -> None:
        """Need MIN_CLEAN_NON_FOVER_FOR_REPIN = 2; one is not enough."""
        repin, _ = exp2826._determine_headline_repin([_exp2821_clean()])
        assert repin is False

    def test_two_clean_non_fover_gives_true(self) -> None:
        repin, _ = exp2826._determine_headline_repin([_exp2821_clean(), _exp2822_clean()])
        assert repin is True

    def test_flagged_non_fover_not_counted(self) -> None:
        repin, _ = exp2826._determine_headline_repin([_exp2823_flagged(), _exp2821_clean()])
        # Only one clean → repin False
        assert repin is False


# ===========================================================================
# Tests for build_artifact (integration-level)
# ===========================================================================


class TestBuildArtifactNominal:
    """SCENARIO-PUBLISH-032: all 7 artifacts present and clean."""

    def test_honest_verdict_starts_with_complete(self, tmp_path: Path) -> None:
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path, started_epoch=1000.0, now_epoch=1000.5)
        assert art["honest_verdict"].startswith("complete:")

    def test_fover_overfit_confirmed_nominal(self, tmp_path: Path) -> None:
        """FoVer arch-only=0.87, MBPP arch-only=0.72, HumanEval=0.68 → delta 0.15 > 0.10."""
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path, started_epoch=2000.0, now_epoch=2001.0)
        assert art["fover_shape_overfit_confirmed"] is True

    def test_self_learning_confirmed_nominal(self, tmp_path: Path) -> None:
        """learning_contribution=0.12 > 0.05 threshold."""
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert art["self_learning_contribution_confirmed"] is True

    def test_headline_repin_recommended_nominal(self, tmp_path: Path) -> None:
        """Two clean non-FoVer corpora → repin viable."""
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert art["recommended_headline_repin"] is True

    def test_experiment_id_and_milestone(self, tmp_path: Path) -> None:
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert art["experiment"] == "exp2826"
        assert art["milestone"] == "2026.05.267"

    def test_corpora_headline_table_has_all_four_corpora(self, tmp_path: Path) -> None:
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        table = art["corpora_headline_table"]
        assert set(table.keys()) == {"FoVer", "MBPP", "HumanEval", "TruthfulQA"}

    def test_duration_s_is_non_negative(self, tmp_path: Path) -> None:
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path, started_epoch=5000.0, now_epoch=5001.3)
        assert abs(art["duration_s"] - 1.3) < 0.01

    def test_acceptance_criteria_high_on_happy_path(self, tmp_path: Path) -> None:
        """On a clean run, at least 8 of 10 criteria should be met."""
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert art["acceptance_criteria_met"] >= 8

    def test_no_execution_layer_gap_on_happy_path(self, tmp_path: Path) -> None:
        _write_full_happy_path(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        flag_kinds = [f["kind"] for f in art["process_flags"]]
        assert "EXECUTION_LAYER_GAP" not in flag_kinds


class TestBuildArtifactDegraded:
    """SCENARIO-PUBLISH-032B: most upstream artifacts missing or flagged."""

    def test_honest_verdict_starts_with_complete(self, tmp_path: Path) -> None:
        """Even with most experiments missing, the verdict is still terminal."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path, started_epoch=3000.0, now_epoch=3002.0)
        assert art["honest_verdict"].startswith("complete:")

    def test_fover_overfit_not_confirmed_degraded(self, tmp_path: Path) -> None:
        """exp2820 missing → overfit unconfirmable."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert art["fover_shape_overfit_confirmed"] is False

    def test_self_learning_not_confirmed_degraded(self, tmp_path: Path) -> None:
        """exp2820 missing → FR-11 contribution unconfirmable."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert art["self_learning_contribution_confirmed"] is False

    def test_headline_repin_false_degraded(self, tmp_path: Path) -> None:
        """No clean non-FoVer data → keep FoVer-only headline."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert art["recommended_headline_repin"] is False

    def test_execution_layer_gap_flagged_degraded(self, tmp_path: Path) -> None:
        """4+ missing experiments → EXECUTION_LAYER_GAP flag."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        flag_kinds = [f["kind"] for f in art["process_flags"]]
        assert "EXECUTION_LAYER_GAP" in flag_kinds

    def test_adversarial_flag_surfaced_degraded(self, tmp_path: Path) -> None:
        """exp2823 flagged adversarially → ADVERSARIALLY_FLAGGED_INPUT flag."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        flag_kinds = [f["kind"] for f in art["process_flags"]]
        assert "ADVERSARIALLY_FLAGGED_INPUT" in flag_kinds

    def test_gaps_for_268_has_minimum_entries(self, tmp_path: Path) -> None:
        """At least 3 gaps must be filed per task spec."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert len(art["gaps_for_268"]) >= 3

    def test_gaps_for_268_are_structured(self, tmp_path: Path) -> None:
        """Each gap must have a 'title' and 'rationale' key."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        for gap in art["gaps_for_268"]:
            assert "title" in gap, f"Gap missing 'title': {gap}"
            assert "rationale" in gap, f"Gap missing 'rationale': {gap}"

    def test_carry_forward_auroc_present_degraded(self, tmp_path: Path) -> None:
        """Carry-forward FoVer AUROC must be preserved when exp2820 is absent."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert abs(art["carry_forward_auroc"] - exp2826.CARRY_FORWARD_AUROC) < 1e-9

    def test_acceptance_criteria_low_degraded(self, tmp_path: Path) -> None:
        """Expect at most 5 criteria met under the degraded .267 reality."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        # Criteria 8, 9, 10 are always met (capstone addresses them honestly).
        # Others depend on upstream data that is missing.
        assert art["acceptance_criteria_met"] >= 3
        assert art["acceptance_criteria_met"] < 8

    def test_verifier_classification_marked_provisional_with_placeholder_matrix(
        self, tmp_path: Path
    ) -> None:
        """exp2824 placeholder-like data → verifier_classification_provisional=True."""
        _write_degraded_inputs(tmp_path)
        art = exp2826.build_artifact(tmp_path)
        assert art["verifier_classification_provisional"] is True

    def test_verifier_classification_not_provisional_with_real_matrix(
        self, tmp_path: Path
    ) -> None:
        """Non-placeholder exp2824 → verifier_classification_provisional=False."""
        _write_full_happy_path(tmp_path)
        _write(root=tmp_path, rel="results/experiment_2824_cross_corpus_verifier_matrix.json",
               payload=_exp2824_real())
        art = exp2826.build_artifact(tmp_path)
        assert art["verifier_classification_provisional"] is False


class TestMatrixLooksProvisional:
    """REQ-PUBLISH-032: _matrix_looks_provisional detects placeholder data."""

    def test_empty_artifact_returns_false(self) -> None:
        assert exp2826._matrix_looks_provisional({}) is False

    def test_mostly_zero_values_returns_true(self) -> None:
        art = _exp2824_placeholder()
        assert exp2826._matrix_looks_provisional(art) is True

    def test_real_values_returns_false(self) -> None:
        art = _exp2824_real()
        assert exp2826._matrix_looks_provisional(art) is False

    def test_missing_matrix_key_returns_false(self) -> None:
        assert exp2826._matrix_looks_provisional({"honest_verdict": "complete: x"}) is False


class TestWriteArtifact:
    """REQ-PUBLISH-032: write_artifact creates a valid JSON file on disk."""

    def test_file_is_created(self, tmp_path: Path) -> None:
        _write_degraded_inputs(tmp_path)
        out = exp2826.write_artifact(tmp_path)
        assert out.is_file()

    def test_required_fields_present(self, tmp_path: Path) -> None:
        _write_degraded_inputs(tmp_path)
        out = exp2826.write_artifact(tmp_path)
        payload = json.loads(out.read_text(encoding="utf-8"))
        required = {
            "honest_verdict",
            "corpora_headline_table",
            "fover_shape_overfit_confirmed",
            "self_learning_contribution_confirmed",
            "architecture_transfer_verifiers",
            "memory_augmented_verifiers",
            "corpus_specific_verifiers",
            "low_signal_verifiers",
            "recommended_headline_repin",
            "gaps_for_268",
            "acceptance_criteria_met",
            "duration_s",
        }
        missing_fields = required - payload.keys()
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_honest_verdict_is_terminal(self, tmp_path: Path) -> None:
        _write_degraded_inputs(tmp_path)
        out = exp2826.write_artifact(tmp_path)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert exp2826.is_terminal_verdict(payload["honest_verdict"])

    def test_duration_s_is_non_negative(self, tmp_path: Path) -> None:
        _write_degraded_inputs(tmp_path)
        out = exp2826.write_artifact(tmp_path)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["duration_s"] >= 0.0

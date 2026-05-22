"""Tests for the Exp 2834 milestone .268 multi-corpus capstone synthesis.

WHY a dedicated test file: the .268 capstone must degrade gracefully under
the specific .268 failure mode — all corpus eval tasks produced honest
``blocked_*`` verdicts (not missing artifacts) because torch/CUDA was not
installed in the execution environment.  This is qualitatively different
from .267 (Gemini crash storm) and the tests verify both the degraded
(.268 reality: all blocked) and nominal (hypothetical clean run) paths.

Spec refs: REQ-BENCH-001, REQ-BENCH-010, REQ-PUBLISH-032,
           SCENARIO-PUBLISH-032, SCENARIO-PUBLISH-032B.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.reporting import capstone_v268_2834 as exp2834


# ---------------------------------------------------------------------------
# Helper: write a JSON file into a tmp_path tree
# ---------------------------------------------------------------------------

def _write(root: Path, rel: str, payload: dict) -> None:
    """Serialise *payload* as JSON at root/rel, creating parent dirs."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture helpers: canonical artifact shapes
# ---------------------------------------------------------------------------

def _exp2827_clean() -> dict:
    """Clean archive/activation artifact (always succeeds)."""
    return {
        "honest_verdict": "complete: archive_ready=true; archived_milestone=2026.05.267",
        "archive_ready": True,
        "archived_milestone": "2026.05.267",
    }


def _exp2828_blocked() -> dict:
    """The real .268 exp2828 — blocked on missing torch/CUDA."""
    return {
        "honest_verdict": (
            "blocked_cuda: Traceback (most recent call last):\n"
            "  File \"<string>\", line 1, in <module>\n"
            "    import torch; assert torch.cuda.is_available()\n"
            "    ^^^^^^^^^^^^\n"
            "ModuleNotFoundError: No module named 'torch'"
        ),
        "condition_a_production_auroc_mean": None,
        "condition_b_architecture_only_auroc_mean": None,
        "learning_contribution": None,
        "methodology_note": (
            "The requested live dual-condition FoVer measurement was not run "
            "because one or more mandatory preconditions failed."
        ),
        "duration_s": 0.054,
        "flagged_adversarial": False,
    }


def _exp2828_clean(learning_contribution: float = 0.12) -> dict:
    """A hypothetical clean exp2828 (FoVer dual-condition)."""
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


def _exp2829_blocked() -> dict:
    """Real .268 exp2829 (MBPP) — blocked_cuda_unavailable."""
    return {
        "honest_verdict": "blocked_cuda_unavailable",
        "condition_a_production_auroc_mean": None,
        "condition_b_architecture_only_auroc_mean": None,
        "learning_contribution": None,
        "flagged_adversarial": False,
        "duration_s": 0.89,
    }


def _exp2829_clean(arch_only: float = 0.72) -> dict:
    """A hypothetical clean exp2829 (MBPP)."""
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


def _exp2830_blocked() -> dict:
    return {
        "honest_verdict": "blocked_cuda_unavailable",
        "condition_a_production_auroc_mean": None,
        "condition_b_architecture_only_auroc_mean": None,
        "learning_contribution": None,
        "flagged_adversarial": False,
        "duration_s": 0.95,
    }


def _exp2830_clean(arch_only: float = 0.68) -> dict:
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


def _exp2831_blocked() -> dict:
    return {
        "honest_verdict": "blocked_cuda_unavailable",
        "condition_a_production_auroc_mean": None,
        "condition_b_architecture_only_auroc_mean": None,
        "learning_contribution": None,
        "flagged_adversarial": False,
        "duration_s": 0.83,
    }


def _exp2831_clean() -> dict:
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


def _exp2832_empty() -> dict:
    """Real .268 exp2832 — empty matrix (all upstream corpora blocked)."""
    return {
        "honest_verdict": "complete: upstream artifacts loaded but no measured per-verifier AUROC rows were present",
        "verifier_corpus_dual_matrix": {},
        "architecture_transfer_verifiers": [],
        "memory_augmented_verifiers": [],
        "corpus_specific_verifiers": [],
        "low_signal_verifiers": [],
        "diversity_gap_on_non_fover": True,
        "flagged_adversarial": False,
        "duration_s": 0.0003,
    }


def _exp2832_real() -> dict:
    """A hypothetical exp2832 with real verifier matrix data."""
    return {
        "honest_verdict": "complete: Matrix generated successfully",
        "verifier_corpus_dual_matrix": {
            "tier_transfer": {
                "FoVer":      {"production": 0.81, "architecture_only": 0.80, "delta": 0.01},
                "MBPP":       {"production": 0.79, "architecture_only": 0.78, "delta": 0.01},
                "HumanEval":  {"production": 0.76, "architecture_only": 0.75, "delta": 0.01},
                "TruthfulQA": {"production": 0.63, "architecture_only": 0.64, "delta": -0.01},
            },
            "tier_memory": {
                "FoVer":      {"production": 0.85, "architecture_only": 0.60, "delta": 0.25},
                "MBPP":       {"production": 0.55, "architecture_only": 0.54, "delta": 0.01},
                "HumanEval":  {"production": 0.53, "architecture_only": 0.52, "delta": 0.01},
                "TruthfulQA": {"production": 0.51, "architecture_only": 0.51, "delta": 0.00},
            },
        },
        "architecture_transfer_verifiers": ["tier_transfer"],
        "memory_augmented_verifiers": ["tier_memory"],
        "corpus_specific_verifiers": ["tier_specific"],
        "low_signal_verifiers": ["tier_low"],
        "flagged_adversarial": False,
        "duration_s": 45.3,
    }


def _exp2833_compiled() -> dict:
    """exp2833 where pdflatex compiled but AUROC data is null."""
    return {
        "honest_verdict": "complete: paper compiled; AUROC data null",
        "paper_v6_compile_success": True,
        "submission_package_ready": False,
        "duration_s": 0.44,
        "flagged_adversarial": False,
    }


def _write_v268_reality(root: Path) -> None:
    """Write the actual .268 artifacts (all corpus evals blocked on CUDA)."""
    _write(root, "results/experiment_2827_archive_v267.json", _exp2827_clean())
    _write(root, "results/experiment_2828_fover_memory_leakage_isolation.json", _exp2828_blocked())
    _write(root, "results/experiment_2829_mbpp_ensemble_eval.json", _exp2829_blocked())
    _write(root, "results/experiment_2830_humaneval_full_ensemble_eval.json", _exp2830_blocked())
    _write(root, "results/experiment_2831_truthfulqa_ensemble_eval.json", _exp2831_blocked())
    _write(root, "results/experiment_2832_cross_corpus_verifier_matrix_v2.json", _exp2832_empty())
    _write(root, "results/experiment_2833_paper_v6_multicorpus_table_v2.json", _exp2833_compiled())


def _write_v268_happy_path(root: Path) -> None:
    """Write hypothetical clean .268 artifacts (all corpus evals succeeded)."""
    _write(root, "results/experiment_2827_archive_v267.json", _exp2827_clean())
    _write(root, "results/experiment_2828_fover_memory_leakage_isolation.json", _exp2828_clean())
    _write(root, "results/experiment_2829_mbpp_ensemble_eval.json", _exp2829_clean())
    _write(root, "results/experiment_2830_humaneval_full_ensemble_eval.json", _exp2830_clean())
    _write(root, "results/experiment_2831_truthfulqa_ensemble_eval.json", _exp2831_clean())
    _write(root, "results/experiment_2832_cross_corpus_verifier_matrix_v2.json", _exp2832_real())
    _write(root, "results/experiment_2833_paper_v6_multicorpus_table_v2.json", _exp2833_compiled())


# ===========================================================================
# Utility function tests
# ===========================================================================


class TestIsTerminalVerdict:
    """REQ-PUBLISH-032: terminal-prefix discipline."""

    def test_complete_colon_accepted(self) -> None:
        assert exp2834.is_terminal_verdict("complete: foo") is True

    def test_complete_underscore_accepted(self) -> None:
        assert exp2834.is_terminal_verdict("complete_under") is True

    def test_success_colon_accepted(self) -> None:
        assert exp2834.is_terminal_verdict("success: ok") is True

    def test_passed_colon_accepted(self) -> None:
        assert exp2834.is_terminal_verdict("passed: tests green") is True

    def test_shipped_colon_accepted(self) -> None:
        assert exp2834.is_terminal_verdict("shipped: artifact") is True

    def test_blocked_prefix_rejected(self) -> None:
        assert exp2834.is_terminal_verdict("blocked_cuda") is False

    def test_none_rejected(self) -> None:
        assert exp2834.is_terminal_verdict(None) is False  # type: ignore[arg-type]

    def test_empty_string_rejected(self) -> None:
        assert exp2834.is_terminal_verdict("") is False

    def test_non_prefix_string_rejected(self) -> None:
        assert exp2834.is_terminal_verdict("gate_passed_without_data") is False


class TestIsBlockedVerdict:
    """REQ-BENCH-010: blocked_* verdicts are distinct from missing artifacts."""

    def test_blocked_cuda_recognised(self) -> None:
        assert exp2834.is_blocked_verdict("blocked_cuda: no torch") is True

    def test_blocked_cuda_unavailable_recognised(self) -> None:
        assert exp2834.is_blocked_verdict("blocked_cuda_unavailable") is True

    def test_complete_not_blocked(self) -> None:
        assert exp2834.is_blocked_verdict("complete: ok") is False

    def test_none_not_blocked(self) -> None:
        assert exp2834.is_blocked_verdict(None) is False  # type: ignore[arg-type]

    def test_empty_string_not_blocked(self) -> None:
        assert exp2834.is_blocked_verdict("") is False


class TestReadJson:
    """REQ-PUBLISH-032: read_json degrades gracefully on missing/malformed files."""

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert exp2834.read_json(tmp_path / "nonexistent.json") == {}

    def test_malformed_json_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("NOT JSON {{", encoding="utf-8")
        assert exp2834.read_json(p) == {}

    def test_json_list_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        assert exp2834.read_json(p) == {}

    def test_valid_dict_returned(self, tmp_path: Path) -> None:
        p = tmp_path / "good.json"
        p.write_text('{"k": "v"}', encoding="utf-8")
        assert exp2834.read_json(p) == {"k": "v"}


# ===========================================================================
# Thesis determination helpers
# ===========================================================================


class TestFoverOverfitDetermination:
    """REQ-PUBLISH-032: fover_shape_overfit_confirmed logic for .268."""

    def test_blocked_fover_gives_false(self) -> None:
        """SCENARIO-PUBLISH-032B: exp2828 blocked → overfit unconfirmable."""
        confirmed, rationale = exp2834._determine_fover_overfit(
            _exp2828_blocked(), [_exp2829_clean(), _exp2830_clean()]
        )
        assert confirmed is False
        assert "blocked" in rationale.lower()

    def test_missing_fover_gives_false(self) -> None:
        """Empty dict (artifact never produced) → overfit unconfirmable."""
        confirmed, rationale = exp2834._determine_fover_overfit({}, [_exp2829_clean()])
        assert confirmed is False
        assert "missing" in rationale.lower() or "flagged" in rationale.lower()

    def test_no_non_fover_gives_false(self) -> None:
        """FoVer measured but all non-FoVer blocked → cannot compare."""
        confirmed, rationale = exp2834._determine_fover_overfit(
            _exp2828_clean(), []
        )
        assert confirmed is False
        assert "non-fover" in rationale.lower()

    def test_overfit_confirmed_when_delta_exceeds_threshold(self) -> None:
        """FoVer arch-only=0.87, MBPP=0.72 → delta=0.15 > 0.10 → confirmed."""
        confirmed, rationale = exp2834._determine_fover_overfit(
            _exp2828_clean(),
            [_exp2829_clean(arch_only=0.72), _exp2830_clean(arch_only=0.68)],
        )
        assert confirmed is True
        assert "CONFIRMED" in rationale

    def test_overfit_not_confirmed_when_delta_at_threshold(self) -> None:
        """FoVer arch-only=0.82, non-FoVer max=0.72 → delta=0.10, NOT > threshold."""
        fover = dict(_exp2828_clean(), condition_b_architecture_only_auroc_mean=0.82)
        confirmed, _ = exp2834._determine_fover_overfit(fover, [_exp2829_clean(arch_only=0.72)])
        assert confirmed is False

    def test_all_non_fover_blocked_gives_false(self) -> None:
        """All three non-FoVer corpora blocked → no comparison possible."""
        confirmed, rationale = exp2834._determine_fover_overfit(
            _exp2828_clean(),
            [_exp2829_blocked(), _exp2830_blocked(), _exp2831_blocked()],
        )
        assert confirmed is False
        assert "non-fover" in rationale.lower()


class TestSelfLearningContribution:
    """REQ-PUBLISH-032: self_learning_contribution_confirmed logic for .268."""

    def test_blocked_fover_gives_false(self) -> None:
        confirmed, rationale = exp2834._determine_self_learning_contribution(
            _exp2828_blocked()
        )
        assert confirmed is False
        assert "blocked" in rationale.lower()

    def test_missing_fover_gives_false(self) -> None:
        confirmed, rationale = exp2834._determine_self_learning_contribution({})
        assert confirmed is False

    def test_contribution_below_threshold_gives_false(self) -> None:
        fover = _exp2828_clean(learning_contribution=0.03)
        confirmed, _ = exp2834._determine_self_learning_contribution(fover)
        assert confirmed is False

    def test_contribution_above_threshold_gives_true(self) -> None:
        fover = _exp2828_clean(learning_contribution=0.12)
        confirmed, rationale = exp2834._determine_self_learning_contribution(fover)
        assert confirmed is True
        assert "CONFIRMED" in rationale

    def test_negative_contribution_gives_false(self) -> None:
        fover = _exp2828_clean(learning_contribution=-0.01)
        confirmed, _ = exp2834._determine_self_learning_contribution(fover)
        assert confirmed is False


class TestHeadlineRepin:
    """REQ-PUBLISH-032: recommended_headline_repin logic."""

    def test_zero_clean_non_fover_gives_false(self) -> None:
        """Zero clean non-FoVer corpora (all blocked) → cannot repin."""
        repin, rationale = exp2834._determine_headline_repin(
            [_exp2829_blocked(), _exp2830_blocked(), _exp2831_blocked()]
        )
        assert repin is False
        assert "0" in rationale

    def test_one_clean_non_fover_gives_false(self) -> None:
        """Need MIN_CLEAN_NON_FOVER_FOR_REPIN = 2; one is not enough."""
        repin, _ = exp2834._determine_headline_repin([_exp2829_clean()])
        assert repin is False

    def test_two_clean_non_fover_gives_true(self) -> None:
        repin, _ = exp2834._determine_headline_repin(
            [_exp2829_clean(), _exp2830_clean()]
        )
        assert repin is True

    def test_empty_list_gives_false(self) -> None:
        repin, _ = exp2834._determine_headline_repin([])
        assert repin is False


class TestMatrixLooksProvisional:
    """REQ-PUBLISH-032: _matrix_looks_provisional detects empty/placeholder matrix."""

    def test_empty_artifact_returns_false(self) -> None:
        """No matrix_art at all → not provisional (no data to judge)."""
        assert exp2834._matrix_looks_provisional({}) is False

    def test_empty_matrix_dict_returns_true(self) -> None:
        """Empty verifier_corpus_dual_matrix is the .268 reality → provisional."""
        assert exp2834._matrix_looks_provisional(_exp2832_empty()) is True

    def test_real_matrix_returns_false(self) -> None:
        assert exp2834._matrix_looks_provisional(_exp2832_real()) is False

    def test_missing_matrix_key_returns_false(self) -> None:
        assert exp2834._matrix_looks_provisional({"honest_verdict": "complete: x"}) is False


# ===========================================================================
# Integration tests: build_artifact with .268 reality (all blocked)
# ===========================================================================


class TestBuildArtifactV268Reality:
    """SCENARIO-PUBLISH-032B: .268 actual state — all corpus evals blocked on CUDA."""

    def test_honest_verdict_starts_with_complete(self, tmp_path: Path) -> None:
        """Even with all corpora blocked, the capstone verdict is terminal."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path, started_epoch=1000.0, now_epoch=1001.0)
        assert art["honest_verdict"].startswith("complete:")

    def test_fover_overfit_not_confirmed_reality(self, tmp_path: Path) -> None:
        """exp2828 blocked → FoVer architecture-only AUROC not measured → False."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["fover_shape_overfit_confirmed"] is False

    def test_self_learning_not_confirmed_reality(self, tmp_path: Path) -> None:
        """exp2828 blocked → FR-11 learning_contribution not measured → False."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["self_learning_contribution_confirmed"] is False

    def test_headline_repin_false_reality(self, tmp_path: Path) -> None:
        """No clean non-FoVer production AUROC → keep FoVer-only headline."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["recommended_headline_repin"] is False

    def test_supersedes_267_capstone_is_true(self, tmp_path: Path) -> None:
        """This capstone must explicitly flag that it supersedes exp2826."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["supersedes_267_capstone"] is True

    def test_precondition_block_storm_detected(self, tmp_path: Path) -> None:
        """4 corpus eval tasks blocked → precondition_block_storm_detected=True."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["precondition_block_storm_detected"] is True

    def test_precondition_block_flag_in_process_flags(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        flag_kinds = [f["kind"] for f in art["process_flags"]]
        assert "PRECONDITION_BLOCK_STORM" in flag_kinds

    def test_carry_forward_auroc_preserved(self, tmp_path: Path) -> None:
        """FoVer-only headline AUROC must be preserved when exp2828 is blocked."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert abs(art["carry_forward_auroc"] - exp2834.CARRY_FORWARD_AUROC) < 1e-9

    def test_acceptance_criteria_met_limited(self, tmp_path: Path) -> None:
        """With all corpus evals blocked, at most 4 of 10 criteria are met."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        # 1_archive_267_landed, 7_paper_compiled, 8/9/10 honesty criteria
        assert art["acceptance_criteria_met"] >= 3
        assert art["acceptance_criteria_met"] < 7

    def test_corpora_headline_table_has_all_four(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert set(art["corpora_headline_table"].keys()) == {
            "FoVer", "MBPP", "HumanEval", "TruthfulQA"
        }

    def test_non_fover_corpora_blocked_status(self, tmp_path: Path) -> None:
        """Non-FoVer corpus table rows should reflect blocked_cuda data_status."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        for corpus in ("MBPP", "HumanEval", "TruthfulQA"):
            row = art["corpora_headline_table"][corpus]
            assert row["data_status"] == "blocked_cuda", (
                f"{corpus} data_status should be blocked_cuda, got {row['data_status']}"
            )

    def test_fover_carry_forward_status(self, tmp_path: Path) -> None:
        """FoVer row should show carry_forward_exp2546 status when exp2828 blocked."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["corpora_headline_table"]["FoVer"]["data_status"] == "carry_forward_exp2546"

    def test_gaps_for_269_has_minimum_entries(self, tmp_path: Path) -> None:
        """At least 5 gaps must be filed for .269."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert len(art["gaps_for_269"]) >= 5

    def test_gaps_for_269_are_structured(self, tmp_path: Path) -> None:
        """Each gap must have a 'title' and 'rationale' key."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        for gap in art["gaps_for_269"]:
            assert "title" in gap, f"Gap missing 'title': {gap}"
            assert "rationale" in gap, f"Gap missing 'rationale': {gap}"

    def test_gaps_mentions_torch_cuda(self, tmp_path: Path) -> None:
        """Gaps must document the torch/CUDA root cause."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        combined = " ".join(g["title"] + g["rationale"] for g in art["gaps_for_269"])
        assert "torch" in combined.lower() or "cuda" in combined.lower()

    def test_duration_s_non_negative(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path, started_epoch=5000.0, now_epoch=5001.3)
        assert art["duration_s"] >= 0.0

    def test_matrix_provisional_true_when_empty(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["verifier_classification_provisional"] is True

    def test_archive_criterion_met_when_exp2827_clean(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["acceptance_criteria_detail"]["1_archive_267_landed"] is True

    def test_paper_compile_criterion_met_when_exp2833_clean(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["acceptance_criteria_detail"]["7_paper_v6_section_5_compiled"] is True

    def test_thesis_criteria_always_true(self, tmp_path: Path) -> None:
        """Criteria 8 and 9 are always True (capstone addresses them honestly)."""
        _write_v268_reality(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["acceptance_criteria_detail"]["8_fover_overfit_thesis_addressed"] is True
        assert art["acceptance_criteria_detail"]["9_fr11_hypothesis_addressed"] is True


# ===========================================================================
# Integration tests: build_artifact with hypothetical nominal path
# ===========================================================================


class TestBuildArtifactNominal:
    """SCENARIO-PUBLISH-032: all .268 artifacts present and clean."""

    def test_honest_verdict_terminal(self, tmp_path: Path) -> None:
        _write_v268_happy_path(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["honest_verdict"].startswith("complete:")

    def test_fover_overfit_confirmed_nominal(self, tmp_path: Path) -> None:
        """FoVer arch-only=0.87, MBPP=0.72, HumanEval=0.68 → delta 0.15 > 0.10."""
        _write_v268_happy_path(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["fover_shape_overfit_confirmed"] is True

    def test_self_learning_confirmed_nominal(self, tmp_path: Path) -> None:
        """learning_contribution=0.12 > 0.05 threshold."""
        _write_v268_happy_path(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["self_learning_contribution_confirmed"] is True

    def test_headline_repin_recommended_nominal(self, tmp_path: Path) -> None:
        """Two clean non-FoVer production AUROCs → repin viable."""
        _write_v268_happy_path(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["recommended_headline_repin"] is True

    def test_high_acceptance_criteria_nominal(self, tmp_path: Path) -> None:
        """At least 8 of 10 criteria met on a clean run."""
        _write_v268_happy_path(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["acceptance_criteria_met"] >= 8

    def test_no_precondition_block_storm_nominal(self, tmp_path: Path) -> None:
        _write_v268_happy_path(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["precondition_block_storm_detected"] is False

    def test_verifier_classification_from_exp2832(self, tmp_path: Path) -> None:
        """architecture_transfer_verifiers comes from exp2832 on the happy path."""
        _write_v268_happy_path(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert "tier_transfer" in art["architecture_transfer_verifiers"]
        assert "tier_memory" in art["memory_augmented_verifiers"]

    def test_matrix_not_provisional_nominal(self, tmp_path: Path) -> None:
        """Real exp2832 data → not provisional."""
        _write_v268_happy_path(tmp_path)
        art = exp2834.build_artifact(tmp_path)
        assert art["verifier_classification_provisional"] is False


# ===========================================================================
# write_artifact: on-disk contract
# ===========================================================================


class TestWriteArtifact:
    """REQ-PUBLISH-032: write_artifact creates a valid JSON file on disk."""

    def test_file_is_created(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        out = exp2834.write_artifact(tmp_path)
        assert out.is_file()

    def test_required_fields_present(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        out = exp2834.write_artifact(tmp_path)
        payload = json.loads(out.read_text(encoding="utf-8"))
        required = {
            "honest_verdict",
            "corpora_headline_table",
            "fover_shape_overfit_confirmed",
            "self_learning_contribution_confirmed",
            "supersedes_267_capstone",
            "architecture_transfer_verifiers",
            "memory_augmented_verifiers",
            "corpus_specific_verifiers",
            "low_signal_verifiers",
            "recommended_headline_repin",
            "gaps_for_269",
            "acceptance_criteria_met",
            "duration_s",
        }
        missing_fields = required - payload.keys()
        assert not missing_fields, f"Missing required fields: {missing_fields}"

    def test_honest_verdict_is_terminal(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        out = exp2834.write_artifact(tmp_path)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert exp2834.is_terminal_verdict(payload["honest_verdict"])

    def test_duration_s_non_negative(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        out = exp2834.write_artifact(tmp_path)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["duration_s"] >= 0.0

    def test_supersedes_267_true(self, tmp_path: Path) -> None:
        _write_v268_reality(tmp_path)
        out = exp2834.write_artifact(tmp_path)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["supersedes_267_capstone"] is True

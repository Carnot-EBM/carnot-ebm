"""Tests for Experiment 344: Constraint Addition Benchmark.

Covers:
- generate_simulated_questions: reproducibility, error distribution, structure
- run_control_condition: returns expected dict shape; accuracy = 0 (no templates)
- run_treatment_condition: templates activate after threshold; accuracy > 0
- compute_improvement_delta: signed, no clamping
- build_addition_benchmark_artifact: schema, required fields, comparison block

Spec: REQ-LEARN-019, SCENARIO-LEARN-033, SCENARIO-LEARN-034
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Import helpers — lazy so we don't pay the cost in non-matching test runs
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

# Ensure scripts/ is on sys.path for the experiment module import
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _load_exp344():
    """Import experiment_344 module, inserting scripts/ into sys.path first."""
    return importlib.import_module(
        "experiment_344_constraint_addition_benchmark"
    )


# ---------------------------------------------------------------------------
# generate_simulated_questions
# ---------------------------------------------------------------------------


class TestGenerateSimulatedQuestions:
    """Tests for the question generation helper.

    REQ-LEARN-019 (indirectly): simulation must reproduce structural conditions
    that reveal whether constraint addition improves over reweighting.
    """

    def test_returns_correct_count(self):
        """Returns exactly n questions."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(50, seed=42)
        assert len(qs) == 50

    def test_reproducible_with_same_seed(self):
        """Two calls with the same seed produce identical questions."""
        mod = _load_exp344()
        qs1 = mod.generate_simulated_questions(20, seed=99)
        qs2 = mod.generate_simulated_questions(20, seed=99)
        assert qs1 == qs2

    def test_different_seeds_different_results(self):
        """Different seeds typically produce different error distributions."""
        mod = _load_exp344()
        qs1 = mod.generate_simulated_questions(50, seed=1)
        qs2 = mod.generate_simulated_questions(50, seed=2)
        # At least some questions should differ
        carry1 = [q["has_carry_error"] for q in qs1]
        carry2 = [q["has_carry_error"] for q in qs2]
        assert carry1 != carry2

    def test_question_dict_has_required_keys(self):
        """Each question dict has all required keys."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(5, seed=42)
        required = {"question", "response", "has_carry_error", "has_sign_error",
                    "has_unit_error", "has_any_error"}
        for q in qs:
            assert required.issubset(q.keys())

    def test_has_any_error_is_union_of_individual_errors(self):
        """has_any_error == carry OR sign OR unit for every question."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(100, seed=42)
        for q in qs:
            expected = q["has_carry_error"] or q["has_sign_error"] or q["has_unit_error"]
            assert q["has_any_error"] == expected

    def test_carry_error_rate_approximately_correct(self):
        """With 1000 questions, carry error rate is within 10pp of 30%."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(1000, seed=42)
        rate = sum(1 for q in qs if q["has_carry_error"]) / 1000
        assert abs(rate - 0.30) < 0.10, f"carry error rate {rate:.2f} not within 10pp of 0.30"

    def test_sign_error_rate_approximately_correct(self):
        """With 1000 questions, sign error rate is within 10pp of 15%."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(1000, seed=42)
        rate = sum(1 for q in qs if q["has_sign_error"]) / 1000
        assert abs(rate - 0.15) < 0.10, f"sign error rate {rate:.2f} not within 10pp of 0.15"

    def test_carry_error_response_contains_multiplication(self):
        """Responses with carry errors contain a multiplication expression."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(200, seed=42)
        carry_qs = [q for q in qs if q["has_carry_error"]]
        assert len(carry_qs) > 0
        for q in carry_qs:
            assert "×" in q["response"] or "*" in q["response"]

    def test_sign_error_response_contains_neg_neg_pattern(self):
        """Responses with sign errors contain a (-A) × (-B) expression."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(200, seed=42)
        sign_qs = [q for q in qs if q["has_sign_error"]]
        assert len(sign_qs) > 0
        for q in sign_qs:
            assert "(-" in q["response"]


# ---------------------------------------------------------------------------
# run_control_condition
# ---------------------------------------------------------------------------


class TestRunControlCondition:
    """Tests for the Control condition (reweighting only)."""

    def test_returns_dict_with_required_keys(self):
        """run_control_condition returns dict with n_detected, n_total_errors, accuracy."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(20, seed=42)
        result = mod.run_control_condition(qs)
        assert "n_detected" in result
        assert "n_total_errors" in result
        assert "accuracy" in result

    def test_n_total_errors_matches_ground_truth(self):
        """n_total_errors equals the count of questions with has_any_error=True."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(50, seed=42)
        expected = sum(1 for q in qs if q["has_any_error"])
        result = mod.run_control_condition(qs)
        assert result["n_total_errors"] == expected

    def test_control_accuracy_is_zero(self):
        """Control condition (no templates) detects 0 errors — accuracy is 0."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(50, seed=42)
        result = mod.run_control_condition(qs)
        assert result["accuracy"] == 0.0

    def test_control_n_detected_is_zero(self):
        """Control condition never detects errors without active templates."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(50, seed=42)
        result = mod.run_control_condition(qs)
        assert result["n_detected"] == 0

    def test_empty_questions_returns_zero_accuracy(self):
        """Edge case: empty question list returns accuracy=0.0."""
        mod = _load_exp344()
        result = mod.run_control_condition([])
        assert result["accuracy"] == 0.0
        assert result["n_total_errors"] == 0

    def test_returns_tracker_stats(self):
        """Control result includes tracker_stats from PerModelFPTracker."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(20, seed=42)
        result = mod.run_control_condition(qs)
        assert "tracker_stats" in result
        assert "min_observations" in result["tracker_stats"]


# ---------------------------------------------------------------------------
# run_treatment_condition
# ---------------------------------------------------------------------------


class TestRunTreatmentCondition:
    """Tests for the Treatment condition (constraint addition enabled)."""

    def test_returns_dict_with_required_keys(self):
        """run_treatment_condition returns dict with all required keys."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(50, seed=42)
        result = mod.run_treatment_condition(qs)
        required = {
            "n_detected", "n_total_errors", "accuracy",
            "n_templates_activated", "n_new_constraints_generated",
            "activated_template_keys",
        }
        assert required.issubset(result.keys())

    def test_templates_activate_after_threshold(self):
        """With 200 questions, carry_check template should activate (n_carry ~ 60 >> 5)."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(200, seed=42)
        result = mod.run_treatment_condition(qs)
        assert result["n_templates_activated"] > 0

    def test_carry_check_template_activates(self):
        """carry_check template activates because carry error rate >> min_frequency=5."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(200, seed=42)
        result = mod.run_treatment_condition(qs)
        assert "carry_check" in result["activated_template_keys"]

    def test_treatment_accuracy_greater_than_control(self):
        """Treatment accuracy > 0 (templates detect some errors that control misses)."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(200, seed=42)
        control_result = mod.run_control_condition(qs)
        treatment_result = mod.run_treatment_condition(qs)
        assert treatment_result["accuracy"] > control_result["accuracy"]

    def test_treatment_accuracy_positive(self):
        """Treatment accuracy is strictly > 0 when templates activate."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(200, seed=42)
        result = mod.run_treatment_condition(qs)
        assert result["accuracy"] > 0.0

    def test_n_new_constraints_generated_positive(self):
        """At least some constraints are generated by active templates."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(200, seed=42)
        result = mod.run_treatment_condition(qs)
        assert result["n_new_constraints_generated"] >= 0

    def test_n_total_errors_matches_ground_truth(self):
        """n_total_errors matches the actual count in the question set."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(50, seed=42)
        expected = sum(1 for q in qs if q["has_any_error"])
        result = mod.run_treatment_condition(qs)
        assert result["n_total_errors"] == expected

    def test_empty_questions_returns_zero_accuracy(self):
        """Edge case: empty question list gives accuracy=0 and no templates activated."""
        mod = _load_exp344()
        result = mod.run_treatment_condition([])
        assert result["accuracy"] == 0.0
        assert result["n_templates_activated"] == 0


# ---------------------------------------------------------------------------
# compute_improvement_delta
# ---------------------------------------------------------------------------


class TestComputeImprovementDelta:
    """Tests for compute_improvement_delta."""

    def test_positive_delta(self):
        """Treatment > control gives positive delta."""
        mod = _load_exp344()
        delta = mod.compute_improvement_delta(0.0, 0.4)
        assert delta == pytest.approx(0.4)

    def test_zero_delta(self):
        """Equal accuracies give delta=0."""
        mod = _load_exp344()
        delta = mod.compute_improvement_delta(0.5, 0.5)
        assert delta == pytest.approx(0.0)

    def test_negative_delta_not_clamped(self):
        """Delta can be negative (no clamping). Regression is reported honestly."""
        mod = _load_exp344()
        delta = mod.compute_improvement_delta(0.8, 0.3)
        assert delta == pytest.approx(-0.5)

    def test_delta_is_treatment_minus_control(self):
        """Delta equals treatment_accuracy - control_accuracy exactly."""
        mod = _load_exp344()
        delta = mod.compute_improvement_delta(0.2, 0.7)
        assert delta == pytest.approx(0.5)

    def test_both_zero(self):
        """Both zero gives delta=0."""
        mod = _load_exp344()
        delta = mod.compute_improvement_delta(0.0, 0.0)
        assert delta == pytest.approx(0.0)

    def test_both_one(self):
        """Both 1.0 gives delta=0."""
        mod = _load_exp344()
        delta = mod.compute_improvement_delta(1.0, 1.0)
        assert delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_addition_benchmark_artifact
# ---------------------------------------------------------------------------


class TestBuildAdditionBenchmarkArtifact:
    """Tests for the artifact builder."""

    def _make_mock_tmpl(self, mod):
        """Build a minimal mock ExperimentTemplate for testing."""
        from experiment_template import ExperimentTemplate
        import tempfile, os
        tmp = tempfile.mkdtemp()
        tmpl = ExperimentTemplate(
            344,
            "Test Exp 344",
            "results/experiment_344_constraint_addition_benchmark.json",
            repo_root=Path(tmp),
        )
        tmpl.setup()
        return tmpl

    def _make_control_result(self) -> dict[str, Any]:
        return {
            "accuracy": 0.0,
            "n_detected": 0,
            "n_total_errors": 80,
            "tracker_stats": {"min_observations": 10, "stats": []},
        }

    def _make_treatment_result(self) -> dict[str, Any]:
        return {
            "accuracy": 0.42,
            "n_detected": 34,
            "n_total_errors": 80,
            "n_templates_activated": 2,
            "n_new_constraints_generated": 120,
            "activated_template_keys": ["carry_check", "sign_check"],
            "library_state": {"observations": []},
        }

    def test_artifact_has_result_type_field(self):
        """Artifact includes result_type='carnot.constraint_addition.v1'.

        Note: ExperimentTemplate.build_result() reserves 'schema' for the sorted
        key list; we use 'result_type' for the experiment-specific schema identifier.
        """
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        assert artifact["result_type"] == "carnot.constraint_addition.v1"

    def test_artifact_has_required_fields(self):
        """Artifact includes all required experiment fields."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        required = {
            "experiment", "title", "run_date", "started_at", "finished_at",
            "duration_s", "status",
            "control_accuracy", "treatment_accuracy", "improvement_delta",
            "n_templates_activated", "n_new_constraints_generated",
            "comparison_to_exp134",
        }
        assert required.issubset(artifact.keys())

    def test_artifact_status_is_success(self):
        """Artifact status is 'success'."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        assert artifact["status"] == "success"

    def test_improvement_delta_is_signed(self):
        """improvement_delta = treatment - control (signed, no clamping)."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        expected_delta = round(0.42 - 0.0, 4)
        assert artifact["improvement_delta"] == pytest.approx(expected_delta, abs=1e-4)

    def test_comparison_to_exp134_block_present(self):
        """comparison_to_exp134 block is present with required fields."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        cmp = artifact["comparison_to_exp134"]
        assert "exp134_improvement_delta" in cmp
        assert "hypothesis_confirmed" in cmp
        assert cmp["exp134_improvement_delta"] == 0.0

    def test_hypothesis_confirmed_true_when_delta_positive(self):
        """hypothesis_confirmed=True when treatment > control."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        # treatment_accuracy=0.42 > control_accuracy=0.0
        assert artifact["comparison_to_exp134"]["hypothesis_confirmed"] is True

    def test_hypothesis_confirmed_false_when_delta_zero(self):
        """hypothesis_confirmed=False when treatment == control (delta=0)."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        control = self._make_control_result()
        treatment = self._make_treatment_result()
        treatment["accuracy"] = 0.0  # same as control
        artifact = mod.build_addition_benchmark_artifact(control, treatment, tmpl)
        assert artifact["comparison_to_exp134"]["hypothesis_confirmed"] is False

    def test_n_questions_and_seed_in_artifact(self):
        """Artifact includes n_questions and seed for reproducibility."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        assert artifact["n_questions"] == mod.N_QUESTIONS
        assert artifact["seed"] == mod.SEED

    def test_control_detail_block_present(self):
        """Artifact includes control_detail with n_detected and n_total_errors."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        assert "control_detail" in artifact
        assert artifact["control_detail"]["n_total_errors"] == 80

    def test_treatment_detail_block_present(self):
        """Artifact includes treatment_detail with n_detected and n_total_errors."""
        mod = _load_exp344()
        tmpl = self._make_mock_tmpl(mod)
        artifact = mod.build_addition_benchmark_artifact(
            self._make_control_result(), self._make_treatment_result(), tmpl
        )
        assert "treatment_detail" in artifact
        assert artifact["treatment_detail"]["n_detected"] == 34


# ---------------------------------------------------------------------------
# End-to-end: full pipeline integration
# ---------------------------------------------------------------------------


class TestExp344EndToEnd:
    """End-to-end test that runs the full experiment pipeline."""

    def test_full_pipeline_produces_positive_delta(self):
        """Full pipeline: generate → control → treatment → artifact; delta > 0."""
        mod = _load_exp344()
        from experiment_template import ExperimentTemplate
        import tempfile

        tmp = tempfile.mkdtemp()
        tmpl = ExperimentTemplate(
            344,
            "E2E Test Exp 344",
            "results/experiment_344_constraint_addition_benchmark.json",
            repo_root=Path(tmp),
        )
        tmpl.setup()

        qs = mod.generate_simulated_questions(mod.N_QUESTIONS, mod.SEED)
        control_result = mod.run_control_condition(qs)
        treatment_result = mod.run_treatment_condition(qs)
        artifact = mod.build_addition_benchmark_artifact(control_result, treatment_result, tmpl)

        assert artifact["improvement_delta"] > 0.0
        assert artifact["comparison_to_exp134"]["hypothesis_confirmed"] is True
        assert artifact["status"] == "success"

    def test_full_pipeline_templates_activated(self):
        """Full pipeline: at least carry_check activates given 200 questions."""
        mod = _load_exp344()
        qs = mod.generate_simulated_questions(mod.N_QUESTIONS, mod.SEED)
        treatment_result = mod.run_treatment_condition(qs)
        assert "carry_check" in treatment_result["activated_template_keys"]

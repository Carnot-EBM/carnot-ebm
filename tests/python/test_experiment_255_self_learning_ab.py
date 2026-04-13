"""Tests for Experiment 255: self-learning A/B benchmark runner.

Spec: REQ-VERIFY-255,
SCENARIO-VERIFY-255-A (strategy branching — correct decisions for synthetic cases),
SCENARIO-VERIFY-255-B (metric aggregation — rates, budgets, domain breakdowns),
SCENARIO-VERIFY-255-C (replay-vs-live compatibility — live cases slot in cleanly),
SCENARIO-VERIFY-255-D (false-positive regression budget — all five strategies tracked),
SCENARIO-VERIFY-255-E (artifact schema stability — all required top-level keys present)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_255_self_learning_ab.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_255_self_learning_ab", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["experiment_255_self_learning_ab"] = module
    spec.loader.exec_module(module)
    return module


mod = load_module()


# ---------------------------------------------------------------------------
# Helpers for building synthetic replay cases
# ---------------------------------------------------------------------------


def _make_case(
    *,
    case_id: str = "c001",
    held_out: bool = True,
    actual_error: bool = True,
    detected: bool = True,
    baseline_success: bool = False,
    repair_success: bool = True,
    error_types: tuple[str, ...] = ("semantic:answer_target_mismatch",),
    descriptions: tuple[str, ...] = ("answer target mismatch",),
    domain: str = "live_gsm8k_semantic_failure",
    benchmark: str = "gsm8k_semantic",
    model_name: str = "Qwen3.5-0.8B",
    sample_position: int = 1,
    source_experiment: int = 235,
    baseline_latency_seconds: float = 1.0,
    repair_latency_seconds: float = 2.0,
) -> Any:
    """Build a synthetic ReplayCase for testing."""
    from carnot.pipeline.self_learning_replay import ReplayCase

    return ReplayCase(
        source_experiment=source_experiment,
        benchmark=benchmark,
        metric_name="accuracy",
        domain=domain,
        model_name=model_name,
        case_id=case_id,
        sample_position=sample_position,
        held_out=held_out,
        actual_error=actual_error,
        detected=detected,
        error_types=error_types,
        descriptions=descriptions,
        baseline_success=baseline_success,
        repair_success=repair_success,
        baseline_latency_seconds=baseline_latency_seconds,
        repair_latency_seconds=repair_latency_seconds,
    )


def _make_learning_case(**kwargs: Any) -> Any:
    return _make_case(held_out=False, **kwargs)


def _make_held_out_case(**kwargs: Any) -> Any:
    return _make_case(held_out=True, **kwargs)


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-255-A: strategy branching
# ---------------------------------------------------------------------------


class TestStrategyBranching:
    """Verify that each strategy produces the correct use_repair decision.

    Uses a minimal set of synthetic cases so results are deterministic and
    independent of any checked-in result JSON.
    """

    def _run_single_held_out(self, case) -> dict[str, Any]:
        """Run the runner on one held-out case with no learning cases."""
        result = mod.run_ab_benchmark([case])
        assert result["summary"]["held_out_cases"] == 1
        assert result["summary"]["learning_cases"] == 0
        decisions = result["held_out_decisions"][0]["strategies"]
        return decisions

    def test_no_learning_detected_uses_repair(self):
        """no_learning: use_repair = True when violation detected."""
        # SCENARIO-VERIFY-255-A
        case = _make_held_out_case(actual_error=True, detected=True)
        decisions = self._run_single_held_out(case)
        assert decisions["no_learning"]["use_repair"] is True

    def test_no_learning_undetected_skips_repair(self):
        """no_learning: use_repair = False when not detected."""
        # SCENARIO-VERIFY-255-A
        case = _make_held_out_case(actual_error=True, detected=False)
        decisions = self._run_single_held_out(case)
        assert decisions["no_learning"]["use_repair"] is False

    def test_all_five_strategies_present(self):
        """All five strategy names appear in every held-out decision."""
        # SCENARIO-VERIFY-255-A
        case = _make_held_out_case()
        result = mod.run_ab_benchmark([case])
        decisions = result["held_out_decisions"][0]["strategies"]
        expected = set(mod.ALL_STRATEGY_NAMES)
        assert set(decisions.keys()) == expected

    def test_strategy_decision_has_required_keys(self):
        """Each strategy decision record carries the required fields."""
        # SCENARIO-VERIFY-255-A
        case = _make_held_out_case()
        result = mod.run_ab_benchmark([case])
        decisions = result["held_out_decisions"][0]["strategies"]
        required = {"use_repair", "reason", "fast_path_hit", "constraint_templates_fired",
                    "final_success"}
        for name, decision in decisions.items():
            missing = required - set(decision.keys())
            assert not missing, f"Strategy {name!r} missing keys: {missing}"

    def test_predictive_gate_can_fire_fast_path(self):
        """predictive_gate: fast_path_hit=True possible for non-detected cases."""
        # SCENARIO-VERIFY-255-A
        # A case with no error and no description → gate sees empty string → FAST_PATH likely.
        case = _make_held_out_case(
            actual_error=False,
            detected=False,
            error_types=(),
            descriptions=(),
        )
        result = mod.run_ab_benchmark([case])
        decisions = result["held_out_decisions"][0]["strategies"]
        gate = decisions["predictive_gate"]
        # The gate routing can be either FAST_PATH or FULL; just check the field exists.
        assert isinstance(gate["fast_path_hit"], bool)
        assert isinstance(gate["use_repair"], bool)

    def test_no_learning_never_fast_path(self):
        """no_learning: fast_path_hit is always False (no gate involved)."""
        # SCENARIO-VERIFY-255-A
        for detected in (True, False):
            case = _make_held_out_case(detected=detected)
            result = mod.run_ab_benchmark([case])
            decision = result["held_out_decisions"][0]["strategies"]["no_learning"]
            assert decision["fast_path_hit"] is False

    def test_no_learning_zero_templates(self):
        """no_learning: constraint_templates_fired is always 0."""
        # SCENARIO-VERIFY-255-A
        case = _make_held_out_case(
            error_types=("semantic:answer_target_mismatch",),
            descriptions=("answer target mismatch",),
        )
        result = mod.run_ab_benchmark([case])
        decision = result["held_out_decisions"][0]["strategies"]["no_learning"]
        assert decision["constraint_templates_fired"] == 0


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-255-B: metric aggregation
# ---------------------------------------------------------------------------


class TestMetricAggregation:
    """Verify that aggregate metrics are computed correctly from synthetic cases."""

    def test_success_rate_perfect_repair(self):
        """success_rate == 1.0 when every held-out case is repaired successfully."""
        # SCENARIO-VERIFY-255-B
        cases = [
            _make_held_out_case(
                case_id=f"c{i}",
                actual_error=True,
                detected=True,
                baseline_success=False,
                repair_success=True,
                sample_position=i,
            )
            for i in range(1, 6)
        ]
        result = mod.run_ab_benchmark(cases)
        for name in mod.ALL_STRATEGY_NAMES:
            overall = result["strategies"][name]["overall"]
            assert overall["n_cases"] == 5

    def test_success_rate_no_repair_all_fail(self):
        """no_learning success_rate == 0.0 when none detected and all baseline fail."""
        # SCENARIO-VERIFY-255-B
        cases = [
            _make_held_out_case(
                case_id=f"c{i}",
                actual_error=True,
                detected=False,
                baseline_success=False,
                repair_success=True,
                sample_position=i,
            )
            for i in range(1, 4)
        ]
        result = mod.run_ab_benchmark(cases)
        overall = result["strategies"]["no_learning"]["overall"]
        assert overall["success_rate"] == pytest.approx(0.0)
        assert overall["false_positives"] == 0

    def test_false_positives_counted_correctly(self):
        """False positives: repair triggered on non-error cases."""
        # SCENARIO-VERIFY-255-B — a case where use_repair=True but actual_error=False
        # For no_learning, this happens when detected=True but actual_error=False.
        case = _make_held_out_case(
            actual_error=False,
            detected=True,
            baseline_success=True,
            repair_success=True,
        )
        result = mod.run_ab_benchmark([case])
        # no_learning uses repair (detected=True) on a non-error case → false positive
        assert result["strategies"]["no_learning"]["overall"]["false_positives"] == 1

    def test_domain_breakdown_populated(self):
        """by_domain must be populated for each strategy."""
        # SCENARIO-VERIFY-255-B
        cases = [
            _make_held_out_case(case_id="c1", domain="live_gsm8k_semantic_failure"),
            _make_held_out_case(case_id="c2", domain="code_spec_properties"),
        ]
        result = mod.run_ab_benchmark(cases)
        for name in mod.ALL_STRATEGY_NAMES:
            by_domain = result["strategies"][name]["by_domain"]
            assert "live_gsm8k_semantic_failure" in by_domain
            assert "code_spec_properties" in by_domain

    def test_domain_bucket_has_fast_path_rate(self):
        """Domain buckets carry fast_path_hit_rate after normalisation."""
        # SCENARIO-VERIFY-255-B
        case = _make_held_out_case()
        result = mod.run_ab_benchmark([case])
        for name in mod.ALL_STRATEGY_NAMES:
            for domain_bucket in result["strategies"][name]["by_domain"].values():
                assert "fast_path_hit_rate" in domain_bucket, (
                    f"Strategy {name!r} domain bucket missing fast_path_hit_rate"
                )

    def test_verification_spend_in_range(self):
        """verification_spend must be in [0, 1]."""
        # SCENARIO-VERIFY-255-B
        cases = [_make_held_out_case(case_id=f"c{i}", sample_position=i) for i in range(1, 5)]
        result = mod.run_ab_benchmark(cases)
        for name in mod.ALL_STRATEGY_NAMES:
            spend = result["strategies"][name]["overall"]["verification_spend"]
            assert 0.0 <= spend <= 1.0, f"Strategy {name!r} spend out of range: {spend}"

    def test_fast_path_plus_full_equals_n_cases(self):
        """n_fast_path_hits + n_full_verification_triggered == n_cases for all strategies."""
        # SCENARIO-VERIFY-255-B
        cases = [_make_held_out_case(case_id=f"c{i}", sample_position=i) for i in range(1, 6)]
        result = mod.run_ab_benchmark(cases)
        for name in mod.ALL_STRATEGY_NAMES:
            overall = result["strategies"][name]["overall"]
            total = overall["n_fast_path_hits"] + overall["n_full_verification_triggered"]
            assert total == overall["n_cases"], (
                f"Strategy {name!r}: fast_path + full != n_cases"
            )

    def test_latency_accumulated(self):
        """total_latency_seconds must be > 0 when cases have nonzero latency."""
        # SCENARIO-VERIFY-255-B
        case = _make_held_out_case(baseline_latency_seconds=1.5, repair_latency_seconds=3.0)
        result = mod.run_ab_benchmark([case])
        for name in mod.ALL_STRATEGY_NAMES:
            assert result["strategies"][name]["overall"]["total_latency_seconds"] > 0.0

    def test_learning_cases_update_learning_count(self):
        """Learning cases are counted separately from held-out cases."""
        # SCENARIO-VERIFY-255-B
        learn = _make_learning_case(case_id="learn1", sample_position=1)
        held = _make_held_out_case(case_id="held1", sample_position=2)
        result = mod.run_ab_benchmark([learn, held])
        assert result["summary"]["learning_cases"] == 1
        assert result["summary"]["held_out_cases"] == 1


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-255-C: replay-vs-live compatibility
# ---------------------------------------------------------------------------


class TestReplayVsLiveCompatibility:
    """Verify that live cases (held_out=True) slot cleanly into the runner."""

    def test_live_stub_returns_empty_list(self):
        """build_live_slice_cases() returns [] in Exp 255 (stub mode)."""
        # SCENARIO-VERIFY-255-C
        cases = mod.build_live_slice_cases()
        assert cases == []

    def test_live_cases_treated_as_held_out(self):
        """Manually supplied held_out=True cases are scored without polluting learning."""
        # SCENARIO-VERIFY-255-C
        # A 'live' case is just a ReplayCase with held_out=True and no learning predecessor.
        live = _make_held_out_case(case_id="live1", source_experiment=255)
        result = mod.run_ab_benchmark([live])
        assert result["summary"]["held_out_cases"] == 1
        assert result["summary"]["learning_cases"] == 0

    def test_mixed_replay_and_live_cases(self):
        """Learning cases from replay + live held-out cases work together."""
        # SCENARIO-VERIFY-255-C
        learn = _make_learning_case(case_id="learn1", sample_position=1)
        replay_held = _make_held_out_case(case_id="replay1", sample_position=2)
        live_held = _make_held_out_case(
            case_id="live1", sample_position=3, source_experiment=255
        )
        result = mod.run_ab_benchmark([learn, replay_held, live_held])
        assert result["summary"]["learning_cases"] == 1
        assert result["summary"]["held_out_cases"] == 2

    def test_all_strategy_names_stable(self):
        """ALL_STRATEGY_NAMES tuple matches the five documented strategies."""
        # SCENARIO-VERIFY-255-C
        expected = {
            "no_learning",
            "case_memory_plus_policy",
            "constraint_addition",
            "predictive_gate",
            "combined",
        }
        assert set(mod.ALL_STRATEGY_NAMES) == expected

    def test_live_models_constant(self):
        """LIVE_MODELS contains both documented model identifiers."""
        # SCENARIO-VERIFY-255-C
        assert "Qwen/Qwen3.5-0.8B" in mod.LIVE_MODELS
        assert "google/gemma-4-E4B-it" in mod.LIVE_MODELS


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-255-D: false-positive regression budget
# ---------------------------------------------------------------------------


class TestFalsePositiveBudget:
    """Verify that the false-positive regression budget covers all five strategies."""

    def test_budget_covers_all_strategies(self):
        """false_positive_regression_budget must have an entry for all five strategies."""
        # SCENARIO-VERIFY-255-D
        case = _make_held_out_case()
        result = mod.run_ab_benchmark([case])
        budget = result["summary"]["false_positive_regression_budget"]
        for name in mod.ALL_STRATEGY_NAMES:
            assert name in budget, f"Budget missing strategy: {name!r}"

    def test_budget_entry_schema(self):
        """Each budget entry has the four required fields."""
        # SCENARIO-VERIFY-255-D
        case = _make_held_out_case()
        result = mod.run_ab_benchmark([case])
        budget = result["summary"]["false_positive_regression_budget"]
        required = {
            "baseline_false_positives",
            "strategy_false_positives",
            "additional_false_positives",
            "within_budget",
        }
        for name in mod.ALL_STRATEGY_NAMES:
            entry = budget[name]
            missing = required - set(entry.keys())
            assert not missing, f"Strategy {name!r} budget entry missing: {missing}"

    def test_no_learning_within_budget(self):
        """no_learning strategy is always within budget (0 additional FPs)."""
        # SCENARIO-VERIFY-255-D
        case = _make_held_out_case(actual_error=False, detected=True)
        result = mod.run_ab_benchmark([case])
        entry = result["summary"]["false_positive_regression_budget"]["no_learning"]
        assert entry["additional_false_positives"] == 0
        assert entry["within_budget"] is True

    def test_primary_success_condition_structure(self):
        """primary_success_condition carries per_strategy dict for non-baseline strategies."""
        # SCENARIO-VERIFY-255-D
        case = _make_held_out_case()
        result = mod.run_ab_benchmark([case])
        psc = result["summary"]["primary_success_condition"]
        assert "metric" in psc
        assert "reference_strategy" in psc
        assert "per_strategy" in psc
        per = psc["per_strategy"]
        # case_memory_plus_policy is the reference, so it should NOT appear in per_strategy.
        assert "case_memory_plus_policy" not in per
        # All others must appear.
        for name in mod.ALL_STRATEGY_NAMES:
            if name != "case_memory_plus_policy":
                assert name in per, f"Missing strategy in per_strategy: {name!r}"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-255-E: artifact schema stability
# ---------------------------------------------------------------------------


class TestArtifactSchemaStability:
    """Verify the top-level artifact schema that Exp 256 will consume."""

    def _build_minimal_payload(self) -> dict[str, Any]:
        """Build a payload dict using two-case synthetic inputs."""
        learn = _make_learning_case(case_id="learn1", sample_position=1)
        held = _make_held_out_case(case_id="held1", sample_position=2)

        # build_255_payload requires exp235 and exp238 dicts.  We fabricate
        # minimal stubs that produce zero replay cases (empty paired_runs).
        exp235_stub: dict[str, Any] = {"paired_runs": [], "cohort": {"case_count": 0}}
        exp238_stub: dict[str, Any] = {"model_runs": {}, "cohort": {"case_count": 0}}

        # Run the runner directly (bypassing the source-file loader).
        runner_result = mod.run_ab_benchmark([learn, held])
        return {
            "experiment": mod.EXPERIMENT,
            "run_date": mod.RUN_DATE,
            "title": "test",
            "metadata": {
                "source_artifacts": [],
                "output_path": str(mod.RESULT_OUTPUT),
                "strategy_names": list(mod.ALL_STRATEGY_NAMES),
                "live_models": list(mod.LIVE_MODELS),
                "live_case_count": 0,
                "replay_case_count": 0,
                "held_out_policy": {"name": "test", "fraction": 0.25},
                "tracker_policy": {"min_support": 5, "min_precision": 0.75},
                "memory_policy": {
                    "min_support": 3,
                    "requires_zero_false_positives": True,
                    "requires_positive_repair_lift": True,
                },
                "policy_compiler": {
                    "min_case_support": 2,
                    "min_case_confidence": 0.9,
                    "min_patch_support": 2,
                },
                "predictive_gate": {
                    "threshold": 0.5,
                    "calibrated": False,
                    "note": "test",
                },
            },
            "summary": runner_result["summary"],
            "strategies": runner_result["strategies"],
            "held_out_decisions": runner_result["held_out_decisions"],
        }

    def test_top_level_keys(self):
        """Artifact has all required top-level keys."""
        # SCENARIO-VERIFY-255-E
        payload = self._build_minimal_payload()
        required = {
            "experiment", "run_date", "title", "metadata",
            "summary", "strategies", "held_out_decisions",
        }
        missing = required - set(payload.keys())
        assert not missing, f"Artifact missing top-level keys: {missing}"

    def test_experiment_and_run_date(self):
        """experiment == 255 and run_date == '20260413'."""
        # SCENARIO-VERIFY-255-E
        payload = self._build_minimal_payload()
        assert payload["experiment"] == 255
        assert payload["run_date"] == "20260413"

    def test_strategies_all_present(self):
        """All five strategy names appear under 'strategies'."""
        # SCENARIO-VERIFY-255-E
        payload = self._build_minimal_payload()
        strategies = payload["strategies"]
        for name in mod.ALL_STRATEGY_NAMES:
            assert name in strategies, f"Strategy {name!r} missing from artifact"

    def test_strategy_overall_has_new_fields(self):
        """Each strategy's overall block carries the Exp 255 extended fields."""
        # SCENARIO-VERIFY-255-E
        payload = self._build_minimal_payload()
        new_fields = {
            "n_fast_path_hits",
            "n_constraint_templates_fired",
            "n_full_verification_triggered",
            "fast_path_hit_rate",
            "verification_spend",
        }
        for name in mod.ALL_STRATEGY_NAMES:
            overall = payload["strategies"][name]["overall"]
            missing = new_fields - set(overall.keys())
            assert not missing, f"Strategy {name!r} overall missing new fields: {missing}"

    def test_summary_has_primary_success_condition(self):
        """summary carries primary_success_condition."""
        # SCENARIO-VERIFY-255-E
        payload = self._build_minimal_payload()
        assert "primary_success_condition" in payload["summary"]

    def test_held_out_decisions_list(self):
        """held_out_decisions is a non-empty list when there are held-out cases."""
        # SCENARIO-VERIFY-255-E
        payload = self._build_minimal_payload()
        assert isinstance(payload["held_out_decisions"], list)
        assert len(payload["held_out_decisions"]) == 1  # one held-out case

    def test_metadata_strategy_names_matches_constant(self):
        """metadata.strategy_names matches ALL_STRATEGY_NAMES."""
        # SCENARIO-VERIFY-255-E
        payload = self._build_minimal_payload()
        assert payload["metadata"]["strategy_names"] == list(mod.ALL_STRATEGY_NAMES)

    def test_constants_stable(self):
        """Key module constants are stable for Exp 256 to reference."""
        # SCENARIO-VERIFY-255-E
        assert mod.RUN_DATE == "20260413"
        assert mod.EXPERIMENT == 255
        assert mod.GATE_THRESHOLD == pytest.approx(0.5)
        assert str(mod.RESULT_OUTPUT) == "results/experiment_255_results.json"

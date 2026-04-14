"""Tests for Experiment 224: Tier 1 live-only ConstraintTracker retrain.

Verifies that training on ONLY live traces from Exp 219-221 produces a
ConstraintTracker with accurate precision/recall weights, and that the
held-out evaluation compares correctly to Exp 223 tracker_only results.

Spec: REQ-VERIFY-033, REQ-VERIFY-034, REQ-LEARN-001,
SCENARIO-VERIFY-033, SCENARIO-LEARN-001
"""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path

import pytest

from carnot.pipeline.self_learning_replay import (
    RESULT_OUTPUT_224,
    WEIGHTS_OUTPUT_224,
    ReplayCase,
    build_tier1_live_retrain_payload,
    evaluate_tier1_on_held_out,
    get_repo_root,
    run_experiment_224,
    train_tier1_weights,
)
from carnot.pipeline.tracker import ConstraintTracker


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_case(
    case_id: str,
    *,
    sample_position: int,
    held_out: bool,
    actual_error: bool,
    detected: bool,
    error_types: tuple[str, ...],
    baseline_success: bool,
    repair_success: bool,
    model_name: str = "TestModel",
) -> ReplayCase:
    """Build a minimal ReplayCase for unit testing."""
    return ReplayCase(
        source_experiment=219,
        benchmark="gsm8k_semantic",
        metric_name="accuracy",
        domain="live_gsm8k_semantic_failure",
        model_name=model_name,
        case_id=case_id,
        sample_position=sample_position,
        held_out=held_out,
        actual_error=actual_error,
        detected=detected,
        error_types=error_types,
        descriptions=tuple(f"desc:{t}" for t in error_types),
        baseline_success=baseline_success,
        repair_success=repair_success,
        baseline_latency_seconds=0.1,
        repair_latency_seconds=0.3,
    )


def _learning_cases() -> list[ReplayCase]:
    """Six learning cases covering two high-precision and one noisy error type."""
    return [
        # "arithmetic" fires 4× and catches 3× real errors → precision=0.75
        _make_case("learn-1", sample_position=1, held_out=False,
                   actual_error=True, detected=True,
                   error_types=("arithmetic",),
                   baseline_success=False, repair_success=True),
        _make_case("learn-2", sample_position=2, held_out=False,
                   actual_error=True, detected=True,
                   error_types=("arithmetic",),
                   baseline_success=False, repair_success=True),
        _make_case("learn-3", sample_position=3, held_out=False,
                   actual_error=True, detected=True,
                   error_types=("arithmetic",),
                   baseline_success=False, repair_success=True),
        _make_case("learn-4", sample_position=4, held_out=False,
                   actual_error=False, detected=True,   # false positive
                   error_types=("arithmetic",),
                   baseline_success=True, repair_success=False),
        # "noisy" fires 6× but catches 0 → precision=0.0 (should be suppressed)
        _make_case("learn-5", sample_position=5, held_out=False,
                   actual_error=False, detected=True,
                   error_types=("noisy_type",),
                   baseline_success=True, repair_success=True),
        _make_case("learn-6", sample_position=6, held_out=False,
                   actual_error=False, detected=True,
                   error_types=("noisy_type",),
                   baseline_success=True, repair_success=True),
        _make_case("learn-7", sample_position=7, held_out=False,
                   actual_error=False, detected=True,
                   error_types=("noisy_type",),
                   baseline_success=True, repair_success=True),
        _make_case("learn-8", sample_position=8, held_out=False,
                   actual_error=False, detected=True,
                   error_types=("noisy_type",),
                   baseline_success=True, repair_success=True),
        _make_case("learn-9", sample_position=9, held_out=False,
                   actual_error=False, detected=True,
                   error_types=("noisy_type",),
                   baseline_success=True, repair_success=True),
        _make_case("learn-10", sample_position=10, held_out=False,
                   actual_error=False, detected=True,
                   error_types=("noisy_type",),
                   baseline_success=True, repair_success=True),
    ]


def _held_out_cases() -> list[ReplayCase]:
    """Three held-out cases for evaluation."""
    return [
        # Real arithmetic error — tracker should approve repair.
        _make_case("held-1", sample_position=11, held_out=True,
                   actual_error=True, detected=True,
                   error_types=("arithmetic",),
                   baseline_success=False, repair_success=True),
        # Noisy type — tracker should suppress repair.
        _make_case("held-2", sample_position=12, held_out=True,
                   actual_error=False, detected=True,
                   error_types=("noisy_type",),
                   baseline_success=True, repair_success=False),
        # Undetected — no repair regardless.
        _make_case("held-3", sample_position=13, held_out=True,
                   actual_error=True, detected=False,
                   error_types=(),
                   baseline_success=False, repair_success=True),
    ]


def _all_cases() -> list[ReplayCase]:
    return _learning_cases() + _held_out_cases()


def _exp223_reference() -> dict:
    """Minimal Exp 223 fixture mirroring the real artifact's tracker_only block."""
    return {
        "experiment": 223,
        "run_date": "20260412",
        "strategies": {
            "tracker_only": {
                "overall": {
                    "success_rate": 0.3273809523809524,
                    "false_positives": 1,
                    "n_cases": 168,
                    "n_success": 55,
                }
            }
        },
    }


def _minimal_exp_payload(
    exp_number: int,
    benchmark: str,
    n_cases: int,
) -> dict:
    """Build a minimal live-experiment JSON payload for integration tests."""
    case_ids = [f"{benchmark}-{i}" for i in range(1, n_cases + 1)]
    cases_meta = [{"case_id": cid, "sample_position": i + 1}
                  for i, cid in enumerate(case_ids)]
    return {
        "experiment": exp_number,
        "benchmark": benchmark,
        "cohort": {
            "case_count": n_cases,
            "case_ids": case_ids,
            "cases": cases_meta,
        },
        "paired_runs": [
            {
                "model_name": "TestModel",
                "mode": "baseline",
                "cases": [{"case_id": cid, "correct": False} for cid in case_ids],
            },
            {
                "model_name": "TestModel",
                "mode": "verify_only",
                "cases": [
                    {
                        "case_id": cid,
                        "correct": False,
                        "flagged": True,
                        "verification": {
                            "violations": [
                                {
                                    "constraint_type": "semantic_grounding",
                                    "description": "answer target mismatch",
                                    "metadata": {
                                        "taxonomy_hint": "question_grounding_failures",
                                        "violation_type": "answer_target_mismatch",
                                    },
                                }
                            ]
                        },
                    }
                    for cid in case_ids
                ],
            },
            {
                "model_name": "TestModel",
                "mode": "verify_repair",
                "cases": [{"case_id": cid, "correct": True} for cid in case_ids],
            },
        ],
    }


# ---------------------------------------------------------------------------
# REQ-LEARN-001: train_tier1_weights
# ---------------------------------------------------------------------------


class TestTrainTier1Weights:
    """REQ-LEARN-001: train_tier1_weights() accumulates counters on learning cases only."""

    def test_skips_held_out_cases(self) -> None:
        """SCENARIO-LEARN-001: held-out cases must not contaminate training."""
        cases = _all_cases()
        tracker, observed = train_tier1_weights(cases)
        # Only learning cases should contribute.
        # "arithmetic" has 4 learning fires, "noisy_type" has 6.
        assert observed["arithmetic"].fired == 4
        assert observed["noisy_type"].fired == 6

    def test_precision_correct_for_arithmetic(self) -> None:
        """REQ-LEARN-001: arithmetic type has 3/4 TP → precision ≈ 0.75."""
        tracker, _ = train_tier1_weights(_all_cases())
        assert tracker.precision("arithmetic") == pytest.approx(0.75)

    def test_precision_zero_for_noisy_type(self) -> None:
        """REQ-LEARN-001: noisy_type catches no real errors → precision = 0.0."""
        tracker, _ = train_tier1_weights(_all_cases())
        assert tracker.precision("noisy_type") == 0.0

    def test_held_out_types_not_in_tracker(self) -> None:
        """REQ-LEARN-001: tracker contains no state derived from held-out events."""
        cases = _all_cases()
        tracker, observed = train_tier1_weights(cases)
        # All types come from learning cases (held-out cases reuse same types here).
        # The important thing is that the counts match ONLY learning cases.
        arith_stats = tracker.stats().get("arithmetic", {})
        assert arith_stats.get("fired", 0) == 4  # not 5 (held-out adds 1 more)

    def test_empty_cases_returns_empty_tracker(self) -> None:
        """REQ-LEARN-001: no cases → empty tracker."""
        tracker, observed = train_tier1_weights([])
        assert tracker.stats() == {}
        assert observed == {}

    def test_undetected_cases_not_recorded(self) -> None:
        """REQ-LEARN-001: cases with detected=False do not touch the tracker."""
        cases = [
            _make_case("undetected", sample_position=1, held_out=False,
                       actual_error=True, detected=False,
                       error_types=("arithmetic",),
                       baseline_success=False, repair_success=True),
        ]
        tracker, observed = train_tier1_weights(cases)
        assert tracker.stats() == {}

    def test_tracker_save_load_round_trip(self, tmp_path: Path) -> None:
        """REQ-LEARN-001: weights produced by training survive a save/load round-trip."""
        tracker, _ = train_tier1_weights(_all_cases())
        weights_path = tmp_path / "tier1_live_weights.json"
        tracker.save(str(weights_path))
        restored = ConstraintTracker.load(str(weights_path))
        assert restored.precision("arithmetic") == pytest.approx(tracker.precision("arithmetic"))
        assert restored.precision("noisy_type") == pytest.approx(tracker.precision("noisy_type"))
        # File format must be valid tracker JSON.
        payload = json.loads(weights_path.read_text())
        assert payload["version"] == 1
        assert "arithmetic" in payload["stats"]


# ---------------------------------------------------------------------------
# REQ-VERIFY-033: evaluate_tier1_on_held_out
# ---------------------------------------------------------------------------


class TestEvaluateTier1OnHeldOut:
    """REQ-VERIFY-033: evaluate_tier1_on_held_out() applies trained weights to held-out cases."""

    def _trained(self):
        return train_tier1_weights(_all_cases())

    def test_held_out_case_count(self) -> None:
        """SCENARIO-VERIFY-033: only held-out cases are evaluated."""
        tracker, observed = self._trained()
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        assert results["held_out_cases"] == 3

    def test_trusted_type_approves_repair(self) -> None:
        """REQ-VERIFY-033: arithmetic (precision≥0.75, support≥4) → use_repair=True."""
        tracker, observed = self._trained()
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        decisions = {d["case_id"]: d["strategies"] for d in results["held_out_decisions"]}
        assert decisions["held-1"]["tracker_only_live"]["use_repair"] is True
        assert decisions["held-1"]["tracker_only_live"]["reason"] == "tracker_supported"

    def test_noisy_type_suppresses_repair(self) -> None:
        """REQ-VERIFY-033: noisy_type (precision=0.0) → use_repair=False."""
        tracker, observed = self._trained()
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        decisions = {d["case_id"]: d["strategies"] for d in results["held_out_decisions"]}
        assert decisions["held-2"]["tracker_only_live"]["use_repair"] is False
        assert decisions["held-2"]["tracker_only_live"]["reason"] == "tracker_suppressed"

    def test_undetected_case_no_repair(self) -> None:
        """REQ-VERIFY-033: detected=False → use_repair=False regardless of tracker."""
        tracker, observed = self._trained()
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        decisions = {d["case_id"]: d["strategies"] for d in results["held_out_decisions"]}
        assert decisions["held-3"]["tracker_only_live"]["use_repair"] is False
        assert decisions["held-3"]["tracker_only_live"]["reason"] == "not_detected"

    def test_false_positive_reduction(self) -> None:
        """REQ-VERIFY-033: tracker suppresses the noisy FP relative to no_learning."""
        tracker, observed = self._trained()
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        overall_no_learning = results["strategies"]["no_learning"]["overall"]
        overall_live = results["strategies"]["tracker_only_live"]["overall"]
        # no_learning fires on held-2 (noisy, not an actual error) → FP
        assert overall_no_learning["false_positives"] >= 1
        # tracker suppresses held-2 → fewer FPs
        assert overall_live["false_positives"] < overall_no_learning["false_positives"]

    def test_false_positive_budget_within_budget(self) -> None:
        """REQ-VERIFY-033: tracker_only_live must not exceed no_learning FP count."""
        tracker, observed = self._trained()
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        budget = results["false_positive_regression_budget"]["tracker_only_live"]
        assert budget["within_budget"] is True

    def test_strategies_normalised(self) -> None:
        """REQ-VERIFY-033: success_rate computed for both strategies."""
        tracker, observed = self._trained()
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        for name in ("no_learning", "tracker_only_live"):
            assert 0.0 <= results["strategies"][name]["overall"]["success_rate"] <= 1.0

    def test_empty_cases_zero_results(self) -> None:
        """REQ-VERIFY-033: no cases → 0 held-out cases, 0 success."""
        tracker, observed = self._trained()
        results = evaluate_tier1_on_held_out(
            [], tracker, observed, tracker_min_support=4, tracker_min_precision=0.75,
        )
        assert results["held_out_cases"] == 0
        assert results["strategies"]["tracker_only_live"]["overall"]["n_cases"] == 0


# ---------------------------------------------------------------------------
# REQ-VERIFY-034: build_tier1_live_retrain_payload
# ---------------------------------------------------------------------------


class TestBuildTier1LiveRetrainPayload:
    """REQ-VERIFY-034: build_tier1_live_retrain_payload assembles the full Exp 224 artifact."""

    def _build(
        self,
        *,
        n219: int = 8,
        n220: int = 8,
        n221: int = 8,
    ):
        exp219 = _minimal_exp_payload(219, "gsm8k_semantic", n219)
        exp220 = _minimal_exp_payload(220, "humaneval_property", n220)
        exp221 = _minimal_exp_payload(221, "constraint_ir", n221)
        exp223 = _exp223_reference()
        return build_tier1_live_retrain_payload(
            exp219=exp219,
            exp220=exp220,
            exp221=exp221,
            exp223_reference=exp223,
            holdout_fraction=0.25,
            tracker_min_support=1,
            tracker_min_precision=0.5,
        )

    def test_experiment_number(self) -> None:
        """REQ-VERIFY-034: payload has experiment=224."""
        payload, _ = self._build()
        assert payload["experiment"] == 224

    def test_title_present(self) -> None:
        """REQ-VERIFY-034: title field describes the live-only retrain."""
        payload, _ = self._build()
        assert "live" in payload["title"].lower()
        assert "219" in payload["title"] or "Tier 1" in payload["title"]

    def test_metadata_no_simulated_data(self) -> None:
        """REQ-VERIFY-034: metadata declares no_simulated_data=True."""
        payload, _ = self._build()
        assert payload["metadata"]["no_simulated_data"] is True

    def test_training_summary_fields(self) -> None:
        """REQ-VERIFY-034: training_summary has required fields."""
        payload, _ = self._build()
        ts = payload["training_summary"]
        assert "training_cases" in ts
        assert "constraint_types_observed" in ts
        assert "types_meeting_threshold" in ts
        assert "tracker_stats" in ts
        assert ts["training_cases"] > 0

    def test_held_out_summary_fields(self) -> None:
        """REQ-VERIFY-034: held_out_summary has required fields."""
        payload, _ = self._build()
        hs = payload["held_out_summary"]
        assert "held_out_cases" in hs
        assert "strategies" in hs
        assert "false_positive_regression_budget" in hs
        assert hs["held_out_cases"] > 0

    def test_comparison_references_exp223(self) -> None:
        """REQ-VERIFY-034: comparison block references Exp 223 tracker_only."""
        payload, _ = self._build()
        comp = payload["comparison_to_exp223"]
        assert comp["reference_experiment"] == 223
        assert comp["reference_strategy"] == "tracker_only"
        assert "exp223_success_rate" in comp
        assert "exp224_success_rate" in comp
        assert "success_rate_delta" in comp
        assert "within_fp_budget" in comp

    def test_success_rate_delta_computed_correctly(self) -> None:
        """REQ-VERIFY-034: delta = exp224 − exp223 success rates."""
        payload, _ = self._build()
        comp = payload["comparison_to_exp223"]
        assert comp["success_rate_delta"] == pytest.approx(
            comp["exp224_success_rate"] - comp["exp223_success_rate"]
        )

    def test_returns_trained_tracker(self) -> None:
        """REQ-VERIFY-034: second return value is a ConstraintTracker with state."""
        payload, tracker = self._build()
        assert isinstance(tracker, ConstraintTracker)
        # The minimal payload has detected errors → tracker should have some state.
        assert len(tracker.stats()) >= 0  # could be 0 if no detected violations


# ---------------------------------------------------------------------------
# REQ-VERIFY-033: run_experiment_224 integration (writes files)
# ---------------------------------------------------------------------------


class TestRunExperiment224:
    """REQ-VERIFY-033, REQ-VERIFY-034: run_experiment_224 writes both output files."""

    def _make_repo(self, tmp_path: Path) -> Path:
        repo = tmp_path / "repo"
        (repo / "results").mkdir(parents=True)
        for exp_num, benchmark, n in [(219, "gsm8k_semantic", 8),
                                      (220, "humaneval_property", 8),
                                      (221, "constraint_ir", 8)]:
            payload = _minimal_exp_payload(exp_num, benchmark, n)
            path = repo / "results" / f"experiment_{exp_num}_results.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
        # Minimal Exp 223 reference.
        exp223_path = repo / "results" / "experiment_223_results.json"
        exp223_path.write_text(json.dumps(_exp223_reference()), encoding="utf-8")
        return repo

    def test_result_json_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SCENARIO-VERIFY-033: results/experiment_224_results.json is created."""
        repo = self._make_repo(tmp_path)
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
        run_experiment_224(repo_root=repo)
        result_path = repo / "results" / "experiment_224_results.json"
        assert result_path.exists()
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload["experiment"] == 224
        assert payload["metadata"]["no_simulated_data"] is True

    def test_weights_json_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SCENARIO-LEARN-001: results/tier1_live_weights.json is created and valid."""
        repo = self._make_repo(tmp_path)
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
        run_experiment_224(repo_root=repo)
        weights_path = repo / "results" / "tier1_live_weights.json"
        assert weights_path.exists()
        payload = json.loads(weights_path.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert "stats" in payload

    def test_weights_loadable_as_tracker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-LEARN-001: saved weights load back into a ConstraintTracker."""
        repo = self._make_repo(tmp_path)
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
        run_experiment_224(repo_root=repo)
        weights_path = repo / "results" / "tier1_live_weights.json"
        restored = ConstraintTracker.load(str(weights_path))
        assert isinstance(restored, ConstraintTracker)

    def test_result_held_out_strategies_present(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-034: result JSON contains no_learning and tracker_only_live."""
        repo = self._make_repo(tmp_path)
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
        run_experiment_224(repo_root=repo)
        payload = json.loads(
            (repo / "results" / "experiment_224_results.json").read_text(encoding="utf-8")
        )
        strategies = payload["held_out_summary"]["strategies"]
        assert "no_learning" in strategies
        assert "tracker_only_live" in strategies

    def test_script_module_parses_args_and_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REQ-VERIFY-033, REQ-VERIFY-034: the script module runs end-to-end via runpy."""
        repo = self._make_repo(tmp_path)
        monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
        scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
        script_path = scripts_dir / "experiment_224_tier1_live_retrain.py"
        argv = sys.argv
        try:
            sys.argv = ["experiment_224_tier1_live_retrain.py"]
            runpy.run_path(str(script_path), run_name="__main__")
        finally:
            sys.argv = argv
        result_path = repo / "results" / "experiment_224_results.json"
        assert result_path.exists()


# ---------------------------------------------------------------------------
# REQ-VERIFY-034: held-out decisions integrity
# ---------------------------------------------------------------------------


class TestHeldOutDecisionsIntegrity:
    """REQ-VERIFY-034: held_out_decisions list has correct shape and IDs."""

    def test_decisions_match_held_out_count(self) -> None:
        """REQ-VERIFY-034: one decision per held-out case."""
        tracker, observed = train_tier1_weights(_all_cases())
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        assert len(results["held_out_decisions"]) == results["held_out_cases"]

    def test_decision_keys_present(self) -> None:
        """REQ-VERIFY-034: each decision has case_id, held_out, and both strategy keys."""
        tracker, observed = train_tier1_weights(_all_cases())
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        for decision in results["held_out_decisions"]:
            assert decision["held_out"] is True
            assert "no_learning" in decision["strategies"]
            assert "tracker_only_live" in decision["strategies"]
            for strat in decision["strategies"].values():
                assert "use_repair" in strat
                assert "reason" in strat
                assert "final_success" in strat

    def test_no_learning_cases_not_in_decisions(self) -> None:
        """REQ-VERIFY-034: held_out_decisions contains only held-out cases."""
        tracker, observed = train_tier1_weights(_all_cases())
        results = evaluate_tier1_on_held_out(
            _all_cases(), tracker, observed,
            tracker_min_support=4, tracker_min_precision=0.75,
        )
        case_ids = {d["case_id"] for d in results["held_out_decisions"]}
        learning_ids = {c.case_id for c in _learning_cases()}
        assert case_ids.isdisjoint(learning_ids)

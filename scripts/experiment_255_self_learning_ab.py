#!/usr/bin/env python3
"""Experiment 255: Self-learning A/B benchmark runner.

Compares five learning strategies on held-out replay cases from Exp 241:

  no_learning               — passthrough baseline; use repair iff violation detected
  case_memory_plus_policy   — current best (Exp 241) tracker + case memory + compiled policy
  constraint_addition       — additive constraint templates compiled from case memory
  predictive_gate           — lightweight logistic gate decides FAST_PATH vs FULL
  combined                  — predictive_gate routing with constraint_addition on FULL path

Both honest chronological replay (using Exp 241 held-out cases) and a small
live slice (on Qwen/Qwen3.5-0.8B and google/gemma-4-E4B-it) are supported.
The live-slice path is wired up but not executed in this task; Exp 256 will
drive execution and analysis.

Per-strategy metrics preserved in the artifact:
  held-out task success rate, false positives, verification spend,
  fast-path hit rate, latency (carried from replay cases), domain breakdowns.

Writes (when run as __main__):
  results/experiment_255_results.json

Run date: 20260413

Spec: REQ-VERIFY-255,
SCENARIO-VERIFY-255-A (strategy branching),
SCENARIO-VERIFY-255-B (metric aggregation),
SCENARIO-VERIFY-255-C (replay-vs-live compatibility),
SCENARIO-VERIFY-255-D (false-positive regression budget),
SCENARIO-VERIFY-255-E (artifact schema stability)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from carnot.pipeline.case_memory import CaseMemory
from carnot.pipeline.constraint_addition import (
    ConstraintAdditionCompiler,
    ConstraintAdditionRegistry,
    ConstraintAdditionResult,
)
from carnot.pipeline.memory import ConstraintMemory
from carnot.pipeline.predictive_verifier import (
    ROUTE_FAST_PATH,
    ROUTE_FULL,
    PredictiveVerifier,
)
from carnot.pipeline.self_learning_policy import PolicyQuery, SelfLearningPolicyCompiler
from carnot.pipeline.self_learning_replay import (
    HOLDOUT_FRACTION,
    MEMORY_MIN_SUPPORT,
    POLICY_MIN_CASE_CONFIDENCE,
    POLICY_MIN_CASE_SUPPORT,
    POLICY_MIN_PATCH_SUPPORT,
    TRACKER_MIN_PRECISION,
    TRACKER_MIN_SUPPORT,
    ReplayCase,
    _ObservedPatternStats,
    _ObservedTypeStats,
    _case_record_for_replay_case,
    _dedupe_preserve,
    _description_for_error_type,
    _normalise_strategy,
    _policy_context_for_replay_case,
    _policy_decision,
    _record_strategy_outcome,
    _tracker_decision,
    _update_transfer_effects,
    build_exp241_replay_cases,
)
from carnot.pipeline.tracker import ConstraintTracker

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

RUN_DATE: str = "20260413"
EXPERIMENT: int = 255
RESULT_OUTPUT: Path = Path("results/experiment_255_results.json")

# Models supported for the live-slice path (not executed in Exp 255).
LIVE_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3.5-0.8B",
    "google/gemma-4-E4B-it",
)

# Source artifacts for replay (same as Exp 241).
SOURCE_ARTIFACTS: tuple[Path, ...] = (
    Path("results/experiment_235_results.json"),
    Path("results/experiment_238_results.json"),
)

# Reference artifact for delta comparison.
EXP241_REFERENCE: Path = Path("results/experiment_241_results.json")

# Predictive gate threshold (same convention as PredictiveVerifier.gate()).
GATE_THRESHOLD: float = 0.5

# All strategy names produced by this runner — stable for Exp 256.
ALL_STRATEGY_NAMES: tuple[str, ...] = (
    "no_learning",
    "case_memory_plus_policy",
    "constraint_addition",
    "predictive_gate",
    "combined",
)

# ---------------------------------------------------------------------------
# Internal decision dataclass (mirrors self_learning_replay._Decision)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Decision255:
    """Routing decision for a single Exp 255 strategy evaluation.

    Fields
    ------
    use_repair
        Whether this strategy triggered the repair path for this case.
    reason
        Short human-readable reason string for traceability.
    fast_path_hit
        True when the predictive gate routed this case to FAST_PATH (skip
        full verification).  Always False for strategies that do not use
        the predictive gate.
    constraint_templates_fired
        Number of constraint templates that fired for this case.  Zero for
        strategies that do not use constraint addition.
    support_models
        Models whose prior cases influenced this decision (from policy/memory).
    candidate_case_keys
        Case key fingerprints considered during memory retrieval.
    matched_case_keys
        Case key fingerprints that were positively matched.
    """

    use_repair: bool
    reason: str
    fast_path_hit: bool = False
    constraint_templates_fired: int = 0
    support_models: tuple[str, ...] = ()
    candidate_case_keys: tuple[str, ...] = ()
    matched_case_keys: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Extended strategy initialiser
# ---------------------------------------------------------------------------


def _empty_strategy_255(name: str, *, track_retrieval: bool = False) -> dict[str, Any]:
    """Return a zeroed Exp 255 strategy record.

    Extends the Exp 241 structure with:
      n_fast_path_hits            — cases routed to FAST_PATH by the predictive gate
      n_constraint_templates_fired— cases where ≥1 constraint template fired
      n_full_verification_triggered — cases that proceeded to full verification
      fast_path_hit_rate          — computed at normalisation time
      verification_spend          — fraction of cases triggering full verification
    """
    return {
        "name": name,
        "track_retrieval": track_retrieval,
        "overall": {
            "n_cases": 0,
            "n_success": 0,
            "false_positives": 0,
            "n_repairs_used": 0,
            # Retrieval fields (populated for strategies that track retrieval).
            "helpful_memory_reuse_events": 0,
            "retrieval_candidate_events": 0,
            "retrieval_candidates": 0,
            "retrieval_hit_events": 0,
            "retrieval_hits": 0,
            # Latency fields (from replay case latency records).
            "total_latency_seconds": 0.0,
            "baseline_reference_latency_seconds": 0.0,
            # New Exp 255 fields.
            "n_fast_path_hits": 0,
            "n_constraint_templates_fired": 0,
            "n_full_verification_triggered": 0,
        },
        "by_domain": {},
        "by_metric": {},
        "by_benchmark": {},
        "by_model": {},
        "over_time": {},
    }


def _normalise_strategy_255(strategy: dict[str, Any]) -> None:
    """Normalise an Exp 255 strategy record in-place.

    Computes derived ratios after all cases have been recorded.  Delegates
    the base normalisation to the shared ``_normalise_strategy`` helper from
    self_learning_replay, then appends the Exp 255-specific ratios.
    """
    _normalise_strategy(strategy)
    overall = strategy["overall"]
    n_cases = overall["n_cases"]
    overall["fast_path_hit_rate"] = (
        overall["n_fast_path_hits"] / n_cases if n_cases else 0.0
    )
    overall["verification_spend"] = (
        overall["n_full_verification_triggered"] / n_cases if n_cases else 0.0
    )
    for bucket_name in ("by_domain",):
        for bucket in strategy[bucket_name].values():
            n = bucket.get("n_cases", 0)
            bucket["success_rate"] = bucket["n_success"] / n if n else 0.0
            bucket["fast_path_hit_rate"] = (
                bucket.get("n_fast_path_hits", 0) / n if n else 0.0
            )
            bucket["verification_spend"] = (
                bucket.get("n_full_verification_triggered", 0) / n if n else 0.0
            )


# ---------------------------------------------------------------------------
# Per-case strategy decisions
# ---------------------------------------------------------------------------


def _base_decision_from_replay(case: ReplayCase) -> _Decision255:
    """no_learning baseline: repair iff violation was detected."""
    return _Decision255(
        use_repair=case.detected,
        reason="detected" if case.detected else "baseline_only",
        fast_path_hit=False,
        constraint_templates_fired=0,
    )


def _constraint_addition_decision(
    case: ReplayCase,
    *,
    registry: ConstraintAdditionRegistry,
    base_decision: _Decision255,
) -> _Decision255:
    """Augment *base_decision* with additive constraint templates.

    Builds a synthetic response text from the case's description strings so
    that ``registry.apply()``'s text-pattern guard can operate in the replay
    setting.  A template firing is interpreted as a signal that the verifier
    would have detected a pattern matching the template — thus requiring repair.

    If the registry fires any template, ``use_repair`` is forced True regardless
    of the base decision.  This is intentionally conservative: the constraint
    templates were compiled from cases where the verifier failed, so a match
    means "this response looks like a failure family we've seen before".
    """
    benchmark_slice = f"{case.benchmark}/{case.domain}"
    violation_types = case.error_types
    response_text = " ".join(case.descriptions) if case.descriptions else ""

    active = registry.apply(
        case.model_name,
        benchmark_slice,
        violation_types,
        response_text,
    )
    n_fired = len(active)

    if n_fired > 0:
        # Templates fired — treat as requiring repair.
        return _Decision255(
            use_repair=True,
            reason="constraint_template_fired",
            fast_path_hit=False,
            constraint_templates_fired=n_fired,
            support_models=base_decision.support_models,
            candidate_case_keys=base_decision.candidate_case_keys,
            matched_case_keys=base_decision.matched_case_keys,
        )

    # No templates fired — fall back to base decision.
    return _Decision255(
        use_repair=base_decision.use_repair,
        reason=base_decision.reason,
        fast_path_hit=False,
        constraint_templates_fired=0,
        support_models=base_decision.support_models,
        candidate_case_keys=base_decision.candidate_case_keys,
        matched_case_keys=base_decision.matched_case_keys,
    )


def _predictive_gate_decision(
    case: ReplayCase,
    *,
    verifier: PredictiveVerifier,
    base_decision: _Decision255,
    threshold: float = GATE_THRESHOLD,
) -> _Decision255:
    """Route the case through the predictive gate.

    Uses the case description text as a stand-in for the partial response.
    FAST_PATH → skip full verification → no repair (optimistic assumption).
    FULL → fall back to base_decision (the gate defers to existing strategy).

    Rationale: in replay we cannot re-run the LLM, so we use the description
    text as a proxy for the partial response that the gate would have seen.
    This under-estimates feature richness but provides an honest lower bound
    on what the gate could achieve with real partial responses.
    """
    response_proxy = " ".join(case.descriptions) if case.descriptions else ""
    # Infer domain: reasoning benchmarks get domain="reasoning", else "code".
    domain = "reasoning" if "gsm8k" in case.benchmark.lower() else "code"
    prior_confidence = 0.5  # neutral prior for replay

    gate = verifier.gate(response_proxy, threshold, domain, prior_confidence)

    if gate.route == ROUTE_FAST_PATH:
        return _Decision255(
            use_repair=False,
            reason="predictive_gate_fast_path",
            fast_path_hit=True,
            constraint_templates_fired=0,
            support_models=base_decision.support_models,
            candidate_case_keys=base_decision.candidate_case_keys,
            matched_case_keys=base_decision.matched_case_keys,
        )

    # FULL — gate is uncertain or predicts violation; defer to base.
    return _Decision255(
        use_repair=base_decision.use_repair,
        reason=f"predictive_gate_full:{base_decision.reason}",
        fast_path_hit=False,
        constraint_templates_fired=0,
        support_models=base_decision.support_models,
        candidate_case_keys=base_decision.candidate_case_keys,
        matched_case_keys=base_decision.matched_case_keys,
    )


def _combined_decision(
    case: ReplayCase,
    *,
    registry: ConstraintAdditionRegistry,
    verifier: PredictiveVerifier,
    base_decision: _Decision255,
    threshold: float = GATE_THRESHOLD,
) -> _Decision255:
    """Combined predictive_gate + constraint_addition strategy.

    Gate is applied first:
      FAST_PATH → skip verification entirely (no repair).
      FULL      → apply constraint_addition templates to guide the verifier.

    This means the predictive gate acts as a cost-saving pre-filter: cheap
    cases skip the whole stack, while uncertain cases get the extra guidance
    from compiled constraint templates.
    """
    response_proxy = " ".join(case.descriptions) if case.descriptions else ""
    domain = "reasoning" if "gsm8k" in case.benchmark.lower() else "code"
    gate = verifier.gate(response_proxy, threshold, domain, prior_confidence=0.5)

    if gate.route == ROUTE_FAST_PATH:
        return _Decision255(
            use_repair=False,
            reason="combined_fast_path",
            fast_path_hit=True,
            constraint_templates_fired=0,
            support_models=base_decision.support_models,
            candidate_case_keys=base_decision.candidate_case_keys,
            matched_case_keys=base_decision.matched_case_keys,
        )

    # FULL path — apply constraint addition on top of base decision.
    ca_decision = _constraint_addition_decision(
        case, registry=registry, base_decision=base_decision
    )
    return _Decision255(
        use_repair=ca_decision.use_repair,
        reason=f"combined_full:{ca_decision.reason}",
        fast_path_hit=False,
        constraint_templates_fired=ca_decision.constraint_templates_fired,
        support_models=base_decision.support_models,
        candidate_case_keys=base_decision.candidate_case_keys,
        matched_case_keys=base_decision.matched_case_keys,
    )


# ---------------------------------------------------------------------------
# Metric recording
# ---------------------------------------------------------------------------


def _record_outcome_255(
    strategy: dict[str, Any],
    case: ReplayCase,
    decision: _Decision255,
) -> None:
    """Update *strategy* counters for one case/decision pair.

    Extends the base ``_record_strategy_outcome`` logic with Exp 255-specific
    counters (fast-path hits, constraint templates fired, full verification
    triggered).  Latency is taken directly from the replay case records.
    """
    success = case.success_for(decision.use_repair)
    overall = strategy["overall"]

    overall["n_cases"] += 1
    overall["n_success"] += int(success)
    overall["false_positives"] += int(decision.use_repair and not case.actual_error)
    overall["n_repairs_used"] += int(decision.use_repair)
    overall["n_fast_path_hits"] += int(decision.fast_path_hit)
    overall["n_constraint_templates_fired"] += int(decision.constraint_templates_fired > 0)
    overall["n_full_verification_triggered"] += int(not decision.fast_path_hit)

    # Latency from replay records.
    latency = (
        case.repair_latency_seconds if decision.use_repair else case.baseline_latency_seconds
    )
    overall["total_latency_seconds"] += latency
    overall["baseline_reference_latency_seconds"] += case.baseline_latency_seconds

    # Domain breakdown.
    by_domain = strategy["by_domain"].setdefault(
        case.domain,
        {
            "n_cases": 0,
            "n_success": 0,
            "false_positives": 0,
            "n_fast_path_hits": 0,
            "n_full_verification_triggered": 0,
        },
    )
    by_domain["n_cases"] += 1
    by_domain["n_success"] += int(success)
    by_domain["false_positives"] += int(decision.use_repair and not case.actual_error)
    by_domain["n_fast_path_hits"] += int(decision.fast_path_hit)
    by_domain["n_full_verification_triggered"] += int(not decision.fast_path_hit)

    # Metric, benchmark, and model breakdowns (reuse existing counters).
    by_metric = strategy["by_metric"].setdefault(
        case.metric_name, {"n_cases": 0, "n_success": 0, "false_positives": 0}
    )
    by_metric["n_cases"] += 1
    by_metric["n_success"] += int(success)
    by_metric["false_positives"] += int(decision.use_repair and not case.actual_error)

    by_benchmark = strategy["by_benchmark"].setdefault(
        case.benchmark, {"n_cases": 0, "n_success": 0, "false_positives": 0}
    )
    by_benchmark["n_cases"] += 1
    by_benchmark["n_success"] += int(success)
    by_benchmark["false_positives"] += int(decision.use_repair and not case.actual_error)

    by_model = strategy["by_model"].setdefault(
        case.model_name, {"n_cases": 0, "n_success": 0, "false_positives": 0}
    )
    by_model["n_cases"] += 1
    by_model["n_success"] += int(success)
    by_model["false_positives"] += int(decision.use_repair and not case.actual_error)

    metric_over_time = strategy["over_time"].setdefault(case.metric_name, [])
    by_metric_now = strategy["by_metric"][case.metric_name]
    metric_over_time.append(
        {
            "source_experiment": case.source_experiment,
            "sample_position": case.sample_position,
            "model_name": case.model_name,
            "case_id": case.case_id,
            "n_cases": by_metric_now["n_cases"],
            "success_rate": (
                by_metric_now["n_success"] / by_metric_now["n_cases"]
                if by_metric_now["n_cases"]
                else 0.0
            ),
        }
    )


# ---------------------------------------------------------------------------
# Primary success condition
# ---------------------------------------------------------------------------


def _primary_success_condition_255(
    strategies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate whether any new strategy beats case_memory_plus_policy.

    The primary condition from Exp 241 remains the floor: the candidate must
    show a non-negative success-rate gain over case_memory_plus_policy without
    introducing new false positives.  All five strategies are evaluated.
    """
    reference_name = "case_memory_plus_policy"
    reference_success = float(
        strategies[reference_name]["overall"].get("success_rate") or 0.0
    )
    no_learning_fps = int(strategies["no_learning"]["overall"]["false_positives"])

    results: dict[str, Any] = {}
    for strategy_name in ALL_STRATEGY_NAMES:
        if strategy_name == reference_name:
            continue
        overall = strategies[strategy_name]["overall"]
        success = float(overall.get("success_rate") or 0.0)
        fps = int(overall.get("false_positives") or 0)
        gain = success - reference_success
        additional_fps = fps - no_learning_fps
        met = gain > 0.0 and additional_fps <= 0
        results[strategy_name] = {
            "success_rate_gain_vs_case_memory_plus_policy": gain,
            "additional_false_positives_vs_no_learning": additional_fps,
            "met": met,
            "statement": "met" if met else "not_met",
        }

    return {
        "metric": "held_out_task_gain_with_no_extra_false_positives",
        "reference_strategy": reference_name,
        "no_learning_false_positives": no_learning_fps,
        "per_strategy": results,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_ab_benchmark(
    cases: list[ReplayCase],
    *,
    tracker_min_support: int = TRACKER_MIN_SUPPORT,
    tracker_min_precision: float = TRACKER_MIN_PRECISION,
    memory_min_support: int = MEMORY_MIN_SUPPORT,
    policy_min_case_support: int = POLICY_MIN_CASE_SUPPORT,
    policy_min_case_confidence: float = POLICY_MIN_CASE_CONFIDENCE,
    policy_min_patch_support: int = POLICY_MIN_PATCH_SUPPORT,
    gate_threshold: float = GATE_THRESHOLD,
) -> dict[str, Any]:
    """Compare all five Exp 255 strategies on *cases* in chronological order.

    Cases are replayed chronologically.  Learning cases (not held-out) update
    the shared tracker, memory, case_memory, and policy compiler exactly as in
    Exp 241.  Constraint templates and the predictive verifier are compiled /
    re-used at held-out evaluation time from the accumulated case_memory.

    Returns a dict with keys:
      summary            — held_out_cases, learning_cases, budgets, success condition
      strategies         — per-strategy metric dicts (schema stable for Exp 256)
      held_out_decisions — per-case decision records for all five strategies
    """
    ordered_cases = sorted(
        cases,
        key=lambda c: (c.source_experiment, c.sample_position, c.model_name, c.case_id),
    )

    # Shared learning state (same as Exp 241 v2).
    tracker = ConstraintTracker()
    memory = ConstraintMemory()
    case_memory = CaseMemory()
    observed_types: dict[str, _ObservedTypeStats] = {}
    observed_patterns: dict[tuple[str, str], _ObservedPatternStats] = {}
    policy_compiler = SelfLearningPolicyCompiler(
        min_case_support=policy_min_case_support,
        min_case_confidence=policy_min_case_confidence,
        min_patch_support=policy_min_patch_support,
    )

    # Predictive verifier with default weights (calibration on live data is
    # Exp 256's job; here we use the out-of-box conservative prior).
    verifier = PredictiveVerifier()

    strategies: dict[str, dict[str, Any]] = {
        "no_learning": _empty_strategy_255("no_learning"),
        "case_memory_plus_policy": _empty_strategy_255(
            "case_memory_plus_policy", track_retrieval=True
        ),
        "constraint_addition": _empty_strategy_255(
            "constraint_addition", track_retrieval=True
        ),
        "predictive_gate": _empty_strategy_255("predictive_gate"),
        "combined": _empty_strategy_255("combined", track_retrieval=True),
    }

    held_out_decisions: list[dict[str, Any]] = []
    held_out_cases = 0
    learning_cases = 0

    for case in ordered_cases:
        if case.held_out:
            held_out_cases += 1

            # ------------------------------------------------------------------
            # Build the shared base decisions (same as Exp 241).
            # ------------------------------------------------------------------
            no_learn_base = _base_decision_from_replay(case)

            tracker_decision_raw = _tracker_decision(
                case,
                tracker=tracker,
                observed_types=observed_types,
                tracker_min_support=tracker_min_support,
                tracker_min_precision=tracker_min_precision,
            )
            from carnot.pipeline.self_learning_replay import _memory_decision

            memory_decision_raw = _memory_decision(
                case,
                tracker_decision=tracker_decision_raw,
                tracker=tracker,
                observed_types=observed_types,
                observed_patterns=observed_patterns,
                case_memory=case_memory,
                memory_min_support=memory_min_support,
            )
            policy_ctx = _policy_context_for_replay_case(
                case,
                policy_compiler=policy_compiler,
                tracker=tracker,
                case_memory=case_memory,
            )
            cmpp_raw = _policy_decision(
                case,
                base_decision=memory_decision_raw,
                policy_context=policy_ctx,
            )

            # Wrap the Exp 241 decisions into _Decision255 wrappers.
            no_learning = _Decision255(
                use_repair=no_learn_base.use_repair,
                reason=no_learn_base.reason,
            )
            cmpp_decision = _Decision255(
                use_repair=cmpp_raw.use_repair,
                reason=cmpp_raw.reason,
                support_models=cmpp_raw.support_models,
                candidate_case_keys=cmpp_raw.candidate_case_keys,
                matched_case_keys=cmpp_raw.matched_case_keys,
            )

            # ------------------------------------------------------------------
            # Compile constraint templates lazily from accumulated case_memory.
            # ------------------------------------------------------------------
            ca_result: ConstraintAdditionResult = ConstraintAdditionCompiler().compile(
                case_memory
            )
            registry = ConstraintAdditionRegistry(ca_result)

            # ------------------------------------------------------------------
            # New strategy decisions.
            # ------------------------------------------------------------------
            ca_decision = _constraint_addition_decision(
                case, registry=registry, base_decision=cmpp_decision
            )
            gate_decision = _predictive_gate_decision(
                case, verifier=verifier, base_decision=cmpp_decision, threshold=gate_threshold
            )
            combined = _combined_decision(
                case,
                registry=registry,
                verifier=verifier,
                base_decision=cmpp_decision,
                threshold=gate_threshold,
            )

            # ------------------------------------------------------------------
            # Record outcomes for all five strategies.
            # ------------------------------------------------------------------
            _record_outcome_255(strategies["no_learning"], case, no_learning)
            _record_outcome_255(strategies["case_memory_plus_policy"], case, cmpp_decision)
            _record_outcome_255(strategies["constraint_addition"], case, ca_decision)
            _record_outcome_255(strategies["predictive_gate"], case, gate_decision)
            _record_outcome_255(strategies["combined"], case, combined)

            held_out_decisions.append(
                {
                    **case.to_dict(),
                    "strategies": {
                        "no_learning": {
                            "use_repair": no_learning.use_repair,
                            "reason": no_learning.reason,
                            "fast_path_hit": no_learning.fast_path_hit,
                            "constraint_templates_fired": no_learning.constraint_templates_fired,
                            "final_success": case.success_for(no_learning.use_repair),
                        },
                        "case_memory_plus_policy": {
                            "use_repair": cmpp_decision.use_repair,
                            "reason": cmpp_decision.reason,
                            "fast_path_hit": cmpp_decision.fast_path_hit,
                            "constraint_templates_fired": cmpp_decision.constraint_templates_fired,
                            "support_models": list(cmpp_decision.support_models),
                            "candidate_case_keys": list(cmpp_decision.candidate_case_keys),
                            "matched_case_keys": list(cmpp_decision.matched_case_keys),
                            "final_success": case.success_for(cmpp_decision.use_repair),
                        },
                        "constraint_addition": {
                            "use_repair": ca_decision.use_repair,
                            "reason": ca_decision.reason,
                            "fast_path_hit": ca_decision.fast_path_hit,
                            "constraint_templates_fired": ca_decision.constraint_templates_fired,
                            "final_success": case.success_for(ca_decision.use_repair),
                        },
                        "predictive_gate": {
                            "use_repair": gate_decision.use_repair,
                            "reason": gate_decision.reason,
                            "fast_path_hit": gate_decision.fast_path_hit,
                            "constraint_templates_fired": gate_decision.constraint_templates_fired,
                            "final_success": case.success_for(gate_decision.use_repair),
                        },
                        "combined": {
                            "use_repair": combined.use_repair,
                            "reason": combined.reason,
                            "fast_path_hit": combined.fast_path_hit,
                            "constraint_templates_fired": combined.constraint_templates_fired,
                            "final_success": case.success_for(combined.use_repair),
                        },
                    },
                }
            )
            continue

        # ------------------------------------------------------------------
        # Learning case — update shared state (identical to Exp 241).
        # ------------------------------------------------------------------
        learning_cases += 1
        if case.detected and case.error_types:
            for error_type in case.error_types:
                tracker.record(
                    error_type,
                    fired=True,
                    caught_error=case.actual_error,
                    any_error_in_batch=case.actual_error,
                )
                stats = observed_types.setdefault(error_type, _ObservedTypeStats())
                stats.fired += 1
                stats.true_positives += int(case.actual_error)
                stats.repair_improvements += int(
                    case.repair_success and not case.baseline_success
                )
                stats.repair_harms += int(case.baseline_success and not case.repair_success)
                stats.source_models.add(case.model_name)

        if case.detected and case.actual_error and case.error_types:
            for error_type in case.error_types:
                description = _description_for_error_type(case, error_type)
                memory.record_pattern(case.domain, error_type, description)
                pattern_stats = observed_patterns.setdefault(
                    (case.domain, error_type), _ObservedPatternStats()
                )
                pattern_stats.support += 1
                pattern_stats.repair_improvements += int(
                    case.repair_success and not case.baseline_success
                )
                pattern_stats.repair_harms += int(case.baseline_success and not case.repair_success)
                pattern_stats.source_models.add(case.model_name)
            case_memory.record(_case_record_for_replay_case(case))

    # Normalise all strategies.
    for strategy in strategies.values():
        _normalise_strategy_255(strategy)

    # False-positive regression budget.
    no_learning_fps = strategies["no_learning"]["overall"]["false_positives"]
    false_positive_regression_budget: dict[str, Any] = {
        "policy": "zero_additional_false_positives_vs_no_learning",
    }
    for strategy_name in ALL_STRATEGY_NAMES:
        strategy_fps = strategies[strategy_name]["overall"]["false_positives"]
        additional = strategy_fps - no_learning_fps
        false_positive_regression_budget[strategy_name] = {
            "baseline_false_positives": no_learning_fps,
            "strategy_false_positives": strategy_fps,
            "additional_false_positives": additional,
            "within_budget": additional <= 0,
        }

    summary = {
        "held_out_cases": held_out_cases,
        "learning_cases": learning_cases,
        "false_positive_regression_budget": false_positive_regression_budget,
        "primary_success_condition": _primary_success_condition_255(strategies),
    }

    return {
        "summary": summary,
        "strategies": strategies,
        "held_out_decisions": held_out_decisions,
    }


# ---------------------------------------------------------------------------
# Live-slice stub (wired; not executed in Exp 255)
# ---------------------------------------------------------------------------


def build_live_slice_cases(
    models: tuple[str, ...] = LIVE_MODELS,
    *,
    n_samples: int = 20,
) -> list[ReplayCase]:
    """Build a small live slice of cases from the specified models.

    This function is a stub for Exp 256 execution.  It documents the interface
    that the live-slice path must satisfy so Exp 256 can plug in real model
    inference without changing the runner schema.

    Args:
        models: HuggingFace model IDs to sample from.
        n_samples: Number of GSM8K questions to sample per model.

    Returns:
        Empty list in Exp 255 (live execution deferred to Exp 256).
    """
    # Exp 256 will replace this stub with real model inference.
    # The returned ReplayCase objects must have held_out=True so the runner
    # scores them without polluting the learning state.
    return []


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def build_255_payload(
    *,
    exp235: dict[str, Any],
    exp238: dict[str, Any],
    live_cases: list[ReplayCase] | None = None,
    holdout_fraction: float = HOLDOUT_FRACTION,
    tracker_min_support: int = TRACKER_MIN_SUPPORT,
    tracker_min_precision: float = TRACKER_MIN_PRECISION,
    memory_min_support: int = MEMORY_MIN_SUPPORT,
    policy_min_case_support: int = POLICY_MIN_CASE_SUPPORT,
    policy_min_case_confidence: float = POLICY_MIN_CASE_CONFIDENCE,
    policy_min_patch_support: int = POLICY_MIN_PATCH_SUPPORT,
    gate_threshold: float = GATE_THRESHOLD,
) -> dict[str, Any]:
    """Build the in-memory Exp 255 artifact without writing files.

    Combines honest chronological replay from Exp 241 replay cases with any
    live-slice cases supplied by the caller (empty in Exp 255).

    Args:
        exp235: Loaded JSON dict for experiment_235_results.json.
        exp238: Loaded JSON dict for experiment_238_results.json.
        live_cases: Additional held-out cases from live model inference
            (empty list or None means replay-only mode).
        holdout_fraction: Fraction of cases to reserve as held-out.
        tracker_min_support: ConstraintTracker minimum support threshold.
        tracker_min_precision: ConstraintTracker minimum precision threshold.
        memory_min_support: CaseMemory minimum support threshold.
        policy_min_case_support: Policy compiler minimum case support.
        policy_min_case_confidence: Policy compiler minimum case confidence.
        policy_min_patch_support: Policy compiler minimum patch support.
        gate_threshold: Predictive gate routing threshold (0..1).

    Returns:
        Artifact dict ready to serialise to JSON.
    """
    replay_cases = build_exp241_replay_cases(
        exp235=exp235,
        exp238=exp238,
        holdout_fraction=holdout_fraction,
    )
    all_cases = replay_cases + (live_cases or [])

    result = run_ab_benchmark(
        all_cases,
        tracker_min_support=tracker_min_support,
        tracker_min_precision=tracker_min_precision,
        memory_min_support=memory_min_support,
        policy_min_case_support=policy_min_case_support,
        policy_min_case_confidence=policy_min_case_confidence,
        policy_min_patch_support=policy_min_patch_support,
        gate_threshold=gate_threshold,
    )

    return {
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "title": "Self-learning A/B benchmark: constraint_addition vs predictive_gate vs combined",
        "metadata": {
            "source_artifacts": [str(p) for p in SOURCE_ARTIFACTS],
            "output_path": str(RESULT_OUTPUT),
            "strategy_names": list(ALL_STRATEGY_NAMES),
            "live_models": list(LIVE_MODELS),
            "live_case_count": len(live_cases) if live_cases else 0,
            "replay_case_count": len(replay_cases),
            "held_out_policy": {
                "name": "final_slice_per_source_artifact",
                "fraction": holdout_fraction,
            },
            "tracker_policy": {
                "min_support": tracker_min_support,
                "min_precision": tracker_min_precision,
            },
            "memory_policy": {
                "min_support": memory_min_support,
                "requires_zero_false_positives": True,
                "requires_positive_repair_lift": True,
            },
            "policy_compiler": {
                "min_case_support": policy_min_case_support,
                "min_case_confidence": policy_min_case_confidence,
                "min_patch_support": policy_min_patch_support,
            },
            "predictive_gate": {
                "threshold": gate_threshold,
                "calibrated": False,
                "note": "Default weights used; calibration is Exp 256's job.",
            },
        },
        "summary": result["summary"],
        "strategies": result["strategies"],
        "held_out_decisions": result["held_out_decisions"],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def get_repo_root() -> Path:
    """Resolve the repository root from the environment or file location."""
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 255: self-learning A/B benchmark runner"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Override repository root (default: auto-detect)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Override output path (default: {RESULT_OUTPUT})",
    )
    args = parser.parse_args()

    repo_root = args.repo_root or get_repo_root()
    output_path = args.output or (repo_root / RESULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def load(path: Path) -> dict[str, Any]:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    exp235 = load(repo_root / SOURCE_ARTIFACTS[0])
    exp238 = load(repo_root / SOURCE_ARTIFACTS[1])

    payload = build_255_payload(exp235=exp235, exp238=exp238)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"Wrote {output_path}")
    summary = payload["summary"]
    print(f"  held-out cases: {summary['held_out_cases']}")
    print(f"  learning cases: {summary['learning_cases']}")
    for sname in ALL_STRATEGY_NAMES:
        overall = payload["strategies"][sname]["overall"]
        print(
            f"  {sname}: success={overall.get('success_rate', 0):.3f}"
            f"  fps={overall['false_positives']}"
            f"  fast_path={overall.get('fast_path_hit_rate', 0):.3f}"
            f"  verif_spend={overall.get('verification_spend', 0):.3f}"
        )


if __name__ == "__main__":
    main()

"""Held-out live self-learning replay benchmark for Exp 223.

Spec: REQ-VERIFY-033, REQ-VERIFY-034, REQ-VERIFY-035,
SCENARIO-VERIFY-033, SCENARIO-VERIFY-034, SCENARIO-VERIFY-035
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.pipeline.case_memory import CaseMemory, CaseQuery, CaseRecord
from carnot.pipeline.live_trace_memory import (
    _extract_exp219_events,
    _extract_exp220_events,
    _extract_exp221_events,
    load_json,
    write_json,
)
from carnot.pipeline.memory import ConstraintMemory
from carnot.pipeline.self_learning_policy import PolicyQuery, SelfLearningPolicyCompiler
from carnot.pipeline.tracker import ConstraintTracker

RUN_DATE = "20260412"
RESULT_OUTPUT = Path("results/experiment_223_results.json")
MEMORY_SOURCE = Path("results/constraint_memory_live_222.json")
SOURCE_ARTIFACTS = (
    Path("results/experiment_219_results.json"),
    Path("results/experiment_220_results.json"),
    Path("results/experiment_221_results.json"),
)
HOLDOUT_FRACTION = 0.25
TRACKER_MIN_SUPPORT = 5
TRACKER_MIN_PRECISION = 0.75
MEMORY_MIN_SUPPORT = 3
RUN_DATE_V2 = "20260413"
RESULT_OUTPUT_V2 = Path("results/experiment_241_results.json")
SOURCE_ARTIFACTS_V2 = (
    Path("results/experiment_235_results.json"),
    Path("results/experiment_238_results.json"),
)
EXP223_REFERENCE = Path("results/experiment_223_results.json")
POLICY_MIN_CASE_SUPPORT = 2
POLICY_MIN_CASE_CONFIDENCE = 0.9
POLICY_MIN_PATCH_SUPPORT = 2


@dataclass(frozen=True)
class ReplayCase:
    """One replayable live case with paired baseline and repair outcomes."""

    source_experiment: int
    benchmark: str
    metric_name: str
    domain: str
    model_name: str
    case_id: str
    sample_position: int
    held_out: bool
    actual_error: bool
    detected: bool
    error_types: tuple[str, ...]
    descriptions: tuple[str, ...]
    baseline_success: bool
    repair_success: bool
    baseline_latency_seconds: float = 0.0
    repair_latency_seconds: float = 0.0

    def success_for(self, use_repair: bool) -> bool:
        return self.repair_success if use_repair else self.baseline_success

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_experiment": self.source_experiment,
            "benchmark": self.benchmark,
            "metric_name": self.metric_name,
            "domain": self.domain,
            "model_name": self.model_name,
            "case_id": self.case_id,
            "sample_position": self.sample_position,
            "held_out": self.held_out,
            "actual_error": self.actual_error,
            "detected": self.detected,
            "error_types": list(self.error_types),
            "descriptions": list(self.descriptions),
            "baseline_success": self.baseline_success,
            "repair_success": self.repair_success,
            "baseline_latency_seconds": self.baseline_latency_seconds,
            "repair_latency_seconds": self.repair_latency_seconds,
        }


@dataclass
class _ObservedTypeStats:
    fired: int = 0
    true_positives: int = 0
    repair_improvements: int = 0
    repair_harms: int = 0
    source_models: set[str] = field(default_factory=set)

    @property
    def precision(self) -> float:
        if self.fired == 0:
            return 0.0
        return self.true_positives / self.fired


@dataclass
class _ObservedPatternStats:
    support: int = 0
    repair_improvements: int = 0
    repair_harms: int = 0
    source_models: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Decision:
    use_repair: bool
    reason: str
    support_models: tuple[str, ...] = ()
    candidate_error_types: tuple[str, ...] = ()
    matched_error_types: tuple[str, ...] = ()
    candidate_case_keys: tuple[str, ...] = ()
    matched_case_keys: tuple[str, ...] = ()
    policy_threshold_ids: tuple[str, ...] = ()
    policy_property_budget_ids: tuple[str, ...] = ()
    policy_patch_ids: tuple[str, ...] = ()
    policy_routing_ids: tuple[str, ...] = ()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _dedupe_preserve(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return tuple(ordered)


def _empty_strategy(name: str, *, track_retrieval: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "track_retrieval": track_retrieval,
        "overall": {
            "n_cases": 0,
            "n_success": 0,
            "false_positives": 0,
            "n_repairs_used": 0,
            "helpful_memory_reuse_events": 0,
            "retrieval_candidate_events": 0,
            "retrieval_candidates": 0,
            "retrieval_hit_events": 0,
            "retrieval_hits": 0,
            "total_latency_seconds": 0.0,
            "baseline_reference_latency_seconds": 0.0,
        },
        "by_metric": {},
        "by_benchmark": {},
        "by_model": {},
        "over_time": {},
    }


def _latency_for_decision(case: ReplayCase, *, use_repair: bool) -> float:
    if use_repair and case.repair_latency_seconds:
        return case.repair_latency_seconds
    return case.baseline_latency_seconds


def get_repo_root() -> Path:
    override = os.environ.get("CARNOT_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _metric_name_for_benchmark(benchmark: str) -> str:
    if benchmark == "gsm8k_semantic":
        return "accuracy"
    if benchmark == "humaneval_property":
        return "pass_rate"
    return "constraint_satisfaction"


def _success_key_for_benchmark(benchmark: str) -> str:
    if benchmark == "gsm8k_semantic":
        return "correct"
    if benchmark == "humaneval_property":
        return "passed"
    return "exact_satisfaction"


def _sample_positions(payload: dict[str, Any]) -> tuple[dict[str, int], int]:
    cohort = payload.get("cohort")
    if not isinstance(cohort, dict):
        return {}, 0
    case_count = int(cohort.get("case_count") or 0)
    positions: dict[str, int] = {}
    cases = cohort.get("cases")
    if isinstance(cases, list):
        for case in cases:
            if not isinstance(case, dict) or not case.get("case_id"):
                continue
            sample_position = int(case.get("sample_position") or 0)
            if sample_position > 0:
                positions[str(case["case_id"])] = sample_position
    if positions:
        return positions, case_count or len(positions)
    case_ids = cohort.get("case_ids")
    if isinstance(case_ids, list):
        positions = {str(case_id): index + 1 for index, case_id in enumerate(case_ids)}
        return positions, case_count or len(positions)
    return {}, case_count


def _paired_case_lookup(
    payload: dict[str, Any],
) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    lookup: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for run in payload.get("paired_runs", []):
        if not isinstance(run, dict):
            continue
        model_name = str(run.get("model_name") or "")
        mode = str(run.get("mode") or "")
        cases_by_id: dict[str, dict[str, Any]] = {}
        for case in run.get("cases", []):
            if isinstance(case, dict) and case.get("case_id"):
                cases_by_id[str(case["case_id"])] = case
        lookup[(model_name, mode)] = cases_by_id
    return lookup


def _extract_exp235_events(payload: dict[str, Any]) -> list[Any]:
    events: list[Any] = []
    for run in payload.get("paired_runs", []):
        if not isinstance(run, dict) or run.get("mode") != "verify_only":
            continue
        model_name = str(run.get("model_name") or "")
        for case in run.get("cases", []):
            if not isinstance(case, dict):
                continue
            verification = _as_dict(case.get("verification"))
            raw_error_types: list[str] = []
            raw_descriptions: list[str] = []
            for violation in verification.get("violations", []):
                if not isinstance(violation, dict):
                    continue
                metadata = _as_dict(violation.get("metadata"))
                legacy_types = metadata.get("legacy_violation_types")
                if isinstance(legacy_types, list):
                    raw_error_types.extend(str(item) for item in legacy_types if item)
                violation_type = str(metadata.get("violation_type") or "").strip()
                taxonomy_hint = str(metadata.get("taxonomy_hint") or "").strip()
                if violation_type:
                    raw_error_types.append(
                        f"{taxonomy_hint}:{violation_type}" if taxonomy_hint else violation_type
                    )
                raw_descriptions.append(
                    str(
                        violation.get("description")
                        or violation.get("constraint_type")
                        or "semantic verification violation"
                    )
                )
            if not raw_error_types:
                certificate = _as_dict(verification.get("certificate"))
                semantic_v2 = _as_dict(certificate.get("semantic_verifier_v2"))
                legacy_types = semantic_v2.get("legacy_violation_types")
                if isinstance(legacy_types, list):
                    raw_error_types.extend(str(item) for item in legacy_types if item)
            events.append(
                SimpleNamespace(
                    source_experiment=235,
                    benchmark="gsm8k_semantic",
                    domain="live_gsm8k_semantic_failure",
                    model_name=model_name,
                    case_id=str(case.get("case_id") or ""),
                    actual_error=not bool(case.get("correct")),
                    detected=bool(case.get("flagged")),
                    error_types=_dedupe_preserve(raw_error_types),
                    descriptions=_dedupe_preserve(raw_descriptions)
                    or _dedupe_preserve([item.replace(":", " ") for item in raw_error_types]),
                )
            )
    return events


def _classify_exp238_text(text: str) -> str:
    lowered = text.lower()
    if "official_tests" in lowered or "assertionerror" in lowered or "nameerror" in lowered:
        return "official_test_failure"
    if "syntax" in lowered or "indentationerror" in lowered:
        return "syntax_error"
    if "no_exception" in lowered:
        return "no_exception"
    if "deterministic" in lowered:
        return "deterministic"
    return "humaneval_failure"


def _extract_exp238_errors(case: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    history = case.get("history")
    if not isinstance(history, list) or not history:
        return (), ()
    first = _as_dict(history[0])
    evaluation = _as_dict(first.get("evaluation"))
    pbt = _as_dict(evaluation.get("pbt"))
    explicit_specs = _as_dict(evaluation.get("explicit_specs"))
    official_tests = _as_dict(evaluation.get("official_tests"))

    raw_error_types: list[str] = []
    raw_descriptions: list[str] = []
    repair_hints = explicit_specs.get("repair_hints")
    if isinstance(repair_hints, list):
        for hint in repair_hints:
            if not isinstance(hint, dict):
                continue
            error_family = str(hint.get("error_family") or "").strip()
            if error_family:
                raw_error_types.append(error_family)
    for key in ("violations",):
        values = explicit_specs.get(key)
        if isinstance(values, list):
            raw_descriptions.extend(str(value) for value in values if value)
    for key in ("violations",):
        values = pbt.get(key)
        if isinstance(values, list):
            raw_descriptions.extend(str(value) for value in values if value)
    error_message = str(official_tests.get("error_message") or "").strip()
    if error_message:
        raw_descriptions.append(error_message)
    raw_error_types.extend(_classify_exp238_text(description) for description in raw_descriptions)
    return _dedupe_preserve(raw_error_types), _dedupe_preserve(raw_descriptions)


def _build_cases_for_payload(
    payload: dict[str, Any],
    events: list[Any],
    *,
    holdout_fraction: float,
) -> list[ReplayCase]:
    benchmark = str(payload.get("benchmark") or "")
    metric_name = _metric_name_for_benchmark(benchmark)
    success_key = _success_key_for_benchmark(benchmark)
    positions, case_count = _sample_positions(payload)
    holdout_start = math.floor(case_count * (1.0 - holdout_fraction))
    lookups = _paired_case_lookup(payload)
    cases: list[ReplayCase] = []
    for event in events:
        sample_position = positions.get(event.case_id, 0)
        baseline_case = lookups.get((event.model_name, "baseline"), {}).get(event.case_id, {})
        repair_case = lookups.get((event.model_name, "verify_repair"), {}).get(event.case_id, {})
        baseline_success = bool(baseline_case.get(success_key, not event.actual_error))
        repair_success = bool(repair_case.get(success_key, baseline_success))
        cases.append(
            ReplayCase(
                source_experiment=event.source_experiment,
                benchmark=event.benchmark,
                metric_name=metric_name,
                domain=event.domain,
                model_name=event.model_name,
                case_id=event.case_id,
                sample_position=sample_position,
                held_out=sample_position > holdout_start,
                actual_error=event.actual_error,
                detected=event.detected,
                error_types=event.error_types,
                descriptions=event.descriptions,
                baseline_success=baseline_success,
                repair_success=repair_success,
                baseline_latency_seconds=float(baseline_case.get("latency_seconds") or 0.0),
                repair_latency_seconds=float(
                    repair_case.get("latency_seconds")
                    or baseline_case.get("latency_seconds")
                    or 0.0
                ),
            )
        )
    return cases


def build_replay_cases(
    *,
    exp219: dict[str, Any],
    exp220: dict[str, Any],
    exp221: dict[str, Any],
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> list[ReplayCase]:
    """Build replay cases from the checked-in live Exp 219-221 artifacts."""

    cases = [
        *_build_cases_for_payload(
            exp219,
            _extract_exp219_events(exp219),
            holdout_fraction=holdout_fraction,
        ),
        *_build_cases_for_payload(
            exp220,
            _extract_exp220_events(exp220),
            holdout_fraction=holdout_fraction,
        ),
        *_build_cases_for_payload(
            exp221,
            _extract_exp221_events(exp221),
            holdout_fraction=holdout_fraction,
        ),
    ]
    return sorted(
        cases,
        key=lambda case: (
            case.source_experiment,
            case.sample_position,
            case.model_name,
            case.case_id,
        ),
    )


def _tracker_decision(
    case: ReplayCase,
    *,
    tracker: ConstraintTracker,
    observed_types: dict[str, _ObservedTypeStats],
    tracker_min_support: int,
    tracker_min_precision: float,
) -> _Decision:
    if not case.detected:
        return _Decision(use_repair=False, reason="not_detected")
    if not case.error_types:
        return _Decision(use_repair=True, reason="detected_without_error_type")
    support_models: set[str] = set()
    trusted_error_types: list[str] = []
    for error_type in case.error_types:
        stats = observed_types.get(error_type)
        if stats is None:
            continue
        support_models.update(stats.source_models)
        if (
            stats.fired >= tracker_min_support
            and tracker.precision(error_type) >= tracker_min_precision
        ):
            trusted_error_types.append(error_type)
    if trusted_error_types:
        return _Decision(
            use_repair=True,
            reason="tracker_supported",
            support_models=tuple(sorted(support_models)),
            matched_error_types=tuple(trusted_error_types),
        )
    return _Decision(
        use_repair=False,
        reason="tracker_suppressed",
        support_models=tuple(sorted(support_models)),
    )


def _memory_candidates(
    *,
    domain: str,
    tracker: ConstraintTracker,
    observed_types: dict[str, _ObservedTypeStats],
    observed_patterns: dict[tuple[str, str], _ObservedPatternStats],
    memory_min_support: int,
) -> list[tuple[str, _ObservedPatternStats]]:
    candidates: list[tuple[str, _ObservedPatternStats]] = []
    for (pattern_domain, error_type), pattern_stats in observed_patterns.items():
        if pattern_domain != domain:
            continue
        type_stats = observed_types.get(error_type)
        if type_stats is None:
            continue
        if pattern_stats.support < memory_min_support:
            continue
        if type_stats.fired < memory_min_support:
            continue
        if type_stats.true_positives != type_stats.fired:
            continue
        if pattern_stats.repair_improvements <= pattern_stats.repair_harms:
            continue
        candidates.append((error_type, pattern_stats))
    candidates.sort(key=lambda item: (-item[1].support, item[0]))
    return candidates


def _memory_decision(
    case: ReplayCase,
    *,
    tracker_decision: _Decision,
    tracker: ConstraintTracker,
    observed_types: dict[str, _ObservedTypeStats],
    observed_patterns: dict[tuple[str, str], _ObservedPatternStats],
    case_memory: CaseMemory | None = None,
    memory_min_support: int,
) -> _Decision:
    candidates = _memory_candidates(
        domain=case.domain,
        tracker=tracker,
        observed_types=observed_types,
        observed_patterns=observed_patterns,
        memory_min_support=memory_min_support,
    )
    candidate_error_types = tuple(error_type for error_type, _ in candidates)
    matched = tuple(
        error_type for error_type in candidate_error_types if error_type in case.error_types
    )
    case_matches = _case_matches_for_replay_case(
        case,
        case_memory=case_memory,
        min_support=memory_min_support,
    )
    candidate_case_keys = tuple(match.entry.key.fingerprint for match in case_matches)
    if tracker_decision.use_repair:
        return _Decision(
            use_repair=True,
            reason=tracker_decision.reason,
            support_models=tracker_decision.support_models,
            candidate_error_types=candidate_error_types,
            matched_error_types=matched,
            candidate_case_keys=candidate_case_keys,
            matched_case_keys=candidate_case_keys,
        )
    if not matched:
        if candidate_case_keys:
            case_support_models = tuple(
                sorted({match.entry.key.model_name for match in case_matches})
            )
            return _Decision(
                use_repair=True,
                reason="case_memory_reuse",
                support_models=case_support_models,
                candidate_error_types=candidate_error_types,
                candidate_case_keys=candidate_case_keys,
                matched_case_keys=candidate_case_keys,
            )
        return _Decision(
            use_repair=False,
            reason="no_memory_match",
            candidate_error_types=candidate_error_types,
            candidate_case_keys=candidate_case_keys,
        )
    matched_support_models: set[str] = set()
    for error_type in matched:
        stats = observed_patterns.get((case.domain, error_type))
        if stats is not None:
            matched_support_models.update(stats.source_models)
    return _Decision(
        use_repair=True,
        reason="memory_reuse",
        support_models=tuple(sorted(matched_support_models)),
        candidate_error_types=candidate_error_types,
        matched_error_types=matched,
        candidate_case_keys=candidate_case_keys,
        matched_case_keys=candidate_case_keys,
    )


def _record_strategy_outcome(
    strategy: dict[str, Any],
    case: ReplayCase,
    *,
    decision: _Decision,
) -> None:
    overall = strategy["overall"]
    overall["n_cases"] += 1
    success = case.success_for(decision.use_repair)
    overall["n_success"] += int(success)
    overall["false_positives"] += int(decision.use_repair and not case.actual_error)
    overall["n_repairs_used"] += int(decision.use_repair)
    overall["total_latency_seconds"] += _latency_for_decision(case, use_repair=decision.use_repair)
    overall["baseline_reference_latency_seconds"] += case.baseline_latency_seconds
    if decision.reason in {"memory_reuse", "case_memory_reuse"}:
        overall["helpful_memory_reuse_events"] += int(success and not case.baseline_success)
    if strategy.get("track_retrieval"):
        candidate_count = len(decision.candidate_error_types) or len(decision.candidate_case_keys)
        hit_count = len(decision.matched_error_types) or len(decision.matched_case_keys)
        overall["retrieval_candidate_events"] += int(candidate_count > 0)
        overall["retrieval_candidates"] += candidate_count
        overall["retrieval_hit_events"] += int(hit_count > 0)
        overall["retrieval_hits"] += hit_count

    by_metric = strategy["by_metric"].setdefault(
        case.metric_name,
        {"n_cases": 0, "n_success": 0, "false_positives": 0},
    )
    by_metric["n_cases"] += 1
    by_metric["n_success"] += int(success)
    by_metric["false_positives"] += int(decision.use_repair and not case.actual_error)

    by_benchmark = strategy["by_benchmark"].setdefault(
        case.benchmark,
        {"n_cases": 0, "n_success": 0, "false_positives": 0},
    )
    by_benchmark["n_cases"] += 1
    by_benchmark["n_success"] += int(success)
    by_benchmark["false_positives"] += int(decision.use_repair and not case.actual_error)

    by_model = strategy["by_model"].setdefault(
        case.model_name,
        {"n_cases": 0, "n_success": 0, "false_positives": 0},
    )
    by_model["n_cases"] += 1
    by_model["n_success"] += int(success)
    by_model["false_positives"] += int(decision.use_repair and not case.actual_error)

    metric_over_time = strategy["over_time"].setdefault(case.metric_name, [])
    metric_over_time.append(
        {
            "source_experiment": case.source_experiment,
            "sample_position": case.sample_position,
            "model_name": case.model_name,
            "case_id": case.case_id,
            "n_cases": by_metric["n_cases"],
            "success_rate": by_metric["n_success"] / by_metric["n_cases"],
        }
    )


def _normalise_strategy(strategy: dict[str, Any]) -> None:
    overall = strategy["overall"]
    n_cases = overall["n_cases"]
    overall["success_rate"] = overall["n_success"] / n_cases if n_cases else 0.0
    overall["mean_latency_seconds"] = overall["total_latency_seconds"] / n_cases if n_cases else 0.0
    overall["latency_overhead_seconds"] = (
        overall["total_latency_seconds"] - overall["baseline_reference_latency_seconds"]
    )
    overall["mean_latency_overhead_seconds"] = (
        overall["latency_overhead_seconds"] / n_cases if n_cases else 0.0
    )
    overall["retrieval_hit_rate"] = (
        overall["retrieval_hit_events"] / overall["retrieval_candidate_events"]
        if overall["retrieval_candidate_events"]
        else 0.0
    )
    overall["retrieval_precision"] = (
        overall["retrieval_hits"] / overall["retrieval_candidates"]
        if overall["retrieval_candidates"]
        else 0.0
    )
    for bucket_name in ("by_metric", "by_benchmark", "by_model"):
        for bucket in strategy[bucket_name].values():
            bucket["success_rate"] = (
                bucket["n_success"] / bucket["n_cases"] if bucket["n_cases"] else 0.0
            )


def _transfer_bucket() -> dict[str, int]:
    return {
        "same_model_support_events": 0,
        "cross_model_support_events": 0,
        "same_model_helpful_events": 0,
        "cross_model_helpful_events": 0,
        "same_model_harmful_events": 0,
        "cross_model_harmful_events": 0,
    }


def _update_transfer_effects(
    transfer_effects: dict[str, dict[str, dict[str, int]]],
    *,
    strategy_name: str,
    case: ReplayCase,
    decision: _Decision,
    reference_use_repair: bool,
) -> None:
    if decision.use_repair == reference_use_repair or not decision.support_models:
        return
    bucket = transfer_effects[strategy_name].setdefault(case.model_name, _transfer_bucket())
    support_models = set(decision.support_models)
    same_model = case.model_name in support_models
    cross_model = any(model_name != case.model_name for model_name in support_models)
    if same_model:
        bucket["same_model_support_events"] += 1
    if cross_model:
        bucket["cross_model_support_events"] += 1
    reference_success = case.success_for(reference_use_repair)
    decision_success = case.success_for(decision.use_repair)
    if decision_success and not reference_success:
        if same_model:
            bucket["same_model_helpful_events"] += 1
        if cross_model:
            bucket["cross_model_helpful_events"] += 1
    elif reference_success and not decision_success:
        if same_model:
            bucket["same_model_harmful_events"] += 1
        if cross_model:
            bucket["cross_model_harmful_events"] += 1


def _description_for_error_type(case: ReplayCase, error_type: str) -> str:
    if case.descriptions:
        index = case.error_types.index(error_type) if error_type in case.error_types else 0
        if index < len(case.descriptions):
            return case.descriptions[index]
        return case.descriptions[0]
    return error_type


def _case_record_for_replay_case(case: ReplayCase) -> CaseRecord:
    confidence = 0.99 if case.actual_error and case.detected else 0.7 if case.detected else 0.45
    return CaseRecord.normalize(
        benchmark=case.benchmark,
        benchmark_slice=f"{case.benchmark}/{case.domain}",
        model_name=case.model_name,
        case_id=case.case_id,
        violation_types=case.error_types,
        prompt_text="",
        description_texts=case.descriptions,
        baseline_success=case.baseline_success,
        repair_success=case.repair_success,
        confidence=confidence,
        source_experiment=case.source_experiment,
        verifier_path="self_learning_replay",
    )


def _case_matches_for_replay_case(
    case: ReplayCase,
    *,
    case_memory: CaseMemory | None,
    min_support: int,
) -> list[Any]:
    if case_memory is None or not case.detected:
        return []
    query = CaseQuery.from_record(
        _case_record_for_replay_case(case),
        preferred_repair_outcome="improved",
    )
    return case_memory.retrieve(query, limit=3, min_support=min_support)


def run_replay_cases(
    cases: list[ReplayCase],
    *,
    tracker_min_support: int = TRACKER_MIN_SUPPORT,
    tracker_min_precision: float = TRACKER_MIN_PRECISION,
    memory_min_support: int = MEMORY_MIN_SUPPORT,
) -> dict[str, Any]:
    """Replay held-out cases in chronological order with live-only updates."""

    ordered_cases = sorted(
        cases,
        key=lambda case: (
            case.source_experiment,
            case.sample_position,
            case.model_name,
            case.case_id,
        ),
    )

    tracker = ConstraintTracker()
    memory = ConstraintMemory()
    case_memory = CaseMemory()
    observed_types: dict[str, _ObservedTypeStats] = {}
    observed_patterns: dict[tuple[str, str], _ObservedPatternStats] = {}

    strategies: dict[str, dict[str, Any]] = {
        "no_learning": _empty_strategy("no_learning"),
        "tracker_only": _empty_strategy("tracker_only"),
        "tracker_plus_memory": _empty_strategy("tracker_plus_memory", track_retrieval=True),
    }

    transfer_effects: dict[str, dict[str, dict[str, int]]] = {
        "tracker_only": {},
        "tracker_plus_memory": {},
    }

    held_out_decisions: list[dict[str, Any]] = []
    held_out_cases = 0
    learning_cases = 0

    for case in ordered_cases:
        if case.held_out:
            held_out_cases += 1
            no_learning = _Decision(
                use_repair=case.detected,
                reason="detected" if case.detected else "baseline_only",
            )
            tracker_only = _tracker_decision(
                case,
                tracker=tracker,
                observed_types=observed_types,
                tracker_min_support=tracker_min_support,
                tracker_min_precision=tracker_min_precision,
            )
            tracker_plus_memory = _memory_decision(
                case,
                tracker_decision=tracker_only,
                tracker=tracker,
                observed_types=observed_types,
                observed_patterns=observed_patterns,
                case_memory=case_memory,
                memory_min_support=memory_min_support,
            )

            _record_strategy_outcome(strategies["no_learning"], case, decision=no_learning)
            _record_strategy_outcome(strategies["tracker_only"], case, decision=tracker_only)
            _record_strategy_outcome(
                strategies["tracker_plus_memory"],
                case,
                decision=tracker_plus_memory,
            )

            _update_transfer_effects(
                transfer_effects,
                strategy_name="tracker_only",
                case=case,
                decision=tracker_only,
                reference_use_repair=no_learning.use_repair,
            )
            _update_transfer_effects(
                transfer_effects,
                strategy_name="tracker_plus_memory",
                case=case,
                decision=tracker_plus_memory,
                reference_use_repair=tracker_only.use_repair,
            )

            held_out_decisions.append(
                {
                    **case.to_dict(),
                    "strategies": {
                        "no_learning": {
                            "use_repair": no_learning.use_repair,
                            "reason": no_learning.reason,
                            "final_success": case.success_for(no_learning.use_repair),
                        },
                        "tracker_only": {
                            "use_repair": tracker_only.use_repair,
                            "reason": tracker_only.reason,
                            "support_models": list(tracker_only.support_models),
                            "final_success": case.success_for(tracker_only.use_repair),
                        },
                        "tracker_plus_memory": {
                            "use_repair": tracker_plus_memory.use_repair,
                            "reason": tracker_plus_memory.reason,
                            "support_models": list(tracker_plus_memory.support_models),
                            "candidate_error_types": list(
                                tracker_plus_memory.candidate_error_types
                            ),
                            "matched_error_types": list(tracker_plus_memory.matched_error_types),
                            "candidate_case_keys": list(tracker_plus_memory.candidate_case_keys),
                            "matched_case_keys": list(tracker_plus_memory.matched_case_keys),
                            "final_success": case.success_for(tracker_plus_memory.use_repair),
                        },
                    },
                }
            )
            continue

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
                stats.repair_improvements += int(case.repair_success and not case.baseline_success)
                stats.repair_harms += int(case.baseline_success and not case.repair_success)
                stats.source_models.add(case.model_name)
        if case.detected and case.actual_error and case.error_types:
            for error_type in case.error_types:
                description = _description_for_error_type(case, error_type)
                memory.record_pattern(case.domain, error_type, description)
                pattern_stats = observed_patterns.setdefault(
                    (case.domain, error_type),
                    _ObservedPatternStats(),
                )
                pattern_stats.support += 1
                pattern_stats.repair_improvements += int(
                    case.repair_success and not case.baseline_success
                )
                pattern_stats.repair_harms += int(case.baseline_success and not case.repair_success)
                pattern_stats.source_models.add(case.model_name)
            case_memory.record(_case_record_for_replay_case(case))

    for strategy in strategies.values():
        _normalise_strategy(strategy)

    no_learning_false_positives = strategies["no_learning"]["overall"]["false_positives"]
    false_positive_regression_budget: dict[str, Any] = {
        "policy": "zero_additional_false_positives_vs_no_learning",
    }
    for strategy_name in ("tracker_only", "tracker_plus_memory"):
        strategy_false_positives = strategies[strategy_name]["overall"]["false_positives"]
        additional = strategy_false_positives - no_learning_false_positives
        false_positive_regression_budget[strategy_name] = {
            "baseline_false_positives": no_learning_false_positives,
            "strategy_false_positives": strategy_false_positives,
            "additional_false_positives": additional,
            "within_budget": additional <= 0,
        }

    return {
        "summary": {
            "held_out_cases": held_out_cases,
            "learning_cases": learning_cases,
            "false_positive_regression_budget": false_positive_regression_budget,
        },
        "strategies": strategies,
        "transfer_effects": transfer_effects,
        "held_out_decisions": held_out_decisions,
    }


def build_self_learning_replay_payload(
    *,
    exp219: dict[str, Any],
    exp220: dict[str, Any],
    exp221: dict[str, Any],
    holdout_fraction: float = HOLDOUT_FRACTION,
    tracker_min_support: int = TRACKER_MIN_SUPPORT,
    tracker_min_precision: float = TRACKER_MIN_PRECISION,
    memory_min_support: int = MEMORY_MIN_SUPPORT,
) -> dict[str, Any]:
    """Build the in-memory Exp 223 replay artifact without writing files."""

    cases = build_replay_cases(
        exp219=exp219,
        exp220=exp220,
        exp221=exp221,
        holdout_fraction=holdout_fraction,
    )
    replay = run_replay_cases(
        cases,
        tracker_min_support=tracker_min_support,
        tracker_min_precision=tracker_min_precision,
        memory_min_support=memory_min_support,
    )
    positions_219, case_count_219 = _sample_positions(exp219)
    positions_220, case_count_220 = _sample_positions(exp220)
    positions_221, case_count_221 = _sample_positions(exp221)
    del positions_219, positions_220, positions_221
    holdout_start_positions = {
        "219": math.floor(case_count_219 * (1.0 - holdout_fraction)) + 1,
        "220": math.floor(case_count_220 * (1.0 - holdout_fraction)) + 1,
        "221": math.floor(case_count_221 * (1.0 - holdout_fraction)) + 1,
    }
    payload = {
        "experiment": 223,
        "run_date": RUN_DATE,
        "title": "Held-out live self-learning replay benchmark",
        "metadata": {
            "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
            "memory_source": str(MEMORY_SOURCE),
            "output_path": str(RESULT_OUTPUT),
            "held_out_policy": {
                "name": "final_quarter_per_experiment",
                "fraction": holdout_fraction,
                "case_counts": {
                    "219": case_count_219,
                    "220": case_count_220,
                    "221": case_count_221,
                },
                "start_positions": holdout_start_positions,
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
        },
        "summary": replay["summary"],
        "strategies": replay["strategies"],
        "transfer_effects": replay["transfer_effects"],
        "held_out_decisions": replay["held_out_decisions"],
    }
    return payload


def _build_exp238_cases(
    payload: dict[str, Any],
    *,
    holdout_fraction: float,
) -> list[ReplayCase]:
    positions, case_count = _sample_positions(payload)
    holdout_start = math.floor(case_count * (1.0 - holdout_fraction))
    model_runs = payload.get("model_runs")
    if not isinstance(model_runs, dict):
        return []

    cases: list[ReplayCase] = []
    for raw_run in model_runs.values():
        run = _as_dict(raw_run)
        model_name = str(run.get("model_name") or "")
        per_problem_results = run.get("per_problem_results")
        if not isinstance(per_problem_results, list):
            continue
        for item in per_problem_results:
            case = _as_dict(item)
            case_id = str(case.get("case_id") or "")
            if not case_id:
                continue
            error_types, descriptions = _extract_exp238_errors(case)
            baseline = _as_dict(case.get("baseline"))
            spec_aware = _as_dict(case.get("spec_aware_verify_only"))
            verify_repair = _as_dict(case.get("verify_repair"))
            history = case.get("history")
            baseline_latency_seconds = 0.0
            repair_latency_seconds = 0.0
            if isinstance(history, list):
                for index, entry in enumerate(history):
                    if not isinstance(entry, dict):
                        continue
                    latency = float(entry.get("latency_seconds") or 0.0)
                    if index == 0:
                        baseline_latency_seconds = latency
                    repair_latency_seconds += latency
            if repair_latency_seconds == 0.0:
                repair_latency_seconds = baseline_latency_seconds
            sample_position = positions.get(case_id, 0)
            baseline_success = bool(baseline.get("official_passed"))
            cases.append(
                ReplayCase(
                    source_experiment=238,
                    benchmark="humaneval_dual_model_spec",
                    metric_name="pass_rate",
                    domain="code_spec_properties",
                    model_name=model_name,
                    case_id=case_id,
                    sample_position=sample_position,
                    held_out=sample_position > holdout_start,
                    actual_error=not baseline_success,
                    detected=not bool(spec_aware.get("accepted", True)),
                    error_types=error_types,
                    descriptions=descriptions,
                    baseline_success=baseline_success,
                    repair_success=bool(verify_repair.get("accepted")),
                    baseline_latency_seconds=baseline_latency_seconds,
                    repair_latency_seconds=repair_latency_seconds,
                )
            )
    return cases


def build_exp241_replay_cases(
    *,
    exp235: dict[str, Any],
    exp238: dict[str, Any],
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> list[ReplayCase]:
    """Build Exp 241 replay cases from Exp 235 semantic and Exp 238 code artifacts."""

    cases = [
        *_build_cases_for_payload(
            exp235,
            _extract_exp235_events(exp235),
            holdout_fraction=holdout_fraction,
        ),
        *_build_exp238_cases(exp238, holdout_fraction=holdout_fraction),
    ]
    return sorted(
        cases,
        key=lambda case: (
            case.source_experiment,
            case.sample_position,
            case.model_name,
            case.case_id,
        ),
    )


def _policy_context_for_replay_case(
    case: ReplayCase,
    *,
    policy_compiler: SelfLearningPolicyCompiler,
    tracker: ConstraintTracker,
    case_memory: CaseMemory,
) -> dict[str, Any]:
    record = _case_record_for_replay_case(case)
    query = PolicyQuery.from_record(record, preferred_repair_outcome="improved")
    policy = policy_compiler.compile(case_memory=case_memory, accepted_repairs=(), tracker=tracker)
    context = policy.runtime_context(query, tracker=tracker, case_memory=case_memory)
    return {
        "case_matches": tuple(match.entry.key.fingerprint for match in context.case_matches),
        "threshold_overrides": tuple(item.update_id for item in context.threshold_overrides),
        "property_budget_updates": tuple(
            item.update_id for item in context.property_budget_updates
        ),
        "repair_prompt_patches": tuple(item.update_id for item in context.repair_prompt_patches),
        "routing_hints": tuple(item.update_id for item in context.routing_hints),
        "routing_targets": tuple(item.route_to for item in context.routing_hints),
    }


def _policy_decision(
    case: ReplayCase,
    *,
    base_decision: _Decision,
    policy_context: dict[str, Any],
) -> _Decision:
    support_models = set(base_decision.support_models)
    if base_decision.use_repair:
        reason = base_decision.reason
        use_repair = True
    elif "case_memory_then_repair" in policy_context["routing_targets"]:
        reason = "policy_routing_hint"
        use_repair = True
    elif case.detected and policy_context["threshold_overrides"]:
        reason = "policy_threshold_override"
        use_repair = True
    elif case.detected and policy_context["repair_prompt_patches"]:
        reason = "policy_repair_patch"
        use_repair = True
    elif case.detected and policy_context["property_budget_updates"] and "code" in case.domain:
        reason = "policy_property_budget"
        use_repair = True
    else:
        reason = base_decision.reason
        use_repair = False

    return _Decision(
        use_repair=use_repair,
        reason=reason,
        support_models=tuple(sorted(support_models)),
        candidate_error_types=base_decision.candidate_error_types,
        matched_error_types=base_decision.matched_error_types,
        candidate_case_keys=base_decision.candidate_case_keys
        or tuple(policy_context["case_matches"]),
        matched_case_keys=base_decision.matched_case_keys
        or tuple(policy_context["case_matches"] if use_repair else ()),
        policy_threshold_ids=tuple(policy_context["threshold_overrides"]),
        policy_property_budget_ids=tuple(policy_context["property_budget_updates"]),
        policy_patch_ids=tuple(policy_context["repair_prompt_patches"]),
        policy_routing_ids=tuple(policy_context["routing_hints"]),
    )


def _policy_context_dict(decision: _Decision) -> dict[str, list[str]]:
    return {
        "threshold_overrides": list(decision.policy_threshold_ids),
        "property_budget_updates": list(decision.policy_property_budget_ids),
        "repair_prompt_patches": list(decision.policy_patch_ids),
        "routing_hints": list(decision.policy_routing_ids),
    }


def _primary_success_condition(strategies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tracker_success = float(strategies["tracker_only"]["overall"]["success_rate"])
    policy_success = float(strategies["case_memory_plus_policy"]["overall"]["success_rate"])
    no_learning_false_positives = int(strategies["no_learning"]["overall"]["false_positives"])
    policy_false_positives = int(
        strategies["case_memory_plus_policy"]["overall"]["false_positives"]
    )
    success_gain = policy_success - tracker_success
    additional_false_positives = policy_false_positives - no_learning_false_positives
    met = success_gain > 0.0 and additional_false_positives <= 0
    return {
        "metric": "real_held_out_task_gain_with_no_extra_false_positives",
        "reference_strategy": "tracker_only",
        "candidate_strategy": "case_memory_plus_policy",
        "success_rate_gain_vs_tracker_only": success_gain,
        "additional_false_positives_vs_no_learning": additional_false_positives,
        "met": met,
        "statement": "met" if met else "not_met",
    }


def run_replay_cases_v2(
    cases: list[ReplayCase],
    *,
    tracker_min_support: int = TRACKER_MIN_SUPPORT,
    tracker_min_precision: float = TRACKER_MIN_PRECISION,
    memory_min_support: int = MEMORY_MIN_SUPPORT,
    policy_min_case_support: int = POLICY_MIN_CASE_SUPPORT,
    policy_min_case_confidence: float = POLICY_MIN_CASE_CONFIDENCE,
    policy_min_patch_support: int = POLICY_MIN_PATCH_SUPPORT,
) -> dict[str, Any]:
    """Replay held-out cases with tracker, case memory, and compiled-policy branches."""

    ordered_cases = sorted(
        cases,
        key=lambda case: (
            case.source_experiment,
            case.sample_position,
            case.model_name,
            case.case_id,
        ),
    )

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

    strategies: dict[str, dict[str, Any]] = {
        "no_learning": _empty_strategy("no_learning"),
        "tracker_only": _empty_strategy("tracker_only"),
        "case_memory": _empty_strategy("case_memory", track_retrieval=True),
        "case_memory_plus_policy": _empty_strategy(
            "case_memory_plus_policy",
            track_retrieval=True,
        ),
    }
    transfer_effects: dict[str, dict[str, dict[str, int]]] = {
        "tracker_only": {},
        "case_memory": {},
        "case_memory_plus_policy": {},
    }

    held_out_decisions: list[dict[str, Any]] = []
    held_out_cases = 0
    learning_cases = 0

    for case in ordered_cases:
        if case.held_out:
            held_out_cases += 1
            no_learning = _Decision(
                use_repair=case.detected,
                reason="detected" if case.detected else "baseline_only",
            )
            tracker_only = _tracker_decision(
                case,
                tracker=tracker,
                observed_types=observed_types,
                tracker_min_support=tracker_min_support,
                tracker_min_precision=tracker_min_precision,
            )
            case_memory_decision = _memory_decision(
                case,
                tracker_decision=tracker_only,
                tracker=tracker,
                observed_types=observed_types,
                observed_patterns=observed_patterns,
                case_memory=case_memory,
                memory_min_support=memory_min_support,
            )
            policy_context = _policy_context_for_replay_case(
                case,
                policy_compiler=policy_compiler,
                tracker=tracker,
                case_memory=case_memory,
            )
            case_memory_plus_policy = _policy_decision(
                case,
                base_decision=case_memory_decision,
                policy_context=policy_context,
            )

            _record_strategy_outcome(strategies["no_learning"], case, decision=no_learning)
            _record_strategy_outcome(strategies["tracker_only"], case, decision=tracker_only)
            _record_strategy_outcome(strategies["case_memory"], case, decision=case_memory_decision)
            _record_strategy_outcome(
                strategies["case_memory_plus_policy"],
                case,
                decision=case_memory_plus_policy,
            )

            _update_transfer_effects(
                transfer_effects,
                strategy_name="tracker_only",
                case=case,
                decision=tracker_only,
                reference_use_repair=no_learning.use_repair,
            )
            _update_transfer_effects(
                transfer_effects,
                strategy_name="case_memory",
                case=case,
                decision=case_memory_decision,
                reference_use_repair=tracker_only.use_repair,
            )
            _update_transfer_effects(
                transfer_effects,
                strategy_name="case_memory_plus_policy",
                case=case,
                decision=case_memory_plus_policy,
                reference_use_repair=case_memory_decision.use_repair,
            )

            held_out_decisions.append(
                {
                    **case.to_dict(),
                    "strategies": {
                        "no_learning": {
                            "use_repair": no_learning.use_repair,
                            "reason": no_learning.reason,
                            "final_success": case.success_for(no_learning.use_repair),
                        },
                        "tracker_only": {
                            "use_repair": tracker_only.use_repair,
                            "reason": tracker_only.reason,
                            "support_models": list(tracker_only.support_models),
                            "final_success": case.success_for(tracker_only.use_repair),
                        },
                        "case_memory": {
                            "use_repair": case_memory_decision.use_repair,
                            "reason": case_memory_decision.reason,
                            "support_models": list(case_memory_decision.support_models),
                            "candidate_error_types": list(
                                case_memory_decision.candidate_error_types
                            ),
                            "matched_error_types": list(case_memory_decision.matched_error_types),
                            "candidate_case_keys": list(case_memory_decision.candidate_case_keys),
                            "matched_case_keys": list(case_memory_decision.matched_case_keys),
                            "final_success": case.success_for(case_memory_decision.use_repair),
                        },
                        "case_memory_plus_policy": {
                            "use_repair": case_memory_plus_policy.use_repair,
                            "reason": case_memory_plus_policy.reason,
                            "support_models": list(case_memory_plus_policy.support_models),
                            "candidate_error_types": list(
                                case_memory_plus_policy.candidate_error_types
                            ),
                            "matched_error_types": list(
                                case_memory_plus_policy.matched_error_types
                            ),
                            "candidate_case_keys": list(
                                case_memory_plus_policy.candidate_case_keys
                            ),
                            "matched_case_keys": list(case_memory_plus_policy.matched_case_keys),
                            "policy_context": _policy_context_dict(case_memory_plus_policy),
                            "final_success": case.success_for(case_memory_plus_policy.use_repair),
                        },
                    },
                }
            )
            continue

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
                stats.repair_improvements += int(case.repair_success and not case.baseline_success)
                stats.repair_harms += int(case.baseline_success and not case.repair_success)
                stats.source_models.add(case.model_name)
        if case.detected and case.actual_error and case.error_types:
            for error_type in case.error_types:
                description = _description_for_error_type(case, error_type)
                memory.record_pattern(case.domain, error_type, description)
                pattern_stats = observed_patterns.setdefault(
                    (case.domain, error_type),
                    _ObservedPatternStats(),
                )
                pattern_stats.support += 1
                pattern_stats.repair_improvements += int(
                    case.repair_success and not case.baseline_success
                )
                pattern_stats.repair_harms += int(case.baseline_success and not case.repair_success)
                pattern_stats.source_models.add(case.model_name)
            case_memory.record(_case_record_for_replay_case(case))

    for strategy in strategies.values():
        _normalise_strategy(strategy)

    no_learning_false_positives = strategies["no_learning"]["overall"]["false_positives"]
    false_positive_regression_budget: dict[str, Any] = {
        "policy": "zero_additional_false_positives_vs_no_learning",
    }
    for strategy_name in ("tracker_only", "case_memory", "case_memory_plus_policy"):
        strategy_false_positives = strategies[strategy_name]["overall"]["false_positives"]
        additional = strategy_false_positives - no_learning_false_positives
        false_positive_regression_budget[strategy_name] = {
            "baseline_false_positives": no_learning_false_positives,
            "strategy_false_positives": strategy_false_positives,
            "additional_false_positives": additional,
            "within_budget": additional <= 0,
        }

    summary = {
        "held_out_cases": held_out_cases,
        "learning_cases": learning_cases,
        "false_positive_regression_budget": false_positive_regression_budget,
    }
    summary["primary_success_condition"] = _primary_success_condition(strategies)
    return {
        "summary": summary,
        "strategies": strategies,
        "transfer_effects": transfer_effects,
        "held_out_decisions": held_out_decisions,
    }


def _build_exp223_comparison(
    strategies: dict[str, dict[str, Any]],
    exp223_reference: dict[str, Any],
) -> dict[str, Any]:
    reference_strategies = _as_dict(exp223_reference.get("strategies"))
    mapping = {
        "no_learning": "no_learning",
        "tracker_only": "tracker_only",
        "case_memory": "tracker_plus_memory",
        "case_memory_plus_policy": "tracker_plus_memory",
    }
    strategy_deltas: dict[str, Any] = {}
    for strategy_name, reference_name in mapping.items():
        reference_overall = _as_dict(
            _as_dict(reference_strategies.get(reference_name)).get("overall")
        )
        current_overall = _as_dict(strategies.get(strategy_name, {}).get("overall"))
        strategy_deltas[strategy_name] = {
            "reference_strategy": reference_name,
            "success_rate_delta": float(current_overall.get("success_rate") or 0.0)
            - float(reference_overall.get("success_rate") or 0.0),
            "false_positive_delta": int(current_overall.get("false_positives") or 0)
            - int(reference_overall.get("false_positives") or 0),
            "retrieval_hit_rate_delta": float(current_overall.get("retrieval_hit_rate") or 0.0)
            - float(reference_overall.get("retrieval_hit_rate") or 0.0),
            "retrieval_precision_delta": float(current_overall.get("retrieval_precision") or 0.0)
            - float(reference_overall.get("retrieval_precision") or 0.0),
        }
    return {
        "reference_experiment": int(exp223_reference.get("experiment") or 223),
        "reference_run_date": str(exp223_reference.get("run_date") or ""),
        "strategy_deltas": strategy_deltas,
    }


def build_self_learning_replay_v2_payload(
    *,
    exp235: dict[str, Any],
    exp238: dict[str, Any],
    exp223_reference: dict[str, Any],
    holdout_fraction: float = HOLDOUT_FRACTION,
    tracker_min_support: int = TRACKER_MIN_SUPPORT,
    tracker_min_precision: float = TRACKER_MIN_PRECISION,
    memory_min_support: int = MEMORY_MIN_SUPPORT,
    policy_min_case_support: int = POLICY_MIN_CASE_SUPPORT,
    policy_min_case_confidence: float = POLICY_MIN_CASE_CONFIDENCE,
    policy_min_patch_support: int = POLICY_MIN_PATCH_SUPPORT,
) -> dict[str, Any]:
    """Build the in-memory Exp 241 replay artifact without writing files."""

    cases = build_exp241_replay_cases(
        exp235=exp235,
        exp238=exp238,
        holdout_fraction=holdout_fraction,
    )
    replay = run_replay_cases_v2(
        cases,
        tracker_min_support=tracker_min_support,
        tracker_min_precision=tracker_min_precision,
        memory_min_support=memory_min_support,
        policy_min_case_support=policy_min_case_support,
        policy_min_case_confidence=policy_min_case_confidence,
        policy_min_patch_support=policy_min_patch_support,
    )
    _, case_count_235 = _sample_positions(exp235)
    _, case_count_238 = _sample_positions(exp238)
    holdout_start_positions = {
        "235": math.floor(case_count_235 * (1.0 - holdout_fraction)) + 1,
        "238": math.floor(case_count_238 * (1.0 - holdout_fraction)) + 1,
    }
    return {
        "experiment": 241,
        "run_date": RUN_DATE_V2,
        "title": "Held-out live self-learning replay benchmark v2",
        "metadata": {
            "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS_V2],
            "comparison_reference": str(EXP223_REFERENCE),
            "output_path": str(RESULT_OUTPUT_V2),
            "held_out_policy": {
                "name": "final_slice_per_source_artifact",
                "fraction": holdout_fraction,
                "case_counts": {"235": case_count_235, "238": case_count_238},
                "start_positions": holdout_start_positions,
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
        },
        "summary": replay["summary"],
        "strategies": replay["strategies"],
        "transfer_effects": replay["transfer_effects"],
        "held_out_decisions": replay["held_out_decisions"],
        "comparison_to_experiment_223": _build_exp223_comparison(
            replay["strategies"],
            exp223_reference,
        ),
    }


def run_experiment(
    repo_root: Path | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    """Build and write Exp 223 for the current repository checkout."""

    resolved_repo = (repo_root or get_repo_root()).resolve()
    resolved_result_path = resolved_repo / (result_path or RESULT_OUTPUT)
    exp219 = load_json(resolved_repo / SOURCE_ARTIFACTS[0])
    exp220 = load_json(resolved_repo / SOURCE_ARTIFACTS[1])
    exp221 = load_json(resolved_repo / SOURCE_ARTIFACTS[2])
    payload = build_self_learning_replay_payload(
        exp219=exp219,
        exp220=exp220,
        exp221=exp221,
    )
    payload["metadata"]["output_path"] = _relative_path(resolved_result_path, resolved_repo)
    write_json(resolved_result_path, payload)
    return payload


def run_experiment_v2(
    repo_root: Path | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    """Build and write Exp 241 for the current repository checkout."""

    resolved_repo = (repo_root or get_repo_root()).resolve()
    resolved_result_path = resolved_repo / (result_path or RESULT_OUTPUT_V2)
    exp235 = load_json(resolved_repo / SOURCE_ARTIFACTS_V2[0])
    exp238 = load_json(resolved_repo / SOURCE_ARTIFACTS_V2[1])
    exp223_reference = load_json(resolved_repo / EXP223_REFERENCE)
    payload = build_self_learning_replay_v2_payload(
        exp235=exp235,
        exp238=exp238,
        exp223_reference=exp223_reference,
    )
    payload["metadata"]["output_path"] = _relative_path(resolved_result_path, resolved_repo)
    write_json(resolved_result_path, payload)
    return payload


__all__ = [
    "EXP223_REFERENCE",
    "HOLDOUT_FRACTION",
    "MEMORY_MIN_SUPPORT",
    "MEMORY_SOURCE",
    "POLICY_MIN_CASE_CONFIDENCE",
    "POLICY_MIN_CASE_SUPPORT",
    "POLICY_MIN_PATCH_SUPPORT",
    "RESULT_OUTPUT",
    "RESULT_OUTPUT_V2",
    "RUN_DATE",
    "RUN_DATE_V2",
    "ReplayCase",
    "SOURCE_ARTIFACTS",
    "SOURCE_ARTIFACTS_V2",
    "TRACKER_MIN_PRECISION",
    "TRACKER_MIN_SUPPORT",
    "build_exp241_replay_cases",
    "build_replay_cases",
    "build_self_learning_replay_payload",
    "build_self_learning_replay_v2_payload",
    "get_repo_root",
    "run_experiment",
    "run_experiment_v2",
    "run_replay_cases",
    "run_replay_cases_v2",
]

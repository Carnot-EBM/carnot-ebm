"""Self-learning from checked-in code-verification traces.

This module turns Exp 225 / Exp 226 style benchmark artifacts into lightweight
learning signals for future code verification. It focuses on three questions:
which PBT properties catch the most bugs, which repair families succeed, and
which problem types benefit most from additive verification.

Spec: REQ-CODE-016, REQ-CODE-017, REQ-CODE-018,
      SCENARIO-CODE-014, SCENARIO-CODE-015
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

_PROPERTY_PRIORITY = {
    "no_exception": 0,
    "deterministic": 1,
    "annotated_return_type": 2,
    "input_immutability": 3,
    "sorted_output": 4,
    "reverse_output": 5,
}

_STRATEGY_PRIORITY = {
    "syntax_recovery": 0,
    "exception_hardening": 1,
    "return_type_alignment": 2,
    "immutability_preserving_fix": 3,
    "ordering_fix": 4,
    "semantic_harness_fix": 5,
    "generic_repair": 6,
}


@dataclass(frozen=True)
class PropertyFailureTrace:
    """One property failure recorded in a verification trace."""

    property_name: str
    source: str
    error: str | None


@dataclass(frozen=True)
class VerificationTraceStep:
    """One baseline or repair iteration inside a code-verification trace."""

    iteration: int
    detected: bool
    accepted: bool
    harness_passed: bool
    harness_error_message: str
    property_failures: tuple[PropertyFailureTrace, ...]

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(failure.property_name for failure in self.property_failures)


@dataclass(frozen=True)
class CodeVerificationTrace:
    """Normalized per-problem verification trace."""

    artifact_name: str
    experiment_id: str
    case_id: str
    task_id: str
    prompt: str
    entry_point: str
    derived_properties: tuple[str, ...]
    baseline_failures: tuple[PropertyFailureTrace, ...]
    baseline_passed: bool
    baseline_accepted: bool
    official_test_miss: bool
    repaired: bool
    steps: tuple[VerificationTraceStep, ...]

    def problem_types(self) -> tuple[str, ...]:
        names = set(self.derived_properties)
        kinds: list[str] = []
        if {"no_exception", "deterministic", "annotated_return_type"} & names:
            kinds.append("signature_robustness")
        if "input_immutability" in names:
            kinds.append("mutation_safety")
        if {"sorted_output", "reverse_output"} & names:
            kinds.append("sequence_intent")
        if not kinds:
            kinds.append("general")
        return tuple(kinds)


@dataclass(frozen=True)
class PropertyScore:
    """Aggregated effectiveness score for one PBT property."""

    property_name: str
    failure_count: int
    affected_cases: int
    official_test_misses: int
    repaired_cases: int
    score: float


@dataclass(frozen=True)
class ProblemTypeScore:
    """Aggregated verifier benefit for one inferred problem type."""

    problem_type: str
    case_count: int
    official_test_misses: int
    repaired_cases: int
    accepted_baselines: int
    score: float


@dataclass(frozen=True)
class StrategyRecommendation:
    """Learned repair recommendation for one error family."""

    strategy_name: str
    error_family: str
    attempts: int
    successes: int
    partial_recoveries: int
    success_rate: float
    support_case_ids: tuple[str, ...]
    score: float


@dataclass(frozen=True)
class LearningCurvePoint:
    """One cumulative-learning checkpoint."""

    prefix_size: int
    top_property: str | None
    top_property_score: float
    top_strategy: str | None
    top_strategy_success_rate: float


@dataclass(frozen=True)
class LearningImprovement:
    """Simple cumulative-learning demonstration payload."""

    points: tuple[LearningCurvePoint, ...]
    improved: bool


@dataclass(frozen=True)
class TraceAnalysis:
    """Top-level trace-analysis summary."""

    trace_artifact_count: int
    skipped_artifact_count: int
    skipped_artifacts: tuple[str, ...]
    case_count: int
    property_rankings: tuple[PropertyScore, ...]
    repair_rankings: tuple[StrategyRecommendation, ...]
    problem_type_rankings: tuple[ProblemTypeScore, ...]


def _unique_strings(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


def _parse_failure_records(records: Any) -> tuple[PropertyFailureTrace, ...]:
    parsed: list[PropertyFailureTrace] = []
    if not isinstance(records, list):
        return ()
    for record in records:
        if not isinstance(record, dict):
            continue
        name = str(record.get("property_name") or "")
        if not name:
            continue
        error = record.get("error")
        parsed.append(
            PropertyFailureTrace(
                property_name=name,
                source=str(record.get("source") or ""),
                error=str(error) if error is not None else None,
            )
        )
    return tuple(parsed)


def _step_from_history_item(item: dict[str, Any]) -> VerificationTraceStep:
    harness = item.get("harness")
    harness_dict = harness if isinstance(harness, dict) else {}
    pbt = item.get("pbt")
    pbt_dict = pbt if isinstance(pbt, dict) else {}
    return VerificationTraceStep(
        iteration=int(item.get("iteration") or 0),
        detected=bool(item.get("detected")),
        accepted=bool(item.get("accepted")),
        harness_passed=bool(harness_dict.get("passed")),
        harness_error_message=str(harness_dict.get("error_message") or ""),
        property_failures=_parse_failure_records(pbt_dict.get("failure_records", [])),
    )


def _baseline_step(row: dict[str, Any]) -> VerificationTraceStep:
    baseline = row.get("baseline")
    baseline_dict = baseline if isinstance(baseline, dict) else {}
    return VerificationTraceStep(
        iteration=0,
        detected=bool(baseline_dict.get("detected")),
        accepted=bool(baseline_dict.get("accepted")),
        harness_passed=bool(baseline_dict.get("passed")),
        harness_error_message=str(baseline_dict.get("error_message") or ""),
        property_failures=_parse_failure_records(baseline_dict.get("pbt_failure_records", [])),
    )


def _display_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _cohort_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cohort = payload.get("cohort")
    cohort_dict = cohort if isinstance(cohort, dict) else {}
    lookup: dict[str, dict[str, Any]] = {}
    cases = cohort_dict.get("cases")
    if not isinstance(cases, list):
        return lookup
    for case in cases:
        if not isinstance(case, dict) or not case.get("case_id"):
            continue
        lookup[str(case["case_id"])] = case
    return lookup


def _extract_experiment_id(payload: dict[str, Any], artifact_name: str) -> str:
    experiment_id = payload.get("experiment_id")
    if experiment_id is not None:
        return str(experiment_id)
    experiment = payload.get("experiment")
    if experiment is not None:
        return str(experiment)
    return artifact_name


def _extract_traces(
    payload: dict[str, Any],
    artifact_name: str,
) -> tuple[CodeVerificationTrace, ...]:
    rows = payload.get("per_problem_results")
    if not isinstance(rows, list) or not rows:
        return ()

    lookup = _cohort_lookup(payload)
    experiment_id = _extract_experiment_id(payload, artifact_name)
    traces: list[CodeVerificationTrace] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("case_id") or row.get("task_id") or "")
        if not case_id:
            continue
        cohort_case = lookup.get(case_id, {})
        baseline = row.get("baseline")
        baseline_dict = baseline if isinstance(baseline, dict) else {}
        derived = baseline_dict.get("pbt_derived_properties", [])
        derived_properties = _unique_strings(
            str(item.get("name") or "") for item in derived if isinstance(item, dict)
        )
        history = row.get("history")
        if isinstance(history, list) and history:
            steps = tuple(
                _step_from_history_item(item) for item in history if isinstance(item, dict)
            )
        else:
            steps = (_baseline_step(row),)
        traces.append(
            CodeVerificationTrace(
                artifact_name=artifact_name,
                experiment_id=experiment_id,
                case_id=case_id,
                task_id=str(row.get("task_id") or case_id),
                prompt=str(cohort_case.get("prompt") or ""),
                entry_point=str(row.get("entry_point") or cohort_case.get("entry_point") or ""),
                derived_properties=derived_properties,
                baseline_failures=_parse_failure_records(
                    baseline_dict.get("pbt_failure_records", [])
                ),
                baseline_passed=bool(baseline_dict.get("passed")),
                baseline_accepted=bool(baseline_dict.get("accepted")),
                official_test_miss=bool(baseline_dict.get("official_test_miss_caught_by_pbt")),
                repaired=bool(
                    (
                        row["verify_repair"] if isinstance(row.get("verify_repair"), dict) else {}
                    ).get("repaired")
                ),
                steps=steps,
            )
        )
    return tuple(traces)


class PropertyRanker:
    """Rank PBT properties by observed verification value."""

    def __init__(self) -> None:
        self._rankings: list[PropertyScore] = []

    def fit(self, traces: Sequence[CodeVerificationTrace]) -> PropertyRanker:
        totals: dict[str, dict[str, Any]] = {}
        for trace in traces:
            seen_in_case: set[str] = set()
            for failure in trace.baseline_failures:
                entry = totals.setdefault(
                    failure.property_name,
                    {
                        "failure_count": 0,
                        "affected_cases": 0,
                        "official_test_misses": 0,
                        "repaired_cases": 0,
                    },
                )
                entry["failure_count"] += 1
                if failure.property_name not in seen_in_case:
                    entry["affected_cases"] += 1
                    if trace.official_test_miss:
                        entry["official_test_misses"] += 1
                    if trace.repaired:
                        entry["repaired_cases"] += 1
                    seen_in_case.add(failure.property_name)

        rankings: list[PropertyScore] = []
        for property_name, entry in totals.items():
            score = float(
                entry["failure_count"]
                + (5 * entry["official_test_misses"])
                + (2 * entry["repaired_cases"])
            )
            rankings.append(
                PropertyScore(
                    property_name=property_name,
                    failure_count=int(entry["failure_count"]),
                    affected_cases=int(entry["affected_cases"]),
                    official_test_misses=int(entry["official_test_misses"]),
                    repaired_cases=int(entry["repaired_cases"]),
                    score=score,
                )
            )

        self._rankings = sorted(
            rankings,
            key=lambda item: (
                -item.score,
                -item.official_test_misses,
                -item.failure_count,
                _PROPERTY_PRIORITY.get(item.property_name, 999),
                item.property_name,
            ),
        )
        return self

    def rank(self, limit: int | None = None) -> list[PropertyScore]:
        if limit is None:
            return list(self._rankings)
        return list(self._rankings[:limit])


def _error_family(
    *,
    harness_error_message: str,
    property_names: set[str],
    official_test_miss: bool,
) -> str:
    if "IndentationError" in harness_error_message or "SyntaxError" in harness_error_message:
        return "syntax"
    if "AssertionError" in harness_error_message:
        return "harness_assertion"
    if harness_error_message:
        return "runtime_exception"
    if official_test_miss:
        return "official_test_gap"
    if property_names:
        return "pbt_contract"
    return "clean"


def _strategy_labels(
    *,
    harness_error_message: str,
    failures: Sequence[PropertyFailureTrace],
    official_test_miss: bool,
) -> tuple[str, ...]:
    property_names = {failure.property_name for failure in failures}
    labels: list[str] = []
    family = _error_family(
        harness_error_message=harness_error_message,
        property_names=property_names,
        official_test_miss=official_test_miss,
    )
    if family == "syntax":
        labels.append("syntax_recovery")
    if any(
        failure.property_name in {"no_exception", "deterministic"} and failure.error
        for failure in failures
    ):
        labels.append("exception_hardening")
    if "annotated_return_type" in property_names:
        labels.append("return_type_alignment")
    if "input_immutability" in property_names:
        labels.append("immutability_preserving_fix")
    if {"sorted_output", "reverse_output"} & property_names:
        labels.append("ordering_fix")
    if family == "harness_assertion":
        labels.append("semantic_harness_fix")
    if not labels:
        labels.append("generic_repair")
    return _unique_strings(labels)


def _count_matching(
    step: VerificationTraceStep,
    property_names: set[str],
) -> int:
    return sum(1 for failure in step.property_failures if failure.property_name in property_names)


def _partial_recovery(
    strategy_name: str,
    current: VerificationTraceStep,
    nxt: VerificationTraceStep,
    official_test_miss: bool,
) -> bool:
    current_names = set(current.property_names)
    next_names = set(nxt.property_names)
    current_family = _error_family(
        harness_error_message=current.harness_error_message,
        property_names=current_names,
        official_test_miss=official_test_miss,
    )
    next_family = _error_family(
        harness_error_message=nxt.harness_error_message,
        property_names=next_names,
        official_test_miss=False,
    )

    if strategy_name == "syntax_recovery":
        return current_family == "syntax" and next_family != "syntax"
    if strategy_name == "exception_hardening":
        return _count_matching(current, {"no_exception", "deterministic"}) > _count_matching(
            nxt, {"no_exception", "deterministic"}
        )
    if strategy_name == "return_type_alignment":
        return _count_matching(current, {"annotated_return_type"}) > _count_matching(
            nxt, {"annotated_return_type"}
        )
    if strategy_name == "immutability_preserving_fix":
        return _count_matching(current, {"input_immutability"}) > _count_matching(
            nxt, {"input_immutability"}
        )
    if strategy_name == "ordering_fix":
        return _count_matching(current, {"sorted_output", "reverse_output"}) > _count_matching(
            nxt, {"sorted_output", "reverse_output"}
        )
    if strategy_name == "semantic_harness_fix":
        return bool(current.harness_error_message) and not nxt.harness_error_message
    return current.detected and not nxt.detected


class RepairStrategy:
    """Learn repair strategy outcomes from trace transitions."""

    def __init__(self) -> None:
        self._rankings: list[StrategyRecommendation] = []

    def fit(self, traces: Sequence[CodeVerificationTrace]) -> RepairStrategy:
        totals: dict[tuple[str, str], dict[str, Any]] = {}

        for trace in traces:
            if len(trace.steps) < 2:
                continue
            for current, nxt in zip(trace.steps, trace.steps[1:], strict=False):
                property_names = set(current.property_names)
                family = _error_family(
                    harness_error_message=current.harness_error_message,
                    property_names=property_names,
                    official_test_miss=trace.official_test_miss,
                )
                for strategy_name in _strategy_labels(
                    harness_error_message=current.harness_error_message,
                    failures=current.property_failures,
                    official_test_miss=trace.official_test_miss,
                ):
                    entry = totals.setdefault(
                        (strategy_name, family),
                        {
                            "attempts": 0,
                            "successes": 0,
                            "partial_recoveries": 0,
                            "support_case_ids": set(),
                        },
                    )
                    entry["attempts"] += 1
                    if nxt.accepted:
                        entry["successes"] += 1
                    elif _partial_recovery(
                        strategy_name,
                        current,
                        nxt,
                        trace.official_test_miss,
                    ):
                        entry["partial_recoveries"] += 1
                    entry["support_case_ids"].add(trace.case_id)

        rankings: list[StrategyRecommendation] = []
        for (strategy_name, family), entry in totals.items():
            attempts = int(entry["attempts"])
            successes = int(entry["successes"])
            partial_recoveries = int(entry["partial_recoveries"])
            success_rate = (successes / attempts) if attempts else 0.0
            score = (success_rate * 10.0) + partial_recoveries + (attempts * 0.01)
            rankings.append(
                StrategyRecommendation(
                    strategy_name=strategy_name,
                    error_family=family,
                    attempts=attempts,
                    successes=successes,
                    partial_recoveries=partial_recoveries,
                    success_rate=success_rate,
                    support_case_ids=tuple(sorted(entry["support_case_ids"])),
                    score=score,
                )
            )

        self._rankings = sorted(
            rankings,
            key=lambda item: (
                -item.score,
                -item.successes,
                -item.partial_recoveries,
                -item.attempts,
                _STRATEGY_PRIORITY.get(item.strategy_name, 999),
                item.strategy_name,
                item.error_family,
            ),
        )
        return self

    def rank(
        self,
        *,
        error_family: str | None = None,
        limit: int | None = None,
    ) -> list[StrategyRecommendation]:
        if error_family is None:
            ranked = self._rankings
        else:
            ranked = [item for item in self._rankings if item.error_family == error_family]
        if limit is None:
            return list(ranked)
        return list(ranked[:limit])

    def recommend(
        self,
        *,
        harness_error_message: str,
        property_names: Sequence[str],
        official_test_miss: bool = False,
        limit: int = 3,
    ) -> list[StrategyRecommendation]:
        names = {name for name in property_names if name}
        family = _error_family(
            harness_error_message=harness_error_message,
            property_names=names,
            official_test_miss=official_test_miss,
        )
        labels = _strategy_labels(
            harness_error_message=harness_error_message,
            failures=tuple(
                PropertyFailureTrace(property_name=name, source="", error=None) for name in names
            ),
            official_test_miss=official_test_miss,
        )
        ranked = self.rank(error_family=family)
        filtered = [item for item in ranked if item.strategy_name in labels]
        if filtered:
            return filtered[:limit]
        if ranked:
            return ranked[:limit]
        return self.rank(limit=limit)


class TraceAnalyzer:
    """Ingest code-verification artifacts and summarize learned signals."""

    def __init__(
        self,
        *,
        cases: Sequence[CodeVerificationTrace],
        skipped_artifacts: Sequence[str],
        trace_artifact_count: int,
    ) -> None:
        self.cases = tuple(cases)
        self._skipped_artifacts = tuple(skipped_artifacts)
        self._trace_artifact_count = trace_artifact_count

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path]) -> TraceAnalyzer:
        cases: list[CodeVerificationTrace] = []
        skipped: list[str] = []
        trace_artifact_count = 0

        for raw_path in paths:
            path = Path(raw_path)
            artifact_name = _display_name(path)
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                skipped.append(artifact_name)
                continue
            extracted = _extract_traces(payload, artifact_name)
            if extracted:
                trace_artifact_count += 1
                cases.extend(extracted)
            else:
                skipped.append(artifact_name)

        return cls(
            cases=cases,
            skipped_artifacts=skipped,
            trace_artifact_count=trace_artifact_count,
        )

    @classmethod
    def from_payloads(
        cls,
        payloads: Sequence[dict[str, Any]],
        *,
        artifact_names: Sequence[str] | None = None,
    ) -> TraceAnalyzer:
        cases: list[CodeVerificationTrace] = []
        skipped: list[str] = []
        trace_artifact_count = 0

        names = list(artifact_names or ())
        for index, payload in enumerate(payloads):
            artifact_name = names[index] if index < len(names) else f"artifact-{index + 1}.json"
            extracted = _extract_traces(payload, artifact_name)
            if extracted:
                trace_artifact_count += 1
                cases.extend(extracted)
            else:
                skipped.append(artifact_name)

        return cls(
            cases=cases,
            skipped_artifacts=skipped,
            trace_artifact_count=trace_artifact_count,
        )

    def analyze(self) -> TraceAnalysis:
        property_rankings = tuple(PropertyRanker().fit(self.cases).rank())
        repair_rankings = tuple(RepairStrategy().fit(self.cases).rank())
        problem_type_rankings = tuple(self._rank_problem_types())
        return TraceAnalysis(
            trace_artifact_count=self._trace_artifact_count,
            skipped_artifact_count=len(self._skipped_artifacts),
            skipped_artifacts=self._skipped_artifacts,
            case_count=len(self.cases),
            property_rankings=property_rankings,
            repair_rankings=repair_rankings,
            problem_type_rankings=problem_type_rankings,
        )

    def _rank_problem_types(self) -> list[ProblemTypeScore]:
        totals: dict[str, dict[str, int]] = {}
        for trace in self.cases:
            for problem_type in trace.problem_types():
                entry = totals.setdefault(
                    problem_type,
                    {
                        "case_count": 0,
                        "official_test_misses": 0,
                        "repaired_cases": 0,
                        "accepted_baselines": 0,
                    },
                )
                entry["case_count"] += 1
                entry["official_test_misses"] += int(trace.official_test_miss)
                entry["repaired_cases"] += int(trace.repaired)
                entry["accepted_baselines"] += int(trace.baseline_accepted)

        ranked: list[ProblemTypeScore] = []
        for problem_type, entry in totals.items():
            score = float(
                entry["case_count"]
                + (5 * entry["official_test_misses"])
                + (3 * entry["repaired_cases"])
            )
            ranked.append(
                ProblemTypeScore(
                    problem_type=problem_type,
                    case_count=int(entry["case_count"]),
                    official_test_misses=int(entry["official_test_misses"]),
                    repaired_cases=int(entry["repaired_cases"]),
                    accepted_baselines=int(entry["accepted_baselines"]),
                    score=score,
                )
            )
        return sorted(
            ranked,
            key=lambda item: (
                -item.score,
                -item.official_test_misses,
                -item.repaired_cases,
                item.problem_type,
            ),
        )

    def learning_curve(
        self,
        *,
        prefix_sizes: Sequence[int] | None = None,
    ) -> tuple[LearningCurvePoint, ...]:
        if not self.cases:
            return ()

        if prefix_sizes is None:
            prefix_sizes = (1, max(1, len(self.cases) // 2), len(self.cases))

        points: list[LearningCurvePoint] = []
        for prefix_size in prefix_sizes:
            size = max(1, min(int(prefix_size), len(self.cases)))
            prefix = self.cases[:size]
            top_property = PropertyRanker().fit(prefix).rank(limit=1)
            top_strategy = RepairStrategy().fit(prefix).rank(limit=1)
            points.append(
                LearningCurvePoint(
                    prefix_size=size,
                    top_property=top_property[0].property_name if top_property else None,
                    top_property_score=top_property[0].score if top_property else 0.0,
                    top_strategy=top_strategy[0].strategy_name if top_strategy else None,
                    top_strategy_success_rate=(
                        top_strategy[0].success_rate if top_strategy else 0.0
                    ),
                )
            )
        return tuple(points)

    def demonstrate_improvement(
        self,
        *,
        prefix_sizes: Sequence[int] | None = None,
    ) -> LearningImprovement:
        points = self.learning_curve(prefix_sizes=prefix_sizes)
        if len(points) < 2:
            return LearningImprovement(points=points, improved=False)
        improved = (
            points[-1].top_property_score > points[0].top_property_score
            or points[-1].top_strategy_success_rate > points[0].top_strategy_success_rate
            or points[-1].top_strategy != points[0].top_strategy
        )
        return LearningImprovement(points=points, improved=improved)

"""EBRM scorer for extracted logical reasoning traces.

The scorer consumes already-extracted logical trace rows rather than raw text.
Each row carries a proposition identity, a truth polarity, confidence, support
links, contradiction links, and constraint IDs.  It assigns a continuous energy:
low energy means the trace is structurally coherent, while high energy means
the trace violates one or more reasoning constraints.

Spec: REQ-VERIFY-1656, SCENARIO-VERIFY-1656.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

JsonDict = dict[str, Any]

RUN_DATE = "20260509"
EXPERIMENT_ID = 1656
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1656_ebrm_trace_scorer.json")
SPEC_TRACES = ("REQ-VERIFY-1656", "SCENARIO-VERIFY-1656")
ACCURACY_GATE = 0.8

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "experiment_id",
    "ebrm_trace_scorer_ready",
    "continuous_energy_used",
    "synthetic_cases_total",
    "consistent_cases",
    "inconsistent_cases",
    "consistent_mean_energy",
    "inconsistent_mean_energy",
    "energy_gap",
    "score_accuracy",
    "spec_traces",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class LogicalTraceStep:
    """One extracted logical claim within a reasoning trace.

    Args:
        step_id: Stable local identifier for this step.
        proposition: Canonical claim/proposition identity.
        truth_value: True for an asserted claim, False for its negation.
        confidence: Extractor confidence in [0, 1].
        supports: Prior step IDs this step relies on.
        contradicts: Prior step IDs this step explicitly negates.
        constraint_ids: Reasoning constraints this step claims to satisfy.
    """

    step_id: str
    proposition: str
    truth_value: bool
    confidence: float = 1.0
    supports: tuple[str, ...] = ()
    contradicts: tuple[str, ...] = ()
    constraint_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class LogicalTrace:
    """A bounded extracted logical trace to score."""

    trace_id: str
    steps: tuple[LogicalTraceStep, ...]
    expected_inconsistent: bool | None = None


@dataclass(frozen=True)
class EBRMTraceScorerConfig:
    """Penalty weights for the deterministic trace energy model."""

    contradiction_weight: float = 3.0
    explicit_contradiction_bonus: float = 0.75
    unsupported_weight: float = 1.25
    confidence_weight: float = 1.0
    coverage_weight: float = 0.65
    ordering_weight: float = 0.4
    min_confidence: float = 0.75
    prediction_threshold: float = 1.0
    coherence_temperature: float = 1.0


@dataclass(frozen=True)
class EBRMTraceScore:
    """Continuous energy score plus auditable component breakdown."""

    trace_id: str
    energy: float
    coherence_score: float
    contradiction_energy: float
    unsupported_energy: float
    confidence_energy: float
    coverage_energy: float
    ordering_energy: float
    violation_count: int
    continuous_energy_used: bool = True

    @property
    def component_energies(self) -> dict[str, float]:
        """Return the component energies in the artifact row shape."""

        return {
            "contradiction_energy": self.contradiction_energy,
            "unsupported_energy": self.unsupported_energy,
            "confidence_energy": self.confidence_energy,
            "coverage_energy": self.coverage_energy,
            "ordering_energy": self.ordering_energy,
        }

    def to_dict(self) -> JsonDict:
        """Return a JSON-safe score row."""

        return {
            "trace_id": self.trace_id,
            "energy": self.energy,
            "coherence_score": self.coherence_score,
            "component_energies": self.component_energies,
            "violation_count": self.violation_count,
            "continuous_energy_used": self.continuous_energy_used,
        }


class EBRMTraceScorer:
    """Deterministic EBRM-style scorer over extracted logical traces."""

    def __init__(self, config: EBRMTraceScorerConfig | None = None) -> None:
        self.config = config or EBRMTraceScorerConfig()

    def score_trace(self, trace: LogicalTrace | Mapping[str, Any]) -> EBRMTraceScore:
        """Score one extracted trace with continuous structural energy."""

        normalized = _coerce_trace(trace)
        _validate_trace(normalized)
        step_index = {step.step_id: index for index, step in enumerate(normalized.steps)}
        seen_by_prop: dict[str, list[LogicalTraceStep]] = {}

        contradiction_energy = 0.0
        unsupported_energy = 0.0
        confidence_energy = 0.0
        coverage_energy = 0.0
        ordering_energy = 0.0
        violation_count = 0

        for index, step in enumerate(normalized.steps):
            bounded_confidence = _clamp01(step.confidence)
            confidence_energy += self.config.confidence_weight * max(
                0.0, self.config.min_confidence - bounded_confidence
            )
            coverage_energy += self.config.coverage_weight * float(not step.constraint_ids)

            for prior in seen_by_prop.get(step.proposition, []):
                if prior.truth_value is not step.truth_value:
                    contradiction_energy += (
                        self.config.contradiction_weight
                        * bounded_confidence
                        * _clamp01(prior.confidence)
                    )
                    violation_count += 1

            for linked_step_id in step.supports:
                linked_index = step_index.get(linked_step_id)
                unsupported = linked_index is None
                out_of_order = linked_index is not None and linked_index >= index
                unsupported_energy += self.config.unsupported_weight * float(unsupported)
                ordering_energy += self.config.ordering_weight * float(out_of_order)
                violation_count += int(unsupported or out_of_order)

            for linked_step_id in step.contradicts:
                linked_index = step_index.get(linked_step_id)
                out_of_order = linked_index is not None and linked_index >= index
                contradiction_energy += self.config.explicit_contradiction_bonus * float(
                    linked_index is not None
                )
                ordering_energy += self.config.ordering_weight * float(out_of_order)
                violation_count += int(linked_index is not None or out_of_order)

            seen_by_prop.setdefault(step.proposition, []).append(step)

        energy = round(
            contradiction_energy
            + unsupported_energy
            + confidence_energy
            + coverage_energy
            + ordering_energy,
            6,
        )
        coherence_score = round(math.exp(-energy / self.config.coherence_temperature), 6)
        return EBRMTraceScore(
            trace_id=normalized.trace_id,
            energy=energy,
            coherence_score=coherence_score,
            contradiction_energy=round(contradiction_energy, 6),
            unsupported_energy=round(unsupported_energy, 6),
            confidence_energy=round(confidence_energy, 6),
            coverage_energy=round(coverage_energy, 6),
            ordering_energy=round(ordering_energy, 6),
            violation_count=violation_count,
        )

    def score_traces(
        self,
        traces: Iterable[LogicalTrace | Mapping[str, Any]],
    ) -> list[EBRMTraceScore]:
        """Score a batch of traces while preserving caller order."""

        return [self.score_trace(trace) for trace in traces]


def default_synthetic_traces() -> list[LogicalTrace]:
    """Return paired coherent and inconsistent traces for Exp 1656 validation."""

    return [
        LogicalTrace(
            trace_id="inventory-consistent",
            expected_inconsistent=False,
            steps=(
                LogicalTraceStep(
                    "s1", "inventory_total_is_five", True, 0.96, constraint_ids=("counting",)
                ),
                LogicalTraceStep(
                    "s2",
                    "answer_uses_inventory_total",
                    True,
                    0.93,
                    supports=("s1",),
                    constraint_ids=("answer_grounding",),
                ),
            ),
        ),
        LogicalTrace(
            trace_id="inventory-contradiction",
            expected_inconsistent=True,
            steps=(
                LogicalTraceStep(
                    "s1", "inventory_total_is_five", True, 0.96, constraint_ids=("counting",)
                ),
                LogicalTraceStep(
                    "s2",
                    "inventory_total_is_five",
                    False,
                    0.95,
                    contradicts=("s1",),
                    constraint_ids=("counting",),
                ),
            ),
        ),
        LogicalTrace(
            trace_id="route-consistent",
            expected_inconsistent=False,
            steps=(
                LogicalTraceStep("s1", "east_bridge_open", True, 0.91, constraint_ids=("route",)),
                LogicalTraceStep(
                    "s2",
                    "route_uses_east_bridge",
                    True,
                    0.9,
                    supports=("s1",),
                    constraint_ids=("route",),
                ),
            ),
        ),
        LogicalTrace(
            trace_id="route-unsupported",
            expected_inconsistent=True,
            steps=(
                LogicalTraceStep("s1", "east_bridge_open", True, 0.91, constraint_ids=("route",)),
                LogicalTraceStep(
                    "s2",
                    "route_is_valid",
                    True,
                    0.52,
                    supports=("missing-route-link",),
                    constraint_ids=(),
                ),
            ),
        ),
    ]


def build_artifact(
    *,
    cases: Iterable[LogicalTrace | Mapping[str, Any]] | None = None,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Build the Exp 1656 artifact without writing it."""

    traces = list(default_synthetic_traces() if cases is None else cases)
    scores = EBRMTraceScorer().score_traces(traces)
    metrics = _aggregate_scores(traces, scores)
    ready = bool(
        metrics["continuous_energy_used"]
        and metrics["energy_gap"] > 0.0
        and metrics["score_accuracy"] >= ACCURACY_GATE
    )
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "schema": "ebrm_trace_scorer_v1",
        "spec_traces": list(SPEC_TRACES),
        "ebrm_trace_scorer_ready": ready,
        **metrics,
        "tests_run": list(tests_run),
        "honest_verdict": _honest_verdict(ready, metrics),
    }
    validate_artifact(artifact)
    return artifact


def write_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Write the Exp 1656 trace-scorer artifact and return its payload."""

    artifact = build_artifact(run_date=run_date, tests_run=tests_run)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert that an Exp 1656 artifact contains the required schema fields."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch"
    assert artifact["spec_traces"] == list(SPEC_TRACES), "spec_traces mismatch"
    assert 0.0 <= artifact["score_accuracy"] <= 1.0, "accuracy out of range"
    if artifact["status"] == "complete":
        assert artifact["ebrm_trace_scorer_ready"] is True, (
            "complete artifact requires ready scorer"
        )
        assert artifact["continuous_energy_used"] is True, (
            "complete artifact requires continuous energy"
        )
        assert artifact["energy_gap"] > 0.0, "complete artifact requires positive energy gap"
        assert artifact["score_accuracy"] >= ACCURACY_GATE, (
            "complete artifact requires accuracy gate"
        )


def _aggregate_scores(
    traces: Sequence[LogicalTrace | Mapping[str, Any]],
    scores: Sequence[EBRMTraceScore],
) -> JsonDict:
    coherent_scores = []
    inconsistent_scores = []
    correct = 0
    for raw_trace, score in zip(traces, scores, strict=True):
        trace = _coerce_trace(raw_trace)
        expected = bool(trace.expected_inconsistent)
        predicted = score.energy >= EBRMTraceScorerConfig().prediction_threshold
        correct += int(predicted is expected)
        if expected:
            inconsistent_scores.append(score.energy)
        else:
            coherent_scores.append(score.energy)
    consistent_mean = _mean(coherent_scores)
    inconsistent_mean = _mean(inconsistent_scores)
    return {
        "continuous_energy_used": all(score.continuous_energy_used for score in scores),
        "synthetic_cases_total": len(scores),
        "consistent_cases": len(coherent_scores),
        "inconsistent_cases": len(inconsistent_scores),
        "consistent_mean_energy": consistent_mean,
        "inconsistent_mean_energy": inconsistent_mean,
        "energy_gap": round(inconsistent_mean - consistent_mean, 6),
        "score_accuracy": round(correct / max(1, len(scores)), 6),
        "case_scores": [score.to_dict() for score in scores],
    }


def _coerce_trace(trace: LogicalTrace | Mapping[str, Any]) -> LogicalTrace:
    if isinstance(trace, LogicalTrace):
        return trace
    if not isinstance(trace, Mapping):
        raise ValueError("logical trace must be a LogicalTrace or mapping")
    raw_steps = trace.get("steps")
    if not raw_steps:
        raise ValueError("logical trace mapping must include non-empty steps")
    return LogicalTrace(
        trace_id=str(trace.get("trace_id", "trace")),
        expected_inconsistent=trace.get("expected_inconsistent"),
        steps=tuple(_coerce_step(step) for step in raw_steps),
    )


def _coerce_step(step: Mapping[str, Any]) -> LogicalTraceStep:
    proposition = str(step.get("proposition", step.get("claim", "")))
    truth_value = bool(step.get("truth_value", step.get("polarity", True)))
    return LogicalTraceStep(
        step_id=str(step.get("step_id", step.get("id", ""))),
        proposition=proposition,
        truth_value=truth_value,
        confidence=float(step.get("confidence", 1.0)),
        supports=_string_tuple(step.get("supports", ())),
        contradicts=_string_tuple(step.get("contradicts", ())),
        constraint_ids=_string_tuple(step.get("constraint_ids", ())),
    )


def _validate_trace(trace: LogicalTrace) -> None:
    if not trace.steps:
        raise ValueError("logical trace must contain at least one step")
    seen: set[str] = set()
    for step in trace.steps:
        if step.step_id in seen:
            raise ValueError(f"duplicate step_id: {step.step_id}")
        seen.add(step.step_id)


def _string_tuple(value: Any) -> tuple[str, ...]:
    return tuple(str(item) for item in value)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def _honest_verdict(ready: bool, metrics: Mapping[str, Any]) -> str:
    if ready:
        return (
            "complete: EBRM trace scorer separates extracted logical traces "
            f"with score_accuracy={metrics['score_accuracy']}"
        )
    return (
        "blocked: EBRM trace scorer did not satisfy separation gate; "
        f"energy_gap={metrics['energy_gap']} score_accuracy={metrics['score_accuracy']}"
    )


__all__ = [
    "ACCURACY_GATE",
    "DEFAULT_ARTIFACT_PATH",
    "EBRMTraceScore",
    "EBRMTraceScorer",
    "EBRMTraceScorerConfig",
    "LogicalTrace",
    "LogicalTraceStep",
    "REQUIRED_ARTIFACT_FIELDS",
    "SPEC_TRACES",
    "build_artifact",
    "default_synthetic_traces",
    "validate_artifact",
    "write_artifact",
]

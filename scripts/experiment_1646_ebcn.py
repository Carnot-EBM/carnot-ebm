#!/usr/bin/env python3
"""Exp 1646 EBCN reasoning-trace coherence prototype.

Spec: REQ-VERIFY-1646, SCENARIO-VERIFY-1646.

The prototype keeps the experiment local and deterministic. It encodes each
reasoning step as a proposition/truth-value row, rolls those rows through a
small state-space recurrence, and reuses the existing EBCN support and
contradiction attention heads to assign structural energy. Direct logical
inconsistencies should have higher energy and lower coherence scores.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from carnot.verify.ebcn_scorer import EBCNScorer, logical_trace_to_hidden_states

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1646_ebcn.json"
RUN_DATE = "20260509"
EXPERIMENT_ID = 1646
EXPERIMENT = "1646_ebcn"
SCHEMA = "ebcn_reasoning_trace_coherence_v1"
SPEC_TRACES = ["REQ-VERIFY-1646", "SCENARIO-VERIFY-1646"]
HIDDEN_STATE_SOURCE = "deterministic_reasoning_trace_state_space"
COHERENCE_THRESHOLD = 0.5
ACCURACY_GATE = 0.8

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "experiment_id",
    "ebcn_prototype_ready",
    "dual_head_attention_used",
    "state_space_transition_used",
    "autoregressive_generation_used",
    "hidden_state_source",
    "reasoning_trace_cases_total",
    "inconsistent_cases",
    "consistent_cases",
    "inconsistent_mean_energy",
    "consistent_mean_energy",
    "energy_gap",
    "coherence_score_accuracy",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class ReasoningStep:
    """One structured proposition extracted from a reasoning trace step."""

    text: str
    proposition: str
    truth_value: bool


@dataclass(frozen=True)
class ReasoningTraceCase:
    """One bounded synthetic reasoning trace for Exp 1646."""

    case_id: str
    expected_inconsistent: bool
    steps: tuple[ReasoningStep, ...]


@dataclass(frozen=True)
class ReasoningTraceScore:
    """Energy and coherence outputs for one scored reasoning trace."""

    case_id: str
    energy: float
    coherence_score: float
    support_energy: float
    contradiction_energy: float
    contradiction_pairs: tuple[tuple[str, int, int], ...]
    dual_head_attention_used: bool
    state_space_transition_used: bool
    autoregressive_generation_used: bool

    def to_dict(self, *, expected_inconsistent: bool) -> JsonDict:
        """Return the score in the JSON row shape used by the result artifact."""

        return {
            "case_id": self.case_id,
            "expected_inconsistent": expected_inconsistent,
            "energy": self.energy,
            "coherence_score": self.coherence_score,
            "support_energy": self.support_energy,
            "contradiction_energy": self.contradiction_energy,
            "contradiction_pairs": [list(pair) for pair in self.contradiction_pairs],
            "dual_head_attention_used": self.dual_head_attention_used,
            "state_space_transition_used": self.state_space_transition_used,
            "autoregressive_generation_used": self.autoregressive_generation_used,
        }


class ReasoningTraceEBCN:
    """Deterministic EBCN wrapper for structured reasoning traces."""

    def __init__(self, *, memory: float = 0.35, scorer: EBCNScorer | None = None) -> None:
        self.memory = float(memory)
        self.scorer = scorer or EBCNScorer()

    def score_case(self, case: ReasoningTraceCase) -> ReasoningTraceScore:
        """Score one reasoning trace after deterministic state-space rollout."""

        if not case.steps:
            raise ValueError("reasoning trace must contain at least one step")

        encoded = logical_trace_to_hidden_states(
            [(step.proposition, step.truth_value) for step in case.steps]
        )
        rolled_states = state_space_rollout(encoded.hidden_states, memory=self.memory)
        score = self.scorer.score_hidden_states(rolled_states, metadata=encoded.metadata)
        coherence_score = round(math.exp(-score.energy), 6)
        return ReasoningTraceScore(
            case_id=case.case_id,
            energy=score.energy,
            coherence_score=coherence_score,
            support_energy=score.support_energy,
            contradiction_energy=score.contradiction_energy,
            contradiction_pairs=tuple(score.contradiction_pairs),
            dual_head_attention_used=score.dual_head_attention_used,
            state_space_transition_used=True,
            autoregressive_generation_used=score.autoregressive_generation_used,
        )


def default_reasoning_trace_cases() -> list[ReasoningTraceCase]:
    """Return deterministic reasoning traces with paired consistent/contradictory rows."""

    return [
        ReasoningTraceCase(
            case_id="inventory-consistent",
            expected_inconsistent=False,
            steps=(
                ReasoningStep("There are 2 red blocks.", "red_blocks_counted", True),
                ReasoningStep("There are 3 blue blocks.", "blue_blocks_counted", True),
                ReasoningStep("The total is 5 blocks.", "total_is_five", True),
                ReasoningStep("The final answer keeps total_is_five.", "total_is_five", True),
            ),
        ),
        ReasoningTraceCase(
            case_id="inventory-contradiction",
            expected_inconsistent=True,
            steps=(
                ReasoningStep("There are 2 red blocks.", "red_blocks_counted", True),
                ReasoningStep("There are 3 blue blocks.", "blue_blocks_counted", True),
                ReasoningStep("The total is 5 blocks.", "total_is_five", True),
                ReasoningStep("The final answer says the total is not 5.", "total_is_five", False),
            ),
        ),
        ReasoningTraceCase(
            case_id="route-consistent",
            expected_inconsistent=False,
            steps=(
                ReasoningStep("The north bridge is closed.", "north_bridge_open", False),
                ReasoningStep("The east bridge is open.", "east_bridge_open", True),
                ReasoningStep("The route uses the east bridge.", "route_uses_east_bridge", True),
                ReasoningStep("The route remains feasible.", "route_feasible", True),
            ),
        ),
        ReasoningTraceCase(
            case_id="route-contradiction",
            expected_inconsistent=True,
            steps=(
                ReasoningStep("The north bridge is closed.", "north_bridge_open", False),
                ReasoningStep("The east bridge is open.", "east_bridge_open", True),
                ReasoningStep("The route uses the east bridge.", "route_uses_east_bridge", True),
                ReasoningStep("The route does not use the east bridge.", "route_uses_east_bridge", False),
            ),
        ),
        ReasoningTraceCase(
            case_id="permission-consistent",
            expected_inconsistent=False,
            steps=(
                ReasoningStep("The policy allows read access.", "read_access_allowed", True),
                ReasoningStep("The user requested read access.", "read_access_requested", True),
                ReasoningStep("The verifier grants read access.", "read_access_granted", True),
                ReasoningStep("The trace preserves read access.", "read_access_granted", True),
            ),
        ),
        ReasoningTraceCase(
            case_id="permission-contradiction",
            expected_inconsistent=True,
            steps=(
                ReasoningStep("The policy allows read access.", "read_access_allowed", True),
                ReasoningStep("The user requested read access.", "read_access_requested", True),
                ReasoningStep("The verifier grants read access.", "read_access_granted", True),
                ReasoningStep("The conclusion denies read access.", "read_access_granted", False),
            ),
        ),
    ]


def state_space_rollout(hidden_states: np.ndarray, *, memory: float) -> np.ndarray:
    """Apply a simple deterministic recurrence over reasoning-step hidden states."""

    states = np.asarray(hidden_states, dtype=np.float32)
    rolled = np.empty_like(states)
    previous = np.zeros(states.shape[1], dtype=np.float32)
    for index, row in enumerate(states):
        current = (1.0 - memory) * row + memory * previous
        rolled[index] = current
        previous = current
    return rolled


def evaluate_cases(
    cases: Iterable[ReasoningTraceCase] | None = None,
    *,
    scorer: ReasoningTraceEBCN | None = None,
) -> list[JsonDict]:
    """Score each reasoning trace and return serializable case rows."""

    active_scorer = scorer or ReasoningTraceEBCN()
    return [
        active_scorer.score_case(case).to_dict(
            expected_inconsistent=case.expected_inconsistent
        )
        for case in (default_reasoning_trace_cases() if cases is None else cases)
    ]


def aggregate_case_scores(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Aggregate case-level EBCN scores into Exp 1646 artifact metrics."""

    if not rows:
        return _empty_metrics()

    inconsistent_energies = [
        float(row["energy"]) for row in rows if row["expected_inconsistent"] is True
    ]
    consistent_energies = [
        float(row["energy"]) for row in rows if row["expected_inconsistent"] is False
    ]
    correct_predictions = sum(
        (float(row["coherence_score"]) < COHERENCE_THRESHOLD)
        is bool(row["expected_inconsistent"])
        for row in rows
    )
    inconsistent_mean = _mean(inconsistent_energies)
    consistent_mean = _mean(consistent_energies)
    return {
        "ebcn_prototype_ready": False,
        "dual_head_attention_used": all(bool(row["dual_head_attention_used"]) for row in rows),
        "state_space_transition_used": all(bool(row["state_space_transition_used"]) for row in rows),
        "autoregressive_generation_used": any(bool(row["autoregressive_generation_used"]) for row in rows),
        "hidden_state_source": HIDDEN_STATE_SOURCE,
        "reasoning_trace_cases_total": len(rows),
        "inconsistent_cases": len(inconsistent_energies),
        "consistent_cases": len(consistent_energies),
        "inconsistent_mean_energy": inconsistent_mean,
        "consistent_mean_energy": consistent_mean,
        "energy_gap": round(inconsistent_mean - consistent_mean, 6),
        "coherence_score_accuracy": round(correct_predictions / len(rows), 6),
        "coherence_threshold": COHERENCE_THRESHOLD,
        "case_scores": [dict(row) for row in rows],
    }


def build_artifact(
    *,
    cases: Iterable[ReasoningTraceCase] | None = None,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Build the Exp 1646 terminal artifact without writing it."""

    rows = evaluate_cases(cases)
    metrics = aggregate_case_scores(rows)
    ready = bool(
        metrics["dual_head_attention_used"]
        and metrics["state_space_transition_used"]
        and metrics["autoregressive_generation_used"] is False
        and metrics["energy_gap"] > 0.0
        and metrics["coherence_score_accuracy"] >= ACCURACY_GATE
    )
    metrics["ebcn_prototype_ready"] = ready
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "spec_traces": list(SPEC_TRACES),
        **metrics,
        "tests_run": list(tests_run),
        "honest_verdict": _honest_verdict(ready, metrics),
    }
    validate_artifact(artifact)
    return artifact


def run_experiment(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] = (),
) -> JsonDict:
    """Run Exp 1646 and write `results/experiment_1646_ebcn.json`."""

    artifact = build_artifact(run_date=run_date, tests_run=tests_run)
    _write_json(Path(output_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert that an Exp 1646 artifact is internally consistent."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch"
    assert 0.0 <= artifact["coherence_score_accuracy"] <= 1.0, "accuracy out of range"
    if artifact["status"] == "complete":
        assert artifact["ebcn_prototype_ready"] is True, "complete artifact requires ready prototype"
        assert artifact["dual_head_attention_used"] is True, "complete artifact requires dual heads"
        assert artifact["state_space_transition_used"] is True, "complete artifact requires state space"
        assert artifact["autoregressive_generation_used"] is False, "complete artifact cannot use generation"
        assert artifact["energy_gap"] > 0.0, "complete artifact requires positive energy gap"
        assert artifact["coherence_score_accuracy"] >= ACCURACY_GATE, (
            "complete artifact requires accuracy gate"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for Exp 1646."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--run-date", default=RUN_DATE)
    args = parser.parse_args(argv)
    artifact = run_experiment(output_path=args.output, run_date=args.run_date)
    print(
        "ready={ready} coherence_score_accuracy={accuracy} energy_gap={gap}".format(
            ready=artifact["ebcn_prototype_ready"],
            accuracy=artifact["coherence_score_accuracy"],
            gap=artifact["energy_gap"],
        )
    )
    return int(artifact["status"] != "complete")


def _honest_verdict(ready: bool, metrics: Mapping[str, Any]) -> str:
    if ready:
        return (
            "complete: EBCN state-space reasoning-trace prototype separates "
            "direct logical inconsistencies with coherence_score_accuracy="
            f"{metrics['coherence_score_accuracy']}"
        )
    return (
        "blocked: EBCN coherence prototype did not satisfy the separation gate; "
        f"energy_gap={metrics['energy_gap']}, "
        f"coherence_score_accuracy={metrics['coherence_score_accuracy']}"
    )


def _empty_metrics() -> JsonDict:
    return {
        "ebcn_prototype_ready": False,
        "dual_head_attention_used": True,
        "state_space_transition_used": True,
        "autoregressive_generation_used": False,
        "hidden_state_source": HIDDEN_STATE_SOURCE,
        "reasoning_trace_cases_total": 0,
        "inconsistent_cases": 0,
        "consistent_cases": 0,
        "inconsistent_mean_energy": 0.0,
        "consistent_mean_energy": 0.0,
        "energy_gap": 0.0,
        "coherence_score_accuracy": 0.0,
        "coherence_threshold": COHERENCE_THRESHOLD,
        "case_scores": [],
    }


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

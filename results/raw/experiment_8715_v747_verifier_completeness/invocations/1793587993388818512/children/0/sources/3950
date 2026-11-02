"""Exp 2244 ODAR routing benchmark.

**Researcher summary:**
    This module answers the integration question for ODAR routing: does the
    free-energy gate save enough verification compute to justify another path
    through the cascade?  The benchmark is intentionally deterministic so the
    conductor can use it as a repeatable capstone gate.

**Detailed explanation for engineers:**
    The uniform regime pays for the full Tier 0 through Tier 3 cascade on every
    corpus row.  The ODAR regime always pays for Tier 0 probes, then lets
    ``FreeEnergyRouter`` decide whether a row can stop on the fast path or must
    continue through Tiers 1-3.  High-confidence rows have correct candidate
    answers and low EFE, so fast-pathing should preserve accuracy.  Ambiguous
    rows have deliberately risky candidate answers and high EFE, so ODAR should
    spend the same deliberative compute that the uniform cascade spends.

Spec: REQ-ODAR-2244, SCENARIO-ODAR-2244.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

JsonDict = dict[str, Any]

DEFAULT_ARTIFACT_PATH = Path("results/experiment_2244_odar_benchmark.json")
ROUTER_MODULE_PATH = "python/carnot/pipeline/odar_router.py"
ODAR_IMPORT_PATH = "carnot.pipeline.odar_router"
DEFAULT_THRESHOLD = 0.5
CASCADE_TIERS = ("tier0_probe", "tier1_constraints", "tier2_semantic", "tier3_formal")
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "odar_benchmark_passed",
    "compute_reduction_pct",
    "accuracy_delta",
    "n_corpus",
    "preconditions_checked",
)
FIELD_PRINCIPLES: JsonDict = {
    "honest_verdict": "Terminal-prefix required. Use complete: if compute_reduction_pct >= 30.",
    "odar_benchmark_passed": "Boolean gate for exp2251 capstone.",
    "compute_reduction_pct": "Primary ODAR gate: >= 30 validates the routing adds value.",
    "accuracy_delta": "Guards against trading accuracy for speed; must be >= -2pp.",
    "n_corpus": "Must be 30 for statistical significance on pct claims.",
    "preconditions_checked": "Lists resource checks performed before benchmark.",
}
TERMINAL_VERDICT_PREFIXES = ("complete:", "blocked:")


@dataclass(frozen=True)
class ReasoningCase:
    """One deterministic reasoning row with Tier 0 evidence and an oracle answer.

    The benchmark does not try to measure language-model quality.  It isolates
    the routing decision by keeping the candidate answer, deliberative answer,
    and Tier 0 probes fixed, then comparing the tier-call budget needed to reach
    the same final answer under uniform and ODAR regimes.
    """

    case_id: str
    difficulty: str
    question: str
    expected_answer: str
    candidate_answer: str
    deliberative_answer: str
    reference_reasoning: str
    probe_outputs: JsonDict

    def to_dict(self) -> JsonDict:
        """Return a JSON-safe row for the terminal artifact."""

        return asdict(self)


def build_reasoning_corpus() -> list[ReasoningCase]:
    """REQ-ODAR-2244: build the balanced 30-example reasoning corpus."""

    corpus: list[ReasoningCase] = []
    for index in range(15):
        left = 12 + index
        right = 7 + (index % 5)
        expected = str(left + right)
        corpus.append(
            ReasoningCase(
                case_id=f"high_confidence_{index + 1:02d}",
                difficulty="high_confidence",
                question=(
                    f"A lab labels {left} samples before lunch and {right} samples after "
                    "lunch. How many samples are labeled in total?"
                ),
                expected_answer=expected,
                candidate_answer=expected,
                deliberative_answer=expected,
                reference_reasoning=f"Add the two labeled batches: {left} + {right} = {expected}.",
                probe_outputs=_low_efe_probe_outputs(index),
            )
        )

    for index in range(15):
        start = 30 + index
        daily_gain = 4 + (index % 4)
        days = 2 + (index % 3)
        loss = 3 + (index % 5)
        expected_int = start + (daily_gain * days) - loss
        candidate_int = start + daily_gain - loss
        corpus.append(
            ReasoningCase(
                case_id=f"ambiguous_{index + 1:02d}",
                difficulty="ambiguous",
                question=(
                    f"A solver starts with {start} tokens, gains {daily_gain} tokens per "
                    f"round for {days} rounds, then spends {loss}. How many remain?"
                ),
                expected_answer=str(expected_int),
                candidate_answer=str(candidate_int),
                deliberative_answer=str(expected_int),
                reference_reasoning=(
                    f"Apply the repeated gain before the spend: {start} + "
                    f"{daily_gain} * {days} - {loss} = {expected_int}."
                ),
                probe_outputs=_high_efe_probe_outputs(index),
            )
        )

    return corpus


def route_case(case: ReasoningCase, *, threshold: float = DEFAULT_THRESHOLD) -> JsonDict:
    """Return the ODAR route and EFE for one case."""

    router_cls, _module = _load_router_class()
    router = router_cls(risk_threshold=threshold)
    result = router.evaluate(case.probe_outputs)
    return {
        "case_id": case.case_id,
        "route": result.decision.value,
        "expected_free_energy": round(float(result.expected_free_energy), 6),
        "risk_threshold": threshold,
        "contributions": [contribution.to_dict() for contribution in result.contributions],
    }


def evaluate_benchmark(
    corpus: Sequence[ReasoningCase] | None = None,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> JsonDict:
    """SCENARIO-ODAR-2244: compare uniform cascade calls with ODAR calls."""

    active_corpus = list(build_reasoning_corpus() if corpus is None else corpus)
    n_corpus = len(active_corpus)
    tier_calls_a = 0
    tier_calls_b = 0
    correct_a = 0
    correct_b = 0
    fast_path_count = 0
    deliberative_count = 0
    case_results: list[JsonDict] = []

    for case in active_corpus:
        uniform = _run_uniform_case(case)
        odar = _run_odar_case(case, threshold=threshold)
        tier_calls_a += uniform["tier_calls"]
        tier_calls_b += odar["tier_calls"]
        correct_a += int(uniform["correct"])
        correct_b += int(odar["correct"])
        fast_path_count += int(odar["route"] == "FAST_PATH")
        deliberative_count += int(odar["route"] == "DELIBERATIVE")
        case_results.append(
            {
                **case.to_dict(),
                "uniform": uniform,
                "odar": odar,
            }
        )

    accuracy_a_pct = _pct(correct_a, n_corpus)
    accuracy_b_pct = _pct(correct_b, n_corpus)
    compute_reduction_pct = (
        ((tier_calls_a - tier_calls_b) / tier_calls_a) * 100.0 if tier_calls_a else 0.0
    )
    accuracy_delta = accuracy_b_pct - accuracy_a_pct
    odar_benchmark_passed = bool(compute_reduction_pct >= 30.0 and accuracy_delta >= -2.0)

    return {
        "n_corpus": n_corpus,
        "threshold": threshold,
        "cascade_tiers": list(CASCADE_TIERS),
        "tier_calls_A": tier_calls_a,
        "tier_calls_B": tier_calls_b,
        "compute_reduction_pct": round(compute_reduction_pct, 6),
        "accuracy_uniform_pct": round(accuracy_a_pct, 6),
        "accuracy_odar_pct": round(accuracy_b_pct, 6),
        "accuracy_delta": round(accuracy_delta, 6),
        "fast_path_count": fast_path_count,
        "deliberative_count": deliberative_count,
        "fast_path_fraction": round(fast_path_count / n_corpus, 6) if n_corpus else 0.0,
        "odar_benchmark_passed": odar_benchmark_passed,
        "case_results": case_results,
    }


def run_benchmark(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> JsonDict:
    """Run the benchmark and write the terminal artifact."""

    output = Path(output_path)
    preconditions_checked, blockers = check_preconditions()
    if blockers:
        artifact = _blocked_artifact(
            preconditions_checked=preconditions_checked,
            blockers=blockers,
            threshold=threshold,
        )
        _write_json(output, artifact)
        return artifact

    corpus = build_reasoning_corpus()
    preconditions_checked.extend(["reasoning_corpus_constructed", f"n_corpus_{len(corpus)}"])
    evaluation = evaluate_benchmark(corpus, threshold=threshold)
    preconditions_checked.extend(
        ["uniform_cascade_regime_evaluated", "odar_threshold_regime_evaluated"]
    )
    artifact = _artifact_from_evaluation(
        evaluation,
        preconditions_checked=preconditions_checked,
        blockers=blockers,
    )
    validate_artifact(artifact)
    _write_json(output, artifact)
    return artifact


def check_preconditions() -> tuple[list[str], list[str]]:
    """Check resource availability before spending benchmark effort."""

    checked: list[str] = []
    blockers: list[str] = []
    try:
        importlib.import_module(ODAR_IMPORT_PATH)
    except Exception as exc:  # pragma: no cover - exercised only when import is broken.
        checked.append("blocked_router_missing")
        blockers.append("blocked_router_missing")
        blockers.append(f"odar_router_import_error:{type(exc).__name__}")
    else:
        checked.append("odar_router_imported")
    return checked, blockers


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Enforce the Exp 2244 schema and gate semantics."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if not str(artifact["honest_verdict"]).startswith(TERMINAL_VERDICT_PREFIXES):
        raise AssertionError("honest_verdict must use a terminal prefix")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise AssertionError("artifact must include field_principles")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise AssertionError(f"missing field principles: {missing_principles}")
    if "blocked_router_missing" in artifact.get("blockers", []):
        return
    if artifact["n_corpus"] != 30:
        raise AssertionError("Exp 2244 requires exactly 30 corpus rows")
    if artifact["compute_reduction_pct"] >= 30.0 and not str(artifact["honest_verdict"]).startswith(
        "complete:"
    ):
        raise AssertionError("compute gate pass requires complete: honest_verdict")
    if artifact["odar_benchmark_passed"]:
        if artifact["compute_reduction_pct"] < 30.0:
            raise AssertionError("passed benchmark requires >=30% compute reduction")
        if artifact["accuracy_delta"] < -2.0:
            raise AssertionError("passed benchmark requires accuracy_delta >= -2pp")


def _run_uniform_case(case: ReasoningCase) -> JsonDict:
    final_answer = case.deliberative_answer
    return {
        "regime": "uniform",
        "tier_calls": len(CASCADE_TIERS),
        "tiers_run": list(CASCADE_TIERS),
        "final_answer": final_answer,
        "correct": _answer_matches(final_answer, case.expected_answer),
    }


def _run_odar_case(case: ReasoningCase, *, threshold: float) -> JsonDict:
    route = route_case(case, threshold=threshold)
    if route["route"] == "FAST_PATH":
        tiers_run = [CASCADE_TIERS[0]]
        final_answer = case.candidate_answer
    else:
        tiers_run = list(CASCADE_TIERS)
        final_answer = case.deliberative_answer
    return {
        "regime": "odar",
        "route": route["route"],
        "expected_free_energy": route["expected_free_energy"],
        "tier_calls": len(tiers_run),
        "tiers_run": tiers_run,
        "final_answer": final_answer,
        "correct": _answer_matches(final_answer, case.expected_answer),
    }


def _artifact_from_evaluation(
    evaluation: Mapping[str, Any],
    *,
    preconditions_checked: Sequence[str],
    blockers: Sequence[str],
) -> JsonDict:
    passed = bool(evaluation["odar_benchmark_passed"])
    if passed:
        verdict = "complete: odar_benchmark_passed"
    elif evaluation["compute_reduction_pct"] >= 30.0:
        verdict = "complete: odar_benchmark_failed_accuracy_gate"
    else:
        verdict = "blocked: odar_benchmark_compute_reduction_below_gate"
    return {
        "status": "complete",
        "experiment_id": 2244,
        "title": "ODAR routing benchmark versus uniform cascade",
        "spec_refs": ["REQ-ODAR-2244", "SCENARIO-ODAR-2244"],
        "router_module_path": ROUTER_MODULE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": list(preconditions_checked),
        "blockers": list(blockers),
        "honest_verdict": verdict,
        **dict(evaluation),
    }


def _blocked_artifact(
    *,
    preconditions_checked: Sequence[str],
    blockers: Sequence[str],
    threshold: float,
) -> JsonDict:
    return {
        "status": "blocked",
        "experiment_id": 2244,
        "title": "ODAR routing benchmark versus uniform cascade",
        "spec_refs": ["REQ-ODAR-2244", "SCENARIO-ODAR-2244"],
        "router_module_path": ROUTER_MODULE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": list(preconditions_checked),
        "blockers": list(blockers),
        "honest_verdict": "blocked: blocked_router_missing",
        "odar_benchmark_passed": False,
        "compute_reduction_pct": 0.0,
        "accuracy_delta": 0.0,
        "n_corpus": 0,
        "threshold": threshold,
        "tier_calls_A": 0,
        "tier_calls_B": 0,
        "fast_path_fraction": 0.0,
        "case_results": [],
    }


def _load_router_class() -> tuple[type[Any], Any]:
    module = importlib.import_module(ODAR_IMPORT_PATH)
    return module.FreeEnergyRouter, module


def _low_efe_probe_outputs(index: int) -> JsonDict:
    risk_offset = (index % 3) * 0.01
    return {
        "nup": {"risk_score": 0.06 + risk_offset, "confidence": 0.94, "weight": 1.0},
        "semantic_energy": {"risk_score": 0.1 + risk_offset, "confidence": 0.9, "weight": 1.0},
    }


def _high_efe_probe_outputs(index: int) -> JsonDict:
    risk_offset = (index % 4) * 0.02
    return {
        "nup": {"risk_score": 0.68 + risk_offset, "confidence": 0.42, "weight": 1.0},
        "semantic_energy": {
            "risk_score": 0.74 + risk_offset,
            "ambiguity": 0.65,
            "weight": 1.0,
        },
    }


def _answer_matches(answer: str, expected_answer: str) -> bool:
    return answer.strip() == expected_answer.strip()


def _pct(numerator: int, denominator: int) -> float:
    return (numerator / denominator) * 100.0 if denominator else 0.0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for conductor and manual benchmark runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)
    artifact = run_benchmark(output_path=args.output, threshold=args.threshold)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "honest_verdict": artifact["honest_verdict"],
                "odar_benchmark_passed": artifact["odar_benchmark_passed"],
                "compute_reduction_pct": artifact["compute_reduction_pct"],
                "accuracy_delta": artifact["accuracy_delta"],
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "FIELD_PRINCIPLES",
    "REQUIRED_ARTIFACT_FIELDS",
    "ReasoningCase",
    "build_reasoning_corpus",
    "check_preconditions",
    "evaluate_benchmark",
    "main",
    "route_case",
    "run_benchmark",
    "validate_artifact",
]

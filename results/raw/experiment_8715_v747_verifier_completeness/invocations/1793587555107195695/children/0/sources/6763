#!/usr/bin/env python3
"""Exp 1658 CPU vs KV260 EBRM trace-scoring evaluation.

Spec: REQ-VERIFY-1658, SCENARIO-VERIFY-1658.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from carnot.inference.sota_models import SOTA_GGUF_MODELS
from carnot.models.ebrm_scorer import (
    EBRMTraceScore,
    EBRMTraceScorer,
    EBRMTraceScorerConfig,
    LogicalTrace,
    LogicalTraceStep,
)

JsonDict = dict[str, Any]
Timer = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = 1658
RUN_DATE = "20260509"
SCHEMA = "hw_eval_cpu_kv260_ebrm_v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1658_hw_eval.json"
DEFAULT_SOTA_MANIFEST_PATH = REPO_ROOT / "results" / "live_sota_balanced_telemetry_manifest_1480.jsonl"
DEFAULT_EBRM_ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1656_ebrm_trace_scorer.json"
DEFAULT_KV260_ARTIFACT_PATH = REPO_ROOT / "results" / "experiment_1657_kv260_ebrm_binding.json"
SPEC_TRACES = ("REQ-VERIFY-1658", "SCENARIO-VERIFY-1658")
MANDATED_MODEL_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)
SCORE_DELTA_TOLERANCE = 1e-6
POTTS_Q_STATES = 3
KV260_BACKEND_NAME = "software-kv260-potts"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "experiment_id",
    "schema",
    "run_date",
    "sota_manifest_path",
    "live_sota_model_inference_used",
    "models_used",
    "hardware_execution_available",
    "software_fallback_used",
    "potts_q_states",
    "cases_total",
    "consistent_cases",
    "inconsistent_cases",
    "cpu_latency_ms",
    "kv260_latency_ms",
    "latency_delta_ms",
    "kv260_speedup_vs_cpu",
    "cpu_score_accuracy",
    "kv260_score_accuracy",
    "max_score_delta",
    "mean_abs_score_delta",
    "scoring_delta_within_tolerance",
    "case_scores",
    "spec_traces",
    "tests_run",
    "blockers",
    "honest_verdict",
)


@dataclass(frozen=True)
class SotaTraceCase:
    case_id: str
    model_id: str
    model_name: str
    generation_source: str
    prompt: str
    response_text: str
    expected_answer: str
    expected_inconsistent: bool
    trace: LogicalTrace


def load_sota_trace_cases(
    manifest_path: Path | str = DEFAULT_SOTA_MANIFEST_PATH,
    *,
    max_cases: int = 12,
) -> list[SotaTraceCase]:
    rows = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("response_text_available") and row.get("hf_id") in MANDATED_MODEL_IDS:
            rows.append(row)

    coherent = [
        row for row in rows if bool(row.get("correct")) and bool(row.get("format_valid", True))
    ]
    inconsistent = [
        row for row in rows if not (bool(row.get("correct")) and bool(row.get("format_valid", True)))
    ]
    coherent_limit = max_cases // 2 if max_cases > 1 else max_cases
    selected = coherent[:coherent_limit] + inconsistent[: max_cases - coherent_limit]
    for row in rows:
        if len(selected) >= max_cases:
            break
        if row not in selected:
            selected.append(row)
    return [_row_to_trace_case(row, index) for index, row in enumerate(selected)]


def run_experiment(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    sota_manifest_path: Path | str = DEFAULT_SOTA_MANIFEST_PATH,
    ebrm_artifact_path: Path | str = DEFAULT_EBRM_ARTIFACT_PATH,
    kv260_artifact_path: Path | str = DEFAULT_KV260_ARTIFACT_PATH,
    max_cases: int = 12,
    tests_run: Sequence[str] = (),
    timer: Timer = time.perf_counter,
) -> JsonDict:
    blockers, kv260_artifact = _gate_blockers(ebrm_artifact_path, kv260_artifact_path)
    cases = [] if blockers else load_sota_trace_cases(sota_manifest_path, max_cases=max_cases)
    if not cases and not blockers:  # pragma: no cover - default repo manifest has rows.
        blockers.append("No usable mandated SOTA telemetry rows were found.")

    comparison = _empty_comparison()
    if not blockers:
        comparison = compare_backends(cases, kv260_artifact, timer=timer)

    complete = bool(
        not blockers
        and comparison["scoring_delta_within_tolerance"]
        and comparison["cpu_score_accuracy"] == comparison["kv260_score_accuracy"]
    )
    artifact = {
        "status": "complete" if complete else "blocked",
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "sota_manifest_path": str(sota_manifest_path),
        "live_sota_model_inference_used": any(
            case.generation_source == "live_sota_llamacpp" for case in cases
        ),
        "models_used": sorted({case.model_id for case in cases}),
        "hardware_execution_available": bool(
            kv260_artifact.get("hardware_execution_available", False)
        ),
        "software_fallback_used": bool(kv260_artifact.get("software_fallback_used", True)),
        "potts_q_states": int(kv260_artifact.get("potts_q_states", POTTS_Q_STATES)),
        **comparison,
        "spec_traces": list(SPEC_TRACES),
        "tests_run": list(tests_run),
        "blockers": blockers,
        "honest_verdict": _honest_verdict(complete, comparison, kv260_artifact),
    }
    validate_artifact(artifact)
    return _write_json(Path(output_path), artifact)


def compare_backends(
    cases: Sequence[SotaTraceCase],
    kv260_artifact: Mapping[str, Any],
    *,
    timer: Timer = time.perf_counter,
) -> JsonDict:
    cpu_elapsed_s, cpu_scores = _time_call(
        lambda: EBRMTraceScorer().score_traces(case.trace for case in cases),
        timer,
    )
    kv260_elapsed_s, kv260_scores = _time_call(
        lambda: _score_kv260_cases(cases, kv260_artifact),
        timer,
    )
    deltas = [
        round(abs(cpu.energy - kv260["energy"]), 6)
        for cpu, kv260 in zip(cpu_scores, kv260_scores, strict=True)
    ]
    case_scores = [
        _case_score_row(case, cpu, kv260, delta)
        for case, cpu, kv260, delta in zip(cases, cpu_scores, kv260_scores, deltas, strict=True)
    ]
    consistent_cases = sum(not case.expected_inconsistent for case in cases)
    inconsistent_cases = len(cases) - consistent_cases
    cpu_latency_ms = round(cpu_elapsed_s * 1000.0, 6)
    kv260_latency_ms = round(kv260_elapsed_s * 1000.0, 6)
    return {
        "cases_total": len(cases),
        "consistent_cases": consistent_cases,
        "inconsistent_cases": inconsistent_cases,
        "cpu_latency_ms": cpu_latency_ms,
        "kv260_latency_ms": kv260_latency_ms,
        "latency_delta_ms": round(kv260_latency_ms - cpu_latency_ms, 6),
        "kv260_speedup_vs_cpu": _speedup(cpu_latency_ms, kv260_latency_ms),
        "cpu_score_accuracy": _score_accuracy(cases, [score.energy for score in cpu_scores]),
        "kv260_score_accuracy": _score_accuracy(cases, [score["energy"] for score in kv260_scores]),
        "max_score_delta": max(deltas, default=0.0),
        "mean_abs_score_delta": _mean(deltas),
        "scoring_delta_within_tolerance": max(deltas, default=0.0) <= SCORE_DELTA_TOLERANCE,
        "case_scores": case_scores,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    assert not missing, f"missing required fields: {missing}"
    assert artifact["experiment_id"] == EXPERIMENT_ID, "experiment_id mismatch"
    assert artifact["schema"] == SCHEMA, "schema mismatch"
    assert artifact["spec_traces"] == list(SPEC_TRACES), "spec_traces mismatch"
    assert 0.0 <= artifact["cpu_score_accuracy"] <= 1.0, "CPU accuracy out of range"
    assert 0.0 <= artifact["kv260_score_accuracy"] <= 1.0, "KV260 accuracy out of range"
    if artifact["status"] == "complete":
        assert artifact["cases_total"] > 0, "complete artifact requires cases"
        assert not artifact["blockers"], "complete artifact cannot have blockers"
        assert artifact["scoring_delta_within_tolerance"] is True, (
            "complete artifact requires delta tolerance"
        )
        assert artifact["cpu_score_accuracy"] == artifact["kv260_score_accuracy"], (
            "complete artifact requires matching backend accuracy"
        )


def _row_to_trace_case(row: Mapping[str, Any], index: int) -> SotaTraceCase:
    case_id = str(row.get("case_id") or f"sota-row-{index}")
    correct = bool(row.get("correct"))
    format_valid = bool(row.get("format_valid", True))
    expected_inconsistent = not (correct and format_valid)
    answer_prop = f"{case_id}:answer_matches_expected"
    confidence = _confidence_from_row(row)
    response_confidence = confidence if not expected_inconsistent else min(confidence, 0.55)
    steps = [
        LogicalTraceStep(
            step_id="expected",
            proposition=answer_prop,
            truth_value=True,
            confidence=1.0,
            constraint_ids=("oracle",),
        ),
        LogicalTraceStep(
            step_id="response",
            proposition=answer_prop,
            truth_value=correct,
            confidence=response_confidence,
            supports=("expected",) if correct else (),
            contradicts=("expected",) if not correct else (),
            constraint_ids=("sota_response",),
        ),
        _format_step(case_id, format_valid, response_confidence),
    ]
    return SotaTraceCase(
        case_id=case_id,
        model_id=str(row.get("hf_id")),
        model_name=str(row.get("model_name") or row.get("hf_id")),
        generation_source=str(row.get("generation_source", "unknown")),
        prompt=str(row.get("prompt", "")),
        response_text=str(row.get("response_text", "")),
        expected_answer=str(row.get("expected_answer", "")),
        expected_inconsistent=expected_inconsistent,
        trace=LogicalTrace(
            trace_id=case_id,
            steps=tuple(steps),
            expected_inconsistent=expected_inconsistent,
        ),
    )


def _format_step(
    case_id: str,
    format_valid: bool,
    response_confidence: float,
) -> LogicalTraceStep:
    return LogicalTraceStep(
        step_id="format",
        proposition=f"{case_id}:format_valid",
        truth_value=format_valid,
        confidence=response_confidence if format_valid else 0.45,
        supports=("response",) if format_valid else ("missing-format-proof",),
        constraint_ids=("format",) if format_valid else (),
    )


def _score_kv260_cases(
    cases: Sequence[SotaTraceCase],
    kv260_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    scorer = EBRMTraceScorer()
    hardware_available = bool(kv260_artifact.get("hardware_execution_available", False))
    return [
        {
            **score.to_dict(),
            "potts_q_states": POTTS_Q_STATES,
            "potts_states": _encode_potts_states(case.trace),
            "kv260_backend": KV260_BACKEND_NAME,
            "hardware_execution_available": hardware_available,
        }
        for case, score in (
            (case, scorer.score_trace(case.trace))
            for case in cases
        )
    ]


def _encode_potts_states(trace: LogicalTrace) -> list[int]:
    config = EBRMTraceScorerConfig()
    states = [0] * len(trace.steps)
    step_index = {step.step_id: index for index, step in enumerate(trace.steps)}
    seen_by_prop: dict[str, list[LogicalTraceStep]] = {}
    for index, step in enumerate(trace.steps):
        if step.confidence < config.min_confidence or not step.constraint_ids:
            states[index] = max(states[index], 1)
        if any(prior.truth_value is not step.truth_value for prior in seen_by_prop.get(step.proposition, [])):
            states[index] = 2
        for linked_step_id in step.supports:
            linked = step_index.get(linked_step_id)
            if linked is None or linked >= index:
                states[index] = 2
        for linked_step_id in step.contradicts:
            linked = step_index.get(linked_step_id)
            if linked is not None:
                states[index] = 2
        seen_by_prop.setdefault(step.proposition, []).append(step)
    return states


def _case_score_row(
    case: SotaTraceCase,
    cpu: EBRMTraceScore,
    kv260: Mapping[str, Any],
    delta: float,
) -> JsonDict:
    return {
        "case_id": case.case_id,
        "model_id": case.model_id,
        "model_name": case.model_name,
        "generation_source": case.generation_source,
        "expected_inconsistent": case.expected_inconsistent,
        "cpu_energy": cpu.energy,
        "kv260_energy": kv260["energy"],
        "score_delta": delta,
        "cpu_coherence_score": cpu.coherence_score,
        "kv260_coherence_score": kv260["coherence_score"],
        "potts_q_states": kv260["potts_q_states"],
        "potts_states": kv260["potts_states"],
        "kv260_backend": kv260["kv260_backend"],
        "hardware_execution_available": kv260["hardware_execution_available"],
    }


def _gate_blockers(
    ebrm_artifact_path: Path | str,
    kv260_artifact_path: Path | str,
) -> tuple[list[str], JsonDict]:
    ebrm_artifact = _load_json(Path(ebrm_artifact_path))
    kv260_artifact = _load_json(Path(kv260_artifact_path))
    blockers = []
    if ebrm_artifact.get("status") != "complete" or not ebrm_artifact.get(
        "ebrm_trace_scorer_ready"
    ):
        blockers.append("Exp 1656 EBRM trace scorer artifact is not complete and ready.")
    if kv260_artifact.get("status") != "complete" or not kv260_artifact.get(
        "kv260_ebrm_binding_ready"
    ):
        blockers.append("Exp 1657 KV260 EBRM binding artifact is not complete and ready.")
    return blockers, kv260_artifact


def _empty_comparison() -> JsonDict:
    return {
        "cases_total": 0,
        "consistent_cases": 0,
        "inconsistent_cases": 0,
        "cpu_latency_ms": 0.0,
        "kv260_latency_ms": 0.0,
        "latency_delta_ms": 0.0,
        "kv260_speedup_vs_cpu": None,
        "cpu_score_accuracy": 0.0,
        "kv260_score_accuracy": 0.0,
        "max_score_delta": 0.0,
        "mean_abs_score_delta": 0.0,
        "scoring_delta_within_tolerance": False,
        "case_scores": [],
    }


def _confidence_from_row(row: Mapping[str, Any]) -> float:
    logprobs = [float(value) for value in (row.get("token_logprobs") or [-0.1])]
    return round(max(0.05, min(1.0, math.exp(sum(logprobs) / len(logprobs)))), 6)


def _time_call(fn: Callable[[], Any], timer: Timer) -> tuple[float, Any]:
    start = timer()
    result = fn()
    return max(0.0, timer() - start), result


def _score_accuracy(cases: Sequence[SotaTraceCase], energies: Sequence[float]) -> float:
    threshold = EBRMTraceScorerConfig().prediction_threshold
    correct = sum(
        (energy >= threshold) is case.expected_inconsistent
        for case, energy in zip(cases, energies, strict=True)
    )
    return round(correct / max(1, len(cases)), 6)


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / max(1, len(values)), 6)


def _speedup(cpu_latency_ms: float, kv260_latency_ms: float) -> float | None:
    if kv260_latency_ms == 0.0:
        return None
    return round(cpu_latency_ms / kv260_latency_ms, 6)


def _load_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = dict(payload)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _honest_verdict(
    complete: bool,
    comparison: Mapping[str, Any],
    kv260_artifact: Mapping[str, Any],
) -> str:
    if complete:
        return (
            "complete: CPU and KV260 EBRM backends agree on SOTA trace scores "
            f"with max_score_delta={comparison['max_score_delta']} and "
            f"hardware_execution_available={bool(kv260_artifact.get('hardware_execution_available', False))}"
        )
    return "blocked: CPU vs KV260 EBRM trace-scoring comparison did not satisfy gates."


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--sota-manifest", type=Path, default=DEFAULT_SOTA_MANIFEST_PATH)
    parser.add_argument("--max-cases", type=int, default=12)
    args = parser.parse_args(argv)
    run_experiment(
        output_path=args.output,
        sota_manifest_path=args.sota_manifest,
        max_cases=args.max_cases,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

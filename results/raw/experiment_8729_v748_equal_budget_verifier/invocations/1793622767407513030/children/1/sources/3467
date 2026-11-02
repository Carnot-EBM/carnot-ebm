"""Deterministic replay benchmark for sampler-backed repair reranking.

Spec: REQ-SAMPLE-008,
SCENARIO-SAMPLE-015, SCENARIO-SAMPLE-016, SCENARIO-SAMPLE-017
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carnot.inference.composite_scorer import CompositeEnergyConfig, CompositeEnergyScorer
from carnot.samplers.backend import CpuBackend, SamplerBackend
from carnot.samplers.fpga_ising import FPGAIsingSampler, SoftwareFPGAOverlay

EXPERIMENT_ID = 243
RUN_DATE = "20260413"
DEFAULT_OUTPUT = Path("results/experiment_243_results.json")
SOURCE_ARTIFACTS = (
    Path("results/experiment_235_results.json"),
    Path("results/experiment_238_results.json"),
    Path("results/experiment_242_results.json"),
)
SPEC_REFS = [
    "REQ-SAMPLE-008",
    "SCENARIO-SAMPLE-015",
    "SCENARIO-SAMPLE-016",
    "SCENARIO-SAMPLE-017",
]
DEFAULT_SAMPLE_COUNT = 6
DEFAULT_ANNEAL_STEPS = 24
DEFAULT_BETA = 6.0
DEFAULT_BITFILE_ENV = "CARNOT_KV260_BITFILE"

_SEMANTIC_SCORER = CompositeEnergyScorer(
    CompositeEnergyConfig(
        logprob_weight=1.0,
        structural_weight=0.5,
        test_failure_penalty=1.0,
    )
)
_CODE_SCORER = CompositeEnergyScorer(
    CompositeEnergyConfig(
        logprob_weight=1.0,
        structural_weight=1.0,
        test_failure_penalty=1.0,
    )
)


@dataclass(frozen=True)
class ReplayCandidate:
    """One saved candidate from a semantic or code repair history."""

    source_experiment: int
    benchmark: str
    domain: str
    model_name: str
    case_id: str
    sample_position: int
    iteration: int
    candidate_id: str
    text: str
    accepted: bool
    verified: bool
    actual_success: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_experiment": self.source_experiment,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "model_name": self.model_name,
            "case_id": self.case_id,
            "sample_position": self.sample_position,
            "iteration": self.iteration,
            "candidate_id": self.candidate_id,
            "accepted": self.accepted,
            "verified": self.verified,
            "actual_success": self.actual_success,
            "text": self.text,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReplayCandidateSet:
    """Saved repair trajectory turned into a deterministic reranking case."""

    source_experiment: int
    benchmark: str
    domain: str
    model_name: str
    case_id: str
    sample_position: int
    initial_candidate_id: str
    original_selected_candidate_id: str
    initial_success: bool
    original_selected_success: bool
    original_selected_accepted: bool
    original_pipeline_latency_seconds: float
    candidates: tuple[ReplayCandidate, ...]

    @property
    def rerankable(self) -> bool:
        return len(self.candidates) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_experiment": self.source_experiment,
            "benchmark": self.benchmark,
            "domain": self.domain,
            "model_name": self.model_name,
            "case_id": self.case_id,
            "sample_position": self.sample_position,
            "initial_candidate_id": self.initial_candidate_id,
            "original_selected_candidate_id": self.original_selected_candidate_id,
            "initial_success": self.initial_success,
            "original_selected_success": self.original_selected_success,
            "original_selected_accepted": self.original_selected_accepted,
            "original_pipeline_latency_seconds": self.original_pipeline_latency_seconds,
            "rerankable": self.rerankable,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_final_number(text: str) -> int | None:
    """Reuse the Exp 218/235 answer-extraction contract for GSM8K replay."""
    numeric_token = r"-?(?:\d[\d,]*)"

    match = re.search(rf"####\s*({numeric_token})", text)
    if match:
        return int(match.group(1).replace(",", ""))

    match = re.search(rf"[Aa]nswer[:\s]+({numeric_token})", text)
    if match:
        return int(match.group(1).replace(",", ""))

    numbers = re.findall(numeric_token, text)
    if numbers:
        return int(numbers[-1].replace(",", ""))
    return None


def _sample_positions(payload: dict[str, Any]) -> dict[str, int]:
    cohort = _as_dict(payload.get("cohort"))
    positions: dict[str, int] = {}
    cases = cohort.get("cases")
    if isinstance(cases, list):
        for case in cases:
            data = _as_dict(case)
            case_id = str(data.get("case_id") or "")
            if case_id:
                positions[case_id] = int(data.get("sample_position") or 0)
    if positions:
        return positions
    case_ids = cohort.get("case_ids")
    if isinstance(case_ids, list):
        return {str(case_id): index + 1 for index, case_id in enumerate(case_ids)}
    return {}


def _semantic_candidate_sets(payload: dict[str, Any]) -> list[ReplayCandidateSet]:
    cohort = _as_dict(payload.get("cohort"))
    cohort_cases = {
        str(_as_dict(case).get("case_id") or ""): _as_dict(case)
        for case in cohort.get("cases", [])
        if isinstance(case, dict)
    }
    results: list[ReplayCandidateSet] = []
    for run in payload.get("paired_runs", []):
        run_data = _as_dict(run)
        if run_data.get("mode") != "verify_repair":
            continue
        model_name = str(run_data.get("model_name") or "")
        for case in run_data.get("cases", []):
            case_data = _as_dict(case)
            case_id = str(case_data.get("case_id") or "")
            history = case_data.get("history")
            cohort_case = cohort_cases.get(case_id, {})
            if not case_id or not isinstance(history, list) or not cohort_case:
                continue
            sample_position = int(cohort_case.get("sample_position") or 0)
            ground_truth = int(cohort_case.get("ground_truth") or 0)
            domain = str(cohort_case.get("task_slice") or "live_gsm8k_semantic_failure")

            candidates: list[ReplayCandidate] = []
            for index, entry in enumerate(history):
                item = _as_dict(entry)
                iteration = int(item.get("iteration", index))
                response = str(item.get("response") or "")
                verification = _as_dict(item.get("verification"))
                semantic_v2 = _as_dict(verification.get("semantic_verifier_v2"))
                actual_success = extract_final_number(response) == ground_truth
                candidate_id = f"{case_id}:{iteration}"
                candidates.append(
                    ReplayCandidate(
                        source_experiment=235,
                        benchmark="gsm8k_semantic",
                        domain=domain,
                        model_name=model_name,
                        case_id=case_id,
                        sample_position=sample_position,
                        iteration=iteration,
                        candidate_id=candidate_id,
                        text=response,
                        accepted=bool(verification.get("verified")),
                        verified=bool(verification.get("verified")),
                        actual_success=actual_success,
                        metadata={
                            "n_violations": int(verification.get("n_violations") or 0),
                            "semantic_verifier_v2": semantic_v2,
                        },
                    )
                )
            if not candidates:
                continue
            results.append(
                ReplayCandidateSet(
                    source_experiment=235,
                    benchmark="gsm8k_semantic",
                    domain=domain,
                    model_name=model_name,
                    case_id=case_id,
                    sample_position=sample_position,
                    initial_candidate_id=candidates[0].candidate_id,
                    original_selected_candidate_id=candidates[-1].candidate_id,
                    initial_success=candidates[0].actual_success,
                    original_selected_success=candidates[-1].actual_success,
                    original_selected_accepted=candidates[-1].accepted,
                    original_pipeline_latency_seconds=float(
                        case_data.get("latency_seconds") or 0.0
                    ),
                    candidates=tuple(candidates),
                )
            )
    return results


def _code_candidate_sets(payload: dict[str, Any]) -> list[ReplayCandidateSet]:
    positions = _sample_positions(payload)
    model_runs = payload.get("model_runs")
    if not isinstance(model_runs, dict):
        return []

    results: list[ReplayCandidateSet] = []
    for raw_run in model_runs.values():
        run = _as_dict(raw_run)
        model_name = str(run.get("model_name") or "")
        per_problem_results = run.get("per_problem_results")
        if not isinstance(per_problem_results, list):
            continue
        for item in per_problem_results:
            case = _as_dict(item)
            case_id = str(case.get("case_id") or "")
            history = case.get("history")
            if not case_id or not isinstance(history, list):
                continue
            sample_position = positions.get(case_id, 0)
            candidates: list[ReplayCandidate] = []
            accumulated_latency = 0.0
            for index, entry in enumerate(history):
                history_entry = _as_dict(entry)
                iteration = int(history_entry.get("iteration", index))
                evaluation = _as_dict(history_entry.get("evaluation"))
                official_tests = _as_dict(evaluation.get("official_tests"))
                pbt = _as_dict(evaluation.get("pbt"))
                explicit_specs = _as_dict(evaluation.get("explicit_specs"))
                stage_acceptance = _as_dict(evaluation.get("stage_acceptance"))
                accumulated_latency += float(evaluation.get("latency_seconds") or 0.0)
                candidate_id = f"{case_id}:{iteration}"
                candidates.append(
                    ReplayCandidate(
                        source_experiment=238,
                        benchmark="humaneval_dual_model_spec",
                        domain="code_spec_properties",
                        model_name=model_name,
                        case_id=case_id,
                        sample_position=sample_position,
                        iteration=iteration,
                        candidate_id=candidate_id,
                        text=str(
                            history_entry.get("candidate_code") or history_entry.get("body") or ""
                        ),
                        accepted=bool(stage_acceptance.get("spec_aware_verify_only")),
                        verified=bool(stage_acceptance.get("spec_aware_verify_only")),
                        actual_success=bool(official_tests.get("passed")),
                        metadata={
                            "official_tests": official_tests,
                            "pbt": pbt,
                            "explicit_specs": explicit_specs,
                            "stage_acceptance": stage_acceptance,
                        },
                    )
                )
            if not candidates:
                continue
            results.append(
                ReplayCandidateSet(
                    source_experiment=238,
                    benchmark="humaneval_dual_model_spec",
                    domain="code_spec_properties",
                    model_name=model_name,
                    case_id=case_id,
                    sample_position=sample_position,
                    initial_candidate_id=candidates[0].candidate_id,
                    original_selected_candidate_id=candidates[-1].candidate_id,
                    initial_success=candidates[0].actual_success,
                    original_selected_success=candidates[-1].actual_success,
                    original_selected_accepted=candidates[-1].accepted,
                    original_pipeline_latency_seconds=accumulated_latency,
                    candidates=tuple(candidates),
                )
            )
    return results


def build_candidate_set_benchmark(
    exp235: dict[str, Any],
    exp238: dict[str, Any],
) -> list[ReplayCandidateSet]:
    """Normalize Exp 235 and Exp 238 histories into deterministic candidate sets."""
    cases = [*_semantic_candidate_sets(exp235), *_code_candidate_sets(exp238)]
    return sorted(
        cases,
        key=lambda case: (
            case.source_experiment,
            case.sample_position,
            case.model_name,
            case.case_id,
        ),
    )


def score_candidate(candidate: ReplayCandidate) -> float:
    """Score one saved candidate through the existing composite scorer path."""
    if candidate.benchmark == "gsm8k_semantic":
        semantic_v2 = _as_dict(candidate.metadata.get("semantic_verifier_v2"))
        error_probability = float(semantic_v2.get("semantic_error_probability") or 1.0)
        verdict = str(semantic_v2.get("verdict") or "unavailable")
        verdict_penalty = {
            "supported": 0,
            "abstain": 1,
            "violated": 2,
            "unavailable": 1,
        }.get(verdict, 1)
        n_failures = int(candidate.metadata.get("n_violations") or 0) + verdict_penalty
        return _SEMANTIC_SCORER.score_candidate(
            candidate.text,
            mean_logprob=-error_probability,
            n_failures=n_failures,
        )

    official_tests = _as_dict(candidate.metadata.get("official_tests"))
    pbt = _as_dict(candidate.metadata.get("pbt"))
    explicit_specs = _as_dict(candidate.metadata.get("explicit_specs"))
    official_failures = 0 if bool(official_tests.get("passed")) else 1
    pbt_failures = int(pbt.get("n_failures") or 0)
    spec_failures = int(explicit_specs.get("n_violations") or 0)
    n_failures = official_failures + pbt_failures + spec_failures
    confidence_penalty = official_failures + (0.25 * pbt_failures) + (0.1 * spec_failures)
    return _CODE_SCORER.score_candidate(
        candidate.text,
        mean_logprob=-confidence_penalty,
        n_failures=n_failures,
    )


def _encode_candidate_selection_problem(scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    if not scores:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    penalty = max(max(scores), 1.0) + 1.0
    biases = np.asarray([penalty - score for score in scores], dtype=np.float32)
    couplings = np.full((len(scores), len(scores)), -penalty, dtype=np.float32)
    np.fill_diagonal(couplings, 0.0)
    return biases, couplings


def _decode_samples(
    samples: np.ndarray,
    scores: list[float],
) -> tuple[int, dict[int, int], bool, str]:
    votes: dict[int, int] = {}
    projected = False
    for sample in np.asarray(samples, dtype=bool):
        active = np.flatnonzero(sample)
        if active.size == 0:
            continue
        if active.size > 1:
            projected = True
        choice = min(int(index) for index in active)
        choice = min((int(index) for index in active), key=lambda index: (scores[index], index))
        votes[choice] = votes.get(choice, 0) + 1
    if votes:
        best_index = min(votes, key=lambda index: (-votes[index], scores[index], index))
        return best_index, votes, projected, "sampler_samples"
    return int(np.argmin(np.asarray(scores, dtype=np.float32))), votes, True, "score_fallback"


def rerank_candidate_set(
    case: ReplayCandidateSet,
    *,
    backend: SamplerBackend,
    n_samples: int = DEFAULT_SAMPLE_COUNT,
    n_steps: int = DEFAULT_ANNEAL_STEPS,
    beta: float = DEFAULT_BETA,
) -> dict[str, Any]:
    """Run sampler-backed top-1 selection over one recorded candidate set."""
    scores = [score_candidate(candidate) for candidate in case.candidates]
    biases, couplings = _encode_candidate_selection_problem(scores)
    started = time.perf_counter()
    samples = backend.minimize_energy(
        biases=biases,
        couplings=couplings,
        n_samples=max(1, n_samples),
        n_steps=max(1, n_steps),
        beta=beta,
    )
    selection_latency_seconds = time.perf_counter() - started
    selected_index, votes, fallback_used, selection_source = _decode_samples(samples, scores)
    selected = case.candidates[selected_index]
    return {
        "benchmark": case.benchmark,
        "domain": case.domain,
        "model_name": case.model_name,
        "case_id": case.case_id,
        "sample_position": case.sample_position,
        "candidate_count": len(case.candidates),
        "rerankable": case.rerankable,
        "initial_candidate_id": case.initial_candidate_id,
        "original_selected_candidate_id": case.original_selected_candidate_id,
        "selected_candidate_id": selected.candidate_id,
        "selected_iteration": selected.iteration,
        "score": round(float(scores[selected_index]), 6),
        "accepted": selected.accepted,
        "verified": selected.verified,
        "actual_success": selected.actual_success,
        "baseline_actual_success": case.original_selected_success,
        "baseline_accepted": case.original_selected_accepted,
        "initial_success": case.initial_success,
        "repair_improved": (not case.original_selected_success) and selected.actual_success,
        "selection_latency_seconds": round(selection_latency_seconds, 6),
        "selection_source": selection_source,
        "fallback_used": fallback_used,
        "sample_vote_counts": {
            case.candidates[index].candidate_id: count for index, count in sorted(votes.items())
        },
        "sampler_backend": backend.backend_name,
    }


def _empty_bucket() -> dict[str, Any]:
    return {
        "n_cases": 0,
        "n_rerankable_cases": 0,
        "baseline_successes": 0,
        "selected_successes": 0,
        "baseline_accepted_cases": 0,
        "selected_accepted_cases": 0,
        "baseline_accepted_successes": 0,
        "selected_accepted_successes": 0,
        "repair_opportunities": 0,
        "baseline_repairs": 0,
        "selected_repairs": 0,
        "saved_pipeline_latency_seconds": 0.0,
        "selection_latency_seconds": 0.0,
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    n_cases = int(bucket["n_cases"])
    repair_opportunities = int(bucket["repair_opportunities"])
    baseline_accepted_cases = int(bucket["baseline_accepted_cases"])
    selected_accepted_cases = int(bucket["selected_accepted_cases"])

    baseline_top1_quality_rate = bucket["baseline_successes"] / n_cases if n_cases else 0.0
    top1_quality_rate = bucket["selected_successes"] / n_cases if n_cases else 0.0
    baseline_verifier_precision = (
        bucket["baseline_accepted_successes"] / baseline_accepted_cases
        if baseline_accepted_cases
        else 0.0
    )
    verifier_precision = (
        bucket["selected_accepted_successes"] / selected_accepted_cases
        if selected_accepted_cases
        else 0.0
    )
    baseline_repair_yield = (
        bucket["baseline_repairs"] / repair_opportunities if repair_opportunities else 0.0
    )
    repair_yield = (
        bucket["selected_repairs"] / repair_opportunities if repair_opportunities else 0.0
    )
    mean_saved_pipeline_latency_seconds = (
        bucket["saved_pipeline_latency_seconds"] / n_cases if n_cases else 0.0
    )
    mean_selection_latency_seconds = (
        bucket["selection_latency_seconds"] / n_cases if n_cases else 0.0
    )

    return {
        "n_cases": n_cases,
        "n_rerankable_cases": int(bucket["n_rerankable_cases"]),
        "baseline_top1_quality_rate": baseline_top1_quality_rate,
        "top1_quality_rate": top1_quality_rate,
        "top1_quality_delta": top1_quality_rate - baseline_top1_quality_rate,
        "baseline_verifier_precision": baseline_verifier_precision,
        "verifier_precision": verifier_precision,
        "verifier_precision_delta": verifier_precision - baseline_verifier_precision,
        "baseline_repair_yield": baseline_repair_yield,
        "repair_yield": repair_yield,
        "repair_yield_delta": repair_yield - baseline_repair_yield,
        "mean_saved_pipeline_latency_seconds": mean_saved_pipeline_latency_seconds,
        "mean_selection_latency_seconds": mean_selection_latency_seconds,
        "mean_selection_minus_saved_pipeline_latency_seconds": (
            mean_selection_latency_seconds - mean_saved_pipeline_latency_seconds
        ),
        "baseline_accept_rate": (baseline_accepted_cases / n_cases if n_cases else 0.0),
        "accept_rate": selected_accepted_cases / n_cases if n_cases else 0.0,
    }


def summarize_backend_results(
    *,
    candidate_sets: list[ReplayCandidateSet],
    reranked_cases: list[dict[str, Any]],
    sampler_backend: str,
    execution_path: str,
    run_status: str,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one backend's reranked top-1 outcomes."""
    overall = _empty_bucket()
    by_benchmark: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}

    indexed_results = {
        (
            str(result.get("benchmark") or ""),
            str(result.get("model_name") or ""),
            str(result.get("case_id") or ""),
        ): result
        for result in reranked_cases
    }

    for case in candidate_sets:
        result = indexed_results.get((case.benchmark, case.model_name, case.case_id))
        if result is None:
            continue
        for bucket in (
            overall,
            by_benchmark.setdefault(case.benchmark, _empty_bucket()),
            by_model.setdefault(case.model_name, _empty_bucket()),
        ):
            bucket["n_cases"] += 1
            bucket["n_rerankable_cases"] += int(case.rerankable)
            bucket["baseline_successes"] += int(case.original_selected_success)
            bucket["selected_successes"] += int(bool(result["actual_success"]))
            bucket["baseline_accepted_cases"] += int(case.original_selected_accepted)
            bucket["selected_accepted_cases"] += int(bool(result["accepted"]))
            bucket["baseline_accepted_successes"] += int(
                case.original_selected_accepted and case.original_selected_success
            )
            bucket["selected_accepted_successes"] += int(
                bool(result["accepted"]) and bool(result["actual_success"])
            )
            if not case.initial_success:
                bucket["repair_opportunities"] += 1
                bucket["baseline_repairs"] += int(case.original_selected_success)
                bucket["selected_repairs"] += int(bool(result["actual_success"]))
            bucket["saved_pipeline_latency_seconds"] += case.original_pipeline_latency_seconds
            bucket["selection_latency_seconds"] += float(result["selection_latency_seconds"])

    return {
        "sampler_backend": sampler_backend,
        "execution_path": execution_path,
        "run_status": run_status,
        "blockers": [dict(blocker) for blocker in blockers],
        "overall": _finalize_bucket(overall),
        "by_benchmark": {
            name: _finalize_bucket(bucket) for name, bucket in sorted(by_benchmark.items())
        },
        "by_model": {name: _finalize_bucket(bucket) for name, bucket in sorted(by_model.items())},
    }


def _candidate_set_summary(candidate_sets: list[ReplayCandidateSet]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, int]] = {}
    by_model: dict[str, dict[str, int]] = {}
    for case in candidate_sets:
        benchmark_bucket = by_benchmark.setdefault(
            case.benchmark,
            {"n_cases": 0, "n_candidates": 0},
        )
        benchmark_bucket["n_cases"] += 1
        benchmark_bucket["n_candidates"] += len(case.candidates)
        model_bucket = by_model.setdefault(case.model_name, {"n_cases": 0, "n_candidates": 0})
        model_bucket["n_cases"] += 1
        model_bucket["n_candidates"] += len(case.candidates)
    return {
        "n_cases": len(candidate_sets),
        "n_rerankable_cases": sum(1 for case in candidate_sets if case.rerankable),
        "n_candidates": sum(len(case.candidates) for case in candidate_sets),
        "by_benchmark": by_benchmark,
        "by_model": by_model,
    }


def build_blocker(
    *,
    code: str,
    message: str,
    setup_step: str,
    bitfile_path: Path | None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "setup_step": setup_step,
        "overlay_path": str(bitfile_path) if bitfile_path is not None else None,
    }


def resolve_optional_kv260_backend(
    *,
    repo_root: Path | None = None,
    bitfile_path: str | Path | None = None,
    overlay_factory: Any | None = None,
) -> dict[str, Any]:
    """Resolve the optional KV260 sampler backend or return an honest blocker."""
    resolved_repo = (repo_root or get_repo_root()).resolve()
    resolved_bitfile = Path(bitfile_path).resolve() if bitfile_path is not None else None
    if overlay_factory is not None:
        transport = overlay_factory(str(resolved_bitfile) if resolved_bitfile is not None else None)
        backend = FPGAIsingSampler(
            mode="hardware",
            allow_cpu_fallback=False,
            overlay_factory=lambda _bitfile: transport,
        )
        execution_path = (
            "software_model" if isinstance(transport, SoftwareFPGAOverlay) else "hardware"
        )
        return {
            "sampler_backend": backend.backend_name,
            "execution_path": execution_path,
            "run_status": "complete",
            "backend": backend,
            "blockers": [],
            "notes": [
                "KV260 reranking backend resolved through the active Exp 242 transport contract."
            ],
        }

    configured_bitfile = resolved_bitfile
    if configured_bitfile is None:
        env_value = os.environ.get(DEFAULT_BITFILE_ENV)
        if env_value:
            configured_bitfile = Path(env_value).resolve()
    if configured_bitfile is not None:
        if configured_bitfile.exists():
            try:
                backend = FPGAIsingSampler(
                    mode="hardware",
                    bitfile_path=str(configured_bitfile),
                    allow_cpu_fallback=False,
                )
                return {
                    "sampler_backend": backend.backend_name,
                    "execution_path": "hardware",
                    "run_status": "complete",
                    "backend": backend,
                    "blockers": [],
                    "notes": ["KV260 hardware backend resolved from the configured bitfile."],
                }
            except Exception as error:
                blockers = [
                    build_blocker(
                        code="overlay_unavailable",
                        message=f"Configured KV260 bitfile could not be activated: {error}",
                        setup_step=(
                            "Load the KV260 overlay through PYNQ and rerun Exp 243 to enable "
                            "hardware-backed reranking."
                        ),
                        bitfile_path=configured_bitfile,
                    )
                ]
                return {
                    "sampler_backend": "kv260",
                    "execution_path": "blocked",
                    "run_status": "blocked",
                    "backend": None,
                    "blockers": blockers,
                    "notes": ["KV260 backend remained blocked while resolving the live overlay."],
                }
        blockers = [
            build_blocker(
                code="bitfile_not_found",
                message=f"Configured KV260 bitfile was not found: {configured_bitfile}",
                setup_step=(
                    "Set CARNOT_KV260_BITFILE to a valid KV260 bitfile path "
                    "before rerunning Exp 243."
                ),
                bitfile_path=configured_bitfile,
            )
        ]
        return {
            "sampler_backend": "kv260",
            "execution_path": "blocked",
            "run_status": "blocked",
            "backend": None,
            "blockers": blockers,
            "notes": [
                "KV260 backend remained blocked because the configured bitfile path was missing."
            ],
        }

    exp242_path = resolved_repo / SOURCE_ARTIFACTS[2]
    if exp242_path.exists():
        exp242 = load_json(exp242_path)
        metadata = _as_dict(exp242.get("metadata"))
        execution_path = str(metadata.get("execution_path") or "blocked")
        raw_blockers = exp242.get("blockers")
        blockers = (
            [dict(blocker) for blocker in raw_blockers if isinstance(blocker, dict)]
            if isinstance(raw_blockers, list)
            else []
        )
        if execution_path == "software_model":
            backend = FPGAIsingSampler(mode="software", seed=EXPERIMENT_ID)
            return {
                "sampler_backend": backend.backend_name,
                "execution_path": "software_model",
                "run_status": "complete",
                "backend": backend,
                "blockers": [],
                "notes": list(metadata.get("notes") or []),
            }
        if execution_path == "hardware":
            return {
                "sampler_backend": "kv260",
                "execution_path": "blocked",
                "run_status": "blocked",
                "backend": None,
                "blockers": [dict(blocker) for blocker in blockers],
                "notes": [
                    "Exp 242 recorded a hardware path, but Exp 243 could not "
                    "re-bind a live KV260 transport in this environment."
                ],
            }
        return {
            "sampler_backend": "kv260",
            "execution_path": "blocked",
            "run_status": "blocked",
            "backend": None,
            "blockers": blockers,
            "notes": list(metadata.get("notes") or []),
        }

    blockers = [
        build_blocker(
            code="missing_exp242_reference",
            message="No Exp 242 artifact or KV260 bitfile configuration was available for Exp 243.",
            setup_step=(
                "Create results/experiment_242_results.json or configure CARNOT_KV260_BITFILE "
                "before rerunning Exp 243 to enable the optional KV260 path."
            ),
            bitfile_path=None,
        )
    ]
    return {
        "sampler_backend": "kv260",
        "execution_path": "blocked",
        "run_status": "blocked",
        "backend": None,
        "blockers": blockers,
        "notes": ["KV260 backend remained blocked because Exp 242 evidence was unavailable."],
    }


def _backend_report(
    *,
    candidate_sets: list[ReplayCandidateSet],
    backend: SamplerBackend | None,
    sampler_backend: str,
    execution_path: str,
    run_status: str,
    blockers: list[dict[str, Any]],
) -> dict[str, Any]:
    reranked_cases: list[dict[str, Any]] = []
    if backend is not None:
        reranked_cases = [rerank_candidate_set(case, backend=backend) for case in candidate_sets]
    summary = summarize_backend_results(
        candidate_sets=candidate_sets,
        reranked_cases=reranked_cases,
        sampler_backend=sampler_backend,
        execution_path=execution_path,
        run_status=run_status,
        blockers=blockers,
    )
    summary["cases"] = reranked_cases
    return summary


def build_experiment_payload(
    *,
    exp235: dict[str, Any],
    exp238: dict[str, Any],
    cpu_report: dict[str, Any],
    kv260_report: dict[str, Any],
    output_path: Path,
    started_at: str,
    finished_at: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    candidate_sets = build_candidate_set_benchmark(exp235, exp238)
    return {
        "experiment": EXPERIMENT_ID,
        "benchmark": "repair_candidate_rerank_replay",
        "title": "Sampler-backed repair reranking replay benchmark",
        "run_date": RUN_DATE,
        "schema": {"artifact": "carnot.repair_rerank_replay.v1"},
        "metadata": {
            "started_at": started_at,
            "finished_at": finished_at,
            "runtime_seconds": round(runtime_seconds, 6),
            "output_path": str(output_path),
            "source_artifacts": [str(path) for path in SOURCE_ARTIFACTS],
            "module": "python/carnot/inference/repair_reranker.py",
            "script": "scripts/experiment_243_energy_reranker.py",
            "spec_requirements": list(SPEC_REFS),
            "rerank_encoding": {
                "selection_problem": "one_hot_candidate_choice",
                "n_samples": DEFAULT_SAMPLE_COUNT,
                "anneal_steps": DEFAULT_ANNEAL_STEPS,
                "beta": DEFAULT_BETA,
            },
            "notes": [
                "Replay uses saved Exp 235 semantic histories and Exp 238 "
                "code histories only; no new model generations were produced.",
                "The existing CompositeEnergyScorer is driven by "
                "verifier-derived confidence proxies because the saved "
                "artifacts do not preserve token logprobs.",
            ],
        },
        "candidate_set_benchmark": {
            "summary": _candidate_set_summary(candidate_sets),
            "cases": [case.to_dict() for case in candidate_sets],
        },
        "backends": {
            "cpu": cpu_report,
            "kv260": kv260_report,
        },
        "run_status": "complete",
    }


def run_experiment(
    repo_root: Path | None = None,
    result_path: Path | None = None,
) -> dict[str, Any]:
    """Build and write the Exp 243 replay artifact for the current checkout."""
    resolved_repo = (repo_root or get_repo_root()).resolve()
    resolved_result_path = resolved_repo / (result_path or DEFAULT_OUTPUT)
    started_at = utc_now()
    started = time.perf_counter()

    exp235 = load_json(resolved_repo / SOURCE_ARTIFACTS[0])
    exp238 = load_json(resolved_repo / SOURCE_ARTIFACTS[1])
    candidate_sets = build_candidate_set_benchmark(exp235, exp238)

    cpu_backend = CpuBackend(seed=EXPERIMENT_ID)
    cpu_report = _backend_report(
        candidate_sets=candidate_sets,
        backend=cpu_backend,
        sampler_backend=cpu_backend.backend_name,
        execution_path="cpu",
        run_status="complete",
        blockers=[],
    )

    kv260_backend = resolve_optional_kv260_backend(repo_root=resolved_repo)
    kv260_report = _backend_report(
        candidate_sets=candidate_sets,
        backend=kv260_backend.get("backend"),
        sampler_backend=str(kv260_backend.get("sampler_backend") or "kv260"),
        execution_path=str(kv260_backend.get("execution_path") or "blocked"),
        run_status=str(kv260_backend.get("run_status") or "blocked"),
        blockers=[dict(blocker) for blocker in kv260_backend.get("blockers", [])],
    )
    if kv260_backend.get("notes"):
        kv260_report["notes"] = list(kv260_backend["notes"])

    finished_at = utc_now()
    payload = build_experiment_payload(
        exp235=exp235,
        exp238=exp238,
        cpu_report=cpu_report,
        kv260_report=kv260_report,
        output_path=resolved_result_path,
        started_at=started_at,
        finished_at=finished_at,
        runtime_seconds=time.perf_counter() - started,
    )
    payload["metadata"]["output_path"] = _relative_path(resolved_result_path, resolved_repo)
    write_json(resolved_result_path, payload)
    return payload


__all__ = [
    "DEFAULT_OUTPUT",
    "EXPERIMENT_ID",
    "ReplayCandidate",
    "ReplayCandidateSet",
    "RUN_DATE",
    "SOURCE_ARTIFACTS",
    "build_candidate_set_benchmark",
    "build_experiment_payload",
    "extract_final_number",
    "get_repo_root",
    "load_json",
    "resolve_optional_kv260_backend",
    "rerank_candidate_set",
    "run_experiment",
    "score_candidate",
    "summarize_backend_results",
    "write_json",
]

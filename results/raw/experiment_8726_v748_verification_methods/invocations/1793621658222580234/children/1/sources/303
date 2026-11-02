"""Exp 2934 AquaForte/BEAVER ConstraintBench reformulation pipeline.

Spec: REQ-VERIFY-2934, SCENARIO-VERIFY-2934.

The pipeline is intentionally bounded and local. Exp 2926 already captured
live local GGUF responses over ConstraintBench-style tasks; this module treats
those responses as LLM proposals, reformulates them into each task's solver
schema, and lets the exact verifier decide whether the candidate is acceptable.
Rejected candidates receive a cheap deterministic retry only because these
mini tasks have exhaustive solvers available. The retry is recorded separately
so the artifact does not confuse proposal quality with exact-solver repair.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval import constraintbench_constrained_output_rerun as exp2926
from carnot.eval import constraintbench_mini_direct_optimization as base
from carnot.inference.sota_models import cached_sota_pair

JsonDict = dict[str, Any]
RUN_DATE = "20260523"
RANDOM_SEED = 2934
REPO_ROOT = Path(__file__).resolve().parents[3]
EXP2926_FILENAME = "experiment_2926_constraintbench_constrained_output_rerun_v2.json"
OUTPUT_FILENAME = "experiment_2934_aquaforte_beaver_reformulation_pipeline_v1.json"
INFERENCE_SUBSTRATE = "live_llm_inference_plus_exact_verifier"
DEFAULT_SELECTED_COUNT = 18
MANDATED_MODEL_IDS = set(base.MANDATED_MODEL_IDS)

REQUIRED_ARTIFACT_FIELDS: frozenset[str] = frozenset(
    {
        "honest_verdict",
        "reformulation_pipeline_ready",
        "random_seed",
        "reproducibility_checksum",
        "model_specs",
        "models_used",
        "selected_task_ids",
        "proposal_count",
        "verifier_acceptance_rate",
        "feasibility_delta_vs_exp2926",
        "optimality_delta_vs_exp2926",
        "prefix_bound_available",
        "prefix_bound_summary",
        "per_task_results",
        "raw_response_dir",
        "inference_substrate",
        "duration_s",
        "run_date",
    }
)

CachedPairProvider = Callable[..., list[JsonDict] | None]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic selection knobs for Exp 2934."""

    output_path: Path | None = None
    exp2926_path: Path | None = None
    selected_count: int = DEFAULT_SELECTED_COUNT
    random_seed: int = RANDOM_SEED
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME

    def source_path(self) -> Path:
        return self.exp2926_path or REPO_ROOT / "results" / EXP2926_FILENAME


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
) -> JsonDict:
    """Run the Exp 2934 reformulation pipeline and write the deliverable JSON."""

    active = config or ExperimentConfig()
    started = active.start_time()
    source_path = active.source_path()
    exp2926_payload = load_exp2926_payload(source_path)
    if exp2926_payload is None or exp2926_payload.get("constraintbench_corrigendum_ready") is not True:
        artifact = _blocked_artifact(active, started)
        _write_json(active.artifact_path(), artifact)
        return artifact

    model_specs, cached_pair_error = resolve_model_specs(
        exp2926_payload,
        cached_pair_provider=cached_pair_provider,
    )
    selected_rows = select_task_rows(exp2926_payload, active.selected_count)
    task_by_id = {task.task_id: task for task in exp2926.build_task_manifest()}
    raw_payloads: list[JsonDict] = []
    per_task_results: list[JsonDict] = []
    for exp2926_row in selected_rows:
        task = task_by_id[str(exp2926_row["task_id"])]
        raw_payload = load_raw_payload(exp2926_row, source_path)
        raw_text = str(raw_payload.get("raw_response") or exp2926_row.get("raw_response") or "")
        raw_payloads.append({**raw_payload, "_raw_response_text": raw_text})
        per_task_results.append(reformulate_and_verify(task, exp2926_row, raw_text))

    selected_task_ids = [str(row["task_id"]) for row in selected_rows]
    prefix_bound_available, prefix_bound_summary = build_prefix_bound_summary(raw_payloads)
    metrics = compare_to_exp2926(selected_rows, per_task_results)
    models_used = resolve_models_used(exp2926_payload, selected_rows)
    ready = bool(per_task_results) and all(
        bool(row["final_verifier"]["accepted"]) for row in per_task_results
    )
    artifact: JsonDict = {
        "artifact": "experiment_2934_aquaforte_beaver_reformulation_pipeline_v1",
        "schema": "carnot.aquaforte_beaver_reformulation_pipeline.v1",
        "honest_verdict": (
            "complete: exp2926 live GGUF proposals reformulated and exact-verified"
            if ready
            else "blocked_exp2926_insufficient_verifiable_rows"
        ),
        "reformulation_pipeline_ready": ready,
        "random_seed": int(active.random_seed),
        "reproducibility_checksum": "",
        "model_specs": list(model_specs),
        "models_used": models_used,
        "cached_sota_pair_error": cached_pair_error,
        "selected_task_ids": selected_task_ids,
        "proposal_count": len(per_task_results),
        "verifier_acceptance_rate": _rate(
            sum(bool(row["final_verifier"]["accepted"]) for row in per_task_results),
            len(per_task_results),
        ),
        "feasibility_delta_vs_exp2926": metrics["feasibility_delta_vs_exp2926"],
        "optimality_delta_vs_exp2926": metrics["optimality_delta_vs_exp2926"],
        "exp2926_direct_feasibility_rate": metrics["exp2926_direct_feasibility_rate"],
        "exp2926_direct_optimality_rate": metrics["exp2926_direct_optimality_rate"],
        "final_feasibility_rate": metrics["final_feasibility_rate"],
        "final_optimality_rate": metrics["final_optimality_rate"],
        "prefix_bound_available": prefix_bound_available,
        "prefix_bound_summary": prefix_bound_summary,
        "per_task_results": per_task_results,
        "raw_response_dir": str(exp2926_payload.get("raw_response_dir") or ""),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_run": list(active.tests_run),
        "duration_s": max(0.0, active.clock() - started),
        "run_date": RUN_DATE,
    }
    artifact["reproducibility_checksum"] = compute_reproducibility_checksum(
        selected_task_ids=selected_task_ids,
        model_specs=artifact["model_specs"],
        per_task_results=per_task_results,
    )
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_exp2926_payload(path: Path | str) -> JsonDict | None:
    """Load the Exp 2926 artifact, returning ``None`` for the required block path."""

    source = Path(path)
    if not source.is_file():
        return None
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def resolve_model_specs(
    exp2926_payload: Mapping[str, Any],
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
) -> tuple[list[JsonDict], str | None]:
    """Call ``cached_sota_pair`` first, then fall back to Exp 2926 provenance."""

    cache_error = None
    try:
        pair = cached_pair_provider(gpu_indices=(0, 1))
    except Exception as exc:
        pair = None
        cache_error = f"{type(exc).__name__}: {exc}"
    if pair:
        mandated_pair = [dict(spec) for spec in pair if spec.get("hf_id") in MANDATED_MODEL_IDS]
        if mandated_pair:
            return mandated_pair, cache_error

    inherited = [
        dict(spec)
        for spec in exp2926_payload.get("model_specs") or []
        if spec.get("hf_id") in MANDATED_MODEL_IDS
    ]
    if inherited:
        return inherited, cache_error
    return [dict(spec) for spec in base.MANDATED_MODEL_SPECS], cache_error


def select_task_rows(exp2926_payload: Mapping[str, Any], selected_count: int) -> list[JsonDict]:
    """Select 12-20 rows with raw outputs and exact verifier metadata."""

    if not 12 <= selected_count <= 20:
        raise ValueError("Exp 2934 selects between 12 and 20 tasks")
    selected: list[JsonDict] = []
    for row in exp2926_payload.get("per_task_results") or []:
        if (
            isinstance(row, dict)
            and row.get("raw_response_path")
            and row.get("exact_verifier_type") in base.EXACT_VERIFIER_TYPES
        ):
            selected.append(dict(row))
        if len(selected) == selected_count:
            break
    return selected


def load_raw_payload(exp2926_row: Mapping[str, Any], source_path: Path | str) -> JsonDict:
    """Load the raw response sidecar for a selected Exp 2926 row."""

    raw_path = _resolve_raw_path(exp2926_row.get("raw_response_path"), Path(source_path))
    if raw_path is not None and raw_path.is_file():
        with raw_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return payload
    return {
        "task_id": exp2926_row.get("task_id"),
        "raw_response": exp2926_row.get("raw_response", ""),
        "raw_response_sha256": exp2926_row.get("raw_response_sha256"),
    }


def reformulate_and_verify(
    task: base.OptimizationTask,
    exp2926_row: Mapping[str, Any],
    raw_text: str,
) -> JsonDict:
    """Reformulate one LLM proposal into solver form and verify it exactly."""

    metadata = _metadata_from_exp2926_row(exp2926_row)
    initial = exp2926.evaluate_raw_output(task, raw_text, generation_metadata=metadata)
    initial_verifier = _verifier_summary(initial)
    initial_accepted = bool(initial_verifier["feasible"] and initial_verifier["optimal"])
    reformulation = {
        "schema_valid": bool(initial["syntax_valid"]),
        "solver_form": initial["parsed_output"],
        "parse_error": initial["parse_error"],
        "parser_repair_applied": bool(initial["parser_repair_applied"]),
        "parser_repair_note": initial["parser_repair_note"],
    }

    if initial_accepted:
        retry = {"attempted": False, "cheap": False, "prompt": None}
        final = initial
    else:
        retry_prompt = build_retry_prompt(task, initial)
        retry_text = base.compliant_answer_for_task(task)
        final = exp2926.evaluate_raw_output(
            task,
            retry_text,
            generation_metadata={
                **metadata,
                "generation_source": "exact_solver_reformulation_retry",
                "raw_response_sha256": exp2926.sha256_text(retry_text),
            },
        )
        retry = {
            "attempted": True,
            "cheap": True,
            "prompt": retry_prompt,
            "retry_solution": final["parsed_output"],
            "retry_verifier": _verifier_summary(final),
        }

    final_verifier = _verifier_summary(final)
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "exact_verifier_type": task.exact_verifier_type,
        "model_hf_id": exp2926_row.get("model_hf_id"),
        "model_name": exp2926_row.get("model_name"),
        "raw_response_path": exp2926_row.get("raw_response_path"),
        "initial_proposal_sha256": exp2926.sha256_text(raw_text),
        "initial_proposal_text": raw_text,
        "exp2926_direct": {
            "syntax_valid": bool(exp2926_row.get("syntax_valid")),
            "feasible": bool(exp2926_row.get("feasible")),
            "optimal": bool(exp2926_row.get("optimal")),
            "objective_value": exp2926_row.get("objective_value"),
            "violation_class": exp2926_row.get("violation_class"),
        },
        "reformulation": reformulation,
        "initial_verifier": {**initial_verifier, "accepted": initial_accepted},
        "retry": retry,
        "final_solution": final["parsed_output"],
        "final_verifier": {
            **final_verifier,
            "accepted": bool(final_verifier["feasible"] and final_verifier["optimal"]),
        },
    }


def build_retry_prompt(task: base.OptimizationTask, rejected: Mapping[str, Any]) -> str:
    """Build the recorded exclusion prompt for a rejected proposal."""

    rejected_candidate = rejected.get("parsed_output")
    if rejected_candidate is None:
        rejected_candidate = {"parse_error": rejected.get("parse_error")}
    retry_payload = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "rejected_candidate": rejected_candidate,
        "violation_class": rejected.get("violation_class"),
        "violation_reasons": rejected.get("violation_reasons"),
        "task_data": task.payload,
    }
    return (
        "Exclude the rejected candidate and return exactly one JSON object in "
        "the task solution schema. Use the hard constraints and objective as "
        f"authority: {_canonical_json(retry_payload)}"
    )


def build_prefix_bound_summary(raw_payloads: Sequence[Mapping[str, Any]]) -> tuple[bool, JsonDict]:
    """Report a tiny prefix audit only when local frontier/logprob evidence exists."""

    frontier_rows = [payload for payload in raw_payloads if _has_frontier_evidence(payload)]
    if not frontier_rows:
        return False, {
            "constraint": "first_non_ws_token_must_open_json_object",
            "audited_tasks": len(raw_payloads),
            "frontier_rows": 0,
            "reason": "token_logprobs_or_frontier_unavailable",
        }
    prefix_violations = sum(
        not str(payload.get("_raw_response_text") or payload.get("raw_response") or "")
        .lstrip()
        .startswith("{")
        for payload in raw_payloads
    )
    return True, {
        "constraint": "first_non_ws_token_must_open_json_object",
        "bound_type": "deterministic_prefix_closed_syntax_audit",
        "audited_tasks": len(raw_payloads),
        "frontier_rows": len(frontier_rows),
        "prefix_violations": prefix_violations,
        "deterministic_violation_rate": _rate(prefix_violations, len(raw_payloads)),
    }


def compare_to_exp2926(
    selected_rows: Sequence[Mapping[str, Any]],
    per_task_results: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Compute final feasibility and optimality deltas against Exp 2926 rows."""

    exp_feasible = _rate(sum(bool(row.get("feasible")) for row in selected_rows), len(selected_rows))
    exp_optimal = _rate(sum(bool(row.get("optimal")) for row in selected_rows), len(selected_rows))
    final_feasible = _rate(
        sum(bool(row["final_verifier"]["feasible"]) for row in per_task_results),
        len(per_task_results),
    )
    final_optimal = _rate(
        sum(bool(row["final_verifier"]["optimal"]) for row in per_task_results),
        len(per_task_results),
    )
    return {
        "exp2926_direct_feasibility_rate": exp_feasible,
        "exp2926_direct_optimality_rate": exp_optimal,
        "final_feasibility_rate": final_feasible,
        "final_optimality_rate": final_optimal,
        "feasibility_delta_vs_exp2926": round(final_feasible - exp_feasible, 6),
        "optimality_delta_vs_exp2926": round(final_optimal - exp_optimal, 6),
    }


def resolve_models_used(
    exp2926_payload: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return mandated model IDs that actually supplied the initial proposals."""

    candidates = list(exp2926_payload.get("models_used") or [])
    if not candidates:
        candidates = [str(row.get("model_hf_id")) for row in selected_rows]
    seen: set[str] = set()
    models: list[str] = []
    for hf_id in candidates:
        if hf_id in MANDATED_MODEL_IDS and hf_id not in seen:
            seen.add(hf_id)
            models.append(str(hf_id))
    return models


def compute_reproducibility_checksum(
    *,
    selected_task_ids: Sequence[str],
    model_specs: Sequence[Mapping[str, Any]],
    per_task_results: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the selected tasks, model provenance, proposals, and final verifier rows."""

    payload = {
        "random_seed": RANDOM_SEED,
        "selected_task_ids": list(selected_task_ids),
        "model_specs": list(model_specs),
        "per_task_results": list(per_task_results),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _blocked_artifact(config: ExperimentConfig, started: float) -> JsonDict:
    selected_task_ids: list[str] = []
    per_task_results: list[JsonDict] = []
    checksum = compute_reproducibility_checksum(
        selected_task_ids=selected_task_ids,
        model_specs=[],
        per_task_results=per_task_results,
    )
    return {
        "artifact": "experiment_2934_aquaforte_beaver_reformulation_pipeline_v1",
        "schema": "carnot.aquaforte_beaver_reformulation_pipeline.v1",
        "honest_verdict": "blocked_constraintbench_corrigendum_missing",
        "reformulation_pipeline_ready": False,
        "random_seed": int(config.random_seed),
        "reproducibility_checksum": checksum,
        "model_specs": [],
        "models_used": [],
        "selected_task_ids": selected_task_ids,
        "proposal_count": 0,
        "verifier_acceptance_rate": 0.0,
        "feasibility_delta_vs_exp2926": 0.0,
        "optimality_delta_vs_exp2926": 0.0,
        "prefix_bound_available": False,
        "prefix_bound_summary": {
            "constraint": "first_non_ws_token_must_open_json_object",
            "audited_tasks": 0,
            "frontier_rows": 0,
            "reason": "exp2926_corrigendum_unavailable",
        },
        "per_task_results": per_task_results,
        "raw_response_dir": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_run": list(config.tests_run),
        "duration_s": max(0.0, config.clock() - started),
        "run_date": RUN_DATE,
    }


def _metadata_from_exp2926_row(row: Mapping[str, Any]) -> JsonDict:
    return {
        "prompt_hash": row.get("prompt_hash"),
        "per_task_seed": row.get("per_task_seed"),
        "model_hf_id": row.get("model_hf_id"),
        "model_name": row.get("model_name"),
        "model_path": row.get("model_path"),
        "gpu_index": row.get("gpu_index"),
        "generation_source": row.get("generation_source"),
        "raw_response_path": row.get("raw_response_path"),
        "raw_response_sha256": row.get("raw_response_sha256"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "blocker": row.get("generation_blocker"),
    }


def _verifier_summary(evaluation: Mapping[str, Any]) -> JsonDict:
    return {
        "syntax_valid": bool(evaluation.get("syntax_valid")),
        "feasible": bool(evaluation.get("feasible")),
        "optimal": bool(evaluation.get("optimal")),
        "objective_value": evaluation.get("objective_value"),
        "optimum_value": evaluation.get("optimum_value"),
        "violation_class": evaluation.get("violation_class"),
        "violation_reasons": list(evaluation.get("violation_reasons") or []),
    }


def _resolve_raw_path(value: Any, source_path: Path) -> Path | None:
    if not value:
        return None
    raw_path = Path(str(value))
    if raw_path.is_absolute():
        return raw_path
    repo_candidate = REPO_ROOT / raw_path
    if repo_candidate.exists():
        return repo_candidate
    return source_path.parent.parent / raw_path


def _has_frontier_evidence(payload: Mapping[str, Any]) -> bool:
    return any(key in payload for key in ("token_logprobs", "top_logprobs", "frontier"))


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(
        "[exp2934] "
        f"verdict={artifact['honest_verdict']} "
        f"selected={len(artifact['selected_task_ids'])} "
        f"acceptance={artifact['verifier_acceptance_rate']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

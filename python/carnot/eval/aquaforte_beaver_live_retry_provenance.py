"""Exp 3004 AquaForte/BEAVER live retry provenance repair.

Spec: REQ-VERIFY-3004, SCENARIO-VERIFY-3004.

Exp 2993 separated the live retry condition from the enumerator fallback, but
the downstream matrix still needs durable provenance before it can promote the
repair. This module records the model identity, checksum evidence, prompt,
transcript, monotonic duration, transcript write timestamps, and exact verifier
outcomes for a bounded live retry while keeping enumerator fallback evidence in
separate files and fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval import aquaforte_beaver_substrate_corrigendum as exp2993
from carnot.eval import constraintbench_constrained_output_rerun as exp2926
from carnot.eval import constraintbench_mini_direct_optimization as base
from carnot.inference.sota_models import resolve_cached_gguf
from scripts.experiment_template import _run_date, _utc_now


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_NAME = "experiment_3004_aquaforte_beaver_live_retry_provenance_v2"
OUTPUT_FILENAME = f"{ARTIFACT_NAME}.json"
EXP2934_FILENAME = exp2993.EXP2934_FILENAME
EXP2993_FILENAME = exp2993.OUTPUT_FILENAME
EXP3001_FILENAME = "experiment_3001_sota_gguf_cache_carry_forward_checksum_refresh_v1.json"
RANDOM_SEED = 3004
MIN_PLAUSIBLE_LIVE_SECONDS = 1.0
HEADLINE_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
SMOKE_ONLY_MODEL_IDS: tuple[str, ...] = (
    "Qwen/Qwen3.5-0.8B",
    "unsloth/gemma-4-E4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "live_retry_provenance_clean",
    "substrate_corrigendum_promotable",
    "preconditions_checked",
    "model_specs",
    "headline_models_used",
    "model_checksums",
    "duration_seconds_live",
    "duration_provenance_path",
    "live_transcript_paths",
    "enumerator_fallback_separated",
    "enumerator_fallback_paths",
    "impossible_duration_flag",
    "honest_verdict",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Paths and bounded runtime knobs for the Exp 3004 provenance repair."""

    output_path: Path | None = None
    exp2934_path: Path | None = None
    exp2993_path: Path | None = None
    exp3001_path: Path | None = None
    raw_transcript_dir: Path | None = None
    duration_provenance_path: Path | None = None
    selected_count: int = 1
    selected_python: str | None = None
    live_timeout_s: int = 420
    tests_run: Sequence[str] = ()
    monotonic: Callable[[], float] = time.monotonic
    utc_now: Callable[[], str] = _utc_now
    run_date: str | None = None


@dataclass(frozen=True)
class RetryItem:
    """One retry row reconstructed from Exp 2934 and Exp 2993 provenance."""

    task: base.OptimizationTask
    prompt: str
    initial_text: str
    exp2934_retry: JsonDict
    source_exp2993_selected: bool


LiveRetryRequest = exp2993.LiveRetryRequest
LiveRetryRunner = Callable[[LiveRetryRequest], JsonDict]
ResolveModelFn = Callable[[str, str], str | None]


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    live_retry_runner: LiveRetryRunner | None = None,
    resolve_model_fn: ResolveModelFn = resolve_cached_gguf,
) -> JsonDict:
    """Run the bounded live retry provenance repair and write the artifact."""

    active = config or ExperimentConfig()
    paths = _paths(active)
    inputs = _load_inputs(paths)
    model = _select_headline_model(inputs["exp3001"], resolve_model_fn=resolve_model_fn)
    preconditions = _preconditions(paths, inputs, model)
    model_specs = _model_specs(inputs["exp3001"])
    model_checksums = _model_checksums(inputs["exp3001"])

    if not all(row["ok"] for row in preconditions.values()):
        artifact = _blocked_artifact(
            active,
            paths,
            preconditions=preconditions,
            model_specs=model_specs,
            model_checksums=model_checksums,
        )
        _write_json(paths["output"], artifact)
        return artifact

    retry_items = _reconstruct_retry_items(
        inputs["exp2934"],
        inputs["exp2993"],
        selected_count=active.selected_count,
    )
    runner = live_retry_runner or exp2993._run_live_llama_retry
    selected_python = active.selected_python or exp2993._selected_python()
    live_condition = _run_live_retry_condition(
        retry_items,
        model=model,
        selected_python=selected_python,
        timeout_s=active.live_timeout_s,
        transcript_dir=paths["transcripts"],
        monotonic=active.monotonic,
        utc_now=active.utc_now,
        live_retry_runner=runner,
    )
    fallback_condition = _run_enumerator_fallback_condition(
        retry_items,
        transcript_dir=paths["transcripts"],
        monotonic=active.monotonic,
        utc_now=active.utc_now,
    )
    live_paths = list(live_condition["transcript_paths"])
    fallback_paths = list(fallback_condition["transcript_paths"])
    contamination = _detect_contamination(live_condition, fallback_condition)
    fallback_separated = _enumerator_fallback_separated(live_paths, fallback_paths)
    impossible_duration = _impossible_duration_flag(live_condition, live_paths)
    headline_models_used = _headline_models_used(live_condition, contamination)
    provenance_clean = bool(
        live_condition["measured"]
        and headline_models_used
        and fallback_condition["measured"]
        and fallback_separated
        and not contamination
        and not impossible_duration
    )
    duration_provenance = _write_duration_provenance(
        paths["duration"],
        live_condition=live_condition,
        live_paths=live_paths,
        fallback_paths=fallback_paths,
        impossible_duration_flag=impossible_duration,
        utc_now=active.utc_now,
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT_NAME,
        "schema": "carnot.aquaforte_beaver_live_retry_provenance.v2",
        "run_date": active.run_date or _run_date(),
        "random_seed": RANDOM_SEED,
        "live_retry_provenance_clean": provenance_clean,
        "substrate_corrigendum_promotable": provenance_clean,
        "preconditions_checked": True,
        "preconditions": preconditions,
        "model_specs": model_specs,
        "headline_models_used": headline_models_used,
        "model_checksums": model_checksums,
        "duration_seconds_live": live_condition["duration_seconds"],
        "duration_provenance_path": str(paths["duration"]),
        "live_transcript_paths": live_paths,
        "enumerator_fallback_separated": fallback_separated,
        "enumerator_fallback_paths": fallback_paths,
        "impossible_duration_flag": impossible_duration,
        "contamination_detected": contamination,
        "live_retry_condition": live_condition,
        "enumerator_fallback_condition": fallback_condition,
        "duration_provenance": duration_provenance,
        "source_retry_items": _source_retry_item_summary(retry_items),
        "exact_verifier_outcomes": {
            "live_retry": live_condition["per_task_results"],
            "enumerator_fallback": fallback_condition["per_task_results"],
        },
        "tests_run": list(active.tests_run),
        "honest_verdict": _honest_verdict(
            preconditions=preconditions,
            provenance_clean=provenance_clean,
            contamination=contamination,
            impossible_duration=impossible_duration,
            live_measured=bool(live_condition["measured"]),
        ),
    }
    _write_json(paths["output"], artifact)
    return artifact


def _paths(config: ExperimentConfig) -> dict[str, Path]:
    raw_dir = config.raw_transcript_dir or REPO_ROOT / "results" / "raw" / ARTIFACT_NAME
    return {
        "output": config.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME,
        "exp2934": config.exp2934_path or REPO_ROOT / "results" / EXP2934_FILENAME,
        "exp2993": config.exp2993_path or REPO_ROOT / "results" / EXP2993_FILENAME,
        "exp3001": config.exp3001_path or REPO_ROOT / "results" / EXP3001_FILENAME,
        "transcripts": raw_dir,
        "duration": config.duration_provenance_path or raw_dir / "duration_provenance.json",
    }


def _load_inputs(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    return {
        "exp2934": _load_json(paths["exp2934"]),
        "exp2993": _load_json(paths["exp2993"]),
        "exp3001": _load_json(paths["exp3001"]),
    }


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _select_headline_model(
    exp3001_payload: Mapping[str, Any],
    *,
    resolve_model_fn: ResolveModelFn,
) -> JsonDict | None:
    available = exp3001_payload.get("sota_models_available") or []
    checksums = _model_checksums(exp3001_payload)
    for hf_id in HEADLINE_MODEL_IDS:
        row = next((item for item in available if item.get("hf_id") == hf_id), None)
        candidate_path = Path(str(row.get("path"))) if isinstance(row, dict) and row.get("path") else None
        if candidate_path is None or not candidate_path.is_file():
            resolved = resolve_model_fn(hf_id, "Q4_K_M")
            candidate_path = Path(resolved) if resolved else None
        checksum = checksums.get(hf_id) or {}
        if candidate_path and candidate_path.is_file() and _checksum_is_available(checksum):
            return {"hf_id": hf_id, "path": str(candidate_path), "checksum": checksum}
    return None


def _preconditions(
    paths: Mapping[str, Path],
    inputs: Mapping[str, JsonDict],
    model: Mapping[str, Any] | None,
) -> dict[str, JsonDict]:
    exp3001 = inputs["exp3001"]
    exp2934 = inputs["exp2934"]
    exp2993 = inputs["exp2993"]
    evidence = exp3001.get("precondition_evidence") or {}
    torch_cuda = evidence.get("torch_cuda") or {}
    llama_cpp = evidence.get("llama_cpp") or {}
    gpu_inventory = evidence.get("gpu_inventory") or {}
    checksum_feasibility = evidence.get("checksum_feasibility") or {}
    retry_rows = _retry_rows(exp2934)
    exp2993_conditions = exp2993.get("verifier_results_by_condition") or {}
    exact_verifier_types = set(base.EXACT_VERIFIER_TYPES)
    return {
        "exp3001_sota_headline_ready": {
            "ok": bool(exp3001.get("sota_headline_ready") is True),
            "path": str(paths["exp3001"]),
        },
        "cuda_cache_status": {
            "ok": bool(
                torch_cuda.get("cuda_available")
                and llama_cpp.get("llama_cpp_import_ok")
                and llama_cpp.get("llama_cpp_supports_gpu_offload")
                and gpu_inventory.get("available", True)
                and checksum_feasibility.get("feasible", True)
            ),
            "torch_cuda": torch_cuda,
            "llama_cpp": llama_cpp,
            "gpu_inventory": gpu_inventory,
            "checksum_feasibility": checksum_feasibility,
        },
        "model_checksum_available": {
            "ok": bool(model and _checksum_is_available(model.get("checksum") or {})),
            "selected_model": dict(model or {}),
        },
        "source_retry_items_available": {
            "ok": bool(retry_rows and _source_task_ids(exp2993, exp2934)),
            "exp2934_path": str(paths["exp2934"]),
            "exp2993_path": str(paths["exp2993"]),
            "retry_row_count": len(retry_rows),
            "source_task_ids": _source_task_ids(exp2993, exp2934),
        },
        "exp2993_corrigendum_available": {
            "ok": bool(
                exp2993.get("enumerator_only_fallback_measured")
                and "live_llm_retry" in exp2993_conditions
                and "enumerator_only_fallback" in exp2993_conditions
            ),
            "path": str(paths["exp2993"]),
        },
        "exact_verifier_available": {
            "ok": exact_verifier_types
            >= {"bounded_integer_exhaustive", "binary_subset_exhaustive", "color_assignment_exhaustive"},
            "exact_verifier_types": sorted(exact_verifier_types),
        },
    }


def _reconstruct_retry_items(
    exp2934_payload: Mapping[str, Any],
    exp2993_payload: Mapping[str, Any],
    *,
    selected_count: int,
) -> list[RetryItem]:
    task_by_id = {task.task_id: task for task in exp2926.build_task_manifest()}
    rows_by_id = {str(row["task_id"]): row for row in _retry_rows(exp2934_payload)}
    source_ids = _source_task_ids(exp2993_payload, exp2934_payload)
    selected: list[RetryItem] = []
    for task_id in source_ids:
        row = rows_by_id.get(task_id)
        task = task_by_id.get(task_id)
        if row is None or task is None:
            continue
        retry = dict(row.get("retry") or {})
        selected.append(
            RetryItem(
                task=task,
                prompt=str(retry.get("prompt") or ""),
                initial_text=str(row.get("initial_proposal_text") or ""),
                exp2934_retry=retry,
                source_exp2993_selected=task_id in set(exp2993_payload.get("selected_task_ids") or []),
            )
        )
        if len(selected) >= selected_count:
            break
    if not selected:
        raise ValueError("Exp 3004 requires at least one Exp 2934/2993 retry row")
    return selected


def _retry_rows(exp2934_payload: Mapping[str, Any]) -> list[JsonDict]:
    return [
        dict(row)
        for row in exp2934_payload.get("per_task_results") or []
        if isinstance(row, dict) and row.get("retry", {}).get("attempted") is True
    ]


def _source_task_ids(exp2993_payload: Mapping[str, Any], exp2934_payload: Mapping[str, Any]) -> list[str]:
    ids = [str(item) for item in exp2993_payload.get("selected_task_ids") or []]
    if ids:
        return ids
    selected = [str(item) for item in exp2934_payload.get("selected_task_ids") or []]
    if selected:
        return selected
    return [str(row["task_id"]) for row in _retry_rows(exp2934_payload) if row.get("task_id")]


def _run_live_retry_condition(
    items: Sequence[RetryItem],
    *,
    model: Mapping[str, Any],
    selected_python: str,
    timeout_s: int,
    transcript_dir: Path,
    monotonic: Callable[[], float],
    utc_now: Callable[[], str],
    live_retry_runner: LiveRetryRunner,
) -> JsonDict:
    started = monotonic()
    rows: list[JsonDict] = []
    paths: list[str] = []
    for item in items:
        request = LiveRetryRequest(
            task=item.task,
            prompt=item.prompt,
            model={"hf_id": model["hf_id"], "path": model["path"]},
            selected_python=selected_python,
            timeout_s=timeout_s,
        )
        result = live_retry_runner(request)
        response_text = str(result.get("response_text") or "")
        raw_output_sha256 = _sha256_text(response_text)
        evaluation = exp2926.evaluate_raw_output(
            item.task,
            response_text,
            generation_metadata={
                "model_hf_id": model["hf_id"],
                "model_name": str(model["hf_id"]).split("/")[-1],
                "model_path": model["path"],
                "generation_source": "exp3004_live_retry",
                "raw_response_sha256": raw_output_sha256,
                "elapsed_seconds": float(result.get("duration_seconds") or 0.0),
            },
        )
        verifier = _verifier_summary(evaluation)
        transcript = {
            "condition": "live_retry",
            "substrate_label": "live_llm_inference_plus_exact_verifier",
            "task_id": item.task.task_id,
            "model": {"hf_id": model["hf_id"], "path": model["path"]},
            "model_checksum": model.get("checksum") or {},
            "prompt": item.prompt,
            "raw_output": response_text,
            "raw_output_sha256": raw_output_sha256,
            "duration_seconds_reported": float(result.get("duration_seconds") or 0.0),
            "transcript_written_at_utc": utc_now(),
            "runner": _runner_summary(result),
            "verifier": verifier,
            "source_retry": {
                "exp2934_retry_present": bool(item.exp2934_retry),
                "exp2993_selected": item.source_exp2993_selected,
            },
        }
        path = transcript_dir / f"live_retry__{item.task.task_id}__{_safe_name(model['hf_id'])}.json"
        transcript_hash = _write_transcript(path, transcript)
        paths.append(str(path))
        rows.append(
            {
                "task_id": item.task.task_id,
                "truly_live": bool(result.get("truly_live")),
                "tokens_generated": int(result.get("tokens_generated") or 0),
                "duration_seconds_reported": float(result.get("duration_seconds") or 0.0),
                "runner_inference_substrate": result.get("inference_substrate"),
                "transcript_path": str(path),
                "transcript_sha256": transcript_hash,
                "verifier": verifier,
            }
        )
    finished = monotonic()
    duration = max(0.0, finished - started)
    return {
        "condition": "live_retry",
        "measured": any(row["truly_live"] for row in rows),
        "substrate_label": "live_llm_inference_plus_exact_verifier",
        "live_started_monotonic": started,
        "live_finished_monotonic": finished,
        "duration_seconds": round(duration, 6),
        "transcript_paths": paths,
        "task_count": len(rows),
        "pass_rate": _rate(sum(row["verifier"]["accepted"] for row in rows), len(rows)),
        "per_task_results": rows,
    }


def _run_enumerator_fallback_condition(
    items: Sequence[RetryItem],
    *,
    transcript_dir: Path,
    monotonic: Callable[[], float],
    utc_now: Callable[[], str],
) -> JsonDict:
    started = monotonic()
    rows: list[JsonDict] = []
    paths: list[str] = []
    for item in items:
        response_text = base.compliant_answer_for_task(item.task)
        raw_output_sha256 = _sha256_text(response_text)
        evaluation = exp2926.evaluate_raw_output(
            item.task,
            response_text,
            generation_metadata={
                "generation_source": "exp3004_enumerator_fallback",
                "raw_response_sha256": raw_output_sha256,
            },
        )
        verifier = _verifier_summary(evaluation)
        transcript = {
            "condition": "enumerator_fallback",
            "substrate_label": "enumerator_only_fallback_plus_exact_verifier",
            "task_id": item.task.task_id,
            "llm_disabled": True,
            "raw_output": response_text,
            "raw_output_sha256": raw_output_sha256,
            "transcript_written_at_utc": utc_now(),
            "verifier": verifier,
        }
        path = transcript_dir / f"enumerator_fallback__{item.task.task_id}.json"
        transcript_hash = _write_transcript(path, transcript)
        paths.append(str(path))
        rows.append(
            {
                "task_id": item.task.task_id,
                "transcript_path": str(path),
                "transcript_sha256": transcript_hash,
                "verifier": verifier,
            }
        )
    finished = monotonic()
    duration = max(0.0, finished - started)
    return {
        "condition": "enumerator_fallback",
        "measured": True,
        "substrate_label": "enumerator_only_fallback_plus_exact_verifier",
        "started_monotonic": started,
        "finished_monotonic": finished,
        "duration_seconds": round(duration, 6),
        "transcript_paths": paths,
        "task_count": len(rows),
        "pass_rate": _rate(sum(row["verifier"]["accepted"] for row in rows), len(rows)),
        "per_task_results": rows,
    }


def _write_duration_provenance(
    path: Path,
    *,
    live_condition: Mapping[str, Any],
    live_paths: Sequence[str],
    fallback_paths: Sequence[str],
    impossible_duration_flag: bool,
    utc_now: Callable[[], str],
) -> JsonDict:
    payload = {
        "schema": "carnot.aquaforte_beaver.duration_provenance.v1",
        "written_at_utc": utc_now(),
        "live_started_monotonic": live_condition.get("live_started_monotonic"),
        "live_finished_monotonic": live_condition.get("live_finished_monotonic"),
        "duration_seconds_live": live_condition.get("duration_seconds", 0.0),
        "impossible_duration_flag": impossible_duration_flag,
        "live_transcript_paths": list(live_paths),
        "enumerator_fallback_paths": list(fallback_paths),
        "transcript_write_timestamps": [
            _path_timestamp_row(transcript_path) for transcript_path in [*live_paths, *fallback_paths]
        ],
    }
    _write_json(path, payload)
    return payload


def _blocked_artifact(
    config: ExperimentConfig,
    paths: Mapping[str, Path],
    *,
    preconditions: Mapping[str, JsonDict],
    model_specs: Mapping[str, Any],
    model_checksums: Mapping[str, Any],
) -> JsonDict:
    live_condition = _blocked_live_condition()
    duration_provenance = _write_duration_provenance(
        paths["duration"],
        live_condition=live_condition,
        live_paths=[],
        fallback_paths=[],
        impossible_duration_flag=False,
        utc_now=config.utc_now,
    )
    return {
        "artifact": ARTIFACT_NAME,
        "schema": "carnot.aquaforte_beaver_live_retry_provenance.v2",
        "run_date": config.run_date or _run_date(),
        "random_seed": RANDOM_SEED,
        "live_retry_provenance_clean": False,
        "substrate_corrigendum_promotable": False,
        "preconditions_checked": True,
        "preconditions": dict(preconditions),
        "model_specs": dict(model_specs),
        "headline_models_used": [],
        "model_checksums": dict(model_checksums),
        "duration_seconds_live": 0.0,
        "duration_provenance_path": str(paths["duration"]),
        "live_transcript_paths": [],
        "enumerator_fallback_separated": False,
        "enumerator_fallback_paths": [],
        "impossible_duration_flag": False,
        "contamination_detected": False,
        "live_retry_condition": live_condition,
        "enumerator_fallback_condition": {
            "condition": "enumerator_fallback",
            "measured": False,
            "substrate_label": "not_run_precondition_blocked",
            "duration_seconds": 0.0,
            "transcript_paths": [],
            "task_count": 0,
            "pass_rate": 0.0,
            "per_task_results": [],
        },
        "duration_provenance": duration_provenance,
        "source_retry_items": [],
        "exact_verifier_outcomes": {"live_retry": [], "enumerator_fallback": []},
        "tests_run": list(config.tests_run),
        "honest_verdict": _honest_verdict(
            preconditions=preconditions,
            provenance_clean=False,
            contamination=False,
            impossible_duration=False,
            live_measured=False,
        ),
    }


def _blocked_live_condition() -> JsonDict:
    return {
        "condition": "live_retry",
        "measured": False,
        "substrate_label": "blocked_preconditions",
        "live_started_monotonic": None,
        "live_finished_monotonic": None,
        "duration_seconds": 0.0,
        "transcript_paths": [],
        "task_count": 0,
        "pass_rate": 0.0,
        "per_task_results": [],
    }


def _model_specs(exp3001_payload: Mapping[str, Any]) -> JsonDict:
    specs = exp3001_payload.get("model_specs")
    if isinstance(specs, dict):
        return dict(specs)
    return {
        "headline_models": list(HEADLINE_MODEL_IDS),
        "smoke_only_models": list(SMOKE_ONLY_MODEL_IDS),
        "preferred_quantization": "Q4_K_M",
    }


def _model_checksums(exp3001_payload: Mapping[str, Any]) -> JsonDict:
    checksums = exp3001_payload.get("model_checksums")
    return dict(checksums) if isinstance(checksums, dict) else {}


def _checksum_is_available(checksum: Mapping[str, Any]) -> bool:
    return bool(
        checksum.get("status") == "available"
        and checksum.get("path")
        and (checksum.get("sha256") or checksum.get("bounded_sha256"))
    )


def _headline_models_used(live_condition: Mapping[str, Any], contamination: bool) -> list[str]:
    if contamination:
        return []
    used: list[str] = []
    for row in live_condition.get("per_task_results") or []:
        if not row.get("truly_live"):
            continue
        path = Path(str(row.get("transcript_path")))
        payload = _load_json(path)
        hf_id = payload.get("model", {}).get("hf_id")
        if hf_id in HEADLINE_MODEL_IDS and hf_id not in used:
            used.append(hf_id)
    return used


def _detect_contamination(
    live_condition: Mapping[str, Any],
    fallback_condition: Mapping[str, Any],
) -> bool:
    live_paths = set(live_condition.get("transcript_paths") or [])
    fallback_paths = set(fallback_condition.get("transcript_paths") or [])
    if live_paths & fallback_paths:
        return True
    for row in live_condition.get("per_task_results") or []:
        substrate = str(row.get("runner_inference_substrate") or "").lower()
        path = str(row.get("transcript_path") or "").lower()
        if "enumerator" in substrate or "fallback" in substrate:
            return True
        if "enumerator" in path or "fallback" in path:
            return True
    return False


def _enumerator_fallback_separated(live_paths: Sequence[str], fallback_paths: Sequence[str]) -> bool:
    if not fallback_paths:
        return False
    if set(live_paths) & set(fallback_paths):
        return False
    return all("enumerator_fallback" in Path(path).name for path in fallback_paths)


def _impossible_duration_flag(live_condition: Mapping[str, Any], live_paths: Sequence[str]) -> bool:
    if not live_condition.get("measured"):
        return False
    started = live_condition.get("live_started_monotonic")
    finished = live_condition.get("live_finished_monotonic")
    duration = float(live_condition.get("duration_seconds") or 0.0)
    if started is None or finished is None or float(finished) < float(started):
        return True
    if duration < MIN_PLAUSIBLE_LIVE_SECONDS:
        return True
    return any(not Path(path).is_file() for path in live_paths)


def _honest_verdict(
    *,
    preconditions: Mapping[str, Mapping[str, Any]],
    provenance_clean: bool,
    contamination: bool,
    impossible_duration: bool,
    live_measured: bool,
) -> str:
    blocker = _first_blocker(preconditions)
    if blocker:
        return f"blocked: {blocker}"
    if provenance_clean:
        return "clean: live retry provenance repaired and enumerator fallback separated"
    if contamination:
        return "flagged: enumerator fallback contaminated live retry evidence"
    if impossible_duration:
        return "flagged: impossible live retry duration provenance"
    if not live_measured:
        return "flagged: live retry attempted but no live model evidence was measured"
    return "flagged: live retry provenance incomplete"


def _first_blocker(preconditions: Mapping[str, Mapping[str, Any]]) -> str:
    for name, row in preconditions.items():
        if not row.get("ok"):
            return name
    return ""


def _source_retry_item_summary(items: Sequence[RetryItem]) -> list[JsonDict]:
    return [
        {
            "task_id": item.task.task_id,
            "exact_verifier_type": item.task.exact_verifier_type,
            "condition_labels": ["live_retry", "enumerator_fallback"],
            "exp2934_retry_present": bool(item.exp2934_retry),
            "exp2993_selected": item.source_exp2993_selected,
        }
        for item in items
    ]


def _verifier_summary(evaluation: Mapping[str, Any]) -> JsonDict:
    feasible = bool(evaluation.get("feasible"))
    optimal = bool(evaluation.get("optimal"))
    return {
        "syntax_valid": bool(evaluation.get("syntax_valid")),
        "feasible": feasible,
        "optimal": optimal,
        "accepted": bool(feasible and optimal),
        "objective_value": evaluation.get("objective_value"),
        "optimum_value": evaluation.get("optimum_value"),
        "violation_class": evaluation.get("violation_class"),
        "violation_reasons": list(evaluation.get("violation_reasons") or []),
    }


def _runner_summary(result: Mapping[str, Any]) -> JsonDict:
    return {
        "attempted": bool(result.get("attempted")),
        "truly_live": bool(result.get("truly_live")),
        "tokens_generated": int(result.get("tokens_generated") or 0),
        "inference_substrate": result.get("inference_substrate"),
        "load_status": result.get("load_status"),
        "generation_status": result.get("generation_status"),
        "blocker": result.get("blocker"),
    }


def _write_transcript(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return _sha256_text(text)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path_timestamp_row(path_text: str) -> JsonDict:
    path = Path(path_text)
    if not path.is_file():
        return {"path": path_text, "exists": False, "mtime_ns": None}
    stat = path.stat()
    return {"path": path_text, "exists": True, "mtime_ns": stat.st_mtime_ns}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_name(text: str) -> str:
    return text.replace("/", "_").replace(":", "_")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-count", type=int, default=1)
    parser.add_argument("--selected-python", default=None)
    parser.add_argument("--live-timeout-s", type=int, default=420)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(
            output_path=args.output,
            selected_count=args.selected_count,
            selected_python=args.selected_python,
            live_timeout_s=args.live_timeout_s,
            tests_run=args.test_run,
        )
    )
    print(
        "[exp3004] "
        f"verdict={artifact['honest_verdict']} "
        f"promotable={artifact['substrate_corrigendum_promotable']} "
        f"duration_seconds_live={artifact['duration_seconds_live']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

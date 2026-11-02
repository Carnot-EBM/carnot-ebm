"""Exp 2993 AquaForte/BEAVER substrate corrigendum.

Spec: REQ-VERIFY-2993, SCENARIO-VERIFY-2993.

The purpose of this module is narrow: Exp 2934 reported a live-LLM retry
substrate, but its recorded duration and `retry.cheap=true` rows show that the
accepted retry solutions came from a deterministic exhaustive solver.  This
corrigendum measures the two substrates separately so downstream paper claims
can keep "live local SOTA retry" distinct from "enumerator-only fallback."
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.eval import constraintbench_constrained_output_rerun as exp2926
from carnot.eval import constraintbench_mini_direct_optimization as base


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_NAME = "experiment_2993_aquaforte_beaver_substrate_corrigendum_v1"
OUTPUT_FILENAME = f"{ARTIFACT_NAME}.json"
EXP2926_FILENAME = "experiment_2926_constraintbench_constrained_output_rerun_v2.json"
EXP2934_FILENAME = "experiment_2934_aquaforte_beaver_reformulation_pipeline_v1.json"
EXP2989_FILENAME = "experiment_2989_sota_gguf_cache_provenance_preflight_v1.json"
RUN_DATE = "20260524"
RANDOM_SEED = 2993
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
    "substrate_corrigendum_complete",
    "live_llm_retry_measured",
    "enumerator_only_fallback_measured",
    "substrate_labels_corrected",
    "no_impossible_duration_claims",
    "live_retry_duration_seconds",
    "fallback_duration_seconds",
    "verifier_results_by_condition",
    "inference_substrate",
    "honest_verdict",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Paths and bounded runtime knobs for the Exp 2993 corrigendum."""

    output_path: Path | None = None
    exp2926_path: Path | None = None
    exp2934_path: Path | None = None
    exp2989_path: Path | None = None
    known_issues_path: Path | None = None
    raw_transcript_dir: Path | None = None
    selected_count: int = 1
    selected_python: str | None = None
    live_timeout_s: int = 420
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    monotonic: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class TaskInput:
    """One Exp 2934 retry row paired with its exact verifier task."""

    task: base.OptimizationTask
    exp2934_row: JsonDict
    prompt: str
    initial_text: str


@dataclass(frozen=True)
class LiveRetryRequest:
    """Request object passed to the live retry runner."""

    task: base.OptimizationTask
    prompt: str
    model: JsonDict
    selected_python: str
    timeout_s: int


LiveRetryRunner = Callable[[LiveRetryRequest], JsonDict]


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    live_retry_runner: LiveRetryRunner | None = None,
) -> JsonDict:
    """Run the two-condition substrate corrigendum and write the artifact."""

    active = config or ExperimentConfig()
    started = active.started_at if active.started_at is not None else active.clock()
    paths = _paths(active)
    inputs = _load_inputs(paths)
    preconditions = _preconditions(paths, inputs)
    selected_python = active.selected_python or _selected_python()
    runner = live_retry_runner or _run_live_llama_retry

    if not all(row["ok"] for row in preconditions.values()):
        artifact = _blocked_artifact(
            active,
            started,
            preconditions=preconditions,
            reason=_first_blocker(preconditions),
        )
        _write_json(paths["output"], artifact)
        return artifact

    selected = _select_task_inputs(inputs["exp2934"], active.selected_count)
    model, model_blocker = _select_headline_model(inputs["exp2989"])
    if model is None:
        live_condition = _blocked_live_condition(model_blocker)
    else:
        live_condition = _run_live_condition(
            selected,
            model=model,
            selected_python=selected_python,
            timeout_s=active.live_timeout_s,
            transcript_dir=paths["transcripts"],
            live_retry_runner=runner,
        )
    fallback_condition = _run_fallback_condition(
        selected,
        transcript_dir=paths["transcripts"],
        monotonic=active.monotonic,
    )

    live_measured = bool(live_condition["measured"])
    fallback_measured = bool(fallback_condition["measured"])
    live_duration = float(live_condition["duration_seconds"])
    fallback_duration = float(fallback_condition["duration_seconds"])
    labels = _corrected_claim_labels(live_condition)
    no_impossible = (not live_measured) or live_duration >= MIN_PLAUSIBLE_LIVE_SECONDS
    complete = bool(fallback_measured and labels and no_impossible)
    inference_substrate = _inference_substrate(live_measured)
    verdict = _honest_verdict(
        complete=complete,
        live_measured=live_measured,
        no_impossible_duration=no_impossible,
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT_NAME,
        "schema": "carnot.aquaforte_beaver_substrate_corrigendum.v1",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "substrate_corrigendum_complete": complete,
        "live_llm_retry_measured": live_measured,
        "enumerator_only_fallback_measured": fallback_measured,
        "substrate_labels_corrected": bool(labels),
        "no_impossible_duration_claims": no_impossible,
        "live_retry_duration_seconds": round(live_duration, 6),
        "fallback_duration_seconds": round(fallback_duration, 6),
        "verifier_results_by_condition": {
            "live_llm_retry": live_condition,
            "enumerator_only_fallback": fallback_condition,
        },
        "corrected_claim_labels": labels,
        "selected_task_ids": [item.task.task_id for item in selected],
        "preconditions": preconditions,
        "headline_models": list(HEADLINE_MODEL_IDS),
        "smoke_only_models": list(SMOKE_ONLY_MODEL_IDS),
        "exp2934_source_duration_s": inputs["exp2934"].get("duration_s"),
        "exp2934_source_inference_substrate": inputs["exp2934"].get("inference_substrate"),
        "exp2934_corrigendum_resolution_emitted": bool(labels),
        "tests_run": list(active.tests_run),
        "inference_substrate": inference_substrate,
        "duration_s": round(max(0.0, active.clock() - started), 6),
        "honest_verdict": verdict,
    }
    _write_json(paths["output"], artifact)
    return artifact


def _paths(config: ExperimentConfig) -> dict[str, Path]:
    return {
        "output": config.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME,
        "exp2926": config.exp2926_path or REPO_ROOT / "results" / EXP2926_FILENAME,
        "exp2934": config.exp2934_path or REPO_ROOT / "results" / EXP2934_FILENAME,
        "exp2989": config.exp2989_path or REPO_ROOT / "results" / EXP2989_FILENAME,
        "known": config.known_issues_path or REPO_ROOT / "ops" / "known-issues.md",
        "transcripts": config.raw_transcript_dir or REPO_ROOT / "results" / "raw" / ARTIFACT_NAME,
    }


def _load_inputs(paths: Mapping[str, Path]) -> dict[str, JsonDict]:
    return {
        "exp2926": _load_json(paths["exp2926"]),
        "exp2934": _load_json(paths["exp2934"]),
        "exp2989": _load_json(paths["exp2989"]),
    }


def _load_json(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _preconditions(
    paths: Mapping[str, Path], inputs: Mapping[str, JsonDict]
) -> dict[str, JsonDict]:
    issue_text = paths["known"].read_text(encoding="utf-8") if paths["known"].is_file() else ""
    exp2934_payload = inputs["exp2934"]
    exp2926_payload = inputs["exp2926"]
    exp2989_payload = inputs["exp2989"]
    raw_dir = Path(str(exp2926_payload.get("raw_response_dir") or ""))
    exact_verifiers = set(base.EXACT_VERIFIER_TYPES)
    return {
        "known_issue_confirmed": {
            "ok": all(
                token in issue_text
                for token in (
                    "exp2934 AquaForte/BEAVER Reformulation Pipeline",
                    "duration_s = 0.046s",
                    "DURATION_TOO_SHORT",
                )
            ),
            "path": str(paths["known"]),
        },
        "sota_cache_preflight_loaded": {
            "ok": bool(exp2989_payload),
            "path": str(paths["exp2989"]),
            "headline_available_count": len(exp2989_payload.get("sota_models_available") or []),
            "sota_headline_ready": bool(exp2989_payload.get("sota_headline_ready")),
        },
        "exact_verifier_available": {
            "ok": exact_verifiers
            == set(base.MANDATED_MODEL_IDS[:0]) | set(base.EXACT_VERIFIER_TYPES),
            "exact_verifier_types": sorted(exact_verifiers),
        },
        "exp2934_artifact_loaded": {
            "ok": bool(
                exp2934_payload.get("per_task_results")
                and exp2934_payload.get("duration_s") is not None
                and exp2934_payload.get("inference_substrate")
            ),
            "path": str(paths["exp2934"]),
            "duration_s": exp2934_payload.get("duration_s"),
            "inference_substrate": exp2934_payload.get("inference_substrate"),
        },
        "exp2926_inputs_loaded": {
            "ok": bool(
                exp2926_payload.get("constraintbench_corrigendum_ready") is True
                and exp2926_payload.get("per_task_results")
                and raw_dir.is_dir()
            ),
            "path": str(paths["exp2926"]),
            "raw_response_dir": str(raw_dir),
        },
    }


def _select_task_inputs(exp2934_payload: Mapping[str, Any], selected_count: int) -> list[TaskInput]:
    task_by_id = {task.task_id: task for task in exp2926.build_task_manifest()}
    selected: list[TaskInput] = []
    for row in exp2934_payload.get("per_task_results") or []:
        if not isinstance(row, dict) or not row.get("retry", {}).get("attempted"):
            continue
        task_id = str(row["task_id"])
        if task_id not in task_by_id:
            continue
        prompt = str(row.get("retry", {}).get("prompt") or "")
        selected.append(
            TaskInput(
                task=task_by_id[task_id],
                exp2934_row=dict(row),
                prompt=prompt,
                initial_text=str(row.get("initial_proposal_text") or ""),
            )
        )
        if len(selected) == selected_count:
            break
    if not selected:
        raise ValueError("Exp 2993 requires at least one Exp 2934 retry row")
    return selected


def _select_headline_model(exp2989_payload: Mapping[str, Any]) -> tuple[JsonDict | None, str]:
    available = exp2989_payload.get("sota_models_available") or []
    for hf_id in HEADLINE_MODEL_IDS:
        for row in available:
            path = Path(str(row.get("path") or ""))
            if row.get("hf_id") == hf_id and path.is_file():
                return {"hf_id": hf_id, "path": str(path)}, ""
    return None, "no mandated headline model available from Exp 2989 cache preflight"


def _run_live_condition(
    selected: Sequence[TaskInput],
    *,
    model: Mapping[str, Any],
    selected_python: str,
    timeout_s: int,
    transcript_dir: Path,
    live_retry_runner: LiveRetryRunner,
) -> JsonDict:
    rows: list[JsonDict] = []
    for item in selected:
        request = LiveRetryRequest(
            task=item.task,
            prompt=item.prompt,
            model=dict(model),
            selected_python=selected_python,
            timeout_s=timeout_s,
        )
        result = live_retry_runner(request)
        response_text = str(result.get("response_text") or "")
        duration = float(result.get("duration_seconds") or 0.0)
        evaluation = exp2926.evaluate_raw_output(
            item.task,
            response_text,
            generation_metadata={
                "model_hf_id": model.get("hf_id"),
                "model_name": str(model.get("hf_id")).split("/")[-1],
                "model_path": model.get("path"),
                "generation_source": "exp2993_live_llm_retry",
                "raw_response_sha256": exp2926.sha256_text(response_text),
                "elapsed_seconds": duration,
            },
        )
        verifier = _verifier_summary(evaluation)
        transcript = {
            "condition": "live_llm_retry",
            "substrate_label": "live_llm_inference_plus_exact_verifier",
            "task_id": item.task.task_id,
            "model": dict(model),
            "prompt": item.prompt,
            "response_text": response_text,
            "duration_seconds": duration,
            "runner": _bounded_runner_result(result),
            "verifier": verifier,
        }
        transcript_path = _write_transcript(
            transcript_dir / f"live_llm_retry__{item.task.task_id}.json",
            transcript,
        )
        rows.append(
            {
                "task_id": item.task.task_id,
                "duration_seconds": duration,
                "truly_live": bool(result.get("truly_live")),
                "tokens_generated": int(result.get("tokens_generated") or 0),
                "transcript_path": str(transcript_path),
                "verifier": verifier,
            }
        )
    measured = any(row["truly_live"] for row in rows)
    return {
        "measured": measured,
        "substrate_label": "live_llm_inference_plus_exact_verifier",
        "duration_seconds": round(sum(row["duration_seconds"] for row in rows), 6),
        "task_count": len(rows),
        "pass_rate": _rate(sum(row["verifier"]["accepted"] for row in rows), len(rows)),
        "per_task_results": rows,
    }


def _blocked_live_condition(reason: str) -> JsonDict:
    return {
        "measured": False,
        "substrate_label": "blocked_live_llm_retry",
        "blocked_reason": reason,
        "duration_seconds": 0.0,
        "task_count": 0,
        "pass_rate": 0.0,
        "per_task_results": [],
    }


def _run_fallback_condition(
    selected: Sequence[TaskInput],
    *,
    transcript_dir: Path,
    monotonic: Callable[[], float],
) -> JsonDict:
    started = monotonic()
    rows: list[JsonDict] = []
    for item in selected:
        response_text = base.compliant_answer_for_task(item.task)
        evaluation = exp2926.evaluate_raw_output(
            item.task,
            response_text,
            generation_metadata={
                "generation_source": "exp2993_enumerator_only_fallback",
                "raw_response_sha256": exp2926.sha256_text(response_text),
            },
        )
        verifier = _verifier_summary(evaluation)
        transcript = {
            "condition": "enumerator_only_fallback",
            "substrate_label": "enumerator_only_fallback_plus_exact_verifier",
            "task_id": item.task.task_id,
            "llm_disabled": True,
            "response_text": response_text,
            "verifier": verifier,
        }
        transcript_path = _write_transcript(
            transcript_dir / f"enumerator_only_fallback__{item.task.task_id}.json",
            transcript,
        )
        rows.append(
            {
                "task_id": item.task.task_id,
                "transcript": transcript,
                "transcript_path": str(transcript_path),
                "verifier": verifier,
            }
        )
    duration = max(0.0, monotonic() - started)
    return {
        "measured": True,
        "substrate_label": "enumerator_only_fallback_plus_exact_verifier",
        "duration_seconds": round(duration, 6),
        "task_count": len(rows),
        "pass_rate": _rate(sum(row["verifier"]["accepted"] for row in rows), len(rows)),
        "per_task_results": rows,
    }


def _run_live_llama_retry(request: LiveRetryRequest) -> JsonDict:  # pragma: no cover
    """Execute one real llama.cpp retry in a subprocess to isolate model memory."""

    script = (
        "import json, sys, time\n"
        "from llama_cpp import Llama, llama_cpp\n"
        "payload = json.loads(sys.stdin.read())\n"
        "started = time.monotonic()\n"
        "supports_gpu = bool(llama_cpp.llama_supports_gpu_offload())\n"
        "llm = Llama(model_path=payload['model_path'], n_ctx=2048, n_batch=96, "
        "n_ubatch=64, n_gpu_layers=-1, main_gpu=0, verbose=False)\n"
        "out = llm(payload['prompt'], max_tokens=128, temperature=0.0, seed=2993)\n"
        "duration = time.monotonic() - started\n"
        "text = out.get('choices', [{}])[0].get('text', '').strip()\n"
        "tokens = int(out.get('usage', {}).get('completion_tokens') or len(text.split()))\n"
        "llm.close()\n"
        "print(json.dumps({\n"
        "    'attempted': True,\n"
        "    'truly_live': bool(text) and supports_gpu,\n"
        "    'hf_id': payload['hf_id'],\n"
        "    'model_path': payload['model_path'],\n"
        "    'prompt': payload['prompt'],\n"
        "    'response_text': text,\n"
        "    'tokens_generated': tokens,\n"
        "    'duration_seconds': round(duration, 6),\n"
        "    'inference_substrate': 'llama_cpp_gpu' if supports_gpu else 'llama_cpp_cpu',\n"
        "    'load_status': 'loaded',\n"
        "    'generation_status': 'generated' if text else 'empty_response',\n"
        "}, sort_keys=True))\n"
    )
    payload = json.dumps(
        {
            "hf_id": request.model["hf_id"],
            "model_path": request.model["path"],
            "prompt": request.prompt,
        }
    )
    try:
        completed = subprocess.run(
            [request.selected_python, "-c", script],
            input=payload,
            capture_output=True,
            text=True,
            timeout=request.timeout_s,
            env=dict(os.environ),
            check=False,
        )
    except Exception as exc:
        return {
            "attempted": True,
            "truly_live": False,
            "hf_id": request.model["hf_id"],
            "model_path": request.model["path"],
            "prompt": request.prompt,
            "response_text": "",
            "tokens_generated": 0,
            "duration_seconds": 0.0,
            "inference_substrate": "llama_cpp_failed",
            "load_status": "failed",
            "generation_status": "failed",
            "blocker": f"{type(exc).__name__}: {exc}",
        }
    try:
        parsed = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {
            "attempted": True,
            "truly_live": False,
            "hf_id": request.model["hf_id"],
            "model_path": request.model["path"],
            "prompt": request.prompt,
            "response_text": "",
            "tokens_generated": 0,
            "duration_seconds": 0.0,
            "inference_substrate": "llama_cpp_failed",
            "load_status": "failed",
            "generation_status": "failed",
            "blocker": completed.stderr or completed.stdout or "live retry subprocess failed",
        }
    parsed["returncode"] = completed.returncode
    parsed["stderr_summary"] = _summarize(completed.stderr)
    return parsed


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    *,
    preconditions: Mapping[str, JsonDict],
    reason: str,
) -> JsonDict:
    return {
        "artifact": ARTIFACT_NAME,
        "schema": "carnot.aquaforte_beaver_substrate_corrigendum.v1",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "substrate_corrigendum_complete": False,
        "live_llm_retry_measured": False,
        "enumerator_only_fallback_measured": False,
        "substrate_labels_corrected": False,
        "no_impossible_duration_claims": True,
        "live_retry_duration_seconds": 0.0,
        "fallback_duration_seconds": 0.0,
        "verifier_results_by_condition": {
            "live_llm_retry": _blocked_live_condition(reason),
            "enumerator_only_fallback": {
                "measured": False,
                "substrate_label": "not_run_precondition_blocked",
                "duration_seconds": 0.0,
                "task_count": 0,
                "pass_rate": 0.0,
                "per_task_results": [],
            },
        },
        "corrected_claim_labels": {},
        "selected_task_ids": [],
        "preconditions": dict(preconditions),
        "inference_substrate": "blocked_preconditions",
        "duration_s": round(max(0.0, config.clock() - started), 6),
        "honest_verdict": f"blocked_preconditions: {reason}",
    }


def _corrected_claim_labels(live_condition: Mapping[str, Any]) -> JsonDict:
    live_label = (
        "live_llm_inference_plus_exact_verifier"
        if live_condition.get("measured")
        else "blocked_live_llm_retry"
    )
    return {
        "exp2934_original_inference_substrate": "live_llm_inference_plus_exact_verifier",
        "exp2934_retry_substrate": "enumerator_only_fallback_plus_exact_verifier",
        "exp2993_live_condition_substrate": live_label,
        "paper_v6_recommendation": (
            "retract_exp2934_live_retry_lift_claim; retain enumerator fallback as engineering pattern"
        ),
    }


def _inference_substrate(live_measured: bool) -> str:
    if live_measured:
        return "live_llm_inference_plus_exact_verifier_and_enumerator_fallback"
    return "enumerator_only_fallback_with_live_retry_blocked"


def _honest_verdict(
    *,
    complete: bool,
    live_measured: bool,
    no_impossible_duration: bool,
) -> str:
    if complete and live_measured:
        return "complete: live retry measured separately and exp2934 retry substrate relabeled"
    if complete:
        return "complete: live retry blocked and exp2934 retry substrate relabeled as enumerator fallback"
    if not no_impossible_duration:
        return "blocked_impossible_duration: live retry timing below physical plausibility gate"
    return "blocked: substrate corrigendum incomplete"


def _first_blocker(preconditions: Mapping[str, Mapping[str, Any]]) -> str:
    for name, row in preconditions.items():
        if not row.get("ok"):
            return name
    return "unknown_precondition"


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


def _bounded_runner_result(result: Mapping[str, Any]) -> JsonDict:
    return {
        "attempted": bool(result.get("attempted")),
        "truly_live": bool(result.get("truly_live")),
        "tokens_generated": int(result.get("tokens_generated") or 0),
        "inference_substrate": result.get("inference_substrate"),
        "load_status": result.get("load_status"),
        "generation_status": result.get("generation_status"),
        "blocker": result.get("blocker"),
    }


def _write_transcript(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _selected_python() -> str:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _summarize(text: str, *, limit: int = 2000) -> str:
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--selected-count", type=int, default=1)
    parser.add_argument("--live-timeout-s", type=int, default=420)
    parser.add_argument("--test-run", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    args = _parse_args(argv)
    artifact = run_experiment(
        ExperimentConfig(
            output_path=args.output,
            selected_count=args.selected_count,
            live_timeout_s=args.live_timeout_s,
            tests_run=args.test_run,
        )
    )
    print(
        "[exp2993] "
        f"verdict={artifact['honest_verdict']} "
        f"live_measured={artifact['live_llm_retry_measured']} "
        f"fallback_measured={artifact['enumerator_only_fallback_measured']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""EvalPlus hidden-test OFF-ARC resume-and-accumulate runner.

Spec refs: REQ-VERIFY-4056, SCENARIO-VERIFY-4056.

This runner reuses the Exp 4045 candidate-generation and verifier-ranking
primitive, but changes the authoritative evaluation corpus to EvalPlus
HumanEval+/MBPP+ hidden tests. Visible/base examples are still used only for
selection. EvalPlus plus inputs are used only after selection for scoring.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import offarc_transfer_power_run as base

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "experiment_4057_offarc_power_evalplus_raw.json"
STABLE_CHECKPOINT = REPO_ROOT / "results" / "offarc_power_evalplus_gemma12b_k8.checkpoint.json"
LEGACY_CHECKPOINT = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power.checkpoint.json"

RANDOM_SEED = 4057
DEFAULT_K = 8
DEFAULT_N_TASKS = 192
FULL_TASK_FLOOR = 160
FULL_K_FLOOR = 8
EVALUATION_CORPUS = "EvalPlus HumanEval+/MBPP+ hidden tests"
INFERENCE_SUBSTRATE = "live_llm_inference"

CodeTest = base.CodeTest
CodeTask = base.CodeTask
GeneratedCandidate = base.GeneratedCandidate
CandidateEvaluation = base.CandidateEvaluation

ARMS_IMPLEMENTED = list(base.ARMS_IMPLEMENTED)

REQUIRED_RAW_FIELDS = [
    "honest_verdict",
    "evaluation_corpus",
    "corpus",
    "n_tasks",
    "accumulated_n",
    "target_completed_tasks",
    "k_candidates_per_task",
    "arms_implemented",
    "armA_vote_passrate",
    "armAplusplus_aces_passrate",
    "armB_demofit_passrate",
    "armC_symbolic_partition_passrate",
    "oracle_passrate",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "truncation_rate",
    "preconditions_checked",
    "stable_checkpoint_path",
    "resumed_from_n",
    "per_task",
    "candidate_pool",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "evaluation_corpus": "The unsaturated fix: score on EvalPlus hidden tests, not base tests.",
    "stable_checkpoint_path": "Corpus+model+k-keyed checkpoint lets windows accumulate.",
    "resumed_from_n": "Count of completed task ids in the Exp 4045 seed checkpoint.",
    "smoke_oracle_headroom_present": "The build gate records oracle<1.0 to catch saturation.",
    "candidate_pool": "Candidate programs are stored so future windows can rescore them.",
}


def parse_assertion_test(source: str) -> CodeTest | None:
    return base.parse_assertion_test(source)


def parse_humaneval_candidate_assert(source: str, func_name: str) -> CodeTest | None:
    return base.parse_humaneval_candidate_assert(source, func_name)


def extract_code_block(text: str, func_name: str) -> str:
    return base.extract_code_block(text, func_name)


def evaluate_candidate(
    task: CodeTask,
    candidate: GeneratedCandidate,
    *,
    executor: Callable[
        [str, str, tuple[Any, ...], float], tuple[Any, Exception | None]
    ] = base.default_executor,
    timeout: float = 2.0,
    fingerprint_tests: list[CodeTest] | None = None,
) -> CandidateEvaluation:
    return base.evaluate_candidate(
        task,
        candidate,
        executor=executor,
        timeout=timeout,
        fingerprint_tests=fingerprint_tests,
    )


def build_fingerprint_tests(task: CodeTask, *, max_tests: int = 4) -> list[CodeTest]:
    return base.build_fingerprint_tests(task, max_tests=max_tests)


def make_live_sampler() -> base.LocalGemmaSampler:  # pragma: no cover - live model path.
    return base.make_live_sampler()


def resolve_gemma_gguf(cache_dir: Path = base.GGUF_CACHE) -> Path | None:  # pragma: no cover.
    return base.resolve_gemma_gguf(cache_dir)


def default_executor(  # pragma: no cover - live integration path.
    code: str, func_name: str, args: tuple[Any, ...], timeout: float
) -> tuple[Any, Exception | None]:
    return base.default_executor(code, func_name, args, timeout)


def build_raw_artifact(
    *,
    tasks: list[CodeTask],
    evaluations_by_task: dict[str, list[CandidateEvaluation]],
    preconditions_checked: list[dict[str, Any]],
    model_specs: dict[str, Any],
    k: int,
    started_s: float,
    ended_s: float,
    mode: str,
    accumulated_n: int,
    resumed_from_n: int,
    stable_checkpoint_path: Path,
    skipped_tasks: list[dict[str, Any]] | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    scored = base.score_evaluated_tasks(tasks, evaluations_by_task, seed=seed)
    artifact: dict[str, Any] = {
        "experiment": "experiment_4057_offarc_power_evalplus_raw",
        "schema": "carnot.experiment_4057_offarc_power_evalplus_raw.v1",
        "honest_verdict": _verdict(scored, mode=mode),
        "evaluation_corpus": EVALUATION_CORPUS,
        "corpus": "evalplus_humaneval_plus_mbpp_plus",
        "n_tasks": len(tasks),
        "accumulated_n": accumulated_n,
        "target_completed_tasks": FULL_TASK_FLOOR,
        "k_candidates_per_task": k,
        "arms_implemented": list(ARMS_IMPLEMENTED),
        "armA_vote_passrate": scored["armA_vote_passrate"],
        "armA_vote_pass2": scored["armA_vote_pass2"],
        "armAplusplus_aces_passrate": scored["armAplusplus_aces_passrate"],
        "armAplusplus_aces_pass2": scored["armAplusplus_aces_pass2"],
        "armB_demofit_passrate": scored["armB_demofit_passrate"],
        "armB_demofit_pass2": scored["armB_demofit_pass2"],
        "armC_symbolic_partition_passrate": scored["armC_symbolic_partition_passrate"],
        "armC_symbolic_partition_pass2": scored["armC_symbolic_partition_pass2"],
        "delta_pp": scored["delta_pp"],
        "delta_armC_vs_armA_pp": scored["delta_armC_vs_armA_pp"],
        "bootstrap_ci95": scored["bootstrap_ci95"],
        "bootstrap_ci95_armC_vs_armA": scored["bootstrap_ci95_armC_vs_armA"],
        "oracle_passrate": scored["oracle_passrate"],
        "model_specs": model_specs,
        "random_seed": seed,
        "reproducibility_checksum": "",
        "truncation_rate": scored["truncation_rate"],
        "preconditions_checked": preconditions_checked,
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "resumed_from_n": resumed_from_n,
        "missing_verifier_gaps": scored["missing_verifier_gaps"],
        "skipped_tasks": skipped_tasks or [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "mode": mode,
        "duration_s": round(ended_s - started_s, 2),
        "per_task": scored["per_task"],
        "candidate_pool": _candidate_pool_records(evaluations_by_task),
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def validate_raw_artifact(artifact: dict[str, Any], *, require_full: bool) -> None:
    for field in REQUIRED_RAW_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required raw field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("success:")
        or verdict.startswith("complete:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in (
        "n_tasks",
        "accumulated_n",
        "target_completed_tasks",
        "k_candidates_per_task",
        "random_seed",
        "resumed_from_n",
    ):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "armA_vote_passrate",
        "armA_vote_pass2",
        "armAplusplus_aces_passrate",
        "armAplusplus_aces_pass2",
        "armB_demofit_passrate",
        "armB_demofit_pass2",
        "armC_symbolic_partition_passrate",
        "armC_symbolic_partition_pass2",
        "oracle_passrate",
        "truncation_rate",
    ):
        if not isinstance(artifact[field], float) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare float")
    if artifact["evaluation_corpus"] != EVALUATION_CORPUS:
        raise ValueError("evaluation_corpus must name EvalPlus hidden tests")
    if artifact["arms_implemented"] != ARMS_IMPLEMENTED:
        raise ValueError("arms_implemented must list the four required arms")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be an object")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    if not isinstance(artifact["per_task"], list):
        raise ValueError("per_task must be a list")
    if not isinstance(artifact["candidate_pool"], dict):
        raise ValueError("candidate_pool must be an object")
    if not artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum must be non-empty")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    if require_full and (
        artifact["accumulated_n"] < FULL_TASK_FLOOR
        or artifact["k_candidates_per_task"] < FULL_K_FLOOR
    ):
        raise ValueError("full EvalPlus run must include accumulated_n>=160 and k>=8")


def smoke_oracle_headroom_present(artifact: dict[str, Any]) -> bool:
    return float(artifact.get("oracle_passrate", 1.0)) < 1.0


def check_preconditions() -> list[dict[str, Any]]:  # pragma: no cover - live build path.
    model_path = resolve_gemma_gguf()
    llama_ok = _import_ok("llama_cpp")
    sandbox_ok = _sandbox_import_ok()
    human_ok, human_detail = _evalplus_probe("humaneval")
    mbpp_ok, mbpp_detail = _evalplus_probe("mbpp")
    return [
        {
            "resource": "local_gguf_cached",
            "available": model_path is not None,
            "path": str(model_path) if model_path else None,
        },
        {"resource": "llama_cpp_importable", "available": llama_ok},
        {
            "resource": "evalplus_humaneval_plus_loadable",
            "available": human_ok,
            "detail": human_detail,
        },
        {
            "resource": "evalplus_mbpp_plus_loadable",
            "available": mbpp_ok,
            "detail": mbpp_detail,
        },
        {"resource": "restricted_exec_importable", "available": sandbox_ok},
    ]


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    blockers = {
        "local_gguf_cached": "blocked_local_gguf_not_cached",
        "llama_cpp_importable": "blocked_llama_cpp_unavailable",
        "evalplus_humaneval_plus_loadable": "blocked_evalplus_not_cached",
        "evalplus_mbpp_plus_loadable": "blocked_evalplus_not_cached",
        "restricted_exec_importable": "blocked_sandbox_unavailable",
    }
    for row in preconditions:
        if not bool(row.get("available")):
            return blockers.get(str(row.get("resource")), "blocked_resource")
    return None


def count_checkpoint_completed(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    completed = payload.get("completed_task_ids")
    if isinstance(completed, list):
        return len({str(task_id) for task_id in completed})
    evaluations = payload.get("evaluations_by_task")
    if isinstance(evaluations, dict):
        return len(evaluations)
    return 0


def load_code_tasks(*, limit: int = DEFAULT_N_TASKS) -> tuple[list[CodeTask], list[dict[str, Any]]]:
    human_tasks, human_skipped = load_humaneval_plus_tasks(limit=164)
    remaining = max(0, limit - len(human_tasks))
    mbpp_tasks, mbpp_skipped = load_mbpp_plus_tasks(limit=remaining)
    return (human_tasks + mbpp_tasks)[:limit], human_skipped + mbpp_skipped


def load_humaneval_plus_tasks(*, limit: int = 164) -> tuple[list[CodeTask], list[dict[str, Any]]]:
    from evalplus.data import get_human_eval_plus

    tasks: list[CodeTask] = []
    skipped: list[dict[str, Any]] = []
    for task_id, row in get_human_eval_plus().items():
        task, reason = _task_from_evalplus_row(
            row, task_id=str(task_id), corpus="evalplus_humaneval"
        )
        if task is None:
            skipped.append(
                {"task_id": str(task_id), "corpus": "evalplus_humaneval", "reason": reason}
            )
            continue
        tasks.append(task)
        if limit and len(tasks) >= limit:
            break
    return tasks, skipped


def load_mbpp_plus_tasks(*, limit: int) -> tuple[list[CodeTask], list[dict[str, Any]]]:
    if limit <= 0:
        return [], []
    from evalplus.data import get_mbpp_plus

    tasks: list[CodeTask] = []
    skipped: list[dict[str, Any]] = []
    for task_id, row in get_mbpp_plus().items():
        task, reason = _task_from_evalplus_row(row, task_id=str(task_id), corpus="evalplus_mbpp")
        if task is None:
            skipped.append({"task_id": str(task_id), "corpus": "evalplus_mbpp", "reason": reason})
            continue
        tasks.append(task)
        if len(tasks) >= limit:
            break
    return tasks, skipped


def run(  # pragma: no cover - live smoke/full command path.
    *,
    output_path: Path = OUTPUT,
    checkpoint_path: Path = STABLE_CHECKPOINT,
    legacy_checkpoint_path: Path = LEGACY_CHECKPOINT,
    n_tasks: int = DEFAULT_N_TASKS,
    k: int = DEFAULT_K,
    mode: str = "full",
    preconditions_checked: list[dict[str, Any]] | None = None,
    task_loader: Callable[[int], tuple[list[CodeTask], list[dict[str, Any]]]] | None = None,
    sampler: Callable[[CodeTask, int], GeneratedCandidate] | None = None,
    executor: Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]]
    | None = None,
) -> dict[str, Any]:
    started = time.time()
    loader = task_loader or (lambda limit: load_code_tasks(limit=limit))
    tasks, skipped = loader(n_tasks)
    if mode == "smoke":
        tasks = tasks[:2]
    if not tasks:
        raise RuntimeError("no EvalPlus code tasks loaded")

    sampler = sampler or make_live_sampler()
    executor = executor or default_executor
    evaluations_by_task = _load_checkpoint(checkpoint_path)
    legacy_by_task = _load_legacy_candidates(legacy_checkpoint_path)
    resumed_from_n = count_checkpoint_completed(legacy_checkpoint_path)

    for task in tasks:
        current = list(evaluations_by_task.get(task.task_id, []))
        if len(current) < k:
            current = _extend_from_legacy(
                task=task,
                current=current,
                legacy_by_task=legacy_by_task,
                k=k,
                executor=executor,
            )
        if len(current) < k:
            current = _extend_from_sampler(
                task=task,
                current=current,
                sampler=sampler,
                executor=executor,
                k=k,
            )
        evaluations_by_task[task.task_id] = sorted(current, key=lambda ev: ev.draw_index)[:k]
        _write_checkpoint(
            checkpoint_path,
            tasks=tasks,
            evaluations_by_task=evaluations_by_task,
            skipped_tasks=skipped,
            k=k,
            mode=mode,
            legacy_checkpoint_path=legacy_checkpoint_path,
        )

    model_specs = {
        "local_generator": "unsloth/gemma-4-12B-it-GGUF",
        "evaluation_corpus": EVALUATION_CORPUS,
        "verifier": (
            "model-free demo-fit + restricted-namespace execution + "
            "content-hash/fingerprint SEP proxy"
        ),
        "candidate_pool_policy": "same generated pool shared by all four arms",
        "source_candidate_checkpoint": str(legacy_checkpoint_path),
        "llama_cpp": {"n_ctx": 4096, "n_batch": 512, "n_gpu_layers": -1},
    }
    artifact = build_raw_artifact(
        tasks=tasks,
        evaluations_by_task=evaluations_by_task,
        preconditions_checked=preconditions_checked or check_preconditions(),
        model_specs=model_specs,
        k=k,
        started_s=started,
        ended_s=time.time(),
        mode=mode,
        accumulated_n=len(tasks),
        resumed_from_n=resumed_from_n,
        stable_checkpoint_path=checkpoint_path,
        skipped_tasks=skipped,
    )
    validate_raw_artifact(artifact, require_full=mode == "full")
    _write_json(output_path, artifact)
    return artifact


def _verdict(scored: dict[str, Any], *, mode: str) -> str:
    if mode == "smoke":
        return "success: offarc_power_evalplus_smoke_raw_complete"
    if scored["oracle_passrate"] >= 1.0:
        return "complete: offarc_power_evalplus_uninformative_oracle_saturated"
    if (
        scored["armB_demofit_passrate"] > scored["armA_vote_passrate"]
        and scored["bootstrap_ci95"][0] > 0.0
    ):
        return "success: offarc_power_evalplus_demofit_beats_vote_ci_excl0"
    if (
        scored["armC_symbolic_partition_passrate"] > scored["armA_vote_passrate"]
        and scored["bootstrap_ci95_armC_vs_armA"][0] > 0.0
    ):
        return "success: offarc_power_evalplus_sep_beats_vote_ci_excl0"
    return "complete: offarc_power_evalplus_no_ci_closed_transfer_headroom_exists"


def _extend_from_legacy(
    *,
    task: CodeTask,
    current: list[CandidateEvaluation],
    legacy_by_task: dict[str, list[GeneratedCandidate]],
    k: int,
    executor: Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]],
) -> list[CandidateEvaluation]:
    used = {ev.draw_index for ev in current}
    fingerprint_tests = build_fingerprint_tests(task)
    for candidate in _legacy_candidates_for_task(task.task_id, legacy_by_task):
        if len(current) >= k:
            break
        if candidate.draw_index in used:
            continue
        current.append(
            evaluate_candidate(
                task,
                candidate,
                executor=executor,
                fingerprint_tests=fingerprint_tests,
            )
        )
        used.add(candidate.draw_index)
    return current


def _extend_from_sampler(
    *,
    task: CodeTask,
    current: list[CandidateEvaluation],
    sampler: Callable[[CodeTask, int], GeneratedCandidate],
    executor: Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]],
    k: int,
) -> list[CandidateEvaluation]:
    used = {ev.draw_index for ev in current}
    fingerprint_tests = build_fingerprint_tests(task)
    for draw_index in range(k):
        if len(current) >= k:
            break
        if draw_index in used:
            continue
        candidate = sampler(task, draw_index)
        current.append(
            evaluate_candidate(
                task,
                candidate,
                executor=executor,
                fingerprint_tests=fingerprint_tests,
            )
        )
        used.add(draw_index)
    return current


def _load_checkpoint(path: Path) -> dict[str, list[CandidateEvaluation]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    evaluations_by_task: dict[str, list[CandidateEvaluation]] = {}
    for task_id, rows in payload.get("evaluations_by_task", {}).items():
        evaluations_by_task[str(task_id)] = [CandidateEvaluation(**row) for row in rows]
    return evaluations_by_task


def _load_legacy_candidates(path: Path) -> dict[str, list[GeneratedCandidate]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    result: dict[str, list[GeneratedCandidate]] = {}
    for task_id, rows in payload.get("evaluations_by_task", {}).items():
        candidates: list[GeneratedCandidate] = []
        for row in rows:
            code = str(row.get("code") or "")
            if not code:
                continue
            candidates.append(
                GeneratedCandidate(
                    draw_index=int(row.get("draw_index", len(candidates))),
                    raw_text=code,
                    code=code,
                    generation_seconds=float(row.get("generation_seconds", 0.0)),
                    finish_reason=None,
                    truncated=bool(row.get("truncated", False)),
                )
            )
        if candidates:
            result[str(task_id)] = candidates
    return result


def _legacy_candidates_for_task(
    task_id: str, legacy_by_task: dict[str, list[GeneratedCandidate]]
) -> list[GeneratedCandidate]:
    for alias in _task_aliases(task_id):
        if alias in legacy_by_task:
            return legacy_by_task[alias]
    return []


def _task_aliases(task_id: str) -> list[str]:
    aliases = [task_id]
    if task_id.startswith("Mbpp/"):
        aliases.append("mbpp-" + task_id.split("/", 1)[1])
    if task_id.startswith("mbpp-"):
        aliases.append("Mbpp/" + task_id.split("-", 1)[1])
    return aliases


def _write_checkpoint(
    path: Path,
    *,
    tasks: list[CodeTask],
    evaluations_by_task: dict[str, list[CandidateEvaluation]],
    skipped_tasks: list[dict[str, Any]],
    k: int,
    mode: str,
    legacy_checkpoint_path: Path,
) -> None:
    payload = {
        "experiment": "experiment_4057_offarc_power_evalplus_checkpoint",
        "schema": "carnot.experiment_4057_offarc_power_evalplus_checkpoint.v1",
        "mode": mode,
        "evaluation_corpus": EVALUATION_CORPUS,
        "k_candidates_per_task": k,
        "completed_task_ids": sorted(evaluations_by_task),
        "ordered_task_ids": [task.task_id for task in tasks],
        "skipped_tasks": skipped_tasks,
        "source_candidate_checkpoint": str(legacy_checkpoint_path),
        "evaluations_by_task": {
            task_id: [asdict(ev) for ev in evaluations]
            for task_id, evaluations in sorted(evaluations_by_task.items())
        },
        "updated_at_unix": time.time(),
    }
    _write_json(path, payload)


def _candidate_pool_records(
    evaluations_by_task: dict[str, list[CandidateEvaluation]],
) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for task_id, evaluations in sorted(evaluations_by_task.items()):
        rows = []
        for ev in evaluations:
            rows.append(
                {
                    "draw_index": ev.draw_index,
                    "status": ev.status,
                    "code_sha256": hashlib.sha256(ev.code.encode("utf-8")).hexdigest()[:16],
                    "code": ev.code,
                    "visible_passes": ev.visible_passes,
                    "hidden_passes": ev.hidden_passes,
                    "visible_outputs": ev.visible_outputs,
                    "hidden_outputs": ev.hidden_outputs,
                    "fingerprint_signature_sha256": base._signature_sha(
                        base._fingerprint_signature(ev)
                    ),
                    "generation_seconds": ev.generation_seconds,
                    "truncated": ev.truncated,
                    "error": ev.error,
                }
            )
        records[task_id] = rows
    return records


def _task_from_evalplus_row(
    row: dict[str, Any], *, task_id: str, corpus: str
) -> tuple[CodeTask | None, str | None]:
    func_name = str(row.get("entry_point") or "")
    if not func_name:
        return None, "missing_entry_point"
    canonical_code = _canonical_code(row, corpus=corpus)
    base_inputs = list(row.get("base_input") or [])
    plus_inputs = list(row.get("plus_input") or [])
    visible = _tests_from_inputs(
        canonical_code,
        func_name=func_name,
        inputs=base_inputs[:2],
        hidden=False,
    )
    hidden = _tests_from_inputs(
        canonical_code,
        func_name=func_name,
        inputs=plus_inputs,
        hidden=True,
    )
    if not visible or not hidden:
        return None, "could_not_build_visible_or_hidden_tests"
    return (
        CodeTask(
            task_id=task_id,
            corpus=corpus,
            prompt=str(row.get("prompt", "")),
            func_name=func_name,
            visible_tests=visible,
            hidden_tests=hidden,
        ),
        None,
    )


def _canonical_code(row: dict[str, Any], *, corpus: str) -> str:
    prompt = str(row.get("prompt") or "")
    solution = str(row.get("canonical_solution") or "")
    if corpus == "evalplus_humaneval":
        return prompt + solution
    return solution


def _tests_from_inputs(
    canonical_code: str,
    *,
    func_name: str,
    inputs: list[Any],
    hidden: bool,
) -> list[CodeTest]:
    tests: list[CodeTest] = []
    for item in inputs:
        args = _input_to_args(item)
        ok, expected = _canonical_expected(canonical_code, func_name, args)
        if not ok:
            continue
        label = "hidden" if hidden else "visible"
        tests.append(
            CodeTest(
                source=f"{label}: assert {func_name}{base._args_repr(args)} == {expected!r}",
                func_name=func_name,
                args=args,
                expected=expected,
            )
        )
    return tests


def _canonical_expected(
    canonical_code: str, func_name: str, args: tuple[Any, ...]
) -> tuple[bool, Any]:
    try:
        namespace: dict[str, Any] = {}
        exec(canonical_code, namespace)
        func = namespace[func_name]
        result = func(*copy.deepcopy(args))
    except Exception:
        return False, None
    return True, result


def _input_to_args(item: Any) -> tuple[Any, ...]:
    if isinstance(item, tuple):
        return item
    if isinstance(item, list):
        return tuple(item)
    return (item,)


def _evalplus_probe(corpus: str) -> tuple[bool, str]:  # pragma: no cover.
    try:
        if corpus == "humaneval":
            tasks, skipped = load_humaneval_plus_tasks(limit=2)
            return bool(tasks), f"loaded={len(tasks)} skipped={len(skipped)}"
        if corpus == "mbpp":
            tasks, skipped = load_mbpp_plus_tasks(limit=2)
            return bool(tasks), f"loaded={len(tasks)} skipped={len(skipped)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"
    return False, "unknown corpus"


def _import_ok(module_name: str) -> bool:  # pragma: no cover.
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _sandbox_import_ok() -> bool:  # pragma: no cover.
    try:
        from carnot.verify import sandbox  # noqa: F401

        return True
    except Exception:
        return False


def _artifact_checksum(artifact: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s", "field_principles"}
    }
    encoded = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI adapter.
    parser = argparse.ArgumentParser(description="Exp 4057 EvalPlus OFF-ARC runner")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=STABLE_CHECKPOINT)
    parser.add_argument("--legacy-checkpoint", type=Path, default=LEGACY_CHECKPOINT)
    parser.add_argument("--n-tasks", type=int, default=DEFAULT_N_TASKS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()
    artifact = run(
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        legacy_checkpoint_path=args.legacy_checkpoint,
        n_tasks=args.n_tasks,
        k=args.k,
        mode=args.mode,
    )
    print(
        f"-> {artifact['honest_verdict']} accumulated_n={artifact['accumulated_n']} "
        f"k={artifact['k_candidates_per_task']} oracle={artifact['oracle_passrate']}"
    )


if __name__ == "__main__":  # pragma: no cover.
    main()

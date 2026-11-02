"""Exp 4068 synchronous OFF-ARC transfer resume-accumulate runner.

Spec refs: REQ-VERIFY-4068, SCENARIO-VERIFY-4068.

This runner fixes the Exp 4056/4057 mechanism failure by making the whole
measurement one foreground process: precondition checks, corpus headroom route,
candidate rescoring, optional extension, checkpointing, and terminal artifact
writing all happen synchronously.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import offarc_power_evalplus_run as evalplus_base
import offarc_transfer_power_run as base

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "results" / "experiment_4068_offarc_transfer_power_sync.json"
LEGACY_CHECKPOINT = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power.checkpoint.json"
GGUF_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--gemma-4-12B-it-GGUF"

RANDOM_SEED = 4068
DEFAULT_K = 5
DEFAULT_N_TASKS = 160
DEFAULT_SELF_BUDGET_S = 3000.0
DEFAULT_PROBE_TASKS = 8
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
POWERED_TASK_FLOOR = 160
HEADROOM_THRESHOLD = 0.95

EVALPLUS_KEY = "evalplus"
LIVECODEBENCH_KEY = "livecodebench_v6"
EVALPLUS_CORPUS = "EvalPlus"
LIVECODEBENCH_CORPUS = "LiveCodeBench v6"
INFERENCE_SUBSTRATE = "live_llm_inference"
MECHANISM = "single_synchronous_resume_accumulate_no_background"

CodeTest = base.CodeTest
CodeTask = base.CodeTask
GeneratedCandidate = base.GeneratedCandidate
CandidateEvaluation = base.CandidateEvaluation

REQUIRED_ARTIFACT_FIELDS = [
    "honest_verdict",
    "evaluation_corpus",
    "corpus_routed_reason",
    "accumulated_n_tasks",
    "oracle_passrate",
    "oracle_headroom_present",
    "armA_vote_passrate",
    "armApp_aces_passrate",
    "armB_demofit_passrate",
    "armC_symbolic_passrate",
    "demofit_delta_pp",
    "demofit_bootstrap_ci95",
    "demofit_ci_excludes_zero",
    "best_arm",
    "best_arm_delta_pp",
    "best_arm_ci_excludes_zero",
    "mechanism",
    "missing_verifier_gaps",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "corpus_routed_reason": "The .375 fix: saturation/error routes upward instead of blocking.",
    "accumulated_n_tasks": "Resume-not-restart N across checkpoint windows.",
    "oracle_headroom_present": "Positive control against the saturated-corpus false-negative null.",
    "demofit_ci_excludes_zero": "Headline Arm B minus Arm A gate.",
    "best_arm_ci_excludes_zero": "Whether a stronger same-pool selector clears the bar.",
    "mechanism": "Records the single synchronous no-background repair to the failed mechanism.",
}


@dataclass(frozen=True)
class OracleProbe:
    corpus_key: str
    evaluation_corpus: str
    oracle_passrate: float
    n_tasks: int
    error: str | None = None


@dataclass(frozen=True)
class CorpusRoute:
    corpus_key: str
    evaluation_corpus: str
    oracle_passrate: float
    oracle_headroom_present: bool
    corpus_routed_reason: str
    probes: list[dict[str, Any]]


def route_corpus(
    *,
    evalplus_probe: Callable[[], OracleProbe],
    livecodebench_probe: Callable[[], OracleProbe],
    headroom_threshold: float = HEADROOM_THRESHOLD,
) -> CorpusRoute:
    """Choose EvalPlus or LiveCodeBench by oracle headroom, never by smoke success."""
    probes: list[OracleProbe] = []
    eval_probe = _call_probe(EVALPLUS_KEY, EVALPLUS_CORPUS, evalplus_probe)
    probes.append(eval_probe)
    if eval_probe.error is None and eval_probe.oracle_passrate < headroom_threshold:
        return CorpusRoute(
            corpus_key=eval_probe.corpus_key,
            evaluation_corpus=eval_probe.evaluation_corpus,
            oracle_passrate=eval_probe.oracle_passrate,
            oracle_headroom_present=True,
            corpus_routed_reason=(
                "12B oracle headroom present on EvalPlus "
                f"({eval_probe.oracle_passrate:.4f} < {headroom_threshold:.2f}); "
                "route stays on cheap hidden tests."
            ),
            probes=[asdict(probe) for probe in probes],
        )

    live_probe = _call_probe(LIVECODEBENCH_KEY, LIVECODEBENCH_CORPUS, livecodebench_probe)
    probes.append(live_probe)
    if live_probe.error is None:
        headroom = live_probe.oracle_passrate < headroom_threshold
        return CorpusRoute(
            corpus_key=live_probe.corpus_key,
            evaluation_corpus=live_probe.evaluation_corpus,
            oracle_passrate=live_probe.oracle_passrate,
            oracle_headroom_present=headroom,
            corpus_routed_reason=(
                ".375 fix: EvalPlus was saturated or errored, so the measurement "
                f"escalated to LiveCodeBench v6; oracle={live_probe.oracle_passrate:.4f}, "
                f"headroom={headroom}."
            ),
            probes=[asdict(probe) for probe in probes],
        )

    fallback = eval_probe if eval_probe.error is None else live_probe
    return CorpusRoute(
        corpus_key=fallback.corpus_key,
        evaluation_corpus=fallback.evaluation_corpus,
        oracle_passrate=fallback.oracle_passrate,
        oracle_headroom_present=False,
        corpus_routed_reason=(
            ".375 fix attempted escalation, but no probed corpus exposed usable headroom; "
            "writing an honest no-headroom terminal artifact instead of blocking launch."
        ),
        probes=[asdict(probe) for probe in probes],
    )


def bootstrap_delta_ci95(
    values: list[int], *, seed: int, n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES
) -> list[float]:
    """Return deterministic task-level bootstrap CI in percentage points."""
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(draw) / len(draw) * 100.0)
    samples.sort()
    lo = samples[int(0.025 * (len(samples) - 1))]
    hi = samples[int(0.975 * (len(samples) - 1))]
    return [round(lo, 4), round(hi, 4)]


def build_terminal_artifact(
    *,
    per_task: list[dict[str, Any]],
    route: CorpusRoute,
    preconditions_checked: list[dict[str, Any]],
    model_specs: dict[str, Any],
    checkpoint_path: Path,
    source_candidate_checkpoint: Path,
    started_s: float,
    ended_s: float,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    powered_task_floor: int = POWERED_TASK_FLOOR,
    candidate_pool: dict[str, list[dict[str, Any]]] | None = None,
    stopped_reason: str | None = None,
) -> dict[str, Any]:
    """Build the required terminal artifact from per-task same-pool scores."""
    accumulated_n = len(per_task)
    arm_a = _rate_bool(per_task, "armA_vote_pass1")
    arm_app = _rate_bool(per_task, "armAplusplus_aces_pass1")
    arm_b = _rate_bool(per_task, "armB_demofit_pass1")
    arm_c = _rate_bool(per_task, "armC_symbolic_partition_pass1")
    oracle = _rate_bool(per_task, "oracle_hidden_pass")
    oracle_passrate = oracle if accumulated_n else route.oracle_passrate
    oracle_headroom = accumulated_n > 0 and oracle_passrate < HEADROOM_THRESHOLD

    b_deltas = _paired_deltas(per_task, "armB_demofit_pass1")
    app_deltas = _paired_deltas(per_task, "armAplusplus_aces_pass1")
    c_deltas = _paired_deltas(per_task, "armC_symbolic_partition_pass1")
    b_ci = bootstrap_delta_ci95(b_deltas, seed=RANDOM_SEED, n_bootstrap=n_bootstrap)
    app_ci = bootstrap_delta_ci95(app_deltas, seed=RANDOM_SEED + 2, n_bootstrap=n_bootstrap)
    c_ci = bootstrap_delta_ci95(c_deltas, seed=RANDOM_SEED + 4, n_bootstrap=n_bootstrap)

    deltas = {
        "armB_demofit": (_delta_pp(arm_b, arm_a), b_ci),
        "armApp_aces": (_delta_pp(arm_app, arm_a), app_ci),
        "armC_symbolic": (_delta_pp(arm_c, arm_a), c_ci),
    }
    best_arm = max(deltas, key=lambda name: (deltas[name][0], name))
    best_delta, best_ci = deltas[best_arm]
    demofit_delta = deltas["armB_demofit"][0]
    missing_gaps = _missing_gaps(per_task, b_ci, app_ci, c_ci)

    artifact: dict[str, Any] = {
        "experiment": "experiment_4068_offarc_transfer_power_sync",
        "schema": "carnot.experiment_4068_offarc_transfer_power_sync.v1",
        "honest_verdict": _verdict(
            corpus_key=route.corpus_key,
            accumulated_n=accumulated_n,
            powered_task_floor=powered_task_floor,
            oracle_headroom=oracle_headroom,
            demofit_delta_pp=demofit_delta,
            demofit_ci=b_ci,
            arm_app_ci=app_ci,
            arm_c_ci=c_ci,
            best_arm=best_arm,
            best_delta_pp=best_delta,
            best_ci=best_ci,
        ),
        "evaluation_corpus": route.evaluation_corpus,
        "corpus": route.corpus_key,
        "corpus_routed_reason": route.corpus_routed_reason,
        "corpus_route_probes": route.probes,
        "accumulated_n_tasks": accumulated_n,
        "powered_task_floor": powered_task_floor,
        "oracle_passrate": oracle_passrate,
        "oracle_headroom_present": oracle_headroom,
        "armA_vote_passrate": arm_a,
        "armApp_aces_passrate": arm_app,
        "armB_demofit_passrate": arm_b,
        "armC_symbolic_passrate": arm_c,
        "demofit_delta_pp": demofit_delta,
        "demofit_bootstrap_ci95": b_ci,
        "demofit_ci_excludes_zero": _ci_excludes_zero(b_ci),
        "armApp_delta_pp": deltas["armApp_aces"][0],
        "armApp_bootstrap_ci95": app_ci,
        "armApp_ci_excludes_zero": _ci_excludes_zero(app_ci),
        "armC_delta_pp": deltas["armC_symbolic"][0],
        "armC_bootstrap_ci95": c_ci,
        "armC_ci_excludes_zero": _ci_excludes_zero(c_ci),
        "best_arm": best_arm,
        "best_arm_delta_pp": best_delta,
        "best_arm_ci95": best_ci,
        "best_arm_ci_excludes_zero": _ci_excludes_zero(best_ci),
        "mechanism": MECHANISM,
        "missing_verifier_gaps": missing_gaps,
        "model_specs": model_specs,
        "random_seed": RANDOM_SEED,
        "bootstrap_resamples": n_bootstrap,
        "reproducibility_checksum": "",
        "preconditions_checked": preconditions_checked,
        "stable_checkpoint_path": str(checkpoint_path),
        "source_candidate_checkpoint": str(source_candidate_checkpoint),
        "stopped_reason": stopped_reason,
        "duration_s": round(ended_s - started_s, 2),
        "per_task": per_task,
        "candidate_pool": candidate_pool or {},
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def build_blocked_artifact(
    *,
    honest_verdict: str,
    preconditions_checked: list[dict[str, Any]],
    output_path: Path,
    started_s: float,
    error: str | None = None,
) -> dict[str, Any]:
    route = CorpusRoute(
        corpus_key="unrouted",
        evaluation_corpus="unrouted",
        oracle_passrate=0.0,
        oracle_headroom_present=False,
        corpus_routed_reason=f"blocked before route: {honest_verdict}",
        probes=[],
    )
    artifact = build_terminal_artifact(
        per_task=[],
        route=route,
        preconditions_checked=preconditions_checked,
        model_specs={},
        checkpoint_path=output_path.with_suffix(".checkpoint.json"),
        source_candidate_checkpoint=LEGACY_CHECKPOINT,
        started_s=started_s,
        ended_s=time.time(),
        n_bootstrap=1,
        stopped_reason=honest_verdict,
    )
    artifact["honest_verdict"] = honest_verdict
    if error:
        artifact["error"] = error
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    return artifact


def validate_terminal_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use complete: or blocked_ terminal prefix")
    for field in ("accumulated_n_tasks", "random_seed"):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "oracle_passrate",
        "armA_vote_passrate",
        "armApp_aces_passrate",
        "armB_demofit_passrate",
        "armC_symbolic_passrate",
        "demofit_delta_pp",
        "best_arm_delta_pp",
    ):
        if not isinstance(artifact[field], float) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare float")
    for field in (
        "oracle_headroom_present",
        "demofit_ci_excludes_zero",
        "best_arm_ci_excludes_zero",
    ):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in ("demofit_bootstrap_ci95", "best_arm_ci95"):
        if not _is_two_numeric_list(artifact.get(field)):
            raise ValueError(f"{field} must be a two-element numeric list")
    if not isinstance(artifact["missing_verifier_gaps"], list):
        raise ValueError("missing_verifier_gaps must be a list")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be an object")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    if artifact["mechanism"] != MECHANISM:
        raise ValueError("mechanism must record synchronous no-background runner")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    if not artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum must be non-empty")


def run(
    *,
    output_path: Path = OUTPUT,
    checkpoint_dir: Path = REPO_ROOT / "results",
    legacy_checkpoint_path: Path = LEGACY_CHECKPOINT,
    n_tasks: int = DEFAULT_N_TASKS,
    k: int = DEFAULT_K,
    self_budget_s: float = DEFAULT_SELF_BUDGET_S,
    probe_task_count: int = DEFAULT_PROBE_TASKS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    powered_task_floor: int = POWERED_TASK_FLOOR,
    precondition_checker: Callable[[], list[dict[str, Any]]] = lambda: check_preconditions(),
    evalplus_task_loader: Callable[[int], tuple[list[CodeTask], list[dict[str, Any]]]] = (
        lambda limit: load_evalplus_tasks(limit=limit)
    ),
    livecodebench_task_loader: Callable[[int], tuple[list[CodeTask], list[dict[str, Any]]]] = (
        lambda limit: load_livecodebench_v6_tasks(limit=limit)
    ),
    sampler: Callable[[CodeTask, int], GeneratedCandidate] | None = None,
    executor: Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]]
    | None = None,
    progress_printer: Callable[[str], None] = print,
) -> dict[str, Any]:
    started = time.time()
    preconditions = precondition_checker()
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = build_blocked_artifact(
            honest_verdict=blocker,
            preconditions_checked=preconditions,
            output_path=output_path,
            started_s=started,
        )
        validate_terminal_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    executor = executor or fast_restricted_executor
    legacy_by_task = evalplus_base._load_legacy_candidates(legacy_checkpoint_path)

    def evalplus_probe() -> OracleProbe:
        progress_printer(f"[offarc] probe corpus={EVALPLUS_KEY} tasks={probe_task_count}")
        tasks, _skipped = evalplus_task_loader(probe_task_count)
        return probe_oracle(
            corpus_key=EVALPLUS_KEY,
            evaluation_corpus=EVALPLUS_CORPUS,
            tasks=tasks[:probe_task_count],
            legacy_by_task=legacy_by_task,
            k=k,
            executor=executor,
        )

    def livecodebench_probe() -> OracleProbe:
        progress_printer(f"[offarc] probe corpus={LIVECODEBENCH_KEY} tasks={probe_task_count}")
        tasks, _skipped = livecodebench_task_loader(probe_task_count)
        return probe_oracle(
            corpus_key=LIVECODEBENCH_KEY,
            evaluation_corpus=LIVECODEBENCH_CORPUS,
            tasks=tasks[:probe_task_count],
            legacy_by_task=legacy_by_task,
            k=k,
            executor=executor,
        )

    route = route_corpus(evalplus_probe=evalplus_probe, livecodebench_probe=livecodebench_probe)
    loader = evalplus_task_loader if route.corpus_key == EVALPLUS_KEY else livecodebench_task_loader
    tasks, skipped = loader(n_tasks)
    checkpoint_path = checkpoint_dir / f"offarc_power_sync_gemma12b_{route.corpus_key}_k{k}.checkpoint.json"
    evaluations_by_task = _load_checkpoint(checkpoint_path)
    sampler_fn = sampler
    completed_tasks: list[CodeTask] = []
    stopped_reason: str | None = None

    for index, task in enumerate(tasks, start=1):
        if time.time() - started >= self_budget_s and completed_tasks:  # pragma: no cover
            stopped_reason = f"self_budget_hit_before_task_{index}"
            break
        current = list(evaluations_by_task.get(task.task_id, []))
        if len(current) < k:
            current = evalplus_base._extend_from_legacy(
                task=task,
                current=current,
                legacy_by_task=legacy_by_task,
                k=k,
                executor=executor,
            )
        if len(current) < k and time.time() - started < self_budget_s:
            if sampler_fn is None:
                progress_printer(
                    "[offarc] loading local sampler "
                    f"corpus={route.corpus_key} elapsed={time.time() - started:.1f}s"
                )
                sampler_fn = make_live_sampler()
            current = evalplus_base._extend_from_sampler(
                task=task,
                current=current,
                sampler=sampler_fn,
                executor=executor,
                k=k,
            )
        evaluations_by_task[task.task_id] = [
            _compact_evaluation(ev) for ev in sorted(current, key=lambda ev: ev.draw_index)[:k]
        ]
        completed_tasks.append(task)
        _write_checkpoint(
            checkpoint_path,
            tasks=tasks,
            evaluations_by_task=evaluations_by_task,
            skipped_tasks=skipped,
            k=k,
            route=route,
            legacy_checkpoint_path=legacy_checkpoint_path,
        )
        partial = base.score_evaluated_tasks(completed_tasks, evaluations_by_task, seed=RANDOM_SEED)
        progress_printer(
            "[offarc] "
            f"task {len(completed_tasks)}/{len(tasks)} corpus={route.corpus_key} "
            f"oracle={partial['oracle_passrate']:.4f} "
            f"armB={partial['armB_demofit_passrate']:.4f} "
            f"elapsed={time.time() - started:.1f}s"
        )
        if time.time() - started >= self_budget_s:
            stopped_reason = f"self_budget_hit_after_task_{index}"
            break

    scored = base.score_evaluated_tasks(completed_tasks, evaluations_by_task, seed=RANDOM_SEED)
    model_specs = {
        "local_generator": "unsloth/gemma-4-12B-it-GGUF",
        "evaluation_corpus": route.evaluation_corpus,
        "verifier": (
            "model-free demo-fit + restricted-namespace execution + "
            "content-hash/fingerprint symbolic partition from offarc_transfer_power_run.py"
        ),
        "candidate_pool_policy": "same generated pool shared by all four arms",
        "source_candidate_checkpoint": str(legacy_checkpoint_path),
        "stable_checkpoint": str(checkpoint_path),
        "llama_cpp": {"n_ctx": 4096, "n_batch": 512, "n_gpu_layers": -1},
    }
    artifact = build_terminal_artifact(
        per_task=scored["per_task"],
        route=route,
        preconditions_checked=preconditions,
        model_specs=model_specs,
        checkpoint_path=checkpoint_path,
        source_candidate_checkpoint=legacy_checkpoint_path,
        started_s=started,
        ended_s=time.time(),
        n_bootstrap=n_bootstrap,
        powered_task_floor=powered_task_floor,
        candidate_pool=base._candidate_pool_records(evaluations_by_task),
        stopped_reason=stopped_reason,
    )
    validate_terminal_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def probe_oracle(
    *,
    corpus_key: str,
    evaluation_corpus: str,
    tasks: list[CodeTask],
    legacy_by_task: dict[str, list[GeneratedCandidate]],
    k: int,
    executor: Callable[[str, str, tuple[Any, ...], float], tuple[Any, Exception | None]],
) -> OracleProbe:
    evaluations_by_task: dict[str, list[CandidateEvaluation]] = {}
    probed_tasks: list[CodeTask] = []
    for task in tasks:
        probe_task = _bounded_probe_task(task)
        current = evalplus_base._extend_from_legacy(
            task=probe_task,
            current=[],
            legacy_by_task=legacy_by_task,
            k=k,
            executor=executor,
        )
        evaluations_by_task[probe_task.task_id] = current[:k]
        probed_tasks.append(probe_task)
    scored = base.score_evaluated_tasks(probed_tasks, evaluations_by_task, seed=RANDOM_SEED)
    return OracleProbe(
        corpus_key=corpus_key,
        evaluation_corpus=evaluation_corpus,
        oracle_passrate=float(scored["oracle_passrate"]),
        n_tasks=len(probed_tasks),
    )


def fast_restricted_executor(
    code: str, func_name: str, args: tuple[Any, ...], timeout: float
) -> tuple[Any, Exception | None]:
    """Fast restricted-namespace execution with a real wall-clock timeout."""
    import signal

    def _timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"restricted execution timed out after {timeout}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    namespace: dict[str, Any] = {}
    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, timeout))
        exec(code, namespace)  # noqa: S102 - intentional generated-code verification.
        func = namespace.get(func_name)
        if func is None:
            return None, NameError(f"Function '{func_name}' not found in code")
        return func(*args), None
    except Exception as exc:
        return None, exc
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
        if old_timer[0] > 0:  # pragma: no cover - defensive nested-alarm restore.
            signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])


def check_preconditions() -> list[dict[str, Any]]:  # pragma: no cover - live resource probe.
    gguf_cached = GGUF_CACHE.exists() and any(GGUF_CACHE.iterdir())
    model_file = resolve_gemma_model_file()
    llama_ok = _import_ok("llama_cpp")
    evalplus_ok, evalplus_detail = _probe_evalplus_loadable()
    livecodebench_ok, livecodebench_detail = _probe_livecodebench_loadable()
    sandbox_ok = _sandbox_import_ok()
    return [
        {
            "resource": "local_gguf_cached",
            "available": gguf_cached,
            "path": str(model_file) if model_file else str(GGUF_CACHE),
        },
        {"resource": "llama_cpp_importable", "available": llama_ok},
        {"resource": "evalplus_loadable", "available": evalplus_ok, "detail": evalplus_detail},
        {
            "resource": "livecodebench_v6_loadable",
            "available": livecodebench_ok,
            "detail": livecodebench_detail,
        },
        {"resource": "restricted_exec_importable", "available": sandbox_ok},
    ]


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    resources = {str(row.get("resource")): bool(row.get("available")) for row in preconditions}
    for resource, blocker in (
        ("local_gguf_cached", "blocked_local_gguf_not_cached"),
        ("llama_cpp_importable", "blocked_llama_cpp_unavailable"),
        ("restricted_exec_importable", "blocked_sandbox_unavailable"),
    ):
        if resources.get(resource) is False:
            return blocker
    evalplus_ok = resources.get("evalplus_loadable", False)
    livecodebench_ok = resources.get("livecodebench_v6_loadable", False)
    if not (evalplus_ok or livecodebench_ok):
        return "blocked_no_code_corpus"
    return None


def load_evalplus_tasks(
    *, limit: int = DEFAULT_N_TASKS,
) -> tuple[list[CodeTask], list[dict[str, Any]]]:  # pragma: no cover - live corpus path.
    return evalplus_base.load_code_tasks(limit=limit)


def load_livecodebench_v6_tasks(
    *, limit: int = DEFAULT_N_TASKS,
) -> tuple[list[CodeTask], list[dict[str, Any]]]:  # pragma: no cover - escalation path.
    from datasets import load_dataset

    skipped: list[dict[str, Any]] = []
    tasks: list[CodeTask] = []
    candidates = (
        ("livecodebench/code_generation_lite", "test"),
        ("livecodebench/code_generation", "test"),
        ("LiveCodeBench/code_generation_lite", "test"),
    )
    last_error = ""
    for dataset_name, split in candidates:
        try:
            dataset = load_dataset(dataset_name, split=split)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
            continue
        for index, row in enumerate(dataset):
            task, reason = _task_from_livecodebench_row(dict(row), index=index)
            if task is None:
                skipped.append(
                    {
                        "task_id": str(row.get("question_id") or index),
                        "corpus": LIVECODEBENCH_KEY,
                        "reason": reason,
                    }
                )
                continue
            tasks.append(task)
            if len(tasks) >= limit:
                return tasks, skipped
        if tasks:
            return tasks, skipped
    if last_error:
        raise RuntimeError(last_error)
    return tasks, skipped


def make_live_sampler() -> base.LocalGemmaSampler:  # pragma: no cover - live model path.
    from llama_cpp import Llama

    model_path = resolve_gemma_model_file()
    if model_path is None:
        raise RuntimeError("local Gemma 4 12B GGUF cache exists but no model file was resolvable")
    llama = Llama(
        model_path=str(model_path),
        n_ctx=4096,
        n_gpu_layers=-1,
        n_batch=512,
        verbose=False,
    )
    return base.LocalGemmaSampler(llama, base_seed=RANDOM_SEED)


def resolve_gemma_model_file(cache_dir: Path = GGUF_CACHE) -> Path | None:  # pragma: no cover.
    ggufs = sorted(cache_dir.glob("**/*.gguf"))
    for candidate in ggufs:
        if candidate.exists():
            return candidate
    files = [path for path in cache_dir.glob("**/*") if path.is_file()]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_size)


def _call_probe(
    corpus_key: str, evaluation_corpus: str, probe: Callable[[], OracleProbe]
) -> OracleProbe:
    try:
        return probe()
    except Exception as exc:
        return OracleProbe(
            corpus_key=corpus_key,
            evaluation_corpus=evaluation_corpus,
            oracle_passrate=1.0,
            n_tasks=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def _load_checkpoint(path: Path) -> dict[str, list[CandidateEvaluation]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    evaluations_by_task: dict[str, list[CandidateEvaluation]] = {}
    for task_id, rows in payload.get("evaluations_by_task", {}).items():
        evaluations_by_task[str(task_id)] = [
            CandidateEvaluation(
                task_id=str(row.get("task_id", task_id)),
                draw_index=int(row.get("draw_index", index)),
                status=str(row.get("status", "ok")),
                code=str(row.get("code", "")),
                visible_passes=list(row.get("visible_passes") or []),
                hidden_passes=list(row.get("hidden_passes") or []),
                visible_outputs=list(row.get("visible_outputs") or []),
                hidden_outputs=list(row.get("hidden_outputs") or []),
                fingerprint_outputs=list(row.get("fingerprint_outputs") or []),
                generation_seconds=float(row.get("generation_seconds", 0.0)),
                truncated=bool(row.get("truncated", False)),
                error=row.get("error"),
            )
            for index, row in enumerate(rows)
        ]
    return evaluations_by_task


def _write_checkpoint(
    path: Path,
    *,
    tasks: list[CodeTask],
    evaluations_by_task: dict[str, list[CandidateEvaluation]],
    skipped_tasks: list[dict[str, Any]],
    k: int,
    route: CorpusRoute,
    legacy_checkpoint_path: Path,
) -> None:
    payload = {
        "experiment": "experiment_4068_offarc_transfer_power_sync_checkpoint",
        "schema": "carnot.experiment_4068_offarc_transfer_power_sync_checkpoint.v1",
        "mode": "sync_resume_accumulate",
        "evaluation_corpus": route.evaluation_corpus,
        "corpus": route.corpus_key,
        "corpus_routed_reason": route.corpus_routed_reason,
        "k_candidates_per_task": k,
        "completed_task_ids": sorted(evaluations_by_task),
        "ordered_task_ids": [task.task_id for task in tasks],
        "skipped_tasks": skipped_tasks,
        "source_candidate_checkpoint": str(legacy_checkpoint_path),
        "evaluations_by_task": {
            task_id: [_checkpoint_eval_row(ev) for ev in evaluations]
            for task_id, evaluations in sorted(evaluations_by_task.items())
        },
        "updated_at_unix": time.time(),
    }
    _write_json(path, payload)


def _compact_evaluation(ev: CandidateEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        task_id=ev.task_id,
        draw_index=ev.draw_index,
        status=ev.status,
        code=ev.code,
        visible_passes=ev.visible_passes,
        hidden_passes=ev.hidden_passes,
        visible_outputs=ev.visible_outputs,
        hidden_outputs=[],
        fingerprint_outputs=ev.fingerprint_outputs,
        generation_seconds=ev.generation_seconds,
        truncated=ev.truncated,
        error=ev.error,
    )


def _checkpoint_eval_row(ev: CandidateEvaluation) -> dict[str, Any]:
    row = asdict(_compact_evaluation(ev))
    row["hidden_outputs_omitted"] = True
    return row


def _task_from_livecodebench_row(
    row: dict[str, Any], *, index: int
) -> tuple[CodeTask | None, str | None]:  # pragma: no cover - best-effort adapter.
    starter = str(row.get("starter_code") or row.get("code") or "")
    func_name = _infer_func_name(starter)
    if not func_name:
        return None, "missing_function_signature"
    visible = _tests_from_lcb_cases(row.get("public_test_cases"), func_name=func_name)
    hidden = _tests_from_lcb_cases(row.get("private_test_cases"), func_name=func_name)
    if not visible or not hidden:
        return None, "missing_public_or_private_exact_tests"
    task_id = str(row.get("question_id") or row.get("id") or f"LiveCodeBench/{index}")
    prompt = str(row.get("question_content") or row.get("prompt") or "")
    return (
        CodeTask(
            task_id=task_id,
            corpus=LIVECODEBENCH_KEY,
            prompt=prompt,
            func_name=func_name,
            visible_tests=visible[:2],
            hidden_tests=hidden,
        ),
        None,
    )


def _infer_func_name(source: str) -> str | None:  # pragma: no cover - LCB adapter.
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name
    return None


def _tests_from_lcb_cases(cases: Any, *, func_name: str) -> list[CodeTest]:  # pragma: no cover.
    if isinstance(cases, str):
        try:
            cases = json.loads(cases)
        except json.JSONDecodeError:
            return []
    if not isinstance(cases, list):
        return []
    tests: list[CodeTest] = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        args = case.get("input")
        expected = case.get("output")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                continue
        if not isinstance(args, list):
            args = [args]
        tests.append(
            CodeTest(
                source=f"assert {func_name}{base._args_repr(tuple(args))} == {expected!r}",
                func_name=func_name,
                args=tuple(args),
                expected=expected,
            )
        )
    return tests


def _probe_evalplus_loadable() -> tuple[bool, str]:  # pragma: no cover.
    try:
        tasks, skipped = load_evalplus_tasks(limit=1)
        return bool(tasks), f"loaded={len(tasks)} skipped={len(skipped)}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


def _probe_livecodebench_loadable() -> tuple[bool, str]:  # pragma: no cover.
    if _import_ok("datasets"):
        return True, "datasets importable; LiveCodeBench load deferred until EvalPlus lacks headroom"
    return False, "datasets import failed"


def _bounded_probe_task(task: CodeTask, *, max_hidden_tests: int = 3) -> CodeTask:
    return CodeTask(
        task_id=task.task_id,
        corpus=task.corpus,
        prompt=task.prompt,
        func_name=task.func_name,
        visible_tests=task.visible_tests,
        hidden_tests=task.hidden_tests[:max_hidden_tests],
    )


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


def _rate_bool(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(1 for row in rows if bool(row.get(key))) / max(1, len(rows)), 6)


def _paired_deltas(rows: list[dict[str, Any]], key: str) -> list[int]:
    return [int(bool(row.get(key))) - int(bool(row.get("armA_vote_pass1"))) for row in rows]


def _delta_pp(left: float, right: float) -> float:
    return round((left - right) * 100.0, 4)


def _ci_excludes_zero(ci: list[float]) -> bool:
    return bool(ci[0] > 0.0 or ci[1] < 0.0)


def _missing_gaps(
    rows: list[dict[str, Any]], b_ci: list[float], app_ci: list[float], c_ci: list[float]
) -> list[str]:
    gaps: list[str] = []
    if rows and not _ci_excludes_zero(b_ci):
        gaps.append("GAP-CODE-EXEC-DEMOFIT")
    if rows and not (_ci_excludes_zero(app_ci) or _ci_excludes_zero(c_ci)):
        gaps.append("GAP-CODE-SPECIALIZED-DISCRIMINATOR")
    for row in rows:
        if row.get("oracle_hidden_pass") and not (
            row.get("armB_demofit_pass1") or row.get("armC_symbolic_partition_pass1")
        ):
            gaps.append(f"UNSELECTABLE:{row.get('task_id')}")
    return sorted(set(gaps))


def _verdict(
    *,
    corpus_key: str,
    accumulated_n: int,
    powered_task_floor: int,
    oracle_headroom: bool,
    demofit_delta_pp: float,
    demofit_ci: list[float],
    arm_app_ci: list[float],
    arm_c_ci: list[float],
    best_arm: str,
    best_delta_pp: float,
    best_ci: list[float],
) -> str:
    if accumulated_n < powered_task_floor:
        return f"complete: offarc_transfer_power_accumulating_n{accumulated_n}_{corpus_key}"
    if not oracle_headroom:
        return f"complete: offarc_transfer_no_oracle_headroom_{corpus_key}_n{accumulated_n}"
    if demofit_delta_pp > 0.0 and demofit_ci[0] > 0.0:
        return f"complete: offarc_demofit_transfers_to_code_ci_excl0_{corpus_key}_n{accumulated_n}"
    if demofit_delta_pp < 0.0 and demofit_ci[1] < 0.0:
        return f"complete: offarc_demofit_negative_ci_excl0_{corpus_key}_n{accumulated_n}"
    if best_arm != "armB_demofit" and best_delta_pp > 0.0 and best_ci[0] > 0.0:
        label = "symbolic" if best_arm == "armC_symbolic" else "aces"
        return f"complete: offarc_demofit_touches0_{label}_excl0_{corpus_key}_n{accumulated_n}"
    if arm_app_ci[0] > 0.0:  # pragma: no cover - dominated by best-arm branch.
        return f"complete: offarc_demofit_touches0_aces_excl0_{corpus_key}_n{accumulated_n}"
    if arm_c_ci[0] > 0.0:  # pragma: no cover - dominated by best-arm branch.
        return f"complete: offarc_demofit_touches0_symbolic_excl0_{corpus_key}_n{accumulated_n}"
    return f"complete: offarc_transfer_power_no_ci_closed_{corpus_key}_n{accumulated_n}"


def _is_two_numeric_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


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
    parser = argparse.ArgumentParser(description="Exp 4068 synchronous OFF-ARC transfer runner")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--legacy-checkpoint", type=Path, default=LEGACY_CHECKPOINT)
    parser.add_argument("--n-tasks", type=int, default=DEFAULT_N_TASKS)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--self-budget-s", type=float, default=DEFAULT_SELF_BUDGET_S)
    parser.add_argument("--probe-task-count", type=int, default=DEFAULT_PROBE_TASKS)
    args = parser.parse_args()
    artifact = run(
        output_path=args.output,
        checkpoint_dir=args.checkpoint_dir,
        legacy_checkpoint_path=args.legacy_checkpoint,
        n_tasks=args.n_tasks,
        k=args.k,
        self_budget_s=args.self_budget_s,
        probe_task_count=args.probe_task_count,
    )
    print(
        f"-> {artifact['honest_verdict']} n={artifact['accumulated_n_tasks']} "
        f"corpus={artifact['corpus']} oracle={artifact['oracle_passrate']:.4f}"
    )


if __name__ == "__main__":  # pragma: no cover.
    main()

"""Exp 4059 MoE resume-and-accumulate ARC best-of-N runner.

Spec refs: REQ-VERIFY-4058, SCENARIO-VERIFY-4058.

This runner resumes the Exp 4048 Qwen MoE checkpoint into a stable
corpus/model/k-keyed checkpoint and keeps accumulating ARC-1 best-of-N
induction evidence. The verifier side stays the Exp 4012 GAP-4 model-free
path; only the local open-weight generator supplies candidate programs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script bootstrap.
    sys.path.insert(0, str(REPO_ROOT))

import experiment_4012_gap4_local_best_of_n as exp4012
import experiment_4048_decentralization_moe_base_best_of_n as run4048
from experiment_4002_gap4_local_generator_arm import CODEX_REF, POOL, SEED
from scripts.experiment_template import _compute_repro_checksum

OUTPUT = REPO_ROOT / "results" / "experiment_4059_decentralization_moe_resume_raw.json"
SOURCE_CHECKPOINT = (
    REPO_ROOT / "results" / "experiment_4048_decentralization_moe_base_raw.checkpoint.json"
)
STABLE_CHECKPOINT = (
    REPO_ROOT / "results" / "decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json"
)

INFERENCE_SUBSTRATE = "live_llm_inference"
DEFAULT_K = 8
DEFAULT_DRAW_BATCH_SIZE = run4048.DEFAULT_DRAW_BATCH_SIZE
STABLE_CHECKPOINT_KEY = "arc1:qwen35a3b:k8"

REQUIRED_RAW_FIELDS = [
    "honest_verdict",
    "runner_ready",
    "moe_base_model",
    "stable_checkpoint_path",
    "source_checkpoint_path",
    "resumed_from_n",
    "ACCUMULATED-N",
    "best_of_n_coverage",
    "local_demo_perfect_coverage_bestofn",
    "k_samples_per_task",
    "gated_pass_at_2",
    "local_gated_pass2",
    "local_seconds",
    "per_task",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "launched_pid",
]


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checkpoint_payload(
    tasks: dict[str, list[dict[str, Any]]],
    *,
    k: int,
    model_name: str,
) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_4059_decentralization_moe_resume.checkpoint.v1",
        "stable_checkpoint_key": STABLE_CHECKPOINT_KEY,
        "k_samples_per_task": int(k),
        "local_model_used": model_name,
        "tasks": tasks,
    }


def _load_checkpoint_tasks(
    checkpoint_path: Path | str,
    *,
    k: int,
    model_name: str,
) -> dict[str, list[dict[str, Any]]] | None:
    try:
        payload = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("k_samples_per_task") != k:
        return None
    if payload.get("local_model_used") != model_name:
        return None
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        return None
    normalized: dict[str, list[dict[str, Any]]] = {}
    for task, samples in tasks.items():
        if not isinstance(task, str) or not isinstance(samples, list):
            return None
        normalized[task] = [sample for sample in samples if isinstance(sample, dict)]
    return normalized


def _save_checkpoint(
    checkpoint_path: Path | str,
    tasks: dict[str, list[dict[str, Any]]],
    *,
    k: int,
    model_name: str,
) -> None:
    _write_json(Path(checkpoint_path), _checkpoint_payload(tasks, k=k, model_name=model_name))


def _source_checkpoint_count(
    checkpoint_path: Path | str,
    *,
    k: int,
    model_name: str,
) -> int:
    tasks = _load_checkpoint_tasks(checkpoint_path, k=k, model_name=model_name)
    return len(tasks) if tasks is not None else 0


def check_preconditions(
    *,
    pool_path: Path | str,
    source_checkpoint_path: Path | str = SOURCE_CHECKPOINT,
    model_key: str = "auto",
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
    cache_dir: Path | str = run4048.MOE_CACHE_DIR,
    llama_available_override: bool | None = None,
    k: int = DEFAULT_K,
) -> tuple[list[dict[str, Any]], dict[str, str] | None, int]:
    """Check all resources before any live inference is attempted."""
    preconditions, chosen = run4048.check_preconditions(
        model_key=model_key,
        pool_path=pool_path,
        resolver=resolver,
        cache_dir=cache_dir,
        llama_available_override=llama_available_override,
    )
    model_name = chosen["name"] if chosen else run4048.MOE_MODEL["name"]
    source_tasks = _load_checkpoint_tasks(source_checkpoint_path, k=k, model_name=model_name)
    resumed_from_n = len(source_tasks) if source_tasks is not None else 0
    preconditions.append(
        {
            "resource": "exp4048_checkpoint",
            "available": source_tasks is not None,
            "path": str(source_checkpoint_path),
            "n_tasks": resumed_from_n,
        }
    )
    return preconditions, chosen, resumed_from_n


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    """Map the first failed precondition to the required blocked verdict."""
    by_resource = {row["resource"]: bool(row["available"]) for row in preconditions}
    if not by_resource.get("moe_base_gguf_cached", False):
        return "blocked_moe_base_not_cached"
    if not by_resource.get("llama_cpp", False):
        return "blocked_llama_cpp_unavailable"
    if not by_resource.get("exp4012_arc1_pool_and_verifier_primitives", False):
        return "blocked_exp4012_pool_unreadable"
    if not by_resource.get("exp4048_checkpoint", False):
        return "blocked_exp4048_checkpoint_unreadable"
    return None


def ensure_stable_checkpoint(
    *,
    source_checkpoint_path: Path | str = SOURCE_CHECKPOINT,
    stable_checkpoint_path: Path | str = STABLE_CHECKPOINT,
    k: int = DEFAULT_K,
    model_name: str = run4048.MOE_MODEL["name"],
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    """Merge the Exp 4048 source checkpoint into the stable accumulation checkpoint."""
    source_tasks = _load_checkpoint_tasks(source_checkpoint_path, k=k, model_name=model_name)
    if source_tasks is None:
        raise ValueError("source checkpoint is unreadable or incompatible")
    stable_tasks = _load_checkpoint_tasks(stable_checkpoint_path, k=k, model_name=model_name) or {}
    merged = {**source_tasks, **stable_tasks}
    _save_checkpoint(stable_checkpoint_path, merged, k=k, model_name=model_name)
    return merged, len(source_tasks)


def _entries_by_task(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_task.setdefault(entry["task"], []).append(entry)
    return by_task


def _done_task_count(samples_by_task: dict[str, list[dict[str, Any]]], *, k: int) -> int:
    return sum(1 for samples in samples_by_task.values() if run4048._task_done(samples, k))


def _next_unfinished_tasks(
    entries: list[dict[str, Any]],
    samples_by_task: dict[str, list[dict[str, Any]]],
    *,
    k: int,
) -> list[str]:
    by_task = _entries_by_task(entries)
    tasks = []
    for task_name in sorted(by_task):
        cached = samples_by_task.get(task_name)
        if isinstance(cached, list) and run4048._task_done(cached, k):
            continue
        tasks.append(task_name)
    return tasks


def induce_pool_resume_batched(
    entries: list[dict[str, Any]],
    sampler: Any,
    *,
    samples_by_task: dict[str, list[dict[str, Any]]],
    stable_checkpoint_path: Path | str,
    k: int,
    model_name: str,
    started_s: float | None = None,
    max_wall_s: float | None = None,
    max_new_tasks: int = 0,
    batch_size: int = DEFAULT_DRAW_BATCH_SIZE,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Continue best-of-N sampling from the first unfinished task and checkpoint each task."""
    by_task = _entries_by_task(entries)
    processed: list[str] = []
    for task_name in sorted(by_task):
        cached = samples_by_task.get(task_name)
        if isinstance(cached, list) and run4048._task_done(cached, k):
            continue
        if max_new_tasks and len(processed) >= max_new_tasks:
            break
        if (
            started_s is not None
            and max_wall_s is not None
            and time.time() - started_s >= max_wall_s
        ):
            break
        samples_by_task[task_name] = run4048.induce_task_samples_batched(
            task_name,
            by_task[task_name][0]["demos"],
            sampler,
            k=k,
            batch_size=batch_size,
        )
        processed.append(task_name)
        _save_checkpoint(stable_checkpoint_path, samples_by_task, k=k, model_name=model_name)
    return samples_by_task, processed


def _fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _raw_verdict(coverage: float, pass2: float, accumulated_n: int) -> str:
    if accumulated_n >= 30:
        return (
            "complete: decentralization_moe_resume_accumulated_"
            + str(accumulated_n)
            + "_cov"
            + _fmt(coverage)
            + "_pass2"
            + _fmt(pass2)
        )
    return "complete: decentralization_moe_resume_partial_" + str(accumulated_n) + "_tasks"


def _scored_entries(
    entries: list[dict[str, Any]],
    samples_by_task: dict[str, list[dict[str, Any]]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    return [
        entry
        for entry in entries
        if run4048._task_done(samples_by_task.get(entry["task"], []), k)
    ]


def _new_task_summary(
    processed_tasks: list[str],
    samples_by_task: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    summary = []
    for task in processed_tasks:
        samples = samples_by_task.get(task, [])
        summary.append(
            {
                "task": task,
                "n_samples": len(samples),
                "n_demo_perfect": sum(1 for sample in samples if sample.get("demo_perfect")),
                "local_seconds": round(
                    sum(float(sample.get("local_s", 0.0)) for sample in samples), 2
                ),
            }
        )
    return summary


def _repro_checksum(output_path: Path | str, pool_path: Path | str) -> str:
    code_files = [
        str(Path(__file__).resolve()),
        str(
            (
                REPO_ROOT
                / "scripts"
                / "experiments"
                / "experiment_4048_decentralization_moe_base_best_of_n.py"
            )
        ),
        str((REPO_ROOT / "scripts" / "experiments" / "experiment_4012_gap4_local_best_of_n.py")),
        str((REPO_ROOT / "python" / "carnot" / "agentic" / "arc_gap4_execution_verifier.py")),
    ]
    checksum = _compute_repro_checksum(SEED, code_files, str(pool_path))
    salt = str(output_path).encode("utf-8")
    return hashlib.sha256((checksum + salt.hex()).encode("utf-8")).hexdigest()[:16]


def blocked_raw_artifact(
    verdict: str,
    *,
    chosen_model: dict[str, str] | None,
    preconditions: list[dict[str, Any]],
    duration_s: float,
    stable_checkpoint_path: Path | str,
    source_checkpoint_path: Path | str = SOURCE_CHECKPOINT,
    k: int = DEFAULT_K,
    output_path: Path | str = OUTPUT,
    pool_path: Path | str = POOL,
) -> dict[str, Any]:
    """Build a valid raw artifact for precondition-blocked resume runs."""
    artifact = {
        "experiment": "experiment_4059_decentralization_moe_resume_raw",
        "schema": "carnot.experiment_4059_decentralization_moe_resume_raw.v1",
        "title": "Decentralization MoE resume-and-accumulate ARC best-of-N raw run",
        "honest_verdict": verdict,
        "runner_ready": False,
        "moe_base_model": chosen_model["name"] if chosen_model else "none",
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "source_checkpoint_path": str(source_checkpoint_path),
        "resumed_from_n": 0,
        "ACCUMULATED-N": 0,
        "accumulated_n": 0,
        "best_of_n_coverage": 0.0,
        "local_demo_perfect_coverage_bestofn": 0.0,
        "k_samples_per_task": int(k),
        "gated_pass_at_2": 0.0,
        "local_gated_pass2": 0.0,
        "local_seconds": 0.0,
        "new_tasks_processed": 0,
        "new_local_seconds": 0.0,
        "new_task_sample_summary": [],
        "per_task": [],
        "per_task_sample_summary": [],
        "model_specs": {
            "generator_model": chosen_model["name"] if chosen_model else "none",
            "generator_hf_id": chosen_model["hf_id"] if chosen_model else "none",
            "generator_gguf_path": chosen_model["model_path"] if chosen_model else "none",
            "verifier": "model-free GAP-4 verifier primitives reused unchanged",
        },
        "random_seed": SEED,
        "reproducibility_checksum": _repro_checksum(output_path, pool_path),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(float(duration_s), 2),
        "launched_pid": 0,
        "verifier_side_unchanged": True,
    }
    validate_raw_artifact(artifact)
    return artifact


def build_raw_artifact(
    *,
    base_artifact: dict[str, Any],
    chosen_model: dict[str, str],
    preconditions: list[dict[str, Any]],
    output_path: Path | str,
    pool_path: Path | str,
    source_checkpoint_path: Path | str,
    stable_checkpoint_path: Path | str,
    resumed_from_n: int,
    accumulated_n: int,
    processed_tasks: list[str],
    samples_by_task: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Adapt the Exp 4012-compatible result into the Exp 4059 raw schema."""
    coverage = float(base_artifact["local_demo_perfect_coverage_bestofn"])
    pass2 = float(base_artifact["local_gated_pass2"])
    raw = run4048.build_raw_artifact(
        base_artifact=base_artifact,
        chosen_model=chosen_model,
        preconditions=preconditions,
        output_path=output_path,
        pool_path=pool_path,
    )
    new_summary = _new_task_summary(processed_tasks, samples_by_task)
    new_seconds = round(sum(float(row["local_seconds"]) for row in new_summary), 2)
    raw.update(
        {
            "experiment": "experiment_4059_decentralization_moe_resume_raw",
            "schema": "carnot.experiment_4059_decentralization_moe_resume_raw.v1",
            "title": "Decentralization MoE resume-and-accumulate ARC best-of-N raw run",
            "honest_verdict": _raw_verdict(coverage, pass2, accumulated_n),
            "stable_checkpoint_path": str(stable_checkpoint_path),
            "source_checkpoint_path": str(source_checkpoint_path),
            "resumed_from_n": int(resumed_from_n),
            "ACCUMULATED-N": int(accumulated_n),
            "accumulated_n": int(accumulated_n),
            "new_tasks_processed": len(processed_tasks),
            "new_local_seconds": new_seconds,
            "new_task_sample_summary": new_summary,
            "reproducibility_checksum": _repro_checksum(output_path, pool_path),
            "preconditions_checked": preconditions,
        }
    )
    validate_raw_artifact(raw)
    return raw


def validate_raw_artifact(artifact: dict[str, Any]) -> None:
    """Validate the fields consumed by the build receipt and later collector."""
    for field in REQUIRED_RAW_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    run4048.validate_raw_artifact(artifact)
    for field in ("stable_checkpoint_path", "source_checkpoint_path"):
        if not isinstance(artifact[field], str):
            raise ValueError(f"{field} must be a string")
    for field in ("resumed_from_n", "ACCUMULATED-N", "new_tasks_processed"):
        if field in artifact and not _is_bare_int(artifact[field]):
            raise ValueError(f"{field} must be a bare int")
    if "accumulated_n" in artifact and not _is_bare_int(artifact["accumulated_n"]):
        raise ValueError("accumulated_n must be a bare int")
    if "new_local_seconds" in artifact and not _is_bare_float(artifact["new_local_seconds"]):
        raise ValueError("new_local_seconds must be a bare float")
    if "new_task_sample_summary" in artifact and not isinstance(
        artifact["new_task_sample_summary"], list
    ):
        raise ValueError("new_task_sample_summary must be a list")


def run(
    *,
    model_key: str = "auto",
    pool_path: Path | str = POOL,
    output_path: Path = OUTPUT,
    codex_ref_path: Path = CODEX_REF,
    source_checkpoint_path: Path | str = SOURCE_CHECKPOINT,
    stable_checkpoint_path: Path | str = STABLE_CHECKPOINT,
    k: int = DEFAULT_K,
    max_new_tasks: int = 0,
    n_ctx: int = 16384,
    max_wall_s: float = 4500.0,
    batch_size: int = DEFAULT_DRAW_BATCH_SIZE,
    sampler: Any | None = None,
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
    cache_dir: Path | str = run4048.MOE_CACHE_DIR,
    llama_available_override: bool | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run the resume-and-accumulate experiment or emit a blocked artifact."""
    started = time.time()
    preconditions, chosen_model, _resumed_count = check_preconditions(
        model_key=model_key,
        pool_path=pool_path,
        source_checkpoint_path=source_checkpoint_path,
        resolver=resolver,
        cache_dir=cache_dir,
        llama_available_override=llama_available_override,
        k=k,
    )
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_raw_artifact(
            blocker,
            chosen_model=chosen_model,
            preconditions=preconditions,
            duration_s=time.time() - started,
            stable_checkpoint_path=stable_checkpoint_path,
            source_checkpoint_path=source_checkpoint_path,
            k=k,
            output_path=output_path,
            pool_path=pool_path,
        )
        if write:
            _write_json(Path(output_path), artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    if chosen_model is None:  # pragma: no cover - defensive; blocker should have caught this.
        raise RuntimeError("MoE model unavailable after precondition pass")

    with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
        pool = json.load(handle)
    entries = pool["entries"]

    samples_by_task, resumed_from_n = ensure_stable_checkpoint(
        source_checkpoint_path=source_checkpoint_path,
        stable_checkpoint_path=stable_checkpoint_path,
        k=k,
        model_name=chosen_model["name"],
    )

    if sampler is None:  # pragma: no cover - live multi-GB model load.
        llama = exp4012.load_local_llama(chosen_model["model_path"], n_ctx=n_ctx, seed=SEED)
        sampler = run4048.BatchedIndependentLocalSampler(llama, base_seed=SEED)

    samples_by_task, processed_tasks = induce_pool_resume_batched(
        entries,
        sampler,
        samples_by_task=samples_by_task,
        stable_checkpoint_path=stable_checkpoint_path,
        k=k,
        model_name=chosen_model["name"],
        started_s=started,
        max_wall_s=max_wall_s,
        max_new_tasks=max_new_tasks,
        batch_size=batch_size,
    )

    scored_entries = _scored_entries(entries, samples_by_task, k=k)
    verifier_t0 = time.time()
    prog_by_entry_id = exp4012.build_entry_programs(scored_entries, samples_by_task)
    scored = exp4012.score_best_of_n_pool(scored_entries, prog_by_entry_id, seed=SEED)
    verifier_s = time.time() - verifier_t0
    base_artifact = exp4012.build_complete_artifact(
        entries=scored_entries,
        samples_by_task={
            task: samples
            for task, samples in samples_by_task.items()
            if run4048._task_done(samples, k)
        },
        prog_by_entry_id=prog_by_entry_id,
        scored=scored,
        chosen_model=chosen_model,
        preconditions=preconditions,
        verifier_seconds=verifier_s,
        started_s=started,
        now_s=time.time(),
        k=k,
        codex_ref_path=codex_ref_path,
    )
    artifact = build_raw_artifact(
        base_artifact=base_artifact,
        chosen_model=chosen_model,
        preconditions=preconditions,
        output_path=output_path,
        pool_path=pool_path,
        source_checkpoint_path=source_checkpoint_path,
        stable_checkpoint_path=stable_checkpoint_path,
        resumed_from_n=resumed_from_n,
        accumulated_n=_done_task_count(samples_by_task, k=k),
        processed_tasks=processed_tasks,
        samples_by_task=samples_by_task,
    )
    if write:
        _write_json(Path(output_path), artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    return artifact


def main() -> None:  # pragma: no cover - exercised by operator command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["auto", "qwen35moe"], default="auto")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--max-new-tasks", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--max-wall-s", type=float, default=4500.0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_DRAW_BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source-checkpoint", type=Path, default=SOURCE_CHECKPOINT)
    parser.add_argument("--stable-checkpoint", type=Path, default=STABLE_CHECKPOINT)
    args = parser.parse_args()
    if args.k != DEFAULT_K:
        raise SystemExit("--k must remain 8 for the stable checkpoint key")
    if args.batch_size < 1 or args.batch_size > 16:
        raise SystemExit("--batch-size must be between 1 and 16")
    run(
        model_key=args.model,
        k=args.k,
        max_new_tasks=args.max_new_tasks,
        n_ctx=args.n_ctx,
        max_wall_s=args.max_wall_s,
        batch_size=args.batch_size,
        output_path=args.output,
        source_checkpoint_path=args.source_checkpoint,
        stable_checkpoint_path=args.stable_checkpoint,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

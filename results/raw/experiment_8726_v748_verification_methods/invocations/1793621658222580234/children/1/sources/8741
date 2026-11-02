"""Exp 4048 MoE-base local best-of-N ARC induction runner.

Spec refs: REQ-VERIFY-4047, SCENARIO-VERIFY-4047.

This is the raw run half for Exp 4047. It mirrors Exp 4012's ARC pool,
restricted execution verifier, candidate snap, and gated pass@2 scoring. The
only intended experimental variable is the local inducer base:
Qwen3.6-35B-A3B-GGUF, selected for MoE throughput rather than dense 31B
latency.
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
from experiment_4002_gap4_local_generator_arm import CODEX_REF, POOL, SEED
from scripts.experiment_template import _compute_repro_checksum

OUTPUT = REPO_ROOT / "results" / "experiment_4048_decentralization_moe_base_raw.json"
CHECKPOINT = REPO_ROOT / "results" / "experiment_4048_decentralization_moe_base_raw.checkpoint.json"

INFERENCE_SUBSTRATE = "live_llm_inference"
DEFAULT_K = 8
DEFAULT_DRAW_BATCH_SIZE = 4
BASELINE_12B_COVERAGE = 0.2581
MOE_CACHE_DIR = (
    Path.home() / ".cache" / "huggingface" / "hub" / ("models--unsloth--Qwen3.6-35B-A3B-GGUF")
)

MOE_MODEL = {
    "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "name": "Qwen3.6-35B-A3B",
    "model_key": "qwen35moe",
    "selection_note": "MoE throughput base, ~3B active params",
}

REQUIRED_RAW_FIELDS = [
    "honest_verdict",
    "runner_ready",
    "moe_base_model",
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


def _cache_dir_nonempty(cache_dir: Path | str) -> bool:
    path = Path(cache_dir).expanduser()
    try:
        return path.is_dir() and any(path.iterdir())
    except OSError:  # pragma: no cover - filesystem race while probing cache dir.
        return False


def select_moe_model(
    model_key: str = "auto",
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
) -> dict[str, str] | None:
    """Return the Qwen MoE model if its concrete GGUF path resolves."""
    if model_key not in {"auto", MOE_MODEL["model_key"]}:
        return None
    model_path = resolver(MOE_MODEL["hf_id"])
    if not model_path:
        return None
    return {**MOE_MODEL, "model_path": str(model_path)}


def check_preconditions(
    *,
    pool_path: Path | str,
    model_key: str = "auto",
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
    cache_dir: Path | str = MOE_CACHE_DIR,
    llama_available_override: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    """Check all resources before any live inference is attempted."""
    chosen = select_moe_model(model_key, resolver=resolver)
    cache_ok = _cache_dir_nonempty(cache_dir)
    if llama_available_override is None:
        try:
            import llama_cpp  # noqa: F401

            llama_ok = True
        except Exception:
            llama_ok = False
    else:
        llama_ok = bool(llama_available_override)

    pool_ok = exp4012._pool_and_verifier_loadable(pool_path)
    gguf_ok = cache_ok and chosen is not None
    preconditions = [
        {
            "resource": "moe_base_gguf_cached",
            "available": gguf_ok,
            "cache_dir": str(Path(cache_dir).expanduser()),
            "selected_model": chosen["name"] if chosen else None,
            "model_path": chosen["model_path"] if chosen else None,
        },
        {"resource": "llama_cpp", "available": llama_ok},
        {"resource": "exp4012_arc1_pool_and_verifier_primitives", "available": pool_ok},
    ]
    return preconditions, chosen if gguf_ok else None


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    """Map the first failed precondition to the required blocked verdict."""
    by_resource = {row["resource"]: bool(row["available"]) for row in preconditions}
    if not by_resource.get("moe_base_gguf_cached", False):
        return "blocked_moe_base_not_cached"
    if not by_resource.get("llama_cpp", False):
        return "blocked_llama_cpp_unavailable"
    if not by_resource.get("exp4012_arc1_pool_and_verifier_primitives", False):
        return "blocked_exp4012_pool_unreadable"
    return None


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_raw_artifact(artifact: dict[str, Any]) -> None:
    """Validate the raw artifact fields consumed by Exp 4047 and Exp 4048 collection."""
    for field in REQUIRED_RAW_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    if not isinstance(artifact["runner_ready"], bool):
        raise ValueError("runner_ready must be a bare bool")
    for field in (
        "best_of_n_coverage",
        "local_demo_perfect_coverage_bestofn",
        "gated_pass_at_2",
        "local_gated_pass2",
        "local_seconds",
        "duration_s",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    for field in ("k_samples_per_task", "random_seed", "launched_pid"):
        if not _is_bare_int(artifact[field]):
            raise ValueError(f"{field} must be a bare int")
    for field in ("moe_base_model", "inference_substrate", "reproducibility_checksum"):
        if not isinstance(artifact[field], str):
            raise ValueError(f"{field} must be a string")
    for field in ("per_task", "preconditions_checked"):
        if not isinstance(artifact[field], list):
            raise ValueError(f"{field} must be a list")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be a dict")


def _repro_checksum(output_path: Path | str, pool_path: Path | str) -> str:
    code_files = [
        str(Path(__file__).resolve()),
        str((REPO_ROOT / "scripts" / "experiments" / "experiment_4012_gap4_local_best_of_n.py")),
        str((REPO_ROOT / "python" / "carnot" / "agentic" / "arc_gap4_execution_verifier.py")),
    ]
    checksum = _compute_repro_checksum(SEED, code_files, str(pool_path))
    salt = str(output_path).encode("utf-8")
    return hashlib.sha256((checksum + salt.hex()).encode("utf-8")).hexdigest()[:16]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class BatchedIndependentLocalSampler:
    """Batch-shaped wrapper around llama.cpp chat completion for independent draws."""

    SYSTEM = exp4012.IndependentLocalSampler.SYSTEM

    def __init__(
        self,
        llama: Any,
        *,
        max_tokens: int = 2048,
        base_seed: int = SEED,
        base_temperature: float = 0.25,
    ) -> None:
        self._llama = llama
        self.max_tokens = max_tokens
        self.base_seed = base_seed
        self.base_temperature = base_temperature

    def sample_many(self, prompt: str, draw_indices: list[int]) -> list[tuple[int, str, float]]:
        """Generate a batch of independent samples for one task prompt."""
        outputs: list[tuple[int, str, float]] = []
        for draw_index in draw_indices:
            temperature = round(min(0.95, self.base_temperature + 0.05 * (draw_index % 8)), 3)
            seed = self.base_seed + 1009 * draw_index
            t0 = time.time()
            try:
                out = self._llama.create_chat_completion(
                    messages=[
                        {"role": "system", "content": self.SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    seed=seed,
                )
                text = out["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # pragma: no cover - defensive around live model failures.
                text = f"__local_error__:{type(exc).__name__}"
            outputs.append((draw_index, text, round(time.time() - t0, 2)))
        return outputs


def _call_sample_many(
    sampler: Any,
    prompt: str,
    draw_indices: list[int],
) -> list[tuple[int, str, float]]:
    if hasattr(sampler, "sample_many"):
        return list(sampler.sample_many(prompt, draw_indices))
    return [(draw_index, *sampler(prompt, draw_index)) for draw_index in draw_indices]


def _grade_sample(
    *,
    task_name: str,
    demos: list[dict[str, Any]],
    draw_index: int,
    raw: str,
    local_s: float,
) -> dict[str, Any]:
    code = exp4012._extract_code(raw)
    if code is None:
        return {
            "task": task_name,
            "draw_index": draw_index,
            "status": "no_code",
            "demo_fit": 0.0,
            "demo_perfect": False,
            "local_s": local_s,
            "code": None,
        }
    fn = exp4012.safe_transform_from_code(code)
    if fn is None:
        return {
            "task": task_name,
            "draw_index": draw_index,
            "status": "unsafe_or_uncompilable",
            "demo_fit": 0.0,
            "demo_perfect": False,
            "local_s": local_s,
            "code": code,
        }
    fit = exp4012.demo_fit(fn, demos)
    return {
        "task": task_name,
        "draw_index": draw_index,
        "status": "graded",
        "demo_fit": round(fit, 4),
        "demo_perfect": bool(fit >= 1.0),
        "local_s": local_s,
        "code_len": len(code),
        "code": code,
    }


def induce_task_samples_batched(
    task_name: str,
    demos: list[dict[str, Any]],
    sampler: Any,
    *,
    k: int,
    batch_size: int = DEFAULT_DRAW_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Draw up to k samples, stopping future batches after the first demo-perfect sample."""
    prompt = exp4012.demo_only_prompt(demos, task_name=task_name)
    samples: list[dict[str, Any]] = []
    for start in range(0, k, max(1, batch_size)):
        draw_indices = list(range(start, min(k, start + max(1, batch_size))))
        batch = sorted(_call_sample_many(sampler, prompt, draw_indices), key=lambda row: row[0])
        for draw_index, raw, local_s in batch:
            sample = _grade_sample(
                task_name=task_name,
                demos=demos,
                draw_index=draw_index,
                raw=raw,
                local_s=local_s,
            )
            samples.append(sample)
            if sample["demo_perfect"]:
                return samples
    return samples


def _checkpoint_payload(
    tasks: dict[str, list[dict[str, Any]]],
    *,
    k: int,
    model_name: str,
) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_4048_decentralization_moe_base_raw.checkpoint.v1",
        "k_samples_per_task": k,
        "local_model_used": model_name,
        "tasks": tasks,
    }


def _load_checkpoint(
    checkpoint_path: Path | None,
    *,
    k: int,
    model_name: str,
) -> dict[str, list[dict[str, Any]]]:
    if checkpoint_path is None or not checkpoint_path.exists():
        return {}
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - malformed checkpoint is treated as absent.
        return {}
    if payload.get("k_samples_per_task") != k or payload.get("local_model_used") != model_name:
        return {}
    tasks = payload.get("tasks")
    return tasks if isinstance(tasks, dict) else {}


def _save_checkpoint(
    checkpoint_path: Path | None,
    tasks: dict[str, list[dict[str, Any]]],
    *,
    k: int,
    model_name: str,
) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps(_checkpoint_payload(tasks, k=k, model_name=model_name), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _entries_by_task(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_task.setdefault(entry["task"], []).append(entry)
    return by_task


def _task_done(samples: list[dict[str, Any]], k: int) -> bool:
    return any(sample.get("demo_perfect") for sample in samples) or len(samples) >= k


def induce_pool_best_of_n_batched(
    entries: list[dict[str, Any]],
    sampler: Any,
    *,
    k: int,
    checkpoint_path: Path | None,
    model_name: str,
    started_s: float | None = None,
    max_wall_s: float | None = None,
    batch_size: int = DEFAULT_DRAW_BATCH_SIZE,
) -> dict[str, list[dict[str, Any]]]:
    """Run batched best-of-N induction per unique task and checkpoint after each task."""
    by_task = _entries_by_task(entries)
    samples_by_task = _load_checkpoint(checkpoint_path, k=k, model_name=model_name)
    for task_name in sorted(by_task):
        cached = samples_by_task.get(task_name)
        if isinstance(cached, list) and _task_done(cached, k):
            samples_by_task[task_name] = cached
            continue
        if (
            started_s is not None
            and max_wall_s is not None
            and time.time() - started_s >= max_wall_s
        ):
            break
        samples_by_task[task_name] = induce_task_samples_batched(
            task_name,
            by_task[task_name][0]["demos"],
            sampler,
            k=k,
            batch_size=batch_size,
        )
        _save_checkpoint(checkpoint_path, samples_by_task, k=k, model_name=model_name)
    return samples_by_task


def blocked_raw_artifact(
    verdict: str,
    *,
    chosen_model: dict[str, str] | None,
    preconditions: list[dict[str, Any]],
    duration_s: float,
    k: int = DEFAULT_K,
    output_path: Path | str = OUTPUT,
    pool_path: Path | str = POOL,
) -> dict[str, Any]:
    """Build a valid raw artifact for precondition-blocked runs."""
    artifact = {
        "experiment": "experiment_4048_decentralization_moe_base_raw",
        "schema": "carnot.experiment_4048_decentralization_moe_base_raw.v1",
        "title": "Decentralization MoE-base ARC best-of-N raw run",
        "honest_verdict": verdict,
        "runner_ready": False,
        "moe_base_model": chosen_model["name"] if chosen_model else "none",
        "best_of_n_coverage": 0.0,
        "local_demo_perfect_coverage_bestofn": 0.0,
        "k_samples_per_task": k,
        "gated_pass_at_2": 0.0,
        "local_gated_pass2": 0.0,
        "local_seconds": 0.0,
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


def _fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _raw_verdict(coverage: float, pass2: float) -> str:
    if coverage > BASELINE_12B_COVERAGE:
        return (
            "success: decentralization_moe_base_latent_support_cov"
            + _fmt(coverage)
            + "_pass2"
            + _fmt(pass2)
            + "_inducerqwen35moe"
        )
    return (
        "complete: decentralization_moe_base_cov"
        + _fmt(coverage)
        + "_pass2"
        + _fmt(pass2)
        + "_absent_or_flat"
    )


def _annotated_per_task(base_artifact: dict[str, Any]) -> list[dict[str, Any]]:
    summary = {row["task"]: row for row in base_artifact.get("per_task_sample_summary", [])}
    annotated = []
    for row in base_artifact.get("per_task", []):
        task_summary = summary.get(row["task"], {})
        annotated.append(
            {
                **row,
                "local_seconds": float(task_summary.get("local_seconds", 0.0)),
                "best_of_n_demo_perfect": bool(row.get("demo_perfect")),
                "n_demo_perfect_samples": int(task_summary.get("n_demo_perfect", 0)),
                "best_demo_fit": float(
                    task_summary.get("best_demo_fit", row.get("demo_fit") or 0.0)
                ),
            }
        )
    return annotated


def build_raw_artifact(
    *,
    base_artifact: dict[str, Any],
    chosen_model: dict[str, str],
    preconditions: list[dict[str, Any]],
    output_path: Path | str,
    pool_path: Path | str,
) -> dict[str, Any]:
    """Adapt the Exp 4012-compatible result into the Exp 4048 raw schema."""
    coverage = float(base_artifact["local_demo_perfect_coverage_bestofn"])
    pass2 = float(base_artifact["local_gated_pass2"])
    raw = {
        **base_artifact,
        "experiment": "experiment_4048_decentralization_moe_base_raw",
        "schema": "carnot.experiment_4048_decentralization_moe_base_raw.v1",
        "title": "Decentralization MoE-base ARC best-of-N raw run",
        "honest_verdict": _raw_verdict(coverage, pass2),
        "runner_ready": True,
        "moe_base_model": chosen_model["name"],
        "local_model_used": chosen_model["name"],
        "best_of_n_coverage": coverage,
        "gated_pass_at_2": pass2,
        "local_seconds": float(base_artifact["total_local_seconds"]),
        "per_task": _annotated_per_task(base_artifact),
        "model_specs": {
            **base_artifact["model_specs"],
            "generator_model": chosen_model["name"],
            "generator_hf_id": chosen_model["hf_id"],
            "generator_gguf_path": chosen_model["model_path"],
            "selection_note": chosen_model["selection_note"],
            "comparator_exp4012_model": "gemma-4-12B",
        },
        "reproducibility_checksum": _repro_checksum(output_path, pool_path),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "launched_pid": 0,
    }
    validate_raw_artifact(raw)
    return raw


def run(
    *,
    model_key: str = "auto",
    pool_path: Path | str = POOL,
    output_path: Path = OUTPUT,
    codex_ref_path: Path = CODEX_REF,
    checkpoint_path: Path | None = CHECKPOINT,
    k: int = DEFAULT_K,
    limit: int = 0,
    n_ctx: int = 16384,
    max_wall_s: float = 4500.0,
    batch_size: int = DEFAULT_DRAW_BATCH_SIZE,
    sampler: Any | None = None,
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
    cache_dir: Path | str = MOE_CACHE_DIR,
    llama_available_override: bool | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run the MoE best-of-N experiment or emit a blocked artifact."""
    started = time.time()
    preconditions, chosen_model = check_preconditions(
        model_key=model_key,
        pool_path=pool_path,
        resolver=resolver,
        cache_dir=cache_dir,
        llama_available_override=llama_available_override,
    )
    blocker = blocker_from_preconditions(preconditions)
    if blocker:
        artifact = blocked_raw_artifact(
            blocker,
            chosen_model=chosen_model,
            preconditions=preconditions,
            duration_s=time.time() - started,
            k=k,
            output_path=output_path,
            pool_path=pool_path,
        )
        if write:
            _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    if chosen_model is None:  # pragma: no cover - defensive; blocker should have caught this.
        raise RuntimeError("MoE model unavailable after precondition pass")

    with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
        pool = json.load(handle)
    entries = pool["entries"]
    if limit:
        entries = entries[:limit]

    if sampler is None:  # pragma: no cover - live multi-GB model load.
        llama = exp4012.load_local_llama(chosen_model["model_path"], n_ctx=n_ctx, seed=SEED)
        sampler = BatchedIndependentLocalSampler(llama, base_seed=SEED)

    samples_by_task = induce_pool_best_of_n_batched(
        entries,
        sampler,
        k=k,
        checkpoint_path=checkpoint_path,
        model_name=chosen_model["name"],
        started_s=started,
        max_wall_s=max_wall_s,
        batch_size=batch_size,
    )
    verifier_t0 = time.time()
    prog_by_entry_id = exp4012.build_entry_programs(entries, samples_by_task)
    scored = exp4012.score_best_of_n_pool(entries, prog_by_entry_id, seed=SEED)
    verifier_s = time.time() - verifier_t0
    base_artifact = exp4012.build_complete_artifact(
        entries=entries,
        samples_by_task=samples_by_task,
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
    )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    return artifact


def main() -> None:  # pragma: no cover - exercised by operator command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["auto", "qwen35moe"], default="auto")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--max-wall-s", type=float, default=4500.0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_DRAW_BATCH_SIZE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if args.k < 1 or args.k > 16:
        raise SystemExit("--k must be between 1 and 16")
    if args.batch_size < 1 or args.batch_size > 16:
        raise SystemExit("--batch-size must be between 1 and 16")
    run(
        model_key=args.model,
        k=args.k,
        limit=args.limit,
        n_ctx=args.n_ctx,
        max_wall_s=args.max_wall_s,
        batch_size=args.batch_size,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

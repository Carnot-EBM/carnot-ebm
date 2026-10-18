"""Exp 4069 synchronous MoE resume-accumulate ARC best-of-N runner.

Spec refs: REQ-VERIFY-4069, SCENARIO-VERIFY-4069.

This script is the mechanism fix for the Exp 4058/4059 split-background run. It
does one foreground resume, checkpoints after each task, prints progress after
each processed task, and writes the final diagnosis artifact directly.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from random import Random
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script bootstrap.
    sys.path.insert(0, str(REPO_ROOT))

import exp4048_decentralization_moe_base_collect as collect4048
import experiment_4012_gap4_local_best_of_n as exp4012
import experiment_4048_decentralization_moe_base_best_of_n as run4048
import experiment_4059_decentralization_moe_resume_best_of_n as resume4059
from experiment_4002_gap4_local_generator_arm import CODEX_REF, POOL, SEED

OUTPUT = REPO_ROOT / "results" / "experiment_4069_decentralization_moe_sync.json"
SOURCE_CHECKPOINT = (
    REPO_ROOT / "results" / "experiment_4048_decentralization_moe_base_raw.checkpoint.json"
)
STABLE_CHECKPOINT = (
    REPO_ROOT / "results" / "decentralization_moe_qwen35a3b_arc1_k8.checkpoint.json"
)
BASELINE_ARTIFACT = REPO_ROOT / "results" / "experiment_4012_gap4_local_best_of_n.json"
GAPS_PATH = REPO_ROOT / "ops" / "verifier_gaps.md"

INFERENCE_SUBSTRATE = "live_llm_inference"
MECHANISM = "single_synchronous_resume_accumulate_no_background"
MOE_MODEL_NAME = run4048.MOE_MODEL["name"]
DEFAULT_K = 8
DEFAULT_EXPECTED_UNIQUE_TASKS = 30
DEFAULT_SELF_BUDGET_S = 3000.0
DEFAULT_BOOTSTRAP_SAMPLES = 2000
ORACLE_SATURATION_TOLERANCE = 0.01
GAP_MARKER = "GAP-DECENTRALIZATION-MOE-SYNC-4069"

REQUIRED_FIELDS = [
    "honest_verdict",
    "moe_base_demo_perfect_coverage",
    "accumulated_n_tasks",
    "coverage_delta_vs_12b",
    "bootstrap_ci95",
    "oracle_coverage",
    "local_support_diagnosis",
    "local_seconds_per_task",
    "mechanism",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "missing_verifier_gaps",
    "preconditions_checked",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix and branch for the Invisible-Leash decision.",
    "moe_base_demo_perfect_coverage": "Gap-closing datum vs the 0.2581 12B ceiling.",
    "accumulated_n_tasks": "Resume metric: seeded checkpoint plus this foreground window.",
    "coverage_delta_vs_12b": "Point lift over the Exp 4012 12B local baseline.",
    "bootstrap_ci95": "CI excluding zero separates real base-size lift from noise.",
    "oracle_coverage": "Positive control: saturated oracle makes the pool uninformative.",
    "local_support_diagnosis": "latent, absent, uninformative, or still accumulating.",
    "local_seconds_per_task": "Sovereign local inference must report wall cost honestly.",
    "mechanism": "Records the no-background fix for the Exp 4058 timeout failure.",
}


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_pool_entries(pool_path: Path | str) -> list[dict[str, Any]] | None:
    try:
        with gzip.open(pool_path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return None
    return [entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("task"), str)]


def _unique_task_count(entries: list[dict[str, Any]] | None) -> int:
    if not entries:
        return 0
    return len({str(entry["task"]) for entry in entries})


def _reference_values(baseline_path: Path | str) -> dict[str, float]:
    return collect4048._reference_values(Path(baseline_path))


def check_preconditions(
    *,
    pool_path: Path | str = POOL,
    source_checkpoint_path: Path | str = SOURCE_CHECKPOINT,
    model_key: str = "auto",
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
    cache_dir: Path | str = run4048.MOE_CACHE_DIR,
    llama_available_override: bool | None = None,
    k: int = DEFAULT_K,
    expected_unique_tasks: int | None = DEFAULT_EXPECTED_UNIQUE_TASKS,
) -> tuple[list[dict[str, Any]], dict[str, str] | None, int]:
    """Check every resource before loading the live local model."""
    preconditions, chosen = run4048.check_preconditions(
        model_key=model_key,
        pool_path=pool_path,
        resolver=resolver,
        cache_dir=cache_dir,
        llama_available_override=llama_available_override,
    )
    entries = _read_pool_entries(pool_path)
    unique_tasks = _unique_task_count(entries)
    expected_ok = (
        entries is not None
        and (expected_unique_tasks is None or expected_unique_tasks <= 0 or unique_tasks == expected_unique_tasks)
    )
    preconditions.append(
        {
            "resource": "exp4012_arc1_30_task_pool",
            "available": expected_ok,
            "path": str(pool_path),
            "unique_tasks": unique_tasks,
            "expected_unique_tasks": expected_unique_tasks,
        }
    )
    model_name = chosen["name"] if chosen else MOE_MODEL_NAME
    source_tasks = resume4059._load_checkpoint_tasks(
        source_checkpoint_path, k=k, model_name=model_name
    )
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
    """Return the required blocked verdict for the first failed precondition."""
    by_resource = {str(row.get("resource")): bool(row.get("available")) for row in preconditions}
    if not by_resource.get("moe_base_gguf_cached", False):
        return "blocked_moe_base_not_cached"
    if not by_resource.get("llama_cpp", False):
        return "blocked_llama_cpp_unavailable"
    if not by_resource.get("exp4012_arc1_pool_and_verifier_primitives", False):
        return "blocked_exp4012_pool_unreadable"
    if not by_resource.get("exp4012_arc1_30_task_pool", False):
        return "blocked_exp4012_pool_unreadable"
    if not by_resource.get("exp4048_checkpoint", False):
        return "blocked_exp4048_checkpoint_unreadable"
    return None


def _rows_from_samples(
    samples_by_task: dict[str, list[dict[str, Any]]],
    *,
    k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, samples in sorted(samples_by_task.items()):
        sample_rows = samples if isinstance(samples, list) else []
        if not run4048._task_done(sample_rows, k):
            continue
        n_perfect = sum(1 for sample in sample_rows if bool(sample.get("demo_perfect")))
        rows.append(
            {
                "task": str(task),
                "best_of_n_demo_perfect": n_perfect > 0,
                "n_demo_perfect_samples": n_perfect,
                "local_seconds": round(
                    sum(float(sample.get("local_s", 0.0) or 0.0) for sample in sample_rows),
                    2,
                ),
            }
        )
    return rows


def _coverage_indicators(rows: list[dict[str, Any]]) -> list[int]:
    return [
        1
        if bool(row.get("best_of_n_demo_perfect"))
        or int(row.get("n_demo_perfect_samples", 0) or 0) > 0
        else 0
        for row in rows
    ]


def _coverage(indicators: list[int]) -> float:
    if not indicators:
        return 0.0
    return round(sum(indicators) / len(indicators), 4)


def _seconds_per_task(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(float(row.get("local_seconds", 0.0) or 0.0) for row in rows) / len(rows), 2)


def _missing_gaps(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row.get("task"))
        for row in rows
        if not bool(row.get("best_of_n_demo_perfect"))
        and int(row.get("n_demo_perfect_samples", 0) or 0) == 0
    ]


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = int(round((len(sorted_values) - 1) * percentile))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def bootstrap_delta_ci95(
    indicators: list[int],
    baseline_coverage: float,
    *,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = SEED,
) -> list[float]:
    """Return deterministic bootstrap CI95 for MoE coverage minus baseline."""
    if not indicators:
        return [0.0, 0.0]
    rng = Random(seed)
    n = len(indicators)
    deltas = []
    for _ in range(n_bootstrap):
        sample_sum = sum(indicators[rng.randrange(n)] for _j in range(n))
        deltas.append(sample_sum / n - baseline_coverage)
    deltas.sort()
    return [round(_percentile(deltas, 0.025), 4), round(_percentile(deltas, 0.975), 4)]


def _oracle_saturated(references: dict[str, float]) -> bool:
    return (
        references["oracle_coverage"] - references["coverage_12b"]
        <= ORACLE_SATURATION_TOLERANCE
    )


def _diagnosis(*, accumulated_n: int, ci95: list[float], references: dict[str, float]) -> str:
    if accumulated_n < DEFAULT_EXPECTED_UNIQUE_TASKS:
        return "accumulating"
    if _oracle_saturated(references):
        return "uninformative"
    if ci95[0] > 0.0:
        return "latent"
    return "absent"


def _fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _verdict(coverage: float, diagnosis: str, accumulated_n: int) -> str:
    if diagnosis == "accumulating":
        return f"complete: decentralization_moe_accumulating_n_{accumulated_n}"
    if diagnosis == "latent":
        return f"complete: decentralization_moe_cov_{_fmt(coverage)}_latent_distill_viable"
    if diagnosis == "absent":
        return f"complete: decentralization_moe_cov_{_fmt(coverage)}_absent_leash_holds_n30"
    return f"complete: decentralization_moe_cov_{_fmt(coverage)}_uninformative_saturated_pool"


def _pass2_comparison(pass2: float, references: dict[str, float]) -> dict[str, float]:
    return {
        "moe_base_gated_pass_at_2": round(pass2, 4),
        "exp4012_12b_gated_pass_at_2": round(references["pass2_12b"], 4),
        "oracle_gated_pass_at_2": round(references["oracle_coverage"], 4),
        "codex_gated_pass_at_2": round(references["codex_pass2"], 4),
        "vs_exp4012_12b_gated_pass2": round(pass2 - references["pass2_12b"], 4),
        "vs_oracle_gated_pass2": round(pass2 - references["oracle_coverage"], 4),
        "vs_codex_gated_pass2": round(pass2 - references["codex_pass2"], 4),
    }


def _stable_checksum(*, seed: int, payloads: list[Any]) -> str:
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    blob = json.dumps(
        {"payloads": payloads, "script_digest": script_digest, "seed": seed},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_terminal_artifact(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    preconditions: list[dict[str, Any]],
    model_specs: dict[str, Any],
    random_seed: int,
    duration_s: float,
    resumed_from_n: int,
    new_tasks_processed: int,
    pass2: float,
    references: dict[str, float],
    stable_checkpoint_path: Path | str,
    source_checkpoint_path: Path | str,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    """Build the required Exp 4069 terminal artifact from checkpointed task rows."""
    indicators = _coverage_indicators(rows)
    coverage = _coverage(indicators)
    delta = round(coverage - references["coverage_12b"], 4)
    ci95 = bootstrap_delta_ci95(
        indicators,
        references["coverage_12b"],
        n_bootstrap=n_bootstrap,
        seed=random_seed,
    )
    accumulated_n = len(rows)
    diagnosis = _diagnosis(accumulated_n=accumulated_n, ci95=ci95, references=references)
    artifact = {
        "experiment": "experiment_4069_decentralization_moe_sync",
        "schema": "carnot.experiment_4069_decentralization_moe_sync.v1",
        "title": "Synchronous MoE resume-accumulate ARC best-of-N diagnosis",
        "honest_verdict": _verdict(coverage, diagnosis, accumulated_n),
        "moe_base_demo_perfect_coverage": coverage,
        "accumulated_n_tasks": accumulated_n,
        "coverage_delta_vs_12b": delta,
        "bootstrap_ci95": ci95,
        "oracle_coverage": round(references["oracle_coverage"], 4),
        "oracle_positive_control_saturated": _oracle_saturated(references),
        "local_support_diagnosis": diagnosis,
        "local_seconds_per_task": _seconds_per_task(rows),
        "mechanism": MECHANISM,
        "model_specs": model_specs,
        "random_seed": random_seed,
        "reproducibility_checksum": _stable_checksum(
            seed=random_seed,
            payloads=[rows, references, model_specs, str(stable_checkpoint_path)],
        ),
        "missing_verifier_gaps": _missing_gaps(rows),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "output_artifact_path": str(output_path),
        "stable_checkpoint_path": str(stable_checkpoint_path),
        "source_checkpoint_path": str(source_checkpoint_path),
        "resumed_from_n": int(resumed_from_n),
        "new_tasks_processed": int(new_tasks_processed),
        "target_n_tasks": DEFAULT_EXPECTED_UNIQUE_TASKS,
        "k_samples_per_task": DEFAULT_K,
        "n_demo_perfect_tasks": int(sum(indicators)),
        "gated_pass_at_2": round(pass2, 4),
        "pass2_comparison": _pass2_comparison(pass2, references),
        "codex_seconds_per_task_reference": round(references["codex_seconds"], 2),
        "local_vs_codex_seconds_ratio": (
            round(_seconds_per_task(rows) / references["codex_seconds"], 4)
            if references["codex_seconds"]
            else 0.0
        ),
        "duration_s": round(float(duration_s), 2),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    verdict: str,
    *,
    preconditions: list[dict[str, Any]],
    references: dict[str, float],
    rows: list[dict[str, Any]],
    output_path: Path,
    duration_s: float,
    model_specs: dict[str, Any] | None = None,
    resumed_from_n: int = 0,
) -> dict[str, Any]:
    """Build a terminal artifact for precondition-blocked runs."""
    model_specs = model_specs or {"generator_model": "none"}
    indicators = _coverage_indicators(rows)
    coverage = _coverage(indicators)
    artifact = {
        "experiment": "experiment_4069_decentralization_moe_sync",
        "schema": "carnot.experiment_4069_decentralization_moe_sync.v1",
        "title": "Synchronous MoE resume-accumulate ARC best-of-N diagnosis",
        "honest_verdict": verdict,
        "moe_base_demo_perfect_coverage": coverage,
        "accumulated_n_tasks": len(rows),
        "coverage_delta_vs_12b": round(coverage - references["coverage_12b"], 4),
        "bootstrap_ci95": bootstrap_delta_ci95(
            indicators,
            references["coverage_12b"],
            n_bootstrap=min(128, DEFAULT_BOOTSTRAP_SAMPLES),
            seed=SEED,
        ),
        "oracle_coverage": round(references["oracle_coverage"], 4),
        "oracle_positive_control_saturated": _oracle_saturated(references),
        "local_support_diagnosis": "uninformative",
        "local_seconds_per_task": _seconds_per_task(rows),
        "mechanism": MECHANISM,
        "model_specs": model_specs,
        "random_seed": SEED,
        "reproducibility_checksum": _stable_checksum(
            seed=SEED, payloads=[verdict, preconditions, rows, references]
        ),
        "missing_verifier_gaps": _missing_gaps(rows),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
        "output_artifact_path": str(output_path),
        "stable_checkpoint_path": str(STABLE_CHECKPOINT),
        "source_checkpoint_path": str(SOURCE_CHECKPOINT),
        "resumed_from_n": int(resumed_from_n),
        "new_tasks_processed": 0,
        "target_n_tasks": DEFAULT_EXPECTED_UNIQUE_TASKS,
        "k_samples_per_task": DEFAULT_K,
        "n_demo_perfect_tasks": int(sum(indicators)),
        "gated_pass_at_2": 0.0,
        "pass2_comparison": _pass2_comparison(0.0, references),
        "codex_seconds_per_task_reference": round(references["codex_seconds"], 2),
        "local_vs_codex_seconds_ratio": 0.0,
        "duration_s": round(float(duration_s), 2),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
        or verdict.startswith("failed:")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in (
        "moe_base_demo_perfect_coverage",
        "coverage_delta_vs_12b",
        "oracle_coverage",
        "local_seconds_per_task",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    if not _is_bare_int(artifact["accumulated_n_tasks"]):
        raise ValueError("accumulated_n_tasks must be a bare int")
    ci95 = artifact["bootstrap_ci95"]
    if not (
        isinstance(ci95, list)
        and len(ci95) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in ci95)
    ):
        raise ValueError("bootstrap_ci95 must be a 2-element numeric list")
    if artifact["local_support_diagnosis"] not in {
        "latent",
        "absent",
        "uninformative",
        "accumulating",
    }:
        raise ValueError("local_support_diagnosis must be latent, absent, uninformative, or accumulating")
    if artifact["mechanism"] != MECHANISM:
        raise ValueError("mechanism must be single synchronous resume-accumulate")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be a dict")
    if not _is_bare_int(artifact["random_seed"]):
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["reproducibility_checksum"], str):
        raise ValueError("reproducibility_checksum must be a string")
    if not isinstance(artifact["missing_verifier_gaps"], list):
        raise ValueError("missing_verifier_gaps must be a list")
    if not isinstance(artifact["preconditions_checked"], list):
        raise ValueError("preconditions_checked must be a list")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")


def record_verifier_gaps(gaps_path: Path, artifact: dict[str, Any]) -> bool:
    gaps = [str(item) for item in artifact.get("missing_verifier_gaps", [])]
    if not gaps:
        return False
    existing = gaps_path.read_text(encoding="utf-8") if gaps_path.exists() else "# Verifier Gaps\n"
    if GAP_MARKER in existing:
        return False
    entry = (
        "\n"
        f"### {GAP_MARKER}: Synchronous MoE local support gaps\n"
        "- status: open\n"
        f"- evidence: {artifact.get('output_artifact_path', OUTPUT)}; "
        f"accumulated_n={artifact.get('accumulated_n_tasks')}; "
        f"coverage={artifact.get('moe_base_demo_perfect_coverage')}; "
        f"diagnosis={artifact.get('local_support_diagnosis')}\n"
        "- failure mode: Qwen3.6-35B-A3B best-of-N did not surface a demo-perfect "
        "local program for the listed ARC tasks under the unchanged GAP-4 verifier.\n"
        "- missing discriminator: a local candidate source or verifier-side signal "
        "that recovers the demonstrated rule before distillation.\n"
        "- priority: high\n"
        f"- missing_verifier_gaps: {', '.join(gaps)}\n"
    )
    gaps_path.parent.mkdir(parents=True, exist_ok=True)
    gaps_path.write_text(existing.rstrip() + "\n" + entry, encoding="utf-8")
    return True


def _run_summarizer(path: Path) -> dict[str, Any]:  # pragma: no cover - subprocess wrapper.
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "summarize_artifact.py"), str(path)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)
    if proc.stderr.strip():
        print(proc.stderr.strip(), file=sys.stderr, flush=True)
    return {"returncode": proc.returncode}


def _finalize_artifact(
    artifact: dict[str, Any],
    *,
    output_path: Path,
    gaps_path: Path,
    write: bool,
    run_summarizer: bool,
    summarizer_fn: Callable[[Path], dict[str, Any]] | None,
) -> dict[str, Any]:
    record_verifier_gaps(gaps_path, artifact)
    if write:
        _write_json(output_path, artifact)
        if run_summarizer:
            summary = (summarizer_fn or _run_summarizer)(output_path)
            artifact["summarize_artifact"] = summary
            _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    return artifact


def _progress_line(
    *,
    done_count: int,
    total_tasks: int,
    demo_perfect: bool,
    rows: list[dict[str, Any]],
    started_s: float,
) -> str:
    return (
        f"[moe] task {done_count}/{total_tasks} demo_perfect={demo_perfect} "
        f"cov={_coverage(_coverage_indicators(rows)):.4f} elapsed={int(time.time() - started_s)}s"
    )


def accumulate_synchronously(
    entries: list[dict[str, Any]],
    sampler: Any,
    *,
    samples_by_task: dict[str, list[dict[str, Any]]],
    stable_checkpoint_path: Path | str,
    k: int,
    model_name: str,
    started_s: float,
    max_wall_s: float,
    max_new_tasks: int,
    batch_size: int,
    progress_fn: Callable[[str], None],
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Sample unfinished tasks in the foreground and checkpoint after each task."""
    by_task = resume4059._entries_by_task(entries)
    total_tasks = len(by_task)
    processed: list[str] = []
    for task_name in sorted(by_task):
        cached = samples_by_task.get(task_name)
        if isinstance(cached, list) and run4048._task_done(cached, k):
            continue
        if max_new_tasks and len(processed) >= max_new_tasks:
            break
        if time.time() - started_s >= max_wall_s:
            break
        samples = run4048.induce_task_samples_batched(
            task_name,
            by_task[task_name][0]["demos"],
            sampler,
            k=k,
            batch_size=batch_size,
        )
        samples_by_task[task_name] = samples
        processed.append(task_name)
        resume4059._save_checkpoint(stable_checkpoint_path, samples_by_task, k=k, model_name=model_name)
        rows = _rows_from_samples(samples_by_task, k=k)
        progress_fn(
            _progress_line(
                done_count=len(rows),
                total_tasks=total_tasks,
                demo_perfect=any(bool(sample.get("demo_perfect")) for sample in samples),
                rows=rows,
                started_s=started_s,
            )
        )
    return samples_by_task, processed


def _score_pass2(
    entries: list[dict[str, Any]],
    samples_by_task: dict[str, list[dict[str, Any]]],
    *,
    k: int,
) -> float:
    scored_entries = [
        entry
        for entry in entries
        if run4048._task_done(samples_by_task.get(str(entry["task"]), []), k)
    ]
    if not scored_entries:
        return 0.0
    done_samples = {
        task: samples
        for task, samples in samples_by_task.items()
        if run4048._task_done(samples, k)
    }
    prog_by_entry_id = exp4012.build_entry_programs(scored_entries, done_samples)
    scored = exp4012.score_best_of_n_pool(scored_entries, prog_by_entry_id, seed=SEED)
    return float(scored.get("g2", 0.0))


def _model_specs(chosen_model: dict[str, str] | None) -> dict[str, Any]:
    return {
        "generator_model": chosen_model["name"] if chosen_model else "none",
        "generator_hf_id": chosen_model["hf_id"] if chosen_model else run4048.MOE_MODEL["hf_id"],
        "generator_gguf_path": chosen_model["model_path"] if chosen_model else "none",
        "verifier": "model-free GAP-4 verifier primitives reused unchanged",
        "mechanism": MECHANISM,
    }


def run(
    *,
    model_key: str = "auto",
    pool_path: Path | str = POOL,
    output_path: Path = OUTPUT,
    baseline_path: Path | str = BASELINE_ARTIFACT,
    source_checkpoint_path: Path | str = SOURCE_CHECKPOINT,
    stable_checkpoint_path: Path | str = STABLE_CHECKPOINT,
    gaps_path: Path = GAPS_PATH,
    k: int = DEFAULT_K,
    n_ctx: int = 16384,
    max_wall_s: float = DEFAULT_SELF_BUDGET_S,
    max_new_tasks: int = 0,
    batch_size: int = run4048.DEFAULT_DRAW_BATCH_SIZE,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sampler: Any | None = None,
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
    cache_dir: Path | str = run4048.MOE_CACHE_DIR,
    llama_available_override: bool | None = None,
    expected_unique_tasks: int | None = DEFAULT_EXPECTED_UNIQUE_TASKS,
    progress_fn: Callable[[str], None] = print,
    write: bool = True,
    run_summarizer: bool = True,
    summarizer_fn: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the foreground resume or write a blocked/failed terminal artifact."""
    started = time.time()
    preconditions: list[dict[str, Any]] = []
    chosen_model: dict[str, str] | None = None
    references = _reference_values(baseline_path)
    try:
        preconditions, chosen_model, resumed_from_n = check_preconditions(
            pool_path=pool_path,
            source_checkpoint_path=source_checkpoint_path,
            model_key=model_key,
            resolver=resolver,
            cache_dir=cache_dir,
            llama_available_override=llama_available_override,
            k=k,
            expected_unique_tasks=expected_unique_tasks,
        )
        source_tasks = resume4059._load_checkpoint_tasks(
            source_checkpoint_path,
            k=k,
            model_name=chosen_model["name"] if chosen_model else MOE_MODEL_NAME,
        )
        source_rows = _rows_from_samples(source_tasks or {}, k=k)
        blocker = blocker_from_preconditions(preconditions)
        if blocker:
            artifact = blocked_artifact(
                blocker,
                preconditions=preconditions,
                references=references,
                rows=source_rows,
                output_path=output_path,
                duration_s=time.time() - started,
                model_specs=_model_specs(chosen_model),
                resumed_from_n=resumed_from_n,
            )
            return _finalize_artifact(
                artifact,
                output_path=output_path,
                gaps_path=gaps_path,
                write=write,
                run_summarizer=run_summarizer,
                summarizer_fn=summarizer_fn,
            )

        if chosen_model is None:  # pragma: no cover - blocker catches this.
            raise RuntimeError("MoE model unavailable after precondition pass")
        entries = _read_pool_entries(pool_path)
        if entries is None:  # pragma: no cover - precondition catches this.
            raise RuntimeError("Exp 4012 pool became unreadable after precondition pass")

        samples_by_task, resumed_from_n = resume4059.ensure_stable_checkpoint(
            source_checkpoint_path=source_checkpoint_path,
            stable_checkpoint_path=stable_checkpoint_path,
            k=k,
            model_name=chosen_model["name"],
        )
        if sampler is None:  # pragma: no cover - live multi-GB model load.
            llama = exp4012.load_local_llama(chosen_model["model_path"], n_ctx=n_ctx, seed=SEED)
            sampler = run4048.BatchedIndependentLocalSampler(llama, base_seed=SEED)

        samples_by_task, processed_tasks = accumulate_synchronously(
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
            progress_fn=progress_fn,
        )
        rows = _rows_from_samples(samples_by_task, k=k)
        pass2 = _score_pass2(entries, samples_by_task, k=k)
        artifact = build_terminal_artifact(
            rows=rows,
            output_path=output_path,
            preconditions=preconditions,
            model_specs=_model_specs(chosen_model),
            random_seed=SEED,
            duration_s=time.time() - started,
            resumed_from_n=resumed_from_n,
            new_tasks_processed=len(processed_tasks),
            pass2=pass2,
            references=references,
            stable_checkpoint_path=stable_checkpoint_path,
            source_checkpoint_path=source_checkpoint_path,
            n_bootstrap=n_bootstrap,
        )
        return _finalize_artifact(
            artifact,
            output_path=output_path,
            gaps_path=gaps_path,
            write=write,
            run_summarizer=run_summarizer,
            summarizer_fn=summarizer_fn,
        )
    except Exception as exc:  # pragma: no cover - defensive terminal artifact path.
        rows = []
        try:
            stable_tasks = resume4059._load_checkpoint_tasks(
                stable_checkpoint_path,
                k=k,
                model_name=chosen_model["name"] if chosen_model else MOE_MODEL_NAME,
            )
            rows = _rows_from_samples(stable_tasks or {}, k=k)
        except Exception:
            rows = []
        artifact = blocked_artifact(
            f"failed: decentralization_moe_sync_exception_{type(exc).__name__}",
            preconditions=preconditions,
            references=references,
            rows=rows,
            output_path=output_path,
            duration_s=time.time() - started,
            model_specs=_model_specs(chosen_model),
            resumed_from_n=len(rows),
        )
        artifact["failure_message"] = str(exc)
        return _finalize_artifact(
            artifact,
            output_path=output_path,
            gaps_path=gaps_path,
            write=write,
            run_summarizer=run_summarizer,
            summarizer_fn=summarizer_fn,
        )


def main() -> None:  # pragma: no cover - exercised by operator command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["auto", "qwen35moe"], default="auto")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--max-wall-s", type=float, default=DEFAULT_SELF_BUDGET_S)
    parser.add_argument("--max-new-tasks", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=run4048.DEFAULT_DRAW_BATCH_SIZE)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
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
        n_ctx=args.n_ctx,
        max_wall_s=args.max_wall_s,
        max_new_tasks=args.max_new_tasks,
        batch_size=args.batch_size,
        n_bootstrap=args.bootstrap_samples,
        output_path=args.output,
        source_checkpoint_path=args.source_checkpoint,
        stable_checkpoint_path=args.stable_checkpoint,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

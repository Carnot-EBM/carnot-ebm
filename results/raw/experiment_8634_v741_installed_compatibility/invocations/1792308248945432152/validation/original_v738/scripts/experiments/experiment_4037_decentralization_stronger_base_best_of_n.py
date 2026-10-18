"""Exp 4037 stronger-base local best-of-N ARC induction runner.

Spec refs: REQ-VERIFY-4036, SCENARIO-VERIFY-4036.

This is the run half of the decentralization stronger-base check. It mirrors
Exp 4012's pool, best-of-N sampling, verifier primitives, candidate snap, and
GAP-4 gated pass@2 scoring. The only intended change is the local inducer base:
prefer cached Gemma4-31B-it, otherwise use cached Qwen3.6-35B-A3B.
"""

from __future__ import annotations

import argparse
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

OUTPUT = REPO_ROOT / "results" / "experiment_4037_decentralization_stronger_base_raw.json"
CHECKPOINT = (
    REPO_ROOT / "results" / "experiment_4037_decentralization_stronger_base_raw.checkpoint.json"
)

INFERENCE_SUBSTRATE = "live_llm_inference"
DEFAULT_K = 8

STRONGER_MODELS: dict[str, dict[str, str]] = {
    "gemma31": {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "selection_note": "preferred cached faster stronger base for ARC code emission",
    },
    "qwen35": {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "selection_note": "fallback cached stronger base",
    },
}
PREFERRED_MODEL_KEYS = ("gemma31", "qwen35")

REQUIRED_RAW_FIELDS = [
    "honest_verdict",
    "runner_ready",
    "stronger_base_model",
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


def select_stronger_model(
    model_key: str = "auto",
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
) -> dict[str, str] | None:
    """Pick the cached stronger base, preferring Gemma4-31B before Qwen."""
    keys = PREFERRED_MODEL_KEYS if model_key == "auto" else (model_key,)
    for key in keys:
        spec = STRONGER_MODELS.get(key)
        if spec is None:
            continue
        model_path = resolver(spec["hf_id"])
        if model_path:
            return {**spec, "model_key": key, "model_path": str(model_path)}
    return None


def check_preconditions(
    *,
    model_key: str,
    pool_path: Path | str,
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
    llama_available_override: bool | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str] | None]:
    """Check all resources before any live inference is attempted."""
    chosen = select_stronger_model(model_key, resolver=resolver)
    if llama_available_override is None:
        try:
            import llama_cpp  # noqa: F401

            llama_ok = True
        except Exception:
            llama_ok = False
    else:
        llama_ok = bool(llama_available_override)

    pool_ok = exp4012._pool_and_verifier_loadable(pool_path)
    preconditions = [
        {
            "resource": "stronger_base_gguf_cached",
            "available": chosen is not None,
            "selected_model": chosen["name"] if chosen else None,
        },
        {"resource": "llama_cpp", "available": llama_ok},
        {"resource": "exp4012_arc1_pool_and_verifier_primitives", "available": pool_ok},
    ]
    return preconditions, chosen


def blocker_from_preconditions(preconditions: list[dict[str, Any]]) -> str | None:
    """Map the first failed precondition to the build/run blocker string."""
    by_resource = {row["resource"]: bool(row["available"]) for row in preconditions}
    if not by_resource.get("stronger_base_gguf_cached", False):
        return "blocked_stronger_base_not_cached"
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
    """Validate the raw artifact fields consumed by the Exp 4036 build gate."""
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
    for field in ("runner_ready",):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
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
    for field in ("stronger_base_model", "inference_substrate", "reproducibility_checksum"):
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
        "experiment": "experiment_4037_decentralization_stronger_base_raw",
        "schema": "carnot.experiment_4037_decentralization_stronger_base_raw.v1",
        "title": "Decentralization stronger-base ARC best-of-N raw run",
        "honest_verdict": verdict,
        "runner_ready": False,
        "stronger_base_model": chosen_model["name"] if chosen_model else "none",
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


def _raw_verdict(local_beats_vote: bool, coverage: float, pass2: float, model_name: str) -> str:
    model_slug = model_name.replace("/", "_")
    if local_beats_vote:
        return (
            "success: decentralization_stronger_base_latent_support_cov"
            + _fmt(coverage)
            + "_pass2"
            + _fmt(pass2)
            + "_inducer"
            + model_slug
        )
    return (
        "complete: decentralization_stronger_base_cov"
        + _fmt(coverage)
        + "_pass2"
        + _fmt(pass2)
        + "_below_codex"
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
    """Adapt the Exp 4012-compatible result into the Exp 4037 raw schema."""
    coverage = float(base_artifact["local_demo_perfect_coverage_bestofn"])
    pass2 = float(base_artifact["local_gated_pass2"])
    raw = {
        **base_artifact,
        "experiment": "experiment_4037_decentralization_stronger_base_raw",
        "schema": "carnot.experiment_4037_decentralization_stronger_base_raw.v1",
        "title": "Decentralization stronger-base ARC best-of-N raw run",
        "honest_verdict": _raw_verdict(
            bool(base_artifact["local_beats_vote"]), coverage, pass2, chosen_model["name"]
        ),
        "runner_ready": True,
        "stronger_base_model": chosen_model["name"],
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    sampler: Callable[[str, int], tuple[str, float]] | None = None,
    resolver: Callable[[str], str | None] = exp4012.resolve_local_gguf,
    llama_available_override: bool | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run the stronger-base best-of-N experiment or emit a blocked artifact."""
    started = time.time()
    preconditions, chosen_model = check_preconditions(
        model_key=model_key,
        pool_path=pool_path,
        resolver=resolver,
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

    if sampler is None:  # pragma: no cover - live multi-GB model load.
        llama = exp4012.load_local_llama(chosen_model["model_path"], n_ctx=n_ctx, seed=SEED)
        sampler = exp4012.IndependentLocalSampler(llama, base_seed=SEED)

    base_artifact = exp4012.run(
        model_key="gemma12",
        pool_path=pool_path,
        output_path=output_path,
        codex_ref_path=codex_ref_path,
        checkpoint_path=checkpoint_path,
        k=k,
        limit=limit,
        n_ctx=n_ctx,
        max_wall_s=max_wall_s,
        sampler=sampler,
        resolver=lambda _hf_id: chosen_model["model_path"],
        llama_available_override=True,
        write=False,
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
    parser.add_argument("--model", choices=["auto", *STRONGER_MODELS], default="auto")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--n-ctx", type=int, default=16384)
    parser.add_argument("--max-wall-s", type=float, default=4500.0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = parser.parse_args()
    if args.k < 1 or args.k > 16:
        raise SystemExit("--k must be between 1 and 16")
    run(
        model_key=args.model,
        k=args.k,
        limit=args.limit,
        n_ctx=args.n_ctx,
        max_wall_s=args.max_wall_s,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

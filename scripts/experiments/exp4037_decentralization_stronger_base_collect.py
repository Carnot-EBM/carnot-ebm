"""Collect and diagnose Exp 4037 stronger-base sovereign coverage.

Spec refs: REQ-VERIFY-4037, SCENARIO-VERIFY-4037.

The collector does not run inference. It reads the Exp 4036 build receipt, polls
for the Exp 4037 raw artifact, validates the raw schema, and writes the terminal
coverage diagnosis. If the raw run never materializes, it reports only the
checkpoint-backed partial coverage that is actually on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from random import Random
from typing import Any, Callable

import experiment_4037_decentralization_stronger_base_best_of_n as run4037

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_ARTIFACT = REPO_ROOT / "results" / "experiment_4036_decentralization_stronger_base_build.json"
RAW_ARTIFACT = (
    REPO_ROOT / "results" / "experiment_4037_decentralization_stronger_base_raw.json"
)
BASELINE_ARTIFACT = REPO_ROOT / "results" / "experiment_4012_gap4_local_best_of_n.json"
CHECKPOINT_ARTIFACT = (
    REPO_ROOT / "results" / "experiment_4037_decentralization_stronger_base_raw.checkpoint.json"
)
LOG_PATH = REPO_ROOT / "logs" / "decentralization_run.log"
OUTPUT = REPO_ROOT / "results" / "experiment_4037_decentralization_stronger_base.json"

INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DEFAULT_BASELINE_COVERAGE = 0.2581
DEFAULT_BASELINE_PASS2 = 0.4516
DEFAULT_ORACLE_PASS2 = 0.6129
DEFAULT_CODEX_PASS2 = 0.5806
DEFAULT_CODEX_SECONDS_PER_TASK = 46.24
DEFAULT_POLL_BUDGET_S = 30.0
DEFAULT_POLL_INTERVAL_S = 5.0
DEFAULT_BOOTSTRAP_SAMPLES = 2000

REQUIRED_FINAL_FIELDS = [
    "honest_verdict",
    "stronger_base_demo_perfect_coverage",
    "coverage_delta_vs_12b",
    "bootstrap_ci95",
    "local_support_diagnosis",
    "local_seconds_per_task",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "missing_verifier_gaps",
    "inference_substrate",
]

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal diagnosis; partial artifacts stay explicit instead of fabricating completion.",
    "stronger_base_demo_perfect_coverage": (
        "The gap-closing datum vs the 0.2581 12B ceiling: is the abstraction in a bigger local model's support?"
    ),
    "coverage_delta_vs_12b": "Coverage lift relative to the Exp 4012 12B baseline.",
    "bootstrap_ci95": "A CI excluding 0 distinguishes a real base-size lift from noise.",
    "local_support_diagnosis": (
        "Invisible-Leash branch: latent means distillation is viable; absent means a stronger base is needed."
    ),
    "local_seconds_per_task": (
        "Sovereignty must be cost-honest; a viable but slower path is a tradeoff, not a free win."
    ),
    "missing_verifier_gaps": "Tasks where no local best-of-N sample surfaced a demo-perfect verifier candidate.",
    "inference_substrate": "Final diagnosis runs over cached verifier artifacts, not fresh inference.",
}


def _is_bare_float(value: Any) -> bool:
    return isinstance(value, float) and not isinstance(value, bool)


def _is_bare_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.exists():
        return None, f"missing:{path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"malformed:{path}:{exc}"
    if not isinstance(payload, dict):
        return None, f"malformed:{path}:top_level_not_object"
    return payload, None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _tail_text(path: Path, max_lines: int = 20) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _reference_values(baseline_path: Path) -> dict[str, float]:
    baseline, _error = _read_json(baseline_path)
    references = baseline.get("references", {}) if baseline else {}
    return {
        "coverage_12b": float(
            baseline.get("local_demo_perfect_coverage_bestofn", DEFAULT_BASELINE_COVERAGE)
            if baseline
            else DEFAULT_BASELINE_COVERAGE
        ),
        "pass2_12b": float(
            baseline.get("local_gated_pass2", DEFAULT_BASELINE_PASS2)
            if baseline
            else DEFAULT_BASELINE_PASS2
        ),
        "oracle_pass2": float(references.get("oracle_pass2", DEFAULT_ORACLE_PASS2)),
        "codex_pass2": float(references.get("codex_gated_pass2", DEFAULT_CODEX_PASS2)),
        "codex_seconds": float(
            baseline.get("cost_codex_seconds_ref", DEFAULT_CODEX_SECONDS_PER_TASK)
            if baseline
            else DEFAULT_CODEX_SECONDS_PER_TASK
        ),
    }


def _stable_checksum(*, seed: int, payloads: list[Any]) -> str:
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    blob = json.dumps(
        {"payloads": payloads, "script_digest": script_digest, "seed": seed},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _coverage_indicators(rows: list[dict[str, Any]]) -> list[int]:
    return [
        1
        if bool(row.get("best_of_n_demo_perfect", row.get("demo_perfect", False)))
        or int(row.get("n_demo_perfect_samples", 0) or 0) > 0
        else 0
        for row in rows
    ]


def _coverage_from_indicators(indicators: list[int]) -> float:
    if not indicators:
        return 0.0
    return round(sum(indicators) / len(indicators), 4)


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
    seed: int = run4037.SEED,
) -> list[float]:
    """Return a deterministic bootstrap CI for stronger coverage minus baseline."""
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


def _complete_raw_payload(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    payload, error = _read_json(path)
    if payload is None:
        return None, error
    try:
        run4037.validate_raw_artifact(payload)
    except ValueError as exc:
        return None, f"raw_schema_invalid:{exc}"
    if not payload.get("runner_ready"):
        return None, "raw_runner_not_ready"
    if str(payload.get("honest_verdict", "")).startswith("blocked_"):
        return None, "raw_blocked"
    if not payload.get("per_task"):
        return None, "raw_has_no_per_task_rows"
    return payload, None


def poll_raw_artifact(
    raw_path: Path,
    *,
    poll_budget_s: float,
    poll_interval_s: float,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any] | None, str | None, int]:
    interval = max(0.001, float(poll_interval_s))
    attempts = max(1, int(float(poll_budget_s) / interval) + 1)
    last_error = "raw_artifact_absent"
    for attempt in range(attempts):
        payload, error = _complete_raw_payload(raw_path)
        if payload is not None:
            return payload, None, attempt + 1
        last_error = error or last_error
        if attempt + 1 < attempts:
            sleep_fn(interval)
    return None, last_error, attempts


def _partial_rows_from_checkpoint(checkpoint_path: Path) -> list[dict[str, Any]]:
    checkpoint, _error = _read_json(checkpoint_path)
    tasks = checkpoint.get("tasks", {}) if checkpoint else {}
    if not isinstance(tasks, dict):
        return []
    rows = []
    for task, samples in sorted(tasks.items()):
        sample_rows = samples if isinstance(samples, list) else []
        demo_perfect = any(bool(sample.get("demo_perfect")) for sample in sample_rows)
        local_seconds = sum(float(sample.get("local_s", 0.0) or 0.0) for sample in sample_rows)
        rows.append(
            {
                "task": str(task),
                "best_of_n_demo_perfect": demo_perfect,
                "n_demo_perfect_samples": sum(
                    1 for sample in sample_rows if bool(sample.get("demo_perfect"))
                ),
                "local_seconds": round(local_seconds, 2),
            }
        )
    return rows


def _missing_gaps_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [
        str(row.get("task"))
        for row in rows
        if not bool(row.get("best_of_n_demo_perfect", row.get("demo_perfect", False)))
        and int(row.get("n_demo_perfect_samples", 0) or 0) == 0
    ]


def _seconds_per_task(rows: list[dict[str, Any]], fallback: Any = None) -> float:
    if isinstance(fallback, (int, float)) and not isinstance(fallback, bool):
        return round(float(fallback), 2)
    if not rows:
        return 0.0
    total = sum(float(row.get("local_seconds", 0.0) or 0.0) for row in rows)
    return round(total / len(rows), 2)


def _base_artifact_fields(
    *,
    build_payload: dict[str, Any],
    output_path: Path,
    duration_s: float,
) -> dict[str, Any]:
    return {
        "experiment": "experiment_4037_decentralization_stronger_base",
        "schema": "carnot.experiment_4037_decentralization_stronger_base.v1",
        "title": "Decentralization stronger-base coverage diagnosis",
        "build_artifact_path": str(build_payload.get("build_artifact_path", BUILD_ARTIFACT)),
        "output_artifact_path": str(output_path),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(float(duration_s), 2),
    }


def blocked_build_artifact(
    *,
    build_payload: dict[str, Any] | None,
    output_path: Path,
    started_s: float,
    duration_s: float,
) -> dict[str, Any]:
    payload = build_payload or {}
    seed = int(payload.get("random_seed", run4037.SEED))
    artifact = {
        **_base_artifact_fields(build_payload=payload, output_path=output_path, duration_s=duration_s),
        "honest_verdict": "blocked_build_runner_not_ready",
        "stronger_base_demo_perfect_coverage": 0.0,
        "coverage_delta_vs_12b": round(0.0 - DEFAULT_BASELINE_COVERAGE, 4),
        "bootstrap_ci95": [0.0, 0.0],
        "local_support_diagnosis": "absent",
        "local_seconds_per_task": 0.0,
        "codex_seconds_per_task_reference": DEFAULT_CODEX_SECONDS_PER_TASK,
        "local_vs_codex_seconds_ratio": 0.0,
        "model_specs": payload.get("model_specs", {"generator_model": "none"}),
        "random_seed": seed,
        "reproducibility_checksum": _stable_checksum(
            seed=seed, payloads=[payload, {"started_s": started_s, "blocked": True}]
        ),
        "missing_verifier_gaps": [],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "raw_complete": False,
        "partial_task_count": 0,
        "poll_attempts": 0,
        "raw_validation_error": "build_runner_not_ready",
    }
    validate_final_artifact(artifact)
    return artifact


def _diagnosis_from_ci(ci95: list[float], raw_complete: bool) -> str:
    if raw_complete and ci95[0] > 0.0:
        return "latent"
    return "absent"


def _verdict_for_complete(coverage: float, diagnosis: str) -> str:
    suffix = "latent_distill_viable" if diagnosis == "latent" else "absent_leash_holds"
    return f"complete: decentralization_stronger_base_cov_{_fmt(coverage)}_{suffix}"


def _pass2_comparison(pass2: float, references: dict[str, float]) -> dict[str, float]:
    return {
        "stronger_base_gated_pass_at_2": round(pass2, 4),
        "exp4012_12b_gated_pass_at_2": round(references["pass2_12b"], 4),
        "oracle_gated_pass_at_2": round(references["oracle_pass2"], 4),
        "codex_gated_pass_at_2": round(references["codex_pass2"], 4),
        "vs_exp4012_12b_gated_pass2": round(pass2 - references["pass2_12b"], 4),
        "vs_oracle_gated_pass2": round(pass2 - references["oracle_pass2"], 4),
        "vs_codex_gated_pass2": round(pass2 - references["codex_pass2"], 4),
    }


def complete_artifact(
    *,
    build_payload: dict[str, Any],
    raw_payload: dict[str, Any],
    references: dict[str, float],
    output_path: Path,
    raw_path: Path,
    baseline_path: Path,
    poll_attempts: int,
    duration_s: float,
    n_bootstrap: int,
) -> dict[str, Any]:
    rows = list(raw_payload.get("per_task", []))
    indicators = _coverage_indicators(rows)
    coverage = _coverage_from_indicators(indicators)
    delta = round(coverage - references["coverage_12b"], 4)
    ci95 = bootstrap_delta_ci95(
        indicators,
        references["coverage_12b"],
        n_bootstrap=n_bootstrap,
        seed=int(raw_payload.get("random_seed", run4037.SEED)),
    )
    diagnosis = _diagnosis_from_ci(ci95, raw_complete=True)
    pass2 = float(raw_payload.get("gated_pass_at_2", raw_payload.get("local_gated_pass2", 0.0)))
    seconds_per_task = _seconds_per_task(rows, fallback=raw_payload.get("cost_local_seconds"))
    codex_seconds = round(references["codex_seconds"], 2)
    missing_gaps = raw_payload.get("missing_verifier_gaps")
    if not isinstance(missing_gaps, list):
        missing_gaps = _missing_gaps_from_rows(rows)
    seed = int(raw_payload.get("random_seed", build_payload.get("random_seed", run4037.SEED)))
    artifact = {
        **_base_artifact_fields(
            build_payload=build_payload, output_path=output_path, duration_s=duration_s
        ),
        "honest_verdict": _verdict_for_complete(coverage, diagnosis),
        "stronger_base_demo_perfect_coverage": coverage,
        "coverage_delta_vs_12b": delta,
        "bootstrap_ci95": ci95,
        "local_support_diagnosis": diagnosis,
        "local_seconds_per_task": seconds_per_task,
        "codex_seconds_per_task_reference": codex_seconds,
        "local_vs_codex_seconds_ratio": round(seconds_per_task / codex_seconds, 4)
        if codex_seconds
        else 0.0,
        "gated_pass_at_2": round(pass2, 4),
        "pass2_comparison": _pass2_comparison(pass2, references),
        "model_specs": raw_payload.get("model_specs", build_payload.get("model_specs", {})),
        "random_seed": seed,
        "reproducibility_checksum": _stable_checksum(
            seed=seed,
            payloads=[
                build_payload.get("reproducibility_checksum"),
                raw_payload.get("reproducibility_checksum"),
                references,
            ],
        ),
        "missing_verifier_gaps": [str(task) for task in missing_gaps],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "source_inference_substrate": raw_payload.get("inference_substrate"),
        "raw_complete": True,
        "partial_task_count": len(rows),
        "poll_attempts": poll_attempts,
        "raw_artifact_path": str(raw_path),
        "baseline_artifact_path": str(baseline_path),
        "n_tasks_scored": len(rows),
        "n_demo_perfect_tasks": int(sum(indicators)),
        "references": {
            "exp4012_12b_demo_perfect_coverage": round(references["coverage_12b"], 4),
            "exp4012_12b_gated_pass_at_2": round(references["pass2_12b"], 4),
            "oracle_gated_pass_at_2": round(references["oracle_pass2"], 4),
            "codex_gated_pass_at_2": round(references["codex_pass2"], 4),
        },
    }
    validate_final_artifact(artifact)
    return artifact


def partial_artifact(
    *,
    build_payload: dict[str, Any],
    partial_rows: list[dict[str, Any]],
    references: dict[str, float],
    output_path: Path,
    checkpoint_path: Path,
    log_path: Path,
    poll_attempts: int,
    raw_error: str | None,
    duration_s: float,
    n_bootstrap: int,
) -> dict[str, Any]:
    indicators = _coverage_indicators(partial_rows)
    coverage = _coverage_from_indicators(indicators)
    delta = round(coverage - references["coverage_12b"], 4)
    ci95 = bootstrap_delta_ci95(
        indicators,
        references["coverage_12b"],
        n_bootstrap=n_bootstrap,
        seed=int(build_payload.get("random_seed", run4037.SEED)),
    )
    seed = int(build_payload.get("random_seed", run4037.SEED))
    seconds_per_task = _seconds_per_task(partial_rows)
    codex_seconds = round(references["codex_seconds"], 2)
    artifact = {
        **_base_artifact_fields(
            build_payload=build_payload, output_path=output_path, duration_s=duration_s
        ),
        "honest_verdict": (
            f"complete: decentralization_stronger_base_partial_{len(partial_rows)}_tasks"
        ),
        "stronger_base_demo_perfect_coverage": coverage,
        "coverage_delta_vs_12b": delta,
        "bootstrap_ci95": ci95,
        "local_support_diagnosis": _diagnosis_from_ci(ci95, raw_complete=False),
        "local_seconds_per_task": seconds_per_task,
        "codex_seconds_per_task_reference": codex_seconds,
        "local_vs_codex_seconds_ratio": round(seconds_per_task / codex_seconds, 4)
        if codex_seconds
        else 0.0,
        "gated_pass_at_2": 0.0,
        "pass2_comparison": _pass2_comparison(0.0, references),
        "model_specs": build_payload.get("model_specs", {"generator_model": "none"}),
        "random_seed": seed,
        "reproducibility_checksum": _stable_checksum(
            seed=seed,
            payloads=[
                build_payload.get("reproducibility_checksum"),
                partial_rows,
                raw_error,
            ],
        ),
        "missing_verifier_gaps": _missing_gaps_from_rows(partial_rows),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "raw_complete": False,
        "partial_task_count": len(partial_rows),
        "poll_attempts": poll_attempts,
        "raw_validation_error": raw_error,
        "checkpoint_artifact_path": str(checkpoint_path),
        "log_path": str(log_path),
        "log_tail": _tail_text(log_path),
        "n_tasks_scored": len(partial_rows),
        "n_demo_perfect_tasks": int(sum(indicators)),
    }
    validate_final_artifact(artifact)
    return artifact


def validate_final_artifact(artifact: dict[str, Any]) -> None:
    for field in REQUIRED_FINAL_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required field: {field}")
    verdict = artifact["honest_verdict"]
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:")
        or verdict.startswith("success:")
        or verdict.startswith("blocked_")
    ):
        raise ValueError("honest_verdict must use a terminal prefix")
    for field in (
        "stronger_base_demo_perfect_coverage",
        "coverage_delta_vs_12b",
        "local_seconds_per_task",
    ):
        if not _is_bare_float(artifact[field]):
            raise ValueError(f"{field} must be a bare float")
    ci95 = artifact["bootstrap_ci95"]
    if not (
        isinstance(ci95, list)
        and len(ci95) == 2
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in ci95)
    ):
        raise ValueError("bootstrap_ci95 must be a 2-element numeric list")
    if artifact["local_support_diagnosis"] not in {"latent", "absent"}:
        raise ValueError("local_support_diagnosis must be latent or absent")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be a dict")
    if not _is_bare_int(artifact["random_seed"]):
        raise ValueError("random_seed must be a bare int")
    if not isinstance(artifact["reproducibility_checksum"], str):
        raise ValueError("reproducibility_checksum must be a string")
    if not isinstance(artifact["missing_verifier_gaps"], list):
        raise ValueError("missing_verifier_gaps must be a list")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be verifier ensemble against cached candidates")


def run_collection(
    *,
    build_path: Path = BUILD_ARTIFACT,
    raw_path: Path = RAW_ARTIFACT,
    baseline_path: Path = BASELINE_ARTIFACT,
    checkpoint_path: Path = CHECKPOINT_ARTIFACT,
    log_path: Path = LOG_PATH,
    output_path: Path = OUTPUT,
    poll_budget_s: float = DEFAULT_POLL_BUDGET_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    sleep_fn: Callable[[float], None] = time.sleep,
    write: bool = True,
) -> dict[str, Any]:
    started = time.time()
    build_payload, build_error = _read_json(build_path)
    if build_payload is None or build_payload.get("runner_ready") is not True:
        artifact = blocked_build_artifact(
            build_payload=build_payload or {"build_error": build_error},
            output_path=output_path,
            started_s=started,
            duration_s=time.time() - started,
        )
        if write:
            _write_json(output_path, artifact)
        print(f"-> {artifact['honest_verdict']}", flush=True)
        return artifact

    references = _reference_values(baseline_path)
    raw_payload, raw_error, poll_attempts = poll_raw_artifact(
        raw_path,
        poll_budget_s=poll_budget_s,
        poll_interval_s=poll_interval_s,
        sleep_fn=sleep_fn,
    )
    if raw_payload is not None:
        artifact = complete_artifact(
            build_payload=build_payload,
            raw_payload=raw_payload,
            references=references,
            output_path=output_path,
            raw_path=raw_path,
            baseline_path=baseline_path,
            poll_attempts=poll_attempts,
            duration_s=time.time() - started,
            n_bootstrap=n_bootstrap,
        )
    else:
        artifact = partial_artifact(
            build_payload=build_payload,
            partial_rows=_partial_rows_from_checkpoint(checkpoint_path),
            references=references,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            log_path=log_path,
            poll_attempts=poll_attempts,
            raw_error=raw_error,
            duration_s=time.time() - started,
            n_bootstrap=n_bootstrap,
        )
    if write:
        _write_json(output_path, artifact)
    print(f"-> {artifact['honest_verdict']}", flush=True)
    print(
        "   coverage="
        f"{artifact['stronger_base_demo_perfect_coverage']} "
        f"delta={artifact['coverage_delta_vs_12b']} "
        f"CI={artifact['bootstrap_ci95']} "
        f"diagnosis={artifact['local_support_diagnosis']}",
        flush=True,
    )
    return artifact


def main() -> None:  # pragma: no cover - exercised by the operator command.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=BUILD_ARTIFACT)
    parser.add_argument("--raw", type=Path, default=RAW_ARTIFACT)
    parser.add_argument("--baseline", type=Path, default=BASELINE_ARTIFACT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_ARTIFACT)
    parser.add_argument("--log", type=Path, default=LOG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--poll-budget-s", type=float, default=DEFAULT_POLL_BUDGET_S)
    parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    run_collection(
        build_path=args.build,
        raw_path=args.raw,
        baseline_path=args.baseline,
        checkpoint_path=args.checkpoint,
        log_path=args.log,
        output_path=args.output,
        poll_budget_s=args.poll_budget_s,
        poll_interval_s=args.poll_interval_s,
        n_bootstrap=args.bootstrap_samples,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

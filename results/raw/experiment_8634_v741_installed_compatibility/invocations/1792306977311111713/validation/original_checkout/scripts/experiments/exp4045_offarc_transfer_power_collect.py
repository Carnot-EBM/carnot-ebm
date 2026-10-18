"""Collect Exp 4045 OFF-ARC full-power transfer evidence.

Spec refs: REQ-VERIFY-4045, SCENARIO-VERIFY-4045.

The runner performs live GGUF generation. This collector does not. It waits for
the runner's raw artifact within a bounded budget, or falls back to the
checkpoint and emits a partial verdict that is explicit about the unfinished
task count. All confidence intervals are recomputed from task-level pass/fail
rows so the final artifact answers the operator's headline question directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import offarc_transfer_power_run as runner

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_PATH = REPO_ROOT / "results" / "experiment_4044_offarc_transfer_power_build.json"
RAW_PATH = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power_raw.json"
CHECKPOINT_PATH = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power.checkpoint.json"
OUTPUT_PATH = REPO_ROOT / "results" / "experiment_4045_offarc_transfer_power.json"
LOG_PATH = REPO_ROOT / "logs" / "offarc_power_run.log"

INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_POLL_BUDGET_S = 60.0
DEFAULT_POLL_INTERVAL_S = 60.0
POWERED_TASK_FLOOR = 160

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefix verdict; partial, no-headroom, and honest negative outcomes are complete.",
    "n_tasks": "N controls whether a CI-excludes-zero transfer claim is statistically powered.",
    "demofit_ci_excludes_zero": "Bare bool for the operator's headline Arm B minus Arm A gate.",
    "best_arm_ci_excludes_zero": "Bare bool for whether a stronger verifier arm clears the bar.",
    "oracle_passrate": "Positive control for selectable headroom and false-negative risk.",
    "missing_verifier_gaps": "Residual selectable slices become future verifier-build specs.",
    "inference_substrate": "Collector scores cached candidates and never invokes live LLM inference.",
}

REQUIRED_FINAL_FIELDS = [
    "honest_verdict",
    "corpus",
    "n_tasks",
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
    "oracle_passrate",
    "missing_verifier_gaps",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "inference_substrate",
]


def bootstrap_delta_ci95(
    values: list[int], *, seed: int, n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES
) -> list[float]:
    """Return a deterministic task-level bootstrap CI in percentage points."""
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


def build_final_artifact(
    raw: dict[str, Any],
    *,
    raw_artifact_present: bool,
    partial_reason: str | None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    powered_task_floor: int = POWERED_TASK_FLOOR,
) -> dict[str, Any]:
    per_task = list(raw.get("per_task") or [])
    n_tasks = len(per_task)

    arm_a = _rate(per_task, "armA_vote_pass1")
    arm_app = _rate(per_task, "armAplusplus_aces_pass1")
    arm_b = _rate(per_task, "armB_demofit_pass1")
    arm_c = _rate(per_task, "armC_symbolic_partition_pass1")
    oracle = _rate(per_task, "oracle_hidden_pass")

    b_deltas = _paired_deltas(per_task, "armB_demofit_pass1")
    app_deltas = _paired_deltas(per_task, "armAplusplus_aces_pass1")
    c_deltas = _paired_deltas(per_task, "armC_symbolic_partition_pass1")
    b_ci = bootstrap_delta_ci95(b_deltas, seed=runner.RANDOM_SEED, n_bootstrap=n_bootstrap)
    app_ci = bootstrap_delta_ci95(
        app_deltas, seed=runner.RANDOM_SEED + 2, n_bootstrap=n_bootstrap
    )
    c_ci = bootstrap_delta_ci95(c_deltas, seed=runner.RANDOM_SEED + 4, n_bootstrap=n_bootstrap)

    arm_summaries = {
        "armB_demofit": (_delta_pp(arm_b, arm_a), b_ci),
        "armApp_aces": (_delta_pp(arm_app, arm_a), app_ci),
        "armC_symbolic": (_delta_pp(arm_c, arm_a), c_ci),
    }
    best_arm = max(arm_summaries, key=lambda name: (arm_summaries[name][0], name))
    best_delta, best_ci = arm_summaries[best_arm]
    oracle_headroom = oracle > arm_a + 0.01
    missing_gaps = _missing_gaps(raw, b_ci, app_ci, c_ci)

    artifact: dict[str, Any] = {
        "experiment": "experiment_4045_offarc_transfer_power",
        "schema": "carnot.experiment_4045_offarc_transfer_power_collect.v1",
        "honest_verdict": _verdict(
            n_tasks=n_tasks,
            powered_task_floor=powered_task_floor,
            partial_reason=partial_reason,
            demofit_ci=b_ci,
            demofit_delta_pp=_delta_pp(arm_b, arm_a),
            arm_app_ci=app_ci,
            arm_c_ci=c_ci,
            oracle_headroom=oracle_headroom,
        ),
        "corpus": str(raw.get("corpus") or "humaneval_plus_mbpp"),
        "n_tasks": n_tasks,
        "powered_task_floor": powered_task_floor,
        "raw_artifact_present": raw_artifact_present,
        "partial_reason": partial_reason,
        "armA_vote_passrate": arm_a,
        "armApp_aces_passrate": arm_app,
        "armB_demofit_passrate": arm_b,
        "armC_symbolic_passrate": arm_c,
        "demofit_delta_pp": _delta_pp(arm_b, arm_a),
        "demofit_bootstrap_ci95": b_ci,
        "demofit_ci_excludes_zero": _ci_excludes_zero(b_ci),
        "armApp_delta_pp": _delta_pp(arm_app, arm_a),
        "armApp_bootstrap_ci95": app_ci,
        "armApp_ci_excludes_zero": _ci_excludes_zero(app_ci),
        "armC_delta_pp": _delta_pp(arm_c, arm_a),
        "armC_bootstrap_ci95": c_ci,
        "armC_ci_excludes_zero": _ci_excludes_zero(c_ci),
        "best_arm": best_arm,
        "best_arm_delta_pp": best_delta,
        "best_arm_ci95": best_ci,
        "best_arm_ci_excludes_zero": _ci_excludes_zero(best_ci),
        "oracle_passrate": oracle,
        "oracle_headroom": oracle_headroom,
        "missing_verifier_gaps": missing_gaps,
        "model_specs": raw.get("model_specs") or {},
        "random_seed": int(raw.get("random_seed", runner.RANDOM_SEED)),
        "bootstrap_resamples": n_bootstrap,
        "source_reproducibility_checksum": str(raw.get("reproducibility_checksum") or ""),
        "reproducibility_checksum": "",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_principles": FIELD_PRINCIPLES,
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_final_artifact(artifact)
    return artifact


def validate_final_artifact(artifact: dict[str, Any]) -> None:
    verdict = artifact.get("honest_verdict")
    if verdict == "blocked_build_runner_not_ready":
        if artifact.get("runner_ready") is not False:
            raise ValueError("blocked artifact must record runner_ready=false")
        return
    for field in REQUIRED_FINAL_FIELDS:
        if field not in artifact:
            raise ValueError(f"missing required final field: {field}")
    if not isinstance(verdict, str) or not verdict.startswith("complete:"):
        raise ValueError("honest_verdict must use a complete: terminal prefix")
    for field in ("n_tasks", "random_seed"):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare int")
    for field in (
        "armA_vote_passrate",
        "armApp_aces_passrate",
        "armB_demofit_passrate",
        "armC_symbolic_passrate",
        "demofit_delta_pp",
        "best_arm_delta_pp",
        "oracle_passrate",
    ):
        if not isinstance(artifact[field], float) or isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare float")
    for field in ("demofit_ci_excludes_zero", "best_arm_ci_excludes_zero"):
        if not isinstance(artifact[field], bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in ("demofit_bootstrap_ci95", "best_arm_ci95"):
        if not _is_two_numeric_list(artifact[field]):
            raise ValueError(f"{field} must be a two-element list")
    if not isinstance(artifact["missing_verifier_gaps"], list):
        raise ValueError("missing_verifier_gaps must be a list")
    if not isinstance(artifact["model_specs"], dict):
        raise ValueError("model_specs must be an object")
    if not artifact["reproducibility_checksum"]:
        raise ValueError("reproducibility_checksum must be non-empty")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be verifier-scoring substrate")


def run_collect(
    *,
    output_path: Path = OUTPUT_PATH,
    build_path: Path = BUILD_PATH,
    raw_path: Path = RAW_PATH,
    checkpoint_path: Path = CHECKPOINT_PATH,
    log_path: Path | None = LOG_PATH,
    poll_budget_s: float = DEFAULT_POLL_BUDGET_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    task_loader: Callable[[], list[runner.CodeTask]] = lambda: runner.load_code_tasks(
        limit=runner.DEFAULT_N_TASKS
    )[0],
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    build = _read_json(build_path)
    if build.get("runner_ready") is not True:
        artifact = {
            "experiment": "experiment_4045_offarc_transfer_power",
            "schema": "carnot.experiment_4045_offarc_transfer_power_collect.v1",
            "honest_verdict": "blocked_build_runner_not_ready",
            "runner_ready": False,
            "build_honest_verdict": build.get("honest_verdict"),
            "preconditions_checked": build.get("preconditions_checked", []),
        }
        validate_final_artifact(artifact)
        _write_json(output_path, artifact)
        return artifact

    candidate_raw_path = Path(build.get("full_raw_path") or raw_path)
    candidate_log_path = Path(build.get("log_path") or log_path or LOG_PATH)
    raw_ready = poll_for_raw(
        candidate_raw_path,
        log_path=candidate_log_path,
        poll_budget_s=poll_budget_s,
        poll_interval_s=poll_interval_s,
    )
    if raw_ready:
        raw = _read_json(candidate_raw_path)
        runner.validate_raw_artifact(raw, require_full=False)
        partial_reason = None
    else:
        raw = raw_from_checkpoint(
            checkpoint_path=checkpoint_path,
            build=build,
            task_loader=task_loader,
        )
        partial_reason = f"offarc_power_run_incomplete_partial_{len(raw.get('per_task') or [])}_tasks"

    artifact = build_final_artifact(
        raw,
        raw_artifact_present=raw_ready,
        partial_reason=partial_reason,
        n_bootstrap=n_bootstrap,
    )
    artifact["log_tail"] = tail_log(candidate_log_path)
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)
    validate_final_artifact(artifact)
    _write_json(output_path, artifact)
    return artifact


def poll_for_raw(
    raw_path: Path,
    *,
    log_path: Path,
    poll_budget_s: float,
    poll_interval_s: float,
    sleeper: Callable[[float], None] = time.sleep,
) -> bool:
    started = time.monotonic()
    while not raw_path.exists():
        elapsed = time.monotonic() - started
        if elapsed >= poll_budget_s:
            return False
        tail_log(log_path)
        wait_s = min(poll_interval_s, poll_budget_s - elapsed)
        if wait_s > 0.0:
            sleeper(wait_s)  # pragma: no cover - real polling delay.
    return True


def raw_from_checkpoint(
    *,
    checkpoint_path: Path,
    build: dict[str, Any],
    task_loader: Callable[[], list[runner.CodeTask]],
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return _empty_partial_raw(build=build, checkpoint_path=checkpoint_path)
    payload = _read_json(checkpoint_path)
    evaluations_by_task = {
        str(task_id): [runner.CandidateEvaluation(**row) for row in rows]
        for task_id, rows in payload.get("evaluations_by_task", {}).items()
    }
    completed = set(evaluations_by_task)
    task_by_id = {task.task_id: task for task in task_loader()}
    ordered_ids = payload.get("ordered_task_ids") or sorted(completed)
    tasks = [
        task_by_id.get(str(task_id)) or _fallback_task(str(task_id))
        for task_id in ordered_ids
        if str(task_id) in completed
    ]
    if not tasks:
        return _empty_partial_raw(build=build, checkpoint_path=checkpoint_path)

    scored = runner.score_evaluated_tasks(tasks, evaluations_by_task, seed=runner.RANDOM_SEED)
    return {
        "experiment": "experiment_4045_offarc_transfer_power_checkpoint_collect",
        "schema": "carnot.experiment_4045_offarc_transfer_power_partial.v1",
        "honest_verdict": "complete: partial_checkpoint_scored",
        "corpus": "humaneval_plus_mbpp_partial_checkpoint",
        "n_tasks": len(tasks),
        "armA_vote_passrate": scored["armA_vote_passrate"],
        "armAplusplus_aces_passrate": scored["armAplusplus_aces_passrate"],
        "armB_demofit_passrate": scored["armB_demofit_passrate"],
        "armC_symbolic_partition_passrate": scored["armC_symbolic_partition_passrate"],
        "oracle_passrate": scored["oracle_passrate"],
        "missing_verifier_gaps": scored["missing_verifier_gaps"],
        "model_specs": _partial_model_specs(checkpoint_path),
        "random_seed": runner.RANDOM_SEED,
        "reproducibility_checksum": _source_checksum(payload),
        "preconditions_checked": build.get("preconditions_checked", []),
        "per_task": scored["per_task"],
    }


def tail_log(path: Path, *, n_lines: int = 30) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n_lines:])


def _missing_gaps(
    raw: dict[str, Any], b_ci: list[float], app_ci: list[float], c_ci: list[float]
) -> list[str]:
    gaps = [str(item) for item in raw.get("missing_verifier_gaps", [])]
    stronger_excludes = app_ci[0] > 0.0 or c_ci[0] > 0.0
    if b_ci[0] <= 0.0 and stronger_excludes and "GAP-CODE-EXEC-DEMOFIT" not in gaps:
        gaps.append("GAP-CODE-EXEC-DEMOFIT")
    return gaps


def _verdict(
    *,
    n_tasks: int,
    powered_task_floor: int,
    partial_reason: str | None,
    demofit_ci: list[float],
    demofit_delta_pp: float,
    arm_app_ci: list[float],
    arm_c_ci: list[float],
    oracle_headroom: bool,
) -> str:
    if partial_reason is not None or n_tasks < powered_task_floor:
        return f"complete: offarc_power_run_incomplete_partial_{n_tasks}_tasks"
    if demofit_ci[0] > 0.0 and demofit_delta_pp > 0.0:
        return f"complete: offarc_demofit_transfers_to_code_ci_excl0_n{n_tasks}"
    if demofit_ci[1] < 0.0:
        return f"complete: offarc_demofit_negative_ci_excl0_n{n_tasks}"
    if arm_c_ci[0] > 0.0:
        return "complete: offarc_demofit_touches0_symbolic_partition_excl0"
    if arm_app_ci[0] > 0.0:
        return "complete: offarc_demofit_touches0_aces_excl0"
    if not oracle_headroom:
        return "complete: offarc_transfer_uninformative_no_oracle_headroom"
    return "complete: offarc_transfer_small_magnitude_all_arms_touch0"


def _rate(per_task: list[dict[str, Any]], field: str) -> float:
    return round(sum(1 for row in per_task if bool(row.get(field))) / max(1, len(per_task)), 6)


def _paired_deltas(per_task: list[dict[str, Any]], arm_field: str) -> list[int]:
    return [int(bool(row.get(arm_field))) - int(bool(row.get("armA_vote_pass1"))) for row in per_task]


def _delta_pp(rate: float, baseline: float) -> float:
    return round((rate - baseline) * 100.0, 4)


def _ci_excludes_zero(ci95: list[float]) -> bool:
    return ci95[0] > 0.0 or ci95[1] < 0.0


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
        if key not in {"reproducibility_checksum"}
    }
    return _source_checksum(payload)


def _source_checksum(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _empty_partial_raw(*, build: dict[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    return {
        "experiment": "experiment_4045_offarc_transfer_power_checkpoint_collect",
        "schema": "carnot.experiment_4045_offarc_transfer_power_partial.v1",
        "honest_verdict": "complete: partial_checkpoint_empty",
        "corpus": "humaneval_plus_mbpp_partial_checkpoint",
        "n_tasks": 0,
        "model_specs": _partial_model_specs(checkpoint_path),
        "random_seed": runner.RANDOM_SEED,
        "reproducibility_checksum": _source_checksum({"checkpoint_path": str(checkpoint_path)}),
        "preconditions_checked": build.get("preconditions_checked", []),
        "missing_verifier_gaps": [],
        "per_task": [],
    }


def _partial_model_specs(checkpoint_path: Path) -> dict[str, Any]:
    return {
        "local_generator": "unsloth/gemma-4-12B-it-GGUF",
        "verifier": "cached candidate verifier ensemble replay",
        "candidate_pool_policy": "same generated pool shared by all four arms",
        "source_checkpoint": str(checkpoint_path),
    }


def _fallback_task(task_id: str) -> runner.CodeTask:
    return runner.CodeTask(
        task_id=task_id,
        corpus="humaneval" if task_id.startswith("HumanEval") else "mbpp",
        prompt="",
        func_name="unknown",
        visible_tests=[],
        hidden_tests=[],
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI adapter.
    parser = argparse.ArgumentParser(description="Collect Exp 4045 OFF-ARC transfer power")
    parser.add_argument("--poll-budget-s", type=float, default=DEFAULT_POLL_BUDGET_S)
    parser.add_argument("--poll-interval-s", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    artifact = run_collect(
        poll_budget_s=args.poll_budget_s,
        poll_interval_s=args.poll_interval_s,
        n_bootstrap=args.bootstrap_samples,
    )
    print(
        f"-> {artifact['honest_verdict']} n={artifact.get('n_tasks')} "
        f"demofit_ci={artifact.get('demofit_bootstrap_ci95')}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI adapter.
    main()

"""Exp 4168 defensive decisive verifier graft over the outer-loop baseline.

This module exists to prevent the `.385` failure mode: a verifier-graft task
must not read a checkpoint while the outer loop is writing it, must not train
on the stable checkpoint in place, and must not turn a non-faithful baseline
into an uninformative negative result. The honest outcome can therefore be a
deferral, a copied-checkpoint verifier win, or a copied-checkpoint null.

Spec refs: REQ-LEARN-4168, SCENARIO-LEARN-4168-DEFER,
SCENARIO-LEARN-4168-COPY-GRAFT.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4158_verifier_rerank_recovery_moat as exp4158
from carnot import experiment_4167_outerloop_training_monitor as exp4167


JsonDict = dict[str, Any]
CandidatePool = exp4109.CandidatePool

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4168_decisive_verifier_graft_defensive.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_DATA_DIR = REPO_ROOT / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
DEFAULT_HELDOUT_SPLIT = "_valsmall"
DEFAULT_COPY_DIR = REPO_ROOT / "results" / "trm_runs" / "experiment_4168_decisive_verifier_graft_defensive"
EXPERIMENT_ID = 4168
RANDOM_SEED = 4168
FAITHFUL_VAL_THRESHOLD = 0.85
DEFAULT_MAX_PUZZLES = 64
DEFAULT_K_CANDIDATES = 8
SCHEMA = "carnot.experiment_4168_decisive_verifier_graft_defensive.v1"
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
SPEC_REFS = [
    "REQ-LEARN-4168",
    "SCENARIO-LEARN-4168-DEFER",
    "SCENARIO-LEARN-4168-COPY-GRAFT",
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "graft_deferred",
    "rerank_lift_vs_vote",
    "rft_vs_ablation_delta",
    "verifier_value_added",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. An honest deferral, an A>B win, or an A~=B null are all COMPLETE.",
    "graft_deferred": (
        "Bare bool: True if the baseline was not faithful+stable -> deferred. Prevents an uninformative "
        "graft + a collision with the outer-loop run."
    ),
    "rerank_lift_vs_vote": (
        "pass@1 lift from verifier-reranking (if grafted); the executable-verifier discrimination signal."
    ),
    "rft_vs_ablation_delta": "The de-confounded A-vs-B held-out delta with CI -- THE moat measurement.",
    "verifier_value_added": (
        "Bare bool: did the graft beat the vote ablation? Resolves the moat question + the DiffusionGemma gate."
    ),
}


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str) and value.strip():
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _metric_has_ci(metric: Any) -> bool:
    if not isinstance(metric, Mapping):
        return False
    ci95 = metric.get("ci95")
    return (
        isinstance(ci95, Sequence)
        and not isinstance(ci95, (str, bytes))
        and len(ci95) == 2
        and _float_or_none(ci95[0]) is not None
        and _float_or_none(ci95[1]) is not None
    )


def _format_val_tag(value: Any) -> str:
    number = _float_or_none(value)
    return "unknown" if number is None else f"{number:.4f}"


def _payload_checksum(payload: Mapping[str, Any]) -> str:
    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_jsonable(filtered), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _monitor_source_snapshot(monitor: Mapping[str, Any]) -> JsonDict:
    return {
        "baseline_faithful": bool(monitor.get("baseline_faithful")),
        "outerloop_train_alive": bool(monitor.get("outerloop_train_alive")),
        "current_val_exact_accuracy": _float_or_none(monitor.get("current_val_exact_accuracy")),
        "checkpoint_path": monitor.get("checkpoint_path"),
        "checkpoint_mtime": monitor.get("checkpoint_mtime"),
        "outerloop_pid": monitor.get("outerloop_pid"),
        "latest_metrics_path": monitor.get("latest_metrics_path"),
    }


def baseline_is_faithful_stable(monitor: Mapping[str, Any]) -> bool:
    """REQ-LEARN-4168: require both the validation threshold and no live writer."""

    current_val = _float_or_none(monitor.get("current_val_exact_accuracy"))
    return (
        monitor.get("baseline_faithful") is True
        and monitor.get("outerloop_train_alive") is False
        and current_val is not None
        and current_val >= FAITHFUL_VAL_THRESHOLD
    )


def _baseline_status(monitor: Mapping[str, Any]) -> JsonDict:
    return {
        **_monitor_source_snapshot(monitor),
        "faithful_threshold": FAITHFUL_VAL_THRESHOLD,
        "faithful_stable": baseline_is_faithful_stable(monitor),
    }


def _deferred_rerank_metric(status: str, monitor: Mapping[str, Any]) -> JsonDict:
    return {
        "metric": "pass@1_exact_accuracy",
        "n_puzzles": 0,
        "vote_at_1": 0.0,
        "verifier_pass_at_1": 0.0,
        "oracle_at_k": 0.0,
        "delta": 0.0,
        "delta_vs_oracle": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
        "current_val_exact_accuracy": _float_or_none(monitor.get("current_val_exact_accuracy")),
    }


def _deferred_rft_metric(status: str, monitor: Mapping[str, Any]) -> JsonDict:
    return {
        "metric": "heldout_exact_accuracy",
        "training_mode": "not_run",
        "n_matched": 0,
        "a_exact_accuracy": 0.0,
        "b_exact_accuracy": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
        "current_val_exact_accuracy": _float_or_none(monitor.get("current_val_exact_accuracy")),
    }


def _preconditions_for_monitor(monitor: Mapping[str, Any]) -> list[JsonDict]:
    stable = monitor.get("checkpoint_path")
    return [
        {
            "resource": "exp4167_monitor",
            "available": True,
            "detail": str(monitor.get("schema", "provided_or_recomputed_monitor")),
        },
        {
            "resource": "baseline_faithful_stable",
            "available": baseline_is_faithful_stable(monitor),
            "detail": json.dumps(_monitor_source_snapshot(monitor), sort_keys=True),
        },
        {
            "resource": "stable_checkpoint_path",
            "available": isinstance(stable, str) and bool(stable),
            "detail": "" if stable is None else str(stable),
        },
    ]


def _copy_target_for(stable_checkpoint_path: Path, *, repo_root: str | Path = REPO_ROOT) -> Path:
    root = Path(repo_root)
    return (
        root
        / "results"
        / "trm_runs"
        / "experiment_4168_decisive_verifier_graft_defensive"
        / f"{stable_checkpoint_path.stem}-exp4168-copy{stable_checkpoint_path.suffix}"
    )


def copy_checkpoint_to_task_local(source_path: Path, target_path: Path) -> Path:
    """SCENARIO-LEARN-4168-COPY-GRAFT: freeze the baseline before model use."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path


def evaluate_rerank_arm(
    pools: Sequence[CandidatePool],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
) -> JsonDict:
    """REQ-LEARN-4168: report verifier lift versus vote and gap to oracle."""

    metrics = exp4158.evaluate_recovery_moat(
        pools,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    rerank = dict(metrics["rerank_lift_vs_vote"])
    verifier_pass_at_1 = _float_or_none(rerank.get("verifier_pass_at_1")) or 0.0
    oracle_at_k = _float_or_none(rerank.get("oracle_at_k")) or 0.0
    rerank["delta_vs_oracle"] = round(float(verifier_pass_at_1 - oracle_at_k), 6)
    rerank["headroom_present"] = bool(metrics.get("headroom_present"))
    rerank["verifier_recovers_outvoted"] = int(metrics.get("verifier_recovers_outvoted", 0))
    return _jsonable(rerank)


def verifier_value_added(rft_vs_ablation_delta: Mapping[str, Any], *, graft_deferred: bool) -> bool:
    """REQ-LEARN-4168: the headline bool comes only from copied-checkpoint RFT."""

    if graft_deferred:
        return False
    ci95 = rft_vs_ablation_delta.get("ci95")
    try:
        return (
            isinstance(ci95, Sequence)
            and not isinstance(ci95, (str, bytes))
            and len(ci95) == 2
            and float(rft_vs_ablation_delta.get("delta", 0.0)) > 0.0
            and float(ci95[0]) > 0.0
        )
    except (TypeError, ValueError):
        return False


def _artifact_verdict(*, graft_deferred: bool, value_added: bool, monitor: Mapping[str, Any]) -> str:
    if graft_deferred:
        return f"complete: graft_deferred_outerloop_training_val_{_format_val_tag(monitor.get('current_val_exact_accuracy'))}"
    if value_added:
        return "success: verifier_value_added_rft_A_gt_B_copy_graft"
    return "complete: A~=B null"


def _acceptance_gate(artifact: Mapping[str, Any]) -> bool:
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        return False
    if type(artifact.get("graft_deferred")) is not bool:
        return False
    if type(artifact.get("verifier_value_added")) is not bool:
        return False
    baseline = artifact.get("baseline_status")
    if not isinstance(baseline, Mapping):
        return False
    if artifact["graft_deferred"] is True:
        return baseline.get("faithful_stable") is False
    return (
        bool(artifact.get("checkpoint_copy_performed"))
        and isinstance(artifact.get("checkpoint_copy_path"), str)
        and _metric_has_ci(artifact.get("rerank_lift_vs_vote"))
        and _metric_has_ci(artifact.get("rft_vs_ablation_delta"))
    )


def build_deferred_artifact(
    *,
    monitor: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """SCENARIO-LEARN-4168-DEFER: write evidence without mutating training state."""

    rerank = _deferred_rerank_metric("deferred_outerloop_training", monitor)
    rft = _deferred_rft_metric("deferred_outerloop_training", monitor)
    artifact: JsonDict = {
        "experiment": "experiment_4168_decisive_verifier_graft_defensive",
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _artifact_verdict(graft_deferred=True, value_added=False, monitor=monitor),
        "graft_deferred": True,
        "rerank_lift_vs_vote": rerank,
        "rft_vs_ablation_delta": rft,
        "verifier_value_added": False,
        "baseline_status": _baseline_status(monitor),
        "preconditions_checked": _preconditions_for_monitor(monitor),
        "stable_checkpoint_path": monitor.get("checkpoint_path"),
        "checkpoint_copy_path": None,
        "checkpoint_copy_performed": False,
        "candidate_source": "none_deferred_outerloop_training",
        "n_candidate_pools": 0,
        "corpus_summary": {"n_matched": 0, "arm_a": "verifier_certified", "arm_b": "vote_certified"},
        "read_only_actions": {
            "training_launched": False,
            "train_process_stop_attempted": False,
            "stable_checkpoint_written": False,
            "candidate_sampling_launched": False,
        },
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_result_artifact(
    *,
    monitor: Mapping[str, Any],
    checkpoint_copy_path: Path,
    rerank_lift_vs_vote: Mapping[str, Any],
    rft_vs_ablation_delta: Mapping[str, Any],
    corpora: Mapping[str, Any],
    candidate_source: str,
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """REQ-LEARN-4168: build the copied-checkpoint graft artifact."""

    value_added = verifier_value_added(rft_vs_ablation_delta, graft_deferred=False)
    artifact: JsonDict = {
        "experiment": "experiment_4168_decisive_verifier_graft_defensive",
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _artifact_verdict(
            graft_deferred=False,
            value_added=value_added,
            monitor=monitor,
        ),
        "graft_deferred": False,
        "rerank_lift_vs_vote": _jsonable(rerank_lift_vs_vote),
        "rft_vs_ablation_delta": _jsonable(rft_vs_ablation_delta),
        "verifier_value_added": bool(value_added),
        "baseline_status": _baseline_status(monitor),
        "preconditions_checked": _preconditions_for_monitor(monitor),
        "stable_checkpoint_path": monitor.get("checkpoint_path"),
        "checkpoint_copy_path": str(checkpoint_copy_path),
        "checkpoint_copy_performed": True,
        "candidate_source": candidate_source,
        "n_candidate_pools": int(rerank_lift_vs_vote.get("n_puzzles", 0)),
        "corpus_summary": {
            "arm_a": corpora.get("arm_a"),
            "arm_b": corpora.get("arm_b"),
            "n_matched": int(corpora.get("n_matched", 0)),
            "skipped_no_verifier_valid": len(corpora.get("skipped_no_verifier_valid", [])),
        },
        "read_only_actions": {
            "training_launched": True,
            "train_process_stop_attempted": False,
            "stable_checkpoint_written": False,
            "candidate_sampling_launched": True,
        },
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    honest_verdict: str,
    *,
    monitor: Mapping[str, Any],
    detail: str,
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """Fail closed when a faithful-branch resource is missing."""

    artifact: JsonDict = {
        "experiment": "experiment_4168_decisive_verifier_graft_defensive",
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "graft_deferred": True,
        "rerank_lift_vs_vote": _deferred_rerank_metric(honest_verdict, monitor),
        "rft_vs_ablation_delta": _deferred_rft_metric(honest_verdict, monitor),
        "verifier_value_added": False,
        "baseline_status": _baseline_status(monitor),
        "preconditions_checked": [
            *_preconditions_for_monitor(monitor),
            {"resource": honest_verdict.removeprefix("blocked_"), "available": False, "detail": detail},
        ],
        "stable_checkpoint_path": monitor.get("checkpoint_path"),
        "checkpoint_copy_path": None,
        "checkpoint_copy_performed": False,
        "candidate_source": "none_blocked",
        "n_candidate_pools": 0,
        "corpus_summary": {"n_matched": 0},
        "read_only_actions": {
            "training_launched": False,
            "train_process_stop_attempted": False,
            "stable_checkpoint_written": False,
            "candidate_sampling_launched": False,
        },
        "blocked_detail": detail,
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _numeric_metric_errors(metric: Any, field_name: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(metric, Mapping):
        return [f"{field_name} must be an object"]
    delta = metric.get("delta")
    if "delta" not in metric:
        errors.append(f"{field_name}.delta is required")
    elif _float_or_none(delta) is None:
        errors.append(f"{field_name}.delta must be numeric")
    if not _metric_has_ci(metric):
        errors.append(f"{field_name}.ci95 must have two numeric bounds")
    return errors


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4168 deliverable."""

    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing required field {field_name}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")

    if type(artifact.get("graft_deferred")) is not bool:
        errors.append("graft_deferred must be a bare bool")
    if type(artifact.get("verifier_value_added")) is not bool:
        errors.append("verifier_value_added must be a bare bool")
    if artifact.get("graft_deferred") is True and artifact.get("verifier_value_added") is True:
        errors.append("verifier_value_added cannot be true when graft_deferred is true")

    errors.extend(_numeric_metric_errors(artifact.get("rerank_lift_vs_vote"), "rerank_lift_vs_vote"))
    errors.extend(_numeric_metric_errors(artifact.get("rft_vs_ablation_delta"), "rft_vs_ablation_delta"))

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field_name, principle in FIELD_PRINCIPLES.items():
            if principles.get(field_name) != principle:
                errors.append(f"field_principles.{field_name} mismatch")

    if "acceptance_gate_passed" in artifact and type(artifact.get("acceptance_gate_passed")) is not bool:
        errors.append("acceptance_gate_passed must be a bare bool")
    checksum = artifact.get("reproducibility_checksum")
    if checksum is not None and not (
        isinstance(checksum, str) and checksum.startswith("sha256:") and len(checksum) == 71
    ):
        errors.append("reproducibility_checksum must be sha256-prefixed")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> JsonDict:
    validate_artifact(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonable(artifact)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _recompute_monitor(repo_root: str | Path) -> JsonDict:  # pragma: no cover - covered by artifact run.
    return exp4167.build_artifact(exp4167.MonitorConfig(repo_root=repo_root))


def _sample_checkpoint_candidate_pools(  # pragma: no cover - live CUDA/checkpoint path.
    *,
    checkpoint_copy_path: Path,
    repo_root: str | Path,
    data_dir: str | Path,
    heldout_split: str,
    max_puzzles: int,
    k_candidates: int,
    random_seed: int,
) -> list[CandidatePool]:
    return exp4109.sample_checkpoint_candidate_pools(
        checkpoint_path=checkpoint_copy_path,
        repo_root=repo_root,
        data_dir=data_dir,
        split=heldout_split,
        max_puzzles=max_puzzles,
        k_candidates=k_candidates,
        random_seed=random_seed,
    )


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    heldout_split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = DEFAULT_MAX_PUZZLES,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
    monitor_provider: Callable[[], Mapping[str, Any]] | None = None,
    checkpoint_copier: Callable[[Path, Path], Path] = copy_checkpoint_to_task_local,
    candidate_pool_provider: Callable[[Path], Sequence[CandidatePool]] | None = None,
    rft_runner: Callable[[Path, dict[str, Any]], Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run Exp 4168 and write the defensive verifier-graft artifact."""

    started = time.time()
    root = Path(repo_root)
    monitor = dict(monitor_provider()) if monitor_provider is not None else _recompute_monitor(root)

    if not baseline_is_faithful_stable(monitor):
        artifact = build_deferred_artifact(
            monitor=monitor,
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    stable_value = monitor.get("checkpoint_path")
    stable_path = Path(stable_value) if isinstance(stable_value, str) and stable_value else Path("")
    if not stable_path.is_file():
        artifact = build_blocked_artifact(
            "blocked_stable_checkpoint",
            monitor=monitor,
            detail=str(stable_path),
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    copy_target = _copy_target_for(stable_path, repo_root=root)
    try:
        checkpoint_copy_path = checkpoint_copier(stable_path, copy_target)
    except Exception as exc:  # pragma: no cover - defensive host filesystem failure.
        artifact = build_blocked_artifact(
            "blocked_checkpoint_copy",
            monitor=monitor,
            detail=f"{type(exc).__name__}: {exc}",
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    if candidate_pool_provider is not None:
        pools = list(candidate_pool_provider(checkpoint_copy_path))
        candidate_source = "provided_candidate_pool"
    else:  # pragma: no cover - live CUDA/checkpoint sampling path.
        pools = _sample_checkpoint_candidate_pools(
            checkpoint_copy_path=checkpoint_copy_path,
            repo_root=root,
            data_dir=data_dir,
            heldout_split=heldout_split,
            max_puzzles=max_puzzles,
            k_candidates=k_candidates,
            random_seed=random_seed,
        )
        candidate_source = "copied_checkpoint_final_logits_k_sampling"

    rerank = evaluate_rerank_arm(
        pools,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    corpora = exp4109.build_matched_corpora(pools)
    if rft_runner is None:
        artifact = build_blocked_artifact(
            "blocked_native_rft_runner_missing",
            monitor=monitor,
            detail="bounded cumulative+budget RFT runner was not provided",
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)
    rft_delta = dict(rft_runner(checkpoint_copy_path, corpora))

    artifact = build_result_artifact(
        monitor=monitor,
        checkpoint_copy_path=checkpoint_copy_path,
        rerank_lift_vs_vote=rerank,
        rft_vs_ablation_delta=rft_delta,
        corpora=corpora,
        candidate_source=candidate_source,
        duration_s=time.time() - started,
        random_seed=random_seed,
    )
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()

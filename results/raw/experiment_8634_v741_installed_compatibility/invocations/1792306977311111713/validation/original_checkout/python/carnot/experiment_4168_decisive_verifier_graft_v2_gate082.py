"""Exp 4168 v2 gate-082 defensive verifier graft.

This module recomputes the decisive verifier-graft preconditions from the
stable run files instead of trusting a stale monitor artifact. It refuses to
touch the outer-loop-owned baseline unless the best validation value is at
least 0.82 and both process-liveness checks show that training has stopped.

Spec refs: REQ-LEARN-4168-V2, SCENARIO-LEARN-4168-V2-DEFER,
SCENARIO-LEARN-4168-V2-COPY-GRAFT.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from carnot import experiment_4109_carnot_verifier_graft_sudoku as exp4109
from carnot import experiment_4158_verifier_rerank_recovery_moat as exp4158


JsonDict = dict[str, Any]
CandidatePool = exp4109.CandidatePool

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4168_decisive_verifier_graft_v2_gate082.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_TRM_RUNS = REPO_ROOT / "results" / "trm_runs"
DEFAULT_STABLE_CHECKPOINT = DEFAULT_TRM_RUNS / "sudoku_extreme_baseline" / "last.ckpt"
DEFAULT_BESTVAL_PATH = DEFAULT_STABLE_CHECKPOINT.with_name("last.ckpt.bestval")
DEFAULT_PID_PATH = DEFAULT_TRM_RUNS / "contiguous_run.pid"
DEFAULT_DATA_DIR = REPO_ROOT / "nano-trm" / "data" / "sudoku_extreme_1k_aug_1k"
DEFAULT_HELDOUT_SPLIT = "_valsmall"
EXPERIMENT_ID = 4168
RANDOM_SEED = 4168
FAITHFUL_VAL_THRESHOLD = 0.82
DEFAULT_MAX_PUZZLES = 64
DEFAULT_K_CANDIDATES = 8
SCHEMA = "carnot.experiment_4168_decisive_verifier_graft_v2_gate082.v1"
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
SPEC_REFS = [
    "REQ-LEARN-4168-V2",
    "SCENARIO-LEARN-4168-V2-DEFER",
    "SCENARIO-LEARN-4168-V2-COPY-GRAFT",
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


def _parse_pid_text(text: str) -> int | None:
    match = re.search(r"\d+", text)
    if match is None:
        return None
    try:
        return int(match.group(0))
    except ValueError:  # pragma: no cover - regex only returns decimal digits.
        return None


def read_bestval(path: str | Path) -> float | None:
    bestval_path = Path(path)
    try:
        return _float_or_none(bestval_path.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def read_pid(path: str | Path) -> int | None:
    pid_path = Path(path)
    try:
        return _parse_pid_text(pid_path.read_text(encoding="utf-8"))
    except OSError:
        return None


def _default_pid_alive_checker(pid: int) -> bool:  # pragma: no cover - host process probe.
    result = subprocess.run(
        ["ps", "-p", str(int(pid))],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _default_process_args(pid: int) -> str | None:  # pragma: no cover - host process probe.
    result = subprocess.run(
        ["ps", "-p", str(int(pid)), "-o", "args="],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _default_gpu_train_process_finder() -> JsonDict:  # pragma: no cover - host GPU probe.
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "detail": "nvidia-smi not found", "train_processes": []}
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "nvidia-smi failed").strip()
        return {"available": False, "detail": detail, "train_processes": []}

    train_processes: list[JsonDict] = []
    for line in result.stdout.splitlines():
        pid = _parse_pid_text(line)
        if pid is None:
            continue
        args = _default_process_args(pid)
        if args and "train.py" in args:
            train_processes.append({"pid": pid, "args": args})
    return {
        "available": True,
        "detail": "nvidia-smi compute-app pid query completed",
        "train_processes": train_processes,
    }


def _paths_for_repo(repo_root: str | Path) -> JsonDict:
    root = Path(repo_root)
    stable = root / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
    return {
        "stable_checkpoint_path": stable,
        "bestval_path": stable.with_name("last.ckpt.bestval"),
        "pid_path": root / "results" / "trm_runs" / "contiguous_run.pid",
    }


def probe_gate082_preconditions(
    *,
    repo_root: str | Path = REPO_ROOT,
    stable_checkpoint_path: str | Path | None = None,
    bestval_path: str | Path | None = None,
    pid_path: str | Path | None = None,
    pid_alive_checker: Callable[[int], bool] = _default_pid_alive_checker,
    gpu_train_process_finder: Callable[[], Mapping[str, Any]] = _default_gpu_train_process_finder,
) -> JsonDict:
    """REQ-LEARN-4168-V2: recompute all gate-082 state from stable files."""

    paths = _paths_for_repo(repo_root)
    stable = Path(stable_checkpoint_path) if stable_checkpoint_path is not None else Path(paths["stable_checkpoint_path"])
    bestval_file = Path(bestval_path) if bestval_path is not None else Path(paths["bestval_path"])
    pid_file = Path(pid_path) if pid_path is not None else Path(paths["pid_path"])

    bestval = read_bestval(bestval_file)
    pid = read_pid(pid_file)
    pid_alive = bool(pid_alive_checker(pid)) if pid is not None else False
    gpu_probe = dict(gpu_train_process_finder())
    gpu_available = gpu_probe.get("available") is True
    gpu_train_processes = list(gpu_probe.get("train_processes", []))

    bestval_passed = bestval is not None and bestval >= FAITHFUL_VAL_THRESHOLD
    pid_not_alive_passed = pid is not None and not pid_alive
    gpu_train_stopped_passed = gpu_available and not gpu_train_processes
    stable_exists = stable.is_file()
    fresh_gate = bool(bestval_passed and pid_not_alive_passed and gpu_train_stopped_passed and stable_exists)
    return {
        "stable_checkpoint_path": str(stable),
        "stable_checkpoint_exists": stable_exists,
        "bestval_path": str(bestval_file),
        "bestval_exact_accuracy": bestval,
        "bestval_threshold": FAITHFUL_VAL_THRESHOLD,
        "bestval_passed": bool(bestval_passed),
        "pid_path": str(pid_file),
        "outerloop_pid": pid,
        "outerloop_pid_alive": pid_alive,
        "pid_not_alive_passed": bool(pid_not_alive_passed),
        "gpu_query_available": bool(gpu_available),
        "gpu_query_detail": str(gpu_probe.get("detail", "")),
        "gpu_train_processes": _jsonable(gpu_train_processes),
        "gpu_train_stopped_passed": bool(gpu_train_stopped_passed),
        "fresh_gate082_stable": fresh_gate,
    }


def _preconditions_for_state(state: Mapping[str, Any]) -> list[JsonDict]:
    return [
        {
            "resource": "stable_checkpoint",
            "available": bool(state.get("stable_checkpoint_exists")),
            "detail": str(state.get("stable_checkpoint_path")),
        },
        {
            "resource": "last_ckpt_bestval_gate082",
            "available": bool(state.get("bestval_passed")),
            "detail": json.dumps(
                {
                    "path": state.get("bestval_path"),
                    "value": state.get("bestval_exact_accuracy"),
                    "threshold": FAITHFUL_VAL_THRESHOLD,
                },
                sort_keys=True,
            ),
        },
        {
            "resource": "outerloop_pid_not_alive",
            "available": bool(state.get("pid_not_alive_passed")),
            "detail": json.dumps(
                {
                    "path": state.get("pid_path"),
                    "pid": state.get("outerloop_pid"),
                    "alive": state.get("outerloop_pid_alive"),
                },
                sort_keys=True,
            ),
        },
        {
            "resource": "nvidia_smi_no_train_py",
            "available": bool(state.get("gpu_train_stopped_passed")),
            "detail": json.dumps(
                {
                    "available": state.get("gpu_query_available"),
                    "detail": state.get("gpu_query_detail"),
                    "train_processes": state.get("gpu_train_processes", []),
                },
                sort_keys=True,
            ),
        },
    ]


def _baseline_status(state: Mapping[str, Any]) -> JsonDict:
    return {
        "stable_checkpoint_path": state.get("stable_checkpoint_path"),
        "stable_checkpoint_exists": bool(state.get("stable_checkpoint_exists")),
        "bestval_path": state.get("bestval_path"),
        "bestval_exact_accuracy": _float_or_none(state.get("bestval_exact_accuracy")),
        "faithful_threshold": FAITHFUL_VAL_THRESHOLD,
        "bestval_passed": bool(state.get("bestval_passed")),
        "outerloop_pid": state.get("outerloop_pid"),
        "outerloop_pid_alive": bool(state.get("outerloop_pid_alive")),
        "pid_not_alive_passed": bool(state.get("pid_not_alive_passed")),
        "gpu_query_available": bool(state.get("gpu_query_available")),
        "gpu_train_processes": _jsonable(state.get("gpu_train_processes", [])),
        "gpu_train_stopped_passed": bool(state.get("gpu_train_stopped_passed")),
        "fresh_gate082_stable": bool(state.get("fresh_gate082_stable")),
    }


def _deferred_rerank_metric(status: str, state: Mapping[str, Any]) -> JsonDict:
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
        "current_val_exact_accuracy": _float_or_none(state.get("bestval_exact_accuracy")),
    }


def _deferred_rft_metric(status: str, state: Mapping[str, Any]) -> JsonDict:
    return {
        "metric": "heldout_exact_accuracy",
        "training_mode": "not_run",
        "n_matched": 0,
        "a_exact_accuracy": 0.0,
        "b_exact_accuracy": 0.0,
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
        "current_val_exact_accuracy": _float_or_none(state.get("bestval_exact_accuracy")),
    }


def _copy_target_for(stable_checkpoint_path: Path, *, repo_root: str | Path = REPO_ROOT) -> Path:
    root = Path(repo_root)
    return (
        root
        / "results"
        / "trm_runs"
        / "experiment_4168_decisive_verifier_graft_v2_gate082"
        / f"{stable_checkpoint_path.stem}-exp4168-v2-gate082-copy{stable_checkpoint_path.suffix}"
    )


def copy_checkpoint_to_task_local(source_path: Path, target_path: Path) -> Path:
    """SCENARIO-LEARN-4168-V2-COPY-GRAFT: freeze the baseline before model use."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path


def evaluate_rerank_arm(
    pools: Sequence[CandidatePool],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
) -> JsonDict:
    """REQ-LEARN-4168-V2: report verifier lift versus vote and oracle."""

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


def evaluate_matched_label_ablation(
    corpora: Mapping[str, Any],
    *,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = 2000,
) -> JsonDict:
    """REQ-LEARN-4168-V2: bounded A-vs-B deconfound when native RFT is absent."""

    metric = exp4109.evaluate_label_arms(
        corpora,
        random_seed=random_seed,
        bootstrap_resamples=bootstrap_resamples,
    )
    metric["training_mode"] = "matched_label_deconfound_no_native_training"
    metric["rft_native_training_launched"] = False
    return _jsonable(metric)


def verifier_value_added(rft_vs_ablation_delta: Mapping[str, Any], *, graft_deferred: bool) -> bool:
    """REQ-LEARN-4168-V2: headline bool comes only from A-vs-B CI separation."""

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


def _artifact_verdict(*, graft_deferred: bool, value_added: bool, state: Mapping[str, Any]) -> str:
    if graft_deferred:
        return f"complete: graft_deferred_outerloop_training_val_{_format_val_tag(state.get('bestval_exact_accuracy'))}"
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
        return baseline.get("fresh_gate082_stable") is False
    return (
        bool(artifact.get("checkpoint_copy_performed"))
        and isinstance(artifact.get("checkpoint_copy_path"), str)
        and _metric_has_ci(artifact.get("rerank_lift_vs_vote"))
        and _metric_has_ci(artifact.get("rft_vs_ablation_delta"))
    )


def _summarize_corpora(corpora: Mapping[str, Any]) -> JsonDict:
    rows = [row for row in corpora.get("rows", []) if isinstance(row, Mapping)]
    return {
        "arm_a": corpora.get("arm_a"),
        "arm_b": corpora.get("arm_b"),
        "n_matched": int(corpora.get("n_matched", 0)),
        "skipped_no_verifier_valid": len(corpora.get("skipped_no_verifier_valid", [])),
        "a_exact_count": sum(bool(row.get("a_exact")) for row in rows),
        "b_exact_count": sum(bool(row.get("b_exact")) for row in rows),
    }


def _finalize_artifact(artifact: JsonDict) -> JsonDict:
    artifact["acceptance_gate_passed"] = _acceptance_gate(artifact)
    artifact["reproducibility_checksum"] = _payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def build_deferred_artifact(
    *,
    state: Mapping[str, Any],
    duration_s: float,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """SCENARIO-LEARN-4168-V2-DEFER: write evidence without mutating training state."""

    rerank = _deferred_rerank_metric("deferred_outerloop_training", state)
    rft = _deferred_rft_metric("deferred_outerloop_training", state)
    artifact: JsonDict = {
        "experiment": "experiment_4168_decisive_verifier_graft_v2_gate082",
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _artifact_verdict(graft_deferred=True, value_added=False, state=state),
        "graft_deferred": True,
        "rerank_lift_vs_vote": rerank,
        "rft_vs_ablation_delta": rft,
        "verifier_value_added": False,
        "baseline_status": _baseline_status(state),
        "preconditions_checked": _preconditions_for_state(state),
        "stable_checkpoint_path": state.get("stable_checkpoint_path"),
        "checkpoint_copy_path": None,
        "checkpoint_copy_performed": False,
        "candidate_source": "none_deferred_outerloop_training",
        "n_candidate_pools": 0,
        "corpus_summary": {"n_matched": 0, "arm_a": "verifier_certified", "arm_b": "vote_certified"},
        "rft_native_training_launched": False,
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
    return _finalize_artifact(artifact)


def build_result_artifact(
    *,
    state: Mapping[str, Any],
    checkpoint_copy_path: Path,
    rerank_lift_vs_vote: Mapping[str, Any],
    rft_vs_ablation_delta: Mapping[str, Any],
    corpora: Mapping[str, Any],
    candidate_source: str,
    duration_s: float,
    rft_native_training_launched: bool,
    random_seed: int = RANDOM_SEED,
) -> JsonDict:
    """SCENARIO-LEARN-4168-V2-COPY-GRAFT: build the copied-checkpoint artifact."""

    value_added = verifier_value_added(rft_vs_ablation_delta, graft_deferred=False)
    artifact: JsonDict = {
        "experiment": "experiment_4168_decisive_verifier_graft_v2_gate082",
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": _artifact_verdict(
            graft_deferred=False,
            value_added=value_added,
            state=state,
        ),
        "graft_deferred": False,
        "rerank_lift_vs_vote": _jsonable(rerank_lift_vs_vote),
        "rft_vs_ablation_delta": _jsonable(rft_vs_ablation_delta),
        "verifier_value_added": bool(value_added),
        "baseline_status": _baseline_status(state),
        "preconditions_checked": _preconditions_for_state(state),
        "stable_checkpoint_path": state.get("stable_checkpoint_path"),
        "checkpoint_copy_path": str(checkpoint_copy_path),
        "checkpoint_copy_performed": True,
        "candidate_source": candidate_source,
        "n_candidate_pools": int(rerank_lift_vs_vote.get("n_puzzles", 0)),
        "corpus_summary": _summarize_corpora(corpora),
        "rft_native_training_launched": bool(rft_native_training_launched),
        "read_only_actions": {
            "training_launched": bool(rft_native_training_launched),
            "train_process_stop_attempted": False,
            "stable_checkpoint_written": False,
            "candidate_sampling_launched": True,
        },
        "random_seed": int(random_seed),
        "duration_s": round(float(duration_s), 3),
        "field_principles": dict(FIELD_PRINCIPLES),
        "acceptance_gate_passed": False,
    }
    return _finalize_artifact(artifact)


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
    """Return explicit schema errors for the Exp 4168 v2 deliverable."""

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
    stable_checkpoint_path: str | Path | None = None,
    bestval_path: str | Path | None = None,
    pid_path: str | Path | None = None,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    heldout_split: str = DEFAULT_HELDOUT_SPLIT,
    max_puzzles: int = DEFAULT_MAX_PUZZLES,
    k_candidates: int = DEFAULT_K_CANDIDATES,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
    pid_alive_checker: Callable[[int], bool] = _default_pid_alive_checker,
    gpu_train_process_finder: Callable[[], Mapping[str, Any]] = _default_gpu_train_process_finder,
    checkpoint_copier: Callable[[Path, Path], Path] = copy_checkpoint_to_task_local,
    candidate_pool_provider: Callable[[Path], Sequence[CandidatePool]] | None = None,
    native_rft_runner: Callable[[Path, dict[str, Any]], Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Run Exp 4168 v2 and write the gate-082 verifier-graft artifact."""

    started = time.time()
    root = Path(repo_root)
    state = probe_gate082_preconditions(
        repo_root=root,
        stable_checkpoint_path=stable_checkpoint_path,
        bestval_path=bestval_path,
        pid_path=pid_path,
        pid_alive_checker=pid_alive_checker,
        gpu_train_process_finder=gpu_train_process_finder,
    )

    if not state["fresh_gate082_stable"]:
        artifact = build_deferred_artifact(
            state=state,
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        return write_artifact(output_path, artifact)

    stable_path = Path(str(state["stable_checkpoint_path"]))
    copy_target = _copy_target_for(stable_path, repo_root=root)
    checkpoint_copy_path = checkpoint_copier(stable_path, copy_target)

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
    if native_rft_runner is None:
        rft_delta = evaluate_matched_label_ablation(
            corpora,
            random_seed=random_seed + 1,
            bootstrap_resamples=bootstrap_resamples,
        )
        rft_native_training_launched = False
    else:
        rft_delta = dict(native_rft_runner(checkpoint_copy_path, corpora))
        rft_delta.setdefault("training_mode", "default_cumulative_plus_budget_same_copy_init")
        rft_native_training_launched = True

    artifact = build_result_artifact(
        state=state,
        checkpoint_copy_path=checkpoint_copy_path,
        rerank_lift_vs_vote=rerank,
        rft_vs_ablation_delta=rft_delta,
        corpora=corpora,
        candidate_source=candidate_source,
        duration_s=time.time() - started,
        rft_native_training_launched=rft_native_training_launched,
        random_seed=random_seed,
    )
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()

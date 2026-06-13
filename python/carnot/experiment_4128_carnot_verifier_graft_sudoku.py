"""Exp 4128 conditional Sudoku verifier graft over the fixed-LR TRM baseline.

Spec refs: REQ-LEARN-4128, SCENARIO-LEARN-4128-DEFER,
SCENARIO-LEARN-4128-GRAFT.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from carnot import experiment_4107_nanotrm_mechanism_smoke as exp4107


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_FILENAME = "experiment_4128_carnot_verifier_graft_sudoku.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_EXP4127_ARTIFACT = REPO_ROOT / "results" / "experiment_4127_sudoku_extreme_accumulate_fixed.json"
DEFAULT_STABLE_CHECKPOINT = REPO_ROOT / "results" / "trm_runs" / "sudoku_extreme_baseline" / "last.ckpt"
FAITHFUL_BASELINE_THRESHOLD = 0.85
RANDOM_SEED = 4128
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
BLOCKED_PREFIX = "blocked_"
SPEC_REFS = ["REQ-LEARN-4128", "SCENARIO-LEARN-4128-DEFER", "SCENARIO-LEARN-4128-GRAFT"]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "graft_deferred",
    "rerank_lift_vs_vote",
    "rft_vs_ablation_delta",
    "verifier_value_added",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal-prefixed. An honest deferral or an A~=B null is COMPLETE.",
    "graft_deferred": (
        "Bare bool: True if the baseline was not faithful enough to graft. "
        "Prevents a meaningless graft being dressed as a result."
    ),
    "rerank_lift_vs_vote": (
        "pass@1 lift from verifier-reranking (if grafted); confirms the executable "
        "Sudoku verifier discriminates (the contrast to .379's ARC-grid anti-discrimination)."
    ),
    "rft_vs_ablation_delta": (
        "The de-confounded A-vs-B held-out delta with CI (if grafted): isolates "
        "the verifier LABEL's training contribution."
    ),
    "verifier_value_added": (
        "Bare bool: did the verifier graft beat the vote ablation? The headline "
        "answer, ONLY meaningful when graft_deferred is False."
    ),
    "preconditions_checked": (
        "Records the baseline checkpoint + CUDA verified; pre-empts "
        "silent-missing-resource fabrication."
    ),
}


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before Exp 4128 chooses a branch."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineContext:
    """Exp 4127 baseline evidence that decides whether a graft is meaningful.

    The verifier graft is only decision-grade after the TRM baseline is close
    to the published Sudoku-Extreme result. A weak baseline can make any
    verifier comparison look better or worse for the wrong reason, so this
    context keeps the exact validation value and checkpoint provenance adjacent
    to the branch decision.
    """

    artifact_path: Path
    stable_checkpoint_path: Path
    val_exact_accuracy: float | None
    previous_val_exact_accuracy: float | None
    matches_published_087: bool
    val_trajectory: Sequence[Mapping[str, Any]]
    raw: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": str(self.artifact_path),
            "stable_checkpoint_path": str(self.stable_checkpoint_path),
            "val_exact_accuracy": self.val_exact_accuracy,
            "previous_val_exact_accuracy": self.previous_val_exact_accuracy,
            "matches_published_087": self.matches_published_087,
            "val_trajectory": _jsonable(self.val_trajectory),
        }


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _trajectory_values(payload: Mapping[str, Any]) -> list[float]:
    trajectory = payload.get("val_trajectory")
    if not isinstance(trajectory, Sequence) or isinstance(trajectory, (str, bytes)):
        return []
    values: list[float] = []
    for entry in trajectory:
        if not isinstance(entry, Mapping):
            continue
        value = _float_or_none(entry.get("val_exact_accuracy"))
        if value is not None:
            values.append(value)
    return values


def _current_val(payload: Mapping[str, Any], values: Sequence[float]) -> float | None:
    explicit = _float_or_none(payload.get("val_exact_accuracy"))
    if explicit is not None:
        return explicit
    final = _float_or_none(payload.get("final_val_exact_accuracy"))
    if final is not None:
        return final
    return values[-1] if values else None


def _previous_val(current: float | None, values: Sequence[float]) -> float | None:
    if current is None:
        return values[-1] if values else None
    for value in reversed(values[:-1] if values and values[-1] == current else values):
        if value != current:
            return value
    return None


def find_exp4127_artifact(repo_root: str | Path = REPO_ROOT) -> Path:
    """REQ-LEARN-4128: find the latest Exp 4127 fixed-LR baseline artifact."""

    root = Path(repo_root)
    matches = sorted((root / "results").glob("experiment_4127_*.json"))
    if matches:
        return matches[-1]
    return root / "results" / DEFAULT_EXP4127_ARTIFACT.name


def load_baseline_context(path: str | Path) -> BaselineContext:
    """REQ-LEARN-4128: load Exp 4127 baseline status before grafting."""

    artifact_path = Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Exp 4127 artifact must be a JSON object")

    stable_value = payload.get("stable_checkpoint_path")
    stable = Path(stable_value) if isinstance(stable_value, str) else DEFAULT_STABLE_CHECKPOINT
    values = _trajectory_values(payload)
    current = _current_val(payload, values)
    trajectory = payload.get("val_trajectory")
    return BaselineContext(
        artifact_path=artifact_path,
        stable_checkpoint_path=stable,
        val_exact_accuracy=current,
        previous_val_exact_accuracy=_previous_val(current, values),
        matches_published_087=payload.get("matches_published_087") is True,
        val_trajectory=list(trajectory) if isinstance(trajectory, list) else [],
        raw=dict(payload),
    )


def baseline_is_faithful(baseline: BaselineContext) -> bool:
    """REQ-LEARN-4128: decide whether verifier-as-reward grafting is meaningful."""

    return baseline.matches_published_087 or (
        baseline.val_exact_accuracy is not None
        and baseline.val_exact_accuracy >= FAITHFUL_BASELINE_THRESHOLD
    )


def estimate_passes_to_converge_for_383(
    baseline: BaselineContext,
    *,
    target: float = FAITHFUL_BASELINE_THRESHOLD,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4128-DEFER: estimate more fixed-LR passes for `.383`."""

    current = baseline.val_exact_accuracy
    previous = baseline.previous_val_exact_accuracy
    delta = None if current is None or previous is None else current - previous
    if current is None:
        estimate = None
        basis = "missing_current_val_exact_accuracy"
    elif current >= target:
        estimate = 0
        basis = "already_at_or_above_target"
    elif delta is None or delta <= 0.0:
        estimate = None
        basis = "no_positive_fixed_lr_pass_improvement"
    else:
        estimate = int(math.ceil((target - current) / delta))
        basis = "latest_fixed_lr_pass_delta"
    return {
        "destination": ".383",
        "target_val_exact_accuracy": float(target),
        "current_val_exact_accuracy": current,
        "previous_val_exact_accuracy": previous,
        "observed_delta_per_pass": None if delta is None else round(float(delta), 6),
        "estimated_additional_passes": estimate,
        "basis": basis,
    }


def _default_cuda_checker() -> tuple[bool, str]:  # pragma: no cover - host dependent.
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - host dependent.
        return False, f"{type(exc).__name__}: {exc}"
    available = bool(torch.cuda.is_available())
    detail = f"torch.cuda.is_available()={available}"
    if available:
        detail += f"; device={torch.cuda.get_device_name(0)}"
    return available, detail


def check_preconditions(
    baseline: BaselineContext,
    *,
    cuda_checker: Callable[[], tuple[bool, str]] = _default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
) -> list[PreconditionCheck]:
    """REQ-LEARN-4128: record baseline artifact, checkpoint, and CUDA checks."""

    checks = [
        PreconditionCheck(
            "baseline_artifact",
            baseline.artifact_path.is_file(),
            str(baseline.artifact_path),
        ),
        PreconditionCheck(
            "stable_checkpoint_path",
            bool(str(baseline.stable_checkpoint_path)),
            str(baseline.stable_checkpoint_path),
        ),
    ]
    if baseline.stable_checkpoint_path.is_file():
        try:
            checkpoint_ok, checkpoint_detail = checkpoint_loader(baseline.stable_checkpoint_path)
        except Exception as exc:
            checkpoint_ok, checkpoint_detail = False, f"{type(exc).__name__}: {exc}"
    else:
        checkpoint_ok = False
        checkpoint_detail = f"missing: {baseline.stable_checkpoint_path}"
    checks.append(PreconditionCheck("stable_checkpoint", bool(checkpoint_ok), str(checkpoint_detail)))

    try:
        cuda_ok, cuda_detail = cuda_checker()
    except Exception as exc:
        cuda_ok, cuda_detail = False, f"{type(exc).__name__}: {exc}"
    checks.append(PreconditionCheck("cuda_available", bool(cuda_ok), str(cuda_detail)))
    return checks


def _checks_to_dicts(checks: Sequence[PreconditionCheck | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [check.to_dict() if isinstance(check, PreconditionCheck) else dict(check) for check in checks]


def _all_preconditions_available(checks: Sequence[PreconditionCheck | Mapping[str, Any]]) -> bool:
    return all(bool(check.available) if isinstance(check, PreconditionCheck) else check.get("available") is True for check in checks)


def verifier_value_added(rft_vs_ablation_delta: Mapping[str, Any]) -> bool:
    """SCENARIO-LEARN-4128-GRAFT: headline bool from A-vs-B CI separation."""

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


def _common_artifact(
    *,
    baseline: BaselineContext | None,
    honest_verdict: str,
    graft_deferred: bool,
    rerank_lift_vs_vote: Mapping[str, Any] | None,
    rft_vs_ablation_delta: Mapping[str, Any] | None,
    verifier_added: bool,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
    acceptance_gate_passed: bool,
) -> dict[str, Any]:
    estimate = None if baseline is None else estimate_passes_to_converge_for_383(baseline)
    return {
        "experiment": "experiment_4128_carnot_verifier_graft_sudoku",
        "schema": "carnot.experiment_4128_carnot_verifier_graft_sudoku.v1",
        "spec_refs": list(SPEC_REFS),
        "honest_verdict": honest_verdict,
        "graft_deferred": bool(graft_deferred),
        "baseline_val_exact_accuracy": None if baseline is None else baseline.val_exact_accuracy,
        "baseline_matches_published_087": False if baseline is None else baseline.matches_published_087,
        "baseline_status_reported": baseline is not None and baseline.val_exact_accuracy is not None,
        "stable_checkpoint_path": None if baseline is None else str(baseline.stable_checkpoint_path),
        "baseline_artifact_path": None if baseline is None else str(baseline.artifact_path),
        "baseline": None if baseline is None else baseline.to_dict(),
        "estimated_passes_to_converge_for_383": estimate,
        "rerank_lift_vs_vote": _jsonable(rerank_lift_vs_vote),
        "rft_vs_ablation_delta": _jsonable(rft_vs_ablation_delta),
        "verifier_value_added": bool(verifier_added),
        "verifier_value_added_meaningful": not graft_deferred,
        "preconditions_checked": _checks_to_dicts(preconditions_checked),
        "acceptance_gate_passed": bool(acceptance_gate_passed),
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "duration_s": round(float(duration_s), 3),
    }


def build_deferred_artifact(
    *,
    baseline: BaselineContext,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4128-DEFER: write a complete non-faithful-baseline verdict."""

    value = baseline.val_exact_accuracy
    rendered = "unknown" if value is None else f"{value:.4f}"
    estimate = estimate_passes_to_converge_for_383(baseline)["estimated_additional_passes"]
    rendered_estimate = "unknown" if estimate is None else str(estimate)
    artifact = _common_artifact(
        baseline=baseline,
        honest_verdict=(
            f"complete: graft_deferred baseline_val={rendered} "
            f"estimated_passes_to_.383={rendered_estimate}"
        ),
        graft_deferred=True,
        rerank_lift_vs_vote=None,
        rft_vs_ablation_delta=None,
        verifier_added=False,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        acceptance_gate_passed=baseline.val_exact_accuracy is not None,
    )
    validate_artifact(artifact)
    return artifact


def build_grafted_artifact(
    *,
    baseline: BaselineContext,
    rerank_lift_vs_vote: Mapping[str, Any],
    rft_vs_ablation_delta: Mapping[str, Any],
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4128-GRAFT: report rerank and RFT deconfound metrics."""

    value_added = verifier_value_added(rft_vs_ablation_delta)
    verdict = (
        "success: verifier_value_added_A_gt_B_ci95_excludes_zero"
        if value_added
        else "complete: A~=B null verifier graft measured"
    )
    artifact = _common_artifact(
        baseline=baseline,
        honest_verdict=verdict,
        graft_deferred=False,
        rerank_lift_vs_vote=rerank_lift_vs_vote,
        rft_vs_ablation_delta=rft_vs_ablation_delta,
        verifier_added=value_added,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        acceptance_gate_passed=True,
    )
    validate_artifact(artifact)
    return artifact


def build_blocked_artifact(
    honest_verdict: str,
    *,
    baseline: BaselineContext | None,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float,
) -> dict[str, Any]:
    """REQ-LEARN-4128: fail closed when baseline or resource evidence is absent."""

    artifact = _common_artifact(
        baseline=baseline,
        honest_verdict=honest_verdict,
        graft_deferred=True,
        rerank_lift_vs_vote=None,
        rft_vs_ablation_delta=None,
        verifier_added=False,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        acceptance_gate_passed=False,
    )
    validate_artifact(artifact)
    return artifact


def _ci_reported(metric: Mapping[str, Any]) -> bool:
    ci95 = metric.get("ci95")
    if not isinstance(ci95, Sequence) or isinstance(ci95, (str, bytes)) or len(ci95) != 2:
        return False
    return _float_or_none(ci95[0]) is not None and _float_or_none(ci95[1]) is not None


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return explicit schema errors for the Exp 4128 deliverable."""

    errors: list[str] = []
    for field_name in REQUIRED_ARTIFACT_FIELDS:
        if field_name not in artifact:
            errors.append(f"missing required field {field_name}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith((*TERMINAL_PREFIXES, BLOCKED_PREFIX)):
        errors.append("honest_verdict must be terminal-prefixed or blocked")

    graft_deferred = artifact.get("graft_deferred")
    if type(graft_deferred) is not bool:
        errors.append("graft_deferred must be a bare bool")
    if type(artifact.get("verifier_value_added")) is not bool:
        errors.append("verifier_value_added must be a bare bool")

    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list):
        errors.append("preconditions_checked must be a list")
    elif any(
        not isinstance(item, Mapping) or "resource" not in item or "available" not in item
        for item in preconditions
    ):
        errors.append("preconditions_checked entries must include resource and available")

    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        errors.append("field_principles must be an object")
    else:
        for field_name, principle in FIELD_PRINCIPLES.items():
            if principles.get(field_name) != principle:
                errors.append(f"field_principles.{field_name} mismatch")

    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        errors.append("duration_s must be numeric")
    if type(artifact.get("acceptance_gate_passed")) is not bool:
        errors.append("acceptance_gate_passed must be a bare bool")

    if graft_deferred is True and artifact.get("acceptance_gate_passed") is True:
        if not isinstance(artifact.get("baseline_val_exact_accuracy"), (int, float)):
            errors.append("accepted deferral requires baseline_val_exact_accuracy")
        if not isinstance(artifact.get("stable_checkpoint_path"), str):
            errors.append("accepted deferral requires stable_checkpoint_path")
        if not isinstance(artifact.get("estimated_passes_to_converge_for_383"), Mapping):
            errors.append("accepted deferral requires estimated_passes_to_converge_for_383")

    if graft_deferred is False:
        for field_name in ("rerank_lift_vs_vote", "rft_vs_ablation_delta"):
            metric = artifact.get(field_name)
            if not isinstance(metric, Mapping):
                errors.append(f"{field_name} must be an object when graft_deferred is false")
                continue
            if "delta" not in metric:
                errors.append(f"{field_name}.delta is required")
            if not _ci_reported(metric):
                errors.append(f"{field_name}.ci95 must have two numeric bounds")
    return errors


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))


def write_artifact(path: str | Path, artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Write the stable Exp 4128 JSON artifact."""

    validate_artifact(artifact)
    payload = _jsonable(artifact)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path = DEFAULT_OUTPUT,
    exp4127_artifact_path: str | Path | None = None,
    cuda_checker: Callable[[], tuple[bool, str]] = _default_cuda_checker,
    checkpoint_loader: Callable[[Path], tuple[bool, str]] = exp4107._load_torch_checkpoint,
    graft_runner: Callable[[BaselineContext, list[PreconditionCheck], float], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run Exp 4128 and write either an honest deferral or grafted measurement."""

    started = time.time()
    baseline_path = Path(exp4127_artifact_path) if exp4127_artifact_path is not None else find_exp4127_artifact(repo_root)
    try:
        baseline = load_baseline_context(baseline_path)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        artifact = build_blocked_artifact(
            "blocked_exp4127_baseline_missing",
            baseline=None,
            preconditions_checked=[
                PreconditionCheck("baseline_artifact", False, str(baseline_path)).to_dict()
            ],
            duration_s=time.time() - started,
        )
        return write_artifact(output_path, artifact)

    checks = check_preconditions(baseline, cuda_checker=cuda_checker, checkpoint_loader=checkpoint_loader)
    if not _all_preconditions_available(checks):
        artifact = build_blocked_artifact(
            "blocked_exp4128_preconditions_missing",
            baseline=baseline,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        return write_artifact(output_path, artifact)

    if not baseline_is_faithful(baseline):
        artifact = build_deferred_artifact(
            baseline=baseline,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        return write_artifact(output_path, artifact)

    if graft_runner is None:
        artifact = build_blocked_artifact(
            "blocked_exp4128_graft_runner_missing",
            baseline=baseline,
            preconditions_checked=checks,
            duration_s=time.time() - started,
        )
        return write_artifact(output_path, artifact)

    artifact = dict(graft_runner(baseline, checks, started))
    return write_artifact(output_path, artifact)


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()

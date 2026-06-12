"""Exp 4080 Sudoku verifier-RFT positive control.

Spec refs: REQ-LEARN-4080, SCENARIO-LEARN-4080,
SCENARIO-LEARN-4080-FAIL.

This module reduces the original Sudoku verifier-as-teacher beachhead into the
same decision shape needed by the ARC RFT pipeline: verifier-certified RFT
versus gold-SFT on held-out seeds. The Sudoku source artifact already contains
the expensive live-GPU training run; Exp 4080 adds the pipeline sanity contract
around it and checks the same GPU/trainer preconditions that Exp 4078 requires.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from carnot.agentic.arc_exp4078_verifier_reward_rft_train_launch import (
    check_cuda_visible as exp4078_check_cuda_visible,
    check_trainer_imports as exp4078_check_trainer_imports,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILENAME = "experiment_4080_sudoku_rft_positive_control.json"
DEFAULT_SOURCE_RESULT = Path("results/sudoku_energy_teacher_v6_v4_decisive.json")
INFERENCE_SUBSTRATE = "live_gpu_sudoku_verifier_certified_rft_with_exp4078_preflight"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "rft_rate",
    "sft_rate",
    "n_seeds",
    "reproduces_beachhead",
    "inference_substrate",
)
TERMINAL_PREFIXES = ("complete:", "blocked_", "failed:")


@dataclass(frozen=True)
class PreconditionCheck:
    """One Exp 4078-style resource check needed before issuing the verdict."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SeedRate:
    """Held-out Sudoku rates for one seed in the positive-control comparison."""

    seed: int
    rft_rate: float
    sft_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coerce_precondition(check: object) -> PreconditionCheck:
    return PreconditionCheck(
        resource=str(getattr(check, "resource")),
        available=bool(getattr(check, "available")),
        detail=str(getattr(check, "detail")),
    )


def check_preconditions(*, repo_root: str | Path = REPO_ROOT) -> list[PreconditionCheck]:  # pragma: no cover
    """REQ-LEARN-4080-1: reuse the Exp 4078 CUDA and trainer checks."""

    del repo_root
    return [
        _coerce_precondition(exp4078_check_cuda_visible()),
        _coerce_precondition(exp4078_check_trainer_imports()),
    ]


def _first_missing(checks: Sequence[PreconditionCheck]) -> PreconditionCheck | None:
    return next((check for check in checks if not check.available), None)


def load_source_artifact(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source artifact must be a JSON object")
    return payload


def extract_seed_rates(source_artifact: Mapping[str, object]) -> list[SeedRate]:
    """REQ-LEARN-4080: extract paired RFT/SFT rates from held-out seed rows."""

    rows = source_artifact.get("per_seed")
    if not isinstance(rows, list):
        raise ValueError("source artifact missing per_seed list")
    if len(rows) < 3:
        raise ValueError("source artifact must contain at least 3 seeds")

    seed_rates: list[SeedRate] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"seed row {index} must be an object")
        missing = [
            field
            for field in ("seed", "energy_distilled_greedy", "gold_distilled_greedy")
            if field not in row
        ]
        if missing:
            raise ValueError(f"seed row {index} missing {', '.join(missing)}")
        seed = row["seed"]
        rft_rate = row["energy_distilled_greedy"]
        sft_rate = row["gold_distilled_greedy"]
        if type(seed) is not int:
            raise ValueError(f"seed row {index} has non-integer seed")
        if (
            not isinstance(rft_rate, int | float)
            or type(rft_rate) is bool
            or not isinstance(sft_rate, int | float)
            or type(sft_rate) is bool
        ):
            raise ValueError(f"seed row {index} has non-numeric rates")
        seed_rates.append(SeedRate(seed=int(seed), rft_rate=float(rft_rate), sft_rate=float(sft_rate)))
    return seed_rates


def mean_rate(values: Iterable[float]) -> float:
    vals = list(values)
    return float(round(sum(vals) / len(vals), 6))


def positive_control_reproduces(seed_rates: Sequence[SeedRate]) -> bool:
    """REQ-LEARN-4080-5: require aggregate and every seed to satisfy RFT >= SFT."""

    return (
        len(seed_rates) >= 3
        and mean_rate(row.rft_rate for row in seed_rates) >= mean_rate(row.sft_rate for row in seed_rates)
        and all(row.rft_rate >= row.sft_rate for row in seed_rates)
    )


def _preconditions_payload(checks: Sequence[PreconditionCheck]) -> list[dict[str, object]]:
    return [check.to_dict() for check in checks]


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal state makes a failed positive control block ARC interpretation instead of silently passing.",
        "rft_rate": "Mean held-out Sudoku solve rate for verifier-certified RFT over the included seeds.",
        "sft_rate": "Mean held-out Sudoku solve rate for gold-SFT over the same seeds.",
        "n_seeds": "Replication count; the beachhead requires at least three seeds.",
        "reproduces_beachhead": "pipeline sanity that makes the ARC headline interpretable.",
        "inference_substrate": "Declares the live-GPU Sudoku source run plus Exp 4078-style preflight.",
    }


def _base_artifact(
    *,
    honest_verdict: str,
    rft_rate: float,
    sft_rate: float,
    n_seeds: int,
    reproduces_beachhead: bool,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
    extra: Mapping[str, object],
) -> dict[str, object]:
    artifact: dict[str, object] = {
        "experiment": "experiment_4080_sudoku_rft_positive_control",
        "schema": "carnot.experiment_4080_sudoku_rft_positive_control.v1",
        "honest_verdict": honest_verdict,
        "rft_rate": float(rft_rate),
        "sft_rate": float(sft_rate),
        "n_seeds": int(n_seeds),
        "reproduces_beachhead": bool(reproduces_beachhead),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": _preconditions_payload(preconditions_checked),
        "field_principles": _field_principles(),
        "duration_s": float(duration_s),
        "spec_refs": ["REQ-LEARN-4080", "SCENARIO-LEARN-4080", "SCENARIO-LEARN-4080-FAIL"],
    }
    artifact.update(dict(extra))
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_blocked_artifact(
    *,
    missing: PreconditionCheck,
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
) -> dict[str, object]:
    """REQ-LEARN-4080-1: block the positive-control claim when resources fail."""

    return _base_artifact(
        honest_verdict=f"blocked_{missing.resource}",
        rft_rate=0.0,
        sft_rate=0.0,
        n_seeds=0,
        reproduces_beachhead=False,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        extra={"source_artifact": None, "per_seed": []},
    )


def build_complete_artifact(
    *,
    source_artifact: Mapping[str, object],
    source_path: str | Path,
    seed_rates: Sequence[SeedRate],
    preconditions_checked: Sequence[PreconditionCheck],
    duration_s: float,
) -> dict[str, object]:
    """SCENARIO-LEARN-4080: emit the reproduced or suspect pipeline verdict."""

    rft_rate = mean_rate(row.rft_rate for row in seed_rates)
    sft_rate = mean_rate(row.sft_rate for row in seed_rates)
    reproduced = positive_control_reproduces(seed_rates)
    verdict = (
        "complete: sudoku_positive_control_rft_ge_sft_reproduced"
        if reproduced
        else "complete: sudoku_positive_control_FAILED_pipeline_suspect"
    )
    return _base_artifact(
        honest_verdict=verdict,
        rft_rate=rft_rate,
        sft_rate=sft_rate,
        n_seeds=len(seed_rates),
        reproduces_beachhead=reproduced,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        extra={
            "source_artifact": str(source_path),
            "source_honest_verdict": source_artifact.get("honest_verdict"),
            "source_inference_substrate": source_artifact.get("inference_substrate"),
            "per_seed": [row.to_dict() for row in seed_rates],
            "comparison": "verifier_certified_rft_energy_distilled_greedy_vs_gold_sft_gold_distilled_greedy",
        },
    )


def artifact_schema_errors(artifact: Mapping[str, object]) -> list[str]:
    """Validate the bare Exp 4080 positive-control artifact contract."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")

    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must be terminal-prefixed")

    for field in ("rft_rate", "sft_rate"):
        value = artifact.get(field)
        if not isinstance(value, float) or type(value) is bool:
            errors.append(f"{field} must be a bare float")
    if type(artifact.get("n_seeds")) is not int:
        errors.append("n_seeds must be a bare int")
    if type(artifact.get("reproduces_beachhead")) is not bool:
        errors.append("reproduces_beachhead must be a bare bool")
    if not artifact.get("inference_substrate") or not isinstance(artifact.get("inference_substrate"), str):
        errors.append("inference_substrate must be a non-empty string")

    preconditions = artifact.get("preconditions_checked")
    if not isinstance(preconditions, list):
        errors.append("preconditions_checked must be a list")
    elif any(not isinstance(item, Mapping) or "resource" not in item or "available" not in item for item in preconditions):
        errors.append("preconditions_checked entries must include resource and available")
    return errors


def write_result_artifact(artifact: Mapping[str, object], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
    preconditions_checker: Callable[..., Sequence[PreconditionCheck]] = check_preconditions,
) -> dict[str, object]:
    """REQ-LEARN-4080: aggregate the Sudoku positive control and write JSON."""

    started = time.perf_counter()
    root = Path(repo_root)
    source = Path(source_path) if source_path is not None else root / DEFAULT_SOURCE_RESULT
    output = Path(output_path) if output_path is not None else root / "results" / RESULT_FILENAME
    checks = list(preconditions_checker(repo_root=root))
    missing = _first_missing(checks)
    if missing is not None:
        artifact = build_blocked_artifact(
            missing=missing,
            preconditions_checked=checks,
            duration_s=time.perf_counter() - started,
        )
        write_result_artifact(artifact, output)
        return artifact

    source_artifact = load_source_artifact(source)
    seed_rates = extract_seed_rates(source_artifact)
    artifact = build_complete_artifact(
        source_artifact=source_artifact,
        source_path=source,
        seed_rates=seed_rates,
        preconditions_checked=checks,
        duration_s=time.perf_counter() - started,
    )
    write_result_artifact(artifact, output)
    return artifact

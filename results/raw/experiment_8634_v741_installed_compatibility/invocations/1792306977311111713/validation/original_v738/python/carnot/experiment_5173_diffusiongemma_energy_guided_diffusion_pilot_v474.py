"""Exp 5173 DiffusionGemma energy-guided diffusion pilot.

Spec refs: REQ-VERIFY-5173, SCENARIO-VERIFY-5173-GATED,
SCENARIO-VERIFY-5173-PILOT.

This module keeps the experiment report honest when the heavyweight GPU path is
available, blocked, or only partially debugged. The executable-code verifier is
the correctness oracle for HumanEval/MBPP, so any completed result is framed as
execution-grounded guidance evidence, not an oracle-distinct moat claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v474"
MILESTONE = "2026.07.474"
RESULT_RELATIVE_PATH = (
    "results/experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v474.json"
)
EXP5171_RELATIVE_PATH = (
    "results/experiment_5171_harden_set_encoder_cross_corpus_n30_v474.json"
)
SPEC_REFS = [
    "REQ-VERIFY-5173",
    "SCENARIO-VERIFY-5173-GATED",
    "SCENARIO-VERIFY-5173-PILOT",
]
RANDOM_SEED = 5173
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "blocked_")
ARMS = ("unguided_diffusion", "guided_diffusion", "ar_best_of_n")
PASS_FIELDS = {
    "unguided_diffusion": "unguided_passed",
    "guided_diffusion": "guided_passed",
    "ar_best_of_n": "ar_passed",
}
GUIDANCE_MECHANISM_DESIGN = (
    "Per denoising step, score the current token-candidate slate with the native "
    "DiffusionGemma logits and an executable-verifier energy. The composed score "
    "for token t at position i is logit_model(i,t) - lambda_energy * "
    "verifier_energy(partial_canvas with t inserted). Sampling/commit uses the "
    "same low-entropy DiffusionGemma commit order after this reweighting; no "
    "post-hoc reranking of completed samples is counted as guided diffusion."
)

FIELD_PRINCIPLES = {
    "meta_tensor_bug_resolution": (
        "Description of the fix applied, or blocked_diffusiongemma_meta_tensor_bug_unresolved with what was tried."
    ),
    "guidance_mechanism_design": (
        "The exact mechanism (logit reweighting vs. resample-on-energy-increase vs. other) must be documented precisely enough for a third party to reproduce."
    ),
    "pass_at_1_unguided": (
        "Executable pass@1 for the unguided DiffusionGemma diffusion baseline."
    ),
    "pass_at_1_guided": "Executable pass@1 for energy-guided DiffusionGemma diffusion.",
    "pass_at_1_ar_baseline": (
        "Executable pass@1 for the best-of-N autoregressive Gemma control."
    ),
    "guided_vs_unguided_delta_ci95": (
        "Bootstrap CI95 over task-level guided-minus-unguided pass@1 deltas."
    ),
    "guided_vs_ar_delta_ci95": (
        "Bootstrap CI95 over task-level guided-diffusion-minus-AR pass@1 deltas."
    ),
    "compute_cost_per_arm": (
        "The north star's efficiency axis is co-equal with accuracy -- a guided win that costs 10x more compute is a different, weaker finding than a Pareto win."
    ),
    "verifier_is_oracle": (
        "The executable test suite IS the oracle here; per the Circularity/Oracle-Distinctness Discipline this is execution-grounded, not a fresh oracle-distinct claim -- must not be headlined as moat proven."
    ),
    "random_seed": "Seed used for task ordering, sampling, and bootstrap.",
    "reproducibility_checksum": (
        "Hash of the gate artifact, smoke outcome, arm rows, metrics, compute costs, and verdict."
    ),
    "gpu1_availability_checked": (
        "True only after the runner explicitly checked GPU 1 before launching the two-GPU DiffusionGemma path."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ or blocked_ and state plainly whether energy guidance helped, hurt, or made no difference."
    ),
}

REQUIRED_PRINCIPLED_FIELDS = tuple(
    field for field in FIELD_PRINCIPLES if field != "honest_verdict"
)
REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "experiment",
        "experiment_id",
        "milestone",
        "spec_refs",
        "result_path",
        "field_principles",
        "inference_substrate",
        "preconditions",
        "arm_rows",
        "duration_s",
        "tests_run",
        *REQUIRED_PRINCIPLED_FIELDS,
    }
)


@dataclass(frozen=True)
class GpuAvailability:
    """Result of the required GPU-1 preflight check."""

    checked: bool
    gpu1_available: bool
    detail: str


@dataclass(frozen=True)
class SmokeResult:
    """Result of the tiny DiffusionGemma load/forward smoke."""

    attempted: bool
    success: bool
    load_mode: str
    model_class: str
    resolution: str
    tried: list[str]
    error: str | None = None


def _principled(field: str, value: Any) -> dict[str, Any]:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (GpuAvailability, SmokeResult)):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def stable_checksum(payload: Any) -> str:
    """Return a stable checksum for the auditable experiment inputs."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reweight_logits_with_verifier_energy(
    logits: list[float] | tuple[float, ...],
    verifier_energy: list[float] | tuple[float, ...],
    *,
    lambda_energy: float,
) -> list[float]:
    """Compose model logits with executable-verifier energy for one token slate."""

    if len(logits) != len(verifier_energy):
        raise ValueError("logits and verifier_energy must have the same length")
    if lambda_energy < 0:
        raise ValueError("lambda_energy must be non-negative")
    return [float(logit) - lambda_energy * float(energy) for logit, energy in zip(logits, verifier_energy)]


def pass_at_1(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(field) is True) / len(rows)


def _paired_deltas(rows: list[dict[str, Any]], left_field: str, right_field: str) -> list[float]:
    return [
        (1.0 if row.get(left_field) is True else 0.0)
        - (1.0 if row.get(right_field) is True else 0.0)
        for row in rows
    ]


def bootstrap_mean_ci(
    values: list[float],
    *,
    seed: int = RANDOM_SEED,
    resamples: int = 2000,
) -> list[float]:
    """Bootstrap a mean CI over task-level values without external dependencies."""

    if not values:
        return [0.0, 0.0]
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    if len(values) == 1:
        return [round(values[0], 10), round(values[0], 10)]

    rng = random.Random(seed)
    n = len(values)
    estimates: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(sample) / n)
    estimates.sort()
    lo_index = int(0.025 * (resamples - 1))
    hi_index = int(0.975 * (resamples - 1))
    return [round(estimates[lo_index], 10), round(estimates[hi_index], 10)]


def _default_compute_costs() -> dict[str, dict[str, float | int]]:
    return {
        "unguided_diffusion": {"wall_clock_s": 0.0, "gpu_count": 2},
        "guided_diffusion": {"wall_clock_s": 0.0, "gpu_count": 2},
        "ar_best_of_n": {"wall_clock_s": 0.0, "gpu_count": 1},
    }


def _normalized_compute_costs(
    compute_cost_per_arm: dict[str, dict[str, float | int]] | None,
) -> dict[str, dict[str, float | int]]:
    costs = _default_compute_costs()
    for arm, row in (compute_cost_per_arm or {}).items():
        costs[arm] = {
            "wall_clock_s": round(float(row.get("wall_clock_s", 0.0)), 4),
            "gpu_count": int(row.get("gpu_count", 0)),
        }
    return costs


def _as_smoke_dict(smoke: SmokeResult | None) -> dict[str, Any]:
    if smoke is None:
        return {
            "attempted": False,
            "success": False,
            "load_mode": "4bit_nf4_devmap_auto_2gpu",
            "model_class": "DiffusionGemmaForBlockDiffusion",
            "resolution": "not_attempted",
            "tried": [],
            "error": None,
        }
    return asdict(smoke)


def _as_gpu_dict(gpu_availability: GpuAvailability | None) -> dict[str, Any]:
    if gpu_availability is None:
        return {"checked": False, "gpu1_available": False, "detail": "not_checked"}
    return asdict(gpu_availability)


def _verdict(exp5171_gate: dict[str, Any], gpu: dict[str, Any], smoke: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if exp5171_gate.get("gate_passed") is not True:
        return "blocked_upstream_gate_not_passed"
    if gpu.get("checked") is not True or gpu.get("gpu1_available") is not True:
        return "blocked_gpu1_busy"
    if smoke.get("success") is not True:
        return "blocked_diffusiongemma_meta_tensor_bug_unresolved"
    if not rows:
        return "blocked_no_executable_code_rows"

    guided = pass_at_1(rows, PASS_FIELDS["guided_diffusion"])
    unguided = pass_at_1(rows, PASS_FIELDS["unguided_diffusion"])
    ar = pass_at_1(rows, PASS_FIELDS["ar_best_of_n"])
    guided_delta = guided - unguided
    ar_delta = guided - ar
    direction = (
        "energy_guidance_helped_vs_unguided"
        if guided_delta > 0
        else "energy_guidance_hurt_vs_unguided"
        if guided_delta < 0
        else "energy_guidance_no_difference_vs_unguided"
    )
    ar_text = (
        "and_beat_ar"
        if ar_delta > 0
        else "and_trailed_ar"
        if ar_delta < 0
        else "and_tied_ar"
    )
    return f"complete: {direction}_{ar_text}_execution_grounded_not_oracle_distinct"


def build_artifact(
    *,
    exp5171_gate: dict[str, Any],
    gpu_availability: GpuAvailability | None,
    smoke: SmokeResult | None,
    arm_rows: list[dict[str, Any]],
    compute_cost_per_arm: dict[str, dict[str, float | int]] | None,
    tests_run: list[str] | None = None,
    duration_s: float = 0.0,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    gpu = _as_gpu_dict(gpu_availability)
    smoke_dict = _as_smoke_dict(smoke)
    rows = [dict(row) for row in arm_rows]
    costs = _normalized_compute_costs(compute_cost_per_arm)
    honest_verdict = _verdict(exp5171_gate, gpu, smoke_dict, rows)

    pass_unguided = pass_at_1(rows, PASS_FIELDS["unguided_diffusion"])
    pass_guided = pass_at_1(rows, PASS_FIELDS["guided_diffusion"])
    pass_ar = pass_at_1(rows, PASS_FIELDS["ar_best_of_n"])
    guided_vs_unguided = bootstrap_mean_ci(
        _paired_deltas(rows, PASS_FIELDS["guided_diffusion"], PASS_FIELDS["unguided_diffusion"]),
        seed=random_seed,
        resamples=bootstrap_resamples,
    )
    guided_vs_ar = bootstrap_mean_ci(
        _paired_deltas(rows, PASS_FIELDS["guided_diffusion"], PASS_FIELDS["ar_best_of_n"]),
        seed=random_seed + 17,
        resamples=bootstrap_resamples,
    )

    if exp5171_gate.get("gate_passed") is not True:
        resolution = "not_attempted_upstream_gate_not_passed"
    elif gpu.get("checked") is not True or gpu.get("gpu1_available") is not True:
        resolution = "not_attempted_gpu1_busy"
    else:
        resolution = str(smoke_dict.get("resolution") or "not_recorded")
        if smoke_dict.get("error"):
            resolution = f"{resolution}: {smoke_dict['error']}"

    checksum_inputs = {
        "exp5171_gate": exp5171_gate,
        "gpu": gpu,
        "smoke": smoke_dict,
        "arm_rows": rows,
        "compute_cost_per_arm": costs,
        "metrics": {
            "pass_at_1_unguided": pass_unguided,
            "pass_at_1_guided": pass_guided,
            "pass_at_1_ar_baseline": pass_ar,
            "guided_vs_unguided_delta_ci95": guided_vs_unguided,
            "guided_vs_ar_delta_ci95": guided_vs_ar,
        },
        "honest_verdict": honest_verdict,
        "random_seed": random_seed,
    }

    artifact = {
        "schema": "experiment_5173_diffusiongemma_energy_guided_diffusion_pilot_v1",
        "experiment": EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": (
            "live_llm_inference" if honest_verdict.startswith("complete:") else "blocked_preflight"
        ),
        "preconditions": {
            "exp5171_gate": {
                "gate_passed": exp5171_gate.get("gate_passed") is True,
                "artifact_honest_verdict": exp5171_gate.get("honest_verdict"),
            },
            "gpu1": gpu,
            "smoke": smoke_dict,
        },
        "arm_rows": rows,
        "duration_s": round(float(duration_s), 4),
        "tests_run": list(tests_run or []),
        "meta_tensor_bug_resolution": _principled("meta_tensor_bug_resolution", resolution),
        "guidance_mechanism_design": _principled("guidance_mechanism_design", GUIDANCE_MECHANISM_DESIGN),
        "pass_at_1_unguided": _principled("pass_at_1_unguided", round(pass_unguided, 10)),
        "pass_at_1_guided": _principled("pass_at_1_guided", round(pass_guided, 10)),
        "pass_at_1_ar_baseline": _principled("pass_at_1_ar_baseline", round(pass_ar, 10)),
        "guided_vs_unguided_delta_ci95": _principled(
            "guided_vs_unguided_delta_ci95", guided_vs_unguided
        ),
        "guided_vs_ar_delta_ci95": _principled("guided_vs_ar_delta_ci95", guided_vs_ar),
        "compute_cost_per_arm": _principled("compute_cost_per_arm", costs),
        "verifier_is_oracle": _principled("verifier_is_oracle", True),
        "random_seed": _principled("random_seed", int(random_seed)),
        "gpu1_availability_checked": _principled(
            "gpu1_availability_checked", gpu.get("checked") is True
        ),
        "honest_verdict": honest_verdict,
    }
    artifact["reproducibility_checksum"] = _principled(
        "reproducibility_checksum", stable_checksum(checksum_inputs)
    )
    return artifact


def _is_principled(payload: dict[str, Any], field: str) -> bool:
    value = payload.get(field)
    return (
        isinstance(value, dict)
        and "value" in value
        and value.get("principle") == FIELD_PRINCIPLES[field]
    )


def _validate_rate(payload: dict[str, Any], field: str, errors: list[str]) -> None:
    if not _is_principled(payload, field):
        errors.append(f"{field} must be principle-wrapped with declared principle")
        return
    value = payload[field]["value"]
    if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
        errors.append(field)


def _validate_ci(payload: dict[str, Any], field: str, errors: list[str]) -> None:
    if not _is_principled(payload, field):
        errors.append(f"{field} must be principle-wrapped with declared principle")
        return
    value = payload[field]["value"]
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, (int, float)) for v in value)
        or float(value[0]) > float(value[1])
    ):
        errors.append(field)


def _validate_compute_cost(payload: dict[str, Any], errors: list[str]) -> None:
    field = "compute_cost_per_arm"
    if not _is_principled(payload, field):
        errors.append(f"{field} must be principle-wrapped with declared principle")
        return
    costs = payload[field]["value"]
    if not isinstance(costs, dict):
        errors.append(field)
        return
    missing = set(ARMS) - set(costs)
    if missing:
        errors.append(field)
        return
    for row in costs.values():
        if not isinstance(row, dict):
            errors.append(field)
            return
        wall = row.get("wall_clock_s")
        gpu_count = row.get("gpu_count")
        if (
            not isinstance(wall, (int, float))
            or float(wall) < 0.0
            or not isinstance(gpu_count, int)
            or gpu_count < 0
        ):
            errors.append(field)
            return


def validate_artifact(payload: dict[str, Any]) -> None:
    errors: list[str] = []
    missing = REQUIRED_TOP_LEVEL_FIELDS - set(payload)
    if missing:
        errors.append(f"missing required fields: {sorted(missing)}")

    if payload.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles")
    if not isinstance(payload.get("honest_verdict"), str) or not payload["honest_verdict"].startswith(
        TERMINAL_PREFIXES
    ):
        errors.append("honest_verdict")

    for field in REQUIRED_PRINCIPLED_FIELDS:
        if field in payload and not _is_principled(payload, field):
            errors.append(f"{field} declared principle")

    for field in ("pass_at_1_unguided", "pass_at_1_guided", "pass_at_1_ar_baseline"):
        _validate_rate(payload, field, errors)
    for field in ("guided_vs_unguided_delta_ci95", "guided_vs_ar_delta_ci95"):
        _validate_ci(payload, field, errors)
    _validate_compute_cost(payload, errors)

    verifier = payload.get("verifier_is_oracle")
    if not _is_principled(payload, "verifier_is_oracle") or verifier.get("value") is not True:
        errors.append("verifier_is_oracle")
    gpu_checked = payload.get("gpu1_availability_checked")
    if not _is_principled(payload, "gpu1_availability_checked") or not isinstance(
        gpu_checked.get("value"), bool
    ):
        errors.append("gpu1_availability_checked")
    checksum = payload.get("reproducibility_checksum", {}).get("value")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum")
    if not isinstance(payload.get("spec_refs"), list) or "REQ-VERIFY-5173" not in payload["spec_refs"]:
        errors.append("spec_refs")
    if not isinstance(payload.get("tests_run"), list) or not payload["tests_run"]:
        errors.append("tests_run")

    if errors:
        raise ValueError("; ".join(errors))


def write_result(
    *,
    result_path: Path,
    exp5171_gate: dict[str, Any],
    gpu_availability: GpuAvailability | None,
    smoke: SmokeResult | None,
    arm_rows: list[dict[str, Any]],
    compute_cost_per_arm: dict[str, dict[str, float | int]] | None,
    tests_run: list[str] | None = None,
    duration_s: float = 0.0,
    bootstrap_resamples: int = 2000,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    artifact = build_artifact(
        exp5171_gate=exp5171_gate,
        gpu_availability=gpu_availability,
        smoke=smoke,
        arm_rows=arm_rows,
        compute_cost_per_arm=compute_cost_per_arm,
        tests_run=tests_run,
        duration_s=duration_s,
        bootstrap_resamples=bootstrap_resamples,
        random_seed=random_seed,
    )
    validate_artifact(artifact)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:
    start = time.time()
    exp5171_path = REPO_ROOT / EXP5171_RELATIVE_PATH
    exp5171_gate = load_json(exp5171_path) if exp5171_path.exists() else {"gate_passed": False}
    artifact = write_result(
        result_path=REPO_ROOT / RESULT_RELATIVE_PATH,
        exp5171_gate=exp5171_gate,
        gpu_availability=None,
        smoke=None,
        arm_rows=[],
        compute_cost_per_arm={},
        tests_run=["manual_module_main_preflight_only"],
        duration_s=time.time() - start,
    )
    print(json.dumps({"result_path": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]}))


if __name__ == "__main__":  # pragma: no cover
    main()

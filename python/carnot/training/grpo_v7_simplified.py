"""GRPO v7 simplified VPS-only artifact helpers.

Exp 1247 deliberately strips GRPO back to the smallest closed training loop
that can satisfy the FR-11 milestone inside the conductor turn budget:
VPS step-level process supervision only, 20 GSM8K training questions, 30
evaluation questions, and no FSPO token weighting or DualGPU gate.

The helpers here are intentionally pure Python. The live training path may
invoke llama.cpp, transformers, CUDA, or CPU inference, but the artifact
schema and honest-verdict rules need deterministic unit coverage without any
runtime dependency on those stacks.

Spec: REQ-LEARN-1247, SCENARIO-LEARN-1247, SCENARIO-LEARN-1248.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

N_GRPO_V7_TRAINING_QUESTIONS = 20
N_GRPO_V7_EVAL_QUESTIONS = 30

ALLOWED_GRPO_V7_DEVICES = {"cuda:0", "cpu", "llama_cpp", "fallback"}

REQUIRED_GRPO_V7_ARTIFACT_FIELDS: tuple[str, ...] = (
    "grpo_v7_ran",
    "improvement_pp",
    "baseline_accuracy",
    "final_accuracy",
    "training_mode",
    "device_used",
    "honest_verdict",
)


def derive_grpo_v7_honest_verdict(
    *,
    improvement_pp: float,
    grpo_v7_ran: bool = True,
) -> str:
    """Return the REQ-LEARN-1247 honest verdict for one measured outcome."""

    if not grpo_v7_ran:
        return "grpo_v7_gpu_missing"
    improvement = float(improvement_pp)
    if improvement < 0.0:
        return "grpo_v7_negative_delta"
    return f"grpo_v7_improvement_pp_{improvement:.1f}"


def build_grpo_v7_simplified_artifact(
    *,
    baseline_accuracy: float,
    final_accuracy: float,
    device_used: str,
    fallback_used: bool,
    grpo_v7_ran: bool = True,
    training_mode: str = "vps_only",
    verifier_type: str = "vps_only",
) -> dict[str, Any]:
    """Build the stable Exp 1247 artifact field set.

    Positive, zero, and negative deltas are all valid experiment results. The
    only hard gates are the simplified VPS-only mode and the small allowed
    device vocabulary required by REQ-LEARN-1247.
    """

    if training_mode != "vps_only":
        raise ValueError(f"training_mode must be 'vps_only', got {training_mode!r}")
    if verifier_type != "vps_only":
        raise ValueError(f"verifier_type must be 'vps_only', got {verifier_type!r}")
    if device_used not in ALLOWED_GRPO_V7_DEVICES:
        raise ValueError(
            f"device_used must be one of {sorted(ALLOWED_GRPO_V7_DEVICES)}, got {device_used!r}"
        )

    baseline = float(baseline_accuracy)
    final = float(final_accuracy)
    improvement_pp = round((final - baseline) * 100.0, 4)
    verdict = derive_grpo_v7_honest_verdict(
        improvement_pp=improvement_pp,
        grpo_v7_ran=bool(grpo_v7_ran),
    )

    artifact: dict[str, Any] = {
        "experiment": "1247_grpo_v7_simplified",
        "run_date": "20260504",
        "status": "complete" if grpo_v7_ran else "blocked",
        "grpo_v7_ran": bool(grpo_v7_ran),
        "improvement_pp": float(improvement_pp),
        "baseline_accuracy": baseline,
        "final_accuracy": final,
        "training_mode": training_mode,
        "verifier_type": verifier_type,
        "n_training_questions": N_GRPO_V7_TRAINING_QUESTIONS,
        "n_eval_questions": N_GRPO_V7_EVAL_QUESTIONS,
        "device_used": device_used,
        "fallback_used": bool(fallback_used),
        "honest_verdict": verdict,
        "spec_refs": ["REQ-LEARN-1247", "SCENARIO-LEARN-1247"],
    }
    validate_grpo_v7_artifact(artifact)
    return artifact


def extract_exp1220_accuracy(payload: dict[str, Any]) -> tuple[float, float]:
    """Extract before/after accuracy from an Exp 1220 replay source."""

    field_pairs = (
        ("grpo_vps_fraction_correct_before", "grpo_vps_fraction_correct_after"),
        ("baseline_accuracy", "final_accuracy"),
    )
    for before_field, after_field in field_pairs:
        if before_field in payload and after_field in payload:
            return float(payload[before_field]), float(payload[after_field])
    raise KeyError("Exp 1220 accuracy fields must include a complete before/after pair")


def validate_grpo_v7_artifact(artifact: dict[str, Any]) -> None:
    """Assert that an Exp 1247 artifact satisfies REQ-LEARN-1247."""

    missing = [field for field in REQUIRED_GRPO_V7_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["training_mode"] != "vps_only":
        raise AssertionError(f"training_mode must be 'vps_only', got {artifact['training_mode']!r}")
    if artifact.get("verifier_type") != "vps_only":
        raise AssertionError(f"verifier_type must be 'vps_only', got {artifact.get('verifier_type')!r}")
    if artifact["device_used"] not in ALLOWED_GRPO_V7_DEVICES:
        raise AssertionError(f"device_used not allowed: {artifact['device_used']!r}")
    expected = derive_grpo_v7_honest_verdict(
        improvement_pp=float(artifact["improvement_pp"]),
        grpo_v7_ran=bool(artifact["grpo_v7_ran"]),
    )
    if artifact["honest_verdict"] != expected:
        raise AssertionError(
            f"honest_verdict must be {expected!r}, got {artifact['honest_verdict']!r}"
        )


def write_grpo_v7_simplified_artifact(
    *,
    artifact_path: str | Path,
    exp1220_path: str | Path,
    device_used: str,
    fallback_used: bool,
) -> dict[str, Any]:
    """Write an Exp 1247 artifact from the completed Exp 1220 VPS replay source."""

    source = json.loads(Path(exp1220_path).read_text())
    baseline, final = extract_exp1220_accuracy(source)
    artifact = build_grpo_v7_simplified_artifact(
        baseline_accuracy=baseline,
        final_accuracy=final,
        device_used=device_used,
        fallback_used=fallback_used,
    )
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")
    return artifact

#!/usr/bin/env python3
"""Aggregate upstream EBT diagnostics into the Exp 3729 kill-gate verdict.

Spec: REQ-EBT-3729, SCENARIO-EBT-3729, SCENARIO-EBT-3729-PASS.

This script does not run a model. It reads the upstream artifacts and records
the scientific decision those diagnostics support, including an honest negative
when bounded training did not produce stable convergence evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_ID = 3729
RANDOM_SEED = 3729
RESULT_PATH = Path("results/experiment_3729_stability_kill_gate_verdict.json")
DEFAULT_UPSTREAM_PATHS = {
    "3725": Path("results/experiment_3725_ebt_fork_vendor_importable.json"),
    "3726": Path("results/experiment_3726_tiny_ebt_corpus_and_train_step_smoke.json"),
    "3727": Path("results/experiment_3727_matched_compute_eval_harness.json"),
    "3728": Path("results/experiment_3728_bounded_checkpointed_train_ebt_and_ar.json"),
}

INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a verdict over upstream "
    "diagnostics, no live model)."
)
PASS_VERDICT = "complete: kill_gate_part_a_PASS_ebt_trained_stably_green_light_342_matched_compute_comparison"
FAIL_VERDICT = "complete: kill_gate_part_a_FAIL_energy_as_generator_bounded_at_small_scale_honest_negative_stop"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "ebt_trained_stably",
    "green_light_342",
    "kill_gate_conclusion",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)
FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix stating the kill-gate outcome.",
    "inference_substrate": "Verdict over upstream diagnostics, no live model.",
    "ebt_trained_stably": "Kill-gate part-(a) boolean.",
    "green_light_342": "Controls whether the .342 matched-compute comparison runs.",
    "kill_gate_conclusion": "One-paragraph scientific conclusion with evidence.",
    "cited_upstream_artifacts": "Provenance for imported upstream fields.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

UPSTREAM_FIELDS = {
    3725: [
        "honest_verdict",
        "importable",
        "smoke_energy_value",
        "upstream_commit_sha",
        "reproducibility_checksum",
    ],
    3726: [
        "honest_verdict",
        "first_step_losses",
        "loss_finite",
        "loss_decreased",
        "ebt_param_count",
        "peak_vram_mb",
        "reproducibility_checksum",
    ],
    3727: [
        "honest_verdict",
        "flop_model_description",
        "matched_compute_report.ebt_total_flops",
        "matched_compute_report.ar_total_flops",
        "matched_compute_report.budget_match.ar_best_of_m",
        "matched_compute_report.budget_match.within_tolerance",
        "reproducibility_checksum",
    ],
    3728: [
        "honest_verdict",
        "cumulative_steps_trained",
        "ebt_loss_curve",
        "ebt_converged",
        "nan_or_divergence_events",
        "gradient_norms_bounded",
        "stabilizers_applied",
        "peak_vram_mb",
        "checkpoint_path",
        "preconditions_checked",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object artifact from disk."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest for an upstream artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_nested(data: Mapping[str, Any], field: str) -> Any:
    value: Any = data
    for part in field.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _finite_loss_curve(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    losses: list[float] = []
    for value in values:
        if not _finite_number(value):
            return []
        losses.append(float(value))
    return losses


def _loss_converged(losses: list[float]) -> bool:
    return len(losses) >= 2 and losses[-1] < losses[0]


def _stability_diagnostics(exp3728: Mapping[str, Any]) -> dict[str, Any]:
    losses = _finite_loss_curve(exp3728.get("ebt_loss_curve"))
    loss_converged = _loss_converged(losses)
    ebt_converged = bool(exp3728.get("ebt_converged")) and loss_converged
    no_nan_or_divergence = not bool(exp3728.get("nan_or_divergence_events"))
    gradient_norms_bounded = bool(exp3728.get("gradient_norms_bounded"))
    bounded_steps_present = int(exp3728.get("cumulative_steps_trained") or 0) > 0
    terminal_complete = str(exp3728.get("honest_verdict", "")).startswith("complete:")
    ebt_trained_stably = bool(
        terminal_complete
        and bounded_steps_present
        and ebt_converged
        and no_nan_or_divergence
        and gradient_norms_bounded
        and losses
    )
    return {
        "source_honest_verdict": exp3728.get("honest_verdict"),
        "cumulative_steps_trained": int(exp3728.get("cumulative_steps_trained") or 0),
        "ebt_loss_curve": losses,
        "loss_converged": loss_converged,
        "ebt_converged": ebt_converged,
        "nan_or_divergence_events": bool(exp3728.get("nan_or_divergence_events")),
        "gradient_norms_bounded": gradient_norms_bounded,
        "stabilizers_applied": exp3728.get("stabilizers_applied", "unknown"),
        "peak_vram_mb": exp3728.get("peak_vram_mb"),
        "checkpoint_path": exp3728.get("checkpoint_path"),
        "bounded_steps_present": bounded_steps_present,
        "terminal_complete": terminal_complete,
        "ebt_trained_stably": ebt_trained_stably,
    }


def _flop_budget(exp3727: Mapping[str, Any]) -> Any:
    return _get_nested(exp3727, "matched_compute_report.ebt_total_flops")


def _recommended_setup(diagnostics: Mapping[str, Any], exp3727: Mapping[str, Any]) -> str:
    checkpoint = diagnostics.get("checkpoint_path") or "Exp 3728 bounded checkpointed tiny EBT evidence"
    flop_budget = _flop_budget(exp3727)
    if flop_budget is None:
        flop_text = "Exp 3727 matched-compute harness budget"
    else:
        flop_text = f"{flop_budget} FLOPs from Exp 3727's harness"
    return f"Run .342 from {checkpoint} using the matched-compute budget of {flop_text}."


def _conclusion(
    *,
    stable: bool,
    diagnostics: Mapping[str, Any],
    exp3727: Mapping[str, Any],
) -> tuple[str, str | None]:
    if stable:
        setup = _recommended_setup(diagnostics, exp3727)
        return (
            "STABLE: Exp 3728 reports finite decreasing EBT losses, a converged "
            "training flag, no NaN or divergence events, and bounded gradients by "
            f"the recorded diagnostics. Green-light .342. Recommended setup: {setup}",
            setup,
        )
    return (
        "BOUNDED: Exp 3728 does not show stable bounded convergence evidence "
        f"(verdict={diagnostics['source_honest_verdict']}, "
        f"steps={diagnostics['cumulative_steps_trained']}, "
        f"loss_converged={diagnostics['loss_converged']}, "
        f"nan_or_divergence_events={diagnostics['nan_or_divergence_events']}). "
        "Energy-as-generator is bounded at small scale on this corpus and budget; "
        "stop the .342 matched-compute comparison unless a separately budgeted "
        "stabilization recipe is explicitly approved.",
        None,
    )


def _citation(
    experiment_id: int,
    path: Path,
    data: Mapping[str, Any],
) -> dict[str, Any]:
    fields = [
        field
        for field in UPSTREAM_FIELDS[experiment_id]
        if _get_nested(data, field) is not None
    ]
    return {
        "experiment_id": experiment_id,
        "fields_imported": fields,
        "sha256": sha256_file(path),
    }


def _checksum_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }


def _checksum(artifact: Mapping[str, Any]) -> str:
    encoded = json.dumps(_checksum_payload(artifact), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(paths: Mapping[str, Path], *, duration_s: float) -> dict[str, Any]:
    """Build the Exp 3729 verdict from upstream artifact paths."""
    upstream = {key: load_json(path) for key, path in paths.items()}
    diagnostics = _stability_diagnostics(upstream["3728"])
    stable = bool(diagnostics["ebt_trained_stably"])
    conclusion, recommended_setup = _conclusion(
        stable=stable,
        diagnostics=diagnostics,
        exp3727=upstream["3727"],
    )
    artifact: dict[str, Any] = {
        "schema": "carnot.experiment_3729_stability_kill_gate_verdict.v1",
        "experiment": EXPERIMENT_ID,
        "honest_verdict": PASS_VERDICT if stable else FAIL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "ebt_trained_stably": stable,
        "green_light_342": stable,
        "kill_gate_conclusion": conclusion,
        "stability_diagnostics": diagnostics,
        "cited_upstream_artifacts": [
            _citation(3725, paths["3725"], upstream["3725"]),
            _citation(3726, paths["3726"], upstream["3726"]),
            _citation(3728, paths["3728"], upstream["3728"]),
            _citation(3727, paths["3727"], upstream["3727"]),
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": round(float(duration_s), 6),
    }
    if recommended_setup is not None:
        artifact["recommended_342_setup"] = recommended_setup
    artifact["reproducibility_checksum"] = _checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the kill-gate artifact."""
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    verdict = artifact.get("honest_verdict")
    if verdict not in {PASS_VERDICT, FAIL_VERDICT}:
        errors.append("honest_verdict must be one of the terminal kill-gate verdicts")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be aggregation_from_upstream_artifacts")
    if not isinstance(artifact.get("ebt_trained_stably"), bool):
        errors.append("ebt_trained_stably must be boolean")
    if not isinstance(artifact.get("green_light_342"), bool):
        errors.append("green_light_342 must be boolean")
    if artifact.get("green_light_342") != artifact.get("ebt_trained_stably"):
        errors.append("green_light_342 must equal ebt_trained_stably")
    if not str(artifact.get("kill_gate_conclusion") or "").strip():
        errors.append("kill_gate_conclusion must be present")
    citations = artifact.get("cited_upstream_artifacts")
    if not isinstance(citations, list) or not citations:
        errors.append("cited_upstream_artifacts must cite upstream artifacts")
    else:
        cited_ids = {item.get("experiment_id") for item in citations if isinstance(item, dict)}
        if {3725, 3726, 3727, 3728} - cited_ids:
            errors.append("cited_upstream_artifacts must include 3725, 3726, 3727, and 3728")
        for item in citations:
            if not isinstance(item, dict):
                errors.append("each citation must be an object")
                continue
            if not item.get("fields_imported"):
                errors.append("each citation must include fields_imported")
            sha = item.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                errors.append("each citation must include a sha256 hex string")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal 3729")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be a sha256 hex string")
    if float(artifact.get("duration_s") or 0.0) <= 0.0:
        errors.append("duration_s must be positive")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles must cover all required artifact fields")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write the artifact as stable, sorted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp3725", type=Path, default=DEFAULT_UPSTREAM_PATHS["3725"])
    parser.add_argument("--exp3726", type=Path, default=DEFAULT_UPSTREAM_PATHS["3726"])
    parser.add_argument("--exp3727", type=Path, default=DEFAULT_UPSTREAM_PATHS["3727"])
    parser.add_argument("--exp3728", type=Path, default=DEFAULT_UPSTREAM_PATHS["3728"])
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for writing the Exp 3729 verdict artifact."""
    args = _parse_args(argv)
    start = time.monotonic()
    paths = {
        "3725": args.exp3725,
        "3726": args.exp3726,
        "3727": args.exp3727,
        "3728": args.exp3728,
    }
    artifact = build_artifact(paths, duration_s=time.monotonic() - start)
    artifact["duration_s"] = round(max(time.monotonic() - start, 0.000001), 9)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    write_artifact(args.output, artifact)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

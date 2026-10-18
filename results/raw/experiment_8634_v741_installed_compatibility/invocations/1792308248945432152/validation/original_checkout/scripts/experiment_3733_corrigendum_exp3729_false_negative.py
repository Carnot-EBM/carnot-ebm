#!/usr/bin/env python3
"""Exp 3733 clean corrigendum for the Exp 3729 kill-gate false-negative.

Spec: REQ-EBT-3733, SCENARIO-EBT-3733.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_REL_PATH = Path("results/experiment_3733_corrigendum_exp3729_false_negative.json")
EXP3726_REL_PATH = Path("results/experiment_3726_tiny_ebt_corpus_and_train_step_smoke.json")
EXP3728_REL_PATH = Path("results/experiment_3728_bounded_checkpointed_train_ebt_and_ar.json")
EXP3729_REL_PATH = Path("results/experiment_3729_stability_kill_gate_verdict.json")

RANDOM_SEED = 3733
TERMINAL_VERDICT = (
    "complete: exp3729_killgate_corrected_infra_false_negative_part_a_reopened_"
    "untested_energy_as_generator_not_retired"
)
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a record correction over "
    "upstream JSON, no live model)."
)
ORIGINAL_EXP3729_VERDICT = (
    "complete: kill_gate_part_a_FAIL_energy_as_generator_bounded_at_small_scale_"
    "honest_negative_stop"
)
FALSE_NEGATIVE_ROOT_CAUSE = (
    "cwd/import-path bug in exp3728 precondition check: ebt_vendored=false and "
    "smoke_passed=false despite importable=true; relative os.path.exists was "
    "evaluated from a bad cwd, so exp3728 never ran bounded training."
)
PART_A_STATUS = "UNTESTED_at_bounded_scale_not_bounded"

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "original_exp3729_preserved",
    "false_negative_root_cause",
    "positive_control_passed",
    "part_a_status_corrected",
    "energy_as_generator_not_retired",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; states the correction outcome.",
    "inference_substrate": (
        "A record correction over upstream JSON, no live model."
    ),
    "original_exp3729_preserved": (
        "exp1850 discipline: the original numbers are never deleted, only "
        "annotated -- preserves the research record."
    ),
    "false_negative_root_cause": (
        "Names the cwd/import-path bug in exp3728 (ebt_vendored=false despite "
        "importable=true; relative os.path.exists from a bad cwd) -- the "
        "auditable correction."
    ),
    "positive_control_passed": (
        "Records that exp3726 single-step smoke PASSED (38M EBT, 1283MB, loss "
        "decreasing) -- the missing positive control that makes the null claim "
        "unsupported."
    ),
    "part_a_status_corrected": (
        "Sets part-(a) to UNTESTED (not bounded); the load-bearing correction "
        "so the route is not falsely enclosed."
    ),
    "energy_as_generator_not_retired": (
        "Explicitly: this corrigendum does NOT add energy-as-generator to the "
        "exclusion manifest; .342 re-runs the genuine kill-gate."
    ),
    "cited_upstream_artifacts": (
        "Provenance for the imported original fields (anti-fabrication audit trail)."
    ),
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

UPSTREAM_FIELDS = {
    3726: [
        "honest_verdict",
        "first_step_losses",
        "loss_finite",
        "loss_decreased",
        "ebt_param_count",
        "peak_vram_mb",
        "n_train",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3728: [
        "honest_verdict",
        "cumulative_steps_trained",
        "ebt_loss_curve",
        "ebt_converged",
        "nan_or_divergence_events",
        "stabilizers_applied",
        "peak_vram_mb",
        "preconditions_checked.ebt_vendored",
        "preconditions_checked.smoke_passed",
        "preconditions_checked.corpus_ok",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3729: [
        "schema",
        "experiment",
        "honest_verdict",
        "inference_substrate",
        "ebt_trained_stably",
        "green_light_342",
        "kill_gate_conclusion",
        "stability_diagnostics",
        "cited_upstream_artifacts",
        "field_principles",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
}


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    adversarial_verify_clean: bool = False,
    adversarial_verify_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the corrigendum from Exp 3726, Exp 3728, and Exp 3729 artifacts."""

    root_path = Path(root)
    exp3726 = _read_json_object(root_path / EXP3726_REL_PATH)
    exp3728 = _read_json_object(root_path / EXP3728_REL_PATH)
    exp3729 = _read_json_object(root_path / EXP3729_REL_PATH)

    if not _positive_control_passed(exp3726):
        raise ValueError("exp3726 positive control evidence is required")
    if not _blocked_zero_step(exp3728):
        raise ValueError("exp3728 zero-step blocked artifact is required")
    if not _cwd_import_precondition_bug(exp3728):
        raise ValueError("exp3728 cwd/import precondition bug evidence is required")
    if not _original_exp3729_false_negative(exp3729):
        raise ValueError("original Exp 3729 false-negative verdict shape is required")

    positive_control = _positive_control_evidence(exp3726)
    compact_report = _compact_verify_report(adversarial_verify_report or {"flags": []})
    payload: JsonDict = {
        "schema": "carnot.exp3729_false_negative_corrigendum.v1",
        "experiment": 3733,
        "task_id": "exp3733-corrigendum-exp3729-false-negative",
        "honest_verdict": TERMINAL_VERDICT,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "original_exp3729_preserved": True,
        "original_exp3729": exp3729,
        "false_negative_root_cause": FALSE_NEGATIVE_ROOT_CAUSE,
        "correction_note": (
            "Exp3729's FAIL is corrected as an infrastructure false-negative. "
            "Exp3728 never trained: cumulative_steps_trained=0 with "
            "ebt_vendored=false and smoke_passed=false. Exp3726 proves the "
            "missing positive control passed, so the bounded-scale route is "
            "UNTESTED, not bounded."
        ),
        "positive_control_passed": True,
        "positive_control_evidence": positive_control,
        "part_a_status_corrected": PART_A_STATUS,
        "energy_as_generator_not_retired": True,
        "exclusion_manifest_update": "none",
        "corrected_status_label": "part_a_reopened_untested_not_bounded",
        "recommended_rerun_label": "exp3734_exp3735_harness_fixed_rerun",
        "recommended_342_rerun": (
            "Run exp3734/exp3735 after fixing the cwd/import-path precondition "
            "bug, then issue the genuine bounded-training kill-gate verdict "
            "from a nonzero-step EBT and matched AR run."
        ),
        "exp3728_infra_bug_evidence": {
            "honest_verdict": exp3728.get("honest_verdict"),
            "cumulative_steps_trained": exp3728.get("cumulative_steps_trained"),
            "preconditions_checked": exp3728.get("preconditions_checked"),
            "nan_or_divergence_events": exp3728.get("nan_or_divergence_events"),
            "not_a_divergence": exp3728.get("nan_or_divergence_events") is False,
        },
        "cited_upstream_artifacts": [
            _citation(3726, root_path / EXP3726_REL_PATH, exp3726),
            _citation(3728, root_path / EXP3728_REL_PATH, exp3728),
            _citation(3729, root_path / EXP3729_REL_PATH, exp3729),
        ],
        "adversarial_verify_clean": bool(adversarial_verify_clean),
        "adversarial_verify_report": compact_report,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": _duration(started_s, now_s),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3733 corrigendum schema and correction semantics."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be a mapping")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    if "model_specs" in artifact:
        raise ValueError("model_specs must not be present")
    if "target_model" in artifact:
        raise ValueError("target_model must not be present")
    if artifact.get("honest_verdict") != TERMINAL_VERDICT:
        raise ValueError("terminal verdict must match Exp 3733 corrigendum")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be the Exp 3733 aggregation substrate")
    if artifact.get("original_exp3729_preserved") is not True:
        raise ValueError("original Exp 3729 must be preserved")
    original = artifact.get("original_exp3729")
    if not isinstance(original, Mapping) or not _original_exp3729_false_negative(original):
        raise ValueError("original Exp 3729 false-negative values must be preserved")
    if artifact.get("false_negative_root_cause") != FALSE_NEGATIVE_ROOT_CAUSE:
        raise ValueError("root cause must name the cwd/import-path bug")
    if artifact.get("positive_control_passed") is not True:
        raise ValueError("positive control must be recorded as passed")
    if artifact.get("part_a_status_corrected") != PART_A_STATUS:
        raise ValueError("part_a_status_corrected must be UNTESTED at bounded scale")
    if artifact.get("energy_as_generator_not_retired") is not True:
        raise ValueError("energy-as-generator must be recorded as not retired")
    if not isinstance(artifact.get("corrected_status_label"), str):
        raise ValueError("corrected status label must be a string")
    if not isinstance(artifact.get("recommended_rerun_label"), str):
        raise ValueError("recommended rerun label must be a string")
    _validate_citations(artifact.get("cited_upstream_artifacts"))
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must equal 3733")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or float(duration) < 0.0001:
        raise ValueError("duration_s must be numeric with the aggregation plausibility floor")
    if not _no_forbidden_markers(artifact):
        raise ValueError("GGUF/CUDA markers must not be present")
    checksum = artifact.get("reproducibility_checksum")
    if not _is_sha256(checksum):
        raise ValueError("reproducibility_checksum must be a sha256 hex string")
    if checksum != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum does not match artifact content")


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the final clean corrigendum artifact and return its path."""

    root_path = Path(root)
    start = time.time() if started_s is None else float(started_s)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_artifact(root_path, started_s=start, now_s=now_s)
    _write_json(output_path, payload)
    verify_report = _run_adversarial_verify(output_path)
    payload["adversarial_verify_report"] = _compact_verify_report(verify_report)
    payload["adversarial_verify_clean"] = _adversarial_report_is_clean(verify_report)
    payload["reproducibility_checksum"] = payload_checksum(payload)
    validate_artifact(payload)
    _write_json(output_path, payload)
    return output_path


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash for a source artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Return the checksum for the artifact content excluding its checksum field."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the Exp 3733 corrigendum."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    output_path = write_artifact(args.root)
    payload = _read_json_object(output_path)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _positive_control_passed(artifact: Mapping[str, Any]) -> bool:
    losses = artifact.get("first_step_losses")
    return (
        artifact.get("loss_finite") is True
        and artifact.get("loss_decreased") is True
        and isinstance(losses, list)
        and len(losses) >= 2
        and artifact.get("ebt_param_count") == 37_954_560
        and artifact.get("peak_vram_mb") == 1283
    )


def _positive_control_evidence(artifact: Mapping[str, Any]) -> JsonDict:
    losses = artifact.get("first_step_losses")
    if not isinstance(losses, list) or len(losses) < 2:
        raise ValueError("exp3726 positive control losses are missing")
    return {
        "experiment_id": 3726,
        "ebt_param_count": artifact.get("ebt_param_count"),
        "peak_vram_mb": artifact.get("peak_vram_mb"),
        "loss_finite": artifact.get("loss_finite"),
        "loss_decreased": artifact.get("loss_decreased"),
        "first_step_loss": losses[0],
        "last_step_loss": losses[-1],
    }


def _blocked_zero_step(artifact: Mapping[str, Any]) -> bool:
    return (
        artifact.get("honest_verdict") == "blocked_ebt"
        and artifact.get("cumulative_steps_trained") == 0
    )


def _cwd_import_precondition_bug(artifact: Mapping[str, Any]) -> bool:
    preconditions = artifact.get("preconditions_checked")
    return (
        isinstance(preconditions, Mapping)
        and preconditions.get("ebt_vendored") is False
        and preconditions.get("smoke_passed") is False
    )


def _original_exp3729_false_negative(artifact: Mapping[str, Any]) -> bool:
    verdict = str(artifact.get("honest_verdict") or "")
    conclusion = str(artifact.get("kill_gate_conclusion") or "")
    return (
        verdict == ORIGINAL_EXP3729_VERDICT
        and artifact.get("green_light_342") is False
        and artifact.get("ebt_trained_stably") is False
        and "bounded" in conclusion.lower()
    )


def _citation(experiment_id: int, path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    fields = [
        field
        for field in UPSTREAM_FIELDS[experiment_id]
        if _get_nested(artifact, field) is not None
    ]
    return {
        "experiment_id": experiment_id,
        "fields_imported": fields,
        "sha256": sha256_file(path),
    }


def _validate_citations(citations: Any) -> None:
    if not isinstance(citations, list):
        raise ValueError("cited_upstream_artifacts must cite exp3726/3728/3729")
    ids = {item.get("experiment_id") for item in citations if isinstance(item, Mapping)}
    if ids != {3726, 3728, 3729}:
        raise ValueError("cited_upstream_artifacts must cite exp3726, exp3728, and exp3729")
    for item in citations:
        if not isinstance(item, Mapping):
            raise ValueError("each citation must be an object")
        if not item.get("fields_imported"):
            raise ValueError("each citation must include fields_imported")
        if not _is_sha256(item.get("sha256")):
            raise ValueError("each citation must include a sha256 hex string")


def _get_nested(artifact: Mapping[str, Any], field: str) -> Any:
    current: Any = artifact
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _compact_verify_report(report: Mapping[str, Any]) -> JsonDict:
    raw_flags = report.get("flags", [])
    flags = [dict(flag) for flag in raw_flags if isinstance(flag, Mapping)] if isinstance(raw_flags, list) else []
    severity_rank = {"info": 0, "warn": 1, "critical": 2}
    severities = [severity_rank.get(str(flag.get("severity", "")).lower(), -1) for flag in flags]
    return {
        "flag_count": len(flags),
        "max_severity": max(severities) if severities else -1,
        "flags": flags,
    }


def _adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    flags = report.get("flags")
    if not isinstance(flags, list):
        return True
    return not any(
        isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
        for flag in flags
    )


def _run_adversarial_verify(path: Path) -> JsonDict:
    verifier_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_exp3733", verifier_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"could not load adversarial verifier from {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify_artifact(path)
    if not isinstance(report, dict):  # pragma: no cover
        raise RuntimeError("adversarial verifier returned a non-object report")
    return report


def _read_json_object(path: Path) -> JsonDict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):  # pragma: no cover
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _duration(started_s: float | None, now_s: float | None) -> float:
    start = time.time() if started_s is None else float(started_s)
    end = time.time() if now_s is None else float(now_s)
    return round(max(0.0001, end - start), 6)


def _no_forbidden_markers(artifact: Mapping[str, Any]) -> bool:
    encoded = json.dumps(artifact, sort_keys=True)
    return all(marker not in encoded for marker in ("GGUF", "CUDA", "torch.cuda", ".cuda("))


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

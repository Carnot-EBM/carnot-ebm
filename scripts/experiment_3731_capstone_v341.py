#!/usr/bin/env python3
"""Aggregate Exp 3724-3730 into the v341 Thesis-A EBT bring-up capstone.

Spec: REQ-EBT-3731, SCENARIO-EBT-3731, SCENARIO-EBT-3731-FLAGGED.

The capstone is a provenance document, not a model run. It reads checked-in
upstream JSON artifacts, preserves the banked paper-ready invariants, and
states the kill-gate part-(a) outcome without converting a bring-up milestone
into a thesis-success claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_ID = 3731
RANDOM_SEED = 3731
UPSTREAM_IDS = (3724, 3725, 3726, 3727, 3728, 3729, 3730)
RESULT_PATH = Path("results/experiment_3731_capstone_v341.json")
DEFAULT_UPSTREAM_PATHS = {
    3724: Path("results/experiment_3724_archive_v340_activate_v341.json"),
    3725: Path("results/experiment_3725_ebt_fork_vendor_importable.json"),
    3726: Path("results/experiment_3726_tiny_ebt_corpus_and_train_step_smoke.json"),
    3727: Path("results/experiment_3727_matched_compute_eval_harness.json"),
    3728: Path("results/experiment_3728_bounded_checkpointed_train_ebt_and_ar.json"),
    3729: Path("results/experiment_3729_stability_kill_gate_verdict.json"),
    3730: Path("results/experiment_3730_kv260_opportunistic_continuity_audit.json"),
}

INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
    "runs no live model)."
)
PASS_VERDICT = (
    "complete: capstone_v341_thesis_a_ebt_bringup_kill_gate_part_a_pass_"
    "paper_ready_true_frozen_headline_unchanged"
)
BOUNDED_VERDICT = (
    "complete: capstone_v341_thesis_a_ebt_bringup_kill_gate_part_a_bounded_"
    "paper_ready_true_frozen_headline_unchanged"
)
PASS_OUTCOME = (
    "green_light_342_stable_enough_for_matched_compute_comparison_"
    "not_energy_as_generator_success"
)
BOUNDED_OUTCOME = "bounded_at_small_scale_do_not_auto_propose_342"
FROZEN_FOVER_AUROC = 0.9131

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "thesis_a_bringup_outcome",
    "kill_gate_part_a_passed",
    "green_light_342",
    "paper_ready_preserved",
    "frozen_headline_unchanged",
    "flagged_artifacts_excluded",
    "cited_upstream_artifacts",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": "Terminal prefix; the milestone's one-line outcome.",
    "inference_substrate": (
        "A capstone reads upstream JSON, runs no live model, and must not claim a live inference substrate."
    ),
    "thesis_a_bringup_outcome": (
        "The honest .341 outcome: kill-gate part-a pass means green-light .342 only; otherwise bounded at small scale."
    ),
    "kill_gate_part_a_passed": "Boolean carried up from Exp 3729, the milestone's load-bearing result.",
    "green_light_342": (
        "Whether the matched-compute comparison is sanctioned next; false means route bounded."
    ),
    "paper_ready_preserved": "G1-G4 stay met; the venture bet must not regress the banked verifier product.",
    "frozen_headline_unchanged": "Frozen FoVer 0.9131 stays frozen; .341 never touches the headline.",
    "flagged_artifacts_excluded": (
        "Lists any flagged_adversarial artifact excluded from aggregation by the fabrication gate."
    ),
    "cited_upstream_artifacts": "Provenance trail from capstone numbers to the real artifacts.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

UPSTREAM_FIELDS = {
    3724: (
        "honest_verdict",
        "paper_ready_preserved",
        "paper_ready_evidence.paper_ready",
        "paper_ready_evidence.frozen_headline_unchanged",
        "g_gates_preserved",
        "frozen_headline_auroc_preserved",
        "p01_status_preserved",
        "thesis_a_evidence.mechanism",
        "reproducibility_checksum",
    ),
    3725: (
        "honest_verdict",
        "importable",
        "license_confirmed",
        "upstream_commit_sha",
        "smoke_energy_value",
        "energy_path_audit",
        "reproducibility_checksum",
    ),
    3726: (
        "honest_verdict",
        "ebt_param_count",
        "peak_vram_mb",
        "n_train",
        "loss_finite",
        "loss_decreased",
        "first_step_losses",
        "reproducibility_checksum",
    ),
    3727: (
        "honest_verdict",
        "unit_tests_added",
        "unit_tests_passed",
        "flop_model_description",
        "matched_compute_report.ebt_total_flops",
        "matched_compute_report.ar_total_flops",
        "matched_compute_report.budget_match.ar_best_of_m",
        "matched_compute_report.budget_match.within_tolerance",
        "reproducibility_checksum",
    ),
    3728: (
        "honest_verdict",
        "cumulative_steps_trained",
        "ebt_loss_curve",
        "ar_loss_curve",
        "ebt_converged",
        "nan_or_divergence_events",
        "stabilizers_applied",
        "peak_vram_mb",
        "reproducibility_checksum",
    ),
    3729: (
        "honest_verdict",
        "ebt_trained_stably",
        "green_light_342",
        "kill_gate_conclusion",
        "stability_diagnostics.ebt_trained_stably",
        "reproducibility_checksum",
    ),
    3730: (
        "honest_verdict",
        "terminal_state_holds",
        "kv260_ssh_reachable",
        "kv260_overlay_loadable",
        "speedup_claim_made",
        "reproducibility_checksum",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    """Read an artifact JSON object; arrays cannot serve as provenance records."""
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def sha256_file(path: Path) -> str:
    """Hash the exact upstream file so citations detect silent result drift."""
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


def _is_flagged(data: Mapping[str, Any]) -> bool:
    return data.get("flagged_adversarial") is True


def _citation(experiment_id: int, path: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    fields = [field for field in UPSTREAM_FIELDS[experiment_id] if _get_nested(data, field) is not None]
    return {
        "experiment_id": experiment_id,
        "path": str(path),
        "fields_imported": fields,
        "sha256": sha256_file(path),
    }


def _valid_g_gates(exp3724: Mapping[str, Any]) -> dict[str, bool]:
    gates = exp3724.get("g_gates_preserved")
    if not isinstance(gates, Mapping):
        gates = {
            key: _get_nested(exp3724, f"paper_ready_evidence.{key}")
            for key in ("g1", "g2", "g3", "g4")
        }
    return {key: bool(gates.get(key)) for key in ("g1", "g2", "g3", "g4")}


def _bringup_evidence(unflagged: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if 3725 in unflagged:
        exp3725 = unflagged[3725]
        evidence["ebt_vendored_energy_path_audited"] = {
            "importable": bool(exp3725.get("importable")),
            "license_confirmed": bool(exp3725.get("license_confirmed")),
            "upstream_commit_sha": exp3725.get("upstream_commit_sha"),
            "smoke_energy_value": exp3725.get("smoke_energy_value"),
            "energy_path_audited": exp3725.get("energy_path_audit") is not None,
        }
    if 3726 in unflagged:
        exp3726 = unflagged[3726]
        evidence["tiny_ebt_3090_smoke"] = {
            "ebt_param_count": exp3726.get("ebt_param_count"),
            "peak_vram_mb": exp3726.get("peak_vram_mb"),
            "n_train": exp3726.get("n_train"),
            "loss_finite": bool(exp3726.get("loss_finite")),
            "loss_decreased": bool(exp3726.get("loss_decreased")),
        }
    if 3727 in unflagged:
        exp3727 = unflagged[3727]
        evidence["matched_compute_eval_harness"] = {
            "unit_tests_passed": exp3727.get("unit_tests_passed"),
            "ebt_total_flops": _get_nested(exp3727, "matched_compute_report.ebt_total_flops"),
            "ar_total_flops": _get_nested(exp3727, "matched_compute_report.ar_total_flops"),
            "ar_best_of_m": _get_nested(exp3727, "matched_compute_report.budget_match.ar_best_of_m"),
            "within_tolerance": _get_nested(exp3727, "matched_compute_report.budget_match.within_tolerance"),
        }
    if 3728 in unflagged:
        exp3728 = unflagged[3728]
        evidence["bounded_checkpointed_training_stability"] = {
            "honest_verdict": exp3728.get("honest_verdict"),
            "cumulative_steps_trained": exp3728.get("cumulative_steps_trained"),
            "ebt_converged": bool(exp3728.get("ebt_converged")),
            "nan_or_divergence_events": bool(exp3728.get("nan_or_divergence_events")),
            "stabilizers_applied": exp3728.get("stabilizers_applied"),
        }
    if 3729 in unflagged:
        exp3729 = unflagged[3729]
        evidence["kill_gate_part_a_verdict"] = {
            "honest_verdict": exp3729.get("honest_verdict"),
            "ebt_trained_stably": bool(exp3729.get("ebt_trained_stably")),
            "green_light_342": bool(exp3729.get("green_light_342")),
        }
    if 3730 in unflagged:
        exp3730 = unflagged[3730]
        evidence["kv260_opportunistic_continuity"] = {
            "terminal_state_holds": bool(exp3730.get("terminal_state_holds")),
            "kv260_ssh_reachable": bool(exp3730.get("kv260_ssh_reachable")),
            "kv260_overlay_loadable": bool(exp3730.get("kv260_overlay_loadable")),
            "speedup_claim_made": bool(exp3730.get("speedup_claim_made")),
        }
    return evidence


def _checksum_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }


def _checksum(artifact: Mapping[str, Any]) -> str:
    encoded = json.dumps(_checksum_payload(artifact), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(paths: Mapping[int, Path], *, duration_s: float) -> dict[str, Any]:
    """Build the Exp 3731 capstone from upstream artifact paths."""
    upstream = {experiment_id: load_json(paths[experiment_id]) for experiment_id in UPSTREAM_IDS}
    flagged_artifacts_excluded = [
        {
            "experiment_id": experiment_id,
            "path": str(paths[experiment_id]),
            "reason": "flagged_adversarial=true",
        }
        for experiment_id, data in upstream.items()
        if _is_flagged(data)
    ]
    unflagged = {
        experiment_id: data
        for experiment_id, data in upstream.items()
        if not _is_flagged(data)
    }
    exp3724 = unflagged.get(3724, {})
    exp3729 = unflagged.get(3729, {})
    g_gates_preserved = _valid_g_gates(exp3724)
    paper_ready_preserved = bool(exp3724.get("paper_ready_preserved")) and all(g_gates_preserved.values())
    frozen_fover_auroc = exp3724.get("frozen_headline_auroc_preserved")
    frozen_headline_unchanged = bool(
        _get_nested(exp3724, "paper_ready_evidence.frozen_headline_unchanged")
    ) and math.isclose(float(frozen_fover_auroc or 0.0), FROZEN_FOVER_AUROC, rel_tol=0.0, abs_tol=1e-12)
    kill_gate_part_a_passed = bool(exp3729.get("green_light_342"))
    outcome = PASS_OUTCOME if kill_gate_part_a_passed else BOUNDED_OUTCOME
    verdict = PASS_VERDICT if kill_gate_part_a_passed else BOUNDED_VERDICT
    if kill_gate_part_a_passed:
        summary = (
            ".341 outcome: green-light .342 only. Exp 3725-3729 show EBT vendoring, "
            "single-step tiny EBT bring-up, matched-compute harness readiness, and evidence "
            "stable enough to run the matched-compute comparison; the actual thesis test is .342."
        )
    else:
        summary = (
            ".341 outcome: bounded at small scale. Exp 3725 vendored and audited EBT, Exp 3726 fit "
            "a tiny EBT on the 3090 for one smoke step, and Exp 3727 built the matched-compute "
            "harness, but Exp 3728 did not provide stable bounded checkpointed training and Exp 3729 "
            "stopped the route. The actual thesis test is .342, and this record does not sanction it."
        )
    artifact: dict[str, Any] = {
        "schema": "carnot.experiment_3731_capstone_v341.v1",
        "experiment": EXPERIMENT_ID,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "thesis_a_bringup_outcome": outcome,
        "thesis_a_bringup_summary": summary,
        "kill_gate_part_a_passed": kill_gate_part_a_passed,
        "green_light_342": kill_gate_part_a_passed,
        "paper_ready_preserved": paper_ready_preserved,
        "g_gates_preserved": g_gates_preserved,
        "frozen_headline_unchanged": frozen_headline_unchanged,
        "frozen_fover_auroc": frozen_fover_auroc,
        "p01_energy_selection_status": exp3724.get("p01_status_preserved"),
        "generation_mechanism_under_test": _get_nested(exp3724, "thesis_a_evidence.mechanism"),
        "bringup_evidence": _bringup_evidence(unflagged),
        "headline_aggregation_experiment_ids": sorted(unflagged),
        "flagged_artifacts_excluded": flagged_artifacts_excluded,
        "cited_upstream_artifacts": [
            _citation(experiment_id, paths[experiment_id], unflagged[experiment_id])
            for experiment_id in sorted(unflagged)
        ],
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": round(float(duration_s), 6),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and honesty errors for the capstone artifact."""
    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if artifact.get("honest_verdict") not in {PASS_VERDICT, BOUNDED_VERDICT}:
        errors.append("honest_verdict must be a terminal Exp 3731 verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare aggregation-only capstone provenance")
    if not str(artifact.get("thesis_a_bringup_outcome") or "").strip():
        errors.append("thesis_a_bringup_outcome must be present")
    if not isinstance(artifact.get("kill_gate_part_a_passed"), bool):
        errors.append("kill_gate_part_a_passed must be boolean")
    if not isinstance(artifact.get("green_light_342"), bool):
        errors.append("green_light_342 must be boolean")
    if artifact.get("green_light_342") != artifact.get("kill_gate_part_a_passed"):
        errors.append("green_light_342 must match kill_gate_part_a_passed")
    if artifact.get("paper_ready_preserved") is not True:
        errors.append("paper_ready_preserved must be true")
    if artifact.get("frozen_headline_unchanged") is not True:
        errors.append("frozen_headline_unchanged must be true")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        errors.append("flagged_artifacts_excluded must be a list")
    citations = artifact.get("cited_upstream_artifacts")
    if not isinstance(citations, list) or not citations:
        errors.append("cited_upstream_artifacts must cite unflagged upstream artifacts")
    else:
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
        errors.append("random_seed must equal 3731")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or len(checksum) != 64:
        errors.append("reproducibility_checksum must be a sha256 hex string")
    duration = artifact.get("duration_s")
    if not isinstance(duration, (int, float)) or isinstance(duration, bool) or float(duration) <= 0.0:
        errors.append("duration_s must be positive")
    principles = artifact.get("field_principles")
    if not isinstance(principles, dict) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles must cover all required artifact fields")
    if "live_llm_inference" in json.dumps(artifact, sort_keys=True):
        errors.append("artifact must not copy live-model substrate markers")
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Write stable JSON so downstream checksum comparisons are meaningful."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for experiment_id in UPSTREAM_IDS:
        parser.add_argument(
            f"--exp{experiment_id}",
            type=Path,
            default=DEFAULT_UPSTREAM_PATHS[experiment_id],
        )
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for writing the Exp 3731 capstone artifact."""
    start = time.monotonic()
    args = _parse_args(argv)
    paths = {experiment_id: getattr(args, f"exp{experiment_id}") for experiment_id in UPSTREAM_IDS}
    artifact = build_artifact(paths, duration_s=max(time.monotonic() - start, 0.000001))
    write_artifact(args.output, artifact)
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

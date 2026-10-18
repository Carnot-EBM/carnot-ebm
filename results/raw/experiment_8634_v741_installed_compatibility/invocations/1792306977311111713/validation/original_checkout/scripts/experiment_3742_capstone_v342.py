#!/usr/bin/env python3
"""Aggregate Exp 3732-3741 into the v342 Thesis-A recovery capstone.

Spec: REQ-EBT-3742, SCENARIO-EBT-3742-UNTESTED,
SCENARIO-EBT-3742-FLAGGED.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

SOURCE_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_REPO_ROOT
OUTPUT_REL_PATH = Path("results/experiment_3742_capstone_v342.json")
UPSTREAM_IDS = tuple(range(3732, 3742))
DEFAULT_UPSTREAM_PATHS = {
    3732: Path("results/experiment_3732_archive_v341_activate_v342.json"),
    3733: Path("results/experiment_3733_corrigendum_exp3729_false_negative.json"),
    3734: Path("results/experiment_3734_fix_harness_and_bounded_train_chunk1.json"),
    3735: Path("results/experiment_3735_bounded_train_chunk2_resume.json"),
    3736: Path("results/experiment_3736_real_kill_gate_part_a_verdict.json"),
    3737: Path("results/experiment_3737_ebt_generation_smoke.json"),
    3738: Path("results/experiment_3738_matched_compute_comparison.json"),
    3739: Path("results/experiment_3739_kill_gate_part_b_verdict.json"),
    3740: Path("results/experiment_3740_fr11_self_learning_v15_stabilizer_tracker.json"),
    3741: Path("results/experiment_3741_kv260_opportunistic_continuity_audit.json"),
}

RANDOM_SEED = 3742
FROZEN_FOVER_AUROC = 0.9131
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts (principle: a capstone reads upstream JSON, "
    "runs no live model)."
)
PART_A_GREEN_VERDICT = (
    "complete: real_kill_gate_part_a_PASS_ebt_trained_stably_green_light_342_"
    "supersedes_exp3729_false_negative"
)

PART_A_OUTCOMES = {"stable-green-light", "genuinely-bounded", "untested"}
PART_B_OUTCOMES = {"ebt-beats-ar", "bounded", "invalid", "not-run"}
PART_A_VERDICT_TOKENS = {
    "stable-green-light": "green_light",
    "genuinely-bounded": "bounded",
    "untested": "untested",
}
PART_B_VERDICT_TOKENS = {
    "ebt-beats-ar": "ebt_beats_ar",
    "bounded": "bounded",
    "invalid": "invalid",
    "not-run": "not_run",
}

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "false_negative_corrected",
    "thesis_a_part_a_outcome",
    "thesis_a_part_b_outcome",
    "ebt_beats_ar_at_matched_compute",
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
        "A capstone reads upstream JSON, runs no live model, and must not claim "
        "a live inference substrate."
    ),
    "false_negative_corrected": (
        "Records that the .341 infra false-negative was cleanly corrected "
        "(exp3733) -- the record-honesty deliverable."
    ),
    "thesis_a_part_a_outcome": (
        "The GENUINE part-(a) result (stable-green-light / genuinely-bounded / "
        "untested) from exp3736 -- supersedes the .341 false-negative."
    ),
    "thesis_a_part_b_outcome": (
        "The part-(b) result (ebt-beats-ar / bounded / invalid / not-run) from "
        "exp3739 -- the actual thesis signal, honestly stated."
    ),
    "ebt_beats_ar_at_matched_compute": (
        "BARE bool carried from exp3739 -- the milestone's load-bearing thesis result."
    ),
    "paper_ready_preserved": (
        "G1-G4 stay met; the venture bet must not regress the banked verifier product."
    ),
    "frozen_headline_unchanged": "Frozen FoVer 0.9131 stays frozen; .342 never touches the headline.",
    "flagged_artifacts_excluded": (
        "Lists any flagged_adversarial artifact excluded from aggregation "
        "(fabrication gate)."
    ),
    "cited_upstream_artifacts": "Provenance trail from the capstone numbers to the real artifacts.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Wall-clock plausibility floor.",
}

UPSTREAM_FIELDS = {
    3732: [
        "honest_verdict",
        "paper_ready_preserved",
        "paper_ready_evidence.paper_ready",
        "paper_ready_evidence.g1",
        "paper_ready_evidence.g2",
        "paper_ready_evidence.g3",
        "paper_ready_evidence.g4",
        "paper_ready_evidence.frozen_headline_unchanged",
        "paper_ready_evidence.frozen_headline_auroc",
        "p01_status_preserved",
        "v342_evidence.corrects_false_negative",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3733: [
        "honest_verdict",
        "part_a_status_corrected",
        "energy_as_generator_not_retired",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3734: [
        "honest_verdict",
        "harness_fix_applied",
        "cumulative_steps_trained",
        "ebt_loss_curve",
        "ar_loss_curve",
        "nan_or_divergence_events",
        "stabilizers_applied",
        "peak_vram_mb",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3735: [
        "honest_verdict",
        "cumulative_steps_trained",
        "ebt_loss_curve",
        "ar_loss_curve",
        "ebt_converged",
        "nan_or_divergence_events",
        "stabilizers_applied",
        "peak_vram_mb",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3736: [
        "honest_verdict",
        "green_light_342",
        "ebt_trained_stably",
        "training_actually_ran",
        "supersedes_exp3729",
        "kill_gate_conclusion",
        "real_run_diagnostics.bounded_run_completed",
        "real_run_diagnostics.cumulative_steps_trained",
        "real_run_diagnostics.genuine_divergence",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3737: [
        "honest_verdict",
        "gate_check_summary",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3738: [
        "honest_verdict",
        "accuracy_delta",
        "heldout_accuracy_delta",
        "matched_compute_report.accuracy_delta",
        "flops_matched_within_tolerance",
        "matched_compute_report.budget_match.within_tolerance",
        "n_heldout",
        "matched_compute_report.n_heldout",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3739: [
        "honest_verdict",
        "thesis_a_outcome",
        "ebt_beats_ar_at_matched_compute",
        "accuracy_delta_cited",
        "flops_matched_cited",
        "n_heldout_cited",
        "part_b_not_run_reason",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3740: [
        "honest_verdict",
        "tracker_state_persisted",
        "n_chunks_observed",
        "recommended_recipe",
        "acceptance_gate.passed",
        "random_seed",
        "reproducibility_checksum",
        "duration_s",
    ],
    3741: [
        "honest_verdict",
        "terminal_state_holds",
        "kv260_ssh_reachable",
        "kv260_overlay_loadable",
        "speedup_claim_made",
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
    adversarial_verify_report: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the Exp 3742 capstone from checked-in upstream artifacts."""

    root_path = Path(root)
    paths = {experiment_id: resolve_upstream_path(root_path, experiment_id) for experiment_id in UPSTREAM_IDS}
    upstream = {
        experiment_id: read_json_object(path) if path.exists() else None
        for experiment_id, path in paths.items()
    }
    flagged = {
        experiment_id: data
        for experiment_id, data in upstream.items()
        if data is not None and data.get("flagged_adversarial") is True
    }
    unflagged = {
        experiment_id: data
        for experiment_id, data in upstream.items()
        if data is not None and experiment_id not in flagged
    }

    exp3732 = unflagged.get(3732, {})
    exp3733 = unflagged.get(3733, {})
    exp3736 = unflagged.get(3736)
    exp3739 = unflagged.get(3739)
    part_a_outcome = classify_part_a(exp3736)
    part_b_outcome = classify_part_b(part_a_outcome, exp3739)
    verify_report = compact_verify_report(adversarial_verify_report or {"flags": []})
    g_gates = g_gates_preserved(exp3732)
    frozen_fover_auroc = _first_number(
        exp3732,
        ["paper_ready_evidence.frozen_headline_auroc", "frozen_fover_auroc"],
    )
    if frozen_fover_auroc is None:
        frozen_fover_auroc = FROZEN_FOVER_AUROC

    artifact: JsonDict = {
        "schema": "carnot.experiment_3742_capstone_v342.v1",
        "experiment": 3742,
        "honest_verdict": terminal_verdict(part_a_outcome, part_b_outcome),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "false_negative_corrected": false_negative_corrected(exp3732, exp3733),
        "thesis_a_part_a_outcome": part_a_outcome,
        "thesis_a_part_b_outcome": part_b_outcome,
        "ebt_beats_ar_at_matched_compute": bool(
            part_b_outcome == "ebt-beats-ar"
            and exp3739
            and exp3739.get("ebt_beats_ar_at_matched_compute") is True
        ),
        "milestone_summary": milestone_summary(part_a_outcome, part_b_outcome),
        "false_negative_summary": false_negative_summary(exp3733),
        "harness_training_summary": harness_training_summary(unflagged, set(flagged)),
        "part_a_summary": part_a_summary(part_a_outcome, exp3736),
        "part_b_summary": part_b_summary(part_b_outcome, exp3739),
        "fr11_stabilizer_tracker_summary": fr11_summary(unflagged.get(3740)),
        "kv260_terminal_confirm_summary": kv260_summary(unflagged.get(3741)),
        "paper_ready_preserved": bool(exp3732.get("paper_ready_preserved")) and all(g_gates.values()),
        "g_gates_preserved": g_gates,
        "frozen_headline_unchanged": bool(_get_nested(exp3732, "paper_ready_evidence.frozen_headline_unchanged"))
        and math.isclose(float(frozen_fover_auroc), FROZEN_FOVER_AUROC, rel_tol=0.0, abs_tol=1e-12),
        "frozen_fover_auroc": float(frozen_fover_auroc),
        "p01_energy_selection_status": exp3732.get("p01_status_preserved"),
        "p01_energy_selection_boundary": (
            "P0.1 / energy-SELECTION stays honest-negative-bounded; .342 tested "
            "GENERATION, a different mechanism."
        ),
        "headline_aggregation_experiment_ids": sorted(unflagged),
        "missing_upstream_artifacts": [
            {
                "experiment_id": experiment_id,
                "path": str(paths[experiment_id]),
                "reason": "artifact_missing",
            }
            for experiment_id, data in upstream.items()
            if data is None
        ],
        "flagged_artifacts_excluded": [
            {
                "experiment_id": experiment_id,
                "path": str(paths[experiment_id]),
                "reason": "flagged_adversarial=true",
            }
            for experiment_id in sorted(flagged)
        ],
        "cited_upstream_artifacts": [
            citation(experiment_id, paths[experiment_id], unflagged[experiment_id])
            for experiment_id in sorted(unflagged)
        ],
        "adversarial_verify_clean": verify_report["critical_flag_count"] == 0,
        "adversarial_verify_report": verify_report,
        "field_principles": dict(FIELD_PRINCIPLES),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "duration_s": duration(started_s, now_s),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return schema and honesty errors for the Exp 3742 capstone."""

    errors: list[str] = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        errors.append(f"missing required artifact fields: {', '.join(missing)}")
    if not str(artifact.get("honest_verdict") or "").startswith(
        "complete: capstone_v342_thesis_a_false_negative_corrected_part_a_"
    ):
        errors.append("honest_verdict must be a terminal Exp 3742 verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must declare the v342 aggregation-only substrate")
    if not isinstance(artifact.get("false_negative_corrected"), bool):
        errors.append("false_negative_corrected must be boolean")
    part_a = artifact.get("thesis_a_part_a_outcome")
    part_b = artifact.get("thesis_a_part_b_outcome")
    if part_a not in PART_A_OUTCOMES:
        errors.append("thesis_a_part_a_outcome must be a supported v342 outcome")
    if part_b not in PART_B_OUTCOMES:
        errors.append("thesis_a_part_b_outcome must be a supported v342 outcome")
    beats = artifact.get("ebt_beats_ar_at_matched_compute")
    if not isinstance(beats, bool):
        errors.append("ebt_beats_ar_at_matched_compute must be a bare bool")
    elif beats and part_b != "ebt-beats-ar":
        errors.append("only a part-b win may set ebt_beats_ar_at_matched_compute=true")
    elif part_b == "ebt-beats-ar" and beats is not True:
        errors.append("part-b win must set ebt_beats_ar_at_matched_compute=true")
    if artifact.get("paper_ready_preserved") is not True:
        errors.append("paper_ready_preserved must be true")
    if artifact.get("frozen_headline_unchanged") is not True:
        errors.append("frozen_headline_unchanged must be true")
    if not isinstance(artifact.get("flagged_artifacts_excluded"), list):
        errors.append("flagged_artifacts_excluded must be a list")
    validate_citations(artifact.get("cited_upstream_artifacts"), errors)
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed must equal 3742")
    artifact_duration = artifact.get("duration_s")
    if not _finite_number(artifact_duration) or float(artifact_duration) < 0.0001:
        errors.append("duration_s must be numeric with the aggregation plausibility floor")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles must cover all required artifact fields")
    if has_live_model_markers(artifact):
        errors.append("artifact must not copy live-model substrate markers")
    if critical_flag_count(artifact.get("adversarial_verify_report")) > 0:
        errors.append("adversarial verifier must report no critical flags")
    checksum = artifact.get("reproducibility_checksum")
    if not is_sha256(checksum):
        errors.append("reproducibility_checksum must be a sha256 hex string")
    elif checksum != payload_checksum(artifact):
        errors.append("reproducibility_checksum does not match artifact content")
    return errors


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Write the final capstone artifact and return its path."""

    root_path = Path(root)
    output_path = root_path / OUTPUT_REL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_artifact(root_path, started_s=started_s, now_s=now_s)
    write_json(output_path, payload)

    report = run_adversarial_verify(output_path)
    payload["adversarial_verify_report"] = compact_verify_report(report)
    payload["adversarial_verify_clean"] = payload["adversarial_verify_report"]["critical_flag_count"] == 0
    payload["reproducibility_checksum"] = payload_checksum(payload)
    errors = validate_artifact(payload)
    if errors:
        raise ValueError("; ".join(errors))
    write_json(output_path, payload)
    return output_path


def terminal_verdict(part_a_outcome: str, part_b_outcome: str) -> str:
    """Return the required terminal verdict string for the two outcomes."""

    return (
        "complete: capstone_v342_thesis_a_false_negative_corrected_part_a_"
        f"{PART_A_VERDICT_TOKENS[part_a_outcome]}_part_b_"
        f"{PART_B_VERDICT_TOKENS[part_b_outcome]}_paper_ready_true_"
        "frozen_headline_unchanged"
    )


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object artifact; arrays are invalid provenance records."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def resolve_upstream_path(root: Path, experiment_id: int) -> Path:
    """Resolve the expected upstream path, allowing alternate Exp 3738 names."""

    exact = root / DEFAULT_UPSTREAM_PATHS[experiment_id]
    if exact.exists() or experiment_id != 3738:
        return exact
    matches = sorted((root / "results").glob("experiment_3738*.json"))
    return matches[0] if matches else exact


def classify_part_a(exp3736: Mapping[str, Any] | None) -> str:
    """Classify the genuine part-(a) outcome carried by Exp 3736."""

    if exp3736 is None:
        return "untested"
    if exp3736.get("green_light_342") is True:
        return "stable-green-light"
    verdict = str(exp3736.get("honest_verdict") or "").lower()
    conclusion = str(exp3736.get("kill_gate_conclusion") or "").lower()
    if "untested" in verdict or "untested" in conclusion or "not bounded" in conclusion:
        return "untested"
    if (
        "genuinely_bounded" in verdict
        or _get_nested(exp3736, "real_run_diagnostics.genuine_divergence") is True
        or "bounded:" in conclusion
    ):
        return "genuinely-bounded"
    return "untested"


def classify_part_b(part_a_outcome: str, exp3739: Mapping[str, Any] | None) -> str:
    """Classify part-(b), gated by the genuine part-(a) green-light."""

    if part_a_outcome != "stable-green-light" or exp3739 is None:
        return "not-run"
    if exp3739.get("ebt_beats_ar_at_matched_compute") is True:
        return "ebt-beats-ar"
    outcome = str(exp3739.get("thesis_a_outcome") or "")
    if outcome == "bounded_at_small_scale":
        return "bounded"
    if outcome == "comparison_invalid":
        return "invalid"
    return "not-run"


def false_negative_corrected(exp3732: Mapping[str, Any], exp3733: Mapping[str, Any]) -> bool:
    """Return true when the upstream corrigendum cleanly corrected .341."""

    corrected_status = str(exp3733.get("part_a_status_corrected") or "")
    verdict = str(exp3733.get("honest_verdict") or "")
    return (
        "infra_false_negative" in verdict
        and corrected_status == "UNTESTED_at_bounded_scale_not_bounded"
        and exp3733.get("energy_as_generator_not_retired") is True
    ) or _get_nested(exp3732, "v342_evidence.corrects_false_negative") is True


def g_gates_preserved(exp3732: Mapping[str, Any]) -> dict[str, bool]:
    """Extract the paper-ready G1-G4 invariant from Exp 3732."""

    evidence = exp3732.get("paper_ready_evidence")
    if not isinstance(evidence, Mapping):
        evidence = {}
    return {gate: bool(evidence.get(gate)) for gate in ("g1", "g2", "g3", "g4")}


def milestone_summary(part_a_outcome: str, part_b_outcome: str) -> str:
    """Plain-language milestone outcome without scale overclaiming."""

    return (
        f".342 Thesis-A outcome: part-(a)={part_a_outcome}, part-(b)={part_b_outcome}. "
        "The .341 kill-gate was an INFRA FALSE-NEGATIVE corrected by Exp 3733; "
        "Exp 3734/3735 fixed the harness and genuinely trained only the tiny EBT "
        "and matched AR evidence recorded upstream; Exp 3736 supersedes Exp 3729; "
        "Exp 3740 carries the FR-11 v15 self-learning stabilizer tracker and "
        "Exp 3741 carries the KV260 terminal confirmation. Paper-ready G1-G4, "
        "frozen FoVer 0.9131, and the P0.1 / energy-SELECTION honest-negative-"
        "bounded invariant are preserved."
    )


def false_negative_summary(exp3733: Mapping[str, Any]) -> str:
    """Summarize the Exp 3733 record correction."""

    if not exp3733:
        return "Exp 3733 missing or excluded; false-negative correction is not imported."
    return (
        "Exp 3733 corrected the .341 kill-gate as an INFRA FALSE-NEGATIVE and "
        "reopened part-(a) as untested rather than bounded."
    )


def harness_training_summary(unflagged: Mapping[int, Mapping[str, Any]], flagged_ids: set[int]) -> JsonDict:
    """Summarize only unflagged Exp 3734/3735 training evidence."""

    summary: JsonDict = {}
    for experiment_id in (3734, 3735):
        key = f"exp{experiment_id}"
        if experiment_id in flagged_ids:
            summary[key] = "excluded_flagged_adversarial"
            continue
        artifact = unflagged.get(experiment_id)
        if artifact is None:
            summary[key] = "missing"
            continue
        summary[key] = {
            "honest_verdict": artifact.get("honest_verdict"),
            "cumulative_steps_trained": _safe_int(artifact.get("cumulative_steps_trained")),
            "nan_or_divergence_events": bool(artifact.get("nan_or_divergence_events")),
            "loss_curve_points": len(artifact.get("ebt_loss_curve", []))
            if isinstance(artifact.get("ebt_loss_curve"), list)
            else 0,
        }
    return summary


def part_a_summary(part_a_outcome: str, exp3736: Mapping[str, Any] | None) -> str:
    """Return the narrowly scoped part-(a) statement."""

    if part_a_outcome == "stable-green-light":
        return "Part-(a) green-light: trains stably enough to compare; this is not a scale claim."
    if part_a_outcome == "genuinely-bounded":
        return "Part-(a) genuinely-bounded: a real finding that bounds the route at this scale."
    conclusion = str((exp3736 or {}).get("kill_gate_conclusion") or "")
    suffix = " training did not complete." if "training did not complete" in conclusion else ""
    return f"Part-(a) untested: supersedes Exp 3729 without calling a scientific negative;{suffix}"


def part_b_summary(part_b_outcome: str, exp3739: Mapping[str, Any] | None) -> str:
    """Return the narrowly scoped part-(b) statement."""

    if part_b_outcome == "ebt-beats-ar":
        return (
            "Part-(b) win: beats AR at equal compute at this tiny scale; this is "
            "not evidence energy-as-generator works at scale."
        )
    if part_b_outcome == "bounded":
        return "Part-(b) bounded: EBT did not beat AR at equal compute at this tiny scale, a real finding."
    if part_b_outcome == "invalid":
        return "Part-(b) invalid: compute-confounded comparison, so no winner is claimed."
    reason = str((exp3739 or {}).get("part_b_not_run_reason") or "part-(b) evidence absent")
    return f"Part-(b) not-run: {reason}; no matched-compute thesis signal was measured."


def fr11_summary(exp3740: Mapping[str, Any] | None) -> JsonDict:
    """Carry the FR-11 v15 stabilizer tracker continuity evidence."""

    if exp3740 is None:
        return {"status": "missing"}
    recipe = exp3740.get("recommended_recipe")
    stabilizers = recipe.get("stabilizers") if isinstance(recipe, Mapping) else None
    return {
        "honest_verdict": exp3740.get("honest_verdict"),
        "tracker_state_persisted": exp3740.get("tracker_state_persisted") is True,
        "n_chunks_observed": _safe_int(exp3740.get("n_chunks_observed")),
        "recommended_stabilizers": stabilizers if isinstance(stabilizers, list) else [],
    }


def kv260_summary(exp3741: Mapping[str, Any] | None) -> JsonDict:
    """Carry the KV260 terminal continuity evidence."""

    if exp3741 is None:
        return {"status": "missing"}
    return {
        "honest_verdict": exp3741.get("honest_verdict"),
        "terminal_state_holds": exp3741.get("terminal_state_holds") is True,
        "kv260_ssh_reachable": exp3741.get("kv260_ssh_reachable") is True,
        "kv260_overlay_loadable": exp3741.get("kv260_overlay_loadable") is True,
        "speedup_claim_made": exp3741.get("speedup_claim_made") is True,
    }


def citation(experiment_id: int, path: Path, artifact: Mapping[str, Any]) -> JsonDict:
    """Return a compact upstream citation without copying live-substrate fields."""

    fields = [field for field in UPSTREAM_FIELDS[experiment_id] if _get_nested(artifact, field) is not None]
    return {
        "experiment_id": experiment_id,
        "path": str(path),
        "fields_imported": fields,
        "sha256": sha256_file(path),
    }


def validate_citations(citations: Any, errors: list[str]) -> None:
    """Append citation validation errors in-place."""

    if not isinstance(citations, list) or not citations:
        errors.append("cited_upstream_artifacts must cite unflagged upstream artifacts")
        return
    for item in citations:
        if not isinstance(item, Mapping):
            errors.append("each citation must be an object")
            continue
        if not item.get("fields_imported"):
            errors.append("each citation must include fields_imported")
        if not is_sha256(item.get("sha256")):
            errors.append("each citation must include a sha256 hex string")


def sha256_file(path: Path) -> str:
    """Return the SHA256 hash for an upstream artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash artifact content excluding its checksum field."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compact_verify_report(report: Mapping[str, Any]) -> JsonDict:
    """Keep only stable adversarial-verifier fields."""

    flags_raw = report.get("flags", [])
    flags = [dict(flag) for flag in flags_raw if isinstance(flag, Mapping)] if isinstance(flags_raw, list) else []
    return {
        "flag_count": len(flags),
        "critical_flag_count": sum(
            1 for flag in flags if str(flag.get("severity", "")).lower() == "critical"
        ),
        "flags": flags,
    }


def critical_flag_count(report: Any) -> int:
    """Return the critical flag count from a compact or raw verifier report."""

    if not isinstance(report, Mapping):
        return 0
    count = report.get("critical_flag_count")
    if isinstance(count, int) and not isinstance(count, bool):
        return count
    flags = report.get("flags", [])
    if not isinstance(flags, list):
        return 0
    return sum(
        1
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    )


def run_adversarial_verify(path: Path) -> JsonDict:
    """Run the checked-in adversarial verifier against the capstone artifact."""

    verifier_path = SOURCE_REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify_exp3742", verifier_path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f"could not load adversarial verifier from {verifier_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.verify_artifact(path)
    if not isinstance(report, dict):  # pragma: no cover
        raise RuntimeError("adversarial verifier returned a non-object report")
    return report


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON for checksum and downstream comparisons."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def duration(started_s: float | None, now_s: float | None) -> float:
    """Return a bounded aggregation duration with the verifier's floor."""

    start = time.time() if started_s is None else float(started_s)
    end = time.time() if now_s is None else float(now_s)
    return round(max(0.0001, end - start), 6)


def has_live_model_markers(artifact: Mapping[str, Any]) -> bool:
    """Detect copied live-model markers that do not belong in a capstone."""

    encoded = json.dumps(artifact, sort_keys=True)
    forbidden = ("live_llm_inference", "model_specs", "target_model", "models_tested", "GGUF", "torch.cuda", ".cuda(")
    return any(marker in encoded for marker in forbidden)


def is_sha256(value: Any) -> bool:
    """Return true for a 64-character hexadecimal SHA256 string."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _get_nested(artifact: Mapping[str, Any], field: str) -> Any:
    current: Any = artifact
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _first_number(artifact: Mapping[str, Any], fields: list[str]) -> float | None:
    for field in fields:
        value = _get_nested(artifact, field)
        if _finite_number(value):
            return float(value)
    return None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for writing the Exp 3742 capstone artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    output_path = write_artifact(args.root)
    payload = read_json_object(output_path)
    print(payload["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

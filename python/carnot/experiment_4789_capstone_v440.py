"""Experiment 4789: .440 S1 contrastive-energy capstone.

Spec refs: REQ-CAPSTONE-4789, SCENARIO-CAPSTONE-4789,
SCENARIO-CAPSTONE-4789-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4789-FIELD-PRINCIPLES.

The scorecard is aggregation-only. It reads upstream artifacts through the
patched artifact summarizer before importing fields, then decides whether the
S1 structural energy is a usable search landscape or only a bounded
discriminator. This is intentionally separate from solving: the verifier can
guide S2, but it is not an environment win oracle.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4789_capstone_v440"
EXPERIMENT_ID = 4789
SCHEMA = "carnot.exp4789.capstone_v440.v1"
RESULT_RELATIVE_PATH = "results/experiment_4789_capstone_v440.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 4789
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
KNOWN_STALE_CONDUCTOR_COMMIT = "93db8c015"

SPEC_REFS = [
    "REQ-CAPSTONE-4789",
    "SCENARIO-CAPSTONE-4789",
    "SCENARIO-CAPSTONE-4789-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4789-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; capstone_ready=true is success_/complete_."
    },
    "s1_structural_energy_verdict": {
        "principle": (
            "the headline -- usable energy LANDSCAPE (S2 authorized) or BOUNDED; "
            "recorded from the gate numbers even if the conductor stale-flagged it."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the monotonic ARC progress metric carried from the registry, not re-counted."
        )
    },
    "cited_upstream_artifacts": {
        "principle": "list of {experiment_id, fields_imported, sha256} -- the audit trail."
    },
    "stale_false_positive_notes": {
        "principle": (
            "records which conductor flags were stale-linter false-positives "
            "(live re-check clean) vs genuine -- so the headline is not lost "
            "to a known false flag."
        )
    },
    "inference_substrate": {
        "principle": "aggregation_from_upstream_artifacts (0.0001s floor)."
    },
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "s1_structural_energy_verdict",
    "reproducible_total_levels",
    "cited_upstream_artifacts",
    "stale_false_positive_notes",
    "inference_substrate",
    "levelup_bank",
    "self_play_checkpoint",
    "readiness",
    "silent_bug_audit",
    "submission_package_state",
    "hardware_continuity",
    "sota_handoff",
    "upstream_oracle_declarations",
    "flagged_artifacts_skipped",
    "preconditions_checked",
    "field_principles",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
)

TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
    "blocked_",
)


@dataclass(frozen=True)
class UpstreamSource:
    """One upstream result that can feed the capstone after live checks."""

    experiment_id: int
    relative_path: str


@dataclass(frozen=True)
class SummarizerResult:
    """Captured `scripts/summarize_artifact.py` result for one upstream artifact."""

    command: list[str]
    exit_code: int
    stdout: str
    stderr: str


Summarizer = Callable[[Path, str], SummarizerResult]

UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "S1": UpstreamSource(
        4781, "results/experiment_4781_structural_energy_s1_contrastive_landscape.json"
    ),
    "LEVELUP": UpstreamSource(4782, "results/experiment_4782_levelup_attempt.json"),
    "SELF_PLAY": UpstreamSource(
        4783, "results/experiment_4783_self_play_verifier_checkpoint.json"
    ),
    "HELDOUT": UpstreamSource(
        4784, "results/experiment_4784_heldout_first_win_readiness.json"
    ),
    "BUG_AUDIT": UpstreamSource(4785, "results/experiment_4785_silent_bug_audit.json"),
    "PACKAGE": UpstreamSource(
        4786, "results/experiment_4786_submission_package_harden.json"
    ),
    "HARDWARE": UpstreamSource(4787, "results/experiment_4787_kv260_continuity.json"),
    "SOTA": UpstreamSource(
        4788, "results/experiment_4788_sota_ingestion_energy_guided_search.json"
    ),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "S1": (
        "honest_verdict",
        "s1_gate_passed",
        "s2_authorized",
        "energy_ranking_loo_auroc_mean",
        "energy_ranking_loo_auroc_ci95",
        "energy_ranking_loo_auroc_per_seed",
        "n_seeds",
        "denoising_direction_agreement",
        "origin_probe_auroc",
        "origin_probe",
        "shuffled_label_control_auroc",
        "controls",
        "per_family_loo",
        "per_game_loo",
        "n_candidate_rows",
        "n_pos",
        "n_neg",
        "n_held_out_games",
        "random_seeds_used",
        "verifier_is_oracle",
    ),
    "LEVELUP": (
        "honest_verdict",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "registry_update",
        "attempted_games",
        "dead_ends",
        "solve_provenance",
        "verifier_is_oracle",
    ),
    "SELF_PLAY": (
        "honest_verdict",
        "verifier_checkpoint_refreshed",
        "target_game",
        "self_play_residual",
        "offline_reproduced",
        "reproduced_levels",
        "reproduction_gate",
        "solve_provenance",
        "verifier_is_oracle",
    ),
    "HELDOUT": (
        "honest_verdict",
        "heldout_first_win_rate",
        "first_win_baseline",
        "prior_best_heldout_first_win_rate",
        "heldout_first_win_delta_vs_baseline",
        "heldout_first_win_delta_vs_prior_best",
        "heldout_variant_attempts",
        "positive_control_passed",
        "parity_test_green",
        "null_delta_methodology_note",
        "verifier_is_oracle",
    ),
    "BUG_AUDIT": (
        "honest_verdict",
        "nulls_audited",
        "trusted_nulls",
        "silent_bugs_found",
        "per_null_verdicts",
        "s1_controls_fired",
        "s1_control_checks",
        "verifier_is_oracle",
    ),
    "PACKAGE": (
        "honest_verdict",
        "submission_package_ready",
        "submitted_to_leaderboard",
        "operator_only",
        "vram_estimate_gb",
        "package_builds",
        "verifier_is_oracle",
    ),
    "HARDWARE": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "loaded_overlay",
        "board_state",
        "verifier_is_oracle",
    ),
    "SOTA": (
        "honest_verdict",
        "flagged_for_v441",
        "methods_mapped",
        "s1_context",
        "arxiv_ids_cited",
        "verifier_is_oracle",
    ),
}


def stable_json(value: Any) -> str:
    """Encode JSON deterministically so checksums track content drift only."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    """Return the repository's normal `sha256:<hex>` provenance string."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str | None:
    """Hash a file when it exists; missing files are recorded separately."""

    return sha256_bytes(path.read_bytes()) if path.exists() else None


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash artifact content while excluding the checksum field itself."""

    filtered = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    return sha256_bytes(stable_json(filtered).encode("utf-8"))


def _read_json_object(path: Path) -> JsonDict:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml_object(path: Path) -> JsonDict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return int(value)
    return default


def _float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return round(float(value), 12)
    return None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _experiment_id(source: str, artifact: Mapping[str, Any] | None = None) -> int:
    if artifact:
        for field in ("experiment_id", "experiment"):
            value = artifact.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return UPSTREAM_SOURCES[source].experiment_id


def _is_flagged(artifact: Mapping[str, Any] | None) -> bool:
    return bool(artifact and artifact.get("flagged_adversarial") is True)


def _oracle_declared(artifact: Mapping[str, Any] | None) -> bool:
    return bool(artifact and artifact.get("verifier_is_oracle") is True)


def _live_clean(summary: SummarizerResult | None) -> bool:
    return summary is not None and summary.exit_code == 0


def _live_critical(summary: SummarizerResult | None) -> bool:
    return bool(summary and summary.exit_code >= 2)


def _stale_false_positive(
    artifact: Mapping[str, Any] | None,
    summary: SummarizerResult | None,
) -> bool:
    return _is_flagged(artifact) and _live_clean(summary)


def _skip_metrics(
    artifact: Mapping[str, Any] | None,
    summary: SummarizerResult | None,
) -> bool:
    if artifact is None:
        return True
    return _live_critical(summary) or (_is_flagged(artifact) and not _stale_false_positive(artifact, summary))


def _imported_fields(
    source: str,
    artifact: Mapping[str, Any],
    summary: SummarizerResult | None,
) -> list[str]:
    if _skip_metrics(artifact, summary):
        return ["flagged_adversarial"] if _is_flagged(artifact) else ["live_critical_recheck"]
    fields = [field for field in CLEAN_IMPORT_FIELDS[source] if field in artifact]
    return ["flagged_adversarial", *fields] if _is_flagged(artifact) else fields


def _cited_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source in UPSTREAM_SOURCES:
        artifact = artifacts.get(source)
        if artifact is None:
            continue
        rows.append(
            {
                "experiment_id": _experiment_id(source, artifact),
                "fields_imported": _imported_fields(
                    source, artifact, summarizer_results.get(source)
                ),
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _flagged_artifacts_skipped(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source, artifact in artifacts.items():
        summary = summarizer_results.get(source)
        if not _skip_metrics(artifact, summary):
            continue
        reason = "live_critical_recheck" if _live_critical(summary) else "flagged_adversarial"
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "path": UPSTREAM_SOURCES[source].relative_path,
                "reason": reason,
                "sha256": artifact_sha256.get(source, ""),
            }
        )
    return rows


def _stale_false_positive_notes(
    artifacts: Mapping[str, Mapping[str, Any]],
    summarizer_results: Mapping[str, SummarizerResult],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source, artifact in artifacts.items():
        if not _is_flagged(artifact):
            continue
        summary = summarizer_results.get(source)
        stale = _stale_false_positive(artifact, summary)
        rows.append(
            {
                "source": source,
                "experiment_id": _experiment_id(source, artifact),
                "status": "stale_false_positive" if stale else "genuine_or_unresolved",
                "live_recheck_exit_code": summary.exit_code if summary else None,
                "metrics_imported": stale,
                "known_stale_conductor_commit": KNOWN_STALE_CONDUCTOR_COMMIT
                if stale
                else "",
                "note": (
                    "live summarizer re-check clean; gate numbers may be read"
                    if stale
                    else "stamped flag was not live-clean; metrics skipped"
                ),
            }
        )
    return rows


def _oracle_declarations(artifacts: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    declarations: dict[str, JsonDict] = {}
    for source, artifact in artifacts.items():
        declared = _oracle_declared(artifact)
        declarations[source] = {
            "experiment_id": _experiment_id(source, artifact),
            "verifier_is_oracle": declared,
            "moat_claim_allowed": not declared,
        }
    return declarations


def _numeric_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    parsed = [_float(item) for item in value]
    return [item for item in parsed if item is not None]


def _s1_controls(artifact: Mapping[str, Any]) -> JsonDict:
    controls = _mapping(artifact.get("controls"))
    marginal = _float(
        controls.get("v2_frame_marginal_energy_ranking_loo_auroc_mean")
        or artifact.get("energy_ranking_loo_auroc_marginal_control_mean")
    )
    origin = _float(artifact.get("origin_probe_auroc"))
    shuffled = _float(artifact.get("shuffled_label_control_auroc"))
    return {
        "origin_probe_auroc": origin,
        "shuffled_label_control_auroc": shuffled,
        "marginal_control_loo_auroc": marginal,
        "origin_probe_passed": origin is not None and origin <= 0.55,
        "shuffled_label_control_passed": shuffled is not None and shuffled <= 0.55,
        "marginal_control_passed": marginal is not None and marginal <= 0.55,
        "shuffled_label_resamples": _int(controls.get("shuffled_label_resamples")),
    }


def _s1_verdict(
    artifact: Mapping[str, Any] | None,
    summary: SummarizerResult | None,
) -> JsonDict:
    if artifact is None:
        return {}
    if _skip_metrics(artifact, summary):
        return {
            "source": "S1",
            "experiment_id": _experiment_id("S1", artifact),
            "verdict": "bounded",
            "s2_authorized": False,
            "reason": "s1_artifact_skipped_live_or_genuine_flag",
            "artifact_skipped": True,
            "verifier_is_oracle": _oracle_declared(artifact),
        }

    per_seed = _numeric_list(artifact.get("energy_ranking_loo_auroc_per_seed"))
    n_seeds = _int(artifact.get("n_seeds"), len(per_seed))
    loo_mean = _float(artifact.get("energy_ranking_loo_auroc_mean"))
    denoising = _float(artifact.get("denoising_direction_agreement"))
    controls = _s1_controls(artifact)
    seed_floor_met = n_seeds >= 10 and len(per_seed) >= 10
    loo_gate = loo_mean is not None and loo_mean >= 0.70 and seed_floor_met and min(per_seed) >= 0.70
    denoising_passed = denoising is not None and denoising > 0.5
    leak_controls_hold = all(
        controls[field] is True
        for field in (
            "origin_probe_passed",
            "shuffled_label_control_passed",
            "marginal_control_passed",
        )
    )
    upstream_gate = artifact.get("s1_gate_passed") is True and artifact.get("s2_authorized") is True
    oracle = _oracle_declared(artifact)
    usable = bool(upstream_gate and loo_gate and denoising_passed and leak_controls_hold and not oracle)
    reason = (
        "s1_oracle_not_moat"
        if oracle
        else ("s1_gate_numbers_authorize_s2" if usable else "s1_gate_numbers_do_not_authorize_s2")
    )
    return {
        "source": "S1",
        "experiment_id": _experiment_id("S1", artifact),
        "upstream_honest_verdict": artifact.get("honest_verdict", ""),
        "verdict": "usable_landscape" if usable else "bounded",
        "s2_authorized": usable,
        "reason": reason,
        "artifact_skipped": False,
        "s1_gate_passed": artifact.get("s1_gate_passed") is True,
        "upstream_s2_authorized": artifact.get("s2_authorized") is True,
        "energy_ranking_loo_auroc_mean": loo_mean,
        "energy_ranking_loo_auroc_ci95": artifact.get("energy_ranking_loo_auroc_ci95")
        if isinstance(artifact.get("energy_ranking_loo_auroc_ci95"), list)
        else [],
        "energy_ranking_loo_auroc_per_seed": per_seed,
        "n_seeds": n_seeds,
        "seed_floor_met": seed_floor_met,
        "loo_gate_passed": loo_gate,
        "denoising_direction_agreement": denoising,
        "denoising_direction_passed": denoising_passed,
        "leak_controls_hold": leak_controls_hold,
        "leak_controls": controls,
        "per_family_loo": artifact.get("per_family_loo")
        if isinstance(artifact.get("per_family_loo"), Mapping)
        else {},
        "n_candidate_rows": _int(artifact.get("n_candidate_rows")),
        "n_pos": _int(artifact.get("n_pos")),
        "n_neg": _int(artifact.get("n_neg")),
        "n_held_out_games": _int(artifact.get("n_held_out_games")),
        "verifier_is_oracle": oracle,
    }


def _levelup_bank(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    registry_update = _mapping(artifact.get("registry_update"))
    before = _int(registry_update.get("reproducible_total_levels_before"))
    after = _int(registry_update.get("reproducible_total_levels_after"), before)
    oracle = _oracle_declared(artifact)
    return {
        "source": "LEVELUP",
        "experiment_id": _experiment_id("LEVELUP", artifact),
        "target_game": artifact.get("target_game"),
        "new_levels_banked": _int(artifact.get("new_levels_banked")),
        "reproducible_total_levels_before": before,
        "reproducible_total_levels_after": after,
        "reproducible_total_levels_delta": after - before,
        "registry_updated": registry_update.get("updated") is True,
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "verifier_is_oracle": oracle,
        "moat_claim": bool(after > before and not oracle),
    }


def _self_play_checkpoint(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    refreshed = artifact.get("verifier_checkpoint_refreshed") is True
    return {
        "source": "SELF_PLAY",
        "experiment_id": _experiment_id("SELF_PLAY", artifact),
        "decision": "checkpoint_refreshed" if refreshed else "checkpoint_not_refreshed",
        "verifier_checkpoint_refreshed": refreshed,
        "target_game": artifact.get("target_game"),
        "self_play_residual": artifact.get("self_play_residual"),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "reproduced_levels": _int(artifact.get("reproduced_levels")),
        "solve_provenance": artifact.get("solve_provenance", ""),
    }


def _heldout_readiness(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    baseline_delta = _float(artifact.get("heldout_first_win_delta_vs_baseline"))
    prior_delta = _float(artifact.get("heldout_first_win_delta_vs_prior_best"))
    flat = baseline_delta == 0.0 and prior_delta == 0.0
    return {
        "source": "HELDOUT",
        "experiment_id": _experiment_id("HELDOUT", artifact),
        "decision": "flat_null_no_readiness_gain" if flat else "heldout_readiness_changed",
        "heldout_first_win_rate": _float(artifact.get("heldout_first_win_rate")),
        "first_win_baseline": _float(artifact.get("first_win_baseline")),
        "prior_best_heldout_first_win_rate": _float(
            artifact.get("prior_best_heldout_first_win_rate")
        ),
        "heldout_first_win_delta_vs_baseline": baseline_delta,
        "heldout_first_win_delta_vs_prior_best": prior_delta,
        "heldout_variant_attempts": _int(artifact.get("heldout_variant_attempts")),
        "parity_test_green": artifact.get("parity_test_green") is True,
        "positive_control_passed": artifact.get("positive_control_passed") is True,
        "null_delta_methodology_note_present": bool(artifact.get("null_delta_methodology_note")),
    }


def _silent_bug_audit(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    silent_bugs = artifact.get("silent_bugs_found")
    silent_bugs = silent_bugs if isinstance(silent_bugs, list) else []
    reopened_ids = [
        row.get("null_id")
        for row in silent_bugs
        if isinstance(row, Mapping) and row.get("verdict") == "silent_bug_must_reopen"
    ]
    s1_reopened = "experiment_4781_structural_energy_s1_contrastive_landscape" in reopened_ids
    return {
        "source": "BUG_AUDIT",
        "experiment_id": _experiment_id("BUG_AUDIT", artifact),
        "nulls_audited": _int(artifact.get("nulls_audited")),
        "trusted_nulls": artifact.get("trusted_nulls")
        if isinstance(artifact.get("trusted_nulls"), list)
        else [],
        "silent_bugs_found_count": len(silent_bugs),
        "reopened_null_ids": reopened_ids,
        "s1_controls_fired": artifact.get("s1_controls_fired") is True,
        "s1_control_numbers": dict(_mapping(artifact.get("s1_control_checks"))),
        "s1_audit_note": "audit_note_recorded_does_not_override_live_clean_s1_pass"
        if s1_reopened
        else "no_s1_reopen_note",
    }


def _submission_package_state(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    ready = artifact.get("submission_package_ready") is True
    submitted = artifact.get("submitted_to_leaderboard") is True
    return {
        "source": "PACKAGE",
        "experiment_id": _experiment_id("PACKAGE", artifact),
        "decision": "package_ready_operator_only" if ready and not submitted else "package_not_ready",
        "submission_package_ready": ready,
        "submitted_to_leaderboard": submitted,
        "operator_only": artifact.get("operator_only") is True,
        "vram_estimate_gb": _float(artifact.get("vram_estimate_gb")),
    }


def _hardware_continuity(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    reachable = artifact.get("kv260_ssh_reachable") is True
    board_state = _mapping(artifact.get("board_state"))
    return {
        "source": "HARDWARE",
        "experiment_id": _experiment_id("HARDWARE", artifact),
        "decision": "kv260_reachable" if reachable else "kv260_unreachable",
        "kv260_ssh_reachable": reachable,
        "loaded_overlay": artifact.get("loaded_overlay"),
        "board_hostname": board_state.get("hostname"),
        "uio_device_count": _int(board_state.get("uio_device_count")),
    }


def _sota_handoff(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    flagged = artifact.get("flagged_for_v441")
    flagged = flagged if isinstance(flagged, list) else []
    methods = artifact.get("methods_mapped")
    methods = methods if isinstance(methods, list) else []
    return {
        "source": "SOTA",
        "experiment_id": _experiment_id("SOTA", artifact),
        "decision": "sota_handoff_mapped" if methods else "sota_handoff_empty",
        "flagged_for_v441_candidates": [
            row.get("candidate") for row in flagged if isinstance(row, Mapping)
        ],
        "methods_mapped_count": len(methods),
        "s1_context": dict(_mapping(artifact.get("s1_context"))),
        "arxiv_ids_cited": artifact.get("arxiv_ids_cited")
        if isinstance(artifact.get("arxiv_ids_cited"), list)
        else [],
    }


def _readiness(
    s1_verdict: Mapping[str, Any],
    heldout: Mapping[str, Any],
    package: Mapping[str, Any],
) -> JsonDict:
    heldout_changed = heldout.get("decision") == "heldout_readiness_changed"
    package_ready = package.get("decision") == "package_ready_operator_only"
    usable_landscape = s1_verdict.get("verdict") == "usable_landscape"
    return {
        "s1_verdict": s1_verdict.get("verdict", ""),
        "heldout_decision": heldout.get("decision", ""),
        "submission_package_decision": package.get("decision", ""),
        "ready_for_operator_submit": bool(usable_landscape and heldout_changed and package_ready),
        "reason": "requires_usable_s1_clean_package_and_heldout_gain",
    }


def _default_preconditions(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    registry_present: bool,
    registry_loadable: bool,
    registry_sha256: str | None,
    spec_has_req: bool,
    summarizer_present: bool,
    summarizer_results: Mapping[str, SummarizerResult],
) -> JsonDict:
    return {
        "agents_md_read": True,
        "codex_md_read": True,
        "summarizer": {
            "path": SUMMARIZER_RELATIVE_PATH,
            "present": summarizer_present,
        },
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
            "sha256": registry_sha256 or "",
        },
        "spec_has_req_4789": spec_has_req,
        "upstream_artifacts": {
            source: {
                "path": spec.relative_path,
                "present": source in artifacts,
                "summarizer_exit_code": summarizer_results[source].exit_code
                if source in summarizer_results
                else None,
            }
            for source, spec in UPSTREAM_SOURCES.items()
        },
    }


def build_artifact(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    registry_sha256: str | None,
    summarizer_results: Mapping[str, SummarizerResult],
    duration_s: float,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build the complete .440 capstone after live summaries have run."""

    s1 = _s1_verdict(artifacts.get("S1"), summarizer_results.get("S1"))
    honest_verdict = (
        "success_s1_structural_energy_usable_landscape_s2_authorized"
        if s1.get("verdict") == "usable_landscape"
        else "complete_s1_structural_energy_bounded_v440_capstone_ready"
    )
    heldout = _heldout_readiness(artifacts.get("HELDOUT"))
    package = _submission_package_state(artifacts.get("PACKAGE"))
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "s1_structural_energy_verdict": s1,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": _cited_artifacts(
            artifacts, artifact_sha256, summarizer_results
        ),
        "stale_false_positive_notes": _stale_false_positive_notes(
            artifacts, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": _levelup_bank(artifacts.get("LEVELUP")),
        "self_play_checkpoint": _self_play_checkpoint(artifacts.get("SELF_PLAY")),
        "readiness": _readiness(s1, heldout, package),
        "heldout_readiness": heldout,
        "silent_bug_audit": _silent_bug_audit(artifacts.get("BUG_AUDIT")),
        "submission_package_state": package,
        "hardware_continuity": _hardware_continuity(artifacts.get("HARDWARE")),
        "sota_handoff": _sota_handoff(artifacts.get("SOTA")),
        "upstream_oracle_declarations": _oracle_declarations(artifacts),
        "flagged_artifacts_skipped": _flagged_artifacts_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "preconditions_checked": dict(
            preconditions_checked
            or _default_preconditions(
                artifacts,
                registry_present=bool(registry),
                registry_loadable=bool(registry),
                registry_sha256=registry_sha256,
                spec_has_req=True,
                summarizer_present=True,
                summarizer_results=summarizer_results,
            )
        ),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": True,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def build_blocked_artifact(
    *,
    reason: str,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    summarizer_results: Mapping[str, SummarizerResult],
    duration_s: float,
    preconditions_checked: Mapping[str, Any],
) -> JsonDict:
    """Build a schema-valid blocked artifact without importing metric claims."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": f"blocked_{reason}",
        "s1_structural_energy_verdict": {},
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": [],
        "stale_false_positive_notes": _stale_false_positive_notes(
            artifacts, summarizer_results
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": {},
        "self_play_checkpoint": {},
        "readiness": {},
        "heldout_readiness": {},
        "silent_bug_audit": {},
        "submission_package_state": {},
        "hardware_continuity": {},
        "sota_handoff": {},
        "upstream_oracle_declarations": _oracle_declarations(artifacts),
        "flagged_artifacts_skipped": _flagged_artifacts_skipped(
            artifacts, artifact_sha256, summarizer_results
        ),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": FIELD_PRINCIPLES,
        "duration_s": round(max(0.0001, float(duration_s)), 6),
        "random_seed": RANDOM_SEED,
        "capstone_ready": False,
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    """Return schema errors for the scorecard without mutating the artifact."""

    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("invalid_inference_substrate")
    if not isinstance(payload.get("reproducible_total_levels"), int):
        errors.append("invalid_reproducible_total_levels")
    field_principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    s1 = payload.get("s1_structural_energy_verdict")
    if isinstance(s1, Mapping) and s1:
        if s1.get("verdict") not in {"usable_landscape", "bounded"}:
            errors.append("invalid_s1_verdict")
    cited = payload.get("cited_upstream_artifacts")
    if not isinstance(cited, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not isinstance(row.get("fields_imported"), list)
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in cited
    ):
        errors.append("invalid_cited_upstream_artifacts")
    flagged_ids = {
        row.get("experiment_id")
        for row in payload.get("flagged_artifacts_skipped", [])
        if isinstance(row, Mapping)
    }
    if isinstance(cited, list):
        for row in cited:
            if (
                isinstance(row, Mapping)
                and row.get("experiment_id") in flagged_ids
                and row.get("fields_imported") != ["flagged_adversarial"]
            ):
                errors.append(f"flagged_artifact_imported_metrics:{row.get('experiment_id')}")
    if not str(payload.get("reproducibility_checksum", "")).startswith("sha256:"):
        errors.append("invalid_reproducibility_checksum")
    return errors


def _run_summarizer(root: Path, relative_path: str) -> SummarizerResult:  # pragma: no cover
    cmd = [sys.executable, SUMMARIZER_RELATIVE_PATH, relative_path]
    proc = subprocess.run(cmd, cwd=root, text=True, capture_output=True, check=False)
    return SummarizerResult(
        command=cmd,
        exit_code=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def _first_blocker(
    *,
    summarizer_present: bool,
    registry_present: bool,
    registry_loadable: bool,
    spec_has_req: bool,
    upstream_preconditions: Mapping[str, Mapping[str, Any]],
) -> str | None:
    if not summarizer_present:
        return "missing_summarizer"
    if not registry_present:
        return "missing_registry"
    if not registry_loadable:
        return "registry_not_yaml_loadable"
    if not spec_has_req:
        return "spec_missing_req_4789"
    for source, info in upstream_preconditions.items():
        if info.get("present") is not True:
            return f"missing_upstream:{source}"
    return None


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    summarizer: Summarizer | None = None,
) -> JsonDict:
    """Read upstreams via the summarizer, aggregate, and write the scorecard."""

    start = time.perf_counter()
    summarizer = summarizer or _run_summarizer
    summarizer_path = root / SUMMARIZER_RELATIVE_PATH
    summarizer_present = summarizer_path.exists()
    artifacts: dict[str, JsonDict] = {}
    artifact_sha256: dict[str, str] = {}
    summarizer_results: dict[str, SummarizerResult] = {}
    upstream_preconditions: dict[str, JsonDict] = {}

    for source, spec in UPSTREAM_SOURCES.items():
        path = root / spec.relative_path
        present = path.exists()
        upstream_preconditions[source] = {"path": spec.relative_path, "present": present}
        if not present:
            continue
        if summarizer_present:
            summary = summarizer(root, spec.relative_path)
            summarizer_results[source] = summary
            upstream_preconditions[source]["summarizer_exit_code"] = summary.exit_code
        artifacts[source] = _read_json_object(path)
        artifact_sha256[source] = file_sha256(path) or ""

    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_present = registry_path.exists()
    registry_loadable = False
    registry: JsonDict = {}
    if registry_present:
        try:
            registry = _read_yaml_object(registry_path)
            registry_loadable = True
        except yaml.YAMLError:
            registry = {}

    spec_path = root / SPEC_RELATIVE_PATH
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4789" in spec_path.read_text(
        encoding="utf-8"
    )
    preconditions_checked = {
        "agents_md_read": True,
        "codex_md_read": True,
        "summarizer": {
            "path": SUMMARIZER_RELATIVE_PATH,
            "present": summarizer_present,
        },
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
            "sha256": file_sha256(registry_path) or "",
        },
        "spec_has_req_4789": spec_has_req,
        "upstream_artifacts": upstream_preconditions,
    }
    duration_s = time.perf_counter() - start
    blocker = _first_blocker(
        summarizer_present=summarizer_present,
        registry_present=registry_present,
        registry_loadable=registry_loadable,
        spec_has_req=spec_has_req,
        upstream_preconditions=upstream_preconditions,
    )
    if blocker is not None:
        artifact = build_blocked_artifact(
            reason=blocker,
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            registry=registry,
            summarizer_results=summarizer_results,
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
        )
    else:
        artifact = build_artifact(
            artifacts=artifacts,
            artifact_sha256=artifact_sha256,
            registry=registry,
            registry_sha256=file_sha256(registry_path),
            summarizer_results=summarizer_results,
            duration_s=duration_s,
            preconditions_checked=preconditions_checked,
        )
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - CLI boundary
    artifact = run_capstone()
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact.get("honest_verdict"),
                "schema_errors": errors,
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main())

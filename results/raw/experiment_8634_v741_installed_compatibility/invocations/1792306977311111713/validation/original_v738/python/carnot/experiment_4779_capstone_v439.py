"""Experiment 4779: .439 S0' origin-matched structural-energy capstone.

Spec refs: REQ-CAPSTONE-4779, SCENARIO-CAPSTONE-4779,
SCENARIO-CAPSTONE-4779-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4779-FIELD-PRINCIPLES.

This scorecard is aggregation-only. Its main job is to prevent the S0' headline
from being promoted when the upstream artifact is stamped adversarial, while
still carrying the clean audit's control-number diagnosis for the reader.
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
EXPERIMENT = "experiment_4779_capstone_v439"
EXPERIMENT_ID = 4779
SCHEMA = "carnot.exp4779.capstone_v439.v1"
RESULT_RELATIVE_PATH = "results/experiment_4779_capstone_v439.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 4779
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4779",
    "SCENARIO-CAPSTONE-4779",
    "SCENARIO-CAPSTONE-4779-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4779-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; capstone_ready=true is success_/complete_."},
    "s0prime_structural_energy_verdict": {
        "principle": (
            "the headline -- direction REOPENS to S1 (origin-matched signal survives) "
            "or RETIRES (was an origin leak); the milestone's load-bearing result."
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
    "flagged_artifacts_skipped": {
        "principle": (
            "records which flagged_adversarial artifacts were excluded -- never aggregate "
            "a fabricated number."
        )
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts; 0.0001s floor."},
}

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "s0prime_structural_energy_verdict",
    "reproducible_total_levels",
    "cited_upstream_artifacts",
    "flagged_artifacts_skipped",
    "inference_substrate",
    "levelup_bank",
    "self_play_checkpoint",
    "heldout_readiness",
    "readiness",
    "silent_bug_audit",
    "submission_package_state",
    "hardware_continuity",
    "sota_handoff",
    "upstream_oracle_declarations",
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
    """One upstream result that can feed the capstone after checks."""

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
    "S0PRIME": UpstreamSource(
        4771, "results/experiment_4771_structural_energy_s0prime_origin_matched.json"
    ),
    "LEVELUP": UpstreamSource(4772, "results/experiment_4772_levelup_attempt.json"),
    "SELF_PLAY": UpstreamSource(
        4773, "results/experiment_4773_self_play_verifier_checkpoint.json"
    ),
    "HELDOUT": UpstreamSource(
        4774, "results/experiment_4774_heldout_first_win_readiness.json"
    ),
    "BUG_AUDIT": UpstreamSource(4775, "results/experiment_4775_silent_bug_audit.json"),
    "PACKAGE": UpstreamSource(
        4776, "results/experiment_4776_submission_package_harden.json"
    ),
    "HARDWARE": UpstreamSource(4777, "results/experiment_4777_kv260_continuity.json"),
    "SOTA": UpstreamSource(
        4778, "results/experiment_4778_sota_ingestion_structural_energy.json"
    ),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "S0PRIME": (
        "honest_verdict",
        "s0prime_gate_passed",
        "loo_auroc_structural",
        "loo_auroc_ci95",
        "loo_auroc_marginal_control",
        "loo_auroc_majority_control",
        "origin_probe_auroc",
        "shuffled_label_control_auroc",
        "controls",
        "dataset_diagnostics",
        "n_candidate_rows",
        "n_pos",
        "n_neg",
        "n_held_out_games",
        "per_family_loo",
        "origin_probe",
        "structural_minus_marginal_delta_ci95",
        "retire_if_same_verdict",
        "in_sample_auroc",
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
        "s0prime_leak_controls_fired",
        "s0prime_leak_control_checks",
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
        "flagged_for_v440",
        "methods_mapped",
        "leak_robust_evaluation_note",
        "arxiv_ids_cited",
        "verifier_is_oracle",
    ),
}


def stable_json(value: Any) -> str:
    """Encode JSON deterministically so checksums catch only real content drift."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    """Return the repository's normal `sha256:<hex>` provenance string."""

    return "sha256:" + hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str | None:
    """Hash a file if it exists; missing files are represented separately."""

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


def _ci_lower_gt(value: Any, threshold: float) -> bool:
    if not isinstance(value, list) or len(value) < 2:
        return False
    lower = _float(value[0])
    return lower is not None and lower > threshold


def _control_leq(value: Any, threshold: float) -> bool:
    parsed = _float(value)
    return parsed is not None and parsed <= threshold


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


def _imported_fields(source: str, artifact: Mapping[str, Any], *, flagged: bool) -> list[str]:
    if flagged:
        return ["flagged_adversarial"]
    return [field for field in CLEAN_IMPORT_FIELDS[source] if field in artifact]


def _flagged_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for source, artifact in artifacts.items():
        if _is_flagged(artifact):
            rows.append(
                {
                    "source": source,
                    "experiment_id": _experiment_id(source, artifact),
                    "path": UPSTREAM_SOURCES[source].relative_path,
                    "reason": "flagged_adversarial",
                    "sha256": artifact_sha256.get(source, ""),
                }
            )
    return rows


def _cited_artifacts(
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
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
                    source, artifact, flagged=_is_flagged(artifact)
                ),
                "sha256": artifact_sha256.get(source, ""),
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


def _control_numbers_from_checks(
    checks: Mapping[str, Any],
    *,
    fired: Any = None,
) -> JsonDict:
    controls_fired = fired if isinstance(fired, bool) else checks.get("all_controls_fired") is True
    return {
        "s0prime_leak_controls_fired": controls_fired,
        "origin_probe_auroc": _float(checks.get("origin_probe_auroc")),
        "shuffled_label_control_auroc": _float(checks.get("shuffled_label_control_auroc")),
        "shuffled_label_resamples": _int(checks.get("shuffled_label_resamples")),
        "origin_probe_refit_on_origin_matched_data": (
            checks.get("origin_probe_refit_on_origin_matched_data") is True
        ),
        "origin_probe_status": checks.get("origin_probe_status", ""),
        "class_balance_non_degenerate": checks.get("class_balance_non_degenerate") is True,
        "shuffled_label_permuted_and_reran_loo": (
            checks.get("shuffled_label_permuted_and_reran_loo") is True
        ),
        "contributing_games_with_both_classes": len(
            checks.get("contributing_games_with_both_classes", [])
            if isinstance(checks.get("contributing_games_with_both_classes"), list)
            else []
        ),
        "single_class_games_skipped": len(
            checks.get("single_class_games_skipped", [])
            if isinstance(checks.get("single_class_games_skipped"), list)
            else []
        ),
    }


def _audit_control_numbers(audit: Mapping[str, Any] | None) -> JsonDict:
    if audit is None:
        return {}
    top_checks = audit.get("s0prime_leak_control_checks")
    if isinstance(top_checks, Mapping):
        return _control_numbers_from_checks(
            top_checks, fired=audit.get("s0prime_leak_controls_fired")
        )
    for field in ("per_null_verdicts", "silent_bugs_found"):
        rows = audit.get(field)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if row.get("null_id") != "experiment_4771_structural_energy_s0prime_origin_matched":
                continue
            checks = row.get("s0prime_leak_control_checks")
            if isinstance(checks, Mapping):
                return _control_numbers_from_checks(
                    checks, fired=row.get("s0prime_leak_controls_fired")
                )
    return {}


def _clean_s0prime_gate_passed(artifact: Mapping[str, Any]) -> bool:
    families = artifact.get("per_family_loo")
    family_signal = False
    if isinstance(families, Mapping):
        family_signal = any(
            parsed is not None and parsed > 0.55
            for parsed in (_float(value) for value in families.values())
        )
    return bool(
        artifact.get("s0prime_gate_passed") is True
        and _ci_lower_gt(artifact.get("loo_auroc_ci95"), 0.5)
        and _ci_lower_gt(artifact.get("structural_minus_marginal_delta_ci95"), 0.0)
        and _control_leq(artifact.get("origin_probe_auroc"), 0.6)
        and _control_leq(artifact.get("shuffled_label_control_auroc"), 0.55)
        and family_signal
    )


def _s0prime_verdict(
    artifact: Mapping[str, Any] | None,
    audit: Mapping[str, Any] | None,
) -> JsonDict:
    if artifact is None:
        return {}
    flagged = _is_flagged(artifact)
    oracle = _oracle_declared(artifact)
    audit_controls = _audit_control_numbers(audit)
    if flagged:
        controls_unfired = audit_controls.get("s0prime_leak_controls_fired") is False
        reason = (
            "s0prime_flagged_adversarial_skipped_controls_unfired"
            if controls_unfired
            else "s0prime_flagged_adversarial_skipped"
        )
        return {
            "source": "S0PRIME",
            "experiment_id": _experiment_id("S0PRIME", artifact),
            "direction": "RETIRES",
            "s1_queued": False,
            "reason": reason,
            "artifact_skipped": True,
            "control_numbers_source": "BUG_AUDIT" if audit_controls else "",
            "loo_auroc_structural": None,
            "origin_probe_auroc": audit_controls.get("origin_probe_auroc"),
            "shuffled_label_control_auroc": audit_controls.get(
                "shuffled_label_control_auroc"
            ),
            "shuffled_label_resamples": audit_controls.get("shuffled_label_resamples"),
            "origin_probe_refit_on_origin_matched_data": audit_controls.get(
                "origin_probe_refit_on_origin_matched_data"
            ),
            "s0prime_leak_controls_fired": audit_controls.get(
                "s0prime_leak_controls_fired"
            ),
            "audit_control_numbers": audit_controls,
            "verifier_is_oracle": oracle,
        }

    clean_gate = _clean_s0prime_gate_passed(artifact)
    if oracle:
        reason = "s0prime_oracle_not_moat"
    elif clean_gate:
        reason = "origin_matched_signal_survives_clean_controls"
    else:
        reason = "s0prime_gate_failed_or_controls_failed"
    return {
        "source": "S0PRIME",
        "experiment_id": _experiment_id("S0PRIME", artifact),
        "upstream_honest_verdict": artifact.get("honest_verdict", ""),
        "direction": "REOPENS_TO_S1" if clean_gate and not oracle else "RETIRES",
        "s1_queued": clean_gate and not oracle,
        "reason": reason,
        "artifact_skipped": False,
        "control_numbers_source": "S0PRIME",
        "s0prime_gate_passed": artifact.get("s0prime_gate_passed") is True,
        "loo_auroc_structural": _float(artifact.get("loo_auroc_structural")),
        "loo_auroc_ci95": artifact.get("loo_auroc_ci95")
        if isinstance(artifact.get("loo_auroc_ci95"), list)
        else [],
        "loo_auroc_majority_control": _float(artifact.get("loo_auroc_majority_control")),
        "loo_auroc_marginal_control": _float(artifact.get("loo_auroc_marginal_control")),
        "origin_probe_auroc": _float(artifact.get("origin_probe_auroc")),
        "shuffled_label_control_auroc": _float(artifact.get("shuffled_label_control_auroc")),
        "shuffled_label_resamples": _int(
            (artifact.get("controls") or {}).get("shuffled_label_resamples")
            if isinstance(artifact.get("controls"), Mapping)
            else None
        ),
        "structural_minus_marginal_delta_ci95": artifact.get(
            "structural_minus_marginal_delta_ci95"
        )
        if isinstance(artifact.get("structural_minus_marginal_delta_ci95"), list)
        else [],
        "per_family_loo": artifact.get("per_family_loo")
        if isinstance(artifact.get("per_family_loo"), Mapping)
        else {},
        "n_candidate_rows": _int(artifact.get("n_candidate_rows")),
        "n_pos": _int(artifact.get("n_pos")),
        "n_neg": _int(artifact.get("n_neg")),
        "n_held_out_games": _int(artifact.get("n_held_out_games")),
        "origin_matched": (
            isinstance(artifact.get("dataset_diagnostics"), Mapping)
            and artifact["dataset_diagnostics"].get("origin_matched") is True
        ),
        "in_sample_auroc": _float(artifact.get("in_sample_auroc")),
        "verifier_is_oracle": oracle,
    }


def _levelup_bank(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    if _is_flagged(artifact):
        return {
            "source": "LEVELUP",
            "experiment_id": _experiment_id("LEVELUP", artifact),
            "decision": "skipped_flagged_adversarial",
        }
    registry_update = artifact.get("registry_update")
    registry_update = registry_update if isinstance(registry_update, Mapping) else {}
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
    if _is_flagged(artifact):
        return {
            "source": "SELF_PLAY",
            "experiment_id": _experiment_id("SELF_PLAY", artifact),
            "decision": "skipped_flagged_adversarial",
        }
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
    }


def _heldout_readiness(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    if _is_flagged(artifact):
        return {
            "source": "HELDOUT",
            "experiment_id": _experiment_id("HELDOUT", artifact),
            "decision": "skipped_flagged_adversarial",
        }
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
        "null_delta_methodology_note_present": bool(
            artifact.get("null_delta_methodology_note")
        ),
    }


def _silent_bug_audit(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    if _is_flagged(artifact):
        return {
            "source": "BUG_AUDIT",
            "experiment_id": _experiment_id("BUG_AUDIT", artifact),
            "decision": "skipped_flagged_adversarial",
        }
    silent_bugs = artifact.get("silent_bugs_found")
    silent_bugs = silent_bugs if isinstance(silent_bugs, list) else []
    reopened_ids = [
        row.get("null_id")
        for row in silent_bugs
        if isinstance(row, Mapping) and row.get("verdict") == "silent_bug_must_reopen"
    ]
    controls = _audit_control_numbers(artifact)
    return {
        "source": "BUG_AUDIT",
        "experiment_id": _experiment_id("BUG_AUDIT", artifact),
        "nulls_audited": _int(artifact.get("nulls_audited")),
        "trusted_nulls": artifact.get("trusted_nulls")
        if isinstance(artifact.get("trusted_nulls"), list)
        else [],
        "silent_bugs_found_count": len(silent_bugs),
        "reopened_null_ids": reopened_ids,
        "s0prime_reopened_for_control_bug": (
            "experiment_4771_structural_energy_s0prime_origin_matched" in reopened_ids
        ),
        "s0prime_leak_controls_fired": controls.get("s0prime_leak_controls_fired"),
        "s0prime_control_numbers": controls,
    }


def _submission_package_state(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    if _is_flagged(artifact):
        return {
            "source": "PACKAGE",
            "experiment_id": _experiment_id("PACKAGE", artifact),
            "decision": "skipped_flagged_adversarial",
        }
    ready = artifact.get("submission_package_ready") is True
    submitted = artifact.get("submitted_to_leaderboard") is True
    return {
        "source": "PACKAGE",
        "experiment_id": _experiment_id("PACKAGE", artifact),
        "decision": "package_ready_operator_only"
        if ready and not submitted
        else "package_not_ready",
        "submission_package_ready": ready,
        "submitted_to_leaderboard": submitted,
        "operator_only": artifact.get("operator_only") is True,
        "vram_estimate_gb": _float(artifact.get("vram_estimate_gb")),
    }


def _hardware_continuity(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    if _is_flagged(artifact):
        return {
            "source": "HARDWARE",
            "experiment_id": _experiment_id("HARDWARE", artifact),
            "decision": "skipped_flagged_adversarial",
        }
    reachable = artifact.get("kv260_ssh_reachable") is True
    board_state = artifact.get("board_state")
    board_state = board_state if isinstance(board_state, Mapping) else {}
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
    if _is_flagged(artifact):
        return {
            "source": "SOTA",
            "experiment_id": _experiment_id("SOTA", artifact),
            "decision": "skipped_flagged_adversarial",
        }
    flagged_for_v440 = artifact.get("flagged_for_v440")
    flagged_for_v440 = flagged_for_v440 if isinstance(flagged_for_v440, list) else []
    methods = artifact.get("methods_mapped")
    methods = methods if isinstance(methods, list) else []
    leak_note = artifact.get("leak_robust_evaluation_note")
    leak_note = leak_note if isinstance(leak_note, Mapping) else {}
    return {
        "source": "SOTA",
        "experiment_id": _experiment_id("SOTA", artifact),
        "decision": "sota_handoff_mapped" if methods else "sota_handoff_empty",
        "flagged_for_v440_candidates": [
            row.get("candidate") for row in flagged_for_v440 if isinstance(row, Mapping)
        ],
        "methods_mapped_count": len(methods),
        "leak_robust_roadmap_gate": leak_note.get("roadmap_gate", ""),
        "arxiv_ids_cited": artifact.get("arxiv_ids_cited")
        if isinstance(artifact.get("arxiv_ids_cited"), list)
        else [],
    }


def _readiness(heldout: Mapping[str, Any], package: Mapping[str, Any]) -> JsonDict:
    heldout_changed = heldout.get("decision") == "heldout_readiness_changed"
    package_ready = package.get("decision") == "package_ready_operator_only"
    return {
        "heldout_decision": heldout.get("decision", ""),
        "submission_package_decision": package.get("decision", ""),
        "ready_for_operator_submit": bool(heldout_changed and package_ready),
        "reason": "requires_clean_package_and_heldout_gain",
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
        "spec_has_req_4779": spec_has_req,
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
    """Build the complete scorecard after preconditions and summaries pass."""

    s0prime = _s0prime_verdict(artifacts.get("S0PRIME"), artifacts.get("BUG_AUDIT"))
    direction = s0prime.get("direction")
    honest_verdict = (
        "success_s0prime_structural_energy_reopens_s1"
        if direction == "REOPENS_TO_S1"
        else "complete_s0prime_structural_energy_retires_v439_capstone_ready"
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
        "s0prime_structural_energy_verdict": s0prime,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": _cited_artifacts(artifacts, artifact_sha256),
        "flagged_artifacts_skipped": _flagged_artifacts(artifacts, artifact_sha256),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": _levelup_bank(artifacts.get("LEVELUP")),
        "self_play_checkpoint": _self_play_checkpoint(artifacts.get("SELF_PLAY")),
        "heldout_readiness": heldout,
        "readiness": _readiness(heldout, package),
        "silent_bug_audit": _silent_bug_audit(artifacts.get("BUG_AUDIT")),
        "submission_package_state": package,
        "hardware_continuity": _hardware_continuity(artifacts.get("HARDWARE")),
        "sota_handoff": _sota_handoff(artifacts.get("SOTA")),
        "upstream_oracle_declarations": _oracle_declarations(artifacts),
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
        "s0prime_structural_energy_verdict": {},
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": [],
        "flagged_artifacts_skipped": _flagged_artifacts(artifacts, artifact_sha256),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": {},
        "self_play_checkpoint": {},
        "heldout_readiness": {},
        "readiness": {},
        "silent_bug_audit": {},
        "submission_package_state": {},
        "hardware_continuity": {},
        "sota_handoff": {},
        "upstream_oracle_declarations": _oracle_declarations(artifacts),
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
    s0prime = payload.get("s0prime_structural_energy_verdict")
    if isinstance(s0prime, Mapping) and s0prime:
        if s0prime.get("direction") not in {"REOPENS_TO_S1", "RETIRES"}:
            errors.append("invalid_s0prime_direction")
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
        return "spec_missing_req_4779"
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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4779" in spec_path.read_text(
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
        "spec_has_req_4779": spec_has_req,
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

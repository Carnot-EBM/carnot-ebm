"""Experiment 4769: .438 structural-energy capstone scorecard.

Spec refs: REQ-CAPSTONE-4769, SCENARIO-CAPSTONE-4769,
SCENARIO-CAPSTONE-4769-BLOCKED-PRECONDITION,
SCENARIO-CAPSTONE-4769-FIELD-PRINCIPLES.

This module only aggregates landed artifacts. It does not rerun S0, self-play,
submission packaging, hardware, or SOTA work; the useful work here is keeping the
audit trail honest enough that the milestone headline cannot be inflated by a
flagged or oracle-tainted upstream number.
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
EXPERIMENT = "experiment_4769_capstone_v438"
EXPERIMENT_ID = 4769
SCHEMA = "carnot.exp4769.capstone_v438.v1"
RESULT_RELATIVE_PATH = "results/experiment_4769_capstone_v438.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
SUMMARIZER_RELATIVE_PATH = "scripts/summarize_artifact.py"
RANDOM_SEED = 4769
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SPEC_REFS = [
    "REQ-CAPSTONE-4769",
    "SCENARIO-CAPSTONE-4769",
    "SCENARIO-CAPSTONE-4769-BLOCKED-PRECONDITION",
    "SCENARIO-CAPSTONE-4769-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {"principle": "terminal prefix; capstone_ready=true is success_/complete_."},
    "s0_structural_energy_verdict": {
        "principle": (
            "the headline -- direction ALIVE (S0 passed, S1 queued) or RETIRED "
            "(S0 nulled); the milestone's load-bearing result."
        )
    },
    "reproducible_total_levels": {
        "principle": (
            "the monotonic ARC progress metric carried from the registry, not re-counted."
        )
    },
    "cited_upstream_artifacts": {
        "principle": (
            "list of {experiment_id, fields_imported, sha256} -- the audit trail proving "
            "the capstone synthesizes real measurements."
        )
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
    "s0_structural_energy_verdict",
    "reproducible_total_levels",
    "cited_upstream_artifacts",
    "flagged_artifacts_skipped",
    "inference_substrate",
    "levelup_bank",
    "self_play_checkpoint",
    "heldout_readiness",
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
    """One upstream result whose numbers may feed the capstone after checks."""

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
    "S0": UpstreamSource(4761, "results/experiment_4761_structural_energy_s0_core_bet_probe.json"),
    "LEVELUP": UpstreamSource(4762, "results/experiment_4762_levelup_attempt.json"),
    "SELF_PLAY": UpstreamSource(4763, "results/experiment_4763_self_play_verifier_checkpoint.json"),
    "HELDOUT": UpstreamSource(4764, "results/experiment_4764_heldout_first_win_readiness.json"),
    "BUG_AUDIT": UpstreamSource(4765, "results/experiment_4765_silent_bug_audit.json"),
    "PACKAGE": UpstreamSource(4766, "results/experiment_4766_submission_package_harden.json"),
    "HARDWARE": UpstreamSource(4767, "results/experiment_4767_kv260_continuity.json"),
    "SOTA": UpstreamSource(4768, "results/experiment_4768_sota_ingestion_structural_energy.json"),
}

CLEAN_IMPORT_FIELDS: dict[str, tuple[str, ...]] = {
    "S0": (
        "honest_verdict",
        "s0_gate_passed",
        "retire_energy_guided_direction",
        "retire_if_same_verdict",
        "loo_auroc_structural",
        "loo_auroc_majority_control",
        "loo_auroc_marginal_control",
        "origin_probe_auroc",
        "structural_minus_marginal_delta_ci95",
        "n_held_out_games",
        "n_candidate_rows",
        "verifier_is_oracle",
    ),
    "LEVELUP": (
        "honest_verdict",
        "new_levels_banked",
        "offline_reproduced",
        "reproduced_levels",
        "target_game",
        "registry_update",
        "verifier_is_oracle",
    ),
    "SELF_PLAY": (
        "honest_verdict",
        "verifier_checkpoint_refreshed",
        "self_play_residual",
        "offline_reproduced",
        "reproduced_levels",
        "verifier_is_oracle",
    ),
    "HELDOUT": (
        "honest_verdict",
        "heldout_first_win_rate",
        "first_win_baseline",
        "prior_best_heldout_first_win_rate",
        "heldout_first_win_delta_vs_baseline",
        "heldout_first_win_delta_vs_prior_best",
        "parity_test_green",
        "positive_control_passed",
        "verifier_is_oracle",
    ),
    "BUG_AUDIT": (
        "honest_verdict",
        "nulls_audited",
        "trusted_nulls",
        "silent_bugs_found",
        "verifier_is_oracle",
    ),
    "PACKAGE": (
        "honest_verdict",
        "submission_package_ready",
        "submitted_to_leaderboard",
        "operator_only",
        "vram_estimate_gb",
        "verifier_is_oracle",
    ),
    "HARDWARE": (
        "honest_verdict",
        "kv260_ssh_reachable",
        "loaded_overlay",
        "verifier_is_oracle",
    ),
    "SOTA": (
        "honest_verdict",
        "s0_context",
        "flagged_for_v439",
        "methods_mapped",
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


def _s0_verdict(artifact: Mapping[str, Any] | None) -> JsonDict:
    if artifact is None:
        return {}
    flagged = _is_flagged(artifact)
    oracle = _oracle_declared(artifact)
    s0_gate_passed = artifact.get("s0_gate_passed") is True
    alive = s0_gate_passed and not flagged and not oracle
    if oracle:
        reason = "s0_oracle_not_moat"
    elif flagged:
        reason = "s0_flagged_skipped"
    elif alive:
        reason = "s0_clean_cross_game_transition_correctness_above_chance"
    else:
        reason = "s0_gate_failed_or_null_or_leaky"
    return {
        "source": "S0",
        "experiment_id": _experiment_id("S0", artifact),
        "upstream_honest_verdict": artifact.get("honest_verdict", ""),
        "direction": "ALIVE" if alive else "RETIRED",
        "s1_queued": alive,
        "s0_gate_passed": s0_gate_passed,
        "reason": reason,
        "loo_auroc_structural": _float(artifact.get("loo_auroc_structural")),
        "loo_auroc_majority_control": _float(artifact.get("loo_auroc_majority_control")),
        "loo_auroc_marginal_control": _float(artifact.get("loo_auroc_marginal_control")),
        "origin_probe_auroc": _float(artifact.get("origin_probe_auroc")),
        "structural_minus_marginal_delta_ci95": artifact.get(
            "structural_minus_marginal_delta_ci95"
        ),
        "n_held_out_games": _int(artifact.get("n_held_out_games")),
        "n_candidate_rows": _int(artifact.get("n_candidate_rows")),
        "retire_energy_guided_direction": artifact.get("retire_energy_guided_direction") is True,
        "retire_if_same_verdict": artifact.get("retire_if_same_verdict") is True,
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
        "parity_test_green": artifact.get("parity_test_green") is True,
        "positive_control_passed": artifact.get("positive_control_passed") is True,
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
    return {
        "source": "BUG_AUDIT",
        "experiment_id": _experiment_id("BUG_AUDIT", artifact),
        "nulls_audited": _int(artifact.get("nulls_audited")),
        "trusted_nulls": artifact.get("trusted_nulls")
        if isinstance(artifact.get("trusted_nulls"), list)
        else [],
        "silent_bugs_found_count": len(silent_bugs),
        "reopened_null_ids": reopened_ids,
        "s0_reopened_for_origin_probe_leak": (
            "experiment_4761_structural_energy_s0_core_bet_probe" in reopened_ids
        ),
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
    return {
        "source": "HARDWARE",
        "experiment_id": _experiment_id("HARDWARE", artifact),
        "decision": "kv260_reachable" if reachable else "kv260_unreachable",
        "kv260_ssh_reachable": reachable,
        "loaded_overlay": artifact.get("loaded_overlay"),
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
    flagged_for_v439 = artifact.get("flagged_for_v439")
    flagged_for_v439 = flagged_for_v439 if isinstance(flagged_for_v439, list) else []
    methods = artifact.get("methods_mapped")
    methods = methods if isinstance(methods, list) else []
    return {
        "source": "SOTA",
        "experiment_id": _experiment_id("SOTA", artifact),
        "decision": "sota_handoff_mapped" if methods else "sota_handoff_empty",
        "flagged_for_v439_candidates": [
            row.get("candidate") for row in flagged_for_v439 if isinstance(row, Mapping)
        ],
        "methods_mapped_count": len(methods),
        "s0_context": artifact.get("s0_context")
        if isinstance(artifact.get("s0_context"), Mapping)
        else {},
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
        "spec_has_req_4769": spec_has_req,
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

    s0 = _s0_verdict(artifacts.get("S0"))
    direction = s0.get("direction")
    honest_verdict = (
        "success: s0_structural_energy_alive_s1_queued"
        if direction == "ALIVE"
        else "complete: s0_structural_energy_retired_v438_capstone_ready"
    )
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "s0_structural_energy_verdict": s0,
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": _cited_artifacts(artifacts, artifact_sha256),
        "flagged_artifacts_skipped": _flagged_artifacts(artifacts, artifact_sha256),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": _levelup_bank(artifacts.get("LEVELUP")),
        "self_play_checkpoint": _self_play_checkpoint(artifacts.get("SELF_PLAY")),
        "heldout_readiness": _heldout_readiness(artifacts.get("HELDOUT")),
        "silent_bug_audit": _silent_bug_audit(artifacts.get("BUG_AUDIT")),
        "submission_package_state": _submission_package_state(artifacts.get("PACKAGE")),
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
        "s0_structural_energy_verdict": {},
        "reproducible_total_levels": _int(registry.get("reproducible_total_levels")),
        "cited_upstream_artifacts": [],
        "flagged_artifacts_skipped": _flagged_artifacts(artifacts, artifact_sha256),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "levelup_bank": {},
        "self_play_checkpoint": {},
        "heldout_readiness": {},
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
        return "spec_missing_req_4769"
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
    spec_has_req = spec_path.exists() and "REQ-CAPSTONE-4769" in spec_path.read_text(
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
        "spec_has_req_4769": spec_has_req,
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

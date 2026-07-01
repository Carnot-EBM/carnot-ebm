#!/usr/bin/env python3
"""Experiment 5095: archive .467 truth and activate the .468 record.

Spec refs: REQ-CAPSTONE-5095, SCENARIO-CAPSTONE-5095,
SCENARIO-CAPSTONE-5095-BLOCKED-YAML,
SCENARIO-CAPSTONE-5095-FIELD-PRINCIPLES.

This module writes a transition truth-record. It reads roadmap YAML plus the
prior milestone's JSON artifacts and records which facts can seed the next
milestone. It does not run a model, and it does not convert flagged runtime,
blocked process-verifier, toy constrained-decoding, no-promote FR-11, or
no-speedup hardware evidence into a broader success claim.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
Clock = Callable[[], float]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5095_archive_467_activate_468"
EXPERIMENT_ID = 5095
SCHEMA = "carnot.experiment_5095_archive_467_activate_468.v1"
RESULT_RELATIVE_PATH = Path("results") / "experiment_5095_archive_467_activate_468.json"
RANDOM_SEED = 5095
MILESTONE_FROM = "2026.07.467"
MILESTONE_TO = "2026.07.468"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
NEXT_MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"

ROADMAP_ACTIVE_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")

SOURCE_ARTIFACTS: dict[int, Path] = {
    5085: Path("results/experiment_5085_llamacpp_logprob_endpoint_bringup_v467.json"),
    5086: Path("results/experiment_5086_uprm_logprob_cache_retry_v467.json"),
    5087: Path("results/experiment_5087_uprm_process_verifier_retry_v467.json"),
    5088: Path("results/experiment_5088_temporal_consistency_prm_v467.json"),
    5089: Path("results/experiment_5089_pbit_guided_cdcl_bridge_v467.json"),
    5090: Path("results/experiment_5090_static_csr_constrained_decoding_v467.json"),
    5091: Path("results/experiment_5091_kan_pwa_milp_scale_v467.json"),
    5092: Path("results/experiment_5092_fr11_budgeted_onpolicy_memory_v467.json"),
    5093: Path("results/experiment_5093_hardware_continuity_v467.json"),
    5094: Path("results/experiment_5094_capstone_v467.json"),
}

SOURCE_LABELS: dict[int, str] = {
    5085: "runtime_endpoint",
    5086: "uprm_logprob_cache",
    5087: "uprm_process_verifier_retry",
    5088: "temporal_consistency_prm",
    5089: "pbit_cdcl_bridge",
    5090: "static_csr_constrained_decoding",
    5091: "kan_pwa_milp_scale",
    5092: "fr11_budgeted_onpolicy_memory",
    5093: "hardware_continuity",
    5094: "capstone_v467",
}

IMPORTED_FIELDS: dict[int, tuple[str, ...]] = {
    5085: (
        "honest_verdict",
        "completion_endpoint_ready",
        "logprob_endpoint_ready",
        "top_logprob_or_confidence_ready",
        "live_completion_invoked",
        "usable_sota_models",
    ),
    5086: ("honest_verdict", "logprob_cache_ready", "step_cache_ready", "endpoint_used"),
    5087: ("honest_verdict", "status", "gate_check_summary"),
    5088: ("honest_verdict", "beats_one_pass", "delta_vs_one_pass"),
    5089: (
        "honest_verdict",
        "correctness_preserved",
        "helps_declared_family",
        "delta_effort_vs_pure",
    ),
    5090: (
        "honest_verdict",
        "beats_cpu_trie",
        "beats_rerank_only_on_validity_or_cost",
        "mask_speedup",
        "validity_rate",
    ),
    5091: (
        "honest_verdict",
        "property_holds",
        "property_status",
        "solver_status",
        "binary_variable_count",
        "pwa_piece_count",
        "constraint_count",
        "global_error_bound",
    ),
    5092: (
        "honest_verdict",
        "heldout_delta",
        "nonforgetting_delta",
        "contamination_guard_passed",
        "poison_guard_passed",
        "rollback_guard_passed",
        "promotion_decision",
    ),
    5093: (
        "honest_verdict",
        "kv260_ssh_ready",
        "kv260_uio_transcript_path",
        "kv260_speedup_claim_allowed",
        "gatemate_detected",
        "gatemate_terminal_state",
        "polarfire_detected",
        "polarfire_dispatch_precheck_ready",
        "destructive_actions_taken",
    ),
    5094: ("honest_verdict", "milestone_decision"),
}

PROMPT_LISTED_MISSING_ARTIFACTS: dict[int, dict[str, str]] = {
    5088: {
        "path": "results/experiment_5088_temporal_consistency_process_verifier_v467.json",
        "reason": (
            "prompt_listed_process_verifier_path_absent; actual available artifact is "
            "results/experiment_5088_temporal_consistency_prm_v467.json"
        ),
    },
}

SPEC_REFS = [
    "REQ-CAPSTONE-5095",
    "SCENARIO-CAPSTONE-5095",
    "SCENARIO-CAPSTONE-5095-BLOCKED-YAML",
    "SCENARIO-CAPSTONE-5095-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_467_archived_468_activated_exact_verifier_pivot_"
            "carried_forward or blocked_yaml_parse."
        )
    },
    "duration_s": {
        "principle": (
            "wall-clock duration for the aggregation run; it must stay compatible "
            "with aggregation_from_upstream_artifacts and not imply live inference."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- reads roadmaps and upstream JSON "
            "only; never live model inference."
        )
    },
    "milestone_from": {
        "principle": "the archived milestone 2026.07.467 from the prior conductor pass.",
    },
    "milestone_to": {
        "principle": "the activated/staged milestone 2026.07.468 from the active roadmap.",
    },
    "source_artifacts": {
        "principle": "the present .467 artifacts loaded with sha256 provenance.",
    },
    "blocked_artifacts": {
        "principle": (
            "gate-blocked or blocked-verdict artifacts such as Exp5086 and Exp5087 "
            "recorded as blocked, not missing or successful."
        )
    },
    "flagged_artifacts": {
        "principle": (
            "flagged .467 artifacts preserved with excluded-from-headline status so "
            "their numbers are not promoted."
        )
    },
    "clean_positive_artifacts": {
        "principle": (
            "clean non-flagged .467 positives that may seed .468, especially Exp5091 KAN/PWA/MILP."
        )
    },
    "missing_artifacts": {
        "principle": "prompt-listed or expected .467 artifacts absent or unreadable, recorded explicitly.",
    },
    "close_state": {
        "principle": (
            "the blunt .467 close-state preserving clean positives, blocked runtime/"
            "process substrate, flagged toy/static claims, no-promote FR-11, and "
            "no-speedup hardware."
        )
    },
    "blockers_carried_forward": {
        "principle": (
            "the exact endpoint/uPRM/process/STATIC/p-bit/FR-11/hardware boundaries "
            "and the clean KAN scale boundary that must constrain .468 planning."
        )
    },
    "exact_verifier_pivot": {
        "principle": "the Exp5091-driven .468 pivot path and its small-property scale boundary.",
    },
    "next_milestone_doc": {
        "principle": "the .468 milestone document path read from the active roadmap.",
    },
    "docs_updated": {
        "principle": (
            "empty when the conductor stop rule delegates ops/status/changelog/"
            "traceability reconciliation."
        )
    },
    "flagged_adversarial": {
        "principle": (
            "false because this record is a transparent aggregation artifact, not a "
            "fast compute-bound claim."
        )
    },
}

REQUIRED_TOP_LEVEL_FIELDS = (
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "milestone_from",
    "milestone_to",
    "source_artifacts",
    "blocked_artifacts",
    "flagged_artifacts",
    "clean_positive_artifacts",
    "missing_artifacts",
    "close_state",
    "blockers_carried_forward",
    "exact_verifier_pivot",
    "next_milestone_doc",
    "docs_updated",
    "flagged_adversarial",
)

REQUIRED_SCHEMA_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "preconditions_checked",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
    *REQUIRED_TOP_LEVEL_FIELDS,
)

COMPLETE_VERDICT = "complete_467_archived_468_activated_exact_verifier_pivot_carried_forward"
TERMINAL_PREFIXES = ("complete_", "success_", "passed_", "shipped_", "blocked_")

BLOCKER_IDS = [
    "clean_kan_pwa_milp_small_property_scale_boundary",
    "flagged_endpoint_live_runtime_claim",
    "blocked_uprm_logprob_cache",
    "non_winning_process_verifier",
    "flagged_static_csr_toy_result",
    "pbit_cdcl_no_effort_win",
    "governed_fr11_no_promote",
    "hardware_continuity_without_speedup",
]


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _bool(value: Any) -> bool:
    return value is True


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nonzero_promoted(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0.0


def read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    if not path.exists():
        return {}, {"exists": False, "loadable": False, "error": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, {"exists": True, "loadable": False, "error": str(exc)}
    if not isinstance(payload, Mapping):
        return {}, {"exists": True, "loadable": False, "error": "json_not_object"}
    return dict(payload), {"exists": True, "loadable": True, "sha256": file_sha256(path)}


def _parse_yaml_status(
    root: Path, rel_path: Path, *, absent_status: str
) -> tuple[JsonDict, JsonDict]:
    path = root / rel_path
    status: JsonDict = {"path": str(rel_path), "exists": path.exists()}
    if not path.exists():
        status.update({"parse_ok": None, "status": absent_status})
        return {}, status
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        status.update({"parse_ok": False, "error": str(exc)})
        return {}, status
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        status.update({"parse_ok": False, "error": "yaml_not_mapping"})
        return {}, status
    status.update(
        {
            "parse_ok": True,
            "status": "parsed",
            "milestone": str(payload.get("milestone", "")),
            "milestone_doc": str(payload.get("milestone_doc", "")),
            "sha256": file_sha256(path),
        }
    )
    return dict(payload), status


def check_roadmaps(root: Path) -> tuple[JsonDict, JsonDict]:
    active, active_status = _parse_yaml_status(
        root,
        ROADMAP_ACTIVE_REL_PATH,
        absent_status="missing_active_roadmap",
    )
    staged, staged_status = _parse_yaml_status(
        root,
        ROADMAP_NEXT_REL_PATH,
        absent_status="absent_already_promoted",
    )
    return (
        {"active": active, "pre_staged": staged},
        {"active": active_status, "pre_staged": staged_status},
    )


def roadmap_blocker(roadmaps_checked: JsonMap) -> str:
    active = _mapping(roadmaps_checked.get("active"))
    staged = _mapping(roadmaps_checked.get("pre_staged"))
    if active.get("parse_ok") is not True or staged.get("parse_ok") is False:
        return "blocked_yaml_parse"
    if str(active.get("milestone", "")) != MILESTONE_TO:
        return "blocked_active_milestone_mismatch"
    return ""


def _artifact_row(exp_id: int, rel_path: Path, payload: JsonMap, status: JsonMap) -> JsonDict:
    verdict = str(payload.get("honest_verdict", ""))
    blocked = verdict.startswith("blocked_") or payload.get("status") == "blocked"
    row: JsonDict = {
        "label": SOURCE_LABELS.get(exp_id, f"experiment_{exp_id}"),
        "experiment_id": exp_id,
        "path": str(rel_path),
        "present": status.get("loadable") is True,
        "exists": status.get("exists") is True,
        "loadable": status.get("loadable") is True,
        "fields_imported": list(IMPORTED_FIELDS.get(exp_id, ("honest_verdict",))),
        "honest_verdict": verdict,
        "flagged_adversarial": payload.get("flagged_adversarial") is True,
        "blocked": blocked,
    }
    if "sha256" in status:
        row["sha256"] = status["sha256"]
    if "error" in status:
        row["error"] = status["error"]
    duration = _number(payload.get("duration_s"))
    if duration is not None:
        row["duration_s"] = duration
    gate_summary = payload.get("gate_check_summary")
    if isinstance(gate_summary, str):
        row["gate_check_summary"] = gate_summary
    return row


def _clean_positive_artifact(exp_id: int, payload: JsonMap, flagged: bool) -> bool:
    if flagged:
        return False
    if exp_id == 5091:
        return (
            _bool(payload.get("property_holds"))
            and str(payload.get("property_status", "")) == "verified"
            and _bool(payload.get("solver_available"))
        )
    return False


def load_v467_artifacts(
    root: Path,
) -> tuple[
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    list[JsonDict],
    dict[int, JsonDict],
]:
    source_rows: list[JsonDict] = []
    blocked_rows: list[JsonDict] = []
    missing_rows: list[JsonDict] = []
    flagged_rows: list[JsonDict] = []
    clean_positive_rows: list[JsonDict] = []
    payloads: dict[int, JsonDict] = {}

    for exp_id, rel_path in SOURCE_ARTIFACTS.items():
        payload, status = read_json_mapping(root / rel_path)
        row = _artifact_row(exp_id, rel_path, payload, status)
        if row["present"]:
            source_rows.append(row)
            payloads[exp_id] = payload
            if row["blocked"]:
                blocked_rows.append({**row, "blocker_reason": "blocked_verdict_or_status"})
            if row["flagged_adversarial"]:
                flagged_rows.append({**row, "excluded_from_headline": True})
            if _clean_positive_artifact(exp_id, payload, row["flagged_adversarial"]):
                clean_positive_rows.append({**row, "positive_reason": "clean_exact_verifier_pivot"})
        else:
            missing_rows.append({**row, "status": "missing_or_unloadable_source"})

    for exp_id, meta in PROMPT_LISTED_MISSING_ARTIFACTS.items():
        rel_path = Path(meta["path"])
        payload, status = read_json_mapping(root / rel_path)
        row = _artifact_row(exp_id, rel_path, payload, status)
        row["label"] = "temporal_consistency_process_verifier_prompt_listed_path"
        if row["present"]:
            source_rows.append({**row, "status": "prompt_listed_path_present"})
        else:
            missing_rows.append(
                {
                    "experiment_id": exp_id,
                    "label": row["label"],
                    "path": meta["path"],
                    "exists": status.get("exists") is True,
                    "loadable": status.get("loadable") is True,
                    "status": "prompt_listed_absent",
                    "reason": meta["reason"],
                }
            )

    return source_rows, blocked_rows, missing_rows, flagged_rows, clean_positive_rows, payloads


def _payload(payloads: Mapping[int, JsonDict], exp_id: int) -> JsonDict:
    return dict(payloads.get(exp_id, {}))


def build_close_state(payloads: Mapping[int, JsonDict], flagged_ids: set[int]) -> JsonDict:
    exp5085 = _payload(payloads, 5085)
    exp5086 = _payload(payloads, 5086)
    exp5087 = _payload(payloads, 5087)
    exp5088 = _payload(payloads, 5088)
    exp5089 = _payload(payloads, 5089)
    exp5090 = _payload(payloads, 5090)
    exp5091 = _payload(payloads, 5091)
    exp5092 = _payload(payloads, 5092)
    exp5093 = _payload(payloads, 5093)
    exp5094 = _payload(payloads, 5094)
    promotion = _mapping(exp5092.get("promotion_decision"))
    gates = _mapping(promotion.get("gate_conditions"))

    process_retry_blocked = (
        str(exp5087.get("honest_verdict", "")).startswith("blocked_")
        or exp5087.get("status") == "blocked"
    )
    pbit_effort_win = _bool(exp5089.get("helps_declared_family"))
    fr11_promoted = _bool(promotion.get("promoted")) or _nonzero_promoted(
        exp5092.get("promoted_count")
    )
    hardware_no_destructive = _list(exp5093.get("destructive_actions_taken")) == []
    hardware_any_board = _bool(exp5093.get("kv260_ssh_ready")) or _bool(
        exp5093.get("polarfire_detected")
    )
    hardware_no_speedup = exp5093.get("kv260_speedup_claim_allowed") is False

    return {
        "transition_record_only": True,
        "scientific_decision_completed": True,
        "capstone_state": {
            "source_experiment_id": 5094,
            "honest_verdict": str(exp5094.get("honest_verdict", "")),
            "milestone_decision": str(exp5094.get("milestone_decision", "")),
            "exact_verifier_pivot_positive": (
                str(exp5094.get("milestone_decision", "")) == "exact_verifier_pivot_positive"
            ),
        },
        "runtime_state": {
            "source_experiment_id": 5085,
            "honest_verdict": str(exp5085.get("honest_verdict", "")),
            "endpoint_live_runtime_claim_flagged": 5085 in flagged_ids,
            "reported_completion_endpoint_ready": _bool(exp5085.get("completion_endpoint_ready")),
            "reported_logprob_endpoint_ready": _bool(exp5085.get("logprob_endpoint_ready")),
            "reported_top_logprob_or_confidence_ready": _bool(
                exp5085.get("top_logprob_or_confidence_ready")
            ),
            "reported_live_completion_invoked": _bool(exp5085.get("live_completion_invoked")),
            "reported_usable_sota_model_count": len(_list(exp5085.get("usable_sota_models"))),
            "headline_runtime_ready": False,
        },
        "uprm_cache_state": {
            "source_experiment_id": 5086,
            "honest_verdict": str(exp5086.get("honest_verdict", "")),
            "blocked": str(exp5086.get("honest_verdict", "")).startswith("blocked_"),
            "logprob_cache_ready": _bool(exp5086.get("logprob_cache_ready")),
            "step_cache_ready": _bool(exp5086.get("step_cache_ready")),
            "endpoint_used_recorded": bool(str(exp5086.get("endpoint_used", ""))),
        },
        "process_verifier_state": {
            "source_experiment_ids": [5087, 5088],
            "uprm_process_retry_blocked": process_retry_blocked,
            "temporal_fallback_flagged": 5088 in flagged_ids,
            "temporal_fallback_reported_win": _bool(exp5088.get("beats_one_pass")),
            "delta_vs_one_pass": _number(exp5088.get("delta_vs_one_pass")),
            "process_verifier_ready": False,
            "process_verifier_win": False,
            "gate_check_summary": str(exp5087.get("gate_check_summary", "")),
        },
        "kan_state": {
            "source_experiment_id": 5091,
            "clean_positive": _clean_positive_artifact(5091, exp5091, 5091 in flagged_ids),
            "property_holds": _bool(exp5091.get("property_holds")),
            "property_status": str(exp5091.get("property_status", "")),
            "solver_available": _bool(exp5091.get("solver_available")),
            "solver_status": str(exp5091.get("solver_status", "")),
            "binary_variable_count": _number(exp5091.get("binary_variable_count")),
            "pwa_piece_count": _number(exp5091.get("pwa_piece_count")),
            "constraint_count": _number(exp5091.get("constraint_count")),
            "global_error_bound": _number(exp5091.get("global_error_bound")),
            "scale_boundary": "small_multi_unit_property_not_architecture_scale_claim",
        },
        "static_csr_state": {
            "source_experiment_id": 5090,
            "flagged_toy_result": 5090 in flagged_ids,
            "headline_allowed": False,
            "reported_beats_cpu_trie": _bool(exp5090.get("beats_cpu_trie")),
            "reported_beats_rerank_only_on_validity_or_cost": _bool(
                exp5090.get("beats_rerank_only_on_validity_or_cost")
            ),
            "reported_mask_speedup": _number(exp5090.get("mask_speedup")),
            "reported_validity_rate": _number(exp5090.get("validity_rate")),
            "reported_rerank_only_validity_rate": _number(exp5090.get("rerank_only_validity_rate")),
        },
        "pbit_cdcl_state": {
            "source_experiment_id": 5089,
            "flagged": 5089 in flagged_ids,
            "correctness_preserved": _bool(exp5089.get("correctness_preserved")),
            "helps_declared_family": _bool(exp5089.get("helps_declared_family")),
            "effort_win": pbit_effort_win,
            "delta_effort_vs_pure": _mapping(exp5089.get("delta_effort_vs_pure")),
            "headline_allowed": False,
        },
        "fr11_state": {
            "source_experiment_id": 5092,
            "safe_governed_mechanism": (
                5092 not in flagged_ids
                and _bool(exp5092.get("fr11_attempt_completed"))
                and exp5092.get("heldout_delta") == 0.0
                and exp5092.get("nonforgetting_delta") == 0.0
                and _bool(exp5092.get("contamination_guard_passed"))
                and _bool(exp5092.get("poison_guard_passed"))
                and _bool(exp5092.get("rollback_guard_passed"))
            ),
            "heldout_delta": _number(exp5092.get("heldout_delta")),
            "nonforgetting_delta": _number(exp5092.get("nonforgetting_delta")),
            "contamination_guard_passed": _bool(exp5092.get("contamination_guard_passed")),
            "poison_guard_passed": _bool(exp5092.get("poison_guard_passed")),
            "rollback_guard_passed": _bool(exp5092.get("rollback_guard_passed")),
            "promoted": fr11_promoted,
            "positive_utility_observed": _bool(gates.get("positive_utility_gt_zero")),
            "no_promote_reason": str(promotion.get("no_promote_reason", "")),
        },
        "hardware_state": {
            "source_experiment_id": 5093,
            "kv260_ssh_ready": _bool(exp5093.get("kv260_ssh_ready")),
            "kv260_uio_transcript_path": exp5093.get("kv260_uio_transcript_path"),
            "speedup_claim_allowed": _bool(exp5093.get("kv260_speedup_claim_allowed")),
            "gatemate_detected": _bool(exp5093.get("gatemate_detected")),
            "gatemate_terminal_state": str(exp5093.get("gatemate_terminal_state", "")),
            "polarfire_detected": _bool(exp5093.get("polarfire_detected")),
            "polarfire_dispatch_precheck_ready": _bool(
                exp5093.get("polarfire_dispatch_precheck_ready")
            ),
            "destructive_actions_taken": _list(exp5093.get("destructive_actions_taken")),
            "clean_continuity_without_speedup": (
                5093 not in flagged_ids
                and hardware_no_destructive
                and hardware_no_speedup
                and hardware_any_board
            ),
        },
    }


def build_exact_verifier_pivot(close_state: JsonMap) -> JsonDict:
    kan = _mapping(close_state.get("kan_state"))
    capstone = _mapping(close_state.get("capstone_state"))
    if kan.get("clean_positive") is not True:
        return {}
    return {
        "pivot_decision": str(capstone.get("milestone_decision", "")),
        "driver_experiment_id": 5091,
        "driver_artifact": str(SOURCE_ARTIFACTS[5091]),
        "clean_positive": True,
        "property_holds": kan.get("property_holds"),
        "property_status": kan.get("property_status"),
        "solver_status": kan.get("solver_status"),
        "binary_variable_count": kan.get("binary_variable_count"),
        "pwa_piece_count": kan.get("pwa_piece_count"),
        "constraint_count": kan.get("constraint_count"),
        "global_error_bound": kan.get("global_error_bound"),
        "scale_boundary": kan.get("scale_boundary"),
        "carried_to_milestone": MILESTONE_TO,
    }


def blockers_carried_forward(close_state: JsonMap) -> list[JsonDict]:
    runtime = _mapping(close_state.get("runtime_state"))
    cache = _mapping(close_state.get("uprm_cache_state"))
    process = _mapping(close_state.get("process_verifier_state"))
    kan = _mapping(close_state.get("kan_state"))
    static = _mapping(close_state.get("static_csr_state"))
    pbit = _mapping(close_state.get("pbit_cdcl_state"))
    fr11 = _mapping(close_state.get("fr11_state"))
    hardware = _mapping(close_state.get("hardware_state"))

    rows = [
        {
            "blocker_id": "clean_kan_pwa_milp_small_property_scale_boundary",
            "label": "clean KAN/PWA/MILP proof with small-property scale boundary",
            "source_experiment_id": 5091,
            "observed_state": {
                "clean_positive": kan.get("clean_positive"),
                "property_holds": kan.get("property_holds"),
                "solver_status": kan.get("solver_status"),
                "binary_variable_count": kan.get("binary_variable_count"),
                "constraint_count": kan.get("constraint_count"),
                "scale_boundary": kan.get("scale_boundary"),
            },
            "principle": "This is the positive path to scale, not a broad architecture-scale proof.",
        },
        {
            "blocker_id": "flagged_endpoint_live_runtime_claim",
            "label": "flagged endpoint/live-runtime claim",
            "source_experiment_id": 5085,
            "observed_state": {
                "endpoint_live_runtime_claim_flagged": runtime.get(
                    "endpoint_live_runtime_claim_flagged"
                ),
                "reported_completion_endpoint_ready": runtime.get(
                    "reported_completion_endpoint_ready"
                ),
                "reported_logprob_endpoint_ready": runtime.get("reported_logprob_endpoint_ready"),
                "reported_top_logprob_or_confidence_ready": runtime.get(
                    "reported_top_logprob_or_confidence_ready"
                ),
                "headline_runtime_ready": runtime.get("headline_runtime_ready"),
            },
            "principle": "A flagged endpoint artifact cannot make local runtime a clean substrate.",
        },
        {
            "blocker_id": "blocked_uprm_logprob_cache",
            "label": "blocked uPRM/logprob cache",
            "source_experiment_id": 5086,
            "observed_state": {
                "blocked": cache.get("blocked"),
                "logprob_cache_ready": cache.get("logprob_cache_ready"),
                "step_cache_ready": cache.get("step_cache_ready"),
            },
            "principle": "uPRM work remains blocked until the cache substrate is real.",
        },
        {
            "blocker_id": "non_winning_process_verifier",
            "label": "non-winning process verifier",
            "source_experiment_ids": [5087, 5088],
            "observed_state": {
                "uprm_process_retry_blocked": process.get("uprm_process_retry_blocked"),
                "temporal_fallback_flagged": process.get("temporal_fallback_flagged"),
                "temporal_fallback_reported_win": process.get("temporal_fallback_reported_win"),
                "process_verifier_win": process.get("process_verifier_win"),
            },
            "principle": "Blocked or non-winning process evidence cannot be headlined.",
        },
        {
            "blocker_id": "flagged_static_csr_toy_result",
            "label": "flagged STATIC CSR toy result",
            "source_experiment_id": 5090,
            "observed_state": {
                "flagged_toy_result": static.get("flagged_toy_result"),
                "reported_mask_speedup": static.get("reported_mask_speedup"),
                "reported_validity_rate": static.get("reported_validity_rate"),
                "headline_allowed": static.get("headline_allowed"),
            },
            "principle": "Syntax-mask toy wins need semantic controls before any headline.",
        },
        {
            "blocker_id": "pbit_cdcl_no_effort_win",
            "label": "p-bit/CDCL no effort win",
            "source_experiment_id": 5089,
            "observed_state": {
                "correctness_preserved": pbit.get("correctness_preserved"),
                "helps_declared_family": pbit.get("helps_declared_family"),
                "effort_win": pbit.get("effort_win"),
                "delta_effort_vs_pure": pbit.get("delta_effort_vs_pure"),
            },
            "principle": "Correctness preservation without effort reduction is not a solver win.",
        },
        {
            "blocker_id": "governed_fr11_no_promote",
            "label": "governed FR-11 no-promote",
            "source_experiment_id": 5092,
            "observed_state": {
                "safe_governed_mechanism": fr11.get("safe_governed_mechanism"),
                "heldout_delta": fr11.get("heldout_delta"),
                "nonforgetting_delta": fr11.get("nonforgetting_delta"),
                "promoted": fr11.get("promoted"),
                "positive_utility_observed": fr11.get("positive_utility_observed"),
            },
            "principle": "Safe no-promote governance is useful, but it is not self-learning progress.",
        },
        {
            "blocker_id": "hardware_continuity_without_speedup",
            "label": "hardware continuity without speedup",
            "source_experiment_id": 5093,
            "observed_state": {
                "kv260_ssh_ready": hardware.get("kv260_ssh_ready"),
                "polarfire_detected": hardware.get("polarfire_detected"),
                "gatemate_detected": hardware.get("gatemate_detected"),
                "speedup_claim_allowed": hardware.get("speedup_claim_allowed"),
                "clean_continuity_without_speedup": hardware.get(
                    "clean_continuity_without_speedup"
                ),
            },
            "principle": "Continuity evidence keeps boards alive; it is not acceleration evidence.",
        },
    ]
    for row in rows:
        row["must_not_be_laundered_into_success"] = True
    return rows


def _next_milestone_doc(roadmaps: JsonMap, roadmaps_checked: JsonMap) -> str:
    active = _mapping(roadmaps.get("active"))
    active_status = _mapping(roadmaps_checked.get("active"))
    return str(
        active.get("milestone_doc") or active_status.get("milestone_doc") or NEXT_MILESTONE_DOC
    )


def build_artifact(
    *,
    honest_verdict: str,
    roadmaps: JsonMap,
    roadmaps_checked: JsonMap,
    source_artifacts: list[JsonDict],
    blocked_artifacts: list[JsonDict],
    missing_artifacts: list[JsonDict],
    flagged_artifacts: list[JsonDict],
    clean_positive_artifacts: list[JsonDict],
    payloads: Mapping[int, JsonDict],
    duration_s: float,
) -> JsonDict:
    complete = honest_verdict == COMPLETE_VERDICT
    flagged_ids = {int(row["experiment_id"]) for row in flagged_artifacts}
    close_state = build_close_state(payloads, flagged_ids) if complete else {}
    blockers = blockers_carried_forward(close_state) if complete else []
    exact_pivot = build_exact_verifier_pivot(close_state) if complete else {}
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(RESULT_RELATIVE_PATH),
        "honest_verdict": honest_verdict,
        "duration_s": round(max(float(duration_s), 0.0001), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "milestone_from": MILESTONE_FROM,
        "milestone_to": MILESTONE_TO,
        "source_artifacts": source_artifacts if complete else [],
        "blocked_artifacts": blocked_artifacts if complete else [],
        "flagged_artifacts": flagged_artifacts if complete else [],
        "clean_positive_artifacts": clean_positive_artifacts if complete else [],
        "missing_artifacts": missing_artifacts if complete else [],
        "close_state": close_state,
        "blockers_carried_forward": blockers,
        "exact_verifier_pivot": exact_pivot,
        "next_milestone_doc": _next_milestone_doc(roadmaps, roadmaps_checked),
        "docs_updated": [],
        "flagged_adversarial": False,
        "preconditions_checked": {
            "roadmaps": dict(roadmaps_checked),
            "expected_artifact_count": len(SOURCE_ARTIFACTS),
            "loaded_artifact_count": len(source_artifacts) if complete else 0,
            "missing_artifact_count": len(missing_artifacts) if complete else 0,
            "blocked_artifact_count": len(blocked_artifacts) if complete else 0,
            "flagged_artifact_count": len(flagged_artifacts) if complete else 0,
            "clean_positive_artifact_count": len(clean_positive_artifacts) if complete else 0,
            "headline_inputs_exclude_flagged": True,
        },
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run(
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    clock: Clock = time.perf_counter,
) -> JsonDict:
    started = clock()
    root = Path(root)
    roadmaps, roadmaps_checked = check_roadmaps(root)
    source_artifacts: list[JsonDict] = []
    blocked_artifacts: list[JsonDict] = []
    missing_artifacts: list[JsonDict] = []
    flagged_artifacts: list[JsonDict] = []
    clean_positive_artifacts: list[JsonDict] = []
    payloads: dict[int, JsonDict] = {}

    blocker = roadmap_blocker(roadmaps_checked)
    if not blocker:
        (
            source_artifacts,
            blocked_artifacts,
            missing_artifacts,
            flagged_artifacts,
            clean_positive_artifacts,
            payloads,
        ) = load_v467_artifacts(root)

    artifact = build_artifact(
        honest_verdict=blocker or COMPLETE_VERDICT,
        roadmaps=roadmaps,
        roadmaps_checked=roadmaps_checked,
        source_artifacts=source_artifacts,
        blocked_artifacts=blocked_artifacts,
        missing_artifacts=missing_artifacts,
        flagged_artifacts=flagged_artifacts,
        clean_positive_artifacts=clean_positive_artifacts,
        payloads=payloads,
        duration_s=clock() - started,
    )
    write_json(artifact_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_SCHEMA_FIELDS:
        if field not in artifact:
            errors.append(f"missing.{field}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict.not_terminal")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate.not_aggregation")
    if artifact.get("milestone_from") != MILESTONE_FROM:
        errors.append("milestone_from.invalid")
    if artifact.get("milestone_to") != MILESTONE_TO:
        errors.append("milestone_to.invalid")
    if artifact.get("next_milestone_doc") != NEXT_MILESTONE_DOC:
        errors.append("next_milestone_doc.invalid")
    if artifact.get("docs_updated") != []:
        errors.append("docs_updated.not_deferred")
    if artifact.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial.must_be_false")
    if "live_llm_inference" in json.dumps(artifact, sort_keys=True, default=str):
        errors.append("forbidden.live_llm_inference_claim")
    if not verdict.startswith("blocked_"):
        blocker_ids = [
            str(row.get("blocker_id"))
            for row in _list(artifact.get("blockers_carried_forward"))
            if isinstance(row, Mapping)
        ]
        if blocker_ids != BLOCKER_IDS:
            errors.append("blockers_carried_forward.invalid")
        pivot = _mapping(artifact.get("exact_verifier_pivot"))
        if pivot.get("driver_experiment_id") != 5091 or pivot.get("clean_positive") is not True:
            errors.append("exact_verifier_pivot.invalid")
        close_state = _mapping(artifact.get("close_state"))
        capstone = _mapping(close_state.get("capstone_state"))
        if capstone.get("milestone_decision") != "exact_verifier_pivot_positive":
            errors.append("close_state.invalid")
    checksum = str(artifact.get("reproducibility_checksum", ""))
    if (
        not checksum.startswith("sha256:")
        or len(checksum) != 71
        or checksum != payload_checksum(artifact)
    ):
        errors.append("reproducibility_checksum.invalid")
    return sorted(set(errors))


def main(
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    clock: Clock = time.perf_counter,
) -> int:
    artifact = run(root=root, artifact_path=artifact_path, clock=clock)
    errors = artifact_schema_errors(artifact)
    if errors:
        print(json.dumps({"schema_errors": errors}, indent=2, sort_keys=True))
        return 1
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - direct experiment entrypoint
    raise SystemExit(main())

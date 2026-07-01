#!/usr/bin/env python3
"""Experiment 5083: archive .466 truth and activate the .467 record.

Spec refs: REQ-CAPSTONE-5083, SCENARIO-CAPSTONE-5083,
SCENARIO-CAPSTONE-5083-BLOCKED-YAML,
SCENARIO-CAPSTONE-5083-FIELD-PRINCIPLES.

This module writes a transition truth-record. It reads roadmap YAML plus the
prior milestone's JSON artifacts and records what happened. It does not run a
model, and it does not convert blocked, skipped, tiny-only, or no-speedup
evidence into a scientific success claim.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5083_archive_466_activate_467"
EXPERIMENT_ID = 5083
SCHEMA = "carnot.experiment_5083_archive_466_activate_467.v1"
RESULT_RELATIVE_PATH = "results/experiment_5083_archive_466_activate_467.json"
RANDOM_SEED = 5083
MILESTONE_FROM = "2026.07.466"
MILESTONE_TO = "2026.07.467"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
NEXT_MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"

ROADMAP_ACTIVE_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")

SOURCE_ARTIFACTS: dict[int, Path] = {
    5071: Path("results/experiment_5071_gguf_logprob_preflight_v466.json"),
    5075: Path("results/experiment_5075_dccd_guided_decoding_scale_v466.json"),
    5076: Path("results/experiment_5076_d6_efficiency_replication_v466.json"),
    5077: Path("results/experiment_5077_fr11_group_sc_memory_v466.json"),
    5079: Path("results/experiment_5079_board_continuity_matrix_v466.json"),
    5080: Path("results/experiment_5080_kan_pwa_milp_bridge_v466.json"),
    5082: Path("results/experiment_5082_capstone_v466.json"),
}

BLOCKED_ARTIFACTS: dict[int, Path] = {
    5072: Path("results/experiment_5072_uprm_logprob_cache_v466.json"),
    5082: SOURCE_ARTIFACTS[5082],
}

SKIPPED_OR_MISSING_ARTIFACTS: dict[int, dict[str, str]] = {
    5073: {
        "path": "results/experiment_5073_uprm_process_verifier_v466.json",
        "reason": "preemptive_skip_upstream_exp5072_retired_no_artifact",
    },
    5074: {
        "path": "results/experiment_5074_vpr_tool_process_reward_v466.json",
        "reason": "preemptive_skip_upstream_exp5072_retired_no_artifact",
    },
    5081: {
        "path": "results/experiment_5081_phase_d_fr11_decision_gate_v466.json",
        "reason": "preemptive_skip_upstream_exp5073_exp5074_retired_no_artifact",
    },
}

SPEC_REFS = [
    "REQ-CAPSTONE-5083",
    "SCENARIO-CAPSTONE-5083",
    "SCENARIO-CAPSTONE-5083-BLOCKED-YAML",
    "SCENARIO-CAPSTONE-5083-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_466_archived_467_activated_endpoint_blockers_"
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
            "aggregation_from_upstream_artifacts -- reads roadmap and upstream JSON "
            "only; never live model inference."
        )
    },
    "milestone_from": {
        "principle": "the archived milestone 2026.07.466 from the prior conductor pass.",
    },
    "milestone_to": {
        "principle": "the activated/staged milestone 2026.07.467 from the active roadmap.",
    },
    "source_artifacts": {
        "principle": "the present .466 artifacts loaded with sha256 provenance.",
    },
    "blocked_artifacts": {
        "principle": (
            "gate-blocked artifacts such as Exp5072 and Exp5082 recorded as blocked, "
            "not missing or successful."
        )
    },
    "missing_artifacts": {
        "principle": (
            "skipped or absent .466 artifacts such as Exp5073, Exp5074, and Exp5081 "
            "recorded explicitly."
        )
    },
    "close_state": {
        "principle": (
            "the blunt .466 close-state preserved without promoting skipped, blocked, "
            "tiny-only, or no-speedup evidence."
        )
    },
    "blockers_carried_forward": {
        "principle": (
            "the exact endpoint/uPRM/VPR/DCCD/D6/FR-11/hardware/KAN/capstone blockers "
            "that must constrain .467 planning."
        )
    },
    "next_milestone_doc": {
        "principle": "the .467 milestone document path read from the active roadmap.",
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

REQUIRED_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "duration_s",
    "inference_substrate",
    "milestone_from",
    "milestone_to",
    "source_artifacts",
    "blocked_artifacts",
    "missing_artifacts",
    "close_state",
    "blockers_carried_forward",
    "next_milestone_doc",
    "docs_updated",
    "flagged_adversarial",
    "preconditions_checked",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
)

COMPLETE_VERDICT = "complete_466_archived_467_activated_endpoint_blockers_carried_forward"
TERMINAL_PREFIXES = ("complete_", "success_", "passed_", "shipped_", "blocked_")

BLOCKER_IDS = [
    "missing_live_completion_logprob_endpoint",
    "skipped_uprm_vpr",
    "dccd_worse_than_rerank_only",
    "d6_no_pareto_win",
    "fr11_no_promote_rollback",
    "hardware_no_speedup_claim",
    "kan_tiny_only_proof",
    "blocked_decision_capstone",
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


def _parse_yaml_status(root: Path, rel_path: Path, *, absent_status: str) -> tuple[JsonDict, JsonDict]:
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
    row: JsonDict = {"experiment_id": exp_id, "path": str(rel_path), **dict(status)}
    row["present"] = status.get("exists") is True and status.get("loadable") is True
    if payload:
        row.update(
            {
                "honest_verdict": str(payload.get("honest_verdict", "")),
                "duration_s": _number(payload.get("duration_s")),
                "flagged_adversarial": payload.get("flagged_adversarial") is True,
            }
        )
        gate_summary = payload.get("gate_check_summary")
        if isinstance(gate_summary, str):
            row["gate_check_summary"] = gate_summary
    return row


def load_v466_artifacts(root: Path) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], dict[int, JsonDict]]:
    source_rows: list[JsonDict] = []
    blocked_rows: list[JsonDict] = []
    missing_rows: list[JsonDict] = []
    payloads: dict[int, JsonDict] = {}

    for exp_id, rel_path in SOURCE_ARTIFACTS.items():
        payload, status = read_json_mapping(root / rel_path)
        row = _artifact_row(exp_id, rel_path, payload, status)
        if row["present"]:
            source_rows.append(row)
            payloads[exp_id] = payload
        else:
            missing_rows.append({**row, "status": "missing_or_unloadable_source"})

    for exp_id, rel_path in BLOCKED_ARTIFACTS.items():
        payload = payloads.get(exp_id)
        status: JsonDict
        if payload is None:
            payload, status = read_json_mapping(root / rel_path)
            if status.get("loadable") is True:
                payloads[exp_id] = payload
        else:
            status = {"exists": True, "loadable": True, "sha256": file_sha256(root / rel_path)}
        row = _artifact_row(exp_id, rel_path, payload, status)
        verdict = str(payload.get("honest_verdict", ""))
        if row["present"] and (verdict.startswith("blocked_") or payload.get("status") == "blocked"):
            row["status"] = "gate_blocked"
            blocked_rows.append(row)
        elif not row["present"]:
            missing_rows.append({**row, "status": "missing_gate_block_artifact"})

    for exp_id, meta in SKIPPED_OR_MISSING_ARTIFACTS.items():
        rel_path = Path(meta["path"])
        payload, status = read_json_mapping(root / rel_path)
        row = _artifact_row(exp_id, rel_path, payload, status)
        if row["present"]:
            source_rows.append(row)
            payloads[exp_id] = payload
        else:
            missing_rows.append(
                {
                    "experiment_id": exp_id,
                    "path": meta["path"],
                    "exists": status.get("exists") is True,
                    "loadable": status.get("loadable") is True,
                    "status": "skipped_or_absent",
                    "reason": meta["reason"],
                }
            )

    return source_rows, blocked_rows, missing_rows, payloads


def _payload(payloads: Mapping[int, JsonDict], exp_id: int) -> JsonDict:
    return dict(payloads.get(exp_id, {}))


def build_close_state(payloads: Mapping[int, JsonDict], blocked_artifacts: list[JsonDict]) -> JsonDict:
    exp5071 = _payload(payloads, 5071)
    exp5072 = _payload(payloads, 5072)
    exp5075 = _payload(payloads, 5075)
    exp5076 = _payload(payloads, 5076)
    exp5077 = _payload(payloads, 5077)
    exp5079 = _payload(payloads, 5079)
    exp5080 = _payload(payloads, 5080)
    exp5082 = _payload(payloads, 5082)
    promotion = _mapping(exp5077.get("promotion_decision"))

    return {
        "transition_record_only": True,
        "scientific_decision_completed": False,
        "endpoint_state": {
            "source_experiment_id": 5071,
            "honest_verdict": str(exp5071.get("honest_verdict", "")),
            "completion_endpoint_ready": _bool(exp5071.get("completion_endpoint_ready")),
            "logprob_endpoint_ready": _bool(exp5071.get("logprob_endpoint_ready")),
            "top_logprob_or_confidence_ready": _bool(exp5071.get("top_logprob_or_confidence_ready")),
            "live_completion_invoked": _bool(exp5071.get("live_completion_invoked")),
            "usable_sota_model_count": len(_list(exp5071.get("usable_sota_models"))),
        },
        "uprm_vpr_state": {
            "uprm_cache_blocked": str(exp5072.get("honest_verdict", "")).startswith("blocked_"),
            "uprm_cache_gate_summary": str(exp5072.get("gate_check_summary", "")),
            "uprm_process_skipped": True,
            "vpr_skipped": True,
        },
        "dccd_state": {
            "honest_verdict": str(exp5075.get("honest_verdict", "")),
            "n_questions": _number(exp5075.get("n_questions")),
            "dccd_accuracy": _number(exp5075.get("dccd_accuracy")),
            "unguided_accuracy": _number(exp5075.get("unguided_accuracy")),
            "rerank_only_accuracy": _number(exp5075.get("rerank_only_accuracy")),
            "delta_dccd_vs_rerank": _number(exp5075.get("delta_dccd_vs_rerank")),
            "beats_rerank_only": _bool(exp5075.get("beats_rerank_only")),
            "ci95_delta": _list(exp5075.get("ci95_delta")),
        },
        "d6_state": {
            "honest_verdict": str(exp5076.get("honest_verdict", "")),
            "cascade_accuracy": _number(exp5076.get("cascade_accuracy")),
            "judge_only_accuracy": _number(exp5076.get("judge_only_accuracy")),
            "delta_vs_judge_only": _number(exp5076.get("delta_vs_judge_only")),
            "ci95_delta": _list(exp5076.get("ci95_delta")),
            "efficiency_win": _bool(exp5076.get("efficiency_win")),
            "accuracy_headline_allowed": _bool(exp5076.get("accuracy_headline_allowed")),
            "judge_call_fraction": _number(exp5076.get("judge_call_fraction")),
        },
        "fr11_state": {
            "honest_verdict": str(exp5077.get("honest_verdict", "")),
            "attempt_completed": _bool(exp5077.get("fr11_attempt_completed")),
            "promoted": _bool(promotion.get("promoted")) or _number(exp5077.get("promoted_count")) not in (None, 0.0),
            "promoted_count": _number(exp5077.get("promoted_count")),
            "quarantined_count": _number(exp5077.get("quarantined_count")),
            "heldout_delta": _number(exp5077.get("heldout_delta")),
            "nonforgetting_delta": _number(exp5077.get("nonforgetting_delta")),
            "no_promote_reason": str(promotion.get("no_promote_reason", "")),
            "rollback_guard_passed": _bool(exp5077.get("rollback_guard_passed")),
        },
        "hardware_state": {
            "honest_verdict": str(exp5079.get("honest_verdict", "")),
            "kv260_ssh_ready": _bool(exp5079.get("kv260_ssh_ready")),
            "kv260_speedup_claim_allowed": _bool(exp5079.get("kv260_speedup_claim_allowed")),
            "polarfire_detected": _bool(exp5079.get("polarfire_detected")),
            "gatemate_detected": _bool(exp5079.get("gatemate_detected")),
            "gatemate_terminal_state": str(exp5079.get("gatemate_terminal_state", "")),
        },
        "kan_state": {
            "honest_verdict": str(exp5080.get("honest_verdict", "")),
            "milp_solver_available": _bool(exp5080.get("milp_solver_available")),
            "pwa_abstraction_built": _bool(exp5080.get("pwa_abstraction_built")),
            "property_holds": _bool(exp5080.get("property_holds")),
            "property_checked": _bool(exp5080.get("property_checked")),
            "binary_variable_count": _number(exp5080.get("binary_variable_count")),
            "error_bound": _number(exp5080.get("error_bound")),
            "solver_status": str(exp5080.get("solver_status", "")),
            "tiny_only_proof": _bool(exp5080.get("property_holds"))
            and (_number(exp5080.get("binary_variable_count")) or 0.0) <= 3.0,
        },
        "capstone_state": {
            "honest_verdict": str(exp5082.get("honest_verdict", "")),
            "status": str(exp5082.get("status", "")),
            "blocked": str(exp5082.get("honest_verdict", "")).startswith("blocked_"),
            "blocked_at_layer": str(exp5082.get("blocked_at_layer", "")),
            "gate_check_summary": str(exp5082.get("gate_check_summary", "")),
        },
        "blocked_artifact_count": len(blocked_artifacts),
    }


def blockers_carried_forward(close_state: JsonMap) -> list[JsonDict]:
    endpoint = _mapping(close_state.get("endpoint_state"))
    uprm = _mapping(close_state.get("uprm_vpr_state"))
    dccd = _mapping(close_state.get("dccd_state"))
    d6 = _mapping(close_state.get("d6_state"))
    fr11 = _mapping(close_state.get("fr11_state"))
    hardware = _mapping(close_state.get("hardware_state"))
    kan = _mapping(close_state.get("kan_state"))
    capstone = _mapping(close_state.get("capstone_state"))

    rows = [
        {
            "blocker_id": "missing_live_completion_logprob_endpoint",
            "label": "missing live completion/logprob endpoint",
            "source_experiment_id": 5071,
            "observed_state": {
                "completion_endpoint_ready": endpoint.get("completion_endpoint_ready"),
                "logprob_endpoint_ready": endpoint.get("logprob_endpoint_ready"),
                "top_logprob_or_confidence_ready": endpoint.get("top_logprob_or_confidence_ready"),
            },
            "principle": "Cached GGUF files are not a live endpoint or token-logprob substrate.",
        },
        {
            "blocker_id": "skipped_uprm_vpr",
            "label": "skipped uPRM/VPR",
            "source_experiment_ids": [5072, 5073, 5074],
            "observed_state": {
                "uprm_cache_blocked": uprm.get("uprm_cache_blocked"),
                "uprm_process_skipped": uprm.get("uprm_process_skipped"),
                "vpr_skipped": uprm.get("vpr_skipped"),
            },
            "principle": "Gated or skipped process-verifier tasks cannot count as null or positive evidence.",
        },
        {
            "blocker_id": "dccd_worse_than_rerank_only",
            "label": "DCCD worse than rerank-only",
            "source_experiment_id": 5075,
            "observed_state": {
                "dccd_accuracy": dccd.get("dccd_accuracy"),
                "rerank_only_accuracy": dccd.get("rerank_only_accuracy"),
                "delta_dccd_vs_rerank": dccd.get("delta_dccd_vs_rerank"),
                "beats_rerank_only": dccd.get("beats_rerank_only"),
            },
            "principle": "A constrained decoding surface that loses to rerank-only is not a headline win.",
        },
        {
            "blocker_id": "d6_no_pareto_win",
            "label": "D6 no Pareto win",
            "source_experiment_id": 5076,
            "observed_state": {
                "efficiency_win": d6.get("efficiency_win"),
                "accuracy_headline_allowed": d6.get("accuracy_headline_allowed"),
                "ci95_delta": d6.get("ci95_delta"),
            },
            "principle": "A point estimate with a CI touching zero is not a Pareto headline.",
        },
        {
            "blocker_id": "fr11_no_promote_rollback",
            "label": "FR-11 no-promote rollback",
            "source_experiment_id": 5077,
            "observed_state": {
                "promoted": fr11.get("promoted"),
                "heldout_delta": fr11.get("heldout_delta"),
                "nonforgetting_delta": fr11.get("nonforgetting_delta"),
                "rollback_guard_passed": fr11.get("rollback_guard_passed"),
            },
            "principle": "The guard prevented harmful memory promotion; that is not self-learning success.",
        },
        {
            "blocker_id": "hardware_no_speedup_claim",
            "label": "hardware no speedup claim",
            "source_experiment_id": 5079,
            "observed_state": {
                "kv260_ssh_ready": hardware.get("kv260_ssh_ready"),
                "kv260_speedup_claim_allowed": hardware.get("kv260_speedup_claim_allowed"),
                "polarfire_detected": hardware.get("polarfire_detected"),
                "gatemate_detected": hardware.get("gatemate_detected"),
            },
            "principle": "Reachability and continuity evidence must not be promoted into speedup evidence.",
        },
        {
            "blocker_id": "kan_tiny_only_proof",
            "label": "KAN tiny-only proof",
            "source_experiment_id": 5080,
            "observed_state": {
                "property_holds": kan.get("property_holds"),
                "binary_variable_count": kan.get("binary_variable_count"),
                "tiny_only_proof": kan.get("tiny_only_proof"),
            },
            "principle": "A tiny PWA/MILP proof is a foothold, not a scaled KAN verification claim.",
        },
        {
            "blocker_id": "blocked_decision_capstone",
            "label": "blocked decision/capstone",
            "source_experiment_ids": [5081, 5082],
            "observed_state": {
                "capstone_blocked": capstone.get("blocked"),
                "gate_check_summary": capstone.get("gate_check_summary"),
            },
            "principle": "The milestone ended with a blocked decision path, not a completed decision.",
        },
    ]
    for row in rows:
        row["must_not_be_laundered_into_success"] = True
    return rows


def _next_milestone_doc(roadmaps: JsonMap, roadmaps_checked: JsonMap) -> str:
    active = _mapping(roadmaps.get("active"))
    active_status = _mapping(roadmaps_checked.get("active"))
    return str(active.get("milestone_doc") or active_status.get("milestone_doc") or NEXT_MILESTONE_DOC)


def build_artifact(
    *,
    honest_verdict: str,
    roadmaps: JsonMap,
    roadmaps_checked: JsonMap,
    source_artifacts: list[JsonDict],
    blocked_artifacts: list[JsonDict],
    missing_artifacts: list[JsonDict],
    payloads: Mapping[int, JsonDict],
    duration_s: float,
) -> JsonDict:
    complete = honest_verdict == COMPLETE_VERDICT
    close_state = build_close_state(payloads, blocked_artifacts) if complete else {}
    blockers = blockers_carried_forward(close_state) if complete else []
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "duration_s": round(max(float(duration_s), 0.0001), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "milestone_from": MILESTONE_FROM,
        "milestone_to": MILESTONE_TO,
        "source_artifacts": source_artifacts if complete else [],
        "blocked_artifacts": blocked_artifacts if complete else [],
        "missing_artifacts": missing_artifacts if complete else [],
        "close_state": close_state,
        "blockers_carried_forward": blockers,
        "next_milestone_doc": _next_milestone_doc(roadmaps, roadmaps_checked),
        "docs_updated": [],
        "flagged_adversarial": False,
        "preconditions_checked": {"roadmaps": dict(roadmaps_checked)},
        "field_principles": FIELD_PRINCIPLES,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def run(root: Path = REPO_ROOT, artifact_path: Path | None = None) -> JsonDict:
    started = time.perf_counter()
    root = Path(root)
    roadmaps, roadmaps_checked = check_roadmaps(root)
    source_artifacts: list[JsonDict] = []
    blocked_artifacts: list[JsonDict] = []
    missing_artifacts: list[JsonDict] = []
    payloads: dict[int, JsonDict] = {}
    blocker = roadmap_blocker(roadmaps_checked)
    if not blocker:
        source_artifacts, blocked_artifacts, missing_artifacts, payloads = load_v466_artifacts(root)
    artifact = build_artifact(
        honest_verdict=blocker or COMPLETE_VERDICT,
        roadmaps=roadmaps,
        roadmaps_checked=roadmaps_checked,
        source_artifacts=source_artifacts,
        blocked_artifacts=blocked_artifacts,
        missing_artifacts=missing_artifacts,
        payloads=payloads,
        duration_s=time.perf_counter() - started,
    )
    write_json(artifact_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(field)
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("milestone_from") != MILESTONE_FROM:
        errors.append("milestone_from")
    if artifact.get("milestone_to") != MILESTONE_TO:
        errors.append("milestone_to")
    if artifact.get("next_milestone_doc") != NEXT_MILESTONE_DOC:
        errors.append("next_milestone_doc")
    if artifact.get("docs_updated") != []:
        errors.append("docs_updated")
    if artifact.get("flagged_adversarial") is not False:
        errors.append("flagged_adversarial")
    if not verdict.startswith("blocked_"):
        blocker_ids = [
            str(row.get("blocker_id"))
            for row in _list(artifact.get("blockers_carried_forward"))
            if isinstance(row, Mapping)
        ]
        if blocker_ids != BLOCKER_IDS:
            errors.append("blockers_carried_forward")
        close_state = _mapping(artifact.get("close_state"))
        if close_state.get("scientific_decision_completed") is not False:
            errors.append("close_state")
    if "live_llm_inference" in json.dumps(artifact, sort_keys=True, default=str):
        errors.append("inference_substrate")
    checksum = str(artifact.get("reproducibility_checksum", ""))
    if not checksum.startswith("sha256:") or len(checksum) != 71:
        errors.append("reproducibility_checksum")
    return sorted(set(errors))


def main(root: Path = REPO_ROOT, artifact_path: Path | None = None) -> int:
    artifact = run(root=root, artifact_path=artifact_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - direct experiment entrypoint
    raise SystemExit(main())

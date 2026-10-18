#!/usr/bin/env python3
"""Experiment 5069: archive .465 truth and activate the .466 record.

Spec refs: REQ-CAPSTONE-5069, SCENARIO-CAPSTONE-5069,
SCENARIO-CAPSTONE-5069-BLOCKED-YAML,
SCENARIO-CAPSTONE-5069-FIELD-PRINCIPLES.

This module is deliberately a truth-record, not a scientific measurement. It
reads roadmap YAML and upstream JSON artifacts, then writes the blockers that
must constrain .466 planning. It never runs an LLM and never upgrades a missing,
flagged, underpowered, or scoped upstream result into a success claim.
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
EXPERIMENT = "experiment_5069_archive_465_activate_466"
EXPERIMENT_ID = 5069
SCHEMA = "carnot.experiment_5069_archive_465_activate_466.v1"
RESULT_RELATIVE_PATH = "results/experiment_5069_archive_465_activate_466.json"
RANDOM_SEED = 5069
MILESTONE_FROM = "2026.06.465"
MILESTONE_TO = "2026.07.466"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
NEXT_MILESTONE_DOC = "openspec/change-proposals/research-roadmap-vNEXT.md"

ROADMAP_ACTIVE_REL_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_REL_PATH = Path("research-roadmap-next.yaml")
CAPSTONE_REL_PATH = Path("results/experiment_5068_capstone_v465.json")

PHASE_ARTIFACTS: dict[int, Path] = {
    5057: Path("results/experiment_5057_gate_state_preflight_v465.json"),
    5058: Path("results/experiment_5058_sota_candidate_refresh_inwriting.json"),
    5059: Path("results/experiment_5059_d1_sota_refresh_audit.json"),
    5060: Path("results/experiment_5060_second_corpus_audit_v2.json"),
    5061: Path("results/experiment_5061_tool_first_cascade.json"),
    5062: Path("results/experiment_5062_guided_decoding_cost_frontier.json"),
    5063: Path("results/experiment_5063_moat_gate_resolution_v465.json"),
    5064: Path("results/experiment_5064_audited_skillgraph_self_learning.json"),
    5065: Path("results/experiment_5065_kv260_testbench_timing_packet.json"),
    5066: Path("results/experiment_5066_sota_ingestion_v466.json"),
    5067: Path("results/experiment_5067_arc_live_path_self_discovery.json"),
}

SPEC_REFS = [
    "REQ-CAPSTONE-5069",
    "SCENARIO-CAPSTONE-5069",
    "SCENARIO-CAPSTONE-5069-BLOCKED-YAML",
    "SCENARIO-CAPSTONE-5069-FIELD-PRINCIPLES",
]

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_465_archived_466_activated_execution_incomplete_"
            "carried_forward or blocked_yaml_parse."
        )
    },
    "inference_substrate": {
        "principle": (
            "aggregation_from_upstream_artifacts -- reads roadmap and upstream JSON only; "
            "never live model inference."
        )
    },
    "milestone_from": {
        "principle": "the archived milestone 2026.06.465 from Exp5068.",
    },
    "milestone_to": {
        "principle": "the activated/staged milestone 2026.07.466 from the active roadmap.",
    },
    "source_capstone_path": {
        "principle": "the authoritative .465 capstone path, never a synthesized closeout.",
    },
    "close_state": {
        "principle": (
            "the blunt .465 capstone state preserved without promoting missing, flagged, "
            "or scoped evidence."
        )
    },
    "blockers_carried_forward": {
        "principle": (
            "the exact D1/D4/D6/guided-decoding/FR-11/KV260/Exp5066 blockers that "
            "must constrain .466 planning."
        )
    },
    "next_milestone_doc": {
        "principle": "the .466 milestone document path read from the active roadmap.",
    },
    "docs_updated": {
        "principle": (
            "empty when the conductor stop rule delegates ops/status/changelog/traceability "
            "reconciliation."
        )
    },
    "flagged_adversarial": {
        "principle": (
            "false because this record is a transparent aggregation artifact, not a fast "
            "compute-bound claim."
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
    "source_capstone_path",
    "close_state",
    "blockers_carried_forward",
    "next_milestone_doc",
    "docs_updated",
    "flagged_adversarial",
    "preconditions_checked",
    "phase_artifacts_loaded",
    "cited_upstream_artifacts",
    "field_principles",
    "random_seed",
    "reproducibility_checksum",
)

COMPLETE_VERDICT = "complete_465_archived_466_activated_execution_incomplete_carried_forward"
TERMINAL_PREFIXES = ("complete_", "success_", "passed_", "shipped_", "blocked_")

BLOCKER_IDS = [
    "d1_bounded_no_proper_win",
    "d4_duplicate_audit_retirement",
    "d6_efficiency_only",
    "guided_decoding_underpowered",
    "fr11_guarded_no_promote",
    "kv260_parity_packet_only",
    "exp5066_missing_unavailable_gemini_routing",
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


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


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
    if active.get("parse_ok") is False or staged.get("parse_ok") is False:
        return "blocked_yaml_parse"
    if active.get("parse_ok") is not True:
        return "blocked_missing_active_roadmap"
    return ""


def load_capstone(root: Path) -> tuple[JsonDict, JsonDict]:
    payload, status = read_json_mapping(root / CAPSTONE_REL_PATH)
    return payload, {"path": str(CAPSTONE_REL_PATH), **status}


def capstone_blocker(status: JsonMap) -> str:
    if status.get("exists") is not True:
        return "blocked_missing_source_capstone"
    if status.get("loadable") is not True:
        return "blocked_unloadable_source_capstone"
    return ""


def load_phase_artifacts(root: Path) -> JsonDict:
    rows: JsonDict = {}
    for exp_id, rel_path in PHASE_ARTIFACTS.items():
        payload, status = read_json_mapping(root / rel_path)
        row: JsonDict = {"experiment_id": exp_id, "path": str(rel_path), **status}
        row["present"] = status.get("exists") is True and status.get("loadable") is True
        if payload:
            row.update(
                {
                    "honest_verdict": str(payload.get("honest_verdict", "")),
                    "duration_s": _number(payload.get("duration_s")),
                    "flagged_adversarial": payload.get("flagged_adversarial") is True,
                }
            )
        rows[str(rel_path)] = row
    return rows


def build_close_state(capstone: JsonMap) -> JsonDict:
    best = _mapping(capstone.get("best_verifier_evidence"))
    fr11 = _mapping(capstone.get("fr11_self_learning_result"))
    hardware = _mapping(capstone.get("hardware_result"))
    sota = _mapping(capstone.get("sota_result"))
    arc = _mapping(capstone.get("arc_result"))
    pointer = _mapping(capstone.get("next_milestone_pointer"))
    return {
        "source_honest_verdict": str(capstone.get("honest_verdict", "")),
        "capstone_ready": capstone.get("capstone_ready") is True,
        "moat_state": str(capstone.get("moat_state", "")),
        "best_verifier_evidence": best,
        "fr11_self_learning_result": fr11,
        "hardware_result": hardware,
        "sota_result": sota,
        "arc_result": arc,
        "missing_upstream_artifacts": _list(capstone.get("missing_upstream_artifacts")),
        "next_milestone_pointer": pointer,
        "execution_incomplete": str(capstone.get("moat_state", "")) == "execution_incomplete",
    }


def _phase_verdict(phase_artifacts: JsonMap, exp_id: int) -> str:
    row = _mapping(phase_artifacts.get(str(PHASE_ARTIFACTS[exp_id])))
    return str(row.get("honest_verdict", ""))


def blockers_carried_forward(close_state: JsonMap, phase_artifacts: JsonMap) -> list[JsonDict]:
    best = _mapping(close_state.get("best_verifier_evidence"))
    fr11 = _mapping(close_state.get("fr11_self_learning_result"))
    hardware = _mapping(close_state.get("hardware_result"))
    sota = _mapping(close_state.get("sota_result"))
    d6_row = _mapping(best.get("source_row"))
    exp5066_row = _mapping(phase_artifacts.get(str(PHASE_ARTIFACTS[5066])))
    rows = [
        {
            "blocker_id": "d1_bounded_no_proper_win",
            "label": "D1 bounded/no proper win",
            "observed_state": _phase_verdict(phase_artifacts, 5059)
            or "complete_d1_sota_refresh_audit_no_proper_win_plus_0p080",
            "principle": "D1 is bounded/null unless it beats genuine tuned self-consistency with CI support.",
        },
        {
            "blocker_id": "d4_duplicate_audit_retirement",
            "label": "D4 duplicate-audit retirement",
            "observed_state": _phase_verdict(phase_artifacts, 5060)
            or "retired_d4_second_corpus_audit_failed_constraintbench_exact_v1_plus_0p370",
            "principle": "Duplicate or unclean second-corpus evidence cannot count as transfer.",
        },
        {
            "blocker_id": "d6_efficiency_only",
            "label": "D6 efficiency-only",
            "observed_state": _phase_verdict(phase_artifacts, 5061)
            or "success_tool_first_cascade_parity_at_0pct_judge_calls",
            "principle": (
                "Tool-first cascade is Pareto/cost evidence until accuracy parity and CI accounting "
                "support a moat claim."
            ),
            "efficiency_win": d6_row.get("efficiency_win") is True or best.get("cascade_efficiency_win") is True,
        },
        {
            "blocker_id": "guided_decoding_underpowered",
            "label": "guided decoding underpowered",
            "observed_state": _phase_verdict(phase_artifacts, 5062)
            or str(best.get("guided_decoding_frontier_state", "")),
            "principle": "A tiny guided-decoding point estimate must be scaled against rerank-only controls.",
        },
        {
            "blocker_id": "fr11_guarded_no_promote",
            "label": "FR-11 guarded no-promote",
            "observed_state": str(fr11.get("honest_verdict", "")),
            "principle": "The guard worked; no harmful replay promotion may be reframed as self-learning success.",
            "promoted": fr11.get("promoted") is True,
        },
        {
            "blocker_id": "kv260_parity_packet_only",
            "label": "KV260 parity packet only",
            "observed_state": str(hardware.get("honest_verdict", "")),
            "principle": "KV260 evidence is local parity/timing packet evidence only, not a speedup claim.",
            "timing_ratio_packet_built": hardware.get("timing_ratio_packet_built") is True,
        },
        {
            "blocker_id": "exp5066_missing_unavailable_gemini_routing",
            "label": "Exp5066 missing due unavailable Gemini routing",
            "observed_state": "missing" if exp5066_row.get("present") is False else str(sota.get("state", "")),
            "principle": (
                "The SOTA ingestion slot must be backfilled through available routing before .466 "
                "claims current literature coverage."
            ),
            "unavailable_model": "gemini-3.1-pro-preview",
        },
    ]
    for row in rows:
        row["must_not_be_laundered_into_success"] = True
    return rows


def cited_upstream_artifacts(root: Path, capstone_status: JsonMap, phase_artifacts: JsonMap) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_path in (ROADMAP_ACTIVE_REL_PATH, ROADMAP_NEXT_REL_PATH):
        path = root / rel_path
        if path.exists():
            rows.append({"path": str(rel_path), "sha256": file_sha256(path)})
    rows.append({"path": str(CAPSTONE_REL_PATH), **dict(capstone_status)})
    for row in phase_artifacts.values():
        mapping = _mapping(row)
        if mapping.get("present") is True or mapping.get("exists") is True:
            rows.append(mapping)
    return rows


def _next_milestone_doc(roadmaps: JsonMap, roadmaps_checked: JsonMap) -> str:
    active = _mapping(roadmaps.get("active"))
    active_status = _mapping(roadmaps_checked.get("active"))
    return str(active.get("milestone_doc") or active_status.get("milestone_doc") or NEXT_MILESTONE_DOC)


def build_artifact(
    *,
    root: Path,
    honest_verdict: str,
    roadmaps: JsonMap,
    roadmaps_checked: JsonMap,
    capstone: JsonMap,
    capstone_status: JsonMap,
    phase_artifacts: JsonMap,
    duration_s: float,
) -> JsonDict:
    close_state = build_close_state(capstone) if capstone else {}
    blockers = (
        blockers_carried_forward(close_state, phase_artifacts)
        if honest_verdict == COMPLETE_VERDICT
        else []
    )
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
        "source_capstone_path": str(CAPSTONE_REL_PATH),
        "close_state": close_state,
        "blockers_carried_forward": blockers,
        "next_milestone_doc": _next_milestone_doc(roadmaps, roadmaps_checked),
        "docs_updated": [],
        "flagged_adversarial": False,
        "preconditions_checked": {
            "roadmaps": dict(roadmaps_checked),
            "source_capstone": dict(capstone_status),
        },
        "phase_artifacts_loaded": dict(phase_artifacts),
        "cited_upstream_artifacts": cited_upstream_artifacts(root, capstone_status, phase_artifacts),
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
    phase_artifacts: JsonDict = {}
    capstone: JsonDict = {}
    capstone_status: JsonDict = {"path": str(CAPSTONE_REL_PATH), "exists": False, "loadable": False}
    blocker = roadmap_blocker(roadmaps_checked)
    if not blocker:
        capstone, capstone_status = load_capstone(root)
        blocker = capstone_blocker(capstone_status)
        if not blocker:
            phase_artifacts = load_phase_artifacts(root)
    artifact = build_artifact(
        root=root,
        honest_verdict=blocker or COMPLETE_VERDICT,
        roadmaps=roadmaps,
        roadmaps_checked=roadmaps_checked,
        capstone=capstone,
        capstone_status=capstone_status,
        phase_artifacts=phase_artifacts,
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
    if artifact.get("source_capstone_path") != str(CAPSTONE_REL_PATH):
        errors.append("source_capstone_path")
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
        if close_state.get("moat_state") != "execution_incomplete":
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

#!/usr/bin/env python3
"""Exp 5068: .465 capstone aggregation.

Spec refs: REQ-CAPSTONE-5068, SCENARIO-CAPSTONE-5068,
SCENARIO-CAPSTONE-5068-FIELD-PRINCIPLES.

This module is intentionally an aggregation step. It does not retry the moat,
FR-11, hardware, SOTA, or ARC work; instead it reads the upstream artifacts and
keeps their boundaries intact so a missing or scoped result cannot drift into a
stronger claim during milestone closeout.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5068_capstone_v465"
EXPERIMENT_ID = 5068
SCHEMA = "carnot.experiment_5068_capstone_v465.v1"
RESULT_RELATIVE_PATH = "results/experiment_5068_capstone_v465.json"
MILESTONE = "2026.06.465"
RANDOM_SEED = 5068
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-5068",
    "SCENARIO-CAPSTONE-5068",
    "SCENARIO-CAPSTONE-5068-FIELD-PRINCIPLES",
]

VALID_MOAT_STATES = {
    "moat_realized",
    "musr_scoped_positive",
    "second_corpus_scoped_positive",
    "retired_bounded",
    "execution_incomplete",
}
MOAT_VERDICT_LABELS = {
    "moat_realized": "realized_moat",
    "musr_scoped_positive": "scoped_positive",
    "second_corpus_scoped_positive": "scoped_positive",
    "retired_bounded": "bounded_retirement",
    "execution_incomplete": "execution_incomplete",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_capstone_v465_"
            "<realized_moat|scoped_positive|bounded_retirement|execution_incomplete>_"
            "<fr11_state>[_missing_<source>]."
        )
    },
    "moat_state": {
        "principle": (
            "the blunt `.465` verdict imported from Exp5063, constrained to realized, "
            "scoped positive, bounded-retired, or execution-incomplete."
        )
    },
    "best_verifier_evidence": {
        "principle": (
            "the best Exp5063 verifier row and gate fields, with headline_countable=false "
            "unless Exp5063's authoritative moat_state permits it."
        )
    },
    "fr11_self_learning_result": {
        "principle": (
            "the Exp5064 audited skill-graph promotion/no-promote result; credible positive "
            "evidence requires the authoritative promotion gate to promote."
        )
    },
    "hardware_result": {
        "principle": (
            "the Exp5065 KV260 result scoped to local SSH transcript-backed parity/timing, "
            "never a general FPGA speedup claim."
        )
    },
    "sota_result": {
        "principle": (
            "the Exp5066 ingestion state, or an explicit missing-artifact state with no "
            "fabricated references."
        )
    },
    "arc_result": {
        "principle": (
            "the Exp5067 live-path ARC result, including no-bank, duplicate-depth, and "
            "provenance boundaries."
        )
    },
    "cited_upstream_artifacts": {
        "principle": (
            "one row per expected upstream with source, path, experiment_id, present, "
            "fields_imported, and sha256 when present."
        )
    },
    "next_milestone_pointer": {
        "principle": (
            "the `.466` route selected from the roadmap state table for realized, "
            "scoped-positive, bounded-retired, or execution-incomplete moat states, annotated "
            "when SOTA ingestion is missing."
        )
    },
    "docs_update_required": {
        "principle": (
            "records required post-run reconciliation for ops/status/changelog/traceability "
            "while honoring conductor stop rules that may defer those edits."
        )
    },
}

REQUIRED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class UpstreamSource:
    """Expected input artifact and the fields this capstone is allowed to import."""

    experiment_id: int
    relative_path: str
    imported_fields: tuple[str, ...]


UPSTREAMS: dict[str, UpstreamSource] = {
    "moat_gate": UpstreamSource(
        5063,
        "results/experiment_5063_moat_gate_resolution_v465.json",
        (
            "honest_verdict",
            "moat_state",
            "best_arm",
            "best_arm_delta",
            "best_arm_ci",
            "second_corpus_confirmed",
            "second_corpus_audit_clean",
            "cascade_efficiency_win",
            "guided_decoding_frontier_state",
            "bounded_retirement_ok",
            "execution_incomplete_reasons",
            "per_arm_table",
            "blocked_upstream_artifacts",
            "flagged_upstream_artifacts",
            "missing_upstream_artifacts",
        ),
    ),
    "fr11_self_learning": UpstreamSource(
        5064,
        "results/experiment_5064_audited_skillgraph_self_learning.json",
        (
            "honest_verdict",
            "continuous_self_learning_task",
            "self_learning_loop_executed",
            "candidate_skill_count",
            "verified_skill_count",
            "promoted",
            "promoted_skill_ids",
            "promotion_decision",
            "no_promote_reason",
            "pre_update_accuracy",
            "post_update_accuracy",
            "heldout_delta",
            "nonforgetting_delta",
            "contamination_guard_passed",
            "external_verifier_audit_receipts",
            "skill_graph_path",
            "skill_graph_sha256",
            "legacy_models_smoke_only",
        ),
    ),
    "hardware": UpstreamSource(
        5065,
        "results/experiment_5065_kv260_testbench_timing_packet.json",
        (
            "honest_verdict",
            "kv260_ssh_reachable",
            "overlay_loaded",
            "loaded_overlay",
            "timing_ratio_packet_built",
            "cpu_reference_ok",
            "kv260_result_ok",
            "board_transcript_path",
            "transcript_sha256",
            "structured_testbench_evidence",
            "local_claim_scope",
            "timing_ratio_packet",
            "optional_board_prechecks",
        ),
    ),
    "sota": UpstreamSource(
        5066,
        "results/experiment_5066_sota_ingestion_v466.json",
        (
            "honest_verdict",
            "research_references_updated",
            "n_sources_checked",
            "selected_sources",
            "duplicate_filter",
            "next_milestone_candidates",
            "preconditions_checked",
        ),
    ),
    "arc": UpstreamSource(
        5067,
        "results/experiment_5067_arc_live_path_self_discovery.json",
        (
            "honest_verdict",
            "target_game",
            "target_level",
            "prior_reproduced_level",
            "new_levels_banked",
            "offline_reproduced",
            "duplicate_solve_avoided",
            "registry_precheck_passed",
            "reproducible_total_levels_before",
            "reproducible_total_levels_after",
            "solve_claim",
            "solve_provenance",
            "provenance_evidence",
            "live_agent_attempts",
        ),
    ),
    "prior_capstone": UpstreamSource(
        5055,
        "results/experiment_5055_capstone_v464.json",
        (
            "honest_verdict",
            "capstone_ready",
            "moat_state",
            "fr11_self_learning_result",
            "next_milestone_pointer",
        ),
    ),
}

ROUTE_TABLE: dict[str, JsonDict] = {
    "moat_realized": {
        "experiment_class": "scale_realized_verifier",
        "concrete_next": (
            "Scale the realized oracle-distinct verifier/cascade with locked controls, "
            "audited second-corpus evidence, and operator-gated activation."
        ),
    },
    "musr_scoped_positive": {
        "experiment_class": "confirm_scoped_musr_positive",
        "concrete_next": (
            "Treat the MuSR signal as scoped progress only; run clean D4 or D6 confirmation "
            "before any PRD-level moat claim."
        ),
    },
    "second_corpus_scoped_positive": {
        "experiment_class": "repair_musr_or_cascade_for_second_corpus_positive",
        "concrete_next": (
            "Treat the second-corpus result as a transfer clue only; repair MuSR D1 or D6 "
            "before making it a headline verifier-moat claim."
        ),
    },
    "retired_bounded": {
        "experiment_class": "pivot_to_new_verifier_direction",
        "concrete_next": (
            "Bound-retire the tested D1/D4/D6 shape and choose a differentiated .466 "
            "verifier direction from current SOTA ingestion."
        ),
    },
    "execution_incomplete": {
        "experiment_class": "execution_repair_before_claim_or_retirement",
        "concrete_next": (
            "Repair flagged, unclean, or statistically incomplete Phase D evidence before "
            "claiming a moat or treating the shape as a clean null."
        ),
    },
}


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("top-level JSON value is not an object")
    return dict(payload)


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


def _moat_state(payload: JsonMap) -> str:
    state = str(payload.get("moat_state") or "execution_incomplete")
    return state if state in VALID_MOAT_STATES else "execution_incomplete"


def _imported_fields(source: str, payload: JsonMap) -> list[str]:
    return [field for field in UPSTREAMS[source].imported_fields if field in payload]


def _citation(
    *,
    source: str,
    config: UpstreamSource,
    status: str,
    path: Path,
    payload: JsonMap | None = None,
    parse_error: str = "",
) -> JsonDict:
    row: JsonDict = {
        "source": source,
        "experiment_id": config.experiment_id,
        "path": config.relative_path,
        "present": status == "present",
        "status": status,
        "fields_imported": _imported_fields(source, payload or {}),
        "sha256": file_sha256(path) if path.exists() else None,
        "honest_verdict": str((payload or {}).get("honest_verdict") or ""),
    }
    if parse_error:
        row["parse_error"] = parse_error
    return row


def load_upstream_artifacts(root: Path = REPO_ROOT) -> JsonDict:
    root = Path(root)
    artifacts: dict[str, JsonDict] = {}
    citations: list[JsonDict] = []
    missing: list[JsonDict] = []
    malformed: list[JsonDict] = []
    for source, config in UPSTREAMS.items():
        path = root / config.relative_path
        if not path.exists():
            citation = _citation(source=source, config=config, status="missing", path=path)
            citations.append(citation)
            missing.append(
                {"source": source, "experiment_id": config.experiment_id, "path": config.relative_path}
            )
            continue
        try:
            payload = read_json_object(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            citation = _citation(
                source=source,
                config=config,
                status="malformed",
                path=path,
                parse_error=str(exc),
            )
            citations.append(citation)
            malformed.append(
                {
                    "source": source,
                    "experiment_id": config.experiment_id,
                    "path": config.relative_path,
                    "parse_error": str(exc),
                }
            )
            continue
        artifacts[source] = payload
        citations.append(_citation(source=source, config=config, status="present", path=path, payload=payload))
    return {
        "artifacts": artifacts,
        "cited_upstream_artifacts": citations,
        "missing_upstream_artifacts": missing,
        "malformed_upstream_artifacts": malformed,
    }


def _best_source_row(moat_gate: JsonMap) -> JsonDict:
    best_arm = str(moat_gate.get("best_arm") or "")
    for row in _list(moat_gate.get("per_arm_table")):
        if not isinstance(row, Mapping):
            continue
        row_id = str(row.get("artifact_id") or row.get("arm_id") or "")
        if row_id == best_arm:
            return dict(row)
    return {}


def _best_verifier_evidence(moat_gate: JsonMap, state: str) -> JsonDict:
    return {
        "source_experiment_id": 5063,
        "source_honest_verdict": str(moat_gate.get("honest_verdict") or ""),
        "moat_state": state,
        "best_arm": moat_gate.get("best_arm"),
        "best_arm_delta": _number(moat_gate.get("best_arm_delta")),
        "best_arm_ci": moat_gate.get("best_arm_ci")
        if isinstance(moat_gate.get("best_arm_ci"), list)
        else None,
        "second_corpus_confirmed": moat_gate.get("second_corpus_confirmed") is True,
        "second_corpus_audit_clean": moat_gate.get("second_corpus_audit_clean") is True,
        "cascade_efficiency_win": moat_gate.get("cascade_efficiency_win") is True,
        "guided_decoding_frontier_state": str(
            moat_gate.get("guided_decoding_frontier_state") or "missing"
        ),
        "bounded_retirement_ok": moat_gate.get("bounded_retirement_ok") is True,
        "execution_incomplete_reasons": [
            str(reason) for reason in _list(moat_gate.get("execution_incomplete_reasons"))
        ],
        "headline_countable": state == "moat_realized",
        "scoped_positive_evidence": state
        in {"musr_scoped_positive", "second_corpus_scoped_positive"},
        "source_row": _best_source_row(moat_gate),
    }


def _fr11_result(payload: JsonMap) -> JsonDict:
    if not payload:
        return {
            "state": "missing",
            "credible_positive_evidence": False,
            "claim_boundary": "missing_artifact_no_fr11_claim",
        }
    decision = _mapping(payload.get("promotion_decision"))
    promoted = payload.get("promoted") is True or decision.get("promoted") is True
    heldout_delta = _number(payload.get("heldout_delta"))
    nonforgetting_delta = _number(payload.get("nonforgetting_delta"))
    guard_passed = payload.get("contamination_guard_passed") is True
    credible = bool(
        promoted
        and guard_passed
        and heldout_delta is not None
        and heldout_delta > 0.0
        and (nonforgetting_delta is None or nonforgetting_delta >= 0.0)
    )
    return {
        "state": "credible_positive" if credible else "no_credible_positive_evidence",
        "credible_positive_evidence": credible,
        "claim_boundary": (
            "audited_promoted_skill_graph_positive"
            if credible
            else "audited_no_promote_or_nonpositive_heldout_delta"
        ),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "continuous_self_learning_task": payload.get("continuous_self_learning_task") is True,
        "self_learning_loop_executed": payload.get("self_learning_loop_executed") is True,
        "candidate_skill_count": _int(payload.get("candidate_skill_count")),
        "verified_skill_count": _int(payload.get("verified_skill_count")),
        "promoted": promoted,
        "promoted_skill_ids": [str(item) for item in _list(payload.get("promoted_skill_ids"))],
        "no_promote_reason": str(payload.get("no_promote_reason") or decision.get("no_promote_reason") or ""),
        "pre_update_accuracy": _number(payload.get("pre_update_accuracy")),
        "post_update_accuracy": _number(payload.get("post_update_accuracy")),
        "heldout_delta": heldout_delta,
        "nonforgetting_delta": nonforgetting_delta,
        "contamination_guard_passed": guard_passed,
        "external_verifier_audit_receipts": _list(payload.get("external_verifier_audit_receipts")),
        "skill_graph_path": payload.get("skill_graph_path"),
        "skill_graph_sha256": payload.get("skill_graph_sha256"),
        "legacy_models_smoke_only": payload.get("legacy_models_smoke_only") is True,
    }


def _hardware_result(payload: JsonMap) -> JsonDict:
    if not payload:
        return {"state": "missing", "claim_boundary": "missing_artifact_no_hardware_claim"}
    packet_built = (
        payload.get("timing_ratio_packet_built") is True
        and payload.get("cpu_reference_ok") is True
        and payload.get("kv260_result_ok") is True
    )
    claim_scope = str(payload.get("local_claim_scope") or "")
    return {
        "state": "packet_built" if packet_built else "blocked_or_not_built",
        "claim_boundary": "local_kv260_transcript_backed_parity_timing_only",
        "no_general_speedup_claim": "no_general_fpga_speedup_claim" in claim_scope,
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "kv260_ssh_reachable": payload.get("kv260_ssh_reachable") is True,
        "overlay_loaded": payload.get("overlay_loaded") is True,
        "loaded_overlay": payload.get("loaded_overlay"),
        "timing_ratio_packet_built": packet_built,
        "cpu_reference_ok": payload.get("cpu_reference_ok") is True,
        "kv260_result_ok": payload.get("kv260_result_ok") is True,
        "board_transcript_path": payload.get("board_transcript_path"),
        "transcript_sha256": payload.get("transcript_sha256"),
        "structured_testbench_evidence": _mapping(payload.get("structured_testbench_evidence")),
        "timing_ratio_packet": _mapping(payload.get("timing_ratio_packet")),
        "optional_board_prechecks": _mapping(payload.get("optional_board_prechecks")),
        "local_claim_scope": claim_scope,
    }


def _sota_result(payload: JsonMap) -> JsonDict:
    if not payload:
        return {
            "state": "missing",
            "claim_boundary": "missing_artifact_no_sota_ingestion_claim",
            "honest_verdict": "",
            "research_references_updated": False,
            "n_sources_checked": 0,
            "selected_sources": [],
            "duplicate_filter": {},
            "next_milestone_candidates": [],
        }
    updated = payload.get("research_references_updated") is True
    return {
        "state": "references_updated" if updated else "no_new_sources_or_not_updated",
        "claim_boundary": "local_research_reference_ingestion_only_no_external_result_claim",
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "research_references_updated": updated,
        "n_sources_checked": _int(payload.get("n_sources_checked")),
        "selected_sources": _list(payload.get("selected_sources")),
        "duplicate_filter": _mapping(payload.get("duplicate_filter")),
        "next_milestone_candidates": _list(payload.get("next_milestone_candidates")),
        "preconditions_checked": _mapping(payload.get("preconditions_checked")),
    }


def _arc_result(payload: JsonMap) -> JsonDict:
    if not payload:
        return {"state": "missing", "claim_boundary": "missing_artifact_no_arc_claim"}
    banked = _int(payload.get("new_levels_banked"))
    reproduced = payload.get("offline_reproduced") is True
    provenance = _mapping(payload.get("provenance_evidence"))
    return {
        "state": "banked_new_level" if banked > 0 and reproduced else "no_bank",
        "claim_boundary": "opportunistic_live_path_reproduction_gated_no_duplicate_credit",
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "target_game": payload.get("target_game"),
        "target_level": _int(payload.get("target_level")),
        "prior_reproduced_level": _int(payload.get("prior_reproduced_level")),
        "new_levels_banked": banked,
        "offline_reproduced": reproduced,
        "duplicate_solve_avoided": payload.get("duplicate_solve_avoided") is True,
        "registry_precheck_passed": payload.get("registry_precheck_passed") is True,
        "reproducible_total_levels_before": _int(payload.get("reproducible_total_levels_before")),
        "reproducible_total_levels_after": _int(payload.get("reproducible_total_levels_after")),
        "solve_claim": _mapping(payload.get("solve_claim")),
        "solve_provenance": str(payload.get("solve_provenance") or ""),
        "provenance_evidence": provenance,
        "live_agent_attempts": _list(payload.get("live_agent_attempts")),
        "no_offline_source_reading": provenance.get("offline_source_reading_used") is False,
        "no_offline_ground_truth_bfs": provenance.get("offline_ground_truth_bfs_used") is False,
        "no_hand_built_adapter": provenance.get("hand_built_adapter_used") is False,
    }


def _candidate_classes(sota: JsonMap) -> list[str]:
    return [
        str(row.get("candidate"))
        for row in _list(sota.get("next_milestone_candidates"))
        if isinstance(row, Mapping) and row.get("candidate")
    ]


def _next_milestone_pointer(state: str, sota: JsonMap, *, sota_missing: bool) -> JsonDict:
    selected = dict(ROUTE_TABLE[state])
    if sota_missing:
        selected["blocked_dependency"] = "exp5066_sota_ingestion_missing"
        selected["concrete_next"] += " Backfill the missing Exp5066 reserved SOTA ingestion before final .466 planning."
    candidates = _candidate_classes(sota)
    return {
        "selected_state": state,
        "selected": selected,
        "by_moat_state": ROUTE_TABLE,
        "candidate_classes": candidates,
        "sota_ingestion_missing": sota_missing,
        "route_basis": "openspec/change-proposals/research-roadmap-vNEXT.md section 6 falsifiable gates",
    }


def _docs_update_required() -> JsonDict:
    return {
        "openspec_capstone_spec": True,
        "ops_status": True,
        "ops_changelog": True,
        "_bmad_traceability": True,
        "updated_by_this_run": ["openspec/capabilities/capstone/spec.md"],
        "deferred_by_stop_rule": True,
        "reason": "operator stop rule delegates ops/status/changelog/traceability reconciliation",
    }


def _honest_verdict(state: str, fr11: JsonMap, missing: list[JsonDict]) -> str:
    label = MOAT_VERDICT_LABELS[state]
    fr11_label = (
        "fr11_credible_positive"
        if fr11.get("credible_positive_evidence") is True
        else "fr11_no_credible_positive_evidence"
    )
    missing_suffix = "_".join(
        f"missing_{str(item.get('source'))}" for item in missing if item.get("source") != "prior_capstone"
    )
    return f"complete_capstone_v465_{label}_{fr11_label}" + (
        f"_{missing_suffix}" if missing_suffix else ""
    )


def build_artifact(loaded: JsonMap, duration_s: float) -> JsonDict:
    artifacts = _mapping(loaded.get("artifacts"))
    moat_gate = _mapping(artifacts.get("moat_gate"))
    state = _moat_state(moat_gate)
    fr11 = _fr11_result(_mapping(artifacts.get("fr11_self_learning")))
    hardware = _hardware_result(_mapping(artifacts.get("hardware")))
    sota_source = _mapping(artifacts.get("sota"))
    sota = _sota_result(sota_source)
    arc = _arc_result(_mapping(artifacts.get("arc")))
    missing = [dict(row) for row in _list(loaded.get("missing_upstream_artifacts")) if isinstance(row, Mapping)]
    malformed = [
        dict(row) for row in _list(loaded.get("malformed_upstream_artifacts")) if isinstance(row, Mapping)
    ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "milestone": MILESTONE,
        "honest_verdict": _honest_verdict(state, fr11, missing),
        "capstone_ready": not missing and not malformed,
        "moat_state": state,
        "best_verifier_evidence": _best_verifier_evidence(moat_gate, state),
        "fr11_self_learning_result": fr11,
        "hardware_result": hardware,
        "sota_result": sota,
        "arc_result": arc,
        "cited_upstream_artifacts": _list(loaded.get("cited_upstream_artifacts")),
        "missing_upstream_artifacts": missing,
        "malformed_upstream_artifacts": malformed,
        "next_milestone_pointer": _next_milestone_pointer(
            state,
            sota,
            sota_missing=any(row.get("source") == "sota" for row in missing),
        ),
        "docs_update_required": _docs_update_required(),
        "field_principles": FIELD_PRINCIPLES,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(max(float(duration_s), 0.0001), 6),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(field)
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete_capstone_v465_"):
        errors.append("honest_verdict")
    if not isinstance(artifact.get("capstone_ready"), bool):
        errors.append("capstone_ready")
    if artifact.get("moat_state") not in VALID_MOAT_STATES:
        errors.append("moat_state")
    if not isinstance(artifact.get("fr11_self_learning_result"), Mapping):
        errors.append("fr11_self_learning_result")
    if not isinstance(artifact.get("cited_upstream_artifacts"), list):
        errors.append("cited_upstream_artifacts")
    if not isinstance(artifact.get("docs_update_required"), Mapping):
        errors.append("docs_update_required")
    return sorted(set(errors))


def run(root: Path = REPO_ROOT, artifact_path: Path | None = None) -> JsonDict:
    started = time.perf_counter()
    loaded = load_upstream_artifacts(root)
    artifact = build_artifact(loaded, duration_s=time.perf_counter() - started)
    write_json(artifact_path or Path(root) / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(root: Path = REPO_ROOT, artifact_path: Path | None = None) -> int:
    artifact = run(root=root, artifact_path=artifact_path)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - direct experiment entrypoint
    raise SystemExit(main())

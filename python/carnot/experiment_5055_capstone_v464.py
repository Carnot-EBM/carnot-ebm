#!/usr/bin/env python3
"""Exp 5055: .464 capstone aggregation.

Spec refs: REQ-CAPSTONE-5055, SCENARIO-CAPSTONE-5055,
SCENARIO-CAPSTONE-5055-FIELD-PRINCIPLES.
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
EXPERIMENT = "experiment_5055_capstone_v464"
EXPERIMENT_ID = 5055
SCHEMA = "carnot.experiment_5055_capstone_v464.v1"
RESULT_RELATIVE_PATH = "results/experiment_5055_capstone_v464.json"
MILESTONE = "2026.06.464"
RANDOM_SEED = 5055
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
SPEC_REFS = [
    "REQ-CAPSTONE-5055",
    "SCENARIO-CAPSTONE-5055",
    "SCENARIO-CAPSTONE-5055-FIELD-PRINCIPLES",
]

VALID_MOAT_STATES = {
    "moat_realized",
    "musr_scoped_positive",
    "retired_bounded",
    "execution_incomplete",
}
MOAT_STATE_LABELS = {
    "moat_realized": "realized_moat",
    "musr_scoped_positive": "musr_scoped_positive",
    "retired_bounded": "bounded_retirement",
    "execution_incomplete": "execution_incomplete",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_capstone_v464_"
            "<realized_moat|musr_scoped_positive|bounded_retirement|execution_incomplete>_"
            "<fr11_state>."
        )
    },
    "moat_state": {
        "principle": (
            "the blunt .464 verdict imported from Exp5050, constrained to realized moat, "
            "MuSR-scoped positive, bounded retirement, or execution-incomplete."
        )
    },
    "best_arm_and_delta": {
        "principle": (
            "the strongest verifier evidence from Exp5050, including arm id, delta, CI, "
            "and whether the evidence is headline-countable."
        )
    },
    "fr11_self_learning_result": {
        "principle": (
            "whether Exp5051 produced credible positive FR-11 self-learning evidence, "
            "with held-out delta and guardrail state."
        )
    },
    "hardware_result": {
        "principle": (
            "the Exp5052 local KV260 timing-ratio packet state without promoting it to a "
            "general speedup claim."
        )
    },
    "arc_result": {
        "principle": (
            "the Exp5054 live-path ARC result, including no-bank and duplicate-depth state."
        )
    },
    "next_milestone_pointer": {
        "principle": (
            "the concrete .465 experiment-class routing table for all four moat states "
            "plus the selected route."
        )
    },
    "docs_updated": {
        "principle": (
            "records only the docs touched by this capstone run; ops reconciliation may "
            "be intentionally delegated."
        )
    },
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "capstone_ready",
    "milestone",
    "moat_state",
    "best_arm_and_delta",
    "best_verifier_evidence",
    "second_corpus_state",
    "cascade_state",
    "fr11_state",
    "fr11_self_learning_result",
    "hardware_state",
    "hardware_result",
    "arc_state",
    "arc_result",
    "next_milestone_pointer",
    "docs_updated",
)


@dataclass(frozen=True)
class UpstreamSource:
    """One source artifact imported into the .464 capstone."""

    experiment_id: int
    relative_path: str
    imported_fields: tuple[str, ...]


UPSTREAMS: dict[str, UpstreamSource] = {
    "moat_gate": UpstreamSource(
        5050,
        "results/experiment_5050_moat_gate_resolution_v464.json",
        (
            "honest_verdict",
            "moat_state",
            "best_arm",
            "best_arm_delta",
            "best_arm_ci",
            "second_corpus_confirmed",
            "cascade_efficiency_win",
            "execution_incomplete_reasons",
            "blocked_upstream_artifacts",
            "flagged_upstream_artifacts",
            "missing_upstream_artifacts",
            "per_arm_table",
            "cascade_artifact",
            "second_corpus_artifact",
        ),
    ),
    "fr11_self_learning": UpstreamSource(
        5051,
        "results/experiment_5051_verifier_trace_self_learning.json",
        (
            "honest_verdict",
            "self_learning_loop_executed",
            "near_miss_count",
            "verified_trace_count",
            "pre_update_accuracy",
            "post_update_accuracy",
            "heldout_delta",
            "contamination_guard_passed",
            "fr11_evidence",
            "delta_vs_genuine_tuned_sc",
            "checkpoint_or_memory_path",
        ),
    ),
    "hardware": UpstreamSource(
        5052,
        "results/experiment_5052_kv260_pbit_timing_ratio.json",
        (
            "honest_verdict",
            "kv260_ssh_reachable",
            "overlay_loaded",
            "loaded_overlay",
            "timing_ratio_packet_built",
            "cpu_reference_ok",
            "kv260_result_ok",
            "local_claim_scope",
            "timing_ratio_packet",
        ),
    ),
    "sota": UpstreamSource(
        5053,
        "results/experiment_5053_sota_ingestion_v465.json",
        (
            "honest_verdict",
            "research_references_updated",
            "n_sources_checked",
            "next_milestone_candidates",
        ),
    ),
    "arc": UpstreamSource(
        5054,
        "results/experiment_5054_arc_live_path_self_discovery.json",
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
        ),
    ),
}


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> JsonDict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _int(value: Any) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _moat_state(gate: JsonMap) -> str:
    state = str(gate.get("moat_state") or "execution_incomplete")
    return state if state in VALID_MOAT_STATES else "execution_incomplete"


def _best_row(gate: JsonMap) -> JsonDict:
    best_arm = str(gate.get("best_arm") or "")
    rows = {
        str(row.get("arm_id")): dict(row)
        for row in _list(gate.get("per_arm_table"))
        if isinstance(row, Mapping)
    }
    return rows.get(best_arm, {})


def _best_arm_and_delta(gate: JsonMap, state: str) -> JsonDict:
    row = _best_row(gate)
    status = str(row.get("execution_status") or "unknown")
    return {
        "arm_id": gate.get("best_arm"),
        "delta": _number(gate.get("best_arm_delta")),
        "ci95": gate.get("best_arm_ci") if isinstance(gate.get("best_arm_ci"), list) else None,
        "evidence_status": status,
        "headline_countable": status == "clean" and state in {"moat_realized", "musr_scoped_positive"},
    }


def _moat_resolution(gate: JsonMap, state: str) -> JsonDict:
    return {
        "state": state,
        "honest_verdict": str(gate.get("honest_verdict") or ""),
        "second_corpus_confirmed": gate.get("second_corpus_confirmed") is True,
        "cascade_efficiency_win": gate.get("cascade_efficiency_win") is True,
        "bounded_retirement_ok": gate.get("bounded_retirement_ok") is True,
        "execution_incomplete_reasons": [
            str(reason) for reason in _list(gate.get("execution_incomplete_reasons"))
        ],
        "blocked_upstream_artifacts": _list(gate.get("blocked_upstream_artifacts")),
        "flagged_upstream_artifacts": _list(gate.get("flagged_upstream_artifacts")),
        "missing_upstream_artifacts": _list(gate.get("missing_upstream_artifacts")),
    }


def _second_corpus_state(gate: JsonMap) -> JsonDict:
    row = _mapping(gate.get("second_corpus_artifact"))
    status = str(row.get("execution_status") or "missing")
    state = "confirmed_clean" if gate.get("second_corpus_confirmed") is True and status == "clean" else f"{status}_not_counted"
    return {
        "state": state,
        "execution_status": status,
        "honest_verdict": str(row.get("honest_verdict") or ""),
        "best_arm": row.get("best_arm"),
        "reported_confirmed": row.get("second_corpus_confirmed") is True,
        "headline_counted": gate.get("second_corpus_confirmed") is True and status == "clean",
        "delta_vs_tuned_sc_second": _number(row.get("delta_vs_tuned_sc_second")),
        "paired_ci95_second": row.get("paired_ci95_second")
        if isinstance(row.get("paired_ci95_second"), list)
        else None,
    }


def _cascade_state(gate: JsonMap) -> JsonDict:
    row = _mapping(gate.get("cascade_artifact"))
    status = str(row.get("execution_status") or "missing")
    state = "efficiency_win" if gate.get("cascade_efficiency_win") is True else status
    return {
        "state": state,
        "execution_status": status,
        "honest_verdict": str(row.get("honest_verdict") or ""),
        "efficiency_win": gate.get("cascade_efficiency_win") is True,
        "judge_call_fraction": _number(row.get("judge_call_fraction")),
        "paired_ci95": row.get("paired_ci95") if isinstance(row.get("paired_ci95"), list) else None,
    }


def _fr11_result(payload: JsonMap) -> JsonDict:
    heldout_delta = _number(payload.get("heldout_delta"))
    loop_executed = payload.get("self_learning_loop_executed") is True
    guard_passed = payload.get("contamination_guard_passed") is True
    credible = bool(loop_executed and guard_passed and heldout_delta is not None and heldout_delta > 0.0)
    state = "credible_positive" if credible else "guarded_negative" if loop_executed and guard_passed else "not_credible"
    return {
        "state": state,
        "credible_evidence": credible,
        "credible_evidence_scope": (
            "credible positive FR-11 held-out improvement"
            if credible
            else "no credible positive FR-11 self-learning evidence"
        ),
        "self_learning_loop_executed": loop_executed,
        "contamination_guard_passed": guard_passed,
        "near_miss_count": _int(payload.get("near_miss_count")),
        "verified_trace_count": _int(payload.get("verified_trace_count")),
        "pre_update_accuracy": _number(payload.get("pre_update_accuracy")),
        "post_update_accuracy": _number(payload.get("post_update_accuracy")),
        "heldout_delta": heldout_delta,
        "delta_vs_genuine_tuned_sc": _number(payload.get("delta_vs_genuine_tuned_sc")),
        "checkpoint_or_memory_path": payload.get("checkpoint_or_memory_path"),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
    }


def _hardware_result(payload: JsonMap) -> JsonDict:
    packet_built = payload.get("timing_ratio_packet_built") is True
    return {
        "state": "packet_built" if packet_built else "blocked_or_not_built",
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "kv260_ssh_reachable": payload.get("kv260_ssh_reachable") is True,
        "overlay_loaded": payload.get("overlay_loaded") is True,
        "loaded_overlay": payload.get("loaded_overlay"),
        "timing_ratio_packet_built": packet_built,
        "cpu_reference_ok": payload.get("cpu_reference_ok") is True,
        "kv260_result_ok": payload.get("kv260_result_ok") is True,
        "claim_scope": str(payload.get("local_claim_scope") or ""),
        "timing_ratio_packet": _mapping(payload.get("timing_ratio_packet")),
    }


def _arc_result(payload: JsonMap) -> JsonDict:
    missing = not payload
    banked = _int(payload.get("new_levels_banked"))
    offline_reproduced = payload.get("offline_reproduced") is True
    state = "missing" if missing else "banked_new_level" if banked > 0 and offline_reproduced else "no_bank"
    return {
        "state": state,
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "target_game": payload.get("target_game"),
        "target_level": _int(payload.get("target_level")),
        "prior_reproduced_level": _int(payload.get("prior_reproduced_level")),
        "new_levels_banked": banked,
        "offline_reproduced": offline_reproduced,
        "duplicate_solve_avoided": payload.get("duplicate_solve_avoided") is True,
        "registry_precheck_passed": payload.get("registry_precheck_passed") is True,
        "reproducible_total_levels_before": _int(payload.get("reproducible_total_levels_before")),
        "reproducible_total_levels_after": _int(payload.get("reproducible_total_levels_after")),
        "solve_claim": _mapping(payload.get("solve_claim")),
    }


def _sota_state(payload: JsonMap) -> JsonDict:
    candidates = [dict(row) for row in _list(payload.get("next_milestone_candidates")) if isinstance(row, Mapping)]
    return {
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "research_references_updated": payload.get("research_references_updated") is True,
        "n_sources_checked": _int(payload.get("n_sources_checked")),
        "next_milestone_candidates": candidates,
    }


def _next_milestone_pointer(state: str, sota: JsonMap) -> JsonDict:
    candidate_names = [str(row.get("candidate")) for row in _sota_state(sota)["next_milestone_candidates"]]
    pointers = {
        "moat_realized": {
            "experiment_class": "scale_realized_verifier",
            "concrete_next": "Scale the winning verifier arm with locked oracle-distinct controls and operator-gated activation.",
        },
        "musr_scoped_positive": {
            "experiment_class": "second_corpus_confirmation",
            "concrete_next": "Run non-flagged second-corpus confirmation plus cascade cost accounting for the winning MuSR arm.",
        },
        "retired_bounded": {
            "experiment_class": "new_verifier_direction_from_sota",
            "concrete_next": "Retire the bounded D1/D2/D3 family and pick a fresh .465 verifier direction from SOTA ingestion.",
            "candidate_classes": candidate_names,
        },
        "execution_incomplete": {
            "experiment_class": "phase_d_execution_repair_and_confirmation",
            "concrete_next": "Repair blocked or flagged Phase D arms before treating any result as a null: D1 SOTA refresh, D6 cascade, and D2/D4 audit cleanup.",
        },
    }
    return {
        "selected_state": state,
        "selected": pointers[state],
        "by_moat_state": pointers,
    }


def _honest_verdict(state: str, fr11: JsonMap) -> str:
    fr11_label = "fr11_credible_positive" if fr11.get("credible_evidence") is True else "fr11_no_credible_positive_evidence"
    return f"complete_capstone_v464_{MOAT_STATE_LABELS[state]}_{fr11_label}"


def _docs_updated() -> JsonDict:
    return {
        "openspec_capstone_spec": True,
        "ops_status": False,
        "ops_changelog": False,
        "_bmad_traceability": False,
        "reason": "operator stop rule delegates ops/status/changelog/traceability reconciliation",
    }


def _imported_fields(source: str, payload: JsonMap) -> list[str]:
    return [field for field in UPSTREAMS[source].imported_fields if field in payload]


def _artifact_state(source: str, payload: JsonMap | None) -> JsonDict:
    config = UPSTREAMS[source]
    if payload is None:
        return {
            "source": source,
            "experiment_id": config.experiment_id,
            "path": config.relative_path,
            "state": "missing",
            "honest_verdict": "",
        }
    return {
        "source": source,
        "experiment_id": config.experiment_id,
        "path": config.relative_path,
        "state": "present",
        "honest_verdict": str(payload.get("honest_verdict") or ""),
    }


def load_upstream_artifacts(root: Path = REPO_ROOT) -> JsonDict:
    artifacts: dict[str, JsonDict] = {}
    citations: list[JsonDict] = []
    missing: list[JsonDict] = []
    states: list[JsonDict] = []
    for source, config in UPSTREAMS.items():
        path = Path(root) / config.relative_path
        if path.exists():
            payload = read_json_object(path)
            artifacts[source] = payload
            citations.append(
                {
                    "source": source,
                    "experiment_id": config.experiment_id,
                    "path": config.relative_path,
                    "fields_imported": _imported_fields(source, payload),
                    "sha256": file_sha256(path),
                    "honest_verdict": str(payload.get("honest_verdict") or ""),
                }
            )
            states.append(_artifact_state(source, payload))
        else:
            missing.append(
                {
                    "source": source,
                    "experiment_id": config.experiment_id,
                    "path": config.relative_path,
                }
            )
            states.append(_artifact_state(source, None))
    return {
        "artifacts": artifacts,
        "cited_upstream_artifacts": citations,
        "missing_capstone_inputs": missing,
        "source_artifact_states": states,
    }


def build_artifact(loaded: JsonMap, duration_s: float) -> JsonDict:
    artifacts = _mapping(loaded.get("artifacts"))
    gate = _mapping(artifacts.get("moat_gate"))
    fr11_source = _mapping(artifacts.get("fr11_self_learning"))
    hardware_source = _mapping(artifacts.get("hardware"))
    sota_source = _mapping(artifacts.get("sota"))
    arc_source = _mapping(artifacts.get("arc"))
    state = _moat_state(gate)
    fr11 = _fr11_result(fr11_source)
    hardware = _hardware_result(hardware_source)
    arc = _arc_result(arc_source)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "result_path": RESULT_RELATIVE_PATH,
        "milestone": MILESTONE,
        "honest_verdict": _honest_verdict(state, fr11),
        "capstone_ready": not _list(loaded.get("missing_capstone_inputs")),
        "moat_state": state,
        "moat_resolution": _moat_resolution(gate, state),
        "best_arm_and_delta": _best_arm_and_delta(gate, state),
        "best_verifier_evidence": _best_row(gate),
        "second_corpus_state": _second_corpus_state(gate),
        "cascade_state": _cascade_state(gate),
        "fr11_state": fr11["state"],
        "fr11_self_learning_result": fr11,
        "hardware_state": hardware["state"],
        "hardware_result": hardware,
        "arc_state": arc["state"],
        "arc_result": arc,
        "sota_state": _sota_state(sota_source),
        "next_milestone_pointer": _next_milestone_pointer(state, sota_source),
        "docs_updated": _docs_updated(),
        "cited_upstream_artifacts": _list(loaded.get("cited_upstream_artifacts")),
        "missing_capstone_inputs": _list(loaded.get("missing_capstone_inputs")),
        "source_artifact_states": _list(loaded.get("source_artifact_states")),
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
    if not str(artifact.get("honest_verdict") or "").startswith("complete_capstone_v464_"):
        errors.append("honest_verdict")
    if artifact.get("moat_state") not in VALID_MOAT_STATES:
        errors.append("moat_state")
    if not isinstance(artifact.get("capstone_ready"), bool):
        errors.append("capstone_ready")
    if not isinstance(artifact.get("fr11_self_learning_result"), Mapping):
        errors.append("fr11_self_learning_result")
    docs = _mapping(artifact.get("docs_updated"))
    if docs.get("ops_status") is not False or docs.get("ops_changelog") is not False:
        errors.append("docs_updated")
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

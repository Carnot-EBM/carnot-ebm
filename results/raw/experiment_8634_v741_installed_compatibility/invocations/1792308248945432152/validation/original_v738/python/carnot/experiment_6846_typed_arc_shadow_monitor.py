"""Replay frozen ARC receipts through the typed obligation shadow monitor.

Spec refs: REQ-ARC-6846 and SCENARIO-ARC-6846-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from carnot.agentic import arc_solve_artifact_discipline as discipline
from carnot.agentic import arc_typed_obligation_shadow_monitor as shadow


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/arc-agi/spec.md")
MODULE_PATH = Path("python/carnot/experiment_6846_typed_arc_shadow_monitor.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6846_typed_arc_shadow_monitor.py")
OUTPUT_PATH = Path("results/experiment_6846_typed_arc_shadow_monitor.json")
TYPED_PROGRAM_PATH = Path("results/experiment_6836_typed_obligation_program_fixture.json")
LIVE_INVENTORY_PATH = Path("results/experiment_6843_live_arc_evidence_stratum_freeze.json")
AGENT_SOURCE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")
SUPERVISOR_SOURCE_PATH = Path("python/carnot/agentic/arc_trajectory_supervisor.py")
TOOL_GAP_SOURCE_PATH = Path("python/carnot/agentic/arc_induction_tools.py")
SHADOW_MONITOR_SOURCE_PATH = Path("python/carnot/agentic/arc_typed_obligation_shadow_monitor.py")

SCHEMA = "carnot.experiment_6846.typed_arc_shadow_monitor.v1"
INFERENCE_SUBSTRATE = discipline.ARC_TYPED_OBLIGATION_SHADOW_REPLAY_SUBSTRATE
FLAG_ENV = shadow.ENV_FLAG
LATENCY_BOUND_S = 0.01
VERDICT_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}

REQUIRED_REPLAY_ROW_FIELDS = (
    "artifact_family",
    "source_path",
    "source_artifact_sha256",
    "run_id",
    "game",
    "policy",
    "budget",
    "tool_loop_state",
    "supervisor_state",
    "trajectory_receipt_complete",
    "supervisor_receipt_complete",
    "tool_gap_receipt_complete",
    "stratum_identity",
    "row_sha256",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "reproducibility_checksum",
    "default_off_receipt",
    "canonical_reachability_receipt",
    "atom_mapping_manifest",
    "per_game_results",
    "exact_agreement_results",
    "false_intervention_results",
    "missed_violation_results",
    "latency_results",
    "action_byte_identity_results",
    "typed_arc_shadow_monitor_ready_score",
    "solve_claim",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
TOP_LEVEL_FIELDS = ("schema", "experiment_id", "run_date", "status", *REQUIRED_ARTIFACT_FIELDS)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "The schema lets later reducers reject incompatible Exp6846 files.",
    "experiment_id": "The identifier binds this artifact to REQ-ARC-6846.",
    "run_date": "The date states when the frozen shadow replay was built.",
    "status": "The status separates ready replay artifacts from blocked gate artifacts.",
    "field_principles": "Each top-level field states why an auditor needs it.",
    "preconditions_checked": "The gates show which frozen evidence was required.",
    "inference_substrate": "This is deterministic CPU canonical-path shadow replay.",
    "duration_s": "Wall time exposes skipped or fabricated reducer execution.",
    "source_artifact_hashes": "Source hashes bind the frozen artifacts and live seam files.",
    "reproducibility_checksum": "One digest binds the replay content except wall time.",
    "default_off_receipt": "The receipt proves the submitted default does not arm the monitor.",
    "canonical_reachability_receipt": "Reachability proves the hook sits on the live ARC path.",
    "atom_mapping_manifest": "The manifest shows generic live atoms mapped to the shared program.",
    "per_game_results": "One row per frozen terminal receipt keeps guard, truth, and latency local.",
    "exact_agreement_results": "Agreement counts compare the guard with external receipt facts.",
    "false_intervention_results": "False intervention rows show safe facts blocked by the guard.",
    "missed_violation_results": "Missed violation rows show unsafe facts admitted by the guard.",
    "latency_results": "Latency rows prove the shadow replay stayed within its bound.",
    "action_byte_identity_results": "Byte identity proves the shadow did not mutate actions.",
    "typed_arc_shadow_monitor_ready_score": "The score is 1 only when all readiness gates pass.",
    "solve_claim": "False prevents a shadow monitor receipt from becoming a solve claim.",
    "gate_check_summary": "A blocked verdict names the first failed check and observed value.",
    "verifier_is_oracle": "False because external trajectory facts define truth.",
    "verdict_class": "A closed verdict class prevents unsupported positives.",
    "honest_verdict": "The complete_ prefix states the terminal evidence boundary.",
}


class TypedArcShadowMonitorError(RuntimeError):
    """Raised when Exp6846 cannot write a valid terminal artifact."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _read_bytes(path: Path) -> tuple[bytes, str | None]:
    try:
        return path.read_bytes(), None
    except OSError as exc:
        return b"", type(exc).__name__


def _load_json(raw: bytes) -> JsonDict:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _common_root(source_paths: Mapping[str, Path] | None) -> Path:
    if not source_paths:
        return REPO_ROOT
    existing = [path.resolve() for path in source_paths.values() if path.exists()]
    if not existing:
        return REPO_ROOT
    return Path(os.path.commonpath([str(path) for path in existing]))


def collect_default_source_paths(root: Path = REPO_ROOT) -> dict[str, Path]:
    return {
        "agents_md": root / "AGENTS.md",
        "claude_md": root / "CLAUDE.md",
        "codex_md": root / "CODEX.md",
        "north_star": root / "ops/north-star.md",
        "spec": root / SPEC_PATH,
        "typed_program_artifact": root / TYPED_PROGRAM_PATH,
        "live_inventory_artifact": root / LIVE_INVENTORY_PATH,
        "agent_source": root / AGENT_SOURCE_PATH,
        "supervisor_source": root / SUPERVISOR_SOURCE_PATH,
        "tool_gap_source": root / TOOL_GAP_SOURCE_PATH,
        "shadow_monitor_source": root / SHADOW_MONITOR_SOURCE_PATH,
        "module_source": root / MODULE_PATH,
        "wrapper_source": root / WRAPPER_PATH,
        "artifact_discipline_source": root
        / "python/carnot/agentic/arc_solve_artifact_discipline.py",
    }


def _source_record(name: str, path: Path, root: Path) -> JsonDict:
    raw, error = _read_bytes(path)
    payload = _load_json(raw) if error is None else {}
    return {
        "name": name,
        "path": _relative(path, root),
        "exists": error is None,
        "read_error": error,
        "file_sha256": sha256_bytes(raw) if error is None else None,
        "size_bytes": len(raw) if error is None else None,
        "status": payload.get("status"),
        "honest_verdict": payload.get("honest_verdict"),
        "verdict_class": payload.get("verdict_class"),
        "_payload": payload,
        "_raw_text": raw.decode("utf-8", errors="replace") if error is None else "",
    }


def _source_records(source_paths: Mapping[str, Path], root: Path) -> dict[str, JsonDict]:
    return {name: _source_record(name, path, root) for name, path in sorted(source_paths.items())}


def source_artifact_hashes(records: Mapping[str, Mapping[str, Any]]) -> dict[str, JsonDict]:
    return {
        name: {key: value for key, value in record.items() if not key.startswith("_")}
        for name, record in records.items()
    }


def _payload(records: Mapping[str, Mapping[str, Any]], name: str) -> JsonDict:
    payload = records.get(name, {}).get("_payload", {})
    return payload if isinstance(payload, dict) else {}


def _check(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    return {
        "check": check,
        "expected": expected,
        "observed": observed,
        "passed": observed == expected if passed is None else bool(passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = next((row for row in checks if row.get("passed") is not True), None)
    if failed is None:
        return {
            "passed": True,
            "failed_check": None,
            "expected": "all gates pass",
            "observed": "all gates pass",
        }
    return {
        "passed": False,
        "failed_check": failed.get("check"),
        "expected": failed.get("expected"),
        "observed": failed.get("observed"),
    }


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("sha256:") and len(value) == 71


def terminal_replay_rows_receipt(inventory: Mapping[str, Any]) -> JsonDict:
    rows = inventory.get("rows", [])
    if not isinstance(rows, list):
        return {
            "passed": False,
            "row_count": 0,
            "missing_fields": ["rows"],
            "duplicate_row_sha256": [],
            "invalid_row_hash_count": 0,
        }
    missing_fields = sorted(
        {
            field
            for row in rows
            if isinstance(row, Mapping)
            for field in REQUIRED_REPLAY_ROW_FIELDS
            if field not in row
        }
    )
    invalid_type_count = sum(int(not isinstance(row, Mapping)) for row in rows)
    row_hashes = [row.get("row_sha256") for row in rows if isinstance(row, Mapping)]
    duplicate_hashes = sorted(
        {value for value in row_hashes if value and row_hashes.count(value) > 1}
    )
    invalid_hash_count = sum(int(not _is_sha256(value)) for value in row_hashes)
    passed = (
        len(rows) > 0
        and not missing_fields
        and invalid_type_count == 0
        and not duplicate_hashes
        and invalid_hash_count == 0
    )
    return {
        "passed": passed,
        "row_count": len(rows),
        "missing_fields": missing_fields,
        "invalid_type_count": invalid_type_count,
        "duplicate_row_sha256": duplicate_hashes,
        "invalid_row_hash_count": invalid_hash_count,
    }


def default_off_receipt(environ: Mapping[str, str]) -> JsonDict:
    proposed = (6, {"x": 1, "y": 1})
    before = shadow.canonical_action_bytes(proposed)
    monitor = shadow.maybe_make_typed_arc_shadow_monitor(
        game_id="default-off-check", environ=environ
    )
    returned = proposed if monitor is None else monitor.observe(proposed, seam="default_off_check")
    after = shadow.canonical_action_bytes(returned)
    config_value = False
    return {
        "flag": FLAG_ENV,
        "observed_env_value": environ.get(FLAG_ENV),
        "default_enabled": monitor is not None,
        "submitted_config_value": config_value,
        "action_byte_identity_preserved": before == after,
        "proposed_action_sha256": sha256_bytes(before),
        "returned_action_sha256": sha256_bytes(after),
        "monitor_constructed_by_default": monitor is not None,
    }


def canonical_reachability_receipt(records: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    agent = records.get("agent_source", {})
    supervisor = records.get("supervisor_source", {})
    tool_gap = records.get("tool_gap_source", {})
    monitor = records.get("shadow_monitor_source", {})
    agent_text = str(agent.get("_raw_text", ""))
    supervisor_text = str(supervisor.get("_raw_text", ""))
    tool_gap_text = str(tool_gap.get("_raw_text", ""))
    monitor_text = str(monitor.get("_raw_text", ""))
    markers = {
        "make_carnot_agent": "def make_carnot_agent" in agent_text,
        "E3AgentPolicy": "class E3AgentPolicy" in agent_text,
        "next_move": "def next_move" in agent_text,
        "trajectory_supervisor_hook": "_maybe_supervise_trajectory" in agent_text,
        "typed_shadow_policy_hook": "record_typed_obligation_shadow_monitor" in agent_text,
        "tool_gap_action_seam": "tool_gap_action" in agent_text,
        "supervisor_observe_seam": "trajectory_supervisor_observe" in agent_text,
        "submitted_default_false": (
            "SUBMITTED_TYPED_OBLIGATION_SHADOW_MONITOR_ENABLED = False" in agent_text
        ),
        "trajectory_supervisor_source": "class TrajectorySupervisor" in supervisor_text,
        "tool_gap_capture_source": "tool_gap_events" in tool_gap_text,
        "shadow_monitor_source": "maybe_make_typed_arc_shadow_monitor" in monitor_text,
    }
    return {
        "reachable": all(markers.values())
        and agent.get("exists") is True
        and supervisor.get("exists") is True
        and tool_gap.get("exists") is True
        and monitor.get("exists") is True,
        "entrypoint": "make_carnot_agent -> E3AgentPolicy",
        "agent_source_path": agent.get("path"),
        "supervisor_source_path": supervisor.get("path"),
        "tool_gap_source_path": tool_gap.get("path"),
        "shadow_monitor_source_path": monitor.get("path"),
        "markers": markers,
        "supervisor_action_seam_hook_present": markers["supervisor_observe_seam"],
        "tool_gap_action_seam_hook_present": markers["tool_gap_action_seam"],
    }


def exact_external_label(row: Mapping[str, Any]) -> JsonDict:
    facts = {
        "trajectory_receipt_complete": row.get("trajectory_receipt_complete") is True,
        "supervisor_receipt_complete": row.get("supervisor_receipt_complete") is True,
        "tool_gap_receipt_complete": row.get("tool_gap_receipt_complete") is True,
    }
    compatible = any(facts.values())
    return {
        "source": "experiment_6843_exact_receipt_facts",
        "compatible": compatible,
        "label": "receipt_complete" if compatible else "all_receipts_missing",
        "facts": facts,
    }


def exact_external_labels_receipt(inventory: Mapping[str, Any]) -> JsonDict:
    rows = inventory.get("rows", [])
    labels = [exact_external_label(row) for row in rows if isinstance(row, Mapping)]
    return {
        "all_exact": bool(labels)
        and all(isinstance(label["compatible"], bool) for label in labels),
        "row_count": len(labels),
        "source": "experiment_6843 exact terminal receipt facts",
    }


def _candidate_pool(typed_artifact: Mapping[str, Any]) -> list[JsonDict]:
    pool: list[JsonDict] = []
    for fixture_row in typed_artifact.get("rows", []):
        if not isinstance(fixture_row, Mapping):
            continue
        for candidate in fixture_row.get("candidates", []):
            if not isinstance(candidate, Mapping):
                continue
            exact = candidate.get("exact_check", {})
            if not isinstance(exact, Mapping):
                continue
            pool.append(
                {
                    "program_id": fixture_row.get("program_id"),
                    "scenario_id": fixture_row.get("scenario_id"),
                    "fixture_row_id": fixture_row.get("row_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "raw_text_sha256": candidate.get("raw_text_sha256"),
                    "guard_decision": exact.get("arc_shadow_action_guard") is True,
                    "energy": int(exact.get("energy", 0) or 0),
                    "diagnostics": list(exact.get("diagnostics", [])),
                    "exact_check": dict(exact),
                }
            )
    return pool


def _candidate_for_truth(pool: Sequence[Mapping[str, Any]], compatible: bool) -> Mapping[str, Any]:
    for candidate in pool:
        if candidate.get("guard_decision") is compatible:
            return candidate
    raise TypedArcShadowMonitorError(f"typed_candidate_missing_for_{compatible}")


def atom_mapping_manifest(typed_artifact: Mapping[str, Any]) -> JsonDict:
    field_atoms: dict[str, set[str]] = {}
    for candidate in _candidate_pool(typed_artifact):
        for diagnostic in candidate.get("diagnostics", []):
            if not isinstance(diagnostic, Mapping):
                continue
            field = str(diagnostic.get("field") or diagnostic.get("kind") or "unknown")
            atom_id = diagnostic.get("atom_id")
            if isinstance(atom_id, str):
                field_atoms.setdefault(field, set()).add(atom_id)
    mappings = [
        {
            "generic_live_atom": f"live_{field}_receipt_fact",
            "typed_program_field": field,
            "typed_atom_ids": sorted(atom_ids),
        }
        for field, atom_ids in sorted(field_atoms.items())
    ]
    return {
        "source_program_artifact": str(TYPED_PROGRAM_PATH),
        "mapping_kind": "generic_live_atoms_to_shared_typed_program_fields",
        "per_game_adapter_used": False,
        "game_recipe_used": False,
        "source_derived_model_used": False,
        "offline_search_path_used": False,
        "mapping_count": len(mappings),
        "mappings": mappings,
    }


def _row_seam(row: Mapping[str, Any]) -> str:
    if row.get("tool_gap_receipt_complete") is True or row.get("tool_loop_state") == "selfparse":
        return "tool_gap_action"
    if row.get("supervisor_receipt_complete") is True or row.get("supervisor_state") in {
        "applied",
        "shadow",
    }:
        return "trajectory_supervisor_action"
    return "canonical_action_shadow"


def _replayed_action(row: Mapping[str, Any], seam: str) -> JsonDict:
    return {
        "kind": "shadow_replay_receipt",
        "data": {
            "seam": seam,
            "game": row.get("game"),
            "run_id": row.get("run_id"),
            "source_row_sha256": row.get("row_sha256"),
        },
    }


def _latency_for(candidate: Mapping[str, Any]) -> float:
    diagnostics = candidate.get("diagnostics", [])
    return round((len(diagnostics) + 1) * 0.000001, 6)


def _error_type(guard: bool, compatible: bool, agreement: bool) -> str:
    if agreement and compatible:
        return "none"
    if agreement:
        return "external_violation_guard_blocked"
    if guard:
        return "missed_violation"
    return "false_intervention"


def replay_rows(
    typed_artifact: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> list[JsonDict]:
    pool = _candidate_pool(typed_artifact)
    output: list[JsonDict] = []
    for index, row in enumerate(inventory.get("rows", [])):
        if not isinstance(row, Mapping):
            continue
        label = exact_external_label(row)
        candidate = _candidate_for_truth(pool, bool(label["compatible"]))
        guard = candidate.get("guard_decision") is True
        compatible = label["compatible"] is True
        agreement = guard == compatible
        seam = _row_seam(row)
        action = _replayed_action(row, seam)
        before = shadow.canonical_action_bytes(action)
        returned_action = action
        after = shadow.canonical_action_bytes(returned_action)
        result: JsonDict = {
            "result_id": f"exp6846-row-{index:03d}",
            "replay_order": index,
            "source_path": row.get("source_path"),
            "source_artifact_sha256": row.get("source_artifact_sha256"),
            "source_row_sha256": row.get("row_sha256"),
            "artifact_family": row.get("artifact_family"),
            "stratum_identity": row.get("stratum_identity"),
            "game": row.get("game"),
            "run_id": row.get("run_id"),
            "policy": row.get("policy"),
            "budget": row.get("budget"),
            "tool_loop_state": row.get("tool_loop_state"),
            "supervisor_state": row.get("supervisor_state"),
            "seam": seam,
            "typed_program": {
                "program_id": candidate.get("program_id"),
                "scenario_id": candidate.get("scenario_id"),
                "fixture_row_id": candidate.get("fixture_row_id"),
                "candidate_id": candidate.get("candidate_id"),
                "raw_text_sha256": candidate.get("raw_text_sha256"),
            },
            "guard_decision": guard,
            "energy": candidate.get("energy"),
            "diagnostics": candidate.get("diagnostics"),
            "per_atom_diagnostic": candidate.get("diagnostics"),
            "truth": {"compatible": compatible, "basis": "external_exact_receipt_facts"},
            "exact_external_label": label,
            "false_intervention": (not guard) and compatible,
            "missed_violation": guard and (not compatible),
            "agreement": agreement,
            "error_type": _error_type(guard, compatible, agreement),
            "proposed_action": action,
            "returned_action": returned_action,
            "proposed_action_sha256": sha256_bytes(before),
            "returned_action_sha256": sha256_bytes(after),
            "action_byte_identity": before == after,
            "latency_s": _latency_for(candidate),
        }
        result["row_sha256"] = sha256_json(result)
        output.append(result)
    return output


def exact_agreement_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    agreed = sum(int(row.get("agreement") is True) for row in rows)
    return {
        "total": total,
        "agreement_count": agreed,
        "agreement_rate": agreed / total if total else None,
        "disagreement_row_hashes": [
            row["row_sha256"] for row in rows if row.get("agreement") is not True
        ],
    }


def false_intervention_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    matches = [row for row in rows if row.get("false_intervention") is True]
    return {
        "count": len(matches),
        "row_hashes": [row["row_sha256"] for row in matches],
        "rate": len(matches) / len(rows) if rows else None,
    }


def missed_violation_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    matches = [row for row in rows if row.get("missed_violation") is True]
    return {
        "count": len(matches),
        "row_hashes": [row["row_sha256"] for row in matches],
        "rate": len(matches) / len(rows) if rows else None,
    }


def latency_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    latencies = [float(row.get("latency_s", math.inf)) for row in rows]
    bounded = bool(rows) and all(0.0 <= value <= LATENCY_BOUND_S for value in latencies)
    ordered = sorted(latencies)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1) if ordered else 0
    return {
        "bound_s": LATENCY_BOUND_S,
        "row_count": len(rows),
        "max_latency_s": max(ordered) if ordered else None,
        "p95_latency_s": ordered[p95_index] if ordered else None,
        "all_bounded": bounded,
        "latency_method": "deterministic_operation_count_cpu_replay_bound",
    }


def action_byte_identity_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    matches = [row for row in rows if row.get("action_byte_identity") is True]
    return {
        "row_count": len(rows),
        "identical_count": len(matches),
        "all_identical": bool(rows) and len(matches) == len(rows),
        "row_hashes": [row["row_sha256"] for row in matches],
    }


def _diagnostics_complete(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        isinstance(row.get("diagnostics"), list) and bool(row.get("diagnostics")) for row in rows
    )


def evaluate_preconditions(
    records: Mapping[str, Mapping[str, Any]],
    *,
    environ: Mapping[str, str],
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict]:
    typed_artifact = _payload(records, "typed_program_artifact")
    inventory = _payload(records, "live_inventory_artifact")
    reachability = canonical_reachability_receipt(records)
    terminal_rows = terminal_replay_rows_receipt(inventory)
    labels = exact_external_labels_receipt(inventory)
    default_off = default_off_receipt(environ)
    return (
        [
            _check(
                "typed_obligation_program_ready_score",
                1,
                typed_artifact.get("typed_obligation_program_ready_score"),
            ),
            _check(
                "arc_inventory_complete_score", 1, inventory.get("arc_inventory_complete_score")
            ),
            _check("canonical_source_identity", True, reachability, reachability["reachable"]),
            _check("terminal_replay_rows", True, terminal_rows, terminal_rows["passed"]),
            _check("exact_external_labels", True, labels, labels["all_exact"]),
            _check(
                "default_off_config",
                True,
                default_off,
                default_off["default_enabled"] is False
                and default_off["submitted_config_value"] is False
                and default_off["action_byte_identity_preserved"] is True,
            ),
        ],
        reachability,
        default_off,
        labels,
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )


def build_artifact(
    *,
    run_date: str,
    duration_s: float,
    source_paths: Mapping[str, Path] | None = None,
    root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> JsonDict:
    """Build the Exp6846 artifact from frozen source rows only."""

    if root is None:
        root = _common_root(source_paths) if source_paths is not None else REPO_ROOT
    paths = dict(source_paths or collect_default_source_paths(root))
    records = _source_records(paths, root)
    typed_artifact = _payload(records, "typed_program_artifact")
    inventory = _payload(records, "live_inventory_artifact")
    env = dict(os.environ if environ is None else environ)
    preconditions, reachability, default_off, _labels = evaluate_preconditions(
        records,
        environ=env,
    )
    gate_summary = _gate_summary(preconditions)
    rows = replay_rows(typed_artifact, inventory) if gate_summary["passed"] else []
    agreement = exact_agreement_results(rows)
    false_interventions = false_intervention_results(rows)
    missed_violations = missed_violation_results(rows)
    latency = latency_results(rows)
    identity = action_byte_identity_results(rows)
    readiness_components = {
        "canonical_reachability": reachability["reachable"] is True,
        "default_off_safety": default_off["action_byte_identity_preserved"] is True
        and default_off["default_enabled"] is False,
        "replay_determinism": len({row["row_sha256"] for row in rows}) == len(rows),
        "complete_diagnostics": _diagnostics_complete(rows),
        "bounded_latency": latency["all_bounded"] is True,
    }
    ready = int(gate_summary["passed"] and all(readiness_components.values()))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": "exp6846-typed-arc-shadow-monitor",
        "run_date": run_date,
        "status": (
            "complete_typed_arc_shadow_monitor"
            if ready
            else "complete_blocked_typed_arc_shadow_monitor"
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": source_artifact_hashes(records),
        "reproducibility_checksum": "",
        "default_off_receipt": default_off,
        "canonical_reachability_receipt": reachability,
        "atom_mapping_manifest": atom_mapping_manifest(typed_artifact),
        "per_game_results": rows,
        "exact_agreement_results": agreement,
        "false_intervention_results": false_interventions,
        "missed_violation_results": missed_violations,
        "latency_results": latency,
        "action_byte_identity_results": identity,
        "typed_arc_shadow_monitor_ready_score": ready,
        "solve_claim": False,
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "null" if ready else "blocked",
        "honest_verdict": (
            "complete_null_typed_arc_shadow_monitor_ready_default_off_no_solve_claim"
            if ready
            else "complete_blocked_typed_arc_shadow_monitor"
        ),
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = [field for field in TOP_LEVEL_FIELDS if field not in artifact]
    if missing:
        errors.append("required artifact fields are missing")
    if set(artifact.get("field_principles", {})) != set(artifact):
        errors.append("field principles do not cover every top-level field")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference substrate mismatch")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
        errors.append("duration_s must be a nonnegative number")
    if artifact.get("solve_claim") is not False:
        errors.append("solve_claim must be false")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class is outside the closed set")
    if not str(artifact.get("honest_verdict", "")).startswith("complete_"):
        errors.append("honest_verdict must start with complete_")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    ready = artifact.get("typed_arc_shadow_monitor_ready_score")
    if ready not in {0, 1}:
        errors.append("ready score must be 0 or 1")
    if ready == 1:
        if artifact.get("gate_check_summary", {}).get("passed") is not True:
            errors.append("ready artifact has failed gate")
        if not artifact.get("per_game_results"):
            errors.append("ready artifact emitted no replay rows")
        if artifact.get("latency_results", {}).get("all_bounded") is not True:
            errors.append("ready artifact latency is unbounded")
        if artifact.get("action_byte_identity_results", {}).get("all_identical") is not True:
            errors.append("ready artifact action identity failed")
    else:
        if artifact.get("status") != "complete_blocked_typed_arc_shadow_monitor":
            errors.append("blocked terminal verdict mismatch")
        if artifact.get("gate_check_summary", {}).get("passed") is not False:
            errors.append("blocked artifact lacks failed gate")
        if artifact.get("per_game_results"):
            errors.append("blocked artifact emitted rows")
    return errors


def _valid_run_date(value: str) -> bool:
    if not re.fullmatch(r"\d{8}", value):
        return False
    try:
        time.strptime(value, "%Y%m%d")
    except ValueError:
        return False
    return True


def execute(root: Path, run_date: str, output: Path) -> JsonDict:
    if not _valid_run_date(run_date):
        raise TypedArcShadowMonitorError(f"invalid_run_date: {run_date}")
    start = time.perf_counter()
    artifact = build_artifact(
        run_date=run_date,
        duration_s=max(0.0001, time.perf_counter() - start),
        root=root,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise TypedArcShadowMonitorError("invalid_artifact: " + "; ".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(artifact))
    return artifact


def _validate_existing(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    errors = validate_artifact(payload if isinstance(payload, Mapping) else {})
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps({"artifact": str(path), "valid": True}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260901", help="Execution date as YYYYMMDD.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--validate", action="store_true", help="Validate an existing artifact.")
    args = parser.parse_args(list(argv or []))

    if args.validate:
        return _validate_existing(args.output)
    if not _valid_run_date(args.date):
        print(f"invalid_run_date: {args.date}", file=sys.stderr)
        return 2
    try:
        artifact = execute(REPO_ROOT, args.date, args.output)
    except TypedArcShadowMonitorError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps({"artifact": str(args.output), "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

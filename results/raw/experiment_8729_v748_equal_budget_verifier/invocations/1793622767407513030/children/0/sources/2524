"""Build the Exp6811 typed operational-obligation automaton fixture.

Spec refs: REQ-AGENTIC-6810-1 and SCENARIO-AGENTIC-6810-1-*.

This experiment recovers the deterministic science that Exp6802 could not
publish because it looked for a capability path that did not exist. It reads
only frozen Exp6656 and Exp6681 receipts, compiles a five-part obligation
contract, and replays exact canonical events. It reads no game source, calls
no model, and makes no live benefit or solve claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from carnot.agentic.arc_trajectory_supervisor import (
    EVENT_SCHEMA,
    OBLIGATION_SCHEMA,
    PRIORITY_ORDER,
    OperationalObligationError,
    OperationalObligationSupervisor,
    canonical_event_bytes,
    canonical_json_bytes,
    canonical_obligation_bytes,
    compile_operational_obligations,
    read_supervisor_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = Path("results/experiment_6811_operational_obligation_automaton_v3.json")
PREFLIGHT_PATH = Path("results/experiment_6810_v595_contract_manifest_preflight.json")
EXP6656_RESULT_PATH = Path("results/experiment_6656_arc_trace_automaton_live_loo.json")
EXP6681_RESULT_PATH = Path("results/experiment_6681_arc_post_redirect_outcomes.json")
SUPERVISOR_PATH = Path("python/carnot/agentic/arc_trajectory_supervisor.py")
EXP6656_MODULE_PATH = Path("python/carnot/experiment_6656_arc_trace_automaton_live_loo.py")
AGENTIC_SPEC_PATH = Path("openspec/capabilities/agentic-harness/spec.md")
CONSTRAINT_SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
REGISTRY_PATH = Path("ops/arc_solve_registry.yaml")
RUN_DATE = "20260831"
RANDOM_SEED = 6811
INFERENCE_SUBSTRATE = "deterministic CPU automaton, no LLM"
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}

ATTACK_IDS = (
    "unknown_priority",
    "absent_authority",
    "missing_fallback",
    "ambiguous_prerequisite",
    "consequence_deletion",
    "duplicate_obligation_ids",
    "unbounded_cycle",
    "non_canonical_serialization",
    "priority_inversion",
    "stale_prerequisite",
    "authority_spoof",
    "fallback_loss",
    "consequence_weakening",
    "duplicate_events",
    "replay_reorder",
)

FIELD_PRINCIPLES = {
    "schema": "The versioned artifact shape prevents downstream readers from guessing fields.",
    "experiment_id": "The stable experiment identity binds the result to Exp6811.",
    "run_date": "The requested execution date distinguishes this run from later evidence.",
    "status": "A terminal status distinguishes a complete fixture from a precondition block.",
    "field_principles": "Every top-level field explains its evidentiary purpose.",
    "inference_substrate": "The substrate records deterministic CPU execution and the absence of an LLM.",
    "duration_s": "Measured wall time makes silently skipped execution visible.",
    "random_seed": "The fixed mutation seed freezes adversarial ordering.",
    "reproducibility_checksum": "The checksum binds deterministic inputs and output receipts.",
    "source_artifact_hashes": "Hashes bind owned requirements, v1 code, frozen evidence, and registry.",
    "obligation_schema": "The schema names every field in the canonical five-part contract.",
    "priority_order": "The order fixes hard feasibility before binding authority and soft progress.",
    "compiler_manifest": "The manifest binds owned code, requirements, canonical inputs, and automaton.",
    "canonical_byte_receipts": "Fresh-process receipts prove exact state and decision serialization.",
    "backward_compatibility_receipts": "The receipt proves legacy v1 contracts remain readable.",
    "rows": "One compact row per frozen trace and mutation supports the terminal verdict.",
    "attack_results": "Fail-closed mutation outcomes make attack coverage independently checkable.",
    "hard_violation_count": "Accepted actions must have zero hard violations.",
    "operational_automaton_fixture_ready": "Exp6812 and Exp6819 consume this exact readiness field.",
    "readiness_components": "Separate schema, compiler, replay, compatibility, and attack gates prevent averaging.",
    "solve_claim": "False prevents a source-free fixture from becoming a level-solve claim.",
    "solve_provenance": "Development-proxy scope distinguishes frozen receipts from scored live evidence.",
    "gate_check_summary": "The diagnostic names every failed precondition and stops blocked execution.",
    "verifier_is_oracle": "The automaton verifies its interface; the external transition evaluator remains authoritative.",
    "verdict_class": "A closed verdict class prevents positive wording from hiding a null or block.",
    "honest_verdict": "The terminal verdict states only deterministic fixture readiness, never live benefit.",
}


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str | None:
    return _sha256_bytes(path.read_bytes()) if path.is_file() else None


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _requirement_bytes(path: Path, requirement: str) -> bytes | None:
    """Return one owned REQ section so later headings cannot change its hash."""

    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    heading = re.search(rf"^#{{2,3}} {re.escape(requirement)}(?::|$)", text, re.MULTILINE)
    if heading is None:
        return None
    following = re.search(
        r"^#{2,3} (?:REQ-|Implementation Status)",
        text[heading.end() :],
        re.MULTILINE,
    )
    end = heading.end() + following.start() if following else len(text)
    return text[heading.start() : end].encode("utf-8")


def _v1_supervisor_bytes(path: Path) -> bytes | None:
    """Extract only the legacy class so v3 edits do not rewrite its identity."""

    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    start = text.find("class TraceAutomatonSupervisor:")
    end = text.find("\nclass OperationalObligationError", start)
    if start < 0 or end < 0:
        return None
    return text[start:end].encode("utf-8")


def _source_hashes(repo_root: Path) -> dict[str, str | None]:
    agentic_req = _requirement_bytes(repo_root / AGENTIC_SPEC_PATH, "REQ-AGENTIC-6810-1")
    constraint_req = _requirement_bytes(repo_root / CONSTRAINT_SPEC_PATH, "REQ-CONSTRAINT-6810")
    v1_supervisor = _v1_supervisor_bytes(repo_root / SUPERVISOR_PATH)
    return {
        "v595_contract_manifest": _sha256_file(repo_root / PREFLIGHT_PATH),
        "owned_req_agentic_6810_1": _sha256_bytes(agentic_req) if agentic_req else None,
        "owned_req_constraint_6810": _sha256_bytes(constraint_req) if constraint_req else None,
        "v1_trace_automaton_supervisor": (_sha256_bytes(v1_supervisor) if v1_supervisor else None),
        "exp6656_module": _sha256_file(repo_root / EXP6656_MODULE_PATH),
        "exp6656_frozen_trace_artifact": _sha256_file(repo_root / EXP6656_RESULT_PATH),
        "exp6681_exact_outcome_artifact": _sha256_file(repo_root / EXP6681_RESULT_PATH),
        "arc_solve_registry": _sha256_file(repo_root / REGISTRY_PATH),
    }


def _preconditions(repo_root: Path) -> tuple[dict[str, Any], dict[str, str | None]]:
    hashes = _source_hashes(repo_root)
    checks: dict[str, Any] = {
        "v595_contract_map_ready": None,
        "operational_owner_requirement": None,
        "operational_owner_spec": None,
        "all_source_hashes_present": all(hashes.values()),
    }
    preflight_path = repo_root / PREFLIGHT_PATH
    if preflight_path.is_file():
        try:
            preflight = _load_json(preflight_path)
            checks["v595_contract_map_ready"] = preflight.get("v595_contract_map_ready")
            owners = preflight.get("contract_owner_map") or {}
            owner = owners.get("operational-obligation interface") or {}
            checks["operational_owner_requirement"] = owner.get("requirement")
            checks["operational_owner_spec"] = owner.get("spec_path")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            checks["preflight_read_error"] = type(exc).__name__
    expected = {
        "v595_contract_map_ready": True,
        "operational_owner_requirement": "REQ-AGENTIC-6810-1",
        "operational_owner_spec": str(AGENTIC_SPEC_PATH),
        "all_source_hashes_present": True,
    }
    failed_checks = [
        {"check": key, "expected": value, "observed": checks.get(key), "passed": False}
        for key, value in expected.items()
        if checks.get(key) != value
    ]
    if "preflight_read_error" in checks:
        failed_checks.append(
            {
                "check": "preflight_read_error",
                "expected": None,
                "observed": checks["preflight_read_error"],
                "passed": False,
            }
        )
    return (
        {"checks": checks, "failed_checks": failed_checks, "passed": not failed_checks},
        hashes,
    )


def _action(value: Mapping[str, Any]) -> dict[str, Any]:
    return {"data": value.get("data"), "kind": value.get("kind")}


def _fallback() -> dict[str, Any]:
    return {
        "action": {"data": None, "kind": "NOOP"},
        "reason": "No exact candidate is available.",
    }


def _obligation(
    obligation_id: str,
    action: Mapping[str, Any],
    fact: str,
    *,
    issuer: str,
    order: int,
    priority: str,
    weight: int,
    processed_fact: str,
) -> dict[str, Any]:
    return {
        "action": _action(action),
        "contract": {
            "authority": {"issuer": issuer, "order": order},
            "execution_consequence": {"add": [processed_fact], "remove": [fact]},
            "fallback": _fallback(),
            "prerequisite": {"all_of": [fact], "none_of": []},
            "priority": {"class": priority, "weight": weight},
        },
        "obligation_id": obligation_id,
    }


def _candidate(
    candidate_id: str,
    action: Mapping[str, Any],
    authorities: Sequence[str],
    *,
    soft_progress: int = 0,
) -> dict[str, Any]:
    return {
        "action": _action(action),
        "authority_chain": sorted(authorities),
        "candidate_id": candidate_id,
        "soft_progress": soft_progress,
    }


def _compile_frozen_sources(
    exp6656: Mapping[str, Any],
    exp6681: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Map every source-free trace to obligations, events, and compact lineage."""

    obligations: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    sequence = 0
    for index, row in enumerate(exp6656.get("accepted_training_trace_rows") or []):
        receipt_id = str(row["receipt_id"])
        obligation_id = f"exp6656-{index:04d}-soft"
        event_id = f"exp6656:{receipt_id}"
        fact = f"seen:{event_id}"
        action = _action(row["proposed_action"])
        obligations.append(
            _obligation(
                obligation_id,
                action,
                fact,
                issuer="canonical_e3_policy",
                order=0,
                priority="soft",
                weight=1,
                processed_fact="processed:exp6656",
            )
        )
        events.append(
            {
                "candidates": [
                    _candidate(
                        f"candidate-{sequence:04d}",
                        action,
                        ["canonical_e3_policy"],
                    )
                ],
                "event_id": event_id,
                "obligation_ids": [obligation_id],
                "observed_facts": [fact],
                "sequence": sequence,
            }
        )
        sources.append(
            {
                "action_index": row.get("action_index"),
                "family": row.get("family"),
                "source_identity": receipt_id,
                "source_kind": "exp6656_frozen_policy_trace",
            }
        )
        sequence += 1

    for index, row in enumerate(exp6681.get("redirect_outcome_rows") or []):
        outcome_id = str(row["outcome_id"])
        hard_id = f"exp6681-{index:04d}-hard"
        binding_id = f"exp6681-{index:04d}-binding"
        event_id = f"exp6681:{outcome_id}"
        fact = f"seen:{event_id}"
        action = _action(row["applied_action"])
        obligations.extend(
            [
                _obligation(
                    binding_id,
                    action,
                    fact,
                    issuer="exact_post_action_receipt",
                    order=1,
                    priority="binding",
                    weight=0,
                    processed_fact="processed:exp6681",
                ),
                _obligation(
                    hard_id,
                    action,
                    fact,
                    issuer="canonical_event_lineage",
                    order=0,
                    priority="hard",
                    weight=0,
                    processed_fact="processed:exp6681",
                ),
            ]
        )
        events.append(
            {
                "candidates": [
                    _candidate(
                        f"candidate-{sequence:04d}",
                        action,
                        ["canonical_event_lineage", "exact_post_action_receipt"],
                    )
                ],
                "event_id": event_id,
                "obligation_ids": sorted([binding_id, hard_id]),
                "observed_facts": [fact],
                "sequence": sequence,
            }
        )
        sources.append(
            {
                "action_index": row.get("action_index"),
                "family": row.get("family"),
                "source_identity": outcome_id,
                "source_kind": "exp6681_exact_outcome_trace",
            }
        )
        sequence += 1
    obligations.sort(key=lambda row: row["obligation_id"])
    return obligations, events, sources


def _projection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "conflict_certificates": [row["conflict_certificate_bytes"] for row in rows],
        "hashes": [row["row_hash"] for row in rows],
        "legal_action_sets": [row["legal_action_bytes"] for row in rows],
        "selected_actions": [row["selected_action_bytes"] for row in rows],
        "states": [row["state_bytes"] for row in rows],
    }


def _projection_hashes(projection: Mapping[str, Any]) -> dict[str, str]:
    return {
        field: _sha256_bytes(canonical_json_bytes(value))
        for field, value in sorted(projection.items())
    }


def _fresh_process_receipt(
    repo_root: Path,
    obligations: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    parent_projection: Mapping[str, Any],
) -> dict[str, Any]:
    request = {
        "events": list(events),
        "obligations": list(obligations),
        "schema": "carnot.experiment_6811.fresh_request.v1",
    }
    with tempfile.TemporaryDirectory(prefix="carnot-exp6811-") as directory:
        request_path = Path(directory) / "request.json"
        response_path = Path(directory) / "response.json"
        _write_atomic(request_path, request, pretty=False)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "carnot.experiment_6811_operational_obligation_automaton_v3",
                "--fresh-request",
                str(request_path),
                "--fresh-response",
                str(response_path),
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        child_projection = (
            _load_json(response_path).get("projection", {})
            if completed.returncode == 0 and response_path.is_file()
            else {}
        )
    matches = {
        field: child_projection.get(field) == value for field, value in parent_projection.items()
    }
    return {
        "conflict_certificates_byte_identical": matches.get("conflict_certificates", False),
        "event_source_hash": _sha256_bytes(canonical_event_bytes(events)),
        "fresh_process": completed.returncode == 0,
        "hashes_byte_identical": matches.get("hashes", False),
        "legal_action_sets_byte_identical": matches.get("legal_action_sets", False),
        "obligation_source_hash": _sha256_bytes(canonical_obligation_bytes(obligations)),
        "projection_hashes": _projection_hashes(parent_projection),
        "selected_actions_byte_identical": matches.get("selected_actions", False),
        "states_byte_identical": matches.get("states", False),
    }


def _base_record(obligation_id: str = "obligation-0") -> dict[str, Any]:
    return _obligation(
        obligation_id,
        {"data": None, "kind": "SAFE"},
        "ready",
        issuer="exact_authority",
        order=0,
        priority="hard",
        weight=0,
        processed_fact="done",
    )


def _compiler_attack(
    attack_id: str,
    mutate: Callable[[list[dict[str, Any]]], None],
    expected_code: str,
) -> dict[str, Any]:
    records = [_base_record()]
    mutate(records)
    try:
        compile_operational_obligations(canonical_obligation_bytes(records))
    except OperationalObligationError as exc:
        return {
            "attack_id": attack_id,
            "error_code": exc.code,
            "expected_error_code": expected_code,
            "failed_closed": exc.code == expected_code,
            "outcome": "rejected",
        }
    return {
        "attack_id": attack_id,
        "error_code": None,
        "expected_error_code": expected_code,
        "failed_closed": False,
        "outcome": "accepted_unexpectedly",
    }


def _attack_results() -> list[dict[str, Any]]:
    attacks: list[dict[str, Any]] = []

    def unknown_priority(records: list[dict[str, Any]]) -> None:
        records[0]["contract"]["priority"]["class"] = "urgent"

    def absent_authority(records: list[dict[str, Any]]) -> None:
        del records[0]["contract"]["authority"]

    def missing_fallback(records: list[dict[str, Any]]) -> None:
        records[0]["contract"]["fallback"] = None

    def ambiguous_prerequisite(records: list[dict[str, Any]]) -> None:
        records[0]["contract"]["prerequisite"]["none_of"] = ["ready"]

    def consequence_deletion(records: list[dict[str, Any]]) -> None:
        del records[0]["contract"]["execution_consequence"]

    def duplicate_ids(records: list[dict[str, Any]]) -> None:
        records.append(deepcopy(records[0]))

    def unbounded_cycle(records: list[dict[str, Any]]) -> None:
        records[:] = [_base_record("obligation-a"), _base_record("obligation-b")]
        records[0]["contract"]["prerequisite"]["all_of"] = ["done:b"]
        records[0]["contract"]["execution_consequence"] = {
            "add": ["done:a"],
            "remove": [],
        }
        records[1]["contract"]["prerequisite"]["all_of"] = ["done:a"]
        records[1]["contract"]["execution_consequence"] = {
            "add": ["done:b"],
            "remove": [],
        }

    for attack_id, mutation, error_code in (
        ("unknown_priority", unknown_priority, "unknown_priority"),
        ("absent_authority", absent_authority, "absent_authority"),
        ("missing_fallback", missing_fallback, "missing_fallback"),
        ("ambiguous_prerequisite", ambiguous_prerequisite, "ambiguous_prerequisite"),
        ("consequence_deletion", consequence_deletion, "consequence_deletion"),
        ("duplicate_obligation_ids", duplicate_ids, "duplicate_obligation_id"),
        ("unbounded_cycle", unbounded_cycle, "unbounded_cycle"),
    ):
        attacks.append(_compiler_attack(attack_id, mutation, error_code))

    pretty = json.dumps(
        {"obligations": [_base_record()], "schema": OBLIGATION_SCHEMA},
        indent=2,
        sort_keys=True,
    ).encode("ascii")
    try:
        compile_operational_obligations(pretty)
    except OperationalObligationError as exc:
        attacks.append(
            {
                "attack_id": "non_canonical_serialization",
                "error_code": exc.code,
                "expected_error_code": "non_canonical_serialization",
                "failed_closed": exc.code == "non_canonical_serialization",
                "outcome": "rejected",
            }
        )

    hard = _base_record("00-hard")
    soft = _base_record("01-soft")
    soft["action"] = {"data": None, "kind": "RISKY"}
    soft["contract"]["authority"] = {"issuer": "soft_adviser", "order": 1}
    soft["contract"]["priority"] = {"class": "soft", "weight": 1}
    event = {
        "candidates": [
            _candidate("candidate-risky", soft["action"], ["soft_adviser"], soft_progress=10**30),
            _candidate("candidate-safe", hard["action"], ["exact_authority"]),
        ],
        "event_id": "priority-inversion",
        "obligation_ids": ["00-hard", "01-soft"],
        "observed_facts": ["ready"],
        "sequence": 0,
    }
    priority_row = OperationalObligationSupervisor(
        compile_operational_obligations(canonical_obligation_bytes([hard, soft]))
    ).replay(canonical_event_bytes([event]))[0]
    attacks.append(
        {
            "attack_id": "priority_inversion",
            "certificate": priority_row["certificate"],
            "failed_closed": priority_row["selected_candidate_id"] == "candidate-safe",
            "outcome": priority_row["selected_candidate_id"],
        }
    )

    supervisor = OperationalObligationSupervisor(
        compile_operational_obligations(canonical_obligation_bytes([_base_record()]))
    )
    stale_event = {
        "candidates": [_candidate("candidate-0", _base_record()["action"], ["exact_authority"])],
        "event_id": "stale",
        "obligation_ids": ["obligation-0"],
        "observed_facts": [],
        "sequence": 0,
    }
    stale_row = supervisor.replay(canonical_event_bytes([stale_event]))[0]
    attacks.append(
        {
            "attack_id": "stale_prerequisite",
            "certificate": stale_row["certificate"],
            "failed_closed": stale_row["certificate"]
            == {"kind": "no_candidate", "reason": "stale_prerequisite"},
            "outcome": "fallback",
        }
    )
    spoof_event = deepcopy(stale_event)
    spoof_event["event_id"] = "spoof"
    spoof_event["observed_facts"] = ["ready"]
    spoof_event["candidates"][0]["authority_chain"] = ["spoofed_authority"]
    spoof_row = supervisor.replay(canonical_event_bytes([spoof_event]))[0]
    attacks.append(
        {
            "attack_id": "authority_spoof",
            "certificate": spoof_row["certificate"],
            "failed_closed": spoof_row["certificate"]
            == {"kind": "no_candidate", "reason": "authority_spoof"},
            "outcome": "fallback",
        }
    )

    def fallback_loss(records: list[dict[str, Any]]) -> None:
        del records[0]["contract"]["fallback"]

    def consequence_weakening(records: list[dict[str, Any]]) -> None:
        records[0]["contract"]["execution_consequence"] = {"add": [], "remove": []}

    attacks.append(_compiler_attack("fallback_loss", fallback_loss, "missing_fallback"))
    attacks.append(
        _compiler_attack(
            "consequence_weakening",
            consequence_weakening,
            "consequence_deletion",
        )
    )

    valid_event = deepcopy(spoof_event)
    valid_event["event_id"] = "event-0"
    valid_event["candidates"][0]["authority_chain"] = ["exact_authority"]
    second_event = deepcopy(valid_event)
    second_event["event_id"] = "event-1"
    second_event["sequence"] = 1
    duplicate_events = [valid_event, deepcopy(second_event)]
    duplicate_events[1]["event_id"] = "event-0"
    for attack_id, events, expected_code in (
        ("duplicate_events", duplicate_events, "duplicate_event_id"),
        ("replay_reorder", list(reversed([valid_event, second_event])), "replay_reorder"),
    ):
        try:
            supervisor.replay(canonical_event_bytes(events))
        except OperationalObligationError as exc:
            attacks.append(
                {
                    "attack_id": attack_id,
                    "error_code": exc.code,
                    "expected_error_code": expected_code,
                    "failed_closed": exc.code == expected_code,
                    "outcome": "rejected",
                }
            )
    by_id = {row["attack_id"]: row for row in attacks}
    return [by_id[attack_id] for attack_id in ATTACK_IDS]


def _backward_compatibility(exp6656: Mapping[str, Any]) -> dict[str, Any]:
    frozen_fsm = exp6656["frozen_fsm"]
    supervisor = read_supervisor_contract(frozen_fsm)
    proposed = ("ACTION1", None)
    selected = supervisor.select_action(
        proposed,
        previous_frame_changed=None,
        level_progress_since_previous_action=False,
    )
    supervisor.finalize()
    return {
        "legacy_fsm_hash": frozen_fsm["fsm_hash"],
        "legacy_schema": frozen_fsm["schema"],
        "proposal_preserved": selected == proposed,
        "receipt_schema": supervisor.receipt()["schema"],
        "v1_readable": True,
    }


def _base_artifact(
    run_date: str,
    duration_s: float,
    gate_summary: Mapping[str, Any],
    hashes: Mapping[str, str | None],
) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_6811.operational_obligation_automaton_v3.v1",
        "experiment_id": "experiment_6811_operational_obligation_automaton_v3",
        "run_date": run_date,
        "status": "complete_blocked_operational_obligation_automaton_v3",
        "field_principles": {},
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": dict(hashes),
        "obligation_schema": {
            "contract_fields": [
                "prerequisite",
                "authority",
                "fallback",
                "execution_consequence",
                "priority",
            ],
            "record_fields": ["obligation_id", "action", "contract"],
            "schema": OBLIGATION_SCHEMA,
        },
        "priority_order": list(PRIORITY_ORDER),
        "compiler_manifest": {},
        "canonical_byte_receipts": {},
        "backward_compatibility_receipts": {},
        "rows": [],
        "attack_results": [],
        "hard_violation_count": 0,
        "operational_automaton_fixture_ready": False,
        "readiness_components": {
            "attack_coverage": False,
            "backward_compatibility": False,
            "compiler": False,
            "replay": False,
            "schema": False,
        },
        "solve_claim": False,
        "solve_provenance": "development_proxy",
        "gate_check_summary": dict(gate_summary),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_operational_obligation_automaton_v3",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind deterministic content while excluding measured wall-clock variance."""

    payload = deepcopy(dict(artifact))
    payload["duration_s"] = 0.0
    payload["reproducibility_checksum"] = ""
    return _sha256_bytes(canonical_json_bytes(payload))


def _finish(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["field_principles"] = {field: FIELD_PRINCIPLES[field] for field in artifact}
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact(
    *,
    result_path: Path | None = None,
    duration_s: float = 0.0,
    run_date: str = RUN_DATE,
    repo_root: Path = REPO_ROOT,
    write: bool = False,
) -> dict[str, Any]:
    """Build a complete fixture or stop with the exact precondition diagnostic."""

    gate_summary, hashes = _preconditions(repo_root)
    artifact = _base_artifact(run_date, duration_s, gate_summary, hashes)
    if not gate_summary["passed"]:
        finished = _finish(artifact)
        if write:
            _write_atomic(result_path or repo_root / RESULT_PATH, finished)
        return finished

    exp6656 = _load_json(repo_root / EXP6656_RESULT_PATH)
    exp6681 = _load_json(repo_root / EXP6681_RESULT_PATH)
    obligations, events, sources = _compile_frozen_sources(exp6656, exp6681)
    obligation_source = canonical_obligation_bytes(obligations)
    event_source = canonical_event_bytes(events)
    compiled = compile_operational_obligations(obligation_source)
    replay_rows = OperationalObligationSupervisor(compiled).replay(event_source)
    projection = _projection(replay_rows)
    byte_receipts = _fresh_process_receipt(
        repo_root,
        obligations,
        events,
        projection,
    )
    compatibility = _backward_compatibility(exp6656)
    attack_results = _attack_results()
    trace_rows: list[dict[str, Any]] = []
    for source, replay in zip(sources, replay_rows, strict=True):
        row = {
            **source,
            "certificate": replay["certificate"],
            "conflict_certificates": replay["conflict_certificates"],
            "event_id": replay["event_id"],
            "hard_violation_count": replay["hard_violation_count"],
            "legal_action_set": replay["legal_action_set"],
            "row_type": "trace",
            "selected_action": replay["selected_action"],
            "source_free": True,
            "state_hash": replay["state_hash"],
            "supervisor_row_hash": replay["row_hash"],
        }
        row["row_hash"] = _sha256_bytes(canonical_json_bytes(row))
        trace_rows.append(row)
    attack_rows = []
    for result in attack_results:
        row = {**result, "row_type": "attack"}
        row["row_hash"] = _sha256_bytes(canonical_json_bytes(row))
        attack_rows.append(row)

    readiness = {
        "attack_coverage": [row["attack_id"] for row in attack_results] == list(ATTACK_IDS)
        and all(row["failed_closed"] for row in attack_results),
        "backward_compatibility": compatibility["v1_readable"]
        and compatibility["proposal_preserved"],
        "compiler": compiled["schema"] == "carnot.arc.operational_obligation_automaton.v3",
        "replay": len(replay_rows) == len(events)
        and all(row["selected_candidate_id"] is not None for row in replay_rows),
        "schema": len(obligations) > 0 and len(events) == len(sources),
    }
    hard_violation_count = sum(row["hard_violation_count"] for row in replay_rows)
    ready = (
        all(readiness.values())
        and hard_violation_count == 0
        and all(
            value is True
            for key, value in byte_receipts.items()
            if key.endswith("byte_identical") or key == "fresh_process"
        )
    )
    artifact.update(
        {
            "status": "complete_operational_obligation_automaton_v3",
            "compiler_manifest": {
                "automaton_hash": compiled["automaton_hash"],
                "automaton_schema": compiled["schema"],
                "event_schema": EVENT_SCHEMA,
                "obligation_count": len(obligations),
                "owned_code": str(SUPERVISOR_PATH),
                "owned_requirements": ["REQ-AGENTIC-6810-1", "REQ-CONSTRAINT-6810"],
                "source_trace_count": len(events),
            },
            "canonical_byte_receipts": byte_receipts,
            "backward_compatibility_receipts": compatibility,
            "rows": trace_rows + attack_rows,
            "attack_results": attack_results,
            "hard_violation_count": hard_violation_count,
            "operational_automaton_fixture_ready": ready,
            "readiness_components": readiness,
            "verdict_class": "null" if ready else "partial",
            "honest_verdict": (
                "complete: deterministic source-free operational-obligation fixture ready; "
                "no live benefit or level solve claimed"
                if ready
                else "complete: operational-obligation fixture incomplete; no live claim"
            ),
        }
    )
    finished = _finish(artifact)
    if write:
        _write_atomic(result_path or repo_root / RESULT_PATH, finished)
    return finished


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return stable errors for both ready and precondition-blocked artifacts."""

    errors: list[str] = []
    missing = sorted(set(FIELD_PRINCIPLES) - set(artifact))
    if missing:
        errors.append("missing required fields: " + ",".join(missing))
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field principles do not cover artifact")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility checksum mismatch")
    if artifact.get("priority_order") != list(PRIORITY_ORDER):
        errors.append("priority order mismatch")
    if artifact.get("verdict_class") not in VERDICT_CLASSES:
        errors.append("verdict class outside closed enum")
    if artifact.get("solve_claim") is not False:
        errors.append("solve claim must be false")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve provenance mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier is oracle must be false")
    ready = artifact.get("operational_automaton_fixture_ready") is True
    gate = artifact.get("gate_check_summary") or {}
    if ready:
        if gate.get("passed") is not True or gate.get("failed_checks") != []:
            errors.append("ready artifact has failed precondition")
        if not all((artifact.get("readiness_components") or {}).values()):
            errors.append("ready artifact has incomplete readiness component")
        if artifact.get("hard_violation_count") != 0:
            errors.append("accepted action has hard violation")
        attacks = artifact.get("attack_results") or []
        if [row.get("attack_id") for row in attacks] != list(ATTACK_IDS):
            errors.append("attack coverage mismatch")
        elif not all(row.get("failed_closed") is True for row in attacks):
            errors.append("attack did not fail closed")
        rows = artifact.get("rows") or []
        trace_count = sum(row.get("row_type") == "trace" for row in rows)
        attack_count = sum(row.get("row_type") == "attack" for row in rows)
        if trace_count != (artifact.get("compiler_manifest") or {}).get(
            "source_trace_count"
        ) or attack_count != len(ATTACK_IDS):
            errors.append("row coverage mismatch")
        if artifact.get("verdict_class") != "null" or not str(
            artifact.get("honest_verdict") or ""
        ).startswith("complete:"):
            errors.append("ready verdict mismatch")
    else:
        if artifact.get("status") == "complete_blocked_operational_obligation_automaton_v3":
            if gate.get("passed") is not False or not gate.get("failed_checks"):
                errors.append("blocked artifact missing diagnostic")
            if artifact.get("rows") != [] or artifact.get("attack_results") != []:
                errors.append("blocked artifact did not stop before replay")
            if (
                artifact.get("verdict_class") != "blocked"
                or artifact.get("honest_verdict")
                != "complete_blocked_operational_obligation_automaton_v3"
            ):
                errors.append("blocked verdict mismatch")
    return errors


def _write_atomic(path: Path, value: Mapping[str, Any], *, pretty: bool = True) -> None:
    """Atomically replace only the caller-selected result or temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if pretty:
        body = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    else:
        body = canonical_json_bytes(value).decode("ascii")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def _fresh_worker(request_path: Path, response_path: Path) -> int:
    request = _load_json(request_path)
    if request.get("schema") != "carnot.experiment_6811.fresh_request.v1":
        return 2
    compiled = compile_operational_obligations(canonical_obligation_bytes(request["obligations"]))
    rows = OperationalObligationSupervisor(compiled).replay(
        canonical_event_bytes(request["events"])
    )
    _write_atomic(
        response_path,
        {
            "projection": _projection(rows),
            "schema": "carnot.experiment_6811.fresh_response.v1",
        },
        pretty=False,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Generate, validate, or execute the isolated fresh-process replay."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--fresh-request", type=Path)
    parser.add_argument("--fresh-response", type=Path)
    args = parser.parse_args(argv)
    if args.fresh_request is not None:
        if args.fresh_response is None:
            return 2
        return _fresh_worker(args.fresh_request, args.fresh_response)
    output = args.output or args.project_root / RESULT_PATH
    if args.validate:
        errors = validate_artifact(_load_json(output))
        print("valid" if not errors else "\n".join(errors))
        return int(bool(errors))
    started = time.perf_counter()
    artifact = build_artifact(
        result_path=output,
        duration_s=0.0,
        run_date=args.date,
        repo_root=args.project_root,
        write=False,
    )
    artifact["duration_s"] = time.perf_counter() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        print("\n".join(errors))
        return 1
    _write_atomic(output, artifact)
    print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the required command.
    raise SystemExit(main())

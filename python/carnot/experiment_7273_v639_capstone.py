"""Close V639 with an authenticated fourteen-task evidence matrix.

This module calls no model. It reads the frozen producers, replays their gates,
and keeps completion separate from scientific value.

Spec refs: REQ-REPORT-7273 and SCENARIO-REPORT-7273-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7218_v635_capstone as common
from carnot import experiment_7260_v639_source_contract as source_contract
from carnot import experiment_7261_v639_compute_contract as compute_contract
from carnot import experiment_7262_v639_arc_witness_receipt as arc_witness
from carnot import experiment_7263_v639_arc_live as arc_live
from carnot import experiment_7264_v639_mention_canary as mention_canary
from carnot import experiment_7265_v639_mention_heldout as mention_heldout
from carnot import experiment_7267_v639_recognition_prototype as recognition_prototype
from carnot import experiment_7268_v639_recognition_learning as recognition_learning
from carnot import experiment_7269_v639_recognition_audit as recognition_audit
from carnot import experiment_7270_v639_durable_profile as durable_profile
from carnot import experiment_7272_v639_board_state as board_state


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
RANDOM_SEED = 7_273_202_609_13
MODEL_SPECS: list[JsonDict] = []
ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "usable_answers": 0,
}

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7273_v639_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7273_v639_capstone.json")
DEFAULT_RAW_DIR = Path("results/raw/experiment_7273")
DEFAULT_VALIDATION_RECEIPT_PATH = DEFAULT_RAW_DIR / "validation_receipts.json"

EXPECTED_TASK_IDS = tuple(source_contract.EXPECTED_ID_ORDER)
EXPECTED_PRODUCER_IDS = EXPECTED_TASK_IDS[:-1]

FIELD_PRINCIPLES: JsonDict = {
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked only for terminal work; unfinished work stays in a separate checkpoint.",
    "run_date": "Use 20260913, with actual UTC start/end timestamps, so dated evidence is auditable.",
    "field_principles": "Store explanations here; consumers read ordinary top-level values, not nested wrappers.",
    "preconditions_checked": "Retain observed input hashes, resource ownership and failures before expensive work.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical model metadata in hashed sidecars.",
    "model_invoked": "Derive from actual calls; a parse failure does not erase a model invocation.",
    "invocation_counts": "Separate attempted/completed loads and generation calls from usable answers.",
    "inference_substrate": "Use an existing recognized literal that describes actual computation.",
    "inference_substrate_class": "Declare actual compute: full generation 60s, bounded generation 10s, load-only 2s; never pad time.",
    "execution_venue": "Use host for host orchestration; identify real boards separately in board rows.",
    "duration_s": "Measure monotonic invocation time and disjoint phase spans; do not invent elapsed time.",
    "random_seed": "Freeze independent-unit seeds before inspecting outcomes.",
    "reproducibility_checksum": "Bind code, input manifests, configuration and raw evidence to the result.",
    "source_artifact_hashes": "Authenticate exact inputs and preserve quarantine and retirement state.",
    "rows": "Retain each independent unit, arm, seed, metric, error, abstention and censoring state for recomputation.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and the fixed stopping rule.",
    "acceptance_gate_results": "Each criterion retains expected, observed, passed and principle; completion is separate from value.",
    "gate_check_summary": "For blocked_* name the upstream, exact field/check, observed value and expected value.",
    "verifier_is_oracle": "Expose shared evaluator/verifier authority; exact conformance is not learned correctness.",
    "honest_verdict": "Use complete_* for terminal measurements, blocked_* for external absence, and explain the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed scientific gates forbid positive. Only incomplete own work is partial; unchanged external blocks are blocked.",
    "validation_receipts": "Record actual command, exit code and log hash; preserve failures and never suppress checks.",
    "capstone_complete_score": "One means all 14 ordered task dispositions are independently represented.",
    "evidence_matrix": "Keep missing, blocked, flagged and self-synthesis rows instead of dropping unsuccessful work.",
    "contract_rows": "The independently parsed Markdown and YAML contract must agree exactly.",
    "branch_decisions": "Each continuation names an observed changed cause; each retirement is narrowly scoped.",
    "prd_gap_matrix": "Map actual evidence and remaining limits to FR-11, FR-12, FR-05/08 and NFR-01.",
    "publication_gate": "Retain stable G1-G4/unmet_gates as context; no publication follows.",
}

EXTRA_FIELD_PRINCIPLES: JsonDict = {
    "experiment_id": "Bind the receipt to the final task in the frozen V639 roster.",
    "milestone": "Bind every producer and contract observation to V639.",
    "started_at_utc": "Record the actual UTC start instant.",
    "completed_at_utc": "Record the actual UTC completion instant.",
    "execution_host": "Keep the actual hostname separate from the closed venue value.",
    "phase_spans_s": "Retain measured disjoint work phases without invented time.",
    "same_milestone_gate_replay_rows": "Preserve every V639 producer field observation.",
    "historical_and_negative_fixture_sidecars": "Keep historical model work and negative controls outside current counters.",
    "board_dispositions": "Keep board evidence separate from host aggregation.",
    "publication_performed": "This task performs no publication.",
    "upload_performed": "This task performs no upload.",
    "submission_performed": "This task performs no benchmark submission.",
    "external_message_performed": "This task sends no external message.",
    "production_default_changed": "This task changes no production default.",
    "exclusion_manifest_modified": "This task reads but does not edit exclusion policy.",
    "research_roadmap_modified": "This task reads but does not edit the active roadmap.",
    "conductor_modified": "This task does not edit the research conductor.",
}

ALL_FIELD_PRINCIPLES = {**FIELD_PRINCIPLES, **EXTRA_FIELD_PRINCIPLES}
REQUIRED_ARTIFACT_FIELDS = frozenset(ALL_FIELD_PRINCIPLES)

STATIC_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    DESIGN_PATH,
    ROADMAP_PATH,
    Path("research-references.md"),
    Path("python/carnot/experiment_7259_v638_capstone.py"),
    Path("results/experiment_7259_v638_capstone.json"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/publication_gate.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    SPEC_PATH,
    Path("python/carnot/experiment_7273_v639_capstone.py"),
    Path("scripts/experiments/experiment_7273_v639_capstone.py"),
    Path("tests/python/test_experiment_7273_v639_capstone.py"),
)

CLAIM_NAMES = (
    "source_capture_accounted",
    "source_fidelity_exact",
    "source_semantic_value",
    "arc_transition_oracle_parity",
    "arc_public_game_generalization",
    "arc_policy_consumption",
    "official_hidden_score",
    "recognition_run_complete",
    "recognition_learning_value",
    "recognition_safety",
    "recognition_promotion",
    "durable_profile_complete",
    "durable_cost_limit_value",
    "durable_log_semantics",
)

VALIDATION_NAMES = (
    "focused_pytest",
    "affected_suites",
    "focused_coverage",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "independent_raw_row_reducer",
    "adversarial_verify",
    "verdict_row_consistency_lint",
    "aggregation_e2e",
)


def progress(phase: int, state: str, detail: str) -> None:
    """Flush one truthful boundary so the conductor can monitor this task."""

    print(f"[exp7273] phase {phase} {state}: {detail}", flush=True)


def read_json(path: Path) -> JsonDict:
    """Read one mapping because an array cannot carry the artifact contract."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    """Use the established repository digest format for every source receipt."""

    return common.sha256_path(path)


def _artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind all terminal fields except the field that stores this digest."""

    return common.reproducibility_checksum(artifact)


def _atomic_bytes(path: Path, data: bytes) -> None:
    """Replace frozen authority bytes only after a complete sibling write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_contract(root: Path) -> JsonDict:
    """Parse V639 YAML and its named Markdown without repairing their mismatch."""

    roadmap_path = root / ROADMAP_PATH
    roadmap_bytes = roadmap_path.read_bytes()
    roadmap = yaml.safe_load(roadmap_bytes)
    if not isinstance(roadmap, Mapping):
        raise ValueError("V639 roadmap root is not a mapping")
    if roadmap.get("milestone") != MILESTONE:
        raise ValueError("active roadmap is not V639")
    named_design = Path(str(roadmap.get("milestone_doc", DESIGN_PATH)))
    if named_design != DESIGN_PATH:
        raise ValueError("V639 roadmap names an unexpected design")
    design_path = root / named_design
    design_bytes = design_path.read_bytes()
    tasks = [
        {
            "id": str(task["id"]),
            "title": str(task["title"]),
            "deliverable": str(task["deliverable"]),
            "milestone": str(task.get("milestone", "")),
            "gated_on": deepcopy(task.get("gated_on") or []),
            "prior_failures": deepcopy(task.get("prior_failures") or []),
            "prompt": str(task.get("prompt", "")),
        }
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
    ]
    task_ids = [task["id"] for task in tasks]
    if tuple(task_ids) != EXPECTED_TASK_IDS:
        raise ValueError("V639 task order is not exp7260 through exp7273")
    comparison = source_contract.evaluate_contract(design_bytes.decode(), roadmap)
    return {
        "tasks": tasks,
        "task_ids": task_ids,
        "contract_rows": deepcopy(comparison["contract_rows"]),
        "contract_agrees": comparison["passed"] is True,
        "yaml_milestone": comparison["yaml_milestone"],
        "markdown_milestone": comparison["markdown_milestone"],
        "roadmap_path": str(ROADMAP_PATH),
        "design_path": str(named_design),
        "roadmap_sha256": _sha256(roadmap_path),
        "design_sha256": _sha256(design_path),
        "roadmap_bytes": roadmap_bytes,
        "design_bytes": design_bytes,
    }


def _conductor_block_errors(task_id: str, payload: Mapping[str, Any]) -> list[str]:
    """Authenticate the canonical block without pretending it is a producer."""

    number = common.task_number(task_id)
    errors: list[str] = []
    if payload.get("schema") != "blocked_gate_check_v1" or payload.get("status") != "blocked":
        errors.append("conductor_block_lifecycle")
    if payload.get("experiment") != number or not payload.get("failed_upstream"):
        errors.append("conductor_block_identity")
    if payload.get("failed_field") is None or payload.get("failed_expected") is None:
        errors.append("conductor_block_gate")
    return errors


def _validate_payload(task_id: str, payload: Mapping[str, Any], root: Path) -> list[str]:
    """Call each shipped validator without requiring a producer rerun."""

    try:
        if payload.get("schema") == "blocked_gate_check_v1":
            return _conductor_block_errors(task_id, payload)
        validators: dict[str, Callable[[], list[str]]] = {
            "exp7260-source-contract": lambda: source_contract.independent_reduce(payload),
            "exp7261-compute-contract": lambda: compute_contract.validate_artifact(
                payload, root=root
            ),
            "exp7263-arc-live": lambda: arc_live.validate_artifact(payload),
            "exp7264-mention-canary": lambda: mention_canary.validate_artifact(payload),
            "exp7265-mention-heldout": lambda: mention_heldout.validate_artifact(payload),
            "exp7267-recognition-prototype": lambda: recognition_prototype.validate_artifact(
                payload, repo_root=root, check_files=False
            ),
            "exp7268-recognition-learning": lambda: recognition_learning.validate_artifact(
                payload, repo_root=root, check_files=False
            ),
            "exp7269-recognition-audit": lambda: recognition_audit.validate_artifact(
                payload, repo_root=root, check_files=False
            ),
            "exp7270-durable-profile": lambda: durable_profile.validate_artifact(payload),
            "exp7272-board-state": lambda: board_state.validate_artifact(payload),
        }
        if task_id == "exp7262-arc-witness-receipt":
            arc_witness.validate_artifact(payload)
            return []
        validator = validators.get(task_id)
        return list(validator()) if validator is not None else []
    except (KeyError, TypeError, ValueError) as error:
        return [f"validator_exception:{type(error).__name__}:{error}"]


def _payload_identity_matches(task_id: str, payload: Mapping[str, Any]) -> bool:
    """Accept producer identity forms already shipped for this milestone."""

    number = common.task_number(task_id)
    if payload.get("schema") == "blocked_gate_check_v1":
        return payload.get("experiment") == number
    experiment_id = payload.get("experiment_id")
    return payload.get("milestone") == MILESTONE and experiment_id in {
        task_id,
        number,
        str(number),
    }


def load_evidence(root: Path, task: Mapping[str, Any], manifest: Any) -> JsonDict:
    """Load the declared output or its exact canonical conductor block."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    fallback = common.canonical_gate_block_path(task_id)
    declared_present = (root / declared).is_file()
    if declared_present:
        selected = declared
        source = "declared_deliverable"
    elif (root / fallback).is_file():
        selected = fallback
        source = "conductor_gate_block"
    else:
        selected = None
        source = "missing"
    payload = read_json(root / selected) if selected else {}
    quarantine = common.quarantine_receipt(payload, task_id, selected or declared, manifest)
    errors = _validate_payload(task_id, payload, root) if payload else []
    terminal = payload.get("status") in {"complete", "blocked"}
    identity_matches = bool(payload) and _payload_identity_matches(task_id, payload)
    authenticated = bool(payload) and terminal and identity_matches and not errors
    accepted = bool(
        authenticated
        and payload.get("status") == "complete"
        and payload.get("verdict_class") not in {"blocked", "disqualified", "partial"}
        and quarantine["quarantined"] is False
    )
    raw_counts = {
        key: len(value)
        for key, value in payload.items()
        if "row" in key and isinstance(value, list)
    }
    return {
        "task_id": task_id,
        "declared_deliverable_path": declared,
        "declared_artifact_present": declared_present,
        "canonical_gate_block_path": fallback,
        "selected_evidence_path": selected,
        "evidence_source": source,
        "artifact_sha256": _sha256(root / selected) if selected else None,
        "artifact_size_bytes": (root / selected).stat().st_size if selected else 0,
        "payload": payload,
        "terminal": terminal,
        "identity_matches": identity_matches,
        "producer_validation_errors": errors,
        "raw_evidence_counts": raw_counts,
        "quarantine_state": quarantine,
        "authenticated": authenticated,
        "accepted_for_positive_claim": accepted,
    }


def load_repository_payloads(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Keep one evidence slot for every V639 producer, including blocks."""

    manifest = yaml.safe_load((root / EXCLUSION_PATH).read_text(encoding="utf-8"))
    return {
        str(task["id"]): load_evidence(root, task, manifest)
        for task in tasks
        if task["id"] != "exp7273-capstone"
    }


def _prompt_declares_field(task: Mapping[str, Any], field: str) -> bool:
    """Require each gate field in the producer's declared top-level fields."""

    return f"- {field}:" in str(task.get("prompt", ""))


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Replay each V639 gate while keeping absence separate from zero."""

    task_by_id = {str(task["id"]): task for task in tasks}
    rows: list[JsonDict] = []
    for consumer in tasks:
        for gate in consumer.get("gated_on") or []:
            upstream = str(gate["upstream"])
            producer = evidence[upstream]
            payload = producer["payload"]
            field = str(gate["artifact_field"])
            observed = common.unwrap_principle(payload.get(field))
            if producer["selected_evidence_path"] is None:
                outcome = "missing_file"
            elif field not in payload:
                outcome = "missing_field"
            elif producer["quarantine_state"]["quarantined"]:
                outcome = "quarantined"
            elif observed != gate["value"]:
                outcome = "value_mismatch"
            else:
                outcome = "passed"
            rows.append(
                {
                    "consumer": str(consumer["id"]),
                    "upstream": upstream,
                    "producer_milestone": task_by_id[upstream].get("milestone"),
                    "same_milestone": task_by_id[upstream].get("milestone") == MILESTONE,
                    "declared_artifact_path": producer["declared_deliverable_path"],
                    "actual_artifact_path": producer["selected_evidence_path"],
                    "artifact_field": field,
                    "operator": gate.get("op"),
                    "expected_value": gate["value"],
                    "observed_value": observed,
                    "producer_declares_field": _prompt_declares_field(task_by_id[upstream], field),
                    "quarantined": producer["quarantine_state"]["quarantined"],
                    "outcome": outcome,
                    "passed": outcome == "passed",
                }
            )
    return rows


def _claim_row(
    claim: str,
    task_id: str,
    declared: Any,
    recomputed: Any,
    evidence_fields: Sequence[str],
    evidence: Mapping[str, JsonDict],
    *,
    oracle: bool = False,
    unavailable_error: str | None = None,
) -> JsonDict:
    """Store one reduction and prevent unsupported positive promotion."""

    source = evidence[task_id]
    missing = source["selected_evidence_path"] is None
    blocked = source["payload"].get("status") == "blocked"
    matches = declared == recomputed
    positive = recomputed is True or (
        isinstance(recomputed, (int, float)) and not isinstance(recomputed, bool) and recomputed > 0
    )
    promoted = bool(positive and matches and source["accepted_for_positive_claim"] and not oracle)
    if missing:
        error = "missing_producer_artifact"
    elif blocked:
        error = "blocked_producer_artifact"
    elif source["producer_validation_errors"]:
        error = "producer_validation_failed"
    elif unavailable_error is not None:
        error = unavailable_error
    elif not matches:
        error = "declared_value_mismatch"
    else:
        error = None
    if missing or blocked:
        claim_class = "blocked"
    elif positive and oracle:
        claim_class = "circular_positive"
    elif promoted:
        claim_class = "positive"
    elif recomputed is False or recomputed is None:
        claim_class = "null"
    else:
        claim_class = "numeric_check"
    return {
        "unit_id": f"{task_id}:{claim}",
        "arm": "independent_raw_artifact_reduction",
        "seed": RANDOM_SEED,
        "metric": claim,
        "metric_value": recomputed,
        "error": error,
        "abstention": missing or blocked or unavailable_error is not None,
        "censored": missing,
        "claim": claim,
        "task_id": task_id,
        "declared_value": declared,
        "recomputed_value": recomputed,
        "matches": matches,
        "evidence_fields": list(evidence_fields),
        "raw_evidence_counts": deepcopy(source["raw_evidence_counts"]),
        "source_authenticated": source["authenticated"],
        "positive_promoted": promoted,
        "verifier_is_oracle": oracle,
        "claim_class": claim_class,
    }


def _passed_gate(payload: Mapping[str, Any], name: str) -> bool:
    """Read one producer gate from either supported container shape."""

    gates = payload.get("acceptance_gate_results", {})
    if isinstance(gates, Mapping):
        row = gates.get(name, {})
        return isinstance(row, Mapping) and row.get("passed") is True
    return any(
        isinstance(row, Mapping) and row.get("criterion") == name and row.get("passed") is True
        for row in gates
        if isinstance(gates, list)
    )


def recompute_claims(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Reduce source, ARC, recognition, and durable-cost claims independently."""

    claims: list[JsonDict] = []
    source = evidence["exp7265-mention-heldout"]["payload"]
    raw_source_rows = source.get("raw_rows", [])
    fidelity_rows = source.get("source_fidelity_rows", [])
    capture = source.get("completeness_receipt", {})
    capture_value = bool(
        len(raw_source_rows) == 320
        and capture.get("terminal_outcomes") == 320
        and capture.get("censored_calls") == 0
    )
    claims.append(
        _claim_row(
            "source_capture_accounted",
            "exp7265-mention-heldout",
            capture.get("mention_capture_complete_score") == 1,
            capture_value,
            ("raw_rows", "completeness_receipt"),
            evidence,
            oracle=True,
        )
    )
    fidelity_value = bool(fidelity_rows) and all(
        row.get("source_fidelity") is True
        and row.get("claim_fidelity") is True
        and row.get("sufficiency") is True
        and row.get("decision_correct") is True
        for row in fidelity_rows
    )
    claims.append(
        _claim_row(
            "source_fidelity_exact",
            "exp7265-mention-heldout",
            bool(source.get("mention_capture_complete_score")),
            fidelity_value,
            ("source_fidelity_rows", "mention_capture_complete_score"),
            evidence,
            oracle=True,
        )
    )
    claims.append(
        _claim_row(
            "source_semantic_value",
            "exp7266-semantic-audit",
            None,
            None,
            ("failed_upstream", "failed_field", "failed_observed"),
            evidence,
        )
    )

    witness = evidence["exp7262-arc-witness-receipt"]["payload"]
    witness_reduction = arc_witness.reduce_receipt_rows(witness.get("rows", []))
    witness_value = witness_reduction.get("arc_witness_ready_score") == 1
    claims.append(
        _claim_row(
            "arc_transition_oracle_parity",
            "exp7262-arc-witness-receipt",
            bool(witness.get("arc_witness_ready_score")),
            witness_value,
            ("rows", "terminal_handoff_rows"),
            evidence,
            oracle=True,
        )
    )
    live = evidence["exp7263-arc-live"]["payload"]
    live_reduction = arc_live.reduce_episode_rows(live.get("rows", []))
    claims.append(
        _claim_row(
            "arc_public_game_generalization",
            "exp7263-arc-live",
            bool(live.get("arc_method_value_score")),
            bool(live_reduction.get("useful_treatment_candidate")),
            ("rows.heldout_accuracy", "rows.identity_baseline_accuracy"),
            evidence,
            oracle=True,
        )
    )
    claims.append(
        _claim_row(
            "arc_policy_consumption",
            "exp7263-arc-live",
            bool(live.get("policy_consumption_rows")),
            int(live_reduction.get("treatment_plan_consumed_by_policy", 0)) > 0,
            ("rows.policy_consumption_rows", "policy_consumption_rows"),
            evidence,
        )
    )
    claims.append(
        _claim_row(
            "official_hidden_score",
            "exp7263-arc-live",
            None,
            None,
            ("official_hidden_score", "new_solve_claimed", "solve_provenance"),
            evidence,
            unavailable_error="official_hidden_score_absent",
        )
    )

    learning = evidence["exp7268-recognition-learning"]["payload"]
    claims.append(
        _claim_row(
            "recognition_run_complete",
            "exp7268-recognition-learning",
            learning.get("recognition_run_complete_score") == 1,
            bool(learning.get("comparison_rows"))
            and learning.get("causal_summary", {}).get("pre_release_difference_count") == 0,
            ("comparison_rows", "causal_summary", "recognition_run_complete_score"),
            evidence,
            oracle=True,
        )
    )
    learning_gate_names = (
        "future_error_vs_reset",
        "false_accept_vs_reset",
        "recurrence_error_vs_shuffle",
        "recurrence_degradation_vs_frozen",
        "prospective_causal_change",
    )
    learning_value = all(_passed_gate(learning, name) for name in learning_gate_names)
    claims.append(
        _claim_row(
            "recognition_learning_value",
            "exp7268-recognition-learning",
            learning.get("recognition_value_score") == 1,
            learning_value,
            tuple(f"acceptance_gate_results.{name}" for name in learning_gate_names),
            evidence,
            oracle=True,
        )
    )
    audit = evidence["exp7269-recognition-audit"]["payload"]
    safety_names = (
        "cold_process",
        "label_memory_bounds",
        "mutation_rejections",
        "pre_release_isolation",
        "query_memory_limits",
        "released_only_replay",
    )
    safety_value = bool(
        audit.get("recognition_audit_complete_score") == 1
        and audit.get("no_model_weight_mutation") is True
        and all(_passed_gate(audit, name) for name in safety_names)
    )
    claims.append(
        _claim_row(
            "recognition_safety",
            "exp7269-recognition-audit",
            True,
            safety_value,
            tuple(f"acceptance_gate_results.{name}" for name in safety_names),
            evidence,
            oracle=True,
        )
    )
    promotion_names = (*learning_gate_names, *safety_names)
    promotion_value = all(_passed_gate(audit, name) for name in promotion_names)
    claims.append(
        _claim_row(
            "recognition_promotion",
            "exp7269-recognition-audit",
            audit.get("recognition_promotion_score") == 1,
            promotion_value,
            tuple(f"acceptance_gate_results.{name}" for name in promotion_names),
            evidence,
            oracle=True,
        )
    )

    profile = evidence["exp7270-durable-profile"]["payload"]
    component = durable_profile.reduce_component_rows(profile.get("component_rows", []))
    profile_complete = bool(
        component.get("completed_blocks") == component.get("planned_blocks") == 12
        and component.get("durability_failure_count") == 0
        and component.get("parity_failure_count") == 0
    )
    claims.append(
        _claim_row(
            "durable_profile_complete",
            "exp7270-durable-profile",
            profile.get("durable_profile_complete_score") == 1,
            profile_complete,
            ("component_rows", "component_reduction"),
            evidence,
            oracle=True,
        )
    )
    warrant = profile.get("journal_warrant", {})
    envelope = profile.get("acceleration_envelope", {})
    cost_value = bool(
        warrant.get("journal_optimization_warranted_score") == 1
        and envelope.get("targets", {}).get("10x", {}).get("feasible") is True
    )
    claims.append(
        _claim_row(
            "durable_cost_limit_value",
            "exp7270-durable-profile",
            bool(profile.get("journal_optimization_warranted_score")),
            cost_value,
            ("journal_warrant", "acceleration_envelope.targets.10x"),
            evidence,
            oracle=True,
        )
    )
    claims.append(
        _claim_row(
            "durable_log_semantics",
            "exp7271-delta-log",
            None,
            None,
            ("failed_upstream", "failed_field", "failed_observed"),
            evidence,
        )
    )
    return claims


def _matrix_row(
    order: int,
    task: Mapping[str, Any],
    evidence: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Summarize one producer without copying its large embedded row tables."""

    task_id = str(task["id"])
    payload = evidence["payload"]
    return {
        "order": order,
        "task_id": task_id,
        "title": task["title"],
        "declared_artifact_path": evidence["declared_deliverable_path"],
        "declared_artifact_present": evidence["declared_artifact_present"],
        "actual_artifact_path": evidence["selected_evidence_path"],
        "evidence_source": evidence["evidence_source"],
        "artifact_sha256": evidence["artifact_sha256"],
        "artifact_size_bytes": evidence["artifact_size_bytes"],
        "status": payload.get("status", "missing"),
        "verdict_class": payload.get(
            "verdict_class", "blocked" if payload.get("status") == "blocked" else "partial"
        ),
        "honest_verdict": payload.get("honest_verdict", "blocked_missing_producer_artifact"),
        "quarantine_state": deepcopy(evidence["quarantine_state"]),
        "producer_validation_errors": deepcopy(evidence["producer_validation_errors"]),
        "authenticated": evidence["authenticated"],
        "accepted_for_positive_claim": evidence["accepted_for_positive_claim"],
        "raw_evidence": {
            "row_counts": deepcopy(evidence["raw_evidence_counts"]),
            "source_artifact_hashes": deepcopy(payload.get("source_artifact_hashes", {})),
        },
        "upstream_gate_observations": [
            deepcopy(row) for row in gates if row["consumer"] == task_id
        ],
        "recomputed_claims": [deepcopy(row) for row in claims if row["task_id"] == task_id],
        "verifier_is_oracle": bool(common.unwrap_principle(payload.get("verifier_is_oracle"))),
    }


def _self_row(task: Mapping[str, Any], gate_summary: Mapping[str, Any]) -> JsonDict:
    """Represent the synthesis task without creating a recursive self-hash."""

    return {
        "order": 14,
        "task_id": str(task["id"]),
        "title": task["title"],
        "declared_artifact_path": task["deliverable"],
        "declared_artifact_present": False,
        "actual_artifact_path": None,
        "evidence_source": "self_synthesis",
        "artifact_sha256": None,
        "artifact_size_bytes": 0,
        "status": "blocked",
        "verdict_class": "blocked",
        "honest_verdict": "blocked_required_semantic_audit",
        "quarantine_state": {
            "declared_flags": {},
            "exclusion_manifest_match": False,
            "quarantined": False,
        },
        "producer_validation_errors": [],
        "authenticated": True,
        "accepted_for_positive_claim": False,
        "raw_evidence": {"row_counts": {}, "source_artifact_hashes": {}},
        "upstream_gate_observations": [],
        "recomputed_claims": [],
        "gate_check_summary": deepcopy(gate_summary),
        "verifier_is_oracle": False,
    }


def _branch_decisions(
    tasks: Sequence[Mapping[str, Any]], matrix: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Apply exact prior verdicts without retiring a broader task family."""

    rows: list[JsonDict] = []
    for task, result in zip(tasks, matrix, strict=True):
        task_id = str(task["id"])
        current = str(result.get("honest_verdict"))
        priors = deepcopy(task.get("prior_failures") or [])
        matches = [
            prior
            for prior in priors
            if prior.get("verdict") == current and prior.get("retire_if_same_verdict") is True
        ]
        if matches:
            action = "retire"
            reason = "The exact prior verdict recurred for this task mechanism."
            condition = "Replace this exact mechanism before another measurement."
        elif task_id in {"exp7260-source-contract", "exp7273-capstone"}:
            action = "retire"
            reason = "This one-time milestone receipt is closed and remains immutable."
            condition = "Use a new milestone identity; do not rerun this receipt."
        elif result.get("status") == "blocked":
            action = "needs_changed_prerequisite"
            reason = "The declared producer did not run because its exact upstream gate failed."
            condition = "Change the named upstream field or mechanism before scheduling new work."
        elif result.get("verdict_class") in {"null", "disqualified"}:
            action = "needs_changed_prerequisite"
            reason = "Completed evidence failed efficacy or contract acceptance."
            condition = (
                "Name a changed cause and a replacement mechanism before another measurement."
            )
        else:
            action = "continue"
            reason = (
                "The narrow capability can support a changed experiment without implying value."
            )
            condition = "Use it only with new oracle-distinct value controls."
        rows.append(
            {
                "task_id": task_id,
                "current_honest_verdict": current,
                "prior_failures": priors,
                "matching_prior_verdicts": matches,
                "exact_same_verdict_recurrence": bool(matches),
                "retire_if_same_verdict_applied": bool(matches),
                "action": action,
                "reason": reason,
                "exact_next_condition": condition,
                "retry_current_mechanism": False,
                "broad_family_retirement_invented": False,
            }
        )
    return rows


def _display_path(root: Path, path: Path) -> str:
    """Use repository-relative paths when evidence belongs to this checkout."""

    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_sidecars(
    root: Path,
    raw_dir: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    claims: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Freeze authorities and isolate historical model and negative evidence."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    roadmap_copy = raw_dir / "selected-roadmap.yaml"
    design_copy = raw_dir / "v639-design.md"
    _atomic_bytes(roadmap_copy, contract["roadmap_bytes"])
    _atomic_bytes(design_copy, contract["design_bytes"])
    model_path = raw_dir / "historical-model-receipts.json"
    negative_path = raw_dir / "negative-gate-fixtures.json"
    claims_path = raw_dir / "recomputed-claim-rows.json"
    common._atomic_write(
        model_path,
        {
            "schema": "carnot.exp7273.historical_model_receipts.v1",
            "rows": [
                {
                    "task_id": task_id,
                    "artifact_sha256": row["artifact_sha256"],
                    "MODEL_SPECS": deepcopy(row["payload"].get("MODEL_SPECS", [])),
                    "model_invoked": row["payload"].get("model_invoked"),
                    "invocation_counts": deepcopy(row["payload"].get("invocation_counts", {})),
                    "inference_substrate": row["payload"].get("inference_substrate"),
                }
                for task_id, row in evidence.items()
                if row["payload"].get("model_invoked") is True
            ],
        },
    )
    common._atomic_write(
        negative_path,
        {
            "schema": "carnot.exp7273.negative_gate_fixtures.v1",
            "failed_gate_rows": [deepcopy(row) for row in gates if row["passed"] is False],
            "blocked_or_quarantined_rows": [
                {
                    "task_id": task_id,
                    "status": row["payload"].get("status", "missing"),
                    "quarantine_state": deepcopy(row["quarantine_state"]),
                    "actual_artifact_path": row["selected_evidence_path"],
                    "artifact_sha256": row["artifact_sha256"],
                }
                for task_id, row in evidence.items()
                if row["payload"].get("status") == "blocked"
                or row["quarantine_state"]["quarantined"]
            ],
        },
    )
    common._atomic_write(
        claims_path,
        {"schema": "carnot.exp7273.recomputed_claim_rows.v1", "rows": list(claims)},
    )
    return [
        {"path": _display_path(root, path), "sha256": _sha256(path)}
        for path in (model_path, negative_path, claims_path)
    ]


def _load_validation_receipts(root: Path, path: Path | None) -> list[JsonDict]:
    """Authenticate existing command logs without inventing missing successes."""

    if path is None or not path.is_file():
        return []
    payload = read_json(path)
    rows: list[JsonDict] = []
    for receipt in payload.get("receipts", []):
        log_path = Path(str(receipt.get("log_path", "")))
        resolved = log_path if log_path.is_absolute() else root / log_path
        observed = _sha256(resolved) if resolved.is_file() else None
        rows.append(
            {
                "name": receipt.get("name"),
                "command": receipt.get("command"),
                "exit_code": receipt.get("exit_code"),
                "log_path": _display_path(root, resolved),
                "log_sha256": receipt.get("log_sha256"),
                "observed_log_sha256": observed,
                "log_hash_matches": observed == receipt.get("log_sha256"),
            }
        )
    return rows


def _preconditions(
    root: Path,
    output: Path,
    checkpoint: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
) -> tuple[list[JsonDict], dict[str, str]]:
    """Record required bytes, ownership, output access, and producer presence."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in STATIC_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": "required_input",
                "upstream": str(relative),
                "artifact_field": "present_nonempty",
                "expected_value": True,
                "observed_value": present,
                "passed": present,
            }
        )
        if present:
            hashes[str(relative)] = _sha256(path)
    for path in (output, checkpoint):
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(path.parent, os.W_OK)
        checks.append(
            {
                "check": "writable_output_parent",
                "upstream": _display_path(root, path.parent),
                "artifact_field": "writable",
                "expected_value": True,
                "observed_value": writable,
                "passed": writable,
            }
        )
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": str(SPEC_PATH),
                "artifact_field": "REQ-REPORT-7273",
                "expected_value": True,
                "observed_value": "REQ-REPORT-7273" in spec_text,
                "passed": "REQ-REPORT-7273" in spec_text,
            },
            {
                "check": "independent_contract",
                "upstream": f"{ROADMAP_PATH}|{DESIGN_PATH}",
                "artifact_field": "milestone|ids|titles|paths|gates",
                "expected_value": True,
                "observed_value": contract["contract_agrees"],
                "passed": contract["contract_agrees"],
            },
            {
                "check": "producer_slots",
                "upstream": MILESTONE,
                "artifact_field": "terminal_artifacts_or_blocks",
                "expected_value": 13,
                "observed_value": sum(row["terminal"] for row in evidence.values()),
                "passed": all(row["terminal"] for row in evidence.values()),
            },
        ]
    )
    for row in evidence.values():
        selected = row["selected_evidence_path"]
        if selected:
            hashes[selected] = str(row["artifact_sha256"])
    return checks, hashes


def _publication_gate(root: Path) -> JsonDict:
    """Run the stable G1-G4 script and retain its unmodified result."""

    command = [sys.executable, "-u", "scripts/publication_gate.py", "--json"]
    progress(6, "subprocess start", "publication_gate.py --json")
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.monotonic() - started
    progress(6, "subprocess end", f"publication gate exit={completed.returncode}")
    if completed.stdout:
        print(completed.stdout.rstrip(), flush=True)
    payload = json.loads(completed.stdout)
    if completed.returncode != 0 or not isinstance(payload, dict):
        raise RuntimeError(f"publication gate failed: {completed.stderr}")
    payload["command"] = " ".join(command)
    payload["exit_code"] = completed.returncode
    payload["elapsed_s"] = elapsed
    payload["stderr"] = completed.stderr
    return payload


def _acceptance_results(
    contract: Mapping[str, Any],
    matrix: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    gates: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Score completion independently from scientific and validation value."""

    receipt_names = [str(row.get("name")) for row in receipts]
    definitions = (
        (
            "fourteen_task_dispositions",
            14,
            len(matrix),
            len(matrix) == 14,
            "Roster completion does not assert scientific value.",
        ),
        (
            "independent_contract_exact",
            True,
            contract["contract_agrees"],
            contract["contract_agrees"] is True,
            "The stale Markdown remains a contract failure.",
        ),
        (
            "claim_reductions_recorded",
            len(CLAIM_NAMES),
            len(claims),
            tuple(row["claim"] for row in claims) == CLAIM_NAMES,
            "Each scientific question retains a separate reduction.",
        ),
        (
            "same_milestone_gates_pass",
            7,
            sum(row["passed"] is True for row in gates),
            len(gates) == 7 and all(row["passed"] is True for row in gates),
            "A failed efficacy gate differs from missing evidence.",
        ),
        (
            "required_semantic_audit_available",
            "complete",
            matrix[6]["status"],
            matrix[6]["status"] == "complete",
            "The gated semantic audit is required before a positive closeout.",
        ),
        (
            "validation_receipts_complete",
            list(VALIDATION_NAMES),
            receipt_names,
            set(VALIDATION_NAMES) <= set(receipt_names)
            and all(
                row.get("exit_code") == 0 and row.get("log_hash_matches") is True
                for row in receipts
                if row.get("name") in VALIDATION_NAMES
            ),
            "Each scoped command keeps its actual exit code and log hash.",
        ),
    )
    return [
        {
            "criterion": name,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }
        for name, expected, observed, passed, principle in definitions
    ]


def _publication_gate_valid(value: Any) -> bool:
    """Check that G1-G4, paper-ready, and unmet names agree internally."""

    if not isinstance(value, Mapping) or value.get("exit_code") != 0:
        return False
    gates = value.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != {"G1", "G2", "G3", "G4"}:
        return False
    observed = [name for name, gate in gates.items() if gate.get("pass") is not True]
    return value.get("unmet_gates") == observed and value.get("paper_ready") is (not observed)


def validate_artifact(artifact: Mapping[str, Any], root: Path | None = None) -> list[str]:
    """Recompute identity, rows, gates, decisions, hashes, and checksum."""

    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(not REQUIRED_ARTIFACT_FIELDS.issubset(artifact), "required_fields")
    add(artifact.get("field_principles") != ALL_FIELD_PRINCIPLES, "field_principles")
    add(
        artifact.get("schema") != "carnot.exp7273.v639_capstone.v1"
        or artifact.get("experiment_id") != "exp7273-capstone"
        or artifact.get("milestone") != MILESTONE,
        "identity",
    )
    add(artifact.get("status") != "blocked" or artifact.get("run_date") != RUN_DATE, "lifecycle")
    add(
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS,
        "model_invocation",
    )
    add(
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
        or not artifact.get("execution_host"),
        "execution_substrate",
    )
    contract_rows = artifact.get("contract_rows", [])
    add(
        not isinstance(contract_rows, list)
        or len(contract_rows) != 14
        or [row.get("unit_id") for row in contract_rows] != list(EXPECTED_TASK_IDS),
        "contract_rows",
    )
    matrix = artifact.get("evidence_matrix", [])
    add(
        not isinstance(matrix, list)
        or len(matrix) != 14
        or [row.get("task_id") for row in matrix] != list(EXPECTED_TASK_IDS)
        or matrix[-1].get("evidence_source") != "self_synthesis"
        or matrix[-1].get("artifact_sha256") is not None,
        "evidence_matrix",
    )
    claims = artifact.get("rows", [])
    add(
        not isinstance(claims, list)
        or tuple(row.get("claim") for row in claims) != CLAIM_NAMES
        or any(
            not {
                "unit_id",
                "arm",
                "seed",
                "metric",
                "metric_value",
                "error",
                "abstention",
                "censored",
            }.issubset(row)
            for row in claims
        ),
        "claim_rows",
    )
    branches = artifact.get("branch_decisions", [])
    add(
        not isinstance(branches, list)
        or len(branches) != 14
        or any(row.get("retry_current_mechanism") is not False for row in branches),
        "branch_decisions",
    )
    gate_rows = artifact.get("same_milestone_gate_replay_rows", [])
    add(not isinstance(gate_rows, list) or len(gate_rows) != 7, "gate_replay_rows")
    add(artifact.get("capstone_complete_score") != 1, "capstone_complete_score")
    add(
        artifact.get("verdict_class") != "blocked"
        or not str(artifact.get("honest_verdict", "")).startswith("blocked_"),
        "verdict_class",
    )
    summary = artifact.get("gate_check_summary", {})
    add(
        not isinstance(summary, Mapping)
        or summary.get("passed") is not False
        or any(
            field not in summary
            for field in (
                "failed_check",
                "upstream",
                "artifact_field",
                "expected_value",
                "observed_value",
            )
        ),
        "gate_check_summary",
    )
    add(not _publication_gate_valid(artifact.get("publication_gate")), "publication_gate")
    add(
        artifact.get("reproducibility_checksum") != _artifact_checksum(artifact),
        "reproducibility_checksum",
    )
    if root is not None:
        contract = load_contract(root)
        evidence = load_repository_payloads(root, contract["tasks"])
        current_gates = replay_gates(contract["tasks"], evidence)
        current_claims = recompute_claims(evidence)
        current_matrix = [
            _matrix_row(index, task, evidence[str(task["id"])], current_gates, current_claims)
            for index, task in enumerate(contract["tasks"][:-1], 1)
        ]
        current_matrix.append(_self_row(contract["tasks"][-1], summary))
        add(contract_rows != contract["contract_rows"], "contract_rows")
        add(claims != current_claims, "claim_rows")
        add(gate_rows != current_gates, "gate_replay_rows")
        add(matrix != current_matrix, "evidence_matrix")
        add(
            branches != _branch_decisions(contract["tasks"], current_matrix),
            "branch_decisions",
        )
        hashes = artifact.get("source_artifact_hashes", {})
        if isinstance(hashes, Mapping):
            for named, expected in hashes.items():
                path = Path(str(named))
                resolved = path if path.is_absolute() else root / path
                if not resolved.is_file() or _sha256(resolved) != expected:
                    add(True, "source_artifact_hashes")
                    break
        else:
            add(True, "source_artifact_hashes")
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    output_path: Path,
    checkpoint_path: Path,
    *,
    raw_dir: Path | None = None,
    validation_receipt_path: Path | None = None,
) -> JsonDict:
    """Aggregate measured V639 evidence and atomically write a blocked result."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "authenticate inputs and writable output paths")
    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    selected_raw_dir = raw_dir or root / DEFAULT_RAW_DIR
    common._atomic_write(
        checkpoint_path,
        {
            "schema": "carnot.exp7273.v639_capstone.checkpoint.v1",
            "experiment_id": "exp7273-capstone",
            "status": "running",
            "run_date": run_date,
            "MODEL_SPECS": [],
            "model_invoked": False,
        },
    )

    spans: dict[str, float] = {}
    phase = time.monotonic()
    progress(1, "start", "parse V639 YAML and named Markdown independently")
    contract = load_contract(root)
    spans["contract_parse"] = time.monotonic() - phase
    progress(1, "end", f"completed=14 contract_agrees={contract['contract_agrees']}")

    phase = time.monotonic()
    progress(2, "start", "authenticate thirteen producer slots")
    evidence = load_repository_payloads(root, contract["tasks"])
    for index, task_id in enumerate(EXPECTED_PRODUCER_IDS, 1):
        progress(
            2,
            "unit",
            f"completed={index}/13 {task_id} source={evidence[task_id]['evidence_source']}",
        )
    checks, source_hashes = _preconditions(root, output_path, checkpoint_path, contract, evidence)
    spans["producer_authentication"] = time.monotonic() - phase
    progress(2, "end", f"terminal_slots={sum(row['terminal'] for row in evidence.values())}")

    phase = time.monotonic()
    progress(3, "start", "replay seven gates and reduce fourteen claims")
    gates = replay_gates(contract["tasks"], evidence)
    claims = recompute_claims(evidence)
    spans["gate_and_claim_reduction"] = time.monotonic() - phase
    progress(3, "end", f"gates={len(gates)} claims={len(claims)}")

    phase = time.monotonic()
    progress(4, "start", "freeze authorities, historical models, and raw reductions")
    sidecars = _write_sidecars(root, selected_raw_dir, contract, evidence, claims, gates)
    frozen_paths = (
        selected_raw_dir / "selected-roadmap.yaml",
        selected_raw_dir / "v639-design.md",
    )
    for path in frozen_paths:
        source_hashes[_display_path(root, path)] = _sha256(path)
    for receipt in sidecars:
        source_hashes[str(receipt["path"])] = str(receipt["sha256"])
    spans["authority_freeze_and_sidecars"] = time.monotonic() - phase
    progress(4, "end", f"sidecars={len(sidecars)} frozen_authorities=2")

    gate_summary = {
        "passed": False,
        "failed_check": "required_semantic_audit_available",
        "upstream": "exp7266-semantic-audit",
        "artifact_field": "status|failed_upstream|failed_field|failed_observed",
        "expected_value": {
            "status": "complete",
            "upstream_field": "exp7265-mention-heldout.mention_capture_complete_score",
            "upstream_value": 1,
        },
        "observed_value": {
            "status": evidence["exp7266-semantic-audit"]["payload"].get("status"),
            "upstream": evidence["exp7266-semantic-audit"]["payload"].get("failed_upstream"),
            "field": evidence["exp7266-semantic-audit"]["payload"].get("failed_field"),
            "value": evidence["exp7266-semantic-audit"]["payload"].get("failed_observed"),
        },
        "terminal_classification": "blocked",
        "retry_allowed": False,
        "additional_failures": [
            {
                "upstream": str(DESIGN_PATH),
                "artifact_field": "milestone|contract_rows",
                "expected_value": {"milestone": MILESTONE, "all_rows_match": True},
                "observed_value": {
                    "milestone": contract["markdown_milestone"],
                    "all_rows_match": contract["contract_agrees"],
                },
            },
            {
                "upstream": "exp7270-durable-profile",
                "artifact_field": "journal_optimization_warranted_score",
                "expected_value": 1,
                "observed_value": evidence["exp7270-durable-profile"]["payload"].get(
                    "journal_optimization_warranted_score"
                ),
            },
        ],
    }

    phase = time.monotonic()
    progress(5, "start", "build evidence matrix and exact branch decisions")
    matrix = [
        _matrix_row(index, task, evidence[str(task["id"])], gates, claims)
        for index, task in enumerate(contract["tasks"][:-1], 1)
    ]
    matrix.append(_self_row(contract["tasks"][-1], gate_summary))
    for index, row in enumerate(matrix, 1):
        progress(5, "unit", f"completed={index}/14 {row['task_id']}")
    decisions = _branch_decisions(contract["tasks"], matrix)
    spans["matrix_and_branches"] = time.monotonic() - phase
    progress(5, "end", f"matrix={len(matrix)} decisions={len(decisions)}")

    phase = time.monotonic()
    progress(6, "start", "record stable G1-G4 publication context")
    publication_gate = _publication_gate(root)
    spans["publication_gate"] = time.monotonic() - phase
    progress(6, "end", f"unmet={publication_gate['unmet_gates']}")

    phase = time.monotonic()
    progress(7, "start", "load command receipts and map PRD gaps")
    receipts = _load_validation_receipts(root, validation_receipt_path)
    if validation_receipt_path is not None and validation_receipt_path.is_file():
        source_hashes[_display_path(root, validation_receipt_path)] = _sha256(
            validation_receipt_path
        )
    claim_by_name = {row["claim"]: row for row in claims}
    prd = {
        "FR-11": {
            "complete": False,
            "recognition_run_complete": claim_by_name["recognition_run_complete"][
                "recomputed_value"
            ],
            "recognition_learning_value": claim_by_name["recognition_learning_value"][
                "recomputed_value"
            ],
            "recognition_safety": claim_by_name["recognition_safety"]["recomputed_value"],
            "remaining_limit": "Active recognition did not beat all frozen efficacy and recurrence controls.",
        },
        "FR-12": {
            "complete": False,
            "source_capture_accounted": claim_by_name["source_capture_accounted"][
                "recomputed_value"
            ],
            "source_fidelity_exact": claim_by_name["source_fidelity_exact"]["recomputed_value"],
            "source_semantic_value": claim_by_name["source_semantic_value"]["recomputed_value"],
            "remaining_limit": "Held-out fidelity failed and the semantic audit was gate-blocked.",
        },
        "FR-05/08": {
            "complete": False,
            "oracle_parity": claim_by_name["arc_transition_oracle_parity"]["recomputed_value"],
            "public_game_generalization": claim_by_name["arc_public_game_generalization"][
                "recomputed_value"
            ],
            "policy_consumption": claim_by_name["arc_policy_consumption"]["recomputed_value"],
            "official_hidden_score": claim_by_name["official_hidden_score"]["recomputed_value"],
            "remaining_limit": "Oracle parity passed, but the live model installed no valid plan and no hidden score exists.",
        },
        "NFR-01": {
            "complete": False,
            "durable_profile_complete": claim_by_name["durable_profile_complete"][
                "recomputed_value"
            ],
            "durable_cost_limit_value": claim_by_name["durable_cost_limit_value"][
                "recomputed_value"
            ],
            "durable_log_semantics": claim_by_name["durable_log_semantics"]["recomputed_value"],
            "remaining_limit": "Fixed sync dominated, the 10x target was infeasible, and the delta-log task stayed blocked.",
        },
    }
    spans["validation_and_prd_mapping"] = time.monotonic() - phase
    progress(7, "end", f"validation_receipts={len(receipts)} PRD_rows={len(prd)}")

    honest_verdict = (
        "blocked_required_semantic_audit: V639 fourteen-task matrix is complete; held-out "
        "source fidelity is null, the semantic audit is gate-blocked, ARC and recognition "
        "value are null, and delta-log semantics remain blocked"
    )
    artifact: JsonDict = {
        "schema": "carnot.exp7273.v639_capstone.v1",
        "experiment_id": "exp7273-capstone",
        "milestone": MILESTONE,
        "status": "blocked",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": "",
        "field_principles": ALL_FIELD_PRINCIPLES,
        "preconditions_checked": checks,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "phase_spans_s": spans,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": source_hashes,
        "rows": claims,
        "sample_size_budget": {
            "contract_rows": {
                "planned": 14,
                "attempted": 14,
                "completed": 14,
                "censored": 0,
                "independent_units": 14,
            },
            "producer_artifacts": {
                "planned": 13,
                "attempted": 13,
                "completed": sum(row["terminal"] for row in evidence.values()),
                "censored": sum(
                    row["declared_artifact_present"] is False for row in evidence.values()
                ),
                "independent_units": 13,
            },
            "claim_rows": {
                "planned": len(CLAIM_NAMES),
                "attempted": len(CLAIM_NAMES),
                "completed": len(claims),
                "censored": sum(row["censored"] is True for row in claims),
                "independent_units": 8,
            },
            "stopping_rule": "Stop after fourteen task dispositions, seven gates, fourteen claims, and four PRD rows are recorded.",
        },
        "acceptance_gate_results": _acceptance_results(contract, matrix, claims, gates, receipts),
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "honest_verdict": honest_verdict,
        "verdict_class": "blocked",
        "validation_receipts": receipts,
        "capstone_complete_score": 1,
        "evidence_matrix": matrix,
        "contract_rows": contract["contract_rows"],
        "branch_decisions": decisions,
        "prd_gap_matrix": prd,
        "publication_gate": publication_gate,
        "same_milestone_gate_replay_rows": gates,
        "historical_and_negative_fixture_sidecars": sidecars,
        "board_dispositions": deepcopy(
            evidence["exp7272-board-state"]["payload"].get("board_rows", [])
        ),
        "publication_performed": False,
        "upload_performed": False,
        "submission_performed": False,
        "external_message_performed": False,
        "production_default_changed": False,
        "exclusion_manifest_modified": False,
        "research_roadmap_modified": False,
        "conductor_modified": False,
    }
    elapsed = time.monotonic() - started
    spans["final_assembly"] = max(elapsed - sum(spans.values()), 0.0)
    artifact["duration_s"] = elapsed
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = _artifact_checksum(artifact)

    progress(8, "start", "write and validate the measured raw terminal candidate")
    candidate_path = selected_raw_dir / "measured-terminal-candidate.json"
    common._atomic_write(candidate_path, artifact)
    candidate = read_json(candidate_path)
    errors = validate_artifact(candidate, root=root)
    if errors:
        raise RuntimeError("capstone candidate validation failed: " + ",".join(errors))
    progress(8, "end", "measured raw candidate passed")

    progress(9, "start", "publish the atomic terminal artifact")
    common._atomic_write(checkpoint_path, artifact)
    common._atomic_write(output_path, artifact)
    final = read_json(output_path)
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError("terminal artifact validation failed: " + ",".join(errors))
    progress(9, "end", f"wrote {output_path}")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Build the capstone or independently validate one terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--validation-receipt-path", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.artifact_path or root / DEFAULT_OUTPUT_PATH
    checkpoint = args.checkpoint_path or root / DEFAULT_CHECKPOINT_PATH
    if args.validate:
        progress(8, "start", f"independent raw-row reducer for {output}")
        errors = validate_artifact(read_json(output), root=root)
        progress(8, "end", "passed" if not errors else ",".join(errors))
        return int(bool(errors))
    receipt_path = args.validation_receipt_path
    if receipt_path is None:
        default_receipts = root / DEFAULT_VALIDATION_RECEIPT_PATH
        receipt_path = default_receipts if default_receipts.is_file() else None
    build_artifact(
        root,
        args.date,
        output,
        checkpoint,
        raw_dir=args.raw_dir,
        validation_receipt_path=receipt_path,
    )
    return 0

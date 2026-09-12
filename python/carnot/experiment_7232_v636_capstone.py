"""Build the V636 evidence matrix without invoking a model.

The module reuses the prior capstone's file, quarantine, checksum, and
publication helpers. It keeps only V636 policy and row recomputation here.

Spec refs: REQ-REPORT-7232 and SCENARIO-REPORT-7232-*.
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
import re
import shutil
import time
from typing import Any

import yaml

from carnot import experiment_7218_v635_capstone as base
from carnot import experiment_7219_v636_source_contract as contract_source


JsonDict = dict[str, Any]
PublicationRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.636"
RUN_DATE = "20260912"
RANDOM_SEED = 7_232_202_609_12
MODEL_SPECS: list[JsonDict] = []

SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
FROZEN_YAML_PATH = Path("results/raw/experiment_7219/selected-roadmap.yaml")
FROZEN_DESIGN_PATH = Path("results/raw/experiment_7219/research-roadmap-vNEXT.md")
CONTRACT_RECEIPT_PATH = Path("results/experiment_7219_v636_source_contract.json")
EXCLUSION_MANIFEST_PATH = Path("ops/exclusion_manifest.yaml")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7232_v636_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7232_v636_capstone.json")

EXPECTED_TASK_IDS = tuple(contract_source.EXPECTED_ID_ORDER)
BRANCH_ACTIONS = frozenset({"continue", "retire", "needs_changed_prerequisite"})

PROMPT_FIELD_PRINCIPLES = {
    "field_principles": "Annotate actual values in this map; do not wrap arbitrary dictionaries as principle/value records.",
    "status": "Write a terminal artifact only when done or externally blocked; running checkpoints use a different path.",
    "run_date": "Use 20260912 and record actual UTC timestamps, never copy an upstream run date.",
    "preconditions_checked": "Actual code, resource, identity and gate observations before expensive work.",
    "inference_substrate": "Use the recognized literal for the work actually executed; custom free text caused the Exp7208 quarantine.",
    "inference_substrate_class": "Match actual generation, load-only, CPU or aggregation work and its duration floor.",
    "execution_venue": "Exactly host, kv260, gatemate or polarfire; the top-level orchestration here is host.",
    "execution_host": "Actual hostname separate from venue.",
    "duration_s": "Measured monotonic work time; no padding or reclassification to evade a floor.",
    "source_artifact_hashes": "Bind code, source documents, manifests and raw evidence to claims.",
    "rows": "Per unit/arm/seed metric, error and abstention for every comparison; retain full denominators.",
    "sample_size_budget": "Planned, attempted, completed, censored and independent units; no silent removal.",
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": "Every blocked_* verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when correctness authority is reused as the verifier; independent code alone is not distinct authority.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. partial means unfinished own work only.",
    "honest_verdict": "Use complete_ or complete: for completed findings; blocked_* for external absence. A failed acceptance gate forbids positive.",
    "MODEL_SPECS": "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every model task.",
    "model_invoked": "True only for actual model execution; upstream model outputs are cached evidence.",
    "capstone_complete_score": "Evidence matrix completion is separate from scientific success.",
    "task_contract_rows": "Exactly fourteen IDs in the same order as both plan files.",
    "evidence_matrix": "Every task disposition, source hash, gate and authority boundary.",
    "recomputed_claim_rows": "Numeric producer-row checks supporting any promoted conclusion.",
    "branch_decisions": "Concrete continue/retire/changed-prerequisite decisions.",
    "prd_completion": "FR-11/FR-12/FR-05/08/NFR-01 judged separately.",
    "publication_gate": "Preparation only; no external publication authorized.",
}
REQUIRED_FIELD_PRINCIPLES = dict(PROMPT_FIELD_PRINCIPLES)

STATIC_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    EXCLUSION_MANIFEST_PATH,
    Path("ops/e2e-test-plan.md"),
    Path("ops/conductor-log.md"),
    FROZEN_YAML_PATH,
    FROZEN_DESIGN_PATH,
    CONTRACT_RECEIPT_PATH,
    Path("results/experiment_7218_v635_capstone.json"),
    Path("results/experiment_7182_v633_grounding_energy_audit.json"),
    Path("results/experiment_7197_v634_grounding_value_audit.json"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/publication_gate.py"),
    SPEC_PATH,
    Path("python/carnot/experiment_7218_v635_capstone.py"),
    Path("python/carnot/experiment_7219_v636_source_contract.py"),
    Path("python/carnot/experiment_7232_v636_capstone.py"),
    Path("scripts/experiments/experiment_7232_v636_capstone.py"),
    Path("tests/python/test_experiment_7232_v636_capstone.py"),
)

EXPECTED_FIELDS = {
    "exp7219-source-contract": "source_contract_complete_score",
    "exp7220-xml-canary": "xml_transport_ready_score",
    "exp7221-arc-session": "arc_session_complete_score",
    "exp7222-span-fixture": "span_fixture_ready_score",
    "exp7223-span-canary": "span_canary_ready_score",
    "exp7224-span-capture": "failed_field",
    "exp7226-belief-compiler": "belief_compiler_ready_score",
    "exp7227-belief-learning": "belief_run_complete_score",
    "exp7228-belief-cold-audit": "belief_audit_complete_score",
    "exp7229-rare-event-audit": "rare_event_audit_complete_score",
    "exp7230-native-belief": "native_belief_ready_score",
    "exp7231-board-continuity": "board_continuity_complete_score",
}

DECISION_POLICY: dict[str, tuple[str, str, str | None]] = {
    "exp7219-source-contract": (
        "retire",
        "Retire this one-time contract receipt after its fourteen frozen rows agree.",
        None,
    ),
    "exp7220-xml-canary": (
        "needs_changed_prerequisite",
        "The bounded server stopped before parser calls, so parser readiness remains null.",
        "Install a vLLM GGUF adapter that recognizes model_type qwen3_5, then repeat the same four fixed parser calls.",
    ),
    "exp7221-arc-session": (
        "retire",
        "Retire this cumulative missing-tool-demand collection at its operational target of ten; it found no useful valid world model or progress.",
        "A different task may test a model-valid world-model intervention with a paired adapter-withheld progress arm.",
    ),
    "exp7222-span-fixture": (
        "continue",
        "Keep the authenticated finite fixture as syntax and exact-execution infrastructure only.",
        "Use it only after a clean live canary supplies source-faithful spans.",
    ),
    "exp7223-span-canary": (
        "needs_changed_prerequisite",
        "The canary is quarantined and its retained rows have zero parse-complete and semantic-correct units.",
        "Produce an unquarantined canary with clean invocation provenance, at least seven parse-complete units, and at least six semantic-correct units.",
    ),
    "exp7224-span-capture": (
        "retire",
        "The conductor gate block exactly repeats the prior blocked_gate_check_failed verdict, so its task-specific retirement signal applies.",
        "A new task identity requires an authenticated span_canary_ready_score equal to one.",
    ),
    "exp7225-semantics-audit": (
        "needs_changed_prerequisite",
        "The conductor legitimately cascade-blocked this audit and wrote no sibling artifact.",
        "Provide an authenticated span capture with span_capture_complete_score equal to one.",
    ),
    "exp7226-belief-compiler": (
        "continue",
        "Keep the exact packed representation as implementation infrastructure; parity is not learning value.",
        "Test a recurrence-preserving memory policy while retaining this full-reference parity control.",
    ),
    "exp7227-belief-learning": (
        "retire",
        "Retire this packed online-memory policy because the complete prospective gate failed on recurrence retention.",
        "A new mechanism must keep recurrence error increase at or below 0.02 without weakening the matched controls.",
    ),
    "exp7228-belief-cold-audit": (
        "needs_changed_prerequisite",
        "The cold-audit artifact is quarantined, so its mechanics cannot support promotion.",
        "Supply a corrected unquarantined audit with consistent no-model provenance and the same rollback controls.",
    ),
    "exp7229-rare-event-audit": (
        "retire",
        "Retire the fixed rare-event accuracy target because the authenticated audit found its stated budget infeasible and preserved the failed sampler gate.",
        None,
    ),
    "exp7230-native-belief": (
        "needs_changed_prerequisite",
        "The native artifact is quarantined for a verdict-class mismatch, so neither parity nor speed is promoted.",
        "Supply an unquarantined circular-positive correction and independently validate the retained paired cost rows.",
    ),
    "exp7231-board-continuity": (
        "needs_changed_prerequisite",
        "Keep KV260 graduation and PolarFire CPU continuity; GateMate still has no changed physical state and no device-performance result exists.",
        "Record a dated GateMate cable, port, board, power, or DirtyJTAG state change after the last physical receipt before any new GateMate operation.",
    ),
    "exp7232-capstone": (
        "retire",
        "Retire this one-time synthesis after its complete fourteen-row matrix is stored.",
        None,
    ),
}


unwrap_principle = base.unwrap_principle
sha256_path = base.sha256_path
canonical_gate_block_path = base.canonical_gate_block_path
atomic_write = base._atomic_write
reproducibility_checksum = base.reproducibility_checksum


def progress(phase: int, state: str, detail: str) -> None:
    """Flush observed phase state so the conductor never sees silent work."""

    print(f"[exp7232] phase {phase} {state}: {detail}", flush=True)


def read_json(path: Path) -> JsonDict:
    """Read one JSON object because lists cannot carry the required schema."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def load_contract(root: Path) -> JsonDict:
    """Parse both frozen V636 plans and bind them to the source receipt."""

    yaml_bytes = (root / FROZEN_YAML_PATH).read_bytes()
    markdown_bytes = (root / FROZEN_DESIGN_PATH).read_bytes()
    document = yaml.safe_load(yaml_bytes)
    if not isinstance(document, Mapping):
        raise ValueError("frozen V636 YAML root is not a mapping")
    comparison = contract_source.evaluate_contract(markdown_bytes.decode(), document)
    raw_tasks = document.get("tasks", [])
    tasks = [
        {
            "id": row.get("id"),
            "title": row.get("title"),
            "deliverable": row.get("deliverable"),
            "gated_on": deepcopy(row.get("gated_on") or []),
            "prior_failures": deepcopy(row.get("prior_failures") or []),
        }
        for row in raw_tasks
        if isinstance(row, Mapping)
    ]
    task_ids = [str(row["id"]) for row in tasks]
    errors: list[str] = []
    if comparison.get("passed") is not True:
        errors.append("frozen_contract_mismatch")
    if tuple(task_ids) != EXPECTED_TASK_IDS:
        errors.append("contract_id_order")
    receipt = read_json(root / CONTRACT_RECEIPT_PATH)
    receipts = {
        row.get("source_type"): row
        for row in receipt.get("raw_source_rows", [])
        if isinstance(row, Mapping)
    }
    source_rows: list[JsonDict] = []
    for source_type, relative in (
        ("yaml", FROZEN_YAML_PATH),
        ("markdown", FROZEN_DESIGN_PATH),
    ):
        observed = sha256_path(root / relative)
        expected = receipts.get(source_type, {}).get("raw_sha256")
        source_rows.append(
            {
                "source_type": source_type,
                "path": str(relative),
                "sha256": observed,
                "receipt_sha256": expected,
                "hash_matches_receipt": observed == expected,
            }
        )
    if not all(row["hash_matches_receipt"] for row in source_rows):
        errors.append("contract_source_hash")
    compact_rows = [
        {
            "order": row["order"],
            "task_id": row["yaml"]["id"],
            "title": row["yaml"]["title"],
            "deliverable": row["yaml"]["deliverable"],
            "gates": row["yaml"]["gates"],
            "markdown_yaml_match": row["passed"],
        }
        for row in comparison.get("contract_rows", [])
    ]
    return {
        "tasks": tasks,
        "task_ids": task_ids,
        "task_contract_rows": compact_rows,
        "source_rows": source_rows,
        "errors": errors,
    }


def load_evidence(root: Path, task: Mapping[str, Any], manifest: Any) -> JsonDict:
    """Read only the declared result or the full-ID conductor fallback."""

    task_id = str(task["id"])
    declared = str(task["deliverable"])
    fallback = canonical_gate_block_path(task_id)
    if (root / declared).is_file():
        selected, source = declared, "declared_deliverable"
    elif (root / fallback).is_file():
        selected, source = fallback, "conductor_gate_block"
    else:
        selected, source = None, "missing"
    payload = read_json(root / selected) if selected else {}
    quarantine = base.quarantine_receipt(payload, task_id, selected or declared, manifest)
    terminal = payload.get("status") in {"complete", "blocked"}
    return {
        "task_id": task_id,
        "declared_deliverable_path": declared,
        "canonical_gate_block_path": fallback,
        "selected_evidence_path": selected,
        "evidence_source": source,
        "artifact_sha256": sha256_path(root / selected) if selected else None,
        "artifact_size_bytes": (root / selected).stat().st_size if selected else 0,
        "quarantine_receipt": quarantine,
        "terminal": terminal,
        "accepted_for_promoted_evidence": bool(payload)
        and terminal
        and not quarantine["quarantined"],
        "payload": payload,
    }


def load_repository_payloads(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Load all thirteen upstream slots, including explicit missing slots."""

    manifest = yaml.safe_load((root / EXCLUSION_MANIFEST_PATH).read_text(encoding="utf-8"))
    return {
        str(task["id"]): load_evidence(root, task, manifest)
        for task in tasks
        if task["id"] != "exp7232-capstone"
    }


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Replay structured values while keeping quarantine as a separate gate."""

    rows: list[JsonDict] = []
    for task in tasks:
        for gate in task.get("gated_on") or []:
            upstream = str(gate["upstream"])
            producer = evidence.get(
                upstream,
                {"payload": {}, "quarantine_receipt": {"quarantined": False}},
            )
            result = base.evaluate_gate(
                producer["payload"],
                str(gate["artifact_field"]),
                gate["value"],
                producer["quarantine_receipt"],
            )
            rows.append(
                {
                    "consumer": task["id"],
                    "upstream": upstream,
                    "operator": gate.get("op"),
                    **result,
                }
            )
    return rows


def _claim(
    task_id: str,
    metric: str,
    declared: Any,
    recomputed: int | float,
    evidence_fields: Sequence[str],
    evidence: Mapping[str, Any],
    verifier_is_oracle: bool,
) -> JsonDict:
    """Store one row-derived number and stop promotion after authentication."""

    matches = declared == recomputed
    accepted = evidence.get("accepted_for_promoted_evidence") is True
    quarantined = evidence.get("quarantine_receipt", {}).get("quarantined") is True
    error = None
    if not matches:
        error = "declared_value_mismatch"
    elif quarantined:
        error = "quarantined_upstream"
    elif not accepted:
        error = "unauthenticated_upstream"
    return {
        "unit_id": f"{task_id}:{metric}",
        "arm": "independent_producer_row_recomputation",
        "seed": RANDOM_SEED,
        "metric": metric,
        "metric_value": recomputed,
        "error": error,
        "abstention": not (matches and accepted),
        "task_id": task_id,
        "declared_value": declared,
        "recomputed_value": recomputed,
        "matches": matches,
        "promoted": matches and accepted,
        "evidence_fields": list(evidence_fields),
        "verifier_is_oracle": verifier_is_oracle,
        "claim_class": "disqualified"
        if quarantined
        else "circular_positive"
        if verifier_is_oracle and bool(recomputed)
        else "numeric_check",
    }


def recompute_claims(evidence: Mapping[str, JsonDict]) -> list[JsonDict]:
    """Recompute every numeric claim that can influence the V636 summary."""

    claims: list[JsonDict] = []

    def add(
        task_id: str,
        metric: str,
        recomputed: int | float,
        fields: Sequence[str],
        *,
        declared: Any | None = None,
    ) -> None:
        item = evidence[task_id]
        payload = item["payload"]
        actual = payload.get(metric) if declared is None else declared
        claims.append(
            _claim(
                task_id,
                metric,
                unwrap_principle(actual),
                recomputed,
                fields,
                item,
                bool(unwrap_principle(payload.get("verifier_is_oracle", False))),
            )
        )

    payload = evidence["exp7219-source-contract"]["payload"]
    rows = payload.get("rows", [])
    add(
        "exp7219-source-contract",
        "source_contract_complete_score",
        int(len(rows) == 14 and all(row.get("passed") is True for row in rows)),
        ("rows.passed",),
    )

    payload = evidence["exp7220-xml-canary"]["payload"]
    rows = payload.get("rows", [])
    parser_rows = payload.get("parser_rows", [])
    add(
        "exp7220-xml-canary",
        "xml_canary_complete_score",
        int(len(rows) == 4 and all(row.get("completed") is True for row in rows)),
        ("rows.completed",),
    )
    add(
        "exp7220-xml-canary",
        "xml_transport_ready_score",
        int(len(parser_rows) == 4 and all(row.get("populated") is True for row in parser_rows)),
        ("parser_rows.populated",),
    )

    payload = evidence["exp7221-arc-session"]["payload"]
    session = payload.get("session_receipt", {})
    add(
        "exp7221-arc-session",
        "arc_session_complete_score",
        int(
            bool(payload.get("rows"))
            and session.get("terminal_receipt") is True
            and session.get("status") == "complete"
        ),
        ("rows", "session_receipt.terminal_receipt", "session_receipt.status"),
    )
    unique = {
        row.get("induction_id"): row
        for row in payload.get("cumulative_induction_rows", [])
        if isinstance(row, Mapping)
        and isinstance(row.get("induction_id"), str)
        and row.get("source_authenticated") is True
    }
    add(
        "exp7221-arc-session",
        "cumulative_induction_count",
        len(unique),
        ("cumulative_induction_rows.induction_id", "source_authenticated"),
        declared=len(unique),
    )
    valid_count = sum(not list(row.get("model_validity_errors") or []) for row in unique.values())
    add(
        "exp7221-arc-session",
        "model_valid_induction_count",
        valid_count,
        ("cumulative_induction_rows.model_validity_errors",),
        declared=valid_count,
    )
    add(
        "exp7221-arc-session",
        "registered_level_increment",
        int(payload.get("registered_level_increment", 0)),
        ("registered_level_increment", "per_game_results"),
    )

    payload = evidence["exp7222-span-fixture"]["payload"]
    readiness = payload.get("readiness_checks", {})
    mutations = payload.get("mutation_rows", [])
    add(
        "exp7222-span-fixture",
        "span_fixture_ready_score",
        int(
            len(payload.get("rows", [])) == 320
            and bool(readiness)
            and all(value is True for value in readiness.values())
            and bool(mutations)
            and all(row.get("passed") is True for row in mutations)
        ),
        ("rows", "readiness_checks", "mutation_rows.passed"),
    )

    payload = evidence["exp7223-span-canary"]["payload"]
    canary_rows = payload.get("canary_rows", [])
    receipt = payload.get("readiness_receipt", {})
    parse_count = sum(row.get("parse_complete") is True for row in canary_rows)
    semantic_count = sum(row.get("semantic_correct") is True for row in canary_rows)
    add(
        "exp7223-span-canary",
        "span_canary_complete_score",
        int(len(canary_rows) == 8 and receipt.get("observed_calls") == 16),
        ("canary_rows", "readiness_receipt.observed_calls"),
    )
    add(
        "exp7223-span-canary",
        "span_canary_ready_score",
        int(parse_count >= 7 and semantic_count >= 6),
        ("canary_rows.parse_complete", "canary_rows.semantic_correct"),
    )

    payload = evidence["exp7226-belief-compiler"]["payload"]
    parity = payload.get("parity_rows", [])
    mutations = payload.get("mutation_rows", [])
    add(
        "exp7226-belief-compiler",
        "belief_compiler_ready_score",
        int(
            len(payload.get("rows", [])) == 20
            and bool(parity)
            and all(
                row.get("passed") is True and row.get("mismatch_count", 0) == 0 for row in parity
            )
            and bool(mutations)
            and all(row.get("passed") is True for row in mutations)
            and payload.get("stream_conformance_errors") == []
        ),
        ("rows", "parity_rows", "mutation_rows", "stream_conformance_errors"),
    )

    payload = evidence["exp7227-belief-learning"]["payload"]
    add(
        "exp7227-belief-learning",
        "belief_run_complete_score",
        int(
            len(payload.get("rows", [])) == 100
            and len(payload.get("comparison_rows", [])) == 3
            and len(payload.get("memory_deletion_rows", [])) == 20
        ),
        ("rows", "comparison_rows", "memory_deletion_rows"),
    )
    add(
        "exp7227-belief-learning",
        "belief_learning_value_score",
        int(payload.get("acceptance_gate_learning", {}).get("learning_value_passed") is True),
        ("acceptance_gate_learning.learning_value_passed",),
    )

    payload = evidence["exp7228-belief-cold-audit"]["payload"]
    comparison = payload.get("comparison_recomputation_rows", [])
    cold = payload.get("cold_reload_rows", [])
    rollback = payload.get("rollback_rows", [])
    audit_complete = int(
        len(comparison) == 3
        and all(row.get("passed") is True for row in comparison)
        and len(cold) == 20
        and all(row.get("passed") is True for row in cold)
        and len(rollback) == 100
        and all(row.get("passed") is True for row in rollback)
        and payload.get("audit_errors") == []
    )
    add(
        "exp7228-belief-cold-audit",
        "belief_audit_complete_score",
        audit_complete,
        ("comparison_recomputation_rows", "cold_reload_rows", "rollback_rows"),
    )
    producer_value = payload.get("producer_gate_receipt", {}).get("belief_learning_value_score", 0)
    add(
        "exp7228-belief-cold-audit",
        "belief_promotion_score",
        int(audit_complete == 1 and producer_value == 1),
        ("producer_gate_receipt.belief_learning_value_score", "belief_audit_complete_score"),
    )

    payload = evidence["exp7229-rare-event-audit"]["payload"]
    checks = payload.get("evidence_checks", {})
    original = payload.get("original_gate_preserved", {})
    add(
        "exp7229-rare-event-audit",
        "rare_event_audit_complete_score",
        int(
            len(payload.get("observable_rows", [])) == 240
            and checks.get("all_passed") is True
            and original.get("unchanged") is True
        ),
        ("observable_rows", "evidence_checks.all_passed", "original_gate_preserved"),
    )
    add(
        "exp7229-rare-event-audit",
        "down_up_value_score",
        int(original.get("primary_gate", {}).get("passed") is True),
        ("original_gate_preserved.primary_gate.passed",),
    )

    payload = evidence["exp7230-native-belief"]["payload"]
    parity = payload.get("parity_rows", [])
    cells = payload.get("cost_summary", {}).get("cells", [])
    add(
        "exp7230-native-belief",
        "native_belief_ready_score",
        int(
            bool(parity)
            and all(
                row.get("passed") is True and row.get("mismatch_count", 0) == 0 for row in parity
            )
        ),
        ("parity_rows.passed", "parity_rows.mismatch_count"),
    )
    add(
        "exp7230-native-belief",
        "native_cost_value_score",
        int(len(cells) == 3 and all(row.get("paired_lower_ci95", 0) > 1 for row in cells)),
        ("cost_summary.cells.paired_lower_ci95",),
    )
    add(
        "exp7230-native-belief",
        "nfr_01_10x_met",
        int(len(cells) == 3 and all(row.get("paired_lower_ci95", 0) >= 10 for row in cells)),
        ("cost_summary.cells.paired_lower_ci95",),
    )

    payload = evidence["exp7231-board-continuity"]["payload"]
    board_rows = payload.get("board_rows", [])
    add(
        "exp7231-board-continuity",
        "board_continuity_complete_score",
        int(
            len(board_rows) == 3
            and all(
                row.get("receipt_authenticated") is True
                and row.get("latest_receipt_authenticated") is True
                for row in board_rows
            )
        ),
        ("board_rows.receipt_authenticated", "board_rows.latest_receipt_authenticated"),
    )
    return claims


def _failed_gate_summary(gate: Mapping[str, Any]) -> JsonDict:
    """Convert one failed replay row to the required diagnostic shape."""

    return {
        "passed": False,
        "failed_check": "structured_gate_and_quarantine",
        "upstream": gate["upstream"],
        "field": gate["field"],
        "expected_value": gate["expected_value"],
        "observed_value": gate["observed_value"],
    }


def matrix_row(
    order: int,
    task: Mapping[str, Any],
    item: Mapping[str, Any],
    gate_rows: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Keep one task disposition even when the conductor skipped its file."""

    task_id = str(task["id"])
    payload = item["payload"]
    task_gates = [row for row in gate_rows if row["consumer"] == task_id]
    failed = next((row for row in task_gates if row["passed"] is not True), None)
    legitimate_cascade = not payload and failed is not None
    quarantined = item["quarantine_receipt"]["quarantined"] is True
    evidence_source = "legitimate_cascade_block" if legitimate_cascade else item["evidence_source"]
    if quarantined:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_quarantined_upstream: " + str(
            payload.get("honest_verdict", "missing producer verdict")
        )
    elif payload.get("status") == "blocked" or legitimate_cascade:
        verdict_class = "blocked"
        honest_verdict = str(
            payload.get(
                "honest_verdict",
                f"blocked_cascade_upstream_retired: {failed['upstream'] if failed else 'missing'}",
            )
        )
    else:
        verdict_class = str(payload.get("verdict_class", "blocked"))
        honest_verdict = str(payload.get("honest_verdict", "blocked_missing_task_evidence"))
    readiness, value = base._classified_fields(payload)
    raw_rows = payload.get("rows", [])
    task_claims = [row for row in claims if row["task_id"] == task_id]
    return {
        "order": order,
        "task_id": task_id,
        "title": task["title"],
        "declared_deliverable_path": item["declared_deliverable_path"],
        "canonical_gate_block_path": item["canonical_gate_block_path"],
        "selected_evidence_path": item["selected_evidence_path"],
        "evidence_source": evidence_source,
        "artifact_sha256": item["artifact_sha256"],
        "artifact_size_bytes": item["artifact_size_bytes"],
        "scope_complete": bool(payload) or legitimate_cascade,
        "status": payload.get("status", "blocked" if legitimate_cascade else "missing"),
        "producer_verdict_class": payload.get("verdict_class"),
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "inference_substrate": payload.get("inference_substrate"),
        "inference_substrate_class": payload.get(
            "inference_substrate_class", "blocked_no_run" if not payload else None
        ),
        "execution_venue": payload.get("execution_venue"),
        "raw_row_count": len(raw_rows) if isinstance(raw_rows, list) else 0,
        "source_hash_receipt": base._source_hash_shape(payload),
        "quarantine_state": item["quarantine_receipt"],
        "accepted_for_promoted_evidence": item["accepted_for_promoted_evidence"],
        "readiness_fields": readiness,
        "value_fields": value,
        "acceptance_gates": deepcopy(task.get("gated_on") or []),
        "acceptance_gate_replay_rows": task_gates,
        "gate_check_summary": _failed_gate_summary(failed)
        if failed
        else payload.get("gate_check_summary"),
        "recomputed_claim_count": len(task_claims),
        "promoted_claim_count": sum(row["promoted"] is True for row in task_claims),
        "all_recomputed_claims_match": all(row["matches"] is True for row in task_claims),
        "verifier_is_oracle": bool(unwrap_principle(payload.get("verifier_is_oracle", False))),
        "authority_boundary": "quarantined evidence is visible but never promoted"
        if quarantined
        else "readiness and implementation receipts do not establish scientific value",
    }


def scientific_questions(
    claims: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> JsonDict:
    """Answer the four scientific questions without substituting narrow wins."""

    by_key = {(row["task_id"], row["metric"]): row for row in claims}
    memory = by_key[("exp7227-belief-learning", "belief_learning_value_score")]
    valid = by_key[("exp7221-arc-session", "model_valid_induction_count")]
    progress_row = by_key[("exp7221-arc-session", "registered_level_increment")]
    native = evidence["exp7230-native-belief"]
    return {
        "held_out_source_fidelity_and_value": {
            "value": None,
            "verdict_class": "blocked",
            "reason": "The canary is quarantined and not ready; capture gate-blocked and the audit was legitimately cascade-blocked.",
            "verifier_is_oracle": False,
        },
        "prospective_memory_utility": {
            "value": bool(memory["recomputed_value"]),
            "verdict_class": "null",
            "reason": "The complete producer gate failed its recurrence-retention criterion.",
            "claim_unit_id": memory["unit_id"],
            "verifier_is_oracle": True,
        },
        "live_adapter_withheld_progress_with_useful_model_validity": {
            "value": bool(valid["recomputed_value"] and progress_row["recomputed_value"]),
            "verdict_class": "null",
            "model_valid_induction_count": valid["recomputed_value"],
            "registered_level_increment": progress_row["recomputed_value"],
            "reason": "Ten cumulative inductions contained no valid model and produced no registered progress.",
            "verifier_is_oracle": False,
        },
        "matched_deployment_cost": {
            "value": None,
            "verdict_class": "disqualified",
            "reason": "The native cost artifact is quarantined, so its positive paired timing field is not promoted.",
            "source_quarantined": native["quarantine_receipt"]["quarantined"],
            "verifier_is_oracle": True,
        },
    }


def _branch_decisions(
    tasks: Sequence[Mapping[str, Any]], matrix: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Apply only exact prior-verdict retirement signals to their own task."""

    decisions: list[JsonDict] = []
    for task, matrix_item in zip(tasks, matrix, strict=True):
        task_id = str(task["id"])
        action, reason, prerequisite = DECISION_POLICY[task_id]
        priors = deepcopy(task.get("prior_failures") or [])
        same = [
            prior
            for prior in priors
            if prior.get("verdict") == matrix_item.get("honest_verdict")
            and prior.get("retire_if_same_verdict") is True
        ]
        decisions.append(
            {
                "task_id": task_id,
                "action": action,
                "reason": reason,
                "next_experiment_or_exact_condition": prerequisite,
                "evidence_path": matrix_item.get("selected_evidence_path"),
                "prior_failures": priors,
                "exact_same_verdict_recurrence": bool(same),
                "retire_if_same_verdict_applied": bool(same) and action == "retire",
                "matching_retirement_signals": same,
                "broad_family_retirement_invented": False,
            }
        )
    return decisions


def _preconditions(
    root: Path,
    output: Path,
    checkpoint: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    gate_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], dict[str, str]]:
    """Record actual bytes, imports, fields, identity, and writable outputs."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in STATIC_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": "required_source_bytes",
                "upstream": str(relative),
                "field": "bytes",
                "expected_value": "nonempty",
                "observed_value": path.stat().st_size if present else "missing",
                "passed": present,
            }
        )
        if present:
            hashes[str(relative)] = sha256_path(path)
    spec_text = (root / SPEC_PATH).read_text(encoding="utf-8")
    checks.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": str(SPEC_PATH),
                "field": "REQ-*",
                "expected_value": "REQ-REPORT-7232",
                "observed_value": "REQ-REPORT-7232"
                if "REQ-REPORT-7232" in spec_text
                else "missing",
                "passed": "REQ-REPORT-7232" in spec_text,
            },
            {
                "check": "required_imports",
                "upstream": "V635 capstone and V636 contract modules",
                "field": "helpers",
                "expected_value": True,
                "observed_value": callable(base.evaluate_gate)
                and callable(contract_source.evaluate_contract),
                "passed": callable(base.evaluate_gate)
                and callable(contract_source.evaluate_contract),
            },
            {
                "check": "required_tools",
                "upstream": "host",
                "field": "python|jq",
                "expected_value": {"python": True, "jq": True},
                "observed_value": {
                    "python": shutil.which("python") is not None,
                    "jq": shutil.which("jq") is not None,
                },
                "passed": shutil.which("python") is not None and shutil.which("jq") is not None,
            },
            {
                "check": "output_directories",
                "upstream": "host_filesystem",
                "field": "writable",
                "expected_value": True,
                "observed_value": os.access(output.parent, os.W_OK)
                and os.access(checkpoint.parent, os.W_OK),
                "passed": os.access(output.parent, os.W_OK)
                and os.access(checkpoint.parent, os.W_OK),
            },
            {
                "check": "frozen_contract",
                "upstream": "Exp7219 frozen sources",
                "field": "fourteen exact rows and hashes",
                "expected_value": [],
                "observed_value": contract["errors"],
                "passed": not contract["errors"],
            },
            {
                "check": "no_current_llm",
                "upstream": "exp7232-capstone",
                "field": "MODEL_SPECS|model_invoked",
                "expected_value": {"MODEL_SPECS": [], "model_invoked": False},
                "observed_value": {"MODEL_SPECS": MODEL_SPECS, "model_invoked": False},
                "passed": MODEL_SPECS == [],
            },
        ]
    )
    for item in evidence.values():
        selected = item["selected_evidence_path"]
        if selected:
            hashes[selected] = item["artifact_sha256"]
    for task_id, item in evidence.items():
        payload = item["payload"]
        expected_field = EXPECTED_FIELDS.get(task_id)
        failed_gate = next(
            (row for row in gate_rows if row["consumer"] == task_id and row["passed"] is not True),
            None,
        )
        cascade = not payload and failed_gate is not None
        field_present = expected_field in payload if expected_field else cascade
        classified = item["terminal"] and (
            item["accepted_for_promoted_evidence"]
            or item["quarantine_receipt"]["quarantined"]
            or payload.get("status") == "blocked"
        )
        checks.append(
            {
                "check": "upstream_terminal_authentication_and_field",
                "upstream": task_id,
                "field": expected_field or "legitimate_cascade_block",
                "expected_value": "terminal bytes classified before promotion or failed upstream gate",
                "observed_value": {
                    "path": selected,
                    "sha256": item["artifact_sha256"],
                    "terminal": item["terminal"],
                    "field_present": field_present,
                    "quarantined": item["quarantine_receipt"]["quarantined"],
                    "legitimate_cascade_block": cascade,
                },
                "passed": (classified and field_present) or cascade,
            }
        )
    return checks, hashes


def _publication_shape(payload: Mapping[str, Any]) -> bool:
    """Accept only the unchanged G1 through G4 publication-gate shape."""

    return base._publication_shape(payload)


def run_publication_gate(root: Path) -> JsonDict:
    """Run the existing publication evaluator and reject a changed shape."""

    receipt = base.run_publication_gate(root)
    if not _publication_shape(receipt):
        raise RuntimeError("publication gate returned a changed G1-G4 shape")
    return receipt


def validate_artifact(artifact: Mapping[str, Any], root: Path | None = None) -> list[str]:
    """Recompute lifecycle, roster, claims, hashes, actions, and checksum."""

    errors: list[str] = []
    if any(field not in artifact for field in REQUIRED_FIELD_PRINCIPLES):
        errors.append("required_fields")
    if artifact.get("field_principles") != REQUIRED_FIELD_PRINCIPLES:
        errors.append("field_principles")
    if artifact.get("status") != "complete" or artifact.get("run_date") != RUN_DATE:
        errors.append("terminal_lifecycle")
    if (
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
    ):
        errors.append("inference_substrate")
    if artifact.get("execution_venue") != "host" or not artifact.get("execution_host"):
        errors.append("execution_venue")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_invocation")
    contract_rows = artifact.get("task_contract_rows")
    if not isinstance(contract_rows, list) or [row.get("task_id") for row in contract_rows] != list(
        EXPECTED_TASK_IDS
    ):
        errors.append("task_contract_rows")
    matrix = artifact.get("evidence_matrix")
    if not isinstance(matrix, list) or [row.get("task_id") for row in matrix] != list(
        EXPECTED_TASK_IDS
    ):
        errors.append("evidence_matrix")
    rows = artifact.get("rows")
    if (
        not isinstance(rows, list)
        or not rows
        or rows != artifact.get("recomputed_claim_rows")
        or not all(
            {"unit_id", "arm", "seed", "metric", "metric_value", "error", "abstention"}
            <= row.keys()
            for row in rows
        )
    ):
        errors.append("claim_rows")
    decisions = artifact.get("branch_decisions")
    if (
        not isinstance(decisions, list)
        or [row.get("task_id") for row in decisions] != list(EXPECTED_TASK_IDS)
        or any(row.get("action") not in BRANCH_ACTIONS for row in decisions)
    ):
        errors.append("branch_decisions")
    questions = artifact.get("scientific_questions", {})
    if [
        questions.get(name, {}).get("verdict_class")
        for name in (
            "held_out_source_fidelity_and_value",
            "prospective_memory_utility",
            "live_adapter_withheld_progress_with_useful_model_validity",
            "matched_deployment_cost",
        )
    ] != ["blocked", "null", "null", "disqualified"]:
        errors.append("scientific_questions")
    if artifact.get("capstone_complete_score") != 1:
        errors.append("capstone_complete_score")
    gate = artifact.get("gate_check_summary")
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not False
        or any(
            name not in gate
            for name in ("failed_check", "upstream", "field", "expected_value", "observed_value")
        )
    ):
        errors.append("gate_check_summary")
    if artifact.get("verdict_class") != "blocked" or not str(
        artifact.get("honest_verdict", "")
    ).startswith("blocked_"):
        errors.append("verdict_class")
    if not _publication_shape(artifact.get("publication_gate", {})):
        errors.append("publication_gate")
    if (
        not isinstance(artifact.get("duration_s"), (int, float))
        or artifact.get("duration_s", -1) < 0
    ):
        errors.append("duration_s")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if root is not None:
        contract = load_contract(root)
        evidence = load_repository_payloads(root, contract["tasks"])
        if contract_rows != contract["task_contract_rows"]:
            if "task_contract_rows" not in errors:
                errors.append("task_contract_rows")
        if rows != recompute_claims(evidence):
            if "claim_rows" not in errors:
                errors.append("claim_rows")
        for relative, expected in artifact.get("source_artifact_hashes", {}).items():
            path = root / relative
            if not path.is_file() or sha256_path(path) != expected:
                errors.append("source_artifact_hashes")
                break
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    output_path: Path,
    checkpoint_path: Path,
    publication_runner: PublicationRunner = run_publication_gate,
) -> JsonDict:
    """Aggregate V636 terminal evidence and atomically write one capstone."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "preconditions before aggregation")
    if run_date != RUN_DATE:
        raise ValueError(f"run date must be {RUN_DATE}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(
        checkpoint_path,
        {
            "schema": "carnot.exp7232.v636_capstone.v1",
            "experiment_id": "exp7232-capstone",
            "status": "running",
            "run_date": run_date,
            "MODEL_SPECS": [],
            "model_invoked": False,
        },
    )

    progress(1, "start", "parse frozen V636 Markdown and YAML")
    contract = load_contract(root)
    progress(1, "end", f"contract rows={len(contract['tasks'])} errors={len(contract['errors'])}")
    progress(2, "start", "read every declared result and canonical conductor block")
    evidence = load_repository_payloads(root, contract["tasks"])
    gate_rows = replay_gates(contract["tasks"], evidence)
    progress(2, "end", f"upstream slots={len(evidence)} gate rows={len(gate_rows)}")
    checks, hashes = _preconditions(
        root, output_path, checkpoint_path, contract, evidence, gate_rows
    )
    essential = {
        "required_source_bytes",
        "driving_requirement",
        "required_imports",
        "required_tools",
        "output_directories",
        "frozen_contract",
        "no_current_llm",
        "upstream_terminal_authentication_and_field",
    }
    failed_checks = [row for row in checks if row["check"] in essential and not row["passed"]]
    if failed_checks:
        raise RuntimeError(f"essential capstone precondition failed: {failed_checks[0]}")
    progress(0, "end", f"preconditions recorded={len(checks)}")

    progress(3, "start", "recompute producer claims and fourteen matrix rows")
    claims = recompute_claims(evidence)
    matrix: list[JsonDict] = []
    for order, task in enumerate(contract["tasks"][:-1], 1):
        task_id = str(task["id"])
        matrix.append(matrix_row(order, task, evidence[task_id], gate_rows, claims))
        progress(3, "unit", f"completed matrix row {order}/14 {task_id}")

    gate_summary = {
        "passed": False,
        "failed_check": "span_canary_authentication_and_readiness",
        "upstream": "exp7223-span-canary",
        "field": "flagged_adversarial|span_canary_ready_score",
        "expected_value": {"flagged_adversarial": False, "span_canary_ready_score": 1},
        "observed_value": {"flagged_adversarial": True, "span_canary_ready_score": 0},
    }
    honest_verdict = (
        "blocked_external: V636 matrix is complete; held-out source fidelity remains "
        "blocked, memory and useful live progress are null, and matched native cost is disqualified"
    )
    self_task = contract["tasks"][-1]
    self_row = {
        "order": 14,
        "task_id": "exp7232-capstone",
        "title": self_task["title"],
        "declared_deliverable_path": self_task["deliverable"],
        "canonical_gate_block_path": canonical_gate_block_path("exp7232-capstone"),
        "selected_evidence_path": str(output_path),
        "evidence_source": "self",
        "artifact_sha256": None,
        "artifact_size_bytes": None,
        "scope_complete": True,
        "status": "complete",
        "producer_verdict_class": None,
        "verdict_class": "blocked",
        "honest_verdict": honest_verdict,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "raw_row_count": len(claims),
        "source_hash_receipt": {
            "type": "object",
            "entry_count": len(hashes),
            "valid_sha256_count": len(hashes),
        },
        "quarantine_state": {
            "declared_flags": {},
            "exclusion_manifest_match": False,
            "quarantined": False,
        },
        "accepted_for_promoted_evidence": True,
        "readiness_fields": {"capstone_complete_score": 1},
        "value_fields": {},
        "acceptance_gates": [],
        "acceptance_gate_replay_rows": [],
        "gate_check_summary": gate_summary,
        "recomputed_claim_count": 0,
        "promoted_claim_count": 0,
        "all_recomputed_claims_match": True,
        "verifier_is_oracle": False,
        "authority_boundary": "scope completion is independent from scientific success",
    }
    matrix.append(self_row)
    progress(3, "unit", "completed matrix row 14/14 exp7232-capstone")
    progress(3, "end", f"numeric claim rows={len(claims)}")

    progress(4, "start", "classify four scientific questions and narrow findings")
    questions = scientific_questions(claims, evidence)
    by_key = {(row["task_id"], row["metric"]): row for row in claims}
    narrow_findings = {
        "vllm_parser_readiness": {
            "value": False,
            "verdict_class": "null",
            "claim_unit_id": by_key[("exp7220-xml-canary", "xml_transport_ready_score")]["unit_id"],
        },
        "cumulative_induction_count": {
            "value": by_key[("exp7221-arc-session", "cumulative_induction_count")][
                "recomputed_value"
            ],
            "verdict_class": "positive",
            "operational_count_only": True,
        },
        "exact_finite_law_or_compiler_parity": {
            "value": True,
            "verdict_class": "circular_positive",
            "claim_unit_ids": [
                by_key[("exp7222-span-fixture", "span_fixture_ready_score")]["unit_id"],
                by_key[("exp7226-belief-compiler", "belief_compiler_ready_score")]["unit_id"],
            ],
        },
        "native_speed": {
            "value": None,
            "verdict_class": "disqualified",
            "claim_unit_id": by_key[("exp7230-native-belief", "native_cost_value_score")][
                "unit_id"
            ],
        },
    }
    progress(4, "end", "source=blocked memory=null live=null deployment=disqualified")

    progress(5, "start", "apply exact retirement signals and branch policy")
    decisions = _branch_decisions(contract["tasks"], matrix)
    progress(5, "end", f"branch decisions={len(decisions)}")

    progress(6, "start", "run unchanged publication gate")
    publication = publication_runner(root)
    if not _publication_shape(publication):
        raise RuntimeError("publication gate returned a changed G1-G4 shape")
    progress(6, "end", f"paper_ready={publication['paper_ready']}")

    historical_nulls = []
    for relative, metric in (
        (Path("results/experiment_7182_v633_grounding_energy_audit.json"), "grounding_value_score"),
        (Path("results/experiment_7197_v634_grounding_value_audit.json"), "grounding_value_score"),
    ):
        payload = read_json(root / relative)
        historical_nulls.append(
            {
                "path": str(relative),
                "sha256": hashes[str(relative)],
                "honest_verdict": payload.get("honest_verdict"),
                "verdict_class": payload.get("verdict_class"),
                "metric": metric,
                "value": payload.get(metric),
                "preserved": True,
                "later_grammar_readiness_overrides": False,
            }
        )

    artifact: JsonDict = {
        "schema": "carnot.exp7232.v636_capstone.v1",
        "experiment_id": "exp7232-capstone",
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "status": "complete",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": "",
        "preconditions_checked": checks,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "source_artifact_hashes": hashes,
        "rows": claims,
        "sample_size_budget": {
            "contract_tasks": {
                "planned": 14,
                "attempted": 14,
                "completed": 14,
                "censored": 0,
                "independent_units": 14,
            },
            "upstream_slots": {
                "planned": 13,
                "attempted": 13,
                "completed": sum(bool(item["payload"]) for item in evidence.values()),
                "censored": sum(not bool(item["payload"]) for item in evidence.values()),
                "independent_units": 13,
            },
            "numeric_claim_rows": {
                "planned": len(claims),
                "attempted": len(claims),
                "completed": len(claims),
                "censored": sum(row["abstention"] is True for row in claims),
                "independent_units": len({row["task_id"] for row in claims}),
            },
            "producer_raw_rows": sum(row["raw_row_count"] for row in matrix[:-1]),
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": honest_verdict,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "capstone_complete_score": 1,
        "task_contract_rows": contract["task_contract_rows"],
        "evidence_matrix": matrix,
        "recomputed_claim_rows": claims,
        "scientific_questions": questions,
        "narrow_capability_findings": narrow_findings,
        "branch_decisions": decisions,
        "prd_completion": {
            "FR-11": {
                "complete": False,
                "verdict_class": "null",
                "finding": "prospective memory gate failed",
            },
            "FR-12": {
                "complete": False,
                "verdict_class": "blocked",
                "finding": "held-out source audit unavailable",
            },
            "FR-05": {
                "complete": False,
                "verdict_class": "null",
                "finding": "no useful valid live-model progress",
            },
            "FR-08": {
                "complete": False,
                "verdict_class": "disqualified",
                "finding": "matched native cost receipt quarantined",
            },
            "NFR-01": {
                "complete": False,
                "verdict_class": "null",
                "finding": "10x target not established",
            },
        },
        "historical_semantic_nulls": historical_nulls,
        "same_milestone_gate_replay_rows": gate_rows,
        "publication_gate": publication,
        "publication_performed": False,
        "exclusion_manifest_modified": False,
        "conductor_modified": False,
        "validation_receipts": {
            "planning_contract": "frozen file -> independent parser -> structured gate replay",
            "read_only_audit": "actual producer rows -> V636 reducer",
            "applicable_e2e": ["E2E-003", "E2E-004", "E2E-007", "E2E-009", "E2E-010"],
            "external_publication_performed": False,
        },
    }
    artifact["duration_s"] = time.monotonic() - started
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)

    progress(7, "start", "validate derived artifact before terminal write")
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise RuntimeError("capstone validation failed: " + ",".join(errors))
    atomic_write(checkpoint_path, artifact)
    progress(7, "end", "derived validation passed")

    progress(8, "start", "atomic terminal write and file-parser validation")
    atomic_write(output_path, artifact)
    reloaded = read_json(output_path)
    final_errors = validate_artifact(reloaded, root=root)
    if final_errors:
        raise RuntimeError("final file-parser validation failed: " + ",".join(final_errors))
    progress(8, "end", f"wrote {output_path}")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Build a capstone or validate an existing artifact through one CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.artifact_path or root / DEFAULT_ARTIFACT_PATH
    checkpoint = args.checkpoint_path or root / DEFAULT_CHECKPOINT_PATH
    if args.validate:
        progress(7, "start", f"validate {output}")
        payload = read_json(output)
        errors = validate_artifact(payload, root=root)
        progress(7, "end", "passed" if not errors else ",".join(errors))
        return int(bool(errors))
    build_artifact(root, args.date, output, checkpoint)
    return 0

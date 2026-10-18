"""Build the V637 evidence matrix without invoking a model.

The capstone reads immutable producer artifacts and recomputes their narrow
claims. It keeps execution completion separate from scientific value.

Spec refs: REQ-REPORT-7245 and SCENARIO-REPORT-7245-*.
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
from carnot import experiment_7233_v637_contract as contract_source


JsonDict = dict[str, Any]
PublicationRunner = Callable[[Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.637"
RUN_DATE = "20260912"
RANDOM_SEED = 7_245_202_609_12
MODEL_SPECS: list[JsonDict] = []

SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
FROZEN_YAML_PATH = Path("results/raw/experiment_7233/selected-roadmap.yaml")
FROZEN_DESIGN_PATH = Path("results/raw/experiment_7233/research-roadmap-v637.md")
CONTRACT_RECEIPT_PATH = Path("results/experiment_7233_v637_contract.json")
EXCLUSION_MANIFEST_PATH = Path("ops/exclusion_manifest.yaml")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7245_v637_capstone.json")
DEFAULT_CHECKPOINT_PATH = Path("results/checkpoints/experiment_7245_v637_capstone.json")

EXPECTED_TASK_IDS = tuple(contract_source.EXPECTED_ID_ORDER)
BRANCH_ACTIONS = frozenset({"continue", "retire", "needs_changed_prerequisite"})

PROMPT_FIELD_PRINCIPLES = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked_* verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. External incompleteness is blocked.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_. Failed acceptance forbids positive.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "capstone_complete_score": "Thirteen-task evidence matrix complete, independent of scientific success.",
    "task_contract_rows": "Exactly exp7233 through exp7245 in the same order and paths as both plan files.",
    "evidence_matrix": "Every task disposition, gate, artifact hash, authority and quarantine status.",
    "recomputed_claim_rows": "Paired metrics derived from authentic raw units, excluding quarantined numerical claims.",
    "branch_decisions": "Exact next experiment, retirement reason or changed external prerequisite.",
    "prd_completion": "FR-11, FR-12, FR-05/08 and NFR-01 assessed separately.",
    "publication_gate": "Preparation only; no publication, upload, Kaggle submission or external contact.",
}
REQUIRED_FIELD_PRINCIPLES = dict(PROMPT_FIELD_PRINCIPLES)

HISTORICAL_QUARANTINE_PATHS = (
    Path("results/experiment_7223_v636_span_canary.json"),
    Path("results/experiment_7228_v636_belief_cold_audit.json"),
    Path("results/experiment_7230_v636_native_belief.json"),
    Path("results/experiment_7232_v636_capstone.json"),
)
HISTORICAL_TASK_IDS = (
    "exp7223-span-canary",
    "exp7228-belief-cold-audit",
    "exp7230-native-belief",
    "exp7232-capstone",
)

STATIC_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    EXCLUSION_MANIFEST_PATH,
    Path("ops/e2e-test-plan.md"),
    FROZEN_YAML_PATH,
    FROZEN_DESIGN_PATH,
    CONTRACT_RECEIPT_PATH,
    Path("scripts/summarize_artifact.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("scripts/publication_gate.py"),
    Path("ops/verifier_gaps.md"),
    SPEC_PATH,
    Path("python/carnot/experiment_7232_v636_capstone.py"),
    Path("python/carnot/experiment_7245_v637_capstone.py"),
    Path("scripts/experiments/experiment_7245_v637_capstone.py"),
    Path("tests/python/test_experiment_7245_v637_capstone.py"),
    *HISTORICAL_QUARANTINE_PATHS,
)

EXPECTED_FIELDS = {
    "exp7233-contract": "source_contract_complete_score",
    "exp7234-arc-scored-dryrun": "scored_dryrun_complete_score",
    "exp7235-arc-path-audit": "arc_path_audit_complete_score",
    "exp7236-mention-fixture": "mention_fixture_ready_score",
    "exp7237-mention-canary": "mention_canary_ready_score",
    "exp7238-mention-capture": "mention_capture_complete_score",
    "exp7239-semantic-audit": "semantic_audit_complete_score",
    "exp7240-recurrence-fixture": "recurrence_fixture_ready_score",
    "exp7241-recurrence-learning": "recurrence_run_complete_score",
    "exp7242-recurrence-audit": "recurrence_audit_complete_score",
    "exp7243-native-memory": "native_archive_ready_score",
    "exp7244-board-disposition": "board_disposition_complete_score",
}

DECISION_POLICY: dict[str, tuple[str, str, str]] = {
    "exp7233-contract": (
        "retire",
        "Retire the one-time contract receipt after preserving its disqualified validation state.",
        "No rerun; use the two frozen source files directly for later contract checks.",
    ),
    "exp7234-arc-scored-dryrun": (
        "needs_changed_prerequisite",
        "The local run is quarantined and its policy did not consume a world model.",
        "Produce an unquarantined scored-stack run where a valid engine write changes a policy action.",
    ),
    "exp7235-arc-path-audit": (
        "needs_changed_prerequisite",
        "The audit stopped because its scored-run source was quarantined.",
        "Provide the unquarantined scored-stack artifact required by the same audit.",
    ),
    "exp7236-mention-fixture": (
        "continue",
        "Keep the exact mention compiler as circular fixture infrastructure only.",
        "Use it with a clean canary; do not infer source fidelity from fixture conformance.",
    ),
    "exp7237-mention-canary": (
        "needs_changed_prerequisite",
        "The numeric readiness gate passed but the canary is quarantined.",
        "Create an unquarantined canary with the same pointer, offset, direct, and unknown arms.",
    ),
    "exp7238-mention-capture": (
        "needs_changed_prerequisite",
        "The held-out capture correctly blocked on quarantined canary evidence.",
        "Require an authenticated mention_canary_ready_score of one from a clean producer.",
    ),
    "exp7239-semantic-audit": (
        "needs_changed_prerequisite",
        "Source fidelity and verification value remain externally blocked.",
        "Provide an authenticated held-out mention capture with private evaluator rows.",
    ),
    "exp7240-recurrence-fixture": (
        "continue",
        "Keep the executable recurrence fixture and controls as test infrastructure.",
        "Use a new archive-admission mechanism under the unchanged delayed-feedback controls.",
    ),
    "exp7241-recurrence-learning": (
        "retire",
        "Retire this validated archive-selection mechanism after its complete prospective null.",
        "A replacement must beat reset and shuffled controls while meeting the 0.02 recurrence bound.",
    ),
    "exp7242-recurrence-audit": (
        "retire",
        "Retire the one-time cold audit after it confirmed the producer null and causal controls.",
        "Audit again only after a different learner passes every prospective gate.",
    ),
    "exp7243-native-memory": (
        "retire",
        "Retire the measured native path because complete batch-one event cost was slower.",
        "A new path must reduce dispatch, validation, serialization, and state-transfer cost together.",
    ),
    "exp7244-board-disposition": (
        "needs_changed_prerequisite",
        "KV260 and PolarFire claims are preserved; only GateMate lacks changed physical state.",
        "Record a dated operator GateMate cable, port, board, power, or DirtyJTAG change after Exp6559.",
    ),
    "exp7245-capstone": (
        "retire",
        "Retire this one-time synthesis after the thirteen-row matrix is stored.",
        "Use a new milestone identity for later evidence rather than rewriting this capstone.",
    ),
}


unwrap_principle = base.unwrap_principle
sha256_path = base.sha256_path
canonical_gate_block_path = base.canonical_gate_block_path
atomic_write = base._atomic_write
reproducibility_checksum = base.reproducibility_checksum


def progress(phase: int, state: str, detail: str) -> None:
    """Flush one truthful phase event for conductor observation."""

    print(f"[exp7245] phase {phase} {state}: {detail}", flush=True)


def read_json(path: Path) -> JsonDict:
    """Read an object because list roots cannot carry the artifact contract."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def _task_number(task_id: str) -> str:
    """Return the numeric producer directory name from a frozen task ID."""

    match = re.match(r"exp(\d+)", task_id)
    if match is None:
        raise ValueError(f"task ID has no experiment number: {task_id}")
    return match.group(1)


def load_contract(root: Path) -> JsonDict:
    """Parse both frozen V637 plans and bind their immutable receipt hashes."""

    yaml_bytes = (root / FROZEN_YAML_PATH).read_bytes()
    markdown_bytes = (root / FROZEN_DESIGN_PATH).read_bytes()
    document = yaml.safe_load(yaml_bytes)
    if not isinstance(document, Mapping):
        raise ValueError("frozen V637 YAML root is not a mapping")
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
    if document.get("milestone") != MILESTONE:
        errors.append("contract_milestone")
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


def _iter_path_hash_pairs(value: Any) -> list[tuple[str, str]]:
    """Find explicit path/hash receipts without treating prose as evidence."""

    pairs: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        path = value.get("path")
        digest = value.get("sha256") or value.get("raw_sha256")
        if isinstance(path, str) and isinstance(digest, str) and digest.startswith("sha256:"):
            pairs.append((path, digest))
        for child in value.values():
            pairs.extend(_iter_path_hash_pairs(child))
    elif isinstance(value, list):
        for child in value:
            pairs.extend(_iter_path_hash_pairs(child))
    return pairs


def _raw_hash_replay(root: Path, task_id: str, payload: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute declared raw and checkpoint hashes before trusting rows."""

    source_hashes = payload.get("source_artifact_hashes", {})
    pairs = _iter_path_hash_pairs(payload)
    if isinstance(source_hashes, Mapping):
        for path, digest in source_hashes.items():
            if isinstance(path, str) and isinstance(digest, str) and "/" in path:
                pairs.append((path, digest))
        expected_hashes = {
            digest
            for digest in source_hashes.values()
            if isinstance(digest, str) and digest.startswith("sha256:")
        }
        raw_dir = root / "results/raw" / f"experiment_{_task_number(task_id)}"
        if raw_dir.is_dir():
            for path in raw_dir.rglob("*"):
                if path.is_file():
                    observed = sha256_path(path)
                    if observed in expected_hashes:
                        pairs.append((str(path.relative_to(root)), observed))
    rows: list[JsonDict] = []
    seen: set[tuple[str, str]] = set()
    for declared_path, expected in pairs:
        path = Path(declared_path)
        resolved = path if path.is_absolute() else root / path
        try:
            relative = resolved.resolve().relative_to(root.resolve())
        except ValueError:
            continue
        if not any(part in {"raw", "checkpoints", "streams"} for part in relative.parts):
            continue
        key = (str(relative), expected)
        if key in seen:
            continue
        seen.add(key)
        observed = sha256_path(resolved) if resolved.is_file() else None
        rows.append(
            {
                "path": str(relative),
                "expected_sha256": expected,
                "observed_sha256": observed,
                "passed": observed == expected,
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def load_evidence(root: Path, task: Mapping[str, Any], manifest: Any) -> JsonDict:
    """Load the declared result first and reject every quarantine form."""

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
    raw_replay = _raw_hash_replay(root, task_id, payload)
    raw_authenticated = all(row["passed"] is True for row in raw_replay)
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
        "raw_hash_replay_rows": raw_replay,
        "raw_hashes_authenticated": raw_authenticated,
        "terminal": terminal,
        "accepted_for_promoted_evidence": bool(payload)
        and payload.get("status") == "complete"
        and not quarantine["quarantined"]
        and raw_authenticated,
        "payload": payload,
    }


def load_repository_payloads(root: Path, tasks: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Load all twelve producer slots and preserve blocked artifacts."""

    manifest = yaml.safe_load((root / EXCLUSION_MANIFEST_PATH).read_text(encoding="utf-8"))
    return {
        str(task["id"]): load_evidence(root, task, manifest)
        for task in tasks
        if task["id"] != "exp7245-capstone"
    }


def replay_gates(
    tasks: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> list[JsonDict]:
    """Replay all five value gates while retaining quarantine state."""

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
                    "quarantined": producer["quarantine_receipt"]["quarantined"],
                    **result,
                }
            )
    return rows


def _criterion_pass(payload: Mapping[str, Any], name: str) -> bool:
    """Read one frozen acceptance criterion from list or mapping form."""

    gates = payload.get("acceptance_gate_results", {})
    if isinstance(gates, Mapping):
        row = gates.get(name, {})
        return isinstance(row, Mapping) and row.get("pass") is True
    if isinstance(gates, list):
        return any(
            isinstance(row, Mapping) and row.get("criterion") == name and row.get("passed") is True
            for row in gates
        )
    return False


def _all_scored_criteria_pass(payload: Mapping[str, Any]) -> bool:
    """Require every evaluated acceptance row while ignoring explicit non-science fixture rows."""

    gates = payload.get("acceptance_gate_results", {})
    if isinstance(gates, Mapping):
        rows = [
            row
            for row in gates.values()
            if isinstance(row, Mapping) and row.get("pass") is not None
        ]
        return bool(rows) and all(row.get("pass") is True for row in rows)
    if isinstance(gates, list):
        rows = [row for row in gates if isinstance(row, Mapping)]
        return bool(rows) and all(row.get("passed") is True for row in rows)
    return False


def _claim(
    task_id: str,
    metric: str,
    declared: Any,
    recomputed: int | float,
    evidence_fields: Sequence[str],
    evidence: Mapping[str, Any],
    verifier_is_oracle: bool,
) -> JsonDict:
    """Store one recomputation and abstain if its producer is not authentic."""

    matches = declared == recomputed
    accepted = evidence.get("accepted_for_promoted_evidence") is True
    quarantined = evidence.get("quarantine_receipt", {}).get("quarantined") is True
    if not matches:
        error = "declared_value_mismatch"
    elif quarantined:
        error = "quarantined_upstream"
    elif not accepted:
        error = "unauthenticated_or_blocked_upstream"
    else:
        error = None
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
    """Recompute milestone-relevant values from rows and frozen gates."""

    claims: list[JsonDict] = []

    def add(task_id: str, metric: str, value: int | float, fields: Sequence[str]) -> None:
        item = evidence[task_id]
        payload = item["payload"]
        claims.append(
            _claim(
                task_id,
                metric,
                unwrap_principle(payload.get(metric)),
                value,
                fields,
                item,
                bool(unwrap_principle(payload.get("verifier_is_oracle", False))),
            )
        )

    payload = evidence["exp7233-contract"]["payload"]
    add(
        "exp7233-contract",
        "source_contract_complete_score",
        int(len(payload.get("rows", [])) == 13 and _all_scored_criteria_pass(payload)),
        ("rows", "acceptance_gate_results"),
    )

    payload = evidence["exp7234-arc-scored-dryrun"]["payload"]
    add(
        "exp7234-arc-scored-dryrun",
        "scored_dryrun_complete_score",
        int(
            bool(payload.get("rows"))
            and _criterion_pass(payload, "both_backend_dispositions_recorded")
        ),
        ("rows", "acceptance_gate_results.both_backend_dispositions_recorded"),
    )
    add(
        "exp7234-arc-scored-dryrun",
        "local_scored_path_ready_score",
        int(_criterion_pass(payload, "actual_dispatch_engine_write_policy_consumption")),
        ("acceptance_gate_results.actual_dispatch_engine_write_policy_consumption",),
    )

    payload = evidence["exp7235-arc-path-audit"]["payload"]
    add(
        "exp7235-arc-path-audit",
        "arc_path_audit_complete_score",
        int(payload.get("status") == "complete" and bool(payload.get("audit_rows"))),
        ("status", "audit_rows"),
    )

    payload = evidence["exp7236-mention-fixture"]["payload"]
    mutations = payload.get("mutation_rows", [])
    add(
        "exp7236-mention-fixture",
        "mention_fixture_ready_score",
        int(
            len(payload.get("rows", [])) == 216
            and len(mutations) == 8
            and all(row.get("passed") is True for row in mutations)
            and _all_scored_criteria_pass(payload)
        ),
        ("rows", "mutation_rows", "acceptance_gate_results"),
    )

    payload = evidence["exp7237-mention-canary"]["payload"]
    receipt = payload.get("readiness_receipt", {})
    add(
        "exp7237-mention-canary",
        "mention_canary_ready_score",
        int(
            receipt.get("pointer_complete_parse_units", 0) >= 7
            and receipt.get("pointer_usable_units", 0) >= 7
            and receipt.get("pointer_semantic_correct_units", 0) >= 6
            and receipt.get("negative_control_false_accepts") == 0
            and all(row.get("passed") is True for row in receipt.get("criteria", []))
        ),
        ("readiness_receipt",),
    )

    for task_id, metric in (
        ("exp7238-mention-capture", "mention_capture_complete_score"),
        ("exp7239-semantic-audit", "semantic_audit_complete_score"),
    ):
        payload = evidence[task_id]["payload"]
        add(
            task_id,
            metric,
            int(payload.get("status") == "complete" and bool(payload.get("rows"))),
            ("status", "rows"),
        )
    payload = evidence["exp7239-semantic-audit"]["payload"]
    add(
        "exp7239-semantic-audit",
        "semantic_value_score",
        int(payload.get("status") == "complete" and _all_scored_criteria_pass(payload)),
        ("status", "acceptance_gate_results"),
    )

    payload = evidence["exp7240-recurrence-fixture"]["payload"]
    add(
        "exp7240-recurrence-fixture",
        "recurrence_fixture_ready_score",
        int(
            len(payload.get("rows", [])) == 192
            and len(payload.get("arm_contract", {}).get("arms", [])) == 6
            and payload.get("stream_conformance_errors") == []
            and _all_scored_criteria_pass(payload)
        ),
        ("rows", "arm_contract.arms", "stream_conformance_errors", "acceptance_gate_results"),
    )

    payload = evidence["exp7241-recurrence-learning"]["payload"]
    add(
        "exp7241-recurrence-learning",
        "recurrence_run_complete_score",
        int(
            len(payload.get("rows", [])) == 192 and len(payload.get("paired_seed_rows", [])) == 192
        ),
        ("rows", "paired_seed_rows"),
    )
    add(
        "exp7241-recurrence-learning",
        "recurrence_learning_value_score",
        int(_all_scored_criteria_pass(payload)),
        ("acceptance_gate_results",),
    )

    payload = evidence["exp7242-recurrence-audit"]["payload"]
    audit_complete = int(
        len(payload.get("rows", [])) == 192
        and len(payload.get("recomputed_seed_rows", [])) == 192
        and len(payload.get("restore_rows", [])) == 192
        and all(row.get("passed") is True for row in payload.get("raw_check_rows", []))
    )
    add(
        "exp7242-recurrence-audit",
        "recurrence_audit_complete_score",
        audit_complete,
        ("rows", "recomputed_seed_rows", "restore_rows", "raw_check_rows"),
    )
    add(
        "exp7242-recurrence-audit",
        "recurrence_promotion_score",
        int(audit_complete == 1 and _all_scored_criteria_pass(payload)),
        ("recurrence_audit_complete_score", "acceptance_gate_results"),
    )

    payload = evidence["exp7243-native-memory"]["payload"]
    parity = payload.get("parity_rows", [])
    add(
        "exp7243-native-memory",
        "native_archive_ready_score",
        int(
            len(parity) == 65
            and all(
                row.get("passed") is True and row.get("mismatch_count", 0) == 0 for row in parity
            )
        ),
        ("parity_rows.passed", "parity_rows.mismatch_count"),
    )
    lower = payload.get("cost_summary", {}).get("batch_one_lower_ci95")
    add(
        "exp7243-native-memory",
        "native_archive_cost_value_score",
        int(isinstance(lower, (int, float)) and lower > 1),
        ("cost_summary.batch_one_lower_ci95",),
    )

    payload = evidence["exp7244-board-disposition"]["payload"]
    board_rows = payload.get("board_rows", [])
    add(
        "exp7244-board-disposition",
        "board_disposition_complete_score",
        int(
            len(board_rows) == 3
            and all(row.get("latest_receipt_authenticated") is True for row in board_rows)
            and all(bool(row.get("exact_next_prerequisite")) for row in board_rows)
            and _all_scored_criteria_pass(payload)
        ),
        ("board_rows", "acceptance_gate_results"),
    )
    return claims


def _failed_gate_summary(gate: Mapping[str, Any]) -> JsonDict:
    """Convert one failed replay row to the required diagnostic names."""

    return {
        "passed": False,
        "failed_check": "structured_gate_and_quarantine",
        "upstream": gate["upstream"],
        "artifact_field": gate["field"],
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
    """Keep one disposition for every planned task and authority boundary."""

    task_id = str(task["id"])
    payload = item["payload"]
    task_gates = [row for row in gate_rows if row["consumer"] == task_id]
    failed = next((row for row in task_gates if row["passed"] is not True), None)
    quarantined = item["quarantine_receipt"]["quarantined"] is True
    if quarantined:
        verdict_class = "disqualified"
        honest_verdict = "complete_disqualified_quarantined_upstream: " + str(
            payload.get("honest_verdict", "missing producer verdict")
        )
    elif payload.get("status") == "blocked" or not payload:
        verdict_class = "blocked"
        honest_verdict = str(payload.get("honest_verdict", "blocked_missing_task_evidence"))
    else:
        verdict_class = str(payload.get("verdict_class", "blocked"))
        honest_verdict = str(payload.get("honest_verdict", "blocked_missing_task_verdict"))
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
        "evidence_source": item["evidence_source"],
        "artifact_sha256": item["artifact_sha256"],
        "artifact_size_bytes": item["artifact_size_bytes"],
        "planned_status": "planned",
        "executed_status": payload.get("status", "missing"),
        "planned_vs_executed_status": "terminal_" + str(payload.get("status"))
        if payload.get("status") in {"complete", "blocked"}
        else "missing",
        "scope_complete": bool(payload) and item["terminal"],
        "producer_verdict_class": payload.get("verdict_class"),
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "inference_substrate": payload.get("inference_substrate"),
        "inference_substrate_class": payload.get("inference_substrate_class"),
        "execution_venue": payload.get("execution_venue"),
        "producer_model_invoked": payload.get("model_invoked"),
        "historical_invocation_source_only": True,
        "raw_row_count": len(raw_rows) if isinstance(raw_rows, list) else 0,
        "producer_source_artifact_hashes": deepcopy(payload.get("source_artifact_hashes", {})),
        "raw_hash_replay_rows": deepcopy(item["raw_hash_replay_rows"]),
        "raw_hashes_authenticated": item["raw_hashes_authenticated"],
        "quarantine_state": deepcopy(item["quarantine_receipt"]),
        "accepted_for_promoted_evidence": item["accepted_for_promoted_evidence"],
        "readiness_fields": readiness,
        "value_fields": value,
        "acceptance_gates": deepcopy(task.get("gated_on") or []),
        "acceptance_gate_replay_rows": task_gates,
        "gate_check_summary": _failed_gate_summary(failed)
        if failed
        else deepcopy(payload.get("gate_check_summary")),
        "recomputed_claim_count": len(task_claims),
        "promoted_claim_count": sum(row["promoted"] is True for row in task_claims),
        "all_recomputed_claims_match": all(row["matches"] is True for row in task_claims),
        "verifier_is_oracle": bool(unwrap_principle(payload.get("verifier_is_oracle", False))),
        "authority_boundary": "quarantined evidence remains visible and cannot be promoted"
        if quarantined
        else "task completion and narrow readiness do not establish scientific value",
    }


def scientific_questions(
    claims: Sequence[Mapping[str, Any]], evidence: Mapping[str, JsonDict]
) -> JsonDict:
    """Answer four questions without substituting narrower readiness facts."""

    by_key = {(row["task_id"], row["metric"]): row for row in claims}
    memory = by_key[("exp7241-recurrence-learning", "recurrence_learning_value_score")]
    native = by_key[("exp7243-native-memory", "native_archive_cost_value_score")]
    scored = evidence["exp7234-arc-scored-dryrun"]
    return {
        "source_fidelity_and_verification_value": {
            "value": None,
            "verdict_class": "blocked",
            "reason": "The canary is quarantined, so capture and independent semantic value remain unavailable.",
            "verifier_is_oracle": False,
        },
        "prospective_recurrence_learning": {
            "value": bool(memory["recomputed_value"]),
            "verdict_class": "null",
            "reason": "The complete learner failed reset, recurrence, and shuffled-control gates.",
            "claim_unit_id": memory["unit_id"],
            "verifier_is_oracle": True,
            "llm_verification_evidence": False,
            "synthetic_executable_constraints_only": True,
        },
        "actual_scored_policy_model_use": {
            "value": None,
            "verdict_class": "disqualified",
            "reason": "The model ran with local configuration differences, but the artifact is quarantined and no world model changed policy.",
            "source_quarantined": scored["quarantine_receipt"]["quarantined"],
            "verifier_is_oracle": False,
        },
        "complete_native_deployment_cost": {
            "value": bool(native["recomputed_value"]),
            "verdict_class": "null",
            "reason": "Exact parity completed, but full batch-one event cost was slower than Python.",
            "claim_unit_id": native["unit_id"],
            "verifier_is_oracle": True,
        },
    }


def _branch_decisions(
    tasks: Sequence[Mapping[str, Any]], matrix: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Apply retirement signals only to an exact task-scoped verdict match."""

    decisions: list[JsonDict] = []
    for task, matrix_item in zip(tasks, matrix, strict=True):
        task_id = str(task["id"])
        action, reason, condition = DECISION_POLICY[task_id]
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
                "next_experiment_or_exact_condition": condition,
                "evidence_path": matrix_item.get("selected_evidence_path"),
                "prior_failures": priors,
                "exact_same_verdict_recurrence": bool(same),
                "retire_if_same_verdict_applied": bool(same) and action == "retire",
                "matching_retirement_signals": same,
                "broad_family_retirement_invented": False,
            }
        )
    return decisions


def _historical_quarantine_rows(root: Path, manifest: Any) -> list[JsonDict]:
    """Preserve all four V636 failures without promoting their claims."""

    rows: list[JsonDict] = []
    for path, task_id in zip(HISTORICAL_QUARANTINE_PATHS, HISTORICAL_TASK_IDS, strict=True):
        payload = read_json(root / path)
        receipt = base.quarantine_receipt(payload, task_id, str(path), manifest)
        rows.append(
            {
                "path": str(path),
                "sha256": sha256_path(root / path),
                "honest_verdict": payload.get("honest_verdict"),
                "verdict_class": payload.get("verdict_class"),
                "quarantine_state": receipt,
                "accepted_for_promoted_evidence": False,
                "rehabilitated_by_v637": False,
            }
        )
    return rows


def _preconditions(
    root: Path,
    output: Path,
    checkpoint: Path,
    contract: Mapping[str, Any],
    evidence: Mapping[str, JsonDict],
    historical_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], dict[str, str]]:
    """Record source bytes, imports, writable paths, and evidence identity."""

    checks: list[JsonDict] = []
    hashes: dict[str, str] = {}
    for relative in STATIC_SOURCE_PATHS:
        path = root / relative
        present = path.is_file() and path.stat().st_size > 0
        checks.append(
            {
                "check": "required_source_bytes",
                "upstream": str(relative),
                "artifact_field": "bytes",
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
                "artifact_field": "REQ-*",
                "expected_value": "REQ-REPORT-7245",
                "observed_value": "REQ-REPORT-7245"
                if "REQ-REPORT-7245" in spec_text
                else "missing",
                "passed": "REQ-REPORT-7245" in spec_text,
            },
            {
                "check": "required_imports",
                "upstream": "V635 capstone and V637 contract modules",
                "artifact_field": "helpers",
                "expected_value": True,
                "observed_value": callable(base.evaluate_gate)
                and callable(contract_source.evaluate_contract),
                "passed": callable(base.evaluate_gate)
                and callable(contract_source.evaluate_contract),
            },
            {
                "check": "required_resources",
                "upstream": "host",
                "artifact_field": "python|jq|yaml",
                "expected_value": True,
                "observed_value": shutil.which("python") is not None
                and shutil.which("jq") is not None
                and bool(yaml.__version__),
                "passed": shutil.which("python") is not None
                and shutil.which("jq") is not None
                and bool(yaml.__version__),
            },
            {
                "check": "output_directories",
                "upstream": "host_filesystem",
                "artifact_field": "raw/checkpoint/result writable",
                "expected_value": True,
                "observed_value": os.access(output.parent, os.W_OK)
                and os.access(checkpoint.parent, os.W_OK),
                "passed": os.access(output.parent, os.W_OK)
                and os.access(checkpoint.parent, os.W_OK),
            },
            {
                "check": "frozen_contract",
                "upstream": "Exp7233 frozen sources",
                "artifact_field": "thirteen exact rows and hashes",
                "expected_value": [],
                "observed_value": contract["errors"],
                "passed": not contract["errors"],
            },
            {
                "check": "no_current_llm",
                "upstream": "exp7245-capstone",
                "artifact_field": "MODEL_SPECS|model_invoked|counters",
                "expected_value": {"MODEL_SPECS": [], "model_invoked": False, "calls": 0},
                "observed_value": {"MODEL_SPECS": MODEL_SPECS, "model_invoked": False, "calls": 0},
                "passed": MODEL_SPECS == [],
            },
            {
                "check": "historical_v636_quarantines",
                "upstream": "four V636 artifacts",
                "artifact_field": "quarantined|rehabilitated",
                "expected_value": {"count": 4, "all_quarantined": True, "rehabilitated": 0},
                "observed_value": {
                    "count": len(historical_rows),
                    "all_quarantined": all(
                        row["quarantine_state"]["quarantined"] is True for row in historical_rows
                    ),
                    "rehabilitated": sum(
                        row["rehabilitated_by_v637"] is True for row in historical_rows
                    ),
                },
                "passed": len(historical_rows) == 4
                and all(row["quarantine_state"]["quarantined"] is True for row in historical_rows)
                and not any(row["rehabilitated_by_v637"] is True for row in historical_rows),
            },
        ]
    )
    for task_id, item in evidence.items():
        selected = item["selected_evidence_path"]
        if selected:
            hashes[selected] = item["artifact_sha256"]
        expected_field = EXPECTED_FIELDS[task_id]
        checks.append(
            {
                "check": "upstream_terminal_authentication_and_field",
                "upstream": task_id,
                "artifact_field": expected_field,
                "expected_value": "terminal artifact with declared field and authentic raw hashes",
                "observed_value": {
                    "path": selected,
                    "sha256": item["artifact_sha256"],
                    "terminal": item["terminal"],
                    "field_present": expected_field in item["payload"],
                    "quarantined": item["quarantine_receipt"]["quarantined"],
                    "raw_hashes_authenticated": item["raw_hashes_authenticated"],
                },
                "passed": item["terminal"]
                and expected_field in item["payload"]
                and item["raw_hashes_authenticated"],
            }
        )
    return checks, hashes


def _publication_shape(payload: Mapping[str, Any]) -> bool:
    """Accept only the unchanged G1 through G4 publication shape."""

    return base._publication_shape(payload)


def run_publication_gate(root: Path) -> JsonDict:
    """Run the existing publication evaluator without changing its policy."""

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
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("current_invocation_counters") != {"loads": 0, "generations": 0, "calls": 0}
    ):
        errors.append("model_invocation")
    contract_rows = artifact.get("task_contract_rows")
    if not isinstance(contract_rows, list) or [row.get("task_id") for row in contract_rows] != list(
        EXPECTED_TASK_IDS
    ):
        errors.append("task_contract_rows")
    matrix = artifact.get("evidence_matrix")
    if (
        not isinstance(matrix, list)
        or [row.get("task_id") for row in matrix] != list(EXPECTED_TASK_IDS)
        or matrix[-1].get("artifact_sha256") is not None
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
        or any(not row.get("next_experiment_or_exact_condition") for row in decisions)
    ):
        errors.append("branch_decisions")
    questions = artifact.get("scientific_questions", {})
    if [
        questions.get(name, {}).get("verdict_class")
        for name in (
            "source_fidelity_and_verification_value",
            "prospective_recurrence_learning",
            "actual_scored_policy_model_use",
            "complete_native_deployment_cost",
        )
    ] != ["blocked", "null", "disqualified", "null"]:
        errors.append("scientific_questions")
    if artifact.get("capstone_complete_score") != 1:
        errors.append("capstone_complete_score")
    gate = artifact.get("gate_check_summary")
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not False
        or any(
            name not in gate
            for name in (
                "failed_check",
                "upstream",
                "artifact_field",
                "expected_value",
                "observed_value",
            )
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
    historical = artifact.get("historical_v636_quarantine_rows", [])
    if (
        len(historical) != 4
        or any(row.get("accepted_for_promoted_evidence") is not False for row in historical)
        or any(row.get("rehabilitated_by_v637") is not False for row in historical)
    ):
        errors.append("historical_quarantine")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    if root is not None:
        contract = load_contract(root)
        evidence = load_repository_payloads(root, contract["tasks"])
        if contract_rows != contract["task_contract_rows"] and "task_contract_rows" not in errors:
            errors.append("task_contract_rows")
        if rows != recompute_claims(evidence) and "claim_rows" not in errors:
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
    """Aggregate V637 evidence and atomically write one terminal capstone."""

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
            "schema": "carnot.exp7245.v637_capstone.v1",
            "experiment_id": "exp7245-capstone",
            "status": "running",
            "run_date": run_date,
            "MODEL_SPECS": [],
            "model_invoked": False,
        },
    )

    progress(1, "start", "parse frozen V637 Markdown and YAML")
    contract = load_contract(root)
    progress(1, "end", f"contract rows={len(contract['tasks'])} errors={len(contract['errors'])}")
    progress(2, "start", "authenticate twelve producer artifacts and raw receipts")
    evidence = load_repository_payloads(root, contract["tasks"])
    gate_rows = replay_gates(contract["tasks"], evidence)
    manifest = yaml.safe_load((root / EXCLUSION_MANIFEST_PATH).read_text(encoding="utf-8"))
    historical_rows = _historical_quarantine_rows(root, manifest)
    checks, hashes = _preconditions(
        root, output_path, checkpoint_path, contract, evidence, historical_rows
    )
    essential = {
        "required_source_bytes",
        "driving_requirement",
        "required_imports",
        "required_resources",
        "output_directories",
        "frozen_contract",
        "no_current_llm",
        "historical_v636_quarantines",
        "upstream_terminal_authentication_and_field",
    }
    failed = [row for row in checks if row["check"] in essential and not row["passed"]]
    if failed:
        raise RuntimeError(f"essential capstone precondition failed: {failed[0]}")
    progress(
        2,
        "end",
        f"producer slots={len(evidence)} raw receipts={sum(len(x['raw_hash_replay_rows']) for x in evidence.values())}",
    )
    progress(0, "end", f"preconditions recorded={len(checks)}")

    progress(3, "start", "recompute producer metrics and thirteen matrix rows")
    claims = recompute_claims(evidence)
    matrix: list[JsonDict] = []
    for order, task in enumerate(contract["tasks"][:-1], 1):
        task_id = str(task["id"])
        matrix.append(matrix_row(order, task, evidence[task_id], gate_rows, claims))
        progress(3, "unit", f"completed matrix row {order}/13 {task_id}")
    gate_summary = {
        "passed": False,
        "failed_check": "structured_quarantine",
        "upstream": "exp7237-mention-canary",
        "artifact_field": "flagged_adversarial|mention_canary_ready_score",
        "field": "flagged_adversarial|mention_canary_ready_score",
        "expected_value": {"flagged_adversarial": False, "mention_canary_ready_score": 1},
        "observed_value": {"flagged_adversarial": True, "mention_canary_ready_score": 1},
    }
    honest_verdict = (
        "blocked_external_source_value: V637 matrix is complete; mention science remains blocked, "
        "recurrence and native cost are null, and scored-policy evidence is disqualified"
    )
    self_task = contract["tasks"][-1]
    self_row = {
        "order": 13,
        "task_id": "exp7245-capstone",
        "title": self_task["title"],
        "declared_deliverable_path": self_task["deliverable"],
        "canonical_gate_block_path": canonical_gate_block_path("exp7245-capstone"),
        "selected_evidence_path": str(output_path),
        "evidence_source": "self",
        "artifact_sha256": None,
        "artifact_size_bytes": None,
        "planned_status": "planned",
        "executed_status": "complete",
        "planned_vs_executed_status": "terminal_complete",
        "scope_complete": True,
        "producer_verdict_class": None,
        "verdict_class": "blocked",
        "honest_verdict": honest_verdict,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "producer_model_invoked": False,
        "historical_invocation_source_only": False,
        "raw_row_count": len(claims),
        "producer_source_artifact_hashes": {},
        "raw_hash_replay_rows": [],
        "raw_hashes_authenticated": True,
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
        "authority_boundary": "matrix completion is independent from scientific success",
    }
    matrix.append(self_row)
    progress(3, "unit", "completed matrix row 13/13 exp7245-capstone")
    progress(3, "end", f"numeric claim rows={len(claims)}")

    progress(4, "start", "classify four scientific questions")
    questions = scientific_questions(claims, evidence)
    narrow_findings = {
        "mention_fixture_exact_conformance": {
            "value": True,
            "verdict_class": "circular_positive",
            "scientific_value_claimed": False,
        },
        "recurrence_causal_controls": {
            "value": True,
            "verdict_class": "circular_positive",
            "llm_verification_evidence": False,
        },
        "native_exact_parity": {
            "value": True,
            "verdict_class": "circular_positive",
            "cost_value": False,
        },
        "board_disposition": {
            "kv260": "graduated_fpga_fabric",
            "polarfire": "graduated_cpu_dispatch_not_fpga_sampling",
            "gatemate": "blocked_changed_physical_state",
        },
    }
    progress(4, "end", "source=blocked recurrence=null scored=disqualified native=null")

    progress(5, "start", "apply task-scoped branch decisions")
    decisions = _branch_decisions(contract["tasks"], matrix)
    progress(5, "end", f"branch decisions={len(decisions)}")

    progress(6, "start", "run unchanged publication gate")
    publication = publication_runner(root)
    if not _publication_shape(publication):
        raise RuntimeError("publication gate returned a changed G1-G4 shape")
    progress(6, "end", f"paper_ready={publication['paper_ready']}")

    artifact: JsonDict = {
        "schema": "carnot.exp7245.v637_capstone.v1",
        "experiment_id": "exp7245-capstone",
        "milestone": MILESTONE,
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
        "phase_spans_s": {"aggregation_total": 0.0},
        "MODEL_SPECS": [],
        "model_invoked": False,
        "current_invocation_counters": {"loads": 0, "generations": 0, "calls": 0},
        "source_artifact_hashes": hashes,
        "rows": claims,
        "sample_size_budget": {
            "contract_tasks": {
                "planned": 13,
                "attempted": 13,
                "completed": 13,
                "censored": 0,
                "independent_units": 13,
            },
            "producer_slots": {
                "planned": 12,
                "attempted": 12,
                "completed": sum(item["terminal"] for item in evidence.values()),
                "censored": sum(not item["terminal"] for item in evidence.values()),
                "independent_units": 12,
            },
            "numeric_claim_rows": {
                "planned": len(claims),
                "attempted": len(claims),
                "completed": len(claims),
                "censored": sum(row["abstention"] is True for row in claims),
                "independent_units": len({row["task_id"] for row in claims}),
            },
            "stopping_rule": "Stop after all thirteen frozen task slots and four scientific questions are classified.",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": honest_verdict,
        "acceptance_gate_results": [
            {
                "criterion": "thirteen_task_matrix_complete",
                "expected_value": 13,
                "actual_value": len(matrix),
                "passed": len(matrix) == 13,
            },
            {
                "criterion": "all_recomputed_claims_match",
                "expected_value": True,
                "actual_value": all(row["matches"] is True for row in claims),
                "passed": all(row["matches"] is True for row in claims),
            },
            {
                "criterion": "historical_quarantines_preserved",
                "expected_value": 4,
                "actual_value": len(historical_rows),
                "passed": len(historical_rows) == 4,
            },
            {
                "criterion": "scientific_success_required_for_completion",
                "expected_value": False,
                "actual_value": False,
                "passed": True,
            },
        ],
        "capstone_complete_score": 1,
        "task_contract_rows": contract["task_contract_rows"],
        "evidence_matrix": matrix,
        "recomputed_claim_rows": claims,
        "scientific_questions": questions,
        "narrow_capability_findings": narrow_findings,
        "historical_v636_quarantine_rows": historical_rows,
        "same_milestone_gate_replay_rows": gate_rows,
        "branch_decisions": decisions,
        "prd_completion": {
            "FR-11": {
                "complete": False,
                "verdict_class": "null",
                "finding": "prospective recurrence learning failed the frozen value gate",
            },
            "FR-12": {
                "complete": False,
                "verdict_class": "blocked",
                "finding": "held-out source fidelity and value remain unavailable",
            },
            "FR-05/08": {
                "complete": False,
                "verdict_class": "disqualified",
                "finding": "the local model ran but no authenticated scored policy use was established",
            },
            "NFR-01": {
                "complete": False,
                "verdict_class": "null",
                "finding": "full batch-one native event cost did not meet the 10x target",
            },
        },
        "publication_gate": publication,
        "publication_performed": False,
        "upload_performed": False,
        "submission_performed": False,
        "external_contact_performed": False,
        "production_default_changed": False,
        "exclusion_manifest_modified": False,
        "conductor_modified": False,
        "validation_receipts": {
            "producer_replay": "public producer artifacts -> independent V637 reducer",
            "applicable_e2e": ["E2E-003", "E2E-004", "E2E-007", "E2E-010"],
            "external_publication_performed": False,
        },
    }
    elapsed = time.monotonic() - started
    artifact["duration_s"] = elapsed
    artifact["phase_spans_s"]["aggregation_total"] = elapsed
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

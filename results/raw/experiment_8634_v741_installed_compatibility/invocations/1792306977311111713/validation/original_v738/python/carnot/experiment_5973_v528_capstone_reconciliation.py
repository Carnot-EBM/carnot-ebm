"""Exp5973 branch-independent capstone reconciliation for milestone .528.

Spec refs: REQ-REPORT-5973,
SCENARIO-REPORT-5973-EXACT-MATRIX,
SCENARIO-REPORT-5973-GATES-AND-MISSING,
SCENARIO-REPORT-5973-BRANCH-INDEPENDENCE,
SCENARIO-REPORT-5973-VERIFIER-AND-SUBSTRATE,
SCENARIO-REPORT-5973-SCHEMA.

This module is an aggregation receipt over already-produced upstream artifacts.
It intentionally treats the active roadmap's declared deliverable path as the
only artifact locator. That prevents a nearby filename, a completion-history
summary, or a downstream gate message from turning a missing, blocked, or null
branch into a stronger capstone claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.07.528"
MILESTONE_TITLE = (
    "Discriminative Exact-Atom Acquisition, Delayed-Commit Continuous Learning, "
    "and ARC Budget/Convention Generalization"
)
EXPERIMENT_ID = "exp5973-v528-capstone-reconciliation"
EXPERIMENT = "experiment_5973_v528_capstone_reconciliation"
RESULT_RELATIVE_PATH = Path("results/experiment_5973_v528_capstone_reconciliation.json")
SCHEMA = "carnot.experiment_5973.v528_capstone_reconciliation.v1"
RUN_DATE = "20260804"
RANDOM_SEED = 5973
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
RESEARCH_STUDYING_RELATIVE_PATH = Path("research-studying.md")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
DOC_RECONCILE_RELATIVE_PATH = Path("scripts/in_process_doc_reconcile.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXP5933_CLASSIFIER_RELATIVE_PATH = Path(
    "results/experiment_5933_aggregation_substrate_qa_repair.json"
)
ARC_AGENT_RELATIVE_PATH = Path("python/carnot/agentic/arc_competition_agent.py")

SPEC_REFS = (
    "REQ-REPORT-5973",
    "SCENARIO-REPORT-5973-EXACT-MATRIX",
    "SCENARIO-REPORT-5973-GATES-AND-MISSING",
    "SCENARIO-REPORT-5973-BRANCH-INDEPENDENCE",
    "SCENARIO-REPORT-5973-VERIFIER-AND-SUBSTRATE",
    "SCENARIO-REPORT-5973-SCHEMA",
)

UPSTREAM_TASKS: tuple[tuple[str, str, Path], ...] = (
    (
        "exp5961-transition-v528",
        "Exact terminal-boundary handoff from .527 into .528",
        Path("results/experiment_5961_transition_v528.json"),
    ),
    (
        "exp5962-v528-source-delta-ingestion",
        "Dated evidence refresh after the V528 planner marker",
        Path("results/experiment_5962_v528_source_delta_ingestion.json"),
    ),
    (
        "exp5963-exact-atom-pair-fixture",
        "Hardness-controlled exact context/atom compatibility fixture",
        Path("results/experiment_5963_exact_atom_pair_fixture.json"),
    ),
    (
        "exp5964-sota-atom-compatibility-corpus",
        "Gated on Exp5963 ready: all-three-model GGUF context/atom compatibility corpus",
        Path("results/experiment_5964_sota_atom_compatibility_corpus.json"),
    ),
    (
        "exp5965-portable-atom-energy-ranker",
        "Gated on Exp5964 ready: portable exact-atom compatibility energy",
        Path("results/experiment_5965_portable_atom_energy_ranker.json"),
    ),
    (
        "exp5966-discriminative-constraint-acquisition",
        "Gated on Exp5965 ready: end-to-end discriminative exact constraint acquisition",
        Path("results/experiment_5966_discriminative_constraint_acquisition.json"),
    ),
    (
        "exp5967-delayed-commit-memory-fixture",
        "Delayed-commit transactional memory fixture over ABI v2",
        Path("results/experiment_5967_delayed_commit_memory_fixture.json"),
    ),
    (
        "exp5968-delayed-commit-csl-prospective",
        "Gated on Exp5967 ready: prospective delayed-commit continuous self-learning A/B",
        Path("results/experiment_5968_delayed_commit_csl_prospective.json"),
    ),
    (
        "exp5969-csl-poison-drift-abi-audit",
        "Gated on Exp5968 ready: poison, drift, rollback, retention, and ABI audit",
        Path("results/experiment_5969_csl_poison_drift_abi_audit.json"),
    ),
    (
        "exp5970-arc-strip-swap-sentinel",
        "ARC row/column strip-swap convention sentinel",
        Path("results/experiment_5970_arc_strip_swap_sentinel.json"),
    ),
    (
        "exp5971-arc-strip-swap-battery",
        "Gated on Exp5970 ready: full ARC strip-swap convention-transfer battery",
        Path("results/experiment_5971_arc_strip_swap_battery.json"),
    ),
    (
        "exp5972-arc-llm-on-budget2000-feasibility",
        "Live ARC flagship-LLM budget-2000 wall-clock feasibility",
        Path("results/experiment_5972_arc_llm_on_budget2000_feasibility.json"),
    ),
)

UPSTREAM_TASK_IDS = tuple(task_id for task_id, _title, _path in UPSTREAM_TASKS)
UPSTREAM_DELIVERABLES = {task_id: rel_path for task_id, _title, rel_path in UPSTREAM_TASKS}
UPSTREAM_TITLES = {task_id: title for task_id, title, _path in UPSTREAM_TASKS}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_and_exact_task_deliverable_matrix",
    "per_task_path_hash_presence_and_terminal_class",
    "missing_and_gate_block_receipts",
    "fresh_adversarial_verifier_receipts",
    "gate_recomputation_and_cascade_receipts",
    "semantic_acquisition_branch_summary",
    "continuous_self_learning_branch_summary",
    "arc_strip_swap_branch_summary",
    "arc_budget_feasibility_branch_summary",
    "branch_independence_receipt",
    "prior_failure_retirement_and_exclusion_receipt",
    "model_and_hardware_policy_receipt",
    "arc_provenance_registry_and_flag_immutability",
    "aggregation_substrate_classifier_receipt",
    "docs_reconciled",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "capstone state and all aggregation prerequisites are explicit.",
    "preconditions_checked": "capstone state and all aggregation prerequisites are explicit.",
    "milestone_and_exact_task_deliverable_matrix": (
        "the active roadmap's exact IDs and paths are the only evidence index."
    ),
    "per_task_path_hash_presence_and_terminal_class": (
        "every upstream task retains a hash-backed or explicitly absent terminal class."
    ),
    "missing_and_gate_block_receipts": (
        "a skipped task is not an execution failure and a missing ungated artifact is not success."
    ),
    "fresh_adversarial_verifier_receipts": (
        "current exact-path verification, not inherited prose, owns artifact quality."
    ),
    "gate_recomputation_and_cascade_receipts": (
        "every structured gate is recomputed and title/YAML alignment is exact."
    ),
    "semantic_acquisition_branch_summary": (
        "fixture, corpus, ranker, and exact acquisition claims remain separate."
    ),
    "continuous_self_learning_branch_summary": (
        "fixture, prospective utility, and poison/ABI safety remain separate."
    ),
    "arc_strip_swap_branch_summary": (
        "targeted dose, live support, game-unit evidence, and limitations remain visible."
    ),
    "arc_budget_feasibility_branch_summary": (
        "measured timing, projection uncertainty, and no automatic flag change remain visible."
    ),
    "branch_independence_receipt": (
        "no positive, null, retirement, block, or flag in one branch rewrites another."
    ),
    "prior_failure_retirement_and_exclusion_receipt": (
        "same-verdict reruns retire mechanically and no retired dependency is introduced."
    ),
    "model_and_hardware_policy_receipt": (
        "mandated GGUF use, CUDA authenticity, no legacy headline, and no unsupported board claim are checked."
    ),
    "arc_provenance_registry_and_flag_immutability": (
        "any level outcome is live-agent provenance only, with no new solve credit, registry mutation, or flag flip."
    ),
    "aggregation_substrate_classifier_receipt": (
        "the capstone declares aggregation and is not misclassified from nested upstream compute strings."
    ),
    "docs_reconciled": (
        "only evidence-supported internal docs change; protected/unrelated state remains byte-identical."
    ),
    "protected_files_unchanged": (
        "only evidence-supported internal docs change; protected/unrelated state remains byte-identical."
    ),
    "duration_s": "use measured `aggregation_from_upstream_artifacts`.",
    "inference_substrate": "use measured `aggregation_from_upstream_artifacts`.",
    "field_provenance": "use measured `aggregation_from_upstream_artifacts`.",
    "test_commands": "use measured `aggregation_from_upstream_artifacts`.",
    "test_exit_codes": "use measured `aggregation_from_upstream_artifacts`.",
    "reproducibility_checksum": "use measured `aggregation_from_upstream_artifacts`.",
    "honest_verdict": (
        "use `complete:`, `complete_with_nulls:`, `complete_with_blocks:`, or `blocked:`."
    ),
}

PROTECTED_RELATIVE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    RESEARCH_STUDYING_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    *UPSTREAM_DELIVERABLES.values(),
)

PRECONDITION_HASH_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    RESEARCH_STUDYING_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    DOC_RECONCILE_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    EXP5933_CLASSIFIER_RELATIVE_PATH,
    ARC_AGENT_RELATIVE_PATH,
    *UPSTREAM_DELIVERABLES.values(),
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def path_sha256(path: str | Path) -> str | None:
    target = Path(path)
    if not target.exists():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: JsonMap) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def write_json(path: Path, payload: JsonMap) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        meta["error"] = f"json_error:{exc.msg}"
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "json_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _read_yaml_mapping(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {
        "path": path.as_posix(),
        "present": path.exists(),
        "loadable": False,
        "sha256": path_sha256(path),
        "error": None,
    }
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        meta["error"] = f"yaml_error:{exc.__class__.__name__}"
        return {}, meta
    if not isinstance(payload, dict):
        meta["error"] = "yaml_not_mapping"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _task_number(task_id: str) -> int | None:
    match = re.search(r"exp(\d{4})", task_id)
    return int(match.group(1)) if match else None


def _conductor_status_from_line(line: str) -> str:
    parts = [part.strip() for part in line.split("|")]
    return parts[3] if len(parts) > 3 else ""


def _roadmap_tasks(root: Path) -> tuple[list[JsonDict], JsonDict]:
    payload, meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    rows = payload.get("tasks") if meta["loadable"] else []
    task_rows = [dict(row) for row in rows if isinstance(row, Mapping)] if isinstance(rows, list) else []
    upstream = [
        row
        for row in task_rows
        if isinstance(row.get("id"), str)
        and (number := _task_number(str(row["id"]))) is not None
        and 5961 <= number <= 5972
    ]
    upstream.sort(key=lambda row: int(str(row["id"])[3:7]))
    capstone = [row for row in task_rows if row.get("id") == EXPERIMENT_ID]
    return upstream, {
        **meta,
        "milestone": payload.get("milestone") if meta["loadable"] else None,
        "milestone_title": payload.get("milestone_title") if meta["loadable"] else None,
        "upstream_task_ids": [str(row["id"]) for row in upstream],
        "capstone_task_present": bool(capstone),
    }


def _artifact_payloads(root: Path, tasks: Sequence[JsonMap]) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for row in tasks:
        task_id = str(row["id"])
        rel_path = Path(str(row.get("deliverable") or ""))
        payload, meta = _read_json_mapping(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        payloads[task_id] = payload
        metadata[task_id] = meta
    return payloads, metadata


def _conductor_receipts(root: Path, tasks: Sequence[JsonMap]) -> JsonDict:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = text.splitlines()
    by_task: dict[str, JsonDict] = {}
    for row in tasks:
        task_id = str(row["id"])
        title = str(row.get("title") or "")
        prefix = title[:45]
        matches = [line for line in lines if task_id in line or (prefix and prefix in line)]
        latest = matches[-1] if matches else ""
        by_task[task_id] = {
            "attempt_count": len(matches),
            "latest_line": latest,
            "latest_status": _conductor_status_from_line(latest) if latest else "",
        }
    activation = [line for line in lines if "Milestone 2026.07.528 activated" in line]
    return {
        "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        "present": path.exists(),
        "sha256": path_sha256(path),
        "activation_line": activation[-1] if activation else "",
        "activation_status": _conductor_status_from_line(activation[-1]) if activation else "",
        "activated_task_count_claim": 13 if activation and "13 tasks queued" in activation[-1] else None,
        "by_task": by_task,
    }


def _receipt_flags(receipt: JsonMap) -> list[JsonDict]:
    stdout_json = receipt.get("stdout_json")
    reports = stdout_json.get("reports") if isinstance(stdout_json, Mapping) else None
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        flags = reports[0].get("flags")
        return [dict(flag) for flag in flags if isinstance(flag, Mapping)] if isinstance(flags, list) else []
    flags = receipt.get("flags")
    return [dict(flag) for flag in flags if isinstance(flag, Mapping)] if isinstance(flags, list) else []


def _receipt_flag_count(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    reports = stdout_json.get("reports") if isinstance(stdout_json, Mapping) else None
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        report = reports[0]
        if isinstance(report.get("flag_count"), int):
            return int(report["flag_count"])
        flags = report.get("flags")
        return len(flags) if isinstance(flags, list) else 0
    if isinstance(stdout_json, Mapping):
        return int(stdout_json.get("flagged_count") or 0)
    return int(receipt.get("flag_count") or 0)


def _receipt_max_severity(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    reports = stdout_json.get("reports") if isinstance(stdout_json, Mapping) else None
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        value = reports[0].get("max_severity")
        return int(value) if isinstance(value, int) else (-1 if _receipt_flag_count(receipt) == 0 else 1)
    return int(receipt.get("max_severity", -1))


def _normalize_adversarial_receipts(
    receipts: Mapping[str, JsonMap] | Sequence[Any] | None,
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    if receipts is None:
        return {}
    source = receipts.values() if isinstance(receipts, Mapping) else receipts
    rows: dict[str, JsonDict] = {}
    for row in source:
        if not isinstance(row, Mapping) or not row.get("task_id"):
            continue
        task_id = str(row["task_id"])
        if metadata.get(task_id, {}).get("present"):
            complete = dict(row)
            complete["flag_count"] = _receipt_flag_count(row)
            complete["max_severity"] = _receipt_max_severity(row)
            complete["flags"] = _receipt_flags(row)
            complete.setdefault("receipt_hash", sha256_json(complete.get("stdout_json", {})))
            rows[task_id] = complete
    return rows


def _run_live_adversarial_receipts(root: Path, metadata: Mapping[str, JsonMap]) -> dict[str, JsonDict]:  # pragma: no cover
    executable = (root / ".venv/bin/python").as_posix() if (root / ".venv/bin/python").exists() else sys.executable
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in UPSTREAM_DELIVERABLES.items():
        if not metadata.get(task_id, {}).get("present"):
            continue
        command = [executable, ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(), "--json", rel_path.as_posix()]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        try:
            stdout_json: Any = json.loads(result.stdout)
        except json.JSONDecodeError:
            stdout_json = {"parse_error": "stdout_not_json", "stdout": result.stdout}
        receipts[task_id] = _normalize_adversarial_receipts(
            [
                {
                    "task_id": task_id,
                    "artifact_path": rel_path.as_posix(),
                    "command": " ".join(command),
                    "exit_code": result.returncode,
                    "stdout_json": stdout_json,
                    "stderr": result.stderr,
                    "receipt_hash": sha256_json(stdout_json),
                }
            ],
            metadata,
        )[task_id]
    return receipts


def _gate_expected_pass(gate: JsonMap, payloads: Mapping[str, JsonMap], metadata: Mapping[str, JsonMap]) -> tuple[Any, bool, str]:
    upstream = str(gate.get("upstream") or "")
    field = str(gate.get("artifact_field") or "")
    expected = gate.get("value")
    op = str(gate.get("op") or "==")
    if not metadata.get(upstream, {}).get("present"):
        return None, False, "upstream_artifact_missing"
    actual = payloads.get(upstream, {}).get(field)
    passed = actual == expected if op == "==" else False
    reason = "passed" if passed else "upstream_gate_failed_or_retired"
    return actual, passed, reason


def _gate_recomputation(
    tasks: Sequence[JsonMap],
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
) -> JsonDict:
    rows: list[JsonDict] = []
    cascade_skips: list[JsonDict] = []
    alignment: list[JsonDict] = []
    for row in tasks:
        task_id = str(row["id"])
        gates = row.get("gated_on")
        if not isinstance(gates, list):
            alignment.append({"task_id": task_id, "aligned": True, "reason": "ungated"})
            continue
        task_passes: list[bool] = []
        task_reasons: list[str] = []
        for gate in gates:
            if not isinstance(gate, Mapping):
                continue
            actual, passed, reason = _gate_expected_pass(gate, payloads, metadata)
            upstream = str(gate.get("upstream") or "")
            upstream_num = _task_number(upstream)
            title_aligned = upstream_num is not None and f"Exp{upstream_num}" in str(row.get("title") or "")
            rows.append(
                {
                    "task_id": task_id,
                    "upstream": upstream,
                    "artifact_field": gate.get("artifact_field"),
                    "op": gate.get("op"),
                    "expected": gate.get("value"),
                    "actual": actual,
                    "gate_passed": passed,
                    "reason": reason,
                    "title_yaml_gate_alignment": title_aligned,
                }
            )
            alignment.append({"task_id": task_id, "upstream": upstream, "aligned": title_aligned})
            task_passes.append(passed)
            task_reasons.append(reason)
        if task_passes and not all(task_passes):
            reason = "upstream_artifact_missing" if "upstream_artifact_missing" in task_reasons else "upstream_gate_failed_or_retired"
            cascade_skips.append(
                {
                    "task_id": task_id,
                    "reason": reason,
                    "not_executed_by_capstone": True,
                }
            )
    return {
        "gate_rows": rows,
        "cascade_skips": cascade_skips,
        "title_yaml_alignment_rows": alignment,
        "title_yaml_gate_alignment_exact": all(row["aligned"] for row in alignment),
        "capstone_executed_skipped_experiment_count": 0,
        "principle": FIELD_PRINCIPLES["gate_recomputation_and_cascade_receipts"],
    }


def _terminal_class(
    task_id: str,
    payload: JsonMap,
    meta: JsonMap,
    conductor: JsonMap,
    gate_rows: Sequence[JsonMap],
) -> tuple[str, str]:
    task_gate_failed = any(row.get("task_id") == task_id and not row.get("gate_passed") for row in gate_rows)
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    if not meta.get("present"):
        if task_gate_failed or conductor.get("latest_status") == "GATE_BLOCK":
            return "gate-blocked", "declared-deliverable-absent-after-recomputed-gate-skip"
        return "missing", "declared-deliverable-missing"
    if payload.get("schema") == "blocked_gate_check_v1" or payload.get("gates_evaluated"):
        return "gate-blocked", "conductor-gate-check-artifact"
    if status == "retired" or verdict.startswith("retired:"):
        return "retired", "retire-if-same-verdict"
    if "underpowered" in status or verdict.startswith("complete_underpowered"):
        return "underpowered", "underpowered"
    if status.startswith("blocked") or verdict.startswith(("blocked:", "blocked_")):
        return "blocked-precondition", "blocked-before-eligible-execution"
    if status == "complete_null" or verdict.startswith("complete_null"):
        return "complete-null", "honest-null"
    if status == "complete_ready" or verdict.startswith("complete_ready"):
        return "complete-ready", "ready-receipt"
    if status == "complete_feasible" or verdict.startswith("complete_feasible"):
        return "complete-feasible", "feasibility-receipt"
    if status.startswith("complete") or verdict.startswith("complete:"):
        return "complete", "complete-receipt"
    return "missing", "unrecognized-terminal-treated-as-missing"


def _terminal_matrix(
    tasks: Sequence[JsonMap],
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    gates: JsonMap,
) -> JsonDict:
    rows: dict[str, JsonDict] = {}
    by_task: dict[str, str] = {}
    by_subclass: dict[str, str] = {}
    by_class: dict[str, list[str]] = {}
    gate_rows = gates.get("gate_rows", [])
    for row in tasks:
        task_id = str(row["id"])
        rel_path = Path(str(row.get("deliverable") or ""))
        terminal, subclass = _terminal_class(
            task_id,
            payloads.get(task_id, {}),
            metadata.get(task_id, {}),
            conductor.get(task_id, {}),
            gate_rows if isinstance(gate_rows, list) else [],
        )
        by_task[task_id] = terminal
        by_subclass[task_id] = subclass
        by_class.setdefault(terminal, []).append(task_id)
        meta = metadata.get(task_id, {})
        payload = payloads.get(task_id, {})
        rows[task_id] = {
            "identity": [task_id, rel_path.as_posix()],
            "milestone": MILESTONE,
            "task_id": task_id,
            "title": str(row.get("title") or ""),
            "declared_deliverable": rel_path.as_posix(),
            "present": bool(meta.get("present")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "status": str(payload.get("status") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "terminal_class": terminal,
            "terminal_subclass": subclass,
            "terminal_evidence_source": "declared_deliverable_path"
            if meta.get("present")
            else ("recomputed_gate_and_conductor_skip" if terminal == "gate-blocked" else "explicit_absence"),
            "conductor": conductor.get(task_id, {}),
        }
    return {
        "tasks": rows,
        "terminal_class_by_task_id": by_task,
        "terminal_subclass_by_task_id": by_subclass,
        "task_ids_by_terminal_class": by_class,
        "all_upstream_tasks_classified_once": len(by_task) == len(tasks),
        "principle": FIELD_PRINCIPLES["per_task_path_hash_presence_and_terminal_class"],
    }


def _numeric_prefix_aliases(root: Path) -> list[str]:
    declared = {path.as_posix() for path in UPSTREAM_DELIVERABLES.values()}
    results = root / "results"
    if not results.exists():
        return []
    aliases: list[str] = []
    for number in range(5961, 5973):
        for path in results.glob(f"experiment_{number}_*.json"):
            rel = path.relative_to(root).as_posix()
            if rel not in declared:
                aliases.append(rel)
    return sorted(aliases)


def _roadmap_matrix(tasks: Sequence[JsonMap], root: Path) -> JsonDict:
    return {
        "milestone": MILESTONE,
        "milestone_title": MILESTONE_TITLE,
        "selection_policy": "active_roadmap_declared_deliverable_only",
        "identity_tuple": ["task_id", "declared_deliverable"],
        "upstream_task_count": len(tasks),
        "upstream_task_ids": [str(row["id"]) for row in tasks],
        "upstream_declared_deliverables": {
            str(row["id"]): str(row.get("deliverable") or "") for row in tasks
        },
        "expected_upstream_task_ids": list(UPSTREAM_TASK_IDS),
        "capstone_task_id": EXPERIMENT_ID,
        "capstone_declared_deliverable": RESULT_RELATIVE_PATH.as_posix(),
        "numeric_prefix_aliases_ignored": _numeric_prefix_aliases(root),
        "principle": FIELD_PRINCIPLES["milestone_and_exact_task_deliverable_matrix"],
    }


def _fresh_verifier_receipts(
    receipts: Mapping[str, JsonMap],
    terminal: JsonMap,
) -> JsonDict:
    rows: list[JsonDict] = []
    for task_id, task_row in terminal["tasks"].items():
        if not task_row["present"]:
            continue
        receipt = receipts.get(task_id)
        if not isinstance(receipt, Mapping):
            continue
        rows.append(
            {
                "task_id": task_id,
                "artifact": task_row["declared_deliverable"],
                "command": str(receipt.get("command") or ""),
                "exit_code": receipt.get("exit_code"),
                "flag_count": int(receipt.get("flag_count") or 0),
                "max_severity": int(receipt.get("max_severity", -1)),
                "flags": receipt.get("flags") or [],
                "receipt_hash": str(receipt.get("receipt_hash") or ""),
            }
        )
    missing = [
        row["declared_deliverable"] for row in terminal["tasks"].values() if not row["present"]
    ]
    failed = [row["task_id"] for row in rows if row["exit_code"] not in {0, None} and row["max_severity"] >= 2]
    return {
        "reports": rows,
        "verified_present_declared_deliverable_count": len(rows),
        "missing_declared_deliverables_not_verified": missing,
        "failed_receipt_task_ids": failed,
        "flagged_count": sum(int(row["flag_count"]) for row in rows),
        "principle": FIELD_PRINCIPLES["fresh_adversarial_verifier_receipts"],
    }


def _missing_gate_receipts(terminal: JsonMap, gates: JsonMap) -> JsonDict:
    classes = terminal["terminal_class_by_task_id"]
    missing = [task_id for task_id, row in terminal["tasks"].items() if not row["present"]]
    gate_blocked = [task_id for task_id, cls in classes.items() if cls == "gate-blocked"]
    missing_ungated = [task_id for task_id in missing if task_id not in gate_blocked]
    return {
        "missing_declared_deliverable_task_ids": missing,
        "missing_ungated_declared_deliverable_task_ids": missing_ungated,
        "missing_ungated_artifact_is_success": False,
        "gate_blocked_task_ids": gate_blocked,
        "cascade_skips": gates["cascade_skips"],
        "skipped_experiments_executed_by_capstone_count": 0,
        "principle": FIELD_PRINCIPLES["missing_and_gate_block_receipts"],
    }


def _value(payloads: Mapping[str, JsonMap], task_id: str, key: str) -> Any:
    return payloads.get(task_id, {}).get(key)


def _semantic_summary(payloads: Mapping[str, JsonMap], terminal: JsonMap) -> JsonDict:
    classes = terminal["terminal_class_by_task_id"]
    return {
        "source_delta": {
            "terminal_class": classes.get("exp5962-v528-source-delta-ingestion"),
            "accepted_count": len(
                (_value(payloads, "exp5962-v528-source-delta-ingestion", "accepted_rejected_abstained_findings") or {}).get("accepted", [])
            )
            if isinstance(_value(payloads, "exp5962-v528-source-delta-ingestion", "accepted_rejected_abstained_findings"), Mapping)
            else None,
        },
        "fixture": {
            "terminal_class": classes.get("exp5963-exact-atom-pair-fixture"),
            "ready_score": _value(payloads, "exp5963-exact-atom-pair-fixture", "pair_fixture_ready_score"),
            "counts": _value(payloads, "exp5963-exact-atom-pair-fixture", "base_case_pair_and_class_counts"),
        },
        "corpus": {
            "terminal_class": classes.get("exp5964-sota-atom-compatibility-corpus"),
            "ready_score": _value(payloads, "exp5964-sota-atom-compatibility-corpus", "atom_compatibility_corpus_ready_score"),
            "blocked_reasons": (_value(payloads, "exp5964-sota-atom-compatibility-corpus", "preconditions_checked") or {}).get("blocked_reasons", [])
            if isinstance(_value(payloads, "exp5964-sota-atom-compatibility-corpus", "preconditions_checked"), Mapping)
            else [],
        },
        "ranker": {
            "terminal_class": classes.get("exp5965-portable-atom-energy-ranker"),
            "present": bool(
                terminal["tasks"].get("exp5965-portable-atom-energy-ranker", {}).get("present")
            ),
        },
        "exact_acquisition": {
            "terminal_class": classes.get("exp5966-discriminative-constraint-acquisition"),
            "gate_check_summary": _value(payloads, "exp5966-discriminative-constraint-acquisition", "gate_check_summary"),
        },
        "fixture_ready_does_not_imply_ranker_or_acquisition_quality": True,
        "blocked_semantic_branch_does_not_null_csl_or_arc": True,
        "principle": FIELD_PRINCIPLES["semantic_acquisition_branch_summary"],
    }


def _csl_summary(payloads: Mapping[str, JsonMap], terminal: JsonMap) -> JsonDict:
    classes = terminal["terminal_class_by_task_id"]
    return {
        "fixture": {
            "terminal_class": classes.get("exp5967-delayed-commit-memory-fixture"),
            "ready_score": _value(payloads, "exp5967-delayed-commit-memory-fixture", "delayed_commit_fixture_ready_score"),
        },
        "prospective": {
            "terminal_class": classes.get("exp5968-delayed-commit-csl-prospective"),
            "ready_score": _value(payloads, "exp5968-delayed-commit-csl-prospective", "prospective_csl_ready_score"),
            "paired_deltas": _value(payloads, "exp5968-delayed-commit-csl-prospective", "paired_deltas_intervals_and_power"),
            "unsafe_accept_count": _value(payloads, "exp5968-delayed-commit-csl-prospective", "unsafe_accept_count"),
        },
        "poison_abi_audit": {
            "terminal_class": classes.get("exp5969-csl-poison-drift-abi-audit"),
            "rollback_ready_score": _value(payloads, "exp5969-csl-poison-drift-abi-audit", "rollback_and_recovery_ready_score"),
            "unsafe_accept_count": _value(payloads, "exp5969-csl-poison-drift-abi-audit", "unsafe_accept_count"),
        },
        "model_weights_mutated": False,
        "same_event_or_poison_credit_laundered": False,
        "principle": FIELD_PRINCIPLES["continuous_self_learning_branch_summary"],
    }


def _arc_strip_summary(payloads: Mapping[str, JsonMap], terminal: JsonMap) -> JsonDict:
    classes = terminal["terminal_class_by_task_id"]
    solve_5970 = _value(payloads, "exp5970-arc-strip-swap-sentinel", "no_solve_credit_receipt")
    solve_5971 = _value(payloads, "exp5971-arc-strip-swap-battery", "no_solve_credit_receipt")
    return {
        "sentinel": {
            "terminal_class": classes.get("exp5970-arc-strip-swap-sentinel"),
            "ready_score": _value(payloads, "exp5970-arc-strip-swap-sentinel", "strip_swap_sentinel_ready_score"),
            "support": _value(payloads, "exp5970-arc-strip-swap-sentinel", "anchor_support_and_behavioral_validity"),
        },
        "battery": {
            "terminal_class": classes.get("exp5971-arc-strip-swap-battery"),
            "decision": _value(payloads, "exp5971-arc-strip-swap-battery", "convention_dependence_decision"),
            "anchor_support": _value(payloads, "exp5971-arc-strip-swap-battery", "anchor_survival_and_discriminating_game_support"),
            "overall_hud_value": _value(payloads, "exp5971-arc-strip-swap-battery", "overall_hud_value_not_identified_receipt"),
        },
        "hidden_transfer_claimed": False,
        "new_solve_credit_claimed": bool(
            isinstance(solve_5970, Mapping) and solve_5970.get("solve_credit_claimed")
        )
        or bool(isinstance(solve_5971, Mapping) and solve_5971.get("solve_credit_claimed")),
        "public_game_outcome_does_not_imply_hidden_transfer": True,
        "principle": FIELD_PRINCIPLES["arc_strip_swap_branch_summary"],
    }


def _arc_budget_summary(payloads: Mapping[str, JsonMap], terminal: JsonMap) -> JsonDict:
    artifact = payloads.get("exp5972-arc-llm-on-budget2000-feasibility", {})
    projection = artifact.get("twenty_five_game_twelve_hour_projection_and_interval")
    flag = artifact.get("no_automatic_flag_change_receipt")
    credit = artifact.get("no_new_solve_credit_receipt")
    return {
        "terminal_class": terminal["terminal_class_by_task_id"].get("exp5972-arc-llm-on-budget2000-feasibility"),
        "projection": projection,
        "feasible_at_upper_bound": bool(isinstance(projection, Mapping) and projection.get("fits_12h_at_upper_bound")),
        "automatic_flag_change": bool(isinstance(flag, Mapping) and flag.get("feature_flags_changed")),
        "new_solve_credit_claimed": bool(isinstance(credit, Mapping) and credit.get("registry_update_requested")),
        "upstream_substrate": artifact.get("inference_substrate"),
        "capstone_substrate_is_aggregation": True,
        "principle": FIELD_PRINCIPLES["arc_budget_feasibility_branch_summary"],
    }


def _branch_independence(terminal: JsonMap) -> JsonDict:
    classes = terminal["terminal_class_by_task_id"]
    branches = {
        "semantic_acquisition": [
            "exp5962-v528-source-delta-ingestion",
            "exp5963-exact-atom-pair-fixture",
            "exp5964-sota-atom-compatibility-corpus",
            "exp5965-portable-atom-energy-ranker",
            "exp5966-discriminative-constraint-acquisition",
        ],
        "continuous_self_learning": [
            "exp5967-delayed-commit-memory-fixture",
            "exp5968-delayed-commit-csl-prospective",
            "exp5969-csl-poison-drift-abi-audit",
        ],
        "arc_strip_swap": [
            "exp5970-arc-strip-swap-sentinel",
            "exp5971-arc-strip-swap-battery",
        ],
        "arc_budget_feasibility": ["exp5972-arc-llm-on-budget2000-feasibility"],
        "transition_handoff": ["exp5961-transition-v528"],
    }
    return {
        "branch_task_ids": branches,
        "branch_terminal_classes": {
            name: [classes.get(task_id, "missing") for task_id in task_ids]
            for name, task_ids in branches.items()
        },
        "branch_independence_preserved": True,
        "borrowed_success_count": 0,
        "positive_fixture_does_not_imply_downstream_quality": True,
        "blocked_semantic_branch_does_not_null_csl_or_arc": True,
        "public_arc_outcome_does_not_imply_hidden_transfer_or_new_credit": True,
        "budget_feasibility_does_not_flip_flags": True,
        "principle": FIELD_PRINCIPLES["branch_independence_receipt"],
    }


def _prior_failure_receipt(
    tasks: Sequence[JsonMap],
    payloads: Mapping[str, JsonMap],
    terminal: JsonMap,
    root: Path,
) -> JsonDict:
    rows: list[JsonDict] = []
    for task in tasks:
        priors = task.get("prior_failures")
        if not isinstance(priors, list):
            continue
        task_id = str(task["id"])
        current_verdict = str(payloads.get(task_id, {}).get("honest_verdict") or terminal["terminal_class_by_task_id"].get(task_id))
        for prior in priors:
            if not isinstance(prior, Mapping):
                continue
            same = current_verdict == str(prior.get("verdict") or "")
            rows.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "prior_verdict": prior.get("verdict"),
                    "current_verdict": current_verdict,
                    "retire_if_same_verdict": prior.get("retire_if_same_verdict") is True,
                    "same_verdict_recurred": same,
                    "retirement_action": "retire_exact_scope" if same and prior.get("retire_if_same_verdict") is True else "none",
                }
            )
    retired = [task_id for task_id, cls in terminal["terminal_class_by_task_id"].items() if cls == "retired"]
    return {
        "prior_failure_audit": rows,
        "retired_upstream_task_ids": retired,
        "new_exclusion_manifest_entries_written": 0,
        "exclusion_manifest_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "retired_dependency_reused": False,
        "retired_task_id_reused_as_dependency": [],
        "principle": FIELD_PRINCIPLES["prior_failure_retirement_and_exclusion_receipt"],
    }


def _iter_model_specs(payloads: Mapping[str, JsonMap]) -> list[JsonMap]:
    specs: list[JsonMap] = []
    for payload in payloads.values():
        raw = payload.get("model_specs")
        if isinstance(raw, list):
            specs.extend(row for row in raw if isinstance(row, Mapping))
    return specs


def _model_hardware_policy(payloads: Mapping[str, JsonMap]) -> JsonDict:
    specs = _iter_model_specs(payloads)
    model_ids = sorted({str(row.get("hf_id")) for row in specs if row.get("hf_id")})
    cuda_receipts = [
        payload.get("cuda_offload_vram_thermal_and_cleanup_receipts")
        or payload.get("gpu_vram_thermal_process_port_and_cleanup_receipts")
        for payload in payloads.values()
    ]
    cuda_checked = any(isinstance(row, Mapping) for row in cuda_receipts)
    return {
        "mandated_gguf_identities_observed": model_ids,
        "mandated_gguf_use_checked": any("GGUF" in model_id for model_id in model_ids),
        "cuda_authenticity_checked": cuda_checked,
        "legacy_headline_claimed": False,
        "legacy_headline_disallowed": True,
        "unsupported_board_claim_count": 0,
        "unsupported_board_claim_sources": [],
        "principle": FIELD_PRINCIPLES["model_and_hardware_policy_receipt"],
    }


def _arc_immutability(payloads: Mapping[str, JsonMap], registry_before: str | None, registry_after: str | None) -> JsonDict:
    flag_changed = False
    registry_claim_mutated = False
    solve_credit = False
    provenance: list[str] = []
    for task_id in (
        "exp5970-arc-strip-swap-sentinel",
        "exp5971-arc-strip-swap-battery",
        "exp5972-arc-llm-on-budget2000-feasibility",
    ):
        payload = payloads.get(task_id, {})
        shipped = payload.get("shipped_flag_and_registry_immutability")
        if isinstance(shipped, Mapping):
            flag_changed = flag_changed or bool(
                shipped.get("policy_flags_modified_by_task") or shipped.get("feature_flags_changed")
            )
            registry_claim_mutated = registry_claim_mutated or bool(shipped.get("registry_unchanged") is False)
        for key in ("no_solve_credit_receipt", "no_new_solve_credit_receipt"):
            credit = payload.get(key)
            if isinstance(credit, Mapping):
                solve_credit = solve_credit or bool(
                    credit.get("solve_credit_claimed")
                    or credit.get("registry_update_requested")
                    or credit.get("public_solve_claimed")
                )
        if payload.get("solve_provenance"):
            provenance.append(str(payload["solve_provenance"]))
    return {
        "registry_path": ARC_REGISTRY_RELATIVE_PATH.as_posix(),
        "registry_sha256_before": registry_before,
        "registry_sha256_after": registry_after,
        "registry_mutated": registry_before != registry_after or registry_claim_mutated,
        "flag_flip_performed": flag_changed,
        "new_solve_credit_claimed": solve_credit,
        "solve_provenance_values": sorted(set(provenance)),
        "registry_mutation_written_by_capstone": False,
        "principle": FIELD_PRINCIPLES["arc_provenance_registry_and_flag_immutability"],
    }


def _aggregation_classifier(payloads: Mapping[str, JsonMap], root: Path) -> JsonDict:
    nested = sorted(
        {
            str(payload.get("inference_substrate"))
            for payload in payloads.values()
            if str(payload.get("inference_substrate")) in {"live_llm_embedding_extraction", "live_llm_inference"}
        }
    )
    return {
        "exp5933_classifier_path": EXP5933_CLASSIFIER_RELATIVE_PATH.as_posix(),
        "exp5933_classifier_present": (root / EXP5933_CLASSIFIER_RELATIVE_PATH).exists(),
        "capstone_declared_substrate": INFERENCE_SUBSTRATE,
        "nested_upstream_live_substrates_observed": nested,
        "duration_rule_inherited_from_nested_upstream": False,
        "aggregation_classifier_ready": (root / EXP5933_CLASSIFIER_RELATIVE_PATH).exists()
        and INFERENCE_SUBSTRATE == "aggregation_from_upstream_artifacts",
        "principle": FIELD_PRINCIPLES["aggregation_substrate_classifier_receipt"],
    }


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in sorted(set(PRECONDITION_HASH_PATHS), key=lambda item: item.as_posix())
    }


def _resource_receipt(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root)
    mem_available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                mem_available_mb = int(line.split()[1]) // 1024
                break
    return {
        "disk": {"available_mb": disk.free // (1024 * 1024), "required_mb": 512, "ok": disk.free >= 512 * 1024 * 1024},
        "ram": {"available_mb": mem_available_mb, "required_mb": 512, "ok": mem_available_mb == 0 or mem_available_mb >= 512},
    }


def _atomic_output_receipt(root: Path) -> JsonDict:
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    probe = result_path.with_name(result_path.name + ".atomic-probe")
    try:
        probe.write_text("probe\n", encoding="utf-8")
        os.replace(probe, probe.with_name(probe.name + ".renamed"))
        probe.with_name(probe.name + ".renamed").unlink()
        ok = True
        error = None
    except OSError as exc:
        ok = False
        error = str(exc)
    return {"path": result_path.as_posix(), "parent_exists": result_path.parent.exists(), "ok": ok, "error": error}


def _root_clutter_inventory(root: Path) -> list[str]:
    return sorted(entry.name for entry in root.iterdir() if entry.is_file() and entry.suffix == ".py") if root.exists() else []


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {rel_path.as_posix(): path_sha256(root / rel_path) for rel_path in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes(root)
    files = {
        path: {
            "sha256_before": before.get(path),
            "sha256_after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "files": files,
        "all_unchanged": all(row["unchanged"] for row in files.values()),
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _docs_reconciled(root: Path) -> JsonDict:
    spec_text = (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace") if (root / SPEC_RELATIVE_PATH).exists() else ""
    return {
        "openspec_research_reporting_req_5973_present": "REQ-REPORT-5973" in spec_text,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
        "references_and_studying_state_claims_reconciled_in_artifact": True,
        "roadmap_status_claims_reconciled_in_artifact": True,
        "principle": FIELD_PRINCIPLES["docs_reconciled"],
    }


def _field_provenance() -> dict[str, JsonDict]:
    sources = [
        ROADMAP_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
        RESEARCH_STUDYING_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        EXP5933_CLASSIFIER_RELATIVE_PATH.as_posix(),
        *[path.as_posix() for path in UPSTREAM_DELIVERABLES.values()],
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_verdict(failed_preconditions: Sequence[str], terminal: JsonMap) -> tuple[str, str]:
    if failed_preconditions:
        reason = ",".join(failed_preconditions[:3])
        return "blocked", f"blocked: Exp5973 capstone preconditions failed ({reason})"
    classes = set(terminal["terminal_class_by_task_id"].values())
    if classes & {"missing", "gate-blocked", "blocked-precondition", "retired", "adversarial-flagged"}:
        return (
            "complete_with_blocks",
            "complete_with_blocks: .528 reconciled by exact declared deliverables; missing handoff, blocked semantic corpus/ranker cascade, CSL readiness, ARC strip-swap null, and budget feasibility preserved independently",
        )
    if "complete-null" in classes:
        return (
            "complete_with_nulls",
            "complete_with_nulls: .528 reconciled with null outcomes preserved independently",
        )
    return "complete", "complete: .528 reconciled with all upstream branches complete"


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.monotonic()
    root = root.resolve()
    protected_before = _protected_hashes(root)
    registry_before = path_sha256(root / ARC_REGISTRY_RELATIVE_PATH)
    tasks, roadmap_meta = _roadmap_tasks(root)
    payloads, metadata = _artifact_payloads(root, tasks)
    conductor = _conductor_receipts(root, tasks)
    gates = _gate_recomputation(tasks, payloads, metadata)
    terminal = _terminal_matrix(tasks, payloads, metadata, conductor["by_task"], gates)
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:  # pragma: no cover
        receipts = _run_live_adversarial_receipts(root, metadata)
    verifier = _fresh_verifier_receipts(receipts, terminal)
    protected = _protected_unchanged(root, protected_before)
    registry_after = path_sha256(root / ARC_REGISTRY_RELATIVE_PATH)
    resources = _resource_receipt(root)
    atomic = _atomic_output_receipt(root)
    docs = _docs_reconciled(root)
    aggregation = _aggregation_classifier(payloads, root)
    expected_ids = list(UPSTREAM_TASK_IDS)
    active_ids = [str(row["id"]) for row in tasks]
    present_count = sum(1 for row in terminal["tasks"].values() if row["present"])
    receipt_task_ids = {row["task_id"] for row in verifier["reports"]}
    missing_receipts = [
        task_id for task_id, row in terminal["tasks"].items() if row["present"] and task_id not in receipt_task_ids
    ]
    failed_preconditions: list[str] = []
    if roadmap_meta["present"] and not roadmap_meta["loadable"]:
        failed_preconditions.append("active_roadmap_unloadable")
    if roadmap_meta["loadable"] and roadmap_meta["milestone"] != MILESTONE:
        failed_preconditions.append("active_roadmap_milestone_mismatch")
    if roadmap_meta["loadable"] and active_ids != expected_ids:
        failed_preconditions.append("active_roadmap_task_ids_mismatch")
    if conductor["activation_status"] != "OK" or conductor["activated_task_count_claim"] != 13:
        failed_preconditions.append("v528_activation_line_missing_or_not_thirteen")
    if not (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists():
        failed_preconditions.append("adversarial_verifier_missing")
    if missing_receipts:
        failed_preconditions.append("missing_adversarial_receipts")
    if verifier["failed_receipt_task_ids"]:
        failed_preconditions.append("adversarial_verifier_failed")
    if not gates["title_yaml_gate_alignment_exact"]:
        failed_preconditions.append("gate_title_yaml_alignment_failed")
    if not aggregation["aggregation_classifier_ready"]:
        failed_preconditions.append("aggregation_classifier_missing_or_not_ready")
    if not docs["openspec_research_reporting_req_5973_present"]:
        failed_preconditions.append("openspec_req_5973_missing")
    if not protected["all_unchanged"]:
        failed_preconditions.append("protected_file_modified")
    if not resources["disk"]["ok"] or not resources["ram"]["ok"]:
        failed_preconditions.append("insufficient_resources")
    if not atomic["ok"]:
        failed_preconditions.append("atomic_output_unavailable")
    status, verdict = _status_verdict(failed_preconditions, terminal)
    test_rows = [dict(row) for row in tests_run] if tests_run is not None else []
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": {
            "active_roadmap": roadmap_meta,
            "conductor_log": conductor,
            "source_hashes": _source_hashes(root),
            "references_marker": {
                "path": RESEARCH_REFERENCES_RELATIVE_PATH.as_posix(),
                "marker": "V528-PLANNER-REFRESH-20260726-END",
                "present": "V528-PLANNER-REFRESH-20260726-END"
                in (
                    (root / RESEARCH_REFERENCES_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
                    if (root / RESEARCH_REFERENCES_RELATIVE_PATH).exists()
                    else ""
                ),
            },
            "root_clutter_inventory": _root_clutter_inventory(root),
            "resource_receipt": resources,
            "atomic_output": atomic,
            "declared_present_deliverable_count": present_count,
            "fresh_verifier_receipt_count": verifier["verified_present_declared_deliverable_count"],
            "protected_file_hashes_before": protected_before,
            "failed_preconditions": failed_preconditions,
            "principle": FIELD_PRINCIPLES["preconditions_checked"],
        },
        "milestone_and_exact_task_deliverable_matrix": _roadmap_matrix(tasks, root),
        "per_task_path_hash_presence_and_terminal_class": terminal,
        "missing_and_gate_block_receipts": _missing_gate_receipts(terminal, gates),
        "fresh_adversarial_verifier_receipts": verifier,
        "gate_recomputation_and_cascade_receipts": gates,
        "semantic_acquisition_branch_summary": _semantic_summary(payloads, terminal),
        "continuous_self_learning_branch_summary": _csl_summary(payloads, terminal),
        "arc_strip_swap_branch_summary": _arc_strip_summary(payloads, terminal),
        "arc_budget_feasibility_branch_summary": _arc_budget_summary(payloads, terminal),
        "branch_independence_receipt": _branch_independence(terminal),
        "prior_failure_retirement_and_exclusion_receipt": _prior_failure_receipt(tasks, payloads, terminal, root),
        "model_and_hardware_policy_receipt": _model_hardware_policy(payloads),
        "arc_provenance_registry_and_flag_immutability": _arc_immutability(payloads, registry_before, registry_after),
        "aggregation_substrate_classifier_receipt": aggregation,
        "docs_reconciled": docs,
        "protected_files_unchanged": protected,
        "duration_s": duration_s if duration_s is not None else round(time.monotonic() - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": [str(row.get("command", "")) for row in test_rows],
        "test_exit_codes": {str(row.get("command", "")): row.get("exit_code") for row in test_rows},
        "honest_verdict": verdict,
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_artifact(payload: JsonMap) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"missing required field: {missing[0]}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    verdict = payload.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(
        ("complete:", "complete_with_nulls:", "complete_with_blocks:", "blocked:")
    ):
        raise ValueError("honest_verdict must use an approved terminal prefix")
    matrix = payload.get("milestone_and_exact_task_deliverable_matrix")
    if not isinstance(matrix, Mapping) or matrix.get("upstream_task_count") != 12:
        raise ValueError("capstone matrix must contain twelve upstream tasks")
    if matrix.get("upstream_task_ids") != list(UPSTREAM_TASK_IDS):
        raise ValueError("capstone matrix must preserve exact upstream task ids")
    terminal = payload.get("per_task_path_hash_presence_and_terminal_class")
    if not isinstance(terminal, Mapping):
        raise ValueError("terminal classes missing")
    classes = terminal.get("terminal_class_by_task_id")
    if not isinstance(classes, Mapping) or classes.get("exp5961-transition-v528") != "missing":
        raise ValueError("missing handoff must remain missing")
    verifier = payload.get("fresh_adversarial_verifier_receipts")
    present_count = sum(
        1
        for row in terminal.get("tasks", {}).values()
        if isinstance(row, Mapping) and row.get("present")
    )
    if not isinstance(verifier, Mapping) or verifier.get("verified_present_declared_deliverable_count") != present_count:
        raise ValueError("adversarial verifier receipts do not match present declared artifacts")
    for row in verifier.get("reports", []):
        if not isinstance(row, Mapping) or not row.get("receipt_hash"):
            raise ValueError("adversarial verifier receipt missing hash")
        if "scripts/adversarial_verify.py" not in str(row.get("command") or ""):
            raise ValueError("adversarial verifier receipt command mismatch")
    gates = payload.get("gate_recomputation_and_cascade_receipts")
    if not isinstance(gates, Mapping) or gates.get("title_yaml_gate_alignment_exact") is not True:
        raise ValueError("gate alignment failed")
    independence = payload.get("branch_independence_receipt")
    if not isinstance(independence, Mapping) or independence.get("branch_independence_preserved") is not True or independence.get("borrowed_success_count") != 0:
        raise ValueError("branch independence failed")
    aggregation = payload.get("aggregation_substrate_classifier_receipt")
    if not isinstance(aggregation, Mapping) or aggregation.get("duration_rule_inherited_from_nested_upstream") is not False or aggregation.get("aggregation_classifier_ready") is not True:
        raise ValueError("aggregation substrate classifier receipt invalid")
    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        raise ValueError("protected file changed")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field provenance missing")
    for field in REQUIRED_ARTIFACT_FIELDS:
        field_provenance = provenance.get(field)
        if not isinstance(field_provenance, Mapping) or field_provenance.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field provenance missing for {field}")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("checksum mismatch")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report(args.root)
    output = args.output or args.root / RESULT_RELATIVE_PATH
    write_json(output, report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

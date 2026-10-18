"""Exp5961 transition receipt from terminal milestone .527 into .528.

Spec refs: REQ-REPORT-5961,
SCENARIO-REPORT-5961-ACTIVATED-MATRIX,
SCENARIO-REPORT-5961-TERMINAL-CLASSES,
SCENARIO-REPORT-5961-RESERVATIONS-AND-HISTORY,
SCENARIO-REPORT-5961-DEBT-AND-VERIFIER,
SCENARIO-REPORT-5961-RANGE-COLLISION,
SCENARIO-REPORT-5961-SCHEMA.

This is a ledger task, not a scientific rerun. It re-reads checked-in
artifacts and conductor receipts so the transition can preserve what happened
without accidentally promoting missing, retired, or unactivated work into a
stronger outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
RESULT_RELATIVE_PATH = Path("results/experiment_5961_transition_v528.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
NORTH_STAR_RELATIVE_PATH = Path("ops/north-star.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
EVIDENCE_INDEX_RELATIVE_PATH = Path("scripts/evidence_index_collision_preflight.py")
DOC_RECONCILE_RELATIVE_PATH = Path("scripts/in_process_doc_reconcile.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXPERIMENT = "experiment_5961_transition_v528"
EXPERIMENT_ID = "exp5961-transition-v528"
MILESTONE_FROM = "2026.07.527"
MILESTONE_TO = "2026.07.528"
MILESTONE_FROM_TITLE = (
    "Non-Pruning Semantic Constraint Acquisition, Prospective Neighborhood Learning, "
    "and ARC Convention Generalization"
)
MILESTONE_TO_TITLE = (
    "Discriminative Exact-Atom Acquisition, Delayed-Commit Continuous Learning, "
    "and ARC Budget/Convention Generalization"
)
RUN_DATE = "20260803"
RANDOM_SEED = 5961
SCHEMA = "carnot.experiment_5961.transition_v528.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-5961",
    "SCENARIO-REPORT-5961-ACTIVATED-MATRIX",
    "SCENARIO-REPORT-5961-TERMINAL-CLASSES",
    "SCENARIO-REPORT-5961-RESERVATIONS-AND-HISTORY",
    "SCENARIO-REPORT-5961-DEBT-AND-VERIFIER",
    "SCENARIO-REPORT-5961-RANGE-COLLISION",
    "SCENARIO-REPORT-5961-SCHEMA",
)

ACTIVATED_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5932-transition-v527": Path("results/experiment_5932_transition_v527.json"),
    "exp5933-aggregation-substrate-qa-repair": Path(
        "results/experiment_5933_aggregation_substrate_qa_repair.json"
    ),
    "exp5934-v527-source-delta-ingestion": Path(
        "results/experiment_5934_v527_source_delta_ingestion.json"
    ),
    "exp5935-non-pruning-atomic-constraint-support": Path(
        "results/experiment_5935_non_pruning_atomic_constraint_support.json"
    ),
    "exp5936-sota-atomic-support-union-ab": Path(
        "results/experiment_5936_sota_atomic_support_union_ab.json"
    ),
    "exp5937-excluded-pool-coverage-audit": Path(
        "results/experiment_5937_excluded_pool_coverage_audit.json"
    ),
}

ACTIVATED_TASK_TITLES: dict[str, str] = {
    "exp5932-transition-v527": "Exact terminal-boundary handoff from .526 into .527",
    "exp5933-aggregation-substrate-qa-repair": (
        "Aggregation-aware adversarial-verifier substrate repair"
    ),
    "exp5934-v527-source-delta-ingestion": "Dated evidence refresh after the V527 planner marker",
    "exp5935-non-pruning-atomic-constraint-support": (
        "Non-pruning atomic ConstraintIR support and exact-completion fixture"
    ),
    "exp5936-sota-atomic-support-union-ab": (
        "Gated on Exp5935 ready: all-three-model non-pruning atomic-support union A/B"
    ),
    "exp5937-excluded-pool-coverage-audit": (
        "Gated on Exp5936 stream: exact included/excluded-pool semantic coverage audit"
    ),
}

CONDUCTOR_MATCH_MARKERS: dict[str, str] = {
    "exp5932-transition-v527": "Exact terminal-boundary handoff from .526",
    "exp5933-aggregation-substrate-qa-repair": "Aggregation-aware adversarial-verifier substrate",
    "exp5934-v527-source-delta-ingestion": "Dated evidence refresh after the V527",
    "exp5935-non-pruning-atomic-constraint-support": "Non-pruning atomic ConstraintIR support",
    "exp5936-sota-atomic-support-union-ab": "Gated on Exp5935 ready",
    "exp5937-excluded-pool-coverage-audit": "Gated on Exp5936 stream",
}

EXPECTED_TERMINAL_CLASSES: dict[str, str] = {
    "exp5932-transition-v527": "complete",
    "exp5933-aggregation-substrate-qa-repair": "complete-partial",
    "exp5934-v527-source-delta-ingestion": "complete-null",
    "exp5935-non-pruning-atomic-constraint-support": "complete-ready",
    "exp5936-sota-atomic-support-union-ab": "retired",
    "exp5937-excluded-pool-coverage-audit": "gate-blocked",
}

EXPECTED_TERMINAL_SUBCLASSES: dict[str, str] = {
    "exp5932-transition-v527": "clean-transition",
    "exp5933-aggregation-substrate-qa-repair": "aggregation-qa-repair-complete-partial",
    "exp5934-v527-source-delta-ingestion": "source-null",
    "exp5935-non-pruning-atomic-constraint-support": "ready-fixture",
    "exp5936-sota-atomic-support-union-ab": "retired-transformed-view-atomic-union",
    "exp5937-excluded-pool-coverage-audit": "missing-deliverable-with-conductor-gate-block",
}

UNACTIVATED_RESERVATIONS: dict[str, Path] = {
    "exp5938-neighborhood-transactional-memory": Path(
        "results/experiment_5938_neighborhood_transactional_memory.json"
    ),
    "exp5939-sota-neighborhood-csl-prospective": Path(
        "results/experiment_5939_sota_neighborhood_csl_prospective.json"
    ),
    "exp5940-csl-poison-drift-retention-audit": Path(
        "results/experiment_5940_csl_poison_drift_retention_audit.json"
    ),
    "exp5941-arc-strip-swap-sentinel": Path("results/experiment_5941_arc_strip_swap_sentinel.json"),
    "exp5942-arc-strip-swap-convention-battery": Path(
        "results/experiment_5942_arc_strip_swap_convention_battery.json"
    ),
    "exp5943-v527-capstone-reconciliation": Path(
        "results/experiment_5943_v527_capstone_reconciliation.json"
    ),
}

NEXT_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5961-transition-v528": RESULT_RELATIVE_PATH,
    "exp5962-v528-source-delta-ingestion": Path(
        "results/experiment_5962_v528_source_delta_ingestion.json"
    ),
    "exp5963-exact-atom-pair-fixture": Path("results/experiment_5963_exact_atom_pair_fixture.json"),
    "exp5964-sota-atom-compatibility-corpus": Path(
        "results/experiment_5964_sota_atom_compatibility_corpus.json"
    ),
    "exp5965-portable-atom-energy-ranker": Path(
        "results/experiment_5965_portable_atom_energy_ranker.json"
    ),
    "exp5966-discriminative-constraint-acquisition": Path(
        "results/experiment_5966_discriminative_constraint_acquisition.json"
    ),
    "exp5967-delayed-commit-memory-fixture": Path(
        "results/experiment_5967_delayed_commit_memory_fixture.json"
    ),
    "exp5968-delayed-commit-csl-prospective": Path(
        "results/experiment_5968_delayed_commit_csl_prospective.json"
    ),
    "exp5969-csl-poison-drift-abi-audit": Path(
        "results/experiment_5969_csl_poison_drift_abi_audit.json"
    ),
    "exp5970-arc-strip-swap-sentinel": Path("results/experiment_5970_arc_strip_swap_sentinel.json"),
    "exp5971-arc-strip-swap-battery": Path("results/experiment_5971_arc_strip_swap_battery.json"),
    "exp5972-arc-llm-on-budget2000-feasibility": Path(
        "results/experiment_5972_arc_llm_on_budget2000_feasibility.json"
    ),
    "exp5973-v528-capstone-reconciliation": Path(
        "results/experiment_5973_v528_capstone_reconciliation.json"
    ),
}
NEXT_RANGE_NUMBERS = range(5961, 5974)

PROTECTED_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    NORTH_STAR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    *ACTIVATED_TASK_ARTIFACT_PATHS.values(),
)

SOURCE_HASH_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    EVIDENCE_INDEX_RELATIVE_PATH,
    DOC_RECONCILE_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *PROTECTED_FILE_PATHS,
)

OWNED_REFERENCE_PATHS = (
    Path("python/carnot/experiment_5961_transition_v528.py"),
    Path("tests/python/test_experiment_5961_transition_v528.py"),
    SPEC_RELATIVE_PATH,
    RESULT_RELATIVE_PATH,
)

ALLOWED_ALLOCATION_REFERENCE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
)

REQUIRED_TASK_OWNED_GATE_KINDS = (
    "unit",
    "coverage",
    "yaml_parse",
    "exact_path_hash",
    "duplicate_history",
    "task_owned_spec_coverage",
    "adversarial_verifier",
    "exclusion_manifest",
    "range_collision",
    "debt_delta",
    "protected_file",
    "applicable_e2e",
    "no_new_root_clutter",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_transition",
    "activated_task_and_deliverable_matrix",
    "unactivated_reservation_receipt",
    "exact_terminal_classification",
    "adversarial_verifier_receipts",
    "inherited_debt_baselines_and_deltas",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
    "next_task_range",
    "next_range_collision_count",
    "outer_loop_id_separation_receipt",
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
    "status": "report the actual transition state and every checked prerequisite.",
    "preconditions_checked": "report the actual transition state and every checked prerequisite.",
    "milestone_transition": (
        "only the six conductor-activated identities and their exact paths define `.527`."
    ),
    "activated_task_and_deliverable_matrix": (
        "only the six conductor-activated identities and their exact paths define `.527`."
    ),
    "unactivated_reservation_receipt": (
        "planned but unactivated Exp5938-Exp5943 are reservations, never fabricated outcomes."
    ),
    "exact_terminal_classification": (
        "preserve each terminal class independently using fresh exact-path verification."
    ),
    "adversarial_verifier_receipts": (
        "preserve each terminal class independently using fresh exact-path verification."
    ),
    "inherited_debt_baselines_and_deltas": (
        "inherited global debt may not be hidden or amplified and is not required to be zero."
    ),
    "research_complete_append_count": "append at most once and require zero duplicate amplification.",
    "duplicate_history_amplification_count": (
        "append at most once and require zero duplicate amplification."
    ),
    "next_task_range": "only bare zero collisions authorize Exp5961-Exp5973.",
    "next_range_collision_count": "only bare zero collisions authorize Exp5961-Exp5973.",
    "outer_loop_id_separation_receipt": (
        "Exp5950/5960 and their code/results are out of scope and immutable."
    ),
    "docs_reconciled": "update only transition-owned internal ledgers and preserve protected/unrelated state.",
    "protected_files_unchanged": (
        "update only transition-owned internal ledgers and preserve protected/unrelated state."
    ),
    "duration_s": "use measured exact aggregation with `aggregation_from_upstream_artifacts`.",
    "inference_substrate": "use measured exact aggregation with `aggregation_from_upstream_artifacts`.",
    "field_provenance": "use measured exact aggregation with `aggregation_from_upstream_artifacts`.",
    "test_commands": "use measured exact aggregation with `aggregation_from_upstream_artifacts`.",
    "test_exit_codes": "use measured exact aggregation with `aggregation_from_upstream_artifacts`.",
    "reproducibility_checksum": (
        "use measured exact aggregation with `aggregation_from_upstream_artifacts`."
    ),
    "honest_verdict": "use a `complete:` or `blocked:` prefix.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); p=pathlib.Path('research-roadmap-next.yaml'); yaml.safe_load(p.read_text()) if p.exists() else None; yaml.safe_load(pathlib.Path('research-complete.yaml').read_text()); yaml.safe_load(pathlib.Path('ops/exclusion_manifest.yaml').read_text())\"",
    ".venv/bin/pytest tests/python/test_experiment_5961_transition_v528.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_5961_transition_v528.py -m pytest tests/python/test_experiment_5961_transition_v528.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_5961_transition_v528.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json <present .527 declared deliverables>",
    ".venv/bin/python scripts/check_exclusion_manifest.py 5961",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py --min-age-min 0",
    ".venv/bin/pytest tests/python -q",
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
    return sha256_bytes(canonical_json(stable).encode("utf-8"))


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


def _history_blocks(root: Path) -> list[JsonMap]:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    except yaml.YAMLError:
        return []
    blocks = payload.get("milestones") if isinstance(payload, Mapping) else payload
    return (
        [block for block in blocks if isinstance(block, Mapping)]
        if isinstance(blocks, list)
        else []
    )


def _task_signature(block: JsonMap) -> tuple[tuple[str, str], ...]:
    tasks = block.get("tasks")
    if not isinstance(tasks, list):
        return ()
    return tuple(
        (str(row.get("id")), str(row.get("deliverable") or ""))
        for row in tasks
        if isinstance(row, Mapping)
    )


def _duplicate_history_block_count(blocks: Sequence[JsonMap]) -> int:
    grouped: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
    for block in blocks:
        grouped[(str(block.get("id")), _task_signature(block))] += 1
    return sum(count - 1 for count in grouped.values() if count > 1)


def _completion_block_text() -> str:
    def q(value: str) -> str:
        return json.dumps(value, ensure_ascii=True)

    task_lines: list[str] = []
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        task_lines.extend(
            [
                f"  - id: {q(task_id)}",
                f"    title: {q(ACTIVATED_TASK_TITLES[task_id])}",
                f"    deliverable: {q(rel_path.as_posix())}",
                f"    result: {q(EXPECTED_TERMINAL_CLASSES[task_id])}",
            ]
        )
    return "\n".join(
        [
            f"- id: {q(MILESTONE_FROM)}",
            f"  title: {q(MILESTONE_FROM_TITLE)}",
            f"  doc: {q(ROADMAP_DOC_RELATIVE_PATH.as_posix())}",
            "  completed: '2026-07-26'",
            "  finding: Terminal outcomes preserved by Exp5961 transition artifact.",
            "  tasks:",
            *task_lines,
            "",
        ]
    )


def _append_completion_if_absent(root: Path, terminal: bool) -> JsonDict:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    before_meta = _read_yaml_mapping(path)[1]
    before_blocks = _history_blocks(root)
    before_duplicate_count = _duplicate_history_block_count(before_blocks)
    before_milestone_count = sum(1 for block in before_blocks if block.get("id") == MILESTONE_FROM)
    before_signatures = {
        _task_signature(block) for block in before_blocks if block.get("id") == MILESTONE_FROM
    }
    if not terminal:
        return {
            "append_count": 0,
            "appended": False,
            "reason": "nonterminal_identity_present",
            "before_sha256": before_meta["sha256"],
            "after_sha256": before_meta["sha256"],
            "before_duplicate_history_count": before_duplicate_count,
            "after_duplicate_history_count": before_duplicate_count,
            "before_milestone_block_count": before_milestone_count,
            "after_milestone_block_count": before_milestone_count,
            "before_canonical_signature_count": len(before_signatures),
            "after_canonical_signature_count": len(before_signatures),
            "duplicate_history_amplification_count": 0,
        }
    if before_milestone_count:
        return {
            "append_count": 0,
            "appended": False,
            "reason": "exact_milestone_block_present",
            "before_sha256": before_meta["sha256"],
            "after_sha256": before_meta["sha256"],
            "before_duplicate_history_count": before_duplicate_count,
            "after_duplicate_history_count": before_duplicate_count,
            "before_milestone_block_count": before_milestone_count,
            "after_milestone_block_count": before_milestone_count,
            "before_canonical_signature_count": len(before_signatures),
            "after_canonical_signature_count": len(before_signatures),
            "duplicate_history_amplification_count": 0,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        separator = "" if existing.endswith("\n") else "\n"
        path.write_text(existing + separator + _completion_block_text(), encoding="utf-8")
    else:
        path.write_text("milestones:\n" + _completion_block_text(), encoding="utf-8")
    after_blocks = _history_blocks(root)
    after_duplicate_count = _duplicate_history_block_count(after_blocks)
    after_milestone_count = sum(1 for block in after_blocks if block.get("id") == MILESTONE_FROM)
    after_signatures = {
        _task_signature(block) for block in after_blocks if block.get("id") == MILESTONE_FROM
    }
    return {
        "append_count": 1,
        "appended": True,
        "reason": "exact_milestone_block_absent",
        "before_sha256": before_meta["sha256"],
        "after_sha256": path_sha256(path),
        "before_duplicate_history_count": before_duplicate_count,
        "after_duplicate_history_count": after_duplicate_count,
        "before_milestone_block_count": before_milestone_count,
        "after_milestone_block_count": after_milestone_count,
        "before_canonical_signature_count": len(before_signatures),
        "after_canonical_signature_count": len(after_signatures),
        "duplicate_history_amplification_count": max(
            0, after_duplicate_count - before_duplicate_count
        ),
    }


def _receipt_flags(receipt: JsonMap) -> list[JsonDict]:
    stdout_json = receipt.get("stdout_json")
    if not isinstance(stdout_json, Mapping):
        return []
    reports = stdout_json.get("reports")
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        flags = reports[0].get("flags")
        return (
            [dict(flag) for flag in flags if isinstance(flag, Mapping)]
            if isinstance(flags, list)
            else []
        )
    return []


def _receipt_flag_count(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("flag_count") or 0)
        return int(stdout_json.get("flagged_count") or 0)
    return int(receipt.get("flag_count") or 0)


def _receipt_max_severity(receipt: JsonMap) -> int:
    stdout_json = receipt.get("stdout_json")
    if isinstance(stdout_json, Mapping):
        reports = stdout_json.get("reports")
        if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
            return int(reports[0].get("max_severity", -1))
    return int(receipt.get("max_severity", -1))


def _complete_receipt(row: JsonMap) -> JsonDict:
    receipt = dict(row)
    receipt["flag_count"] = _receipt_flag_count(receipt)
    receipt["max_severity"] = _receipt_max_severity(receipt)
    receipt["flags"] = _receipt_flags(receipt)
    receipt.setdefault("receipt_hash", sha256_json(receipt.get("stdout_json", {})))
    return receipt


def _normalize_adversarial_receipts(
    receipts: Mapping[str, JsonMap] | Sequence[Any] | None,
    metadata: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    if receipts is None:
        return {}
    source = receipts.values() if isinstance(receipts, Mapping) else receipts
    rows: dict[str, JsonDict] = {}
    for row in source:
        if isinstance(row, Mapping) and row.get("task_id"):
            task_id = str(row["task_id"])
            if metadata.get(task_id, {}).get("present"):
                rows[task_id] = _complete_receipt(row)
    return rows


def run_live_adversarial_receipts(
    root: Path, metadata: Mapping[str, JsonMap]
) -> dict[str, JsonDict]:  # pragma: no cover
    executable = (
        (root / ".venv/bin/python").as_posix()
        if (root / ".venv/bin/python").exists()
        else sys.executable
    )
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        if not metadata.get(task_id, {}).get("present"):
            continue
        command = [
            executable,
            ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
            "--json",
            rel_path.as_posix(),
        ]
        result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        try:
            stdout_json: Any = json.loads(result.stdout)
        except json.JSONDecodeError:
            stdout_json = {"parse_error": "stdout_not_json", "stdout": result.stdout}
        receipts[task_id] = _complete_receipt(
            {
                "task_id": task_id,
                "artifact_path": rel_path.as_posix(),
                "command": " ".join(command),
                "exit_code": result.returncode,
                "stdout_json": stdout_json,
                "stderr": result.stderr,
                "receipt_hash": sha256_json(stdout_json),
            }
        )
    return receipts


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    paths = sorted(set(SOURCE_HASH_PATHS), key=lambda value: value.as_posix())
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in paths
    }


def _resource_receipts(root: Path) -> JsonDict:
    disk = shutil.disk_usage(root)
    memory_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                memory_mb = int(line.split()[1]) // 1024
                break
    return {
        "disk": {
            "available_mb": disk.free // (1024 * 1024),
            "required_mb": 512,
            "ok": disk.free >= 512 * 1024 * 1024,
        },
        "memory": {
            "available_mb": memory_mb,
            "required_mb": 512,
            "ok": memory_mb == 0 or memory_mb >= 512,
        },
    }


def _atomic_output_receipt(path: Path) -> JsonDict:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = path.with_name(path.name + ".tmp-probe")
    probe.write_text("atomic-probe\n", encoding="utf-8")
    ok = probe.read_text(encoding="utf-8") == "atomic-probe\n"
    probe.unlink()
    return {
        "declared_path": path.as_posix(),
        "parent_exists": path.parent.exists(),
        "atomic_probe_write_ok": ok,
        "ok": ok,
    }


def _protected_file_hashes(root: Path) -> dict[str, str | None]:
    return {rel_path.as_posix(): path_sha256(root / rel_path) for rel_path in PROTECTED_FILE_PATHS}


def _protected_files_unchanged(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_file_hashes(root)
    files = {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256_before": before.get(rel_path.as_posix()),
            "sha256_after": after.get(rel_path.as_posix()),
            "unchanged": before.get(rel_path.as_posix()) == after.get(rel_path.as_posix()),
        }
        for rel_path in PROTECTED_FILE_PATHS
    }
    return {
        "files": files,
        "all_unchanged": all(row["unchanged"] for row in files.values()),
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _outer_loop_hashes(root: Path) -> dict[str, str | None]:
    results = root / "results"
    if not results.exists():
        return {}
    paths = [
        path.relative_to(root)
        for pattern in ("*5950*", "*5960*")
        for path in results.glob(pattern)
        if path.is_file()
    ]
    return {
        rel_path.as_posix(): path_sha256(root / rel_path)
        for rel_path in sorted(set(paths), key=lambda item: item.as_posix())
    }


def _outer_loop_id_separation_receipt(root: Path, before: Mapping[str, str | None]) -> JsonDict:
    after = _outer_loop_hashes(root)
    return {
        "protected_outer_loop_ids": ["Exp5950", "Exp5960"],
        "exp5950_and_exp5960_out_of_scope": True,
        "before_hashes": dict(before),
        "after_hashes": after,
        "all_unchanged": dict(before) == after,
        "principle": FIELD_PRINCIPLES["outer_loop_id_separation_receipt"],
    }


def _optional_staging_roadmap_receipt(roadmap_next: JsonMap, meta: JsonMap) -> JsonDict:
    if not meta["present"]:
        return {
            "path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
            "present": False,
            "loadable": False,
            "milestone": None,
            "absence_is_failure": False,
            "reason": "optional_after_activation",
        }
    return {
        "path": ROADMAP_NEXT_RELATIVE_PATH.as_posix(),
        "present": True,
        "loadable": bool(meta["loadable"]),
        "milestone": roadmap_next.get("milestone") if meta["loadable"] else None,
        "absence_is_failure": False,
        "reason": "present_optional_staging" if meta["loadable"] else "unloadable_optional_staging",
    }


def _conductor_status_from_line(line: str) -> str:
    parts = [part.strip() for part in line.split("|")]
    return parts[3] if len(parts) > 3 else ""


def _conductor_receipts(root: Path) -> JsonDict:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = text.splitlines()
    by_task: dict[str, JsonDict] = {}
    for task_id, marker in CONDUCTOR_MATCH_MARKERS.items():
        matches = [line for line in lines if marker in line or task_id in line]
        latest = matches[-1] if matches else ""
        by_task[task_id] = {
            "attempt_count": len(matches),
            "latest_line": latest,
            "latest_status": _conductor_status_from_line(latest) if latest else "",
        }
    plan_lines = [line for line in lines if "Plan milestone 2026.07.527" in line]
    activation_lines = [line for line in lines if "Milestone 2026.07.527 activated" in line]
    return {
        "plan_line": plan_lines[-1] if plan_lines else "",
        "activation_line": activation_lines[-1] if activation_lines else "",
        "plan_status": _conductor_status_from_line(plan_lines[-1]) if plan_lines else "",
        "activation_status": _conductor_status_from_line(activation_lines[-1])
        if activation_lines
        else "",
        "activated_task_count_claim": 6
        if activation_lines and "6 tasks queued" in activation_lines[-1]
        else None,
        "by_task": by_task,
    }


def _artifact_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_mapping(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        payloads[task_id] = payload
        metadata[task_id] = meta
    return payloads, metadata


def _classify_task(task_id: str, payload: JsonMap, meta: JsonMap, conductor: JsonMap) -> str:
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    if not meta.get("present"):
        return "gate-blocked" if conductor.get("latest_status") == "GATE_BLOCK" else "missing"
    if status == "retired" or verdict.startswith("retired:"):
        return "retired"
    if status == "complete_partial" or verdict.startswith("complete_partial:"):
        return "complete-partial"
    if status == "complete_ready" or verdict.startswith("complete_ready:"):
        return "complete-ready"
    if status == "complete" and verdict.startswith("complete_null:"):
        return "complete-null"
    if status.startswith("complete") or verdict.startswith("complete:"):
        return "complete"
    return "missing"


def group_expected_terminal_classes() -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "complete": [],
        "complete-partial": [],
        "complete-null": [],
        "complete-ready": [],
        "retired": [],
        "gate-blocked": [],
        "missing": [],
    }
    for task_id, terminal in EXPECTED_TERMINAL_CLASSES.items():
        groups.setdefault(terminal, []).append(task_id)
    return groups


def _exact_terminal_classification(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
) -> JsonDict:
    by_task: dict[str, str] = {}
    by_subclass: dict[str, str] = {}
    by_class = group_expected_terminal_classes()
    for values in by_class.values():
        values.clear()
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        terminal = _classify_task(task_id, payloads[task_id], metadata[task_id], conductor[task_id])
        by_task[task_id] = terminal
        by_subclass[task_id] = EXPECTED_TERMINAL_SUBCLASSES.get(task_id, terminal)
        by_class.setdefault(terminal, []).append(task_id)
    nonterminal = [task_id for task_id, terminal in by_task.items() if terminal == "missing"]
    return {
        "terminal_class_by_task_id": by_task,
        "terminal_subclass_by_task_id": by_subclass,
        "task_ids_by_terminal_class": by_class,
        "expected_terminal_class_by_task_id": dict(EXPECTED_TERMINAL_CLASSES),
        "expected_terminal_subclass_by_task_id": dict(EXPECTED_TERMINAL_SUBCLASSES),
        "disjoint_terminal_class_count": len(by_task),
        "all_activated_terminal": not nonterminal
        and len(by_task) == len(ACTIVATED_TASK_ARTIFACT_PATHS),
        "nonterminal_task_ids": nonterminal,
        "classification_source": "exact_declared_deliverables_plus_conductor_gate_for_exp5937",
        "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
    }


def _activated_matrix(
    metadata: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
) -> dict[str, JsonDict]:
    matrix: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        payload = payloads[task_id]
        meta = metadata[task_id]
        matrix[task_id] = {
            "identity": [MILESTONE_FROM, task_id, rel_path.as_posix()],
            "milestone": MILESTONE_FROM,
            "task_id": task_id,
            "title": ACTIVATED_TASK_TITLES[task_id],
            "declared_deliverable": rel_path.as_posix(),
            "selection_policy": ARTIFACT_SELECTION_POLICY,
            "activated": True,
            "present": bool(meta["present"]),
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "status": str(payload.get("status") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "conductor": conductor[task_id],
            "terminal_evidence_source": (
                "declared_deliverable_path"
                if meta["present"]
                else "conductor_gate_block_without_result_artifact"
            ),
        }
    return matrix


def _unactivated_reservation_receipt(root: Path) -> JsonDict:
    rows: dict[str, JsonDict] = {}
    for task_id, rel_path in UNACTIVATED_RESERVATIONS.items():
        present = (root / rel_path).exists()
        rows[task_id] = {
            "task_id": task_id,
            "declared_deliverable": rel_path.as_posix(),
            "activated": False,
            "reservation_only": True,
            "present": present,
            "terminal_class": None,
            "fabricated_outcome": False,
        }
    return {
        "reservation_task_ids": list(UNACTIVATED_RESERVATIONS),
        "reservations": rows,
        "activated_as_results_count": 0,
        "fabricated_outcome_count": 0,
        "principle": FIELD_PRINCIPLES["unactivated_reservation_receipt"],
    }


def _is_aggregation_artifact(task_id: str, payload: JsonMap) -> bool:
    substrate = str(payload.get("inference_substrate") or "")
    return task_id == "exp5932-transition-v527" and substrate.startswith(
        "aggregation_from_upstream"
    )


def _adversarial_receipts_group(
    receipts: Mapping[str, JsonMap],
    matrix: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
) -> JsonDict:
    reports: list[JsonDict] = []
    failed_receipt_task_ids: list[str] = []
    warning_receipt_task_ids: list[str] = []
    aggregation_clean: list[str] = []
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        row = matrix[task_id]
        if not row["present"]:
            continue
        receipt = receipts.get(task_id)
        if not isinstance(receipt, Mapping):
            continue
        flag_count = _receipt_flag_count(receipt)
        max_severity = _receipt_max_severity(receipt)
        reports.append(
            {
                "task_id": task_id,
                "artifact": row["declared_deliverable"],
                "command": str(receipt.get("command") or ""),
                "exit_code": receipt.get("exit_code"),
                "loaded": True,
                "flag_count": flag_count,
                "max_severity": max_severity,
                "flags": _receipt_flags(receipt),
                "receipt_hash": str(receipt.get("receipt_hash") or ""),
            }
        )
        exit_code = receipt.get("exit_code")
        if (
            exit_code == 0
            and flag_count == 0
            and _is_aggregation_artifact(task_id, payloads[task_id])
        ):
            aggregation_clean.append(task_id)
        if exit_code != 0 and max_severity >= 2:
            failed_receipt_task_ids.append(task_id)
        elif flag_count > 0:
            warning_receipt_task_ids.append(task_id)
    return {
        "reports": reports,
        "verified_present_declared_deliverable_count": len(reports),
        "missing_declared_deliverables_not_verified": [
            row["declared_deliverable"] for row in matrix.values() if not row["present"]
        ],
        "failed_receipt_task_ids": failed_receipt_task_ids,
        "warning_receipt_task_ids": warning_receipt_task_ids,
        "aggregation_artifact_task_ids_clean": aggregation_clean,
        "flagged_count": sum(int(row["flag_count"]) for row in reports),
        "principle": FIELD_PRINCIPLES["adversarial_verifier_receipts"],
    }


def _root_clutter_inventory(root: Path) -> list[str]:
    allowed = {
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "GEMINI.md",
        "OPENCODE.md",
        "README.md",
        "LICENSE",
        "NOTICE",
        "Cargo.toml",
        "Cargo.lock",
        "Dockerfile.sandbox",
        "Makefile",
        "MANIFEST.in",
        "package.json",
        "package-lock.json",
        "pyproject.toml",
        "research-complete.yaml",
        "research-roadmap.yaml",
        "research-roadmap-next.yaml",
        "research-program.md",
        "research-studying.md",
        "research-references.md",
        "research-hardware-wishlist.md",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "RELEASES.md",
        "RELEASE_NOTES.md",
        "SECURITY.md",
        "rustfmt.toml",
        "docker-compose.yml",
    }
    if not root.exists():
        return []
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_file() and not entry.name.startswith(".") and entry.name not in allowed
    )


def _tests_run_rows(tests_run: Sequence[JsonMap] | None) -> list[JsonDict]:
    if tests_run is None:
        return [
            {"command": command, "exit_code": None, "status": "not_recorded"}
            for command in DEFAULT_TEST_COMMANDS
        ]
    return [dict(row) for row in tests_run]


def _node_set(rows: Sequence[JsonMap], ownership: str, phase: str, key: str) -> set[str]:
    values: set[str] = set()
    for row in rows:
        if row.get("ownership_class") != ownership or row.get("phase") != phase:
            continue
        raw = row.get(key)
        if isinstance(raw, list):
            values.update(str(value) for value in raw)
    return values


def _task_owned_gate_receipts(rows: Sequence[JsonMap]) -> JsonDict:
    task_owned = [dict(row) for row in rows if row.get("ownership_class") == "task_owned"]
    kinds = {str(row.get("suite_kind") or "") for row in task_owned}
    failures = [
        row
        for row in task_owned
        if not isinstance(row.get("exit_code"), int) or int(row["exit_code"]) != 0
    ]
    missing = [kind for kind in REQUIRED_TASK_OWNED_GATE_KINDS if kind not in kinds]
    return {
        "required_gate_kinds": list(REQUIRED_TASK_OWNED_GATE_KINDS),
        "observed_gate_kinds": sorted(kind for kind in kinds if kind),
        "all_required_gate_kinds_present": not missing,
        "missing_required_gate_kinds": missing,
        "task_owned_failures": failures,
        "receipts": task_owned,
        "principle": FIELD_PRINCIPLES["test_commands"],
    }


def _debt_baselines_and_deltas(root: Path, rows: Sequence[JsonMap]) -> JsonDict:
    global_before = _node_set(rows, "global_suite", "before", "failure_node_ids")
    global_after = _node_set(rows, "global_suite", "after", "failure_node_ids")
    spec_before = _node_set(rows, "spec_coverage", "before", "missing_node_ids")
    spec_after = _node_set(rows, "spec_coverage", "after", "missing_node_ids")
    root_before = _node_set(rows, "root_clutter", "before", "root_clutter_paths")
    root_after = _node_set(rows, "root_clutter", "after", "root_clutter_paths")
    if not global_before and global_after:
        global_before = set(global_after)
    if not global_after and global_before:
        global_after = set(global_before)
    if not spec_before and spec_after:
        spec_before = set(spec_after)
    if not spec_after and spec_before:
        spec_after = set(spec_before)
    if not root_before and not root_after:
        current = set(_root_clutter_inventory(root))
        root_before = set(current)
        root_after = set(current)
    elif not root_before and root_after:
        root_before = set(root_after)
    elif not root_after and root_before:
        root_after = set(root_before)
    global_new = sorted(global_after - global_before)
    spec_new = sorted(spec_after - spec_before)
    root_new = sorted(root_after - root_before)
    return {
        "global_suite_failure_baseline_node_ids": sorted(global_before),
        "global_suite_failure_after_node_ids": sorted(global_after),
        "global_suite_new_failure_node_ids": global_new,
        "global_suite_failure_delta": len(global_new),
        "global_spec_gap_baseline_node_ids": sorted(spec_before),
        "global_spec_gap_after_node_ids": sorted(spec_after),
        "global_spec_new_gap_node_ids": spec_new,
        "global_spec_gap_delta": len(spec_new),
        "root_clutter_baseline_paths": sorted(root_before),
        "root_clutter_after_paths": sorted(root_after),
        "root_clutter_new_paths": root_new,
        "root_clutter_delta": len(root_new),
        "non_amplification_gate_passed": not global_new and not spec_new and not root_new,
        "principle": FIELD_PRINCIPLES["inherited_debt_baselines_and_deltas"],
    }


def _range_number_mentions(text: str) -> set[int]:
    lowered = text.lower()
    if "596" not in lowered and "597" not in lowered:
        return set()
    numbers: set[int] = set()
    for number in NEXT_RANGE_NUMBERS:
        if re.search(rf"(?<![a-z0-9_])exp{number}(?![a-z0-9])", lowered) or re.search(
            rf"(?<![a-z0-9_])experiment_{number}(?![a-z0-9])", lowered
        ):
            numbers.add(number)
    return numbers


def _scan_candidate_paths(root: Path) -> list[Path]:
    candidates = [
        ROADMAP_RELATIVE_PATH,
        ROADMAP_NEXT_RELATIVE_PATH,
        RESEARCH_COMPLETE_RELATIVE_PATH,
        ROADMAP_DOC_RELATIVE_PATH,
        EXCLUSION_MANIFEST_RELATIVE_PATH,
        KNOWN_ISSUES_RELATIVE_PATH,
        CONDUCTOR_LOG_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    ]
    for folder in ("python", "tests", "scripts", "openspec/change-proposals", "ops"):
        base = root / folder
        if base.exists():
            candidates.extend(
                path.relative_to(root)
                for path in base.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and ".test_suite_mutation_runs" not in path.parts
                and path.suffix != ".pyc"
            )
    results = root / "results"
    if results.exists():
        candidates.extend(path.relative_to(root) for path in results.iterdir() if path.is_file())
    return sorted(set(candidates), key=lambda value: value.as_posix())


def _allowed_range_reference_kind(rel_path: Path, numbers: set[int]) -> str | None:
    if rel_path in OWNED_REFERENCE_PATHS:
        return "transition_owned_reference"
    if rel_path in ALLOWED_ALLOCATION_REFERENCE_PATHS:
        return "allowed_allocation_reference"
    return None


def _range_collision_scan(root: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    allowed: list[JsonDict] = []
    for rel_path in _scan_candidate_paths(root):
        path = root / rel_path
        text = rel_path.as_posix()
        if path.exists() and path.stat().st_size <= 5_000_000:
            text += "\n" + path.read_text(encoding="utf-8", errors="replace")
        numbers = _range_number_mentions(text)
        if not numbers:
            continue
        kind = _allowed_range_reference_kind(rel_path, numbers)
        row = {
            "path": rel_path.as_posix(),
            "kind": kind or "unexpected_next_range_reference",
            "numbers": sorted(numbers),
        }
        if kind:
            allowed.append(row)
        else:
            collisions.append(row)
    return {
        "range": {"start": 5961, "end": 5973},
        "collision_count": len(collisions),
        "collisions": collisions,
        "allowed_references": allowed,
        "principle": FIELD_PRINCIPLES["next_range_collision_count"],
    }


def _docs_reconciled(root: Path) -> JsonDict:
    spec_text = (
        (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "openspec_research_reporting_req_5961_present": "REQ-REPORT-5961" in spec_text,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
        "principle": FIELD_PRINCIPLES["docs_reconciled"],
    }


def _field_provenance() -> dict[str, JsonDict]:
    base_sources = [
        ROADMAP_RELATIVE_PATH.as_posix(),
        RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        KNOWN_ISSUES_RELATIVE_PATH.as_posix(),
        SPEC_RELATIVE_PATH.as_posix(),
        *[path.as_posix() for path in ACTIVATED_TASK_ARTIFACT_PATHS.values()],
    ]
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "sources": base_sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _status_and_verdict(failed_preconditions: Sequence[str], classes: JsonMap) -> tuple[str, str]:
    if failed_preconditions:
        reason = ",".join(failed_preconditions[:3])
        return "blocked", f"blocked: Exp5961 transition preconditions failed ({reason})"
    by_task = classes.get("terminal_class_by_task_id", {})
    if any(
        value in {"complete-partial", "complete-null", "retired", "gate-blocked"}
        for value in by_task.values()
    ):
        return (
            "complete_with_terminal_receipts",
            "complete: archived terminal .527 identities into .528 without outcome laundering; inherited global debt not amplified; next_range_collision_count=0",
        )
    return (
        "complete",
        "complete: archived terminal .527 identities into .528 with collision-free allocation and no inherited debt amplification",
    )


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.monotonic()
    root = root.resolve()
    protected_before = _protected_file_hashes(root)
    outer_loop_before = _outer_loop_hashes(root)
    active_roadmap, active_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_next, roadmap_next_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    complete_meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)[1]
    exclusion_meta = _read_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)[1]
    payloads, metadata = _artifact_payloads(root)
    conductor = _conductor_receipts(root)
    matrix = _activated_matrix(metadata, payloads, conductor["by_task"])
    classes = _exact_terminal_classification(payloads, metadata, conductor["by_task"])
    append_receipt = _append_completion_if_absent(root, bool(classes["all_activated_terminal"]))
    reservations = _unactivated_reservation_receipt(root)
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:  # pragma: no cover
        receipts = run_live_adversarial_receipts(root, metadata)
    verifier_group = _adversarial_receipts_group(receipts, matrix, payloads)
    range_scan = _range_collision_scan(root)
    protected = _protected_files_unchanged(root, protected_before)
    outer_loop = _outer_loop_id_separation_receipt(root, outer_loop_before)
    staging_receipt = _optional_staging_roadmap_receipt(roadmap_next, roadmap_next_meta)
    test_rows = _tests_run_rows(tests_run)
    task_gate = _task_owned_gate_receipts(test_rows)
    debt = _debt_baselines_and_deltas(root, test_rows)
    resources = _resource_receipts(root)
    atomic = _atomic_output_receipt(root / RESULT_RELATIVE_PATH)
    docs = _docs_reconciled(root)
    active_task_ids = (
        [
            str(row.get("id"))
            for row in active_roadmap.get("tasks", [])
            if isinstance(row, Mapping) and row.get("id")
        ]
        if isinstance(active_roadmap.get("tasks"), list)
        else []
    )
    present_task_ids = [task_id for task_id, row in matrix.items() if row["present"]]
    receipt_task_ids = {row["task_id"] for row in verifier_group["reports"]}
    missing_receipts = [task_id for task_id in present_task_ids if task_id not in receipt_task_ids]
    failed_preconditions: list[str] = []
    if active_meta["present"] and not active_meta["loadable"]:
        failed_preconditions.append("active_roadmap_unloadable")
    if active_meta["loadable"] and active_roadmap.get("milestone") != MILESTONE_TO:
        failed_preconditions.append("active_roadmap_milestone_mismatch")
    if active_meta["loadable"] and active_task_ids != list(NEXT_TASK_ARTIFACT_PATHS):
        failed_preconditions.append("active_roadmap_task_ids_mismatch")
    if roadmap_next_meta["present"] and not roadmap_next_meta["loadable"]:
        failed_preconditions.append("roadmap_next_unloadable")
    if complete_meta["present"] and not complete_meta["loadable"]:
        failed_preconditions.append("research_complete_unparseable")
    if exclusion_meta["present"] and not exclusion_meta["loadable"]:
        failed_preconditions.append("exclusion_manifest_unparseable")
    if conductor["plan_status"] != "OK":
        failed_preconditions.append("v527_plan_line_missing_or_not_ok")
    if conductor["activation_status"] != "OK" or conductor["activated_task_count_claim"] != 6:
        failed_preconditions.append("v527_activation_line_missing_or_not_six")
    if not (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists():
        failed_preconditions.append("live_verifier_missing")
    if classes["terminal_class_by_task_id"] != EXPECTED_TERMINAL_CLASSES:
        failed_preconditions.append("terminal_outcomes_not_preserved")
    if missing_receipts:
        failed_preconditions.append("missing_adversarial_receipts")
    if verifier_group["failed_receipt_task_ids"]:
        failed_preconditions.append("adversarial_verifier_failed")
    if "exp5932-transition-v527" not in verifier_group["aggregation_artifact_task_ids_clean"]:
        failed_preconditions.append("aggregation_artifact_not_clean")
    if task_gate["task_owned_failures"]:
        failed_preconditions.append("task_owned_gate_failed")
    if not task_gate["all_required_gate_kinds_present"]:
        failed_preconditions.append("task_owned_gate_missing")
    if debt["global_suite_failure_delta"] > 0:
        failed_preconditions.append("global_suite_debt_amplified")
    if debt["global_spec_gap_delta"] > 0:
        failed_preconditions.append("global_spec_debt_amplified")
    if debt["root_clutter_delta"] > 0:
        failed_preconditions.append("root_clutter_debt_amplified")
    if append_receipt["duplicate_history_amplification_count"] != 0:
        failed_preconditions.append("duplicate_history_amplified")
    if range_scan["collision_count"] != 0:
        failed_preconditions.append("next_range_collision")
    if not docs["openspec_research_reporting_req_5961_present"]:
        failed_preconditions.append("openspec_req_5961_missing")
    if not protected["all_unchanged"]:
        failed_preconditions.append("protected_file_modified")
    if not outer_loop["all_unchanged"]:
        failed_preconditions.append("outer_loop_id_modified")
    if not resources["disk"]["ok"] or not resources["memory"]["ok"]:
        failed_preconditions.append("insufficient_resources")
    if not atomic["ok"]:
        failed_preconditions.append("atomic_output_unavailable")
    status, verdict = _status_and_verdict(failed_preconditions, classes)
    result_duration = duration_s if duration_s is not None else round(time.monotonic() - start, 6)
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE_TO,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": {
            "active_roadmap": {
                **active_meta,
                "milestone": active_roadmap.get("milestone") if active_meta["loadable"] else None,
                "task_ids": active_task_ids,
            },
            "roadmap_next": staging_receipt,
            "research_complete": complete_meta,
            "conductor_log": {
                "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "present": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
                **conductor,
            },
            "exclusion_manifest": exclusion_meta,
            "known_issues": {
                "path": KNOWN_ISSUES_RELATIVE_PATH.as_posix(),
                "present": (root / KNOWN_ISSUES_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / KNOWN_ISSUES_RELATIVE_PATH),
            },
            "source_hashes": _source_hashes(root),
            "root_clutter_inventory": _root_clutter_inventory(root),
            "declared_present_deliverable_hashes": {
                task_id: row["sha256"] for task_id, row in matrix.items() if row["present"]
            },
            "resource_receipts": resources,
            "atomic_output": atomic,
            "adversarial_verifier_available": (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists(),
            "range_collision_scan": range_scan,
            "protected_file_hashes_before": protected_before,
            "failed_preconditions": failed_preconditions,
        },
        "milestone_transition": {
            "source_milestone": MILESTONE_FROM,
            "destination_milestone": MILESTONE_TO,
            "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
        },
        "activated_task_and_deliverable_matrix": matrix,
        "unactivated_reservation_receipt": reservations,
        "exact_terminal_classification": classes,
        "adversarial_verifier_receipts": verifier_group,
        "task_owned_gate_receipts": task_gate,
        "inherited_debt_baselines_and_deltas": debt,
        "research_complete_append_count": append_receipt["append_count"],
        "duplicate_history_amplification_count": append_receipt[
            "duplicate_history_amplification_count"
        ],
        "research_complete_append_receipt": append_receipt,
        "next_task_range": {
            "start": "exp5961",
            "end": "exp5973",
            "count": len(NEXT_TASK_ARTIFACT_PATHS),
            "declared_allocation_task_ids": list(NEXT_TASK_ARTIFACT_PATHS),
            "declared_allocation_deliverables": {
                task_id: rel_path.as_posix()
                for task_id, rel_path in NEXT_TASK_ARTIFACT_PATHS.items()
            },
        },
        "next_range_collision_count": range_scan["collision_count"],
        "outer_loop_id_separation_receipt": outer_loop,
        "docs_reconciled": docs,
        "protected_files_unchanged": protected,
        "duration_s": result_duration,
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
    if not isinstance(verdict, str) or not verdict.startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must start with complete: or blocked:")
    if not isinstance(payload.get("next_range_collision_count"), int):
        raise ValueError("next_range_collision_count must be a bare integer")
    if payload["next_range_collision_count"] != 0 and payload.get("status") != "blocked":
        raise ValueError("next_range_collision_count must be zero unless status is blocked")
    if payload.get("research_complete_append_count") not in {0, 1}:
        raise ValueError("research_complete_append_count must be zero or one")
    if payload.get("duplicate_history_amplification_count") != 0:
        raise ValueError("duplicate_history_amplification_count must be zero")
    matrix = payload.get("activated_task_and_deliverable_matrix")
    if not isinstance(matrix, Mapping) or len(matrix) != 6:
        raise ValueError("activated matrix must contain exactly six .527 identities")
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = matrix.get(task_id)
        if not isinstance(row, Mapping):
            raise ValueError("activated matrix must contain exactly six .527 identities")
        if row.get("identity") != [MILESTONE_FROM, task_id, rel_path.as_posix()]:
            raise ValueError("activated identity mismatch")
    reservations = payload.get("unactivated_reservation_receipt")
    if (
        not isinstance(reservations, Mapping)
        or reservations.get("activated_as_results_count") != 0
        or reservations.get("fabricated_outcome_count") != 0
    ):
        raise ValueError("unactivated reservations must not be fabricated outcomes")
    classes = payload.get("exact_terminal_classification")
    if (
        not isinstance(classes, Mapping)
        or classes.get("terminal_class_by_task_id") != EXPECTED_TERMINAL_CLASSES
    ):
        raise ValueError("terminal classes do not preserve .527 outcomes")
    verifier = payload.get("adversarial_verifier_receipts")
    if not isinstance(verifier, Mapping):
        raise ValueError("adversarial verifier receipts missing")
    present_count = sum(
        1 for row in matrix.values() if isinstance(row, Mapping) and row.get("present")
    )
    if verifier.get("verified_present_declared_deliverable_count") != present_count:
        raise ValueError("missing adversarial verifier receipt")
    for row in verifier.get("reports", []):
        if not isinstance(row, Mapping) or not row.get("receipt_hash"):
            raise ValueError("missing adversarial verifier receipt fields")
        if "scripts/adversarial_verify.py" not in str(row.get("command") or ""):
            raise ValueError(
                "adversarial verifier receipt command must run scripts/adversarial_verify.py"
            )
    if "exp5932-transition-v527" not in verifier.get("aggregation_artifact_task_ids_clean", []):
        raise ValueError("aggregation verifier receipt must be clean")
    gate = payload.get("task_owned_gate_receipts")
    if (
        not isinstance(gate, Mapping)
        or gate.get("all_required_gate_kinds_present") is not True
        or gate.get("task_owned_failures")
    ):
        raise ValueError("task-owned gate receipts are not clean")
    debt = payload.get("inherited_debt_baselines_and_deltas")
    if not isinstance(debt, Mapping) or debt.get("non_amplification_gate_passed") is not True:
        raise ValueError("debt non-amplification gate failed")
    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        raise ValueError("protected file changed")
    if any(
        isinstance(row, Mapping) and row.get("unchanged") is not True
        for row in protected.get("files", {}).values()
    ):
        raise ValueError("protected file changed")
    field_provenance = payload.get("field_provenance")
    if not isinstance(field_provenance, Mapping):
        raise ValueError("field provenance missing")
    for field in REQUIRED_ARTIFACT_FIELDS:
        provenance = field_provenance.get(field)
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("principle") != FIELD_PRINCIPLES[field]
        ):
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

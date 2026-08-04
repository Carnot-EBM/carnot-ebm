"""Exp6100 transition receipt from terminal milestone .528 into .529.

Spec refs: REQ-REPORT-6100,
SCENARIO-REPORT-6100-ACTIVATED-MATRIX,
SCENARIO-REPORT-6100-TERMINAL-CLASSES,
SCENARIO-REPORT-6100-MISSING-AND-GATE-BLOCKS,
SCENARIO-REPORT-6100-DUPLICATE-DEBT-AND-VERIFIER,
SCENARIO-REPORT-6100-RANGE-COLLISION,
SCENARIO-REPORT-6100-SCHEMA.

This task is a ledger handoff. It reads checked-in evidence and conductor
receipts, then records what actually happened without letting nearby filenames,
completion-history summaries, or gate-check artifacts strengthen a missing or
preemptively skipped experiment.
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
RESULT_RELATIVE_PATH = Path("results/experiment_6100_transition_v529.json")

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

EXPERIMENT = "experiment_6100_transition_v529"
EXPERIMENT_ID = "exp6100-transition-v529"
MILESTONE_FROM = "2026.07.528"
MILESTONE_TO = "2026.08.529"
MILESTONE_FROM_TITLE = (
    "Discriminative Exact-Atom Acquisition, Delayed-Commit Continuous Learning, "
    "and ARC Budget/Convention Generalization"
)
MILESTONE_TO_TITLE = (
    "Calibrated Phase-D Candidate Headroom, Internal-State Verification, "
    "and Reduced-Order Continuous Learning"
)
RUN_DATE = "20260804"
RANDOM_SEED = 6100
SCHEMA = "carnot.experiment_6100.transition_v529.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-6100",
    "SCENARIO-REPORT-6100-ACTIVATED-MATRIX",
    "SCENARIO-REPORT-6100-TERMINAL-CLASSES",
    "SCENARIO-REPORT-6100-MISSING-AND-GATE-BLOCKS",
    "SCENARIO-REPORT-6100-DUPLICATE-DEBT-AND-VERIFIER",
    "SCENARIO-REPORT-6100-RANGE-COLLISION",
    "SCENARIO-REPORT-6100-SCHEMA",
)

ACTIVATED_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5961-transition-v528": Path("results/experiment_5961_transition_v528.json"),
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

ACTIVATED_TASK_TITLES: dict[str, str] = {
    "exp5961-transition-v528": "Exact terminal-boundary handoff from .527 into .528",
    "exp5962-v528-source-delta-ingestion": "Dated evidence refresh after the V528 planner marker",
    "exp5963-exact-atom-pair-fixture": "Hardness-controlled exact context/atom compatibility fixture",
    "exp5964-sota-atom-compatibility-corpus": (
        "Gated on Exp5963 ready: all-three-model GGUF context/atom compatibility corpus"
    ),
    "exp5965-portable-atom-energy-ranker": (
        "Gated on Exp5964 ready: portable exact-atom compatibility energy"
    ),
    "exp5966-discriminative-constraint-acquisition": (
        "Gated on Exp5965 ready: end-to-end discriminative exact constraint acquisition"
    ),
    "exp5967-delayed-commit-memory-fixture": "Delayed-commit transactional memory fixture over ABI v2",
    "exp5968-delayed-commit-csl-prospective": (
        "Gated on Exp5967 ready: prospective delayed-commit continuous self-learning A/B"
    ),
    "exp5969-csl-poison-drift-abi-audit": (
        "Gated on Exp5968 ready: poison, drift, rollback, retention, and ABI audit"
    ),
    "exp5970-arc-strip-swap-sentinel": "ARC row/column strip-swap convention sentinel",
    "exp5971-arc-strip-swap-battery": (
        "Gated on Exp5970 ready: full ARC strip-swap convention-transfer battery"
    ),
    "exp5972-arc-llm-on-budget2000-feasibility": (
        "Live ARC flagship-LLM budget-2000 wall-clock feasibility"
    ),
    "exp5973-v528-capstone-reconciliation": (
        "Branch-independent .528 capstone and exact reconciliation"
    ),
}

CONDUCTOR_MATCH_MARKERS: dict[str, str] = {
    "exp5961-transition-v528": "Exact terminal-boundary handoff from .527",
    "exp5962-v528-source-delta-ingestion": "Dated evidence refresh after the V528",
    "exp5963-exact-atom-pair-fixture": "Hardness-controlled exact context/atom compatibili",
    "exp5964-sota-atom-compatibility-corpus": "Gated on Exp5963 ready",
    "exp5965-portable-atom-energy-ranker": "Gated on Exp5964 ready",
    "exp5966-discriminative-constraint-acquisition": "Gated on Exp5965 ready",
    "exp5967-delayed-commit-memory-fixture": "Delayed-commit transactional memory fixture",
    "exp5968-delayed-commit-csl-prospective": "Gated on Exp5967 ready",
    "exp5969-csl-poison-drift-abi-audit": "Gated on Exp5968 ready",
    "exp5970-arc-strip-swap-sentinel": "ARC row/column strip-swap convention sentinel",
    "exp5971-arc-strip-swap-battery": "Gated on Exp5970 ready",
    "exp5972-arc-llm-on-budget2000-feasibility": "Live ARC flagship-LLM budget-2000",
    "exp5973-v528-capstone-reconciliation": "Branch-independent .528 capstone",
}

EXPECTED_TERMINAL_CLASSES: dict[str, str] = {
    "exp5961-transition-v528": "missing",
    "exp5962-v528-source-delta-ingestion": "complete-null",
    "exp5963-exact-atom-pair-fixture": "complete-ready",
    "exp5964-sota-atom-compatibility-corpus": "blocked-precondition",
    "exp5965-portable-atom-energy-ranker": "conductor-gate-blocked",
    "exp5966-discriminative-constraint-acquisition": "conductor-gate-blocked",
    "exp5967-delayed-commit-memory-fixture": "complete-ready",
    "exp5968-delayed-commit-csl-prospective": "complete-ready",
    "exp5969-csl-poison-drift-abi-audit": "complete-ready",
    "exp5970-arc-strip-swap-sentinel": "complete-ready",
    "exp5971-arc-strip-swap-battery": "complete-null",
    "exp5972-arc-llm-on-budget2000-feasibility": "feasible-only",
    "exp5973-v528-capstone-reconciliation": "complete-with-blocks",
}

EXPECTED_TERMINAL_SUBCLASSES: dict[str, str] = {
    "exp5961-transition-v528": "declared-artifact-missing-after-three-wall-clock-attempts",
    "exp5962-v528-source-delta-ingestion": "source-null",
    "exp5963-exact-atom-pair-fixture": "ready-fixture",
    "exp5964-sota-atom-compatibility-corpus": "insufficient-free-vram",
    "exp5965-portable-atom-energy-ranker": "preemptive-skip-upstream-retired",
    "exp5966-discriminative-constraint-acquisition": "gate-check-blocked-on-exp5965",
    "exp5967-delayed-commit-memory-fixture": "ready-fixture",
    "exp5968-delayed-commit-csl-prospective": "ready-prospective-csl",
    "exp5969-csl-poison-drift-abi-audit": "ready-safety-abi-audit",
    "exp5970-arc-strip-swap-sentinel": "ready-sentinel",
    "exp5971-arc-strip-swap-battery": "public-game-null",
    "exp5972-arc-llm-on-budget2000-feasibility": "wall-clock-feasibility-envelope",
    "exp5973-v528-capstone-reconciliation": "capstone-with-blocks",
}

ACTIVE_V529_TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp6100-transition-v529": RESULT_RELATIVE_PATH,
    "exp6101-v529-source-delta-ingestion": Path(
        "results/experiment_6101_v529_source_delta_ingestion.json"
    ),
    "exp6102-sota-atom-corpus-vram-recovery": Path(
        "results/experiment_6102_sota_atom_corpus_vram_recovery.json"
    ),
    "exp6103-phase-d-difficulty-ladder-fixture": Path(
        "results/experiment_6103_phase_d_difficulty_ladder_fixture.json"
    ),
}

NEXT_RANGE_NUMBERS = range(6100, 6112)
GATE_BLOCK_TASK_IDS = {
    "exp5965-portable-atom-energy-ranker",
    "exp5966-discriminative-constraint-acquisition",
}

PROTECTED_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
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
    Path("python/carnot/experiment_6100_transition_v529.py"),
    Path("tests/python/test_experiment_6100_transition_v529.py"),
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
    "duplicate_history",
    "exact_path_hash",
    "adversarial_verifier",
    "exclusion_manifest",
    "range_collision",
    "debt_delta",
    "protected_file",
    "applicable_e2e",
    "no_new_root_clutter",
    "task_owned_spec_coverage",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_transition",
    "activated_task_and_deliverable_matrix",
    "exact_terminal_classification",
    "missing_artifact_receipt",
    "conductor_gate_block_receipts",
    "adversarial_verifier_receipts",
    "inherited_debt_baselines_and_deltas",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
    "next_task_range",
    "next_range_collision_count",
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
    "status": "state the actual transition and every prerequisite checked.",
    "preconditions_checked": "state the actual transition and every prerequisite checked.",
    "milestone_transition": "only the thirteen activated identities and declared paths define `.528`.",
    "activated_task_and_deliverable_matrix": (
        "only the thirteen activated identities and declared paths define `.528`."
    ),
    "exact_terminal_classification": (
        "only the thirteen activated identities and declared paths define `.528`."
    ),
    "missing_artifact_receipt": (
        "absence and preemptive skips remain distinct terminal evidence classes."
    ),
    "conductor_gate_block_receipts": (
        "absence and preemptive skips remain distinct terminal evidence classes."
    ),
    "adversarial_verifier_receipts": (
        "verify present evidence freshly without laundering unrelated debt."
    ),
    "inherited_debt_baselines_and_deltas": (
        "verify present evidence freshly without laundering unrelated debt."
    ),
    "research_complete_append_count": "append at most once and amplify no history.",
    "duplicate_history_amplification_count": "append at most once and amplify no history.",
    "next_task_range": "bare zero collisions authorize Exp6100-Exp6111.",
    "next_range_collision_count": "bare zero collisions authorize Exp6100-Exp6111.",
    "docs_reconciled": "update only owned internal ledgers.",
    "protected_files_unchanged": "update only owned internal ledgers.",
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
    ".venv/bin/pytest tests/python/test_experiment_6100_transition_v529.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6100_transition_v529.py -m pytest tests/python/test_experiment_6100_transition_v529.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6100_transition_v529.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json <present .528 declared deliverables>",
    ".venv/bin/python scripts/check_exclusion_manifest.py 6100",
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


def _load_yaml_any(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _history_blocks(root: Path) -> list[JsonMap]:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    try:
        payload = _load_yaml_any(path)
    except yaml.YAMLError:
        return []
    blocks = payload.get("milestones") if isinstance(payload, Mapping) else payload
    return [block for block in blocks if isinstance(block, Mapping)] if isinstance(blocks, list) else []


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


def _completion_block_data() -> JsonDict:
    return {
        "id": MILESTONE_FROM,
        "title": MILESTONE_FROM_TITLE,
        "doc": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-04",
        "finding": "Terminal outcomes preserved by Exp6100 transition artifact.",
        "tasks": [
            {
                "id": task_id,
                "title": ACTIVATED_TASK_TITLES[task_id],
                "deliverable": rel_path.as_posix(),
                "result": EXPECTED_TERMINAL_CLASSES[task_id],
            }
            for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items()
        ],
    }


def _write_history_blocks(path: Path, original: Any, blocks: list[JsonMap]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(original, Mapping):
        updated = dict(original)
        updated["milestones"] = blocks
        path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(blocks, sort_keys=False), encoding="utf-8")


def _append_completion_if_absent(root: Path, terminal: bool) -> JsonDict:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    before_meta = _read_yaml_mapping(path)[1]
    before_blocks = _history_blocks(root)
    before_duplicate_count = _duplicate_history_block_count(before_blocks)
    before_milestone_count = sum(1 for block in before_blocks if block.get("id") == MILESTONE_FROM)
    before_signatures = {
        _task_signature(block) for block in before_blocks if block.get("id") == MILESTONE_FROM
    }
    base = {
        "before_sha256": before_meta["sha256"],
        "before_duplicate_history_count": before_duplicate_count,
        "before_milestone_block_count": before_milestone_count,
        "before_canonical_signature_count": len(before_signatures),
    }
    if not terminal:
        return {
            **base,
            "append_count": 0,
            "appended": False,
            "reason": "nonterminal_identity_present",
            "after_sha256": before_meta["sha256"],
            "after_duplicate_history_count": before_duplicate_count,
            "after_milestone_block_count": before_milestone_count,
            "after_canonical_signature_count": len(before_signatures),
            "duplicate_history_amplification_count": 0,
        }
    if before_milestone_count:
        return {
            **base,
            "append_count": 0,
            "appended": False,
            "reason": "exact_milestone_block_present",
            "after_sha256": before_meta["sha256"],
            "after_duplicate_history_count": before_duplicate_count,
            "after_milestone_block_count": before_milestone_count,
            "after_canonical_signature_count": len(before_signatures),
            "duplicate_history_amplification_count": 0,
        }
    try:
        original = _load_yaml_any(path)
    except yaml.YAMLError:
        original = {}
    after_blocks = list(before_blocks)
    after_blocks.append(_completion_block_data())
    _write_history_blocks(path, original, after_blocks)
    written_blocks = _history_blocks(root)
    after_duplicate_count = _duplicate_history_block_count(written_blocks)
    after_milestone_count = sum(1 for block in written_blocks if block.get("id") == MILESTONE_FROM)
    after_signatures = {
        _task_signature(block) for block in written_blocks if block.get("id") == MILESTONE_FROM
    }
    return {
        **base,
        "append_count": 1,
        "appended": True,
        "reason": "exact_milestone_block_absent",
        "after_sha256": path_sha256(path),
        "after_duplicate_history_count": after_duplicate_count,
        "after_milestone_block_count": after_milestone_count,
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
        return [dict(flag) for flag in flags if isinstance(flag, Mapping)] if isinstance(flags, list) else []
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
            raw = reports[0].get("max_severity")
            return int(raw) if raw is not None else -1
    raw = receipt.get("max_severity")
    return int(raw) if raw is not None else -1


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
    plan_lines = [line for line in lines if "Plan milestone 2026.07.528" in line]
    activation_lines = [line for line in lines if "Milestone 2026.07.528 activated" in line]
    return {
        "plan_line": plan_lines[-1] if plan_lines else "",
        "plan_status": _conductor_status_from_line(plan_lines[-1]) if plan_lines else "",
        "activation_line": activation_lines[-1] if activation_lines else "",
        "activation_status": _conductor_status_from_line(activation_lines[-1]) if activation_lines else "",
        "activated_task_count_claim": 13
        if activation_lines and "13 tasks queued" in activation_lines[-1]
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
    if task_id in GATE_BLOCK_TASK_IDS:
        if (
            conductor.get("latest_status") == "GATE_BLOCK"
            or not meta.get("present")
            or payload.get("schema") == "blocked_gate_check_v1"
            or payload.get("gates_evaluated")
        ):
            return "conductor-gate-blocked"
    if not meta.get("present"):
        return "missing"
    if status == "complete_with_blocks" or verdict.startswith("complete_with_blocks:"):
        return "complete-with-blocks"
    if status == "complete_feasible" or verdict.startswith("complete_feasible:"):
        return "feasible-only"
    if status == "complete_null" or verdict.startswith("complete_null:"):
        return "complete-null"
    if status == "complete_ready" or verdict.startswith("complete_ready:"):
        return "complete-ready"
    if status == "blocked" or verdict.startswith("blocked:"):
        return "blocked-precondition"
    if status.startswith("complete") or verdict.startswith("complete:"):
        return "complete"
    return "missing"


def _exact_terminal_classification(
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
) -> JsonDict:
    by_task: dict[str, str] = {}
    by_subclass: dict[str, str] = {}
    by_class: dict[str, list[str]] = {}
    for task_id in ACTIVATED_TASK_ARTIFACT_PATHS:
        terminal = _classify_task(task_id, payloads[task_id], metadata[task_id], conductor[task_id])
        by_task[task_id] = terminal
        by_subclass[task_id] = EXPECTED_TERMINAL_SUBCLASSES.get(task_id, terminal)
        by_class.setdefault(terminal, []).append(task_id)
    nonterminal = [task_id for task_id, terminal in by_task.items() if terminal == "missing"]
    tolerated_missing = ["exp5961-transition-v528"]
    return {
        "terminal_class_by_task_id": by_task,
        "terminal_subclass_by_task_id": by_subclass,
        "task_ids_by_terminal_class": by_class,
        "expected_terminal_class_by_task_id": dict(EXPECTED_TERMINAL_CLASSES),
        "expected_terminal_subclass_by_task_id": dict(EXPECTED_TERMINAL_SUBCLASSES),
        "disjoint_terminal_class_count": len(by_task),
        "all_activated_terminal": nonterminal == tolerated_missing
        and by_task == EXPECTED_TERMINAL_CLASSES,
        "nonterminal_task_ids": nonterminal,
        "classification_source": "exact_declared_deliverables_plus_conductor_gate_receipts",
        "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
    }


def _activated_matrix(
    metadata: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    conductor: Mapping[str, JsonMap],
    classes: Mapping[str, str],
) -> dict[str, JsonDict]:
    matrix: dict[str, JsonDict] = {}
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        payload = payloads[task_id]
        meta = metadata[task_id]
        terminal = classes[task_id]
        if terminal == "missing":
            evidence_source = "declared_absence"
        elif terminal == "conductor-gate-blocked":
            evidence_source = "conductor_gate_block_without_execution"
        else:
            evidence_source = "declared_deliverable_path"
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
            "terminal_class": terminal,
            "terminal_evidence_source": evidence_source,
        }
    return matrix


def _task_number(task_id: str) -> int | None:
    match = re.search(r"exp(\d{4})", task_id, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _same_number_aliases(root: Path, task_id: str, declared_path: Path) -> list[str]:
    number = _task_number(task_id)
    results = root / "results"
    if number is None or not results.exists():
        return []
    aliases: list[str] = []
    for path in results.glob(f"*{number}*"):
        rel_path = path.relative_to(root)
        if path.is_file() and rel_path != declared_path:
            aliases.append(rel_path.as_posix())
    return sorted(aliases)


def _missing_artifact_receipt(root: Path, matrix: Mapping[str, JsonMap]) -> JsonDict:
    missing = [
        task_id for task_id, row in matrix.items() if row.get("terminal_class") == "missing"
    ]
    return {
        "missing_declared_artifact_task_ids": missing,
        "missing_declared_artifact_paths": [
            ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix() for task_id in missing
        ],
        "missing_artifact_count": len(missing),
        "missing_artifact_is_success": False,
        "same_number_aliases_ignored": {
            task_id: _same_number_aliases(root, task_id, ACTIVATED_TASK_ARTIFACT_PATHS[task_id])
            for task_id in missing
        },
        "principle": FIELD_PRINCIPLES["missing_artifact_receipt"],
    }


def _conductor_gate_block_receipts(matrix: Mapping[str, JsonMap]) -> JsonDict:
    gate_blocked = [
        task_id
        for task_id, row in matrix.items()
        if row.get("terminal_class") == "conductor-gate-blocked"
    ]
    return {
        "gate_blocked_task_ids": gate_blocked,
        "gate_blocked_declared_artifact_paths": [
            ACTIVATED_TASK_ARTIFACT_PATHS[task_id].as_posix() for task_id in gate_blocked
        ],
        "receipts_by_task_id": {
            task_id: {
                "declared_deliverable": matrix[task_id]["declared_deliverable"],
                "present": bool(matrix[task_id]["present"]),
                "conductor": matrix[task_id]["conductor"],
                "not_executed_by_transition": True,
                "artifact_present_but_gate_class_preserved": bool(matrix[task_id]["present"]),
            }
            for task_id in gate_blocked
        },
        "executed_experiment_claim_count": 0,
        "principle": FIELD_PRINCIPLES["conductor_gate_block_receipts"],
    }


def _adversarial_receipts_group(
    receipts: Mapping[str, JsonMap],
    matrix: Mapping[str, JsonMap],
) -> JsonDict:
    reports: list[JsonDict] = []
    failed_receipt_task_ids: list[str] = []
    warning_receipt_task_ids: list[str] = []
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
        if receipt.get("exit_code") != 0 and max_severity >= 2:
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
    if "610" not in lowered and "611" not in lowered:
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
    for folder in ("python", "tests", "scripts", "openspec/change-proposals", "openspec/capabilities", "ops"):
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
        "range": {"start": 6100, "end": 6111},
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
        "openspec_research_reporting_req_6100_present": "REQ-REPORT-6100" in spec_text,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "ops_conductor_log_deferred_to_conductor_stop_rule": True,
        "principle": FIELD_PRINCIPLES["docs_reconciled"],
    }


def _field_provenance() -> dict[str, JsonDict]:
    base_sources = [
        ROADMAP_RELATIVE_PATH.as_posix(),
        ROADMAP_DOC_RELATIVE_PATH.as_posix(),
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


def _status_and_verdict(failed_preconditions: Sequence[str]) -> tuple[str, str]:
    if failed_preconditions:
        reason = ",".join(failed_preconditions[:3])
        return "blocked", f"blocked: Exp6100 transition preconditions failed ({reason})"
    return (
        "complete_with_terminal_receipts",
        "complete: archived terminal .528 identities into .529 without outcome laundering; missing artifact and conductor gate blocks preserved; next_range_collision_count=0",
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
    active_roadmap, active_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    roadmap_next, roadmap_next_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    roadmap_doc_present = (root / ROADMAP_DOC_RELATIVE_PATH).exists()
    complete_meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)[1]
    exclusion_meta = _read_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)[1]
    payloads, metadata = _artifact_payloads(root)
    conductor = _conductor_receipts(root)
    classes = _exact_terminal_classification(payloads, metadata, conductor["by_task"])
    matrix = _activated_matrix(
        metadata,
        payloads,
        conductor["by_task"],
        classes["terminal_class_by_task_id"],
    )
    append_receipt = _append_completion_if_absent(root, bool(classes["all_activated_terminal"]))
    receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    if adversarial_receipts is None:  # pragma: no cover
        receipts = run_live_adversarial_receipts(root, metadata)
    verifier_group = _adversarial_receipts_group(receipts, matrix)
    range_scan = _range_collision_scan(root)
    protected = _protected_files_unchanged(root, protected_before)
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
    if active_meta["loadable"] and active_task_ids != list(ACTIVE_V529_TASK_ARTIFACT_PATHS):
        failed_preconditions.append("active_roadmap_task_ids_mismatch")
    if roadmap_next_meta["present"] and not roadmap_next_meta["loadable"]:
        failed_preconditions.append("roadmap_next_unloadable")
    if not roadmap_doc_present:
        failed_preconditions.append("vnext_proposal_missing")
    if complete_meta["present"] and not complete_meta["loadable"]:
        failed_preconditions.append("research_complete_unparseable")
    if exclusion_meta["present"] and not exclusion_meta["loadable"]:
        failed_preconditions.append("exclusion_manifest_unparseable")
    if conductor["activation_status"] != "OK" or conductor["activated_task_count_claim"] != 13:
        failed_preconditions.append("v528_activation_line_missing_or_not_thirteen")
    if not (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists():
        failed_preconditions.append("live_verifier_missing")
    if classes["terminal_class_by_task_id"] != EXPECTED_TERMINAL_CLASSES:
        failed_preconditions.append("terminal_outcomes_not_preserved")
    if missing_receipts:
        failed_preconditions.append("missing_adversarial_receipts")
    if verifier_group["failed_receipt_task_ids"]:
        failed_preconditions.append("adversarial_verifier_failed")
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
    if not docs["openspec_research_reporting_req_6100_present"]:
        failed_preconditions.append("openspec_req_6100_missing")
    if not protected["all_unchanged"]:
        failed_preconditions.append("protected_file_modified")
    if not resources["disk"]["ok"] or not resources["memory"]["ok"]:
        failed_preconditions.append("insufficient_resources")
    if not atomic["ok"]:
        failed_preconditions.append("atomic_output_unavailable")
    status, verdict = _status_and_verdict(failed_preconditions)
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
            "vnext_proposal": {
                "path": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                "present": roadmap_doc_present,
                "sha256": path_sha256(root / ROADMAP_DOC_RELATIVE_PATH),
            },
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
        "exact_terminal_classification": classes,
        "missing_artifact_receipt": _missing_artifact_receipt(root, matrix),
        "conductor_gate_block_receipts": _conductor_gate_block_receipts(matrix),
        "adversarial_verifier_receipts": verifier_group,
        "task_owned_gate_receipts": task_gate,
        "inherited_debt_baselines_and_deltas": debt,
        "research_complete_append_count": append_receipt["append_count"],
        "duplicate_history_amplification_count": append_receipt[
            "duplicate_history_amplification_count"
        ],
        "research_complete_append_receipt": append_receipt,
        "next_task_range": {
            "start": "exp6100",
            "end": "exp6111",
            "reserved_count": len(list(NEXT_RANGE_NUMBERS)),
            "active_allocation_task_count": len(ACTIVE_V529_TASK_ARTIFACT_PATHS),
            "active_declared_allocation_task_ids": list(ACTIVE_V529_TASK_ARTIFACT_PATHS),
            "active_declared_allocation_deliverables": {
                task_id: rel_path.as_posix()
                for task_id, rel_path in ACTIVE_V529_TASK_ARTIFACT_PATHS.items()
            },
        },
        "next_range_collision_count": range_scan["collision_count"],
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
    if not isinstance(matrix, Mapping) or len(matrix) != 13:
        raise ValueError("activated matrix must contain exactly thirteen .528 identities")
    for task_id, rel_path in ACTIVATED_TASK_ARTIFACT_PATHS.items():
        row = matrix.get(task_id)
        if not isinstance(row, Mapping):
            raise ValueError("activated matrix must contain exactly thirteen .528 identities")
        if row.get("identity") != [MILESTONE_FROM, task_id, rel_path.as_posix()]:
            raise ValueError("activated identity mismatch")
    classes = payload.get("exact_terminal_classification")
    if (
        not isinstance(classes, Mapping)
        or classes.get("terminal_class_by_task_id") != EXPECTED_TERMINAL_CLASSES
    ):
        raise ValueError("terminal classes do not preserve .528 outcomes")
    missing_receipt = payload.get("missing_artifact_receipt")
    if (
        not isinstance(missing_receipt, Mapping)
        or missing_receipt.get("missing_declared_artifact_task_ids") != ["exp5961-transition-v528"]
        or missing_receipt.get("missing_artifact_is_success") is not False
    ):
        raise ValueError("missing artifact receipt must preserve Exp5961 absence")
    gate_receipt = payload.get("conductor_gate_block_receipts")
    if (
        not isinstance(gate_receipt, Mapping)
        or gate_receipt.get("gate_blocked_task_ids")
        != [
            "exp5965-portable-atom-energy-ranker",
            "exp5966-discriminative-constraint-acquisition",
        ]
        or gate_receipt.get("executed_experiment_claim_count") != 0
    ):
        raise ValueError("gate block receipts must preserve conductor skips")
    verifier = payload.get("adversarial_verifier_receipts")
    if not isinstance(verifier, Mapping):
        raise ValueError("adversarial verifier receipts missing")
    present_count = sum(
        1 for row in matrix.values() if isinstance(row, Mapping) and row.get("present")
    )
    if verifier.get("verified_present_declared_deliverable_count") != present_count:
        raise ValueError("adversarial verifier receipts do not match present declared artifacts")
    for row in verifier.get("reports", []):
        if not isinstance(row, Mapping) or not row.get("receipt_hash"):
            raise ValueError("adversarial verifier receipt missing hash")
        if "scripts/adversarial_verify.py" not in str(row.get("command") or ""):
            raise ValueError("adversarial verifier receipt command mismatch")
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


def _load_tests_run(path: Path | None) -> list[JsonDict] | None:  # pragma: no cover
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--tests-run-json must contain a JSON list")
    return [dict(row) for row in payload if isinstance(row, Mapping)]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report(args.root, tests_run=_load_tests_run(args.tests_run_json))
    output = args.output or args.root / RESULT_RELATIVE_PATH
    write_json(output, report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

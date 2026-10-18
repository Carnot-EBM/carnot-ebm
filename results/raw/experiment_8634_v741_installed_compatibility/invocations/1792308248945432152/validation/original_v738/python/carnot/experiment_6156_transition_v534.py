"""Exp6156 transition receipt from terminal milestone .533 into .534.

Spec refs: REQ-REPORT-6156,
SCENARIO-REPORT-6156-ACTIVATED-MATRIX,
SCENARIO-REPORT-6156-TERMINAL-QUARANTINE-AND-SKIP,
SCENARIO-REPORT-6156-DUPLICATE-ACTIVATION,
SCENARIO-REPORT-6156-RANGE-COLLISION,
SCENARIO-REPORT-6156-SCHEMA.

This task is a ledger transition. It archives only declared upstream artifact
paths, records quarantine overlays separately from underlying scientific
outcomes, and activates the next roadmap only when the staged file exists or
the conductor already consumed it.
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
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6156_transition_v534.json")

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_NEXT_RELATIVE_PATH = Path("research-roadmap-next.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
EVIDENCE_INDEX_RELATIVE_PATH = Path("scripts/evidence_index_collision_preflight.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXPERIMENT = "experiment_6156_transition_v534"
EXPERIMENT_ID = "exp6156-transition-v534"
MILESTONE_FROM = "2026.08.533"
MILESTONE_TO = "2026.08.534"
MILESTONE_FROM_TITLE = (
    "Task-Aware Energy Calibration, Certified Continuous Learning, "
    "and Stochastic Program Compilation"
)
MILESTONE_TO_TITLE = (
    "Decision-Calibrated Energy, Prospective Strategy Learning, "
    "and Nontrivial Stochastic Compilation"
)
RUN_DATE = "20260806"
RANDOM_SEED = 6156
SCHEMA = "carnot.experiment_6156.transition_v534.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

ACTIVATED_TASKS: tuple[tuple[str, str, Path], ...] = (
    (
        "exp6142-transition-v533",
        "Exact terminal-boundary handoff from .532 into .533",
        Path("results/experiment_6142_transition_v533.json"),
    ),
    (
        "exp6143-test-artifact-isolation",
        "Tracked-result test artifact isolation and quarantine preservation",
        Path("results/experiment_6143_test_artifact_isolation.json"),
    ),
    (
        "exp6144-v533-source-delta-ingestion",
        "Reliable dated evidence refresh after the V533 planner marker",
        Path("results/experiment_6144_v533_source_delta_ingestion.json"),
    ),
    (
        "exp6145-constraint-shift-stream",
        "Exact chronological constraint-event stream with held family shifts",
        Path("results/experiment_6145_constraint_shift_stream.json"),
    ),
    (
        "exp6146-sota-constraint-event-corpus",
        "Gated on Exp6145 readiness: flagship-GGUF chronological event corpus",
        Path("results/experiment_6146_sota_constraint_event_corpus.json"),
    ),
    (
        "exp6147-task-aware-energy-calibration",
        "Gated on Exp6146 corpus readiness: TOOD-style task-aware energy calibration",
        Path("results/experiment_6147_task_aware_energy_calibration.json"),
    ),
    (
        "exp6148-shifted-family-admission-held",
        "Gated on Exp6147 calibration readiness: one-shot shifted-family admission evaluation",
        Path("results/experiment_6148_shifted_family_admission_held.json"),
    ),
    (
        "exp6149-certified-strategy-schema-fixture",
        "Gated on Exp6145 stream readiness: certified strategy-schema and idempotence fixture",
        Path("results/experiment_6149_certified_strategy_schema_fixture.json"),
    ),
    (
        "exp6150-frozen-qwen-continuous-self-learning-ab",
        "Gated on Exp6148 and Exp6149 readiness: frozen-Qwen prospective continuous self-learning A/B",
        Path("results/experiment_6150_frozen_qwen_continuous_self_learning_ab.json"),
    ),
    (
        "exp6151-strategy-memory-shadow-adapter",
        "Gated on Exp6150 positive utility: default-off transactional strategy-memory adapter",
        Path("results/experiment_6151_strategy_memory_shadow_adapter.json"),
    ),
    (
        "exp6152-typed-stochastic-constraint-ir",
        "Gated on Exp6145 stream readiness: typed Torx-compatible stochastic constraint IR",
        Path("results/experiment_6152_typed_stochastic_constraint_ir.json"),
    ),
    (
        "exp6153-thermalized-program-error-audit",
        "Gated on Exp6152 IR readiness: software thermalization and compositional-error audit",
        Path("results/experiment_6153_thermalized_program_error_audit.json"),
    ),
    (
        "exp6154-arc-task-aware-energy-generalization",
        "ARC live-path adapter-disabled task-aware energy generalization",
        Path("results/experiment_6154_arc_task_aware_energy_generalization.json"),
    ),
    (
        "exp6155-v533-capstone-reconciliation",
        "Branch-independent .533 capstone, adversarial verification, and reconciliation",
        Path("results/experiment_6155_v533_capstone_reconciliation.json"),
    ),
)

NEXT_TASKS: tuple[tuple[str, str, Path], ...] = (
    (
        "exp6156-transition-v534",
        "Exact terminal-boundary handoff from .533 into .534",
        RESULT_RELATIVE_PATH,
    ),
    (
        "exp6157-repo-wide-artifact-isolation-closure",
        "Repository-wide test artifact-isolation compatibility closure",
        Path("results/experiment_6157_repo_wide_artifact_isolation_closure.json"),
    ),
    (
        "exp6158-v534-source-delta-ingestion",
        "Reliable dated evidence refresh after the V534 planner marker",
        Path("results/experiment_6158_v534_source_delta_ingestion.json"),
    ),
    (
        "exp6159-decision-calibrated-stream",
        "Fresh chronological decision-calibration stream and sealed endpoint preregistration",
        Path("results/experiment_6159_decision_calibrated_stream.json"),
    ),
    (
        "exp6160-sota-decision-calibration-corpus",
        "Gated on Exp6159 readiness: fresh flagship-GGUF decision-calibration corpus",
        Path("results/experiment_6160_sota_decision_calibration_corpus.json"),
    ),
    (
        "exp6161-decision-calibrated-energy-policy",
        "Gated on Exp6160 readiness: freeze a decision-calibrated task-energy policy",
        Path("results/experiment_6161_decision_calibrated_energy_policy.json"),
    ),
    (
        "exp6162-prospective-admission-replication",
        "Gated on Exp6161 readiness: one-shot decision-utility admission replication",
        Path("results/experiment_6162_prospective_admission_replication.json"),
    ),
    (
        "exp6163-certified-strategy-store-scaleup",
        "Gated on Exp6157 and Exp6159 readiness: certified strategy-store family scale-up",
        Path("results/experiment_6163_certified_strategy_store_scaleup.json"),
    ),
    (
        "exp6164-continuous-strategy-learning-ab",
        "Mandatory prospective continuous strategy-learning A/B",
        Path("results/experiment_6164_continuous_strategy_learning_ab.json"),
    ),
    (
        "exp6165-strategy-memory-shadow-adapter",
        "Gated on Exp6164 positive utility: default-off transactional strategy-memory adapter",
        Path("results/experiment_6165_strategy_memory_shadow_adapter.json"),
    ),
    (
        "exp6166-mode-jumping-factor-thermalization",
        "Mode-jumping approximate-factor thermalization and composition",
        Path("results/experiment_6166_mode_jumping_factor_thermalization.json"),
    ),
    (
        "exp6167-arc-task-aware-multiseed-replication",
        "ARC task-aware multi-seed replication with no solve claim",
        Path("results/experiment_6167_arc_task_aware_multiseed_replication.json"),
    ),
    (
        "exp6168-v534-capstone-reconciliation",
        "Branch-independent .534 capstone and reconciliation",
        Path("results/experiment_6168_v534_capstone_reconciliation.json"),
    ),
)

ACTIVATED_TASK_PATHS = {task_id: rel_path for task_id, _title, rel_path in ACTIVATED_TASKS}
STRUCTURED_SKIP_TASK_ID = "exp6151-strategy-memory-shadow-adapter"
NEXT_RANGE_NUMBERS = range(6156, 6169)

PROTECTED_FILE_PATHS = (
    ROADMAP_DOC_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    EVIDENCE_INDEX_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    *ACTIVATED_TASK_PATHS.values(),
)

PRECONDITION_CONTEXT_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    EVIDENCE_INDEX_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
)

OWNED_REFERENCE_PATHS = (
    Path("python/carnot/experiment_6156_transition_v534.py"),
    Path("tests/python/test_experiment_6156_transition_v534.py"),
    SPEC_RELATIVE_PATH,
    RESULT_RELATIVE_PATH,
)

CANONICAL_PLAN_REFERENCE_PATHS = (ROADMAP_RELATIVE_PATH, ROADMAP_NEXT_RELATIVE_PATH)
VNEXT_REFERENCE_PATHS = (ROADMAP_DOC_RELATIVE_PATH,)

REQUIRED_TASK_OWNED_GATE_KINDS = (
    "unit",
    "coverage",
    "spec_coverage",
    "yaml_parse",
    "exact_path",
    "terminal_quarantine",
    "duplicate_history",
    "activation",
    "exclusion_manifest",
    "range_collision",
    "adversarial_verifier",
    "protected_file",
    "applicable_e2e",
    "no_new_root_clutter",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "milestone_transition",
    "activated_task_and_deliverable_matrix",
    "exact_terminal_classification",
    "adversarial_verifier_receipts",
    "quarantine_and_null_preservation_receipts",
    "structured_gate_skip_receipts",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
    "staged_roadmap_activation_receipt",
    "next_task_range",
    "next_range_collision_count",
    "docs_reconciled",
    "protected_files_unchanged",
    "preexisting_worktree_changes_preserved",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": (
        "terminal transition state follows activated identity, exact path, terminal-class, "
        "quarantine, skip, history, activation, and collision receipts."
    ),
    "preconditions_checked": (
        "active and staged roadmaps, conductor receipts, declared matrix, history, "
        "exclusions, dirty worktree, root clutter, and protected hashes are parsed "
        "before mutation."
    ),
    "milestone_transition": (
        "only the fourteen activated task identities and declared paths define `.533`."
    ),
    "activated_task_and_deliverable_matrix": (
        "only the fourteen activated task identities and declared paths define `.533`."
    ),
    "exact_terminal_classification": (
        "missing, skip, block, partial, null, flagged, complete, and positive remain distinct."
    ),
    "adversarial_verifier_receipts": (
        "present exact artifacts are freshly checked and verifier flags remain quarantine evidence."
    ),
    "quarantine_and_null_preservation_receipts": (
        "Exp6147 and Exp6148 quarantine fields stay binding, Exp6148 remains a null, "
        "and diagnostics are not promoted."
    ),
    "structured_gate_skip_receipts": (
        "Exp6151 is a conductor structured skip, not a missing success or fabricated run."
    ),
    "research_complete_append_count": "append `.533` at most once and amplify no history.",
    "duplicate_history_amplification_count": "append `.533` at most once and amplify no history.",
    "staged_roadmap_activation_receipt": (
        "activation is exact when staged YAML exists and already-active when the conductor "
        "has consumed it into `research-roadmap.yaml`."
    ),
    "next_task_range": "bare zero collisions authorize exactly Exp6156-Exp6168.",
    "next_range_collision_count": "bare zero collisions authorize exactly Exp6156-Exp6168.",
    "docs_reconciled": (
        "transition-owned spec updates are recorded while conductor-owned ops "
        "reconciliation is deferred."
    ),
    "protected_files_unchanged": (
        "historical artifacts, conductor, exclusions, and unrelated files remain "
        "byte-identical except for intentional ledger/result writes."
    ),
    "preexisting_worktree_changes_preserved": (
        "pre-existing worktree changes are recorded and not staged or reverted."
    ),
    "duration_s": "measured aggregation duration for upstream-artifact transition.",
    "inference_substrate": (
        "set `aggregation_from_upstream_artifacts`; this task invokes no research LLM."
    ),
    "field_provenance": (
        "every required field traces to exact local receipts instead of broad glob inference."
    ),
    "test_commands": (
        "commands document focused unit/spec coverage, YAML parse, exact path, terminal "
        "and quarantine preservation, duplicate-history, activation, exclusion, collision, "
        "adversarial-verifier, protected-file, E2E, full-suite, and root-clutter checks."
    ),
    "test_exit_codes": "exit codes prevent failed checks from becoming success.",
    "reproducibility_checksum": (
        "a checksum detects later transition, activation, history, collision, or evidence drift."
    ),
    "honest_verdict": (
        "use a terminal `complete:` or `blocked:` prefix and state whether `.534` was activated."
    ),
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6156_transition_v534.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include="
    "python/carnot/experiment_6156_transition_v534.py -m pytest "
    "tests/python/test_experiment_6156_transition_v534.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include="
    "python/carnot/experiment_6156_transition_v534.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json <present .533 declared deliverables>",
    ".venv/bin/python scripts/check_spec_coverage.py",
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


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json(path: Path, payload: JsonMap) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_write_bytes(path, data)


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


def _completion_block_data() -> JsonDict:
    return {
        "id": MILESTONE_FROM,
        "title": MILESTONE_FROM_TITLE,
        "doc": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-06",
        "finding": "Terminal outcomes preserved by transition artifact.",
        "tasks": [
            {
                "id": task_id,
                "title": title,
                "deliverable": rel_path.as_posix(),
                "result": "terminal preserved",
            }
            for task_id, title, rel_path in ACTIVATED_TASKS
        ],
    }


def _write_history_blocks(path: Path, original: Any, blocks: list[JsonMap]) -> None:
    if isinstance(original, Mapping):
        updated = dict(original)
        updated["milestones"] = blocks
        data = yaml.safe_dump(updated, sort_keys=False).encode("utf-8")
    else:
        data = yaml.safe_dump(blocks, sort_keys=False).encode("utf-8")
    _atomic_write_bytes(path, data)


def _append_completion_if_absent(root: Path, terminal: bool) -> JsonDict:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    before_meta = _read_yaml_mapping(path)[1]
    before_blocks = _history_blocks(root)
    before_duplicate_count = _duplicate_history_block_count(before_blocks)
    canonical_signature = _task_signature(_completion_block_data())
    before_signatures = {
        _task_signature(block) for block in before_blocks if block.get("id") == MILESTONE_FROM
    }
    before_milestone_count = sum(1 for block in before_blocks if block.get("id") == MILESTONE_FROM)
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
    if canonical_signature in before_signatures:
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
        "after_milestone_block_count": sum(
            1 for block in written_blocks if block.get("id") == MILESTONE_FROM
        ),
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


def _run_live_adversarial_receipts(
    root: Path, metadata: Mapping[str, JsonMap]
) -> dict[str, JsonDict]:  # pragma: no cover
    executable = (
        (root / ".venv/bin/python").as_posix()
        if (root / ".venv/bin/python").exists()
        else sys.executable
    )
    receipts: dict[str, JsonDict] = {}
    for task_id, _title, rel_path in ACTIVATED_TASKS:
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


def _conductor_status_from_line(line: str) -> str:
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    return parts[2] if len(parts) > 2 else ""


def _queued_count(line: str) -> int | None:
    match = re.search(r"(\d+)\s+tasks queued", line)
    return int(match.group(1)) if match else None


def _latest_conductor_receipt(log_text: str, title: str) -> JsonDict:
    lines = log_text.splitlines()
    markers = [title[:size] for size in (58, 52, 46, 40, 34, 28, 22) if len(title) >= size]
    matches = [line for line in lines if any(marker and marker in line for marker in markers)]
    if not matches:
        return {"present": False, "status": None, "line": None, "detail": None}
    line = matches[-1]
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    return {
        "present": True,
        "timestamp": parts[0] if len(parts) > 0 else None,
        "status": parts[2] if len(parts) > 2 else None,
        "detail": parts[3] if len(parts) > 3 else None,
        "line": line,
    }


def _conductor_receipts(root: Path) -> JsonDict:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    lines = text.splitlines()
    plan_534 = [line for line in lines if "Plan milestone 2026.08.534" in line]
    activation_533 = [line for line in lines if "Milestone 2026.08.533 activated" in line]
    activation_534 = [line for line in lines if "Milestone 2026.08.534 activated" in line]
    return {
        "source_activation_line": activation_533[-1] if activation_533 else "",
        "source_activation_status": _conductor_status_from_line(activation_533[-1])
        if activation_533
        else "",
        "source_activated_task_count_claim": _queued_count(activation_533[-1])
        if activation_533
        else None,
        "destination_plan_line": plan_534[-1] if plan_534 else "",
        "destination_plan_status": _conductor_status_from_line(plan_534[-1]) if plan_534 else "",
        "destination_activation_line": activation_534[-1] if activation_534 else "",
        "destination_activation_status": _conductor_status_from_line(activation_534[-1])
        if activation_534
        else "",
        "destination_activated_task_count_claim": _queued_count(activation_534[-1])
        if activation_534
        else None,
    }


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


def _terminal_marker(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    marker = text.split(":", 1)[0].strip().split(None, 1)[0]
    if marker.startswith("retired"):
        return "retired"
    if marker.startswith("blocked"):
        return "block"
    if marker.startswith("complete_null") or marker == "null":
        return "null"
    if marker.startswith("complete_partial") or marker == "partial":
        return "partial"
    if (
        marker.startswith("complete_positive")
        or marker.startswith("complete_ready")
        or marker == "positive"
        or marker == "ready"
    ):
        return "positive"
    if marker.startswith("complete"):
        return "complete"
    return None


def _classify_underlying(payload: JsonMap, present: bool, receipt: JsonMap) -> str:
    if not present:
        if receipt.get("status") == "GATE_BLOCK" or receipt.get("latest_status") == "GATE_BLOCK":
            return "skip"
        return "missing"
    if payload.get("retirement_triggered") in {True, "retired"}:
        return "retired"
    status_marker = _terminal_marker(payload.get("status"))
    verdict_marker = _terminal_marker(payload.get("honest_verdict"))
    return status_marker or verdict_marker or "missing"


def _artifact_has_quarantine(payload: JsonMap) -> bool:
    return bool(payload.get("flagged_adversarial") or payload.get("corrigendum_pending"))


def _archive_terminal_class(underlying: str, payload: JsonMap, receipt: JsonMap) -> str:
    if underlying in {"missing", "skip"}:
        return underlying
    if _artifact_has_quarantine(payload) or _receipt_flag_count(receipt) > 0:
        return "flagged"
    return underlying


def _artifact_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, _title, rel_path in ACTIVATED_TASKS:
        payload, meta = _read_json_mapping(root / rel_path)
        meta["declared_deliverable"] = rel_path.as_posix()
        payloads[task_id] = payload
        metadata[task_id] = meta
    return payloads, metadata


def _activated_matrix(
    root: Path,
    payloads: Mapping[str, JsonMap],
    metadata: Mapping[str, JsonMap],
    receipts: Mapping[str, JsonMap],
    log_text: str,
) -> dict[str, JsonDict]:
    matrix: dict[str, JsonDict] = {}
    for task_id, title, rel_path in ACTIVATED_TASKS:
        payload = payloads[task_id]
        meta = metadata[task_id]
        conductor = _latest_conductor_receipt(log_text, title)
        present = bool(meta["present"] and meta["loadable"])
        underlying = _classify_underlying(payload, present, conductor)
        receipt = receipts.get(task_id, {})
        archive_class = _archive_terminal_class(underlying, payload, receipt)
        matrix[task_id] = {
            "identity": [MILESTONE_FROM, task_id, rel_path.as_posix()],
            "milestone": MILESTONE_FROM,
            "task_id": task_id,
            "title": title,
            "declared_deliverable": rel_path.as_posix(),
            "selection_policy": ARTIFACT_SELECTION_POLICY,
            "activated": True,
            "present": present,
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "status": str(payload.get("status") or ""),
            "honest_verdict": str(payload.get("honest_verdict") or ""),
            "artifact_quarantine_fields_present": _artifact_has_quarantine(payload),
            "verifier_flag_count": _receipt_flag_count(receipt),
            "same_number_aliases_ignored": _same_number_aliases(root, task_id, rel_path),
            "same_number_alias_used": False,
            "conductor": conductor,
            "underlying_terminal_class": underlying,
            "terminal_class": archive_class,
            "terminal_evidence_source": (
                "exact_declared_artifact"
                if present
                else "conductor_structured_gate_receipt"
                if underlying == "skip"
                else "declared_path_absent"
            ),
        }
    return matrix


def _exact_terminal_classification(matrix: Mapping[str, JsonMap]) -> JsonDict:
    underlying_by_task = {
        task_id: str(row["underlying_terminal_class"]) for task_id, row in matrix.items()
    }
    archive_by_task = {task_id: str(row["terminal_class"]) for task_id, row in matrix.items()}
    return {
        "terminal_class_by_task_id": archive_by_task,
        "underlying_terminal_class_by_task_id": underlying_by_task,
        "task_ids_by_terminal_class": {
            klass: [task_id for task_id, value in archive_by_task.items() if value == klass]
            for klass in sorted(set(archive_by_task.values()))
        },
        "task_ids_by_underlying_terminal_class": {
            klass: [task_id for task_id, value in underlying_by_task.items() if value == klass]
            for klass in sorted(set(underlying_by_task.values()))
        },
        "flagged_underlying_class_by_task_id": {
            task_id: underlying_by_task[task_id]
            for task_id, value in archive_by_task.items()
            if value == "flagged"
        },
        "all_activated_terminal": all(
            value not in {"missing"} for value in underlying_by_task.values()
        ),
        "nonterminal_task_ids": [
            task_id for task_id, value in underlying_by_task.items() if value == "missing"
        ],
        "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
    }


def _adversarial_receipts_group(
    receipts: Mapping[str, JsonMap],
    matrix: Mapping[str, JsonMap],
) -> JsonDict:
    reports: list[JsonDict] = []
    flagged_task_ids: list[str] = []
    for task_id, _title, _rel_path in ACTIVATED_TASKS:
        row = matrix[task_id]
        if not row["present"]:
            continue
        receipt = receipts.get(task_id)
        if not isinstance(receipt, Mapping):
            continue
        report = {
            "task_id": task_id,
            "artifact": row["declared_deliverable"],
            "command": str(receipt.get("command") or ""),
            "exit_code": receipt.get("exit_code"),
            "loaded": True,
            "flag_count": _receipt_flag_count(receipt),
            "max_severity": _receipt_max_severity(receipt),
            "flags": _receipt_flags(receipt),
            "receipt_hash": str(receipt.get("receipt_hash") or ""),
        }
        if report["flag_count"] > 0 or row["artifact_quarantine_fields_present"]:
            flagged_task_ids.append(task_id)
        reports.append(report)
    return {
        "reports": reports,
        "verified_present_declared_deliverable_count": len(reports),
        "missing_declared_deliverables_not_verified": [
            row["declared_deliverable"] for row in matrix.values() if not row["present"]
        ],
        "flagged_task_ids": flagged_task_ids,
        "flagged_count": len(flagged_task_ids),
        "principle": FIELD_PRINCIPLES["adversarial_verifier_receipts"],
    }


def _quarantine_and_null_preservation(
    payloads: Mapping[str, JsonMap],
    matrix: Mapping[str, JsonMap],
    verifier: JsonMap,
) -> JsonDict:
    flagged_task_ids = list(verifier.get("flagged_task_ids", []))
    exp6148 = payloads.get("exp6148-shifted-family-admission-held", {})
    exp6153 = payloads.get("exp6153-thermalized-program-error-audit", {})
    exp6154 = payloads.get("exp6154-arc-task-aware-energy-generalization", {})
    violation_count = None
    bound_counts = exp6153.get("bound_slack_and_violation_counts")
    if isinstance(bound_counts, Mapping):
        violation_count = bound_counts.get("violation_count")
    return {
        "quarantined_task_ids": flagged_task_ids,
        "exp6147_flag_preserved": {
            "artifact_quarantine_fields_present": bool(
                matrix["exp6147-task-aware-energy-calibration"][
                    "artifact_quarantine_fields_present"
                ]
            ),
            "underlying_terminal_class": matrix["exp6147-task-aware-energy-calibration"][
                "underlying_terminal_class"
            ],
            "archive_terminal_class": matrix["exp6147-task-aware-energy-calibration"][
                "terminal_class"
            ],
        },
        "exp6148_null_preserved": {
            "artifact_quarantine_fields_present": bool(
                matrix["exp6148-shifted-family-admission-held"][
                    "artifact_quarantine_fields_present"
                ]
            ),
            "underlying_terminal_class": matrix["exp6148-shifted-family-admission-held"][
                "underlying_terminal_class"
            ],
            "archive_terminal_class": matrix["exp6148-shifted-family-admission-held"][
                "terminal_class"
            ],
            "shifted_family_admission_ready_score": exp6148.get(
                "shifted_family_admission_ready_score"
            ),
            "diagnostic_fields_promoted": False,
        },
        "exp6150_block_preserved": {
            "terminal_class": matrix["exp6150-frozen-qwen-continuous-self-learning-ab"][
                "underlying_terminal_class"
            ],
            "gate_check_summary": payloads.get(
                "exp6150-frozen-qwen-continuous-self-learning-ab", {}
            ).get("gate_check_summary"),
        },
        "exp6153_zero_error_block_preserved": {
            "terminal_class": matrix["exp6153-thermalized-program-error-audit"][
                "underlying_terminal_class"
            ],
            "archive_terminal_class": matrix["exp6153-thermalized-program-error-audit"][
                "terminal_class"
            ],
            "violation_count": violation_count,
            "zero_error_receipt_preserved": violation_count == 0,
        },
        "exp6154_arc_no_solve_preserved": {
            "archive_terminal_class": matrix["exp6154-arc-task-aware-energy-generalization"][
                "terminal_class"
            ],
            "solve_claimed": bool(exp6154.get("solve_claimed", False)),
            "offline_reproduced": bool(exp6154.get("offline_reproduced", False)),
            "level_credit_delta": exp6154.get("level_credit_delta", 0),
        },
        "principle": FIELD_PRINCIPLES["quarantine_and_null_preservation_receipts"],
    }


def _structured_gate_skip_receipts(matrix: Mapping[str, JsonMap]) -> JsonDict:
    row = matrix[STRUCTURED_SKIP_TASK_ID]
    conductor = row.get("conductor", {}) if isinstance(row.get("conductor"), Mapping) else {}
    return {
        "task_id": STRUCTURED_SKIP_TASK_ID,
        "declared_deliverable": ACTIVATED_TASK_PATHS[STRUCTURED_SKIP_TASK_ID].as_posix(),
        "terminal_class": row["terminal_class"],
        "underlying_terminal_class": row["underlying_terminal_class"],
        "declared_artifact_present": bool(row["present"]),
        "conductor_latest_status": conductor.get("status"),
        "conductor_latest_line": conductor.get("line"),
        "reported_as_run": False,
        "artifact_invented": bool(row["present"]),
        "same_number_aliases_ignored": row["same_number_aliases_ignored"],
        "same_number_alias_used": bool(row["same_number_alias_used"]),
        "principle": FIELD_PRINCIPLES["structured_gate_skip_receipts"],
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


def _dirty_worktree_receipt(root: Path) -> JsonDict:
    if not (root / ".git").exists():
        return {"git_present": False, "dirty_paths": [], "command_exit_code": None}
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    paths = [line[3:] for line in result.stdout.splitlines() if len(line) > 3]
    return {
        "git_present": True,
        "dirty_paths": sorted(paths),
        "command_exit_code": result.returncode,
    }


def _activate_staged_roadmap(root: Path) -> JsonDict:
    active_path = root / ROADMAP_RELATIVE_PATH
    staged_path = root / ROADMAP_NEXT_RELATIVE_PATH
    active_payload, active_meta = _read_yaml_mapping(active_path)
    staged_payload, staged_meta = _read_yaml_mapping(staged_path)
    before_active_sha = active_meta["sha256"]
    if staged_meta["present"] and not staged_meta["loadable"]:
        return {
            "mode": "staged_unloadable",
            "activated": False,
            "staged_present": True,
            "staged_loadable": False,
            "active_before_sha256": before_active_sha,
            "active_after_sha256": path_sha256(active_path),
            "copied_exactly": False,
            "principle": FIELD_PRINCIPLES["staged_roadmap_activation_receipt"],
        }
    if staged_meta["loadable"]:
        if staged_payload.get("milestone") != MILESTONE_TO:
            return {
                "mode": "staged_milestone_mismatch",
                "activated": False,
                "staged_present": True,
                "staged_loadable": True,
                "staged_milestone": staged_payload.get("milestone"),
                "active_before_sha256": before_active_sha,
                "active_after_sha256": path_sha256(active_path),
                "copied_exactly": False,
                "principle": FIELD_PRINCIPLES["staged_roadmap_activation_receipt"],
            }
        data = staged_path.read_bytes()
        _atomic_write_bytes(active_path, data)
        active_after, active_meta_after = _read_yaml_mapping(active_path)
        copied_exactly = active_path.read_bytes() == data
        return {
            "mode": "copied_staged_roadmap",
            "activated": copied_exactly and active_after.get("milestone") == MILESTONE_TO,
            "staged_present": True,
            "staged_loadable": True,
            "staged_sha256": staged_meta["sha256"],
            "active_before_sha256": before_active_sha,
            "active_after_sha256": active_meta_after["sha256"],
            "active_milestone_after": active_after.get("milestone"),
            "active_roadmap_task_count": len(active_after.get("tasks", []))
            if isinstance(active_after.get("tasks"), list)
            else 0,
            "copied_exactly": copied_exactly,
            "principle": FIELD_PRINCIPLES["staged_roadmap_activation_receipt"],
        }
    active_task_count = (
        len(active_payload.get("tasks", [])) if isinstance(active_payload.get("tasks"), list) else 0
    )
    already_active = active_meta["loadable"] and active_payload.get("milestone") == MILESTONE_TO
    return {
        "mode": "already_active" if already_active else "not_active_no_staged_roadmap",
        "activated": already_active,
        "staged_present": False,
        "staged_loadable": False,
        "active_before_sha256": before_active_sha,
        "active_after_sha256": path_sha256(active_path),
        "active_milestone_after": active_payload.get("milestone"),
        "active_roadmap_task_count": active_task_count,
        "copied_exactly": False,
        "principle": FIELD_PRINCIPLES["staged_roadmap_activation_receipt"],
    }


def _range_number_mentions(text: str) -> set[int]:
    lowered = text.lower()
    if "615" not in lowered and "616" not in lowered:
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
        CONDUCTOR_LOG_RELATIVE_PATH,
        SPEC_RELATIVE_PATH,
    ]
    for folder in (
        "python",
        "tests",
        "scripts",
        "openspec/change-proposals",
        "openspec/capabilities",
        "ops",
    ):
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
    if rel_path in CANONICAL_PLAN_REFERENCE_PATHS:
        return "canonical_v534_plan_reference"
    if rel_path in VNEXT_REFERENCE_PATHS:
        return "vnext_v534_proposal_reference"
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
        "range": {"start": 6156, "end": 6168},
        "collision_count": len(collisions),
        "collisions": collisions,
        "allowed_references": allowed,
        "principle": FIELD_PRINCIPLES["next_range_collision_count"],
    }


def _suite_kinds(row: JsonMap) -> set[str]:
    values: set[str] = set()
    raw = row.get("suite_kind")
    if raw:
        values.add(str(raw))
    raw_many = row.get("suite_kinds")
    if isinstance(raw_many, list):
        values.update(str(value) for value in raw_many)
    return values


def _infer_test_row(command: str, exit_code: int | None) -> JsonDict:
    row: JsonDict = {"command": command, "exit_code": exit_code}
    if "test_experiment_6156_transition_v534.py" in command and "coverage" not in command:
        row["ownership_class"] = "task_owned"
        row["suite_kinds"] = [
            "unit",
            "yaml_parse",
            "exact_path",
            "terminal_quarantine",
            "duplicate_history",
            "activation",
            "exclusion_manifest",
            "range_collision",
            "adversarial_verifier",
            "protected_file",
            "applicable_e2e",
            "no_new_root_clutter",
        ]
    elif "coverage" in command and "experiment_6156_transition_v534.py" in command:
        row["ownership_class"] = "task_owned"
        row["suite_kind"] = "coverage"
    elif "check_spec_coverage.py" in command:
        row["ownership_class"] = "task_owned"
        row["suite_kind"] = "spec_coverage"
    elif "adversarial_verify.py" in command:
        row["ownership_class"] = "task_owned"
        row["suite_kind"] = "adversarial_verifier"
    elif "root_clutter" in command or "find . -maxdepth 1" in command:
        row["ownership_class"] = "root_clutter"
    else:
        row["ownership_class"] = "global_suite"
    return row


def _tests_run_rows(tests_run: Mapping[str, int] | Sequence[JsonMap] | None) -> list[JsonDict]:
    if tests_run is None:
        return [
            _infer_test_row(command, None) | {"status": "not_recorded"}
            for command in DEFAULT_TEST_COMMANDS
        ]
    if isinstance(tests_run, Mapping):
        return [
            _infer_test_row(str(command), int(exit_code))
            for command, exit_code in tests_run.items()
        ]
    return [dict(row) for row in tests_run]


def _task_owned_gate_receipts(rows: Sequence[JsonMap]) -> JsonDict:
    task_owned = [dict(row) for row in rows if row.get("ownership_class") == "task_owned"]
    kinds: set[str] = set()
    for row in task_owned:
        kinds.update(_suite_kinds(row))
    failures = [
        row
        for row in task_owned
        if not isinstance(row.get("exit_code"), int) or int(row["exit_code"]) != 0
    ]
    missing = [kind for kind in REQUIRED_TASK_OWNED_GATE_KINDS if kind not in kinds]
    return {
        "required_gate_kinds": list(REQUIRED_TASK_OWNED_GATE_KINDS),
        "observed_gate_kinds": sorted(kinds),
        "all_required_gate_kinds_present": not missing,
        "missing_required_gate_kinds": missing,
        "task_owned_failures": failures,
        "receipts": task_owned,
        "principle": FIELD_PRINCIPLES["test_commands"],
    }


def _root_clutter_delta(rows: Sequence[JsonMap]) -> int:
    before: set[str] = set()
    after: set[str] = set()
    before_seen = False
    after_seen = False
    for row in rows:
        if row.get("ownership_class") != "root_clutter":
            continue
        raw = row.get("root_clutter_paths")
        paths = {str(value) for value in raw} if isinstance(raw, list) else set()
        if row.get("phase") == "before":
            before_seen = True
            before.update(paths)
        if row.get("phase") == "after":
            after_seen = True
            after.update(paths)
    if not before_seen and after_seen:
        before = set(after)
    if not after_seen and before_seen:
        after = set(before)
    return len(after - before)


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": [
                ROADMAP_RELATIVE_PATH.as_posix(),
                RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
                CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "exact_declared_upstream_artifacts",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _source_hashes(root: Path) -> dict[str, JsonDict]:
    return {
        rel_path.as_posix(): {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
        }
        for rel_path in sorted(set(PRECONDITION_CONTEXT_PATHS), key=lambda value: value.as_posix())
    }


def _docs_reconciled(root: Path) -> JsonDict:
    spec_text = (
        (root / SPEC_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / SPEC_RELATIVE_PATH).exists()
        else ""
    )
    return {
        "openspec_research_reporting_req_6156_present": "REQ-REPORT-6156" in spec_text,
        "ops_status_deferred_to_conductor_stop_rule": True,
        "ops_changelog_deferred_to_conductor_stop_rule": True,
        "traceability_deferred_to_conductor_stop_rule": True,
        "principle": FIELD_PRINCIPLES["docs_reconciled"],
    }


def _preexisting_worktree_preserved(before: JsonMap, after: JsonMap) -> JsonDict:
    return {
        "preserved": before.get("dirty_paths") == after.get("dirty_paths")
        or before.get("git_present") is False,
        "before": before,
        "after": after,
        "staged_anything": False,
        "principle": FIELD_PRINCIPLES["preexisting_worktree_changes_preserved"],
    }


def _failed_preconditions(report: JsonMap) -> list[str]:
    failed: list[str] = []
    pre = report["preconditions_checked"]
    if pre["active_roadmap"]["present"] and not pre["active_roadmap"]["loadable"]:
        failed.append("active_roadmap_unloadable")
    if pre["research_complete"]["present"] and not pre["research_complete"]["loadable"]:
        failed.append("research_complete_unparseable")
    if pre["exclusion_manifest"]["present"] and not pre["exclusion_manifest"]["loadable"]:
        failed.append("exclusion_manifest_unparseable")
    conductor = pre["conductor_receipts"]
    if conductor.get("source_activated_task_count_claim") != 14:
        failed.append("v533_activation_line_missing_or_not_fourteen")
    if conductor.get("destination_activated_task_count_claim") != 13:
        failed.append("v534_activation_line_missing_or_not_thirteen")
    if not pre["live_adversarial_verifier"]["present"]:
        failed.append("live_verifier_missing")
    if not pre["docs_reconciled"]["openspec_research_reporting_req_6156_present"]:
        failed.append("openspec_req_6156_missing")
    if not report["exact_terminal_classification"]["all_activated_terminal"]:
        failed.append("terminal_outcomes_not_preserved")
    quarantine = report["quarantine_and_null_preservation_receipts"]
    if quarantine["exp6147_flag_preserved"]["archive_terminal_class"] != "flagged":
        failed.append("exp6147_quarantine_not_preserved")
    if quarantine["exp6148_null_preserved"]["underlying_terminal_class"] != "null":
        failed.append("exp6148_null_not_preserved")
    if quarantine["exp6148_null_preserved"]["archive_terminal_class"] != "flagged":
        failed.append("exp6148_quarantine_not_preserved")
    if quarantine["exp6148_null_preserved"]["diagnostic_fields_promoted"]:
        failed.append("diagnostic_fields_promoted")
    if report["structured_gate_skip_receipts"]["underlying_terminal_class"] != "skip":
        failed.append("structured_gate_skip_not_preserved")
    verifier = report["adversarial_verifier_receipts"]
    present_count = sum(
        1 for row in report["activated_task_and_deliverable_matrix"].values() if row["present"]
    )
    if verifier["verified_present_declared_deliverable_count"] != present_count:
        failed.append("missing_adversarial_receipts")
    if report["research_complete_append_receipt"]["duplicate_history_amplification_count"] != 0:
        failed.append("duplicate_history_amplified")
    if not report["staged_roadmap_activation_receipt"]["activated"]:
        failed.append("staged_roadmap_activation_failed")
    if report["next_range_collision_count"] != 0:
        failed.append("next_range_collision_detected")
    gates = report["task_owned_gate_receipts"]
    if not gates["all_required_gate_kinds_present"]:
        failed.append("task_owned_gate_missing")
    if gates["task_owned_failures"]:
        failed.append("task_owned_gate_failed")
    if report["root_clutter_delta_count"] != 0:
        failed.append("root_clutter_debt_amplified")
    if not report["protected_files_unchanged"]["all_unchanged"]:
        failed.append("protected_file_modified")
    return failed


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.monotonic()
    protected_before = _protected_file_hashes(root)
    dirty_before = _dirty_worktree_receipt(root)
    active_payload, active_meta = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    staged_payload, staged_meta = _read_yaml_mapping(root / ROADMAP_NEXT_RELATIVE_PATH)
    complete_payload, complete_meta = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    exclusion_payload, exclusion_meta = _read_yaml_mapping(root / EXCLUSION_MANIFEST_RELATIVE_PATH)
    payloads, metadata = _artifact_payloads(root)
    if adversarial_receipts is None:  # pragma: no cover
        receipts = _run_live_adversarial_receipts(root, metadata)
    else:
        receipts = _normalize_adversarial_receipts(adversarial_receipts, metadata)
    log_text = (
        (root / CONDUCTOR_LOG_RELATIVE_PATH).read_text(encoding="utf-8", errors="replace")
        if (root / CONDUCTOR_LOG_RELATIVE_PATH).exists()
        else ""
    )
    matrix = _activated_matrix(root, payloads, metadata, receipts, log_text)
    terminal = _exact_terminal_classification(matrix)
    terminal_ok = bool(terminal["all_activated_terminal"])
    append_receipt = _append_completion_if_absent(root, terminal_ok)
    activation = _activate_staged_roadmap(root)
    range_scan = _range_collision_scan(root)
    rows = _tests_run_rows(tests_run)
    commands = [str(row.get("command")) for row in rows]
    exits = {str(row.get("command")): row.get("exit_code") for row in rows}
    verifier = _adversarial_receipts_group(receipts, matrix)
    docs = _docs_reconciled(root)
    protected = _protected_files_unchanged(root, protected_before)
    dirty_after = _dirty_worktree_receipt(root)
    actual_duration = float(duration_s if duration_s is not None else time.monotonic() - started)
    task_gates = _task_owned_gate_receipts(rows)

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE_TO,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": "complete_with_terminal_receipts",
        "preconditions_checked": {
            "source_hashes": _source_hashes(root),
            "active_roadmap": {
                **active_meta,
                "milestone": active_payload.get("milestone"),
                "task_count": len(active_payload.get("tasks", []))
                if isinstance(active_payload.get("tasks"), list)
                else 0,
            },
            "staged_roadmap": {
                **staged_meta,
                "milestone": staged_payload.get("milestone"),
                "task_count": len(staged_payload.get("tasks", []))
                if isinstance(staged_payload.get("tasks"), list)
                else 0,
            },
            "research_complete": complete_meta,
            "research_complete_loadable_type": type(complete_payload).__name__,
            "conductor_log": {
                "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "present": (root / CONDUCTOR_LOG_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
            },
            "conductor_receipts": _conductor_receipts(root),
            "exclusion_manifest": exclusion_meta,
            "exclusion_loadable_type": type(exclusion_payload).__name__,
            "dirty_worktree_before": dirty_before,
            "root_clutter_python_files_before": _root_clutter_inventory(root),
            "protected_hashes_before": protected_before,
            "live_adversarial_verifier": {
                "path": ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
                "present": (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).exists(),
                "sha256": path_sha256(root / ADVERSARIAL_VERIFY_RELATIVE_PATH),
            },
            "docs_reconciled": docs,
            "range_collision_scan": range_scan,
            "failed_preconditions": [],
        },
        "milestone_transition": {
            "source_milestone": MILESTONE_FROM,
            "destination_milestone": MILESTONE_TO,
            "task_identity_tuple": ["milestone", "task_id", "declared_deliverable"],
        },
        "activated_task_and_deliverable_matrix": matrix,
        "exact_terminal_classification": terminal,
        "adversarial_verifier_receipts": verifier,
        "quarantine_and_null_preservation_receipts": _quarantine_and_null_preservation(
            payloads, matrix, verifier
        ),
        "structured_gate_skip_receipts": _structured_gate_skip_receipts(matrix),
        "research_complete_append_count": append_receipt["append_count"],
        "research_complete_append_receipt": append_receipt,
        "duplicate_history_amplification_count": append_receipt[
            "duplicate_history_amplification_count"
        ],
        "staged_roadmap_activation_receipt": activation,
        "next_task_range": {
            "start": "exp6156",
            "end": "exp6168",
            "reserved_count": 13,
            "principle": FIELD_PRINCIPLES["next_task_range"],
        },
        "next_range_collision_count": range_scan["collision_count"],
        "docs_reconciled": docs,
        "protected_files_unchanged": protected,
        "preexisting_worktree_changes_preserved": _preexisting_worktree_preserved(
            dirty_before, dirty_after
        ),
        "duration_s": actual_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exits,
        "task_owned_gate_receipts": task_gates,
        "root_clutter_delta_count": _root_clutter_delta(rows),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    failed = _failed_preconditions(report)
    report["preconditions_checked"]["failed_preconditions"] = failed
    if failed:
        report["status"] = "blocked"
        report["honest_verdict"] = (
            "blocked: .534 activation not complete; failed_preconditions=" + ",".join(failed)
        )
    else:
        report["honest_verdict"] = (
            "complete: .533 archived exactly once with quarantine, null, block, partial, "
            f"and skip states preserved; .534 activation mode={activation['mode']}"
        )
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_artifact(payload: JsonMap) -> None:
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            raise ValueError(f"missing required field: {field}")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        row = provenance.get(field)
        if not isinstance(row, Mapping) or row.get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field provenance missing or wrong for {field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    if not str(payload.get("honest_verdict", "")).startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict must use complete: or blocked:")
    if not isinstance(payload.get("next_range_collision_count"), int):
        raise ValueError("next_range_collision_count must be a bare integer")
    if int(payload["research_complete_append_count"]) > 1:
        raise ValueError("research_complete_append_count must be at most one")
    if int(payload["duplicate_history_amplification_count"]) != 0:
        raise ValueError("duplicate_history_amplification_count must remain zero")
    matrix = payload.get("activated_task_and_deliverable_matrix")
    if not isinstance(matrix, Mapping) or len(matrix) != len(ACTIVATED_TASKS):
        raise ValueError("activated matrix must contain exactly fourteen tasks")
    for task_id, _title, rel_path in ACTIVATED_TASKS:
        row = matrix.get(task_id)
        if not isinstance(row, Mapping):
            raise ValueError("activated matrix must contain exactly fourteen mapping rows")
        if row.get("identity") != [MILESTONE_FROM, task_id, rel_path.as_posix()]:
            raise ValueError(f"activated identity mismatch for {task_id}")
    terminal = payload.get("exact_terminal_classification")
    if not isinstance(terminal, Mapping):
        raise ValueError("terminal classification missing")
    underlying = terminal.get("underlying_terminal_class_by_task_id")
    archive = terminal.get("terminal_class_by_task_id")
    if not isinstance(underlying, Mapping) or not isinstance(archive, Mapping):
        raise ValueError("terminal classification mappings missing")
    if underlying.get("exp6148-shifted-family-admission-held") != "null":
        raise ValueError("Exp6148 null classification must be preserved")
    if archive.get("exp6147-task-aware-energy-calibration") != "flagged":
        raise ValueError("Exp6147 quarantine must remain flagged")
    if archive.get("exp6148-shifted-family-admission-held") != "flagged":
        raise ValueError("Exp6148 quarantine must remain flagged")
    quarantine = payload.get("quarantine_and_null_preservation_receipts")
    if not isinstance(quarantine, Mapping):
        raise ValueError("quarantine receipts missing")
    exp6148 = quarantine.get("exp6148_null_preserved")
    if not isinstance(exp6148, Mapping):
        raise ValueError("Exp6148 null receipt missing")
    if exp6148.get("diagnostic_fields_promoted") is not False:
        raise ValueError("diagnostic fields must not be promoted")
    skip = payload.get("structured_gate_skip_receipts")
    if not isinstance(skip, Mapping) or skip.get("reported_as_run") is not False:
        raise ValueError("structured gate skip must not be reported as a run")
    verifier = payload.get("adversarial_verifier_receipts")
    if not isinstance(verifier, Mapping):
        raise ValueError("adversarial verifier receipts missing")
    present_count = sum(1 for row in matrix.values() if isinstance(row, Mapping) and row["present"])
    if verifier.get("verified_present_declared_deliverable_count") != present_count:
        raise ValueError("adversarial verifier receipt count mismatch")
    activation = payload.get("staged_roadmap_activation_receipt")
    if not isinstance(activation, Mapping) or not activation.get("activated"):
        raise ValueError("activation receipt must show .534 activated")
    gates = payload.get("task_owned_gate_receipts")
    if not isinstance(gates, Mapping) or not gates.get("all_required_gate_kinds_present"):
        raise ValueError("task-owned gate receipts are incomplete")
    if gates.get("task_owned_failures"):
        raise ValueError("task-owned gate receipts include failures")
    protected = payload.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("all_unchanged") is not True:
        raise ValueError("protected file hashes changed")
    if payload["next_range_collision_count"] != 0 and str(payload.get("status")) != "blocked":
        raise ValueError("next_range_collision_count must be zero for a complete artifact")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        raise ValueError("checksum mismatch")


def run(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    payload = build_report(
        root,
        adversarial_receipts=adversarial_receipts,
        tests_run=tests_run,
        duration_s=duration_s,
    )
    write_json(root / RESULT_RELATIVE_PATH, payload)
    return payload


def _parse_recorded_tests(values: Sequence[str]) -> dict[str, int]:  # pragma: no cover
    out: dict[str, int] = {}
    for value in values:
        command, sep, code = value.rpartition("=")
        if not sep:
            raise ValueError(f"--record-test requires COMMAND=EXIT_CODE, got {value!r}")
        out[command] = int(code)
    return out


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--record-test",
        action="append",
        default=[],
        help="Record a completed check as COMMAND=EXIT_CODE.",
    )
    args = parser.parse_args(argv)
    tests = _parse_recorded_tests(args.record_test) if args.record_test else None
    payload = run(REPO_ROOT, tests_run=tests)
    if args.output != REPO_ROOT / RESULT_RELATIVE_PATH:
        write_json(args.output, payload)
    print(
        json.dumps(
            {
                "path": args.output.as_posix(),
                "status": payload["status"],
                "checksum": payload["reproducibility_checksum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

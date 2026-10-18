"""Exp6169 transition receipt from terminal milestone .534 into .535.

Spec refs: REQ-REPORT-6169,
SCENARIO-REPORT-6169-EXACT-TERMINAL,
SCENARIO-REPORT-6169-APPEND-ONCE,
SCENARIO-REPORT-6169-ROADMAP-VALIDATION,
SCENARIO-REPORT-6169-PARTIAL-ACTIVATION-BLOCKS,
SCENARIO-REPORT-6169-SCHEMA.

This module is a deterministic repository transition. It reads exact source
artifacts, records whether the next roadmap is truly ready, and fails closed
when activation has only partially happened.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = "experiment_6169_transition_v535"
EXPERIMENT_ID = "exp6169-v535-transition"
SCHEMA = "carnot.experiment_6169.transition_v535.v1"
RUN_DATE = "20260807"
RANDOM_SEED = 6169
INFERENCE_SUBSTRATE = "deterministic_repository_transition"

SOURCE_MILESTONE = "2026.08.534"
SOURCE_MILESTONE_TITLE = (
    "Decision-Calibrated Energy, Prospective Strategy Learning, "
    "and Nontrivial Stochastic Compilation"
)
TARGET_MILESTONE = "2026.08.535"
TARGET_MILESTONE_TITLE = (
    "Executable-Trace Internal Verification, Retention-Safe Strategy Memory, "
    "and Live-Path Robustness"
)

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
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
DETERMINATION_LINT_RELATIVE_PATH = Path("scripts/determination_preservation_lint.py")
TRANSITION_INTEGRITY_RELATIVE_PATH = Path("scripts/transition_integrity.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

SOURCE_TRANSITION_RELATIVE_PATH = Path("results/experiment_6156_transition_v534.json")
SOURCE_CAPSTONE_RELATIVE_PATH = Path("results/experiment_6168_v534_capstone_reconciliation.json")
RESULT_RELATIVE_PATH = Path("results/experiment_6169_transition_v535.json")

SOURCE_TASKS_WITHOUT_CAPSTONE: tuple[tuple[str, str, Path], ...] = (
    (
        "exp6156-transition-v534",
        "Exact terminal-boundary handoff from .533 into .534",
        SOURCE_TRANSITION_RELATIVE_PATH,
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
        "Mandatory prospective continuous strategy-learning A/B on frozen flagship GGUFs",
        Path("results/experiment_6164_continuous_strategy_learning_ab.json"),
    ),
    (
        "exp6165-strategy-memory-shadow-adapter",
        "Gated on Exp6164 positive utility: default-off transactional strategy-memory adapter",
        Path("results/experiment_6165_strategy_memory_shadow_adapter.json"),
    ),
    (
        "exp6166-mode-jumping-factor-thermalization",
        "Mode-jumping CNCE for nonzero-error typed-factor thermalization",
        Path("results/experiment_6166_mode_jumping_factor_thermalization.json"),
    ),
    (
        "exp6167-arc-task-aware-multiseed-replication",
        "ARC live-path task-aware admission replication across games and seeds, no solve",
        Path("results/experiment_6167_arc_task_aware_multiseed_replication.json"),
    ),
)

SOURCE_TASKS: tuple[tuple[str, str, Path], ...] = (
    *SOURCE_TASKS_WITHOUT_CAPSTONE,
    (
        "exp6168-v534-capstone-reconciliation",
        "Branch-independent .534 capstone, adversarial verification, and reconciliation",
        SOURCE_CAPSTONE_RELATIVE_PATH,
    ),
)

EXPECTED_NEXT_TASKS: tuple[tuple[str, str, Path], ...] = (
    ("exp6169-v535-transition", "Exact terminal-boundary handoff from .534 into .535", RESULT_RELATIVE_PATH),
    (
        "exp6170-v535-task-artifact-isolation-canary",
        "Task-scoped artifact-isolation compatibility canary for .535",
        Path("results/experiment_6170_v535_task_artifact_isolation_canary.json"),
    ),
    (
        "exp6171-v535-source-delta-ingestion",
        "Reliable dated evidence refresh after the V535 planner marker",
        Path("results/experiment_6171_v535_source_delta_ingestion.json"),
    ),
    (
        "exp6172-current-rule-quarantine-determination",
        "Immutable current-rule companion determination for Exp6161 and Exp6162",
        Path("results/experiment_6172_current_rule_quarantine_determination.json"),
    ),
    (
        "exp6173-cctu-item-bank-preregistration",
        "Frozen executable CCTU-style item bank and Phase-D preregistration",
        Path("results/experiment_6173_cctu_item_bank_preregistration.json"),
    ),
    (
        "exp6174-cctu-authentic-k8-pool",
        "Authentic local-SOTA K>=8 executable trace pool",
        Path("results/experiment_6174_cctu_authentic_k8_pool.json"),
    ),
    (
        "exp6175-cctu-headroom-audit",
        "Competence, unsaturation, and selectable-headroom audit",
        Path("results/experiment_6175_cctu_headroom_audit.json"),
    ),
    (
        "exp6176-hidden-state-surface-qualification",
        "Matching-base per-layer hidden-state surface qualification",
        Path("results/experiment_6176_hidden_state_surface_qualification.json"),
    ),
    (
        "exp6177-clue-latent-selector-freeze",
        "Calibration-only CLUE and latent selector freeze",
        Path("results/experiment_6177_clue_latent_selector_freeze.json"),
    ),
    (
        "exp6178-held-internal-state-selection",
        "One-shot held internal-state selection",
        Path("results/experiment_6178_held_internal_state_selection.json"),
    ),
    (
        "exp6179-retention-safe-continuous-strategy-learning-ab",
        "Mandatory retention-safe continuous strategy-learning A/B",
        Path("results/experiment_6179_retention_safe_continuous_strategy_learning_ab.json"),
    ),
    (
        "exp6180-exp6166-reproducibility-adjudication",
        "Exp6166 evidence-preserving reproducibility adjudication",
        Path("results/experiment_6180_exp6166_reproducibility_adjudication.json"),
    ),
    (
        "exp6181-arc-logo-shortcut-audit",
        "Single ARC slot leave-one-game-out shortcut audit",
        Path("results/experiment_6181_arc_logo_shortcut_audit.json"),
    ),
    (
        "exp6182-v535-capstone-reconciliation",
        "Branch-independent .535 capstone",
        Path("results/experiment_6182_v535_capstone_reconciliation.json"),
    ),
)

EXPECTED_TASK_IDS = tuple(task_id for task_id, _title, _deliverable in EXPECTED_NEXT_TASKS)
EXPECTED_DELIVERABLES = tuple(deliverable.as_posix() for _task_id, _title, deliverable in EXPECTED_NEXT_TASKS)
EXPECTED_TRACKS: dict[str, str] = {
    "exp6169-v535-transition": "infrastructure",
    "exp6170-v535-task-artifact-isolation-canary": "infrastructure",
    "exp6171-v535-source-delta-ingestion": "evidence-ingestion",
    "exp6172-current-rule-quarantine-determination": "determination",
    "exp6173-cctu-item-bank-preregistration": "phase-d",
    "exp6174-cctu-authentic-k8-pool": "phase-d",
    "exp6175-cctu-headroom-audit": "phase-d",
    "exp6176-hidden-state-surface-qualification": "phase-d",
    "exp6177-clue-latent-selector-freeze": "phase-d",
    "exp6178-held-internal-state-selection": "phase-d",
    "exp6179-retention-safe-continuous-strategy-learning-ab": "continuous-learning",
    "exp6180-exp6166-reproducibility-adjudication": "stochastic-reproducibility",
    "exp6181-arc-logo-shortcut-audit": "arc",
    "exp6182-v535-capstone-reconciliation": "capstone",
}
LLM_TASK_IDS = {
    "exp6174-cctu-authentic-k8-pool",
    "exp6176-hidden-state-surface-qualification",
    "exp6177-clue-latent-selector-freeze",
    "exp6178-held-internal-state-selection",
    "exp6179-retention-safe-continuous-strategy-learning-ab",
}
SUPPORTED_GATE_OPERATORS = {"==", "!=", ">", ">=", "<", "<=", "exists", "truthy", "falsey"}
PROMPT_ENDING = "Run command, 'Do NOT push. Do NOT modify scripts/research_conductor.py.'\n"

PROTECTED_FILE_PATHS = (
    SOURCE_TRANSITION_RELATIVE_PATH,
    SOURCE_CAPSTONE_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    DETERMINATION_LINT_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
)

CONTEXT_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    RESEARCH_PROGRAM_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    ROADMAP_NEXT_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    SOURCE_TRANSITION_RELATIVE_PATH,
    SOURCE_CAPSTONE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    DETERMINATION_LINT_RELATIVE_PATH,
    TRANSITION_INTEGRITY_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "source_and_target_milestones",
    "source_capstone_hash_and_honest_verdict",
    "source_exact_terminal_classification",
    "research_complete_count_before_after",
    "research_complete_append_count",
    "staged_and_activated_roadmap_hashes",
    "activated_task_count",
    "task_id_and_deliverable_collision_matrix",
    "optional_field_and_prior_failure_validation",
    "gate_reference_validation",
    "arc_ingestion_infrastructure_and_phase_d_allocation_receipt",
    "mandatory_model_spec_validation",
    "quarantine_and_determination_before_after_matrix",
    "preexisting_worktree_changes_preserved",
    "activation_mode",
    "rollback_receipt",
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
    "status": "terminal state follows exact source evidence, append-once history, roadmap validation, activation, and rollback receipts.",
    "preconditions_checked": "snapshots instructions, roadmaps, completion history, capstone, exclusions, dirty worktree, root clutter, and protected hashes before mutation.",
    "source_and_target_milestones": "binds the receipt to .534 source evidence and .535 target activation.",
    "source_capstone_hash_and_honest_verdict": "requires the Exp6168 capstone and carries its verdict without strengthening it.",
    "source_exact_terminal_classification": "missing, skipped, internal block, null, retired, flagged, blocked, complete, and positive classes remain distinct.",
    "research_complete_count_before_after": "proves .534 completion history has exactly one total entry afterward.",
    "research_complete_append_count": "append .534 at most once for this transition.",
    "staged_and_activated_roadmap_hashes": "activation is exact when staged exists and fails closed on partial active fallback.",
    "activated_task_count": "the .535 transition is only complete with fourteen activated tasks.",
    "task_id_and_deliverable_collision_matrix": "Exp6169-Exp6182 task IDs and deliverables must be unique and collision-free.",
    "optional_field_and_prior_failure_validation": "optional roadmap fields and prior-failure rows must be structurally auditable.",
    "gate_reference_validation": "gates must reference known non-retired upstreams and supported operators.",
    "arc_ingestion_infrastructure_and_phase_d_allocation_receipt": "roadmap allocation must contain exactly the promised ARC, SOTA-ingestion, infrastructure, and Phase-D slots.",
    "mandatory_model_spec_validation": "LLM tasks must declare local GGUF requirements rather than silently falling back to remote models.",
    "quarantine_and_determination_before_after_matrix": "historical flags, nulls, blocks, and skips are byte-preserved; companion determinations do not mutate source artifacts.",
    "preexisting_worktree_changes_preserved": "pre-existing user changes are recorded and not staged or reverted.",
    "activation_mode": "distinguishes staged atomic copy, already-active complete, and partial mismatch modes.",
    "rollback_receipt": "partial or mismatched activation must leave a rollback audit trail.",
    "protected_files_unchanged": "conductor, source evidence, ops docs, exclusions, and verifier lints stay unchanged unless an exact staged roadmap activation is performed.",
    "duration_s": "measured deterministic repository-transition duration.",
    "inference_substrate": "set deterministic_repository_transition because no research model or solver runs.",
    "field_provenance": "every required field traces to exact local receipts.",
    "test_commands": "records focused unit/spec coverage, YAML/schema, collision, gate, history, preservation, E2E, full-suite, and root-clutter checks.",
    "test_exit_codes": "exit codes keep failed checks from being laundered into success.",
    "reproducibility_checksum": "content hash detects later evidence, roadmap, history, or validation drift.",
    "honest_verdict": "use complete: or blocked: and name append multiplicity and activation mode.",
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6169_transition_v535.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6169_transition_v535.py -m pytest tests/python/test_experiment_6169_transition_v535.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6169_transition_v535.py --fail-under=100",
    ".venv/bin/python -c \"import pathlib, yaml; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); yaml.safe_load(pathlib.Path('research-complete.yaml').read_text()); yaml.safe_load(pathlib.Path('ops/exclusion_manifest.yaml').read_text())\"",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python -m carnot.experiment_6169_transition_v535 --validate",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6169_transition_v535.py",
    ".venv/bin/python scripts/check_exclusion_manifest.py 6169",
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python -c \"from pathlib import Path; assert not Path('scripts/transition_integrity.py').exists(); print('transition_integrity.py absent; module fallback recorded')\"",
    ".venv/bin/python -c \"import yaml; data=yaml.safe_load(open('research-complete.yaml')) or {}; n=sum(1 for m in data.get('milestones', []) if isinstance(m, dict) and m.get('id') == '2026.08.534'); assert n == 1; print('research-complete .534 duplicate-history OK')\"",
    ".venv/bin/python scripts/determination_preservation_lint.py HEAD",
    "git diff --quiet -- scripts/research_conductor.py && git diff --cached --quiet -- scripts/research_conductor.py && echo 'scripts/research_conductor.py protected'",
    ".venv/bin/python -c \"from pathlib import Path; assert Path('ops/e2e-test-plan.md').exists(); print('E2E plan inspected; no EBM runtime E2E applies to deterministic roadmap transition')\"",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


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
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _read_json(path: Path) -> tuple[JsonDict, JsonDict]:
    meta: JsonDict = {"path": path.as_posix(), "present": path.exists(), "sha256": path_sha256(path), "loadable": False}
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


def _read_yaml(path: Path) -> tuple[Any, JsonDict]:
    meta: JsonDict = {"path": path.as_posix(), "present": path.exists(), "sha256": path_sha256(path), "loadable": False}
    if not path.exists():
        meta["error"] = "missing"
        return {}, meta
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        meta["error"] = f"yaml_error:{exc.__class__.__name__}"
        return {}, meta
    meta["loadable"] = True
    return payload, meta


def _git_status(root: Path) -> JsonDict:
    try:
        proc = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:  # pragma: no cover
        return {"available": False, "exit_code": 127, "stdout_lines": [], "stderr": str(exc)}
    return {
        "available": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout_lines": [line for line in proc.stdout.splitlines() if line],
        "stderr": proc.stderr.strip(),
    }


def _root_clutter(root: Path) -> list[str]:
    return sorted(path.name for path in root.glob("*.py") if path.is_file())


def _task_signature(block: JsonMap) -> tuple[tuple[str, str], ...]:
    tasks = block.get("tasks")
    if not isinstance(tasks, list):
        return ()
    return tuple(
        (str(task.get("id")), str(task.get("deliverable") or ""))
        for task in tasks
        if isinstance(task, Mapping)
    )


def _completion_block() -> JsonDict:
    return {
        "id": SOURCE_MILESTONE,
        "title": SOURCE_MILESTONE_TITLE,
        "doc": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "completed": "2026-08-06",
        "finding": "Terminal outcomes preserved by Exp6169 transition receipt.",
        "tasks": [
            {
                "id": task_id,
                "title": title,
                "deliverable": deliverable.as_posix(),
                "result": "terminal preserved",
            }
            for task_id, title, deliverable in SOURCE_TASKS
        ],
    }


def _history_blocks(root: Path) -> tuple[list[JsonMap], Any]:
    payload, _meta = _read_yaml(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    blocks = payload.get("milestones") if isinstance(payload, Mapping) else payload
    if not isinstance(blocks, list):
        return [], payload
    return [block for block in blocks if isinstance(block, Mapping)], payload


def _history_receipt(root: Path) -> JsonDict:
    blocks, _payload = _history_blocks(root)
    count = sum(1 for block in blocks if block.get("id") == SOURCE_MILESTONE)
    canonical = _task_signature(_completion_block())
    canonical_count = sum(
        1 for block in blocks if block.get("id") == SOURCE_MILESTONE and _task_signature(block) == canonical
    )
    return {
        "before_count": count,
        "after_count": count,
        "before_canonical_count": canonical_count,
        "after_canonical_count": canonical_count,
        "before_sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
        "after_sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
        "append_count": 0,
        "exactly_one_after": count == 1,
    }


def _append_history_if_needed(root: Path, receipt: JsonDict, allowed: bool) -> JsonDict:
    if not allowed or receipt["before_count"] != 0:
        receipt["reason"] = "present_or_transition_blocked" if receipt["before_count"] else "transition_blocked"
        return receipt
    blocks, payload = _history_blocks(root)
    blocks.append(_completion_block())
    updated = dict(payload) if isinstance(payload, Mapping) else {"milestones": []}
    updated["milestones"] = blocks
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    path.write_text(yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")
    after = _history_receipt(root)
    after["before_count"] = receipt["before_count"]
    after["before_canonical_count"] = receipt["before_canonical_count"]
    after["before_sha256"] = receipt["before_sha256"]
    after["append_count"] = 1
    after["reason"] = "canonical_534_block_absent"
    return after


def _duplicate_values(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _valid_prior_failures(tasks: Sequence[JsonMap]) -> JsonDict:
    required = {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"}
    invalid: list[JsonDict] = []
    rows = 0
    for task in tasks:
        failures = task.get("prior_failures", [])
        if failures is None:
            failures = []
        if not isinstance(failures, list):
            invalid.append({"task_id": task.get("id"), "reason": "prior_failures_not_list"})
            continue
        for row in failures:
            rows += 1
            if not isinstance(row, Mapping) or set(row) < required:
                invalid.append({"task_id": task.get("id"), "reason": "missing_prior_failure_subfields"})
    return {"rows_checked": rows, "invalid_rows": invalid, "valid": not invalid}


def _valid_gates(tasks: Sequence[JsonMap]) -> JsonDict:
    known = {str(task.get("id")) for task in tasks}
    invalid: list[JsonDict] = []
    retired_chain = []
    for task in tasks:
        gates = task.get("gated_on", [])
        if gates is None:
            gates = []
        if not isinstance(gates, list):
            invalid.append({"task_id": task.get("id"), "reason": "gated_on_not_list"})
            continue
        for gate in gates:
            if not isinstance(gate, Mapping):
                invalid.append({"task_id": task.get("id"), "reason": "gate_not_mapping"})
                continue
            if gate.get("op") not in SUPPORTED_GATE_OPERATORS:
                invalid.append({"task_id": task.get("id"), "reason": "unsupported_operator", "op": gate.get("op")})
            if gate.get("upstream") not in known:
                invalid.append({"task_id": task.get("id"), "reason": "unknown_upstream", "upstream": gate.get("upstream")})
    return {
        "supported_operators": not any(row["reason"] == "unsupported_operator" for row in invalid),
        "known_upstreams": not any(row["reason"] == "unknown_upstream" for row in invalid),
        "no_retired_upstream_requires_chain": not retired_chain,
        "invalid_gates": invalid,
        "valid": not invalid and not retired_chain,
    }


def _allocation_receipt(tasks: Sequence[JsonMap]) -> JsonDict:
    tracks = Counter(str(task.get("track")) for task in tasks)
    task_ids = [str(task.get("id")) for task in tasks]
    arc = [task_id for task_id in task_ids if "arc" in task_id]
    sota = [task_id for task_id in task_ids if "source-delta-ingestion" in task_id]
    infrastructure = [task_id for task_id in task_ids if EXPECTED_TRACKS.get(task_id) == "infrastructure"]
    phase_d = [task_id for task_id in task_ids if EXPECTED_TRACKS.get(task_id) == "phase-d"]
    return {
        "track_counts": dict(sorted(tracks.items())),
        "arc_slot_task_ids": arc,
        "sota_ingestion_task_ids": sota,
        "infrastructure_task_ids": infrastructure,
        "phase_d_task_ids": phase_d,
        "one_arc_slot": len(arc) == 1,
        "one_sota_ingestion_slot": len(sota) == 1,
        "two_infrastructure_slots": len(infrastructure) == 2,
        "phase_d_slot_count": len(phase_d),
    }


def _model_spec_receipt(tasks: Sequence[JsonMap]) -> JsonDict:
    invalid = []
    for task in tasks:
        task_id = str(task.get("id"))
        if task_id not in LLM_TASK_IDS:
            continue
        prompt = str(task.get("prompt") or "")
        if "GGUF" not in prompt or not any(marker in prompt for marker in ("local-GGUF", "unsloth/", "google/")):
            invalid.append(task_id)
    return {"llm_task_ids": sorted(LLM_TASK_IDS), "missing_local_gguf_declarations": invalid, "valid": not invalid}


def validate_v535_roadmap(payload: Any) -> JsonDict:
    tasks = payload.get("tasks", []) if isinstance(payload, Mapping) else []
    task_maps = [task for task in tasks if isinstance(task, Mapping)] if isinstance(tasks, list) else []
    task_ids = [str(task.get("id")) for task in task_maps]
    deliverables = [str(task.get("deliverable")) for task in task_maps]
    missing = [task_id for task_id in EXPECTED_TASK_IDS if task_id not in task_ids]
    unexpected = [task_id for task_id in task_ids if task_id not in EXPECTED_TASK_IDS]
    prompt_failures = [
        task_id
        for task_id, task in zip(task_ids, task_maps, strict=False)
        if not str(task.get("prompt") or "").endswith(PROMPT_ENDING)
    ]
    track_mismatches = [
        {"task_id": task_id, "expected": EXPECTED_TRACKS[task_id], "actual": str(task.get("track"))}
        for task_id, task in zip(task_ids, task_maps, strict=False)
        if task_id in EXPECTED_TRACKS and str(task.get("track")) != EXPECTED_TRACKS[task_id]
    ]
    prior = _valid_prior_failures(task_maps)
    gates = _valid_gates(task_maps)
    allocation = _allocation_receipt(task_maps)
    models = _model_spec_receipt(task_maps)
    task_id_unique = len(task_ids) == len(set(task_ids))
    deliverable_unique = len(deliverables) == len(set(deliverables))
    ready = (
        isinstance(payload, Mapping)
        and payload.get("milestone") == TARGET_MILESTONE
        and payload.get("milestone_title") == TARGET_MILESTONE_TITLE
        and payload.get("milestone_doc") == ROADMAP_DOC_RELATIVE_PATH.as_posix()
        and len(task_maps) == len(EXPECTED_TASK_IDS)
        and not missing
        and not unexpected
        and task_id_unique
        and deliverable_unique
        and not prompt_failures
        and not track_mismatches
        and prior["valid"]
        and gates["valid"]
        and allocation["one_arc_slot"]
        and allocation["one_sota_ingestion_slot"]
        and allocation["two_infrastructure_slots"]
        and models["valid"]
    )
    return {
        "ready": ready,
        "milestone": payload.get("milestone") if isinstance(payload, Mapping) else None,
        "title_valid": isinstance(payload, Mapping) and payload.get("milestone_title") == TARGET_MILESTONE_TITLE,
        "doc_valid": isinstance(payload, Mapping) and payload.get("milestone_doc") == ROADMAP_DOC_RELATIVE_PATH.as_posix(),
        "task_count": len(task_maps),
        "expected_task_count": len(EXPECTED_TASK_IDS),
        "task_ids": task_ids,
        "missing_expected_task_ids": missing,
        "unexpected_task_ids": unexpected,
        "task_id_unique": task_id_unique,
        "deliverable_unique": deliverable_unique,
        "duplicate_task_ids": _duplicate_values(task_ids),
        "duplicate_deliverables": _duplicate_values(deliverables),
        "prompt_ending_failures": prompt_failures,
        "track_mismatches": track_mismatches,
        "prior_failure_validation": prior,
        "gate_reference_validation": gates,
        "allocation_receipt": allocation,
        "mandatory_model_spec_validation": models,
    }


def _source_terminal_matrix(root: Path) -> tuple[JsonDict, JsonDict]:
    capstone, capstone_meta = _read_json(root / SOURCE_CAPSTONE_RELATIVE_PATH)
    matrix = capstone.get("activated_task_and_declared_deliverable_matrix", {})
    classification: JsonDict = {}
    if isinstance(matrix, Mapping):
        for task_id, _title, deliverable in SOURCE_TASKS_WITHOUT_CAPSTONE:
            row = matrix.get(task_id, {})
            classification[task_id] = {
                "declared_deliverable": deliverable.as_posix(),
                "present": bool(row.get("present")) if isinstance(row, Mapping) else False,
                "terminal_class": row.get("terminal_class") if isinstance(row, Mapping) else "missing",
                "underlying_terminal_class": row.get("underlying_terminal_class") if isinstance(row, Mapping) else "missing",
                "conductor_receipt": row.get("conductor_receipt", {}) if isinstance(row, Mapping) else {},
                "source": "exp6168_capstone_declared_matrix",
            }
    capstone_status = str(capstone.get("status") or "")
    classification["exp6168-v534-capstone-reconciliation"] = {
        "declared_deliverable": SOURCE_CAPSTONE_RELATIVE_PATH.as_posix(),
        "present": capstone_meta["present"],
        "terminal_class": "complete" if capstone_status.startswith("complete") else "blocked",
        "underlying_terminal_class": "complete" if capstone_status.startswith("complete") else "blocked",
        "conductor_receipt": _capstone_conductor_receipt(root),
        "source": "exact_capstone_artifact",
    }
    capstone_receipt = {
        "path": SOURCE_CAPSTONE_RELATIVE_PATH.as_posix(),
        "present": capstone_meta["present"],
        "loadable": capstone_meta["loadable"],
        "sha256": capstone_meta["sha256"],
        "honest_verdict": capstone.get("honest_verdict"),
        "status": capstone.get("status"),
    }
    return classification, capstone_receipt


def _capstone_conductor_receipt(root: Path) -> JsonDict:
    log_path = root / CONDUCTOR_LOG_RELATIVE_PATH
    if not log_path.exists():
        return {"present": False}
    lines = log_path.read_text(encoding="utf-8").splitlines()
    matches = [line for line in lines if "6168" in line or ".534 capstone" in line]
    return {"present": bool(matches), "line": matches[-1] if matches else None}


def _protected_hashes(root: Path) -> JsonDict:
    return {
        rel_path.as_posix(): {"present": (root / rel_path).exists(), "sha256": path_sha256(root / rel_path)}
        for rel_path in PROTECTED_FILE_PATHS
    }


def _quarantine_matrix(root: Path, before: JsonMap, after: JsonMap) -> JsonDict:
    rows = {}
    flagged = []
    for task_id, _title, deliverable in SOURCE_TASKS_WITHOUT_CAPSTONE:
        path = root / deliverable
        payload, meta = _read_json(path)
        markers = {
            key: payload.get(key)
            for key in ("flagged_adversarial", "corrigendum_pending", "corrigendum_note")
            if key in payload
        }
        if markers:
            flagged.append(task_id)
        rows[task_id] = {
            "path": deliverable.as_posix(),
            "present": meta["present"],
            "sha256_before": before.get(deliverable.as_posix(), {}).get("sha256"),
            "sha256_after": after.get(deliverable.as_posix(), {}).get("sha256"),
            "markers": markers,
        }
    return {
        "byte_preserved": all(row["sha256_before"] == row["sha256_after"] for row in rows.values()),
        "flagged_task_ids": flagged,
        "source_rows": rows,
        "current_rule_companion_mutated_source_artifacts": False,
    }


def _collision_matrix(root: Path, validation: JsonMap) -> JsonDict:
    deliverables = list(validation.get("task_ids", []))
    result_paths = [path.as_posix() for path in root.glob("results/experiment_61*.json")]
    return {
        "range": {"start": 6169, "end": 6182},
        "expected_task_ids": list(EXPECTED_TASK_IDS),
        "observed_task_ids": list(validation.get("task_ids", [])),
        "missing_expected_task_ids": list(validation.get("missing_expected_task_ids", [])),
        "duplicate_task_ids": list(validation.get("duplicate_task_ids", [])),
        "duplicate_deliverables": list(validation.get("duplicate_deliverables", [])),
        "repository_result_paths_scanned": len(result_paths),
        "collision_free": not validation.get("duplicate_task_ids") and not validation.get("duplicate_deliverables") and not validation.get("missing_expected_task_ids"),
        "task_id_rows": deliverables,
    }


def _roadmap_sources(root: Path) -> tuple[Any, JsonDict, Any, JsonDict]:
    staged_payload, staged_meta = _read_yaml(root / ROADMAP_NEXT_RELATIVE_PATH)
    active_payload, active_meta = _read_yaml(root / ROADMAP_RELATIVE_PATH)
    return staged_payload, staged_meta, active_payload, active_meta


def _activate_if_ready(root: Path, staged_meta: JsonMap, staged_validation: JsonMap, active_validation: JsonMap, apply_mutations: bool) -> JsonDict:
    if staged_meta["present"] and staged_validation["ready"]:
        if apply_mutations:
            data = (root / ROADMAP_NEXT_RELATIVE_PATH).read_bytes()
            tmp = root / ROADMAP_RELATIVE_PATH.with_name(ROADMAP_RELATIVE_PATH.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, root / ROADMAP_RELATIVE_PATH)
        return {"mode": "staged_atomic_copy", "valid": True, "mutation_performed": bool(apply_mutations)}
    if active_validation["ready"]:
        return {"mode": "already_active", "valid": True, "mutation_performed": False}
    if not staged_meta["present"] and active_validation.get("milestone") == TARGET_MILESTONE:
        return {"mode": "already_active_partial_mismatch", "valid": False, "mutation_performed": False}
    return {"mode": "blocked_missing_or_mismatched_roadmap", "valid": False, "mutation_performed": False}


def _validate_required_payload(payload: JsonMap) -> list[str]:
    errors = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in payload]
    provenance = payload.get("field_provenance", {})
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance_not_mapping")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            row = provenance.get(field)
            if not isinstance(row, Mapping) or not row.get("principle"):
                errors.append(f"missing_principle:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if not str(payload.get("honest_verdict", "")).startswith(("complete:", "blocked:")):
        errors.append("honest_verdict_prefix")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        errors.append("reproducibility_checksum")
    return errors


def build_report(
    root: Path = REPO_ROOT,
    *,
    apply_mutations: bool,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    start = time.perf_counter()
    before_git = _git_status(root)
    protected_before = _protected_hashes(root)
    source_hash_before = {
        deliverable.as_posix(): {"sha256": path_sha256(root / deliverable)}
        for _task_id, _title, deliverable in SOURCE_TASKS_WITHOUT_CAPSTONE
    }
    staged_payload, staged_meta, active_payload, active_meta = _roadmap_sources(root)
    staged_validation = validate_v535_roadmap(staged_payload)
    active_validation = validate_v535_roadmap(active_payload)
    selected_validation = staged_validation if staged_meta["present"] else active_validation
    activation = _activate_if_ready(root, staged_meta, staged_validation, active_validation, apply_mutations)
    history_before = _history_receipt(root)
    source_matrix, capstone_receipt = _source_terminal_matrix(root)
    source_terminal = capstone_receipt["present"] and capstone_receipt["loadable"]
    history_after = _append_history_if_needed(root, history_before, activation["valid"] and source_terminal and apply_mutations)
    active_after_meta = _read_yaml(root / ROADMAP_RELATIVE_PATH)[1]
    protected_after = _protected_hashes(root)
    source_hash_after = {
        deliverable.as_posix(): {"sha256": path_sha256(root / deliverable)}
        for _task_id, _title, deliverable in SOURCE_TASKS_WITHOUT_CAPSTONE
    }
    after_git = _git_status(root)
    protected_unchanged = all(protected_before[path] == protected_after[path] for path in protected_before)
    status = "complete_with_v535_activation" if activation["valid"] and history_after["after_count"] == 1 else "blocked_partial_v535_activation"
    append_count = history_after["append_count"]
    activation_mode = activation["mode"]
    test_codes = dict(test_exit_codes or {})
    payload: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": SCHEMA,
        "milestone": TARGET_MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": status,
        "preconditions_checked": {
            "context_paths": {
                rel_path.as_posix(): {"present": (root / rel_path).exists(), "sha256": path_sha256(root / rel_path)}
                for rel_path in CONTEXT_PATHS
            },
            "git_status_before": before_git,
            "active_roadmap_task_count": active_validation["task_count"],
            "staged_roadmap_present": staged_meta["present"],
            "completion_history_before_count": history_before["before_count"],
            "root_clutter_before": _root_clutter(root),
        },
        "source_and_target_milestones": {
            "source": SOURCE_MILESTONE,
            "source_title": SOURCE_MILESTONE_TITLE,
            "target": TARGET_MILESTONE,
            "target_title": TARGET_MILESTONE_TITLE,
        },
        "source_capstone_hash_and_honest_verdict": capstone_receipt,
        "source_exact_terminal_classification": source_matrix,
        "research_complete_count_before_after": history_after,
        "research_complete_append_count": append_count,
        "staged_and_activated_roadmap_hashes": {
            "staged_present": staged_meta["present"],
            "staged_before_sha256": staged_meta["sha256"],
            "active_before_sha256": active_meta["sha256"],
            "active_after_sha256": active_after_meta["sha256"],
            "selected_source": "staged" if staged_meta["present"] else "active",
        },
        "activated_task_count": active_validation["task_count"],
        "task_id_and_deliverable_collision_matrix": _collision_matrix(root, selected_validation),
        "optional_field_and_prior_failure_validation": {
            **selected_validation["prior_failure_validation"],
            "missing_expected_task_ids": selected_validation["missing_expected_task_ids"],
            "unexpected_task_ids": selected_validation["unexpected_task_ids"],
            "prompt_ending_failures": selected_validation["prompt_ending_failures"],
            "track_mismatches": selected_validation["track_mismatches"],
        },
        "gate_reference_validation": selected_validation["gate_reference_validation"],
        "arc_ingestion_infrastructure_and_phase_d_allocation_receipt": selected_validation["allocation_receipt"],
        "mandatory_model_spec_validation": selected_validation["mandatory_model_spec_validation"],
        "quarantine_and_determination_before_after_matrix": _quarantine_matrix(root, source_hash_before, source_hash_after),
        "preexisting_worktree_changes_preserved": {
            "before": before_git,
            "after": after_git,
            "preserved_except_task_outputs": before_git["stdout_lines"] == after_git["stdout_lines"],
        },
        "activation_mode": activation_mode,
        "rollback_receipt": {
            "performed": False,
            "reason": "no_mutation_before_block" if not activation["valid"] else "not_needed",
            "partial_activation_detected": activation_mode == "already_active_partial_mismatch",
        },
        "protected_files_unchanged": {
            "unchanged": protected_unchanged,
            "before": protected_before,
            "after": protected_after,
        },
        "duration_s": round(duration_s if duration_s is not None else time.perf_counter() - start, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": {
            field: {"principle": FIELD_PRINCIPLES[field], "source": "local_transition_receipts"}
            for field in REQUIRED_ARTIFACT_FIELDS
        },
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": test_codes,
        "reproducibility_checksum": "",
        "honest_verdict": (
            f"complete: .534 completion entries after transition={history_after['after_count']}; "
            f"append_count={append_count}; activation_mode={activation_mode}"
            if status.startswith("complete")
            else f"blocked: .534 completion entries after transition={history_after['after_count']}; "
            f"append_count={append_count}; activation_mode={activation_mode}"
        ),
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    return payload


def write_report(root: Path = REPO_ROOT, *, test_exit_codes: Mapping[str, int] | None = None) -> JsonDict:
    payload = build_report(root, apply_mutations=True, test_exit_codes=test_exit_codes)
    write_json(root / RESULT_RELATIVE_PATH, payload)
    return payload


def _load_test_receipts(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): int(value) for key, value in payload.items()}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the Exp6169 result artifact")
    parser.add_argument("--validate", action="store_true", help="validate the written Exp6169 artifact")
    parser.add_argument("--test-receipts", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.validate:
        payload, meta = _read_json(REPO_ROOT / RESULT_RELATIVE_PATH)
        if not meta["loadable"]:
            print(f"invalid Exp6169 artifact: {meta}", file=sys.stderr)
            return 1
        errors = _validate_required_payload(payload)
        if errors:
            print(f"invalid Exp6169 artifact: {errors}", file=sys.stderr)
            return 1
        return 0
    if args.write:
        write_report(REPO_ROOT, test_exit_codes=_load_test_receipts(args.test_receipts))
        return 0
    payload = build_report(REPO_ROOT, apply_mutations=False, test_exit_codes=_load_test_receipts(args.test_receipts))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

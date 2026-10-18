"""Exp5769 transition receipt from terminal milestone .514 into .515.

Spec refs: REQ-REPORT-5769, SCENARIO-REPORT-5769,
SCENARIO-REPORT-5769-COLLISION-BLOCK,
SCENARIO-REPORT-5769-DECLARED-DELIVERABLE-BLOCK,
SCENARIO-REPORT-5769-FIELD-PRINCIPLES.

This module is deliberately a ledger reconciler, not a new experiment. Its job
is to make the evidence boundary boring and inspectable: the canonical V514
artifact for each task is the deliverable path declared in the completion
ledger, while same-number outer-loop ARC induction files are disclosed only as
aliases. That avoids the historical bug where a numeric glob could silently
pick the wrong result file for a task number.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from carnot.experiment_5754_v513_capstone_reconciliation import (
    _read_json_any,
    _read_yaml_mapping,
    path_sha256,
    payload_checksum,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5769_transition_v515.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXPERIMENT = "experiment_5769_transition_v515"
EXPERIMENT_ID = "exp5769-transition-v515"
MILESTONE_FROM = "2026.07.514"
MILESTONE_TO = "2026.07.515"
NEXT_TASK_RANGE = "exp5769-exp5781"
RUN_DATE = "2026-07-22"
RANDOM_SEED = 5769
SCHEMA = "carnot.experiment_5769.transition_v515.v1"
INFERENCE_SUBSTRATE = "cached_artifact_reconciliation_no_llm"
ARTIFACT_SELECTION_POLICY = "exact_declared_deliverable"

SPEC_REFS = (
    "REQ-REPORT-5769",
    "SCENARIO-REPORT-5769",
    "SCENARIO-REPORT-5769-COLLISION-BLOCK",
    "SCENARIO-REPORT-5769-DECLARED-DELIVERABLE-BLOCK",
    "SCENARIO-REPORT-5769-FIELD-PRINCIPLES",
)

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5755-transition-v514": Path("results/experiment_5755_transition_v514.json"),
    "exp5756-v514-source-delta-ingestion": Path(
        "results/experiment_5756_v514_source_delta_ingestion.json"
    ),
    "exp5757-proposal-benchmark-scalar-bridge": Path(
        "results/experiment_5757_proposal_benchmark_scalar_bridge.json"
    ),
    "exp5758-rust-parity-scalar-bridge": Path(
        "results/experiment_5758_rust_parity_scalar_bridge.json"
    ),
    "exp5759-sota-exact-proposal-utility-panel": Path(
        "results/experiment_5759_sota_exact_proposal_utility_panel.json"
    ),
    "exp5760-selective-exact-feedback-search": Path(
        "results/experiment_5760_selective_exact_feedback_search.json"
    ),
    "exp5761-exact-constraint-acquisition-benchmark": Path(
        "results/experiment_5761_exact_constraint_acquisition_benchmark.json"
    ),
    "exp5762-query-driven-constraint-lifecycle": Path(
        "results/experiment_5762_query_driven_constraint_lifecycle.json"
    ),
    "exp5763-dependent-task-constraint-acquisition": Path(
        "results/experiment_5763_dependent_task_constraint_acquisition.json"
    ),
    "exp5764-one-axis-profiled-allocation-free-hot-path": Path(
        "results/experiment_5764_one_axis_profiled_allocation_free_hot_path.json"
    ),
    "exp5765-one-axis-final-10x-crossover": Path(
        "results/experiment_5765_one_axis_final_10x_crossover.json"
    ),
    "exp5766-arc-loo-component-interaction-audit": Path(
        "results/experiment_5766_arc_loo_component_interaction_audit.json"
    ),
    "exp5767-arc-game-blind-composition-hardening": Path(
        "results/experiment_5767_arc_game_blind_composition_hardening.json"
    ),
    "exp5768-v514-capstone-reconciliation": Path(
        "results/experiment_5768_v514_capstone_reconciliation.json"
    ),
}
EXPECTED_TASK_IDS = tuple(TASK_ARTIFACT_PATHS)

NEXT_TASK_IDS = (
    "exp5769-transition-v515",
    "exp5770-v515-source-delta-ingestion",
    "exp5771-evidence-index-collision-preflight",
    "exp5772-sota-constraint-drift-stream",
    "exp5773-prospective-constraint-acquisition-ab",
    "exp5774-constraint-transfer-forgetting-audit",
    "exp5775-constraint-sidecar-shadow-integration",
    "exp5776-arc-world-model-admissibility-contract",
    "exp5777-arc-sota-singleshot-inducer-panel",
    "exp5778-arc-calibrated-world-model-selector",
    "exp5779-arc-live-world-model-generalization-ab",
    "exp5780-hardware-terminal-state-receipt",
    "exp5781-v515-capstone-reconciliation",
)

OUTER_LOOP_ALIAS_PATHS: dict[str, tuple[str, Path]] = {
    "5760": (
        "exp5760-selective-exact-feedback-search",
        Path("results/experiment_5760_cegis_refinement_induction_ab.json"),
    ),
    "5764": (
        "exp5764-one-axis-profiled-allocation-free-hot-path",
        Path("results/experiment_5764_gemma31b_singleshot_induction_ab.json"),
    ),
    "5766": (
        "exp5766-arc-loo-component-interaction-audit",
        Path("results/experiment_5766_gemma31b_cegis_refinement_ab.json"),
    ),
}

GATE_SKIPPED_TASK_IDS = (
    "exp5760-selective-exact-feedback-search",
    "exp5767-arc-game-blind-composition-hardening",
)
BLOCKED_TASK_IDS = ("exp5755-transition-v514",) + GATE_SKIPPED_TASK_IDS
SCIENTIFIC_NULL_TASK_IDS = ("exp5766-arc-loo-component-interaction-audit",)
NEGATIVE_RESULT_TASK_IDS = ("exp5759-sota-exact-proposal-utility-panel",)
RETIRED_TECHNIQUE_IDS = ("exp5765-one-axis-final-10x-crossover",)
DEFAULT_POSITIVE_RESULT_TASK_IDS = (
    "exp5757-proposal-benchmark-scalar-bridge",
    "exp5758-rust-parity-scalar-bridge",
    "exp5761-exact-constraint-acquisition-benchmark",
    "exp5762-query-driven-constraint-lifecycle",
    "exp5763-dependent-task-constraint-acquisition",
    "exp5764-one-axis-profiled-allocation-free-hot-path",
)

PROTECTED_FILE_PATHS = (ROADMAP_RELATIVE_PATH, CONDUCTOR_RELATIVE_PATH)
SELF_OWNED_RELATIVE_PATHS = {
    Path("python/carnot/experiment_5769_transition_v515.py"),
    Path("tests/python/test_experiment_5769_transition_v515.py"),
    RESULT_RELATIVE_PATH,
}
ALLOWED_CONTENT_REFERENCE_PATHS = {
    Path(".coverage"),
    ROADMAP_RELATIVE_PATH,
    VNEXT_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    *SELF_OWNED_RELATIVE_PATHS,
}
IGNORED_SCAN_DIRS = {
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "data",
    "datasets",
    "dist",
    "external",
    "logs",
    "models",
    "node_modules",
    "secrets",
    "target",
}

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/python -c \"import yaml, pathlib; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); yaml.safe_load(pathlib.Path('research-complete.yaml').read_text())\"",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5769_transition_v515.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage run --include=python/carnot/experiment_5769_transition_v515.py -m pytest tests/python/test_experiment_5769_transition_v515.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage report --include=python/carnot/experiment_5769_transition_v515.py --fail-under=100",
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
    {
        "command": ".venv/bin/python scripts/check_spec_coverage.py",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/python scripts/root_clutter_sweep.py",
        "exit_code": None,
        "status": "not_run",
    },
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Identifies the versioned Exp5769 transition artifact schema.",
    "experiment": "Names the local experiment slug without relying on paths.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "status": "Bare terminal state derived from explicit precondition checks.",
    "run_date": "Records the operator-specified transition date as a fixed value.",
    "random_seed": "Deterministic metadata for checksum stability; no stochastic run occurs.",
    "spec_refs": "Anchors the artifact to REQ-REPORT-5769 and its scenarios.",
    "result_path": "Names the emitted deliverable path.",
    "field_principles": "Maps every top-level artifact field to its evidence boundary.",
    "preconditions_checked": "Records roadmap, vNEXT, ledger, hash, alias, collision, and protected-file checks before claims are trusted.",
    "milestone_from": "Names the terminal milestone whose evidence is archived.",
    "milestone_to": "Names the milestone receiving the archived evidence.",
    "declared_deliverable_matrix": "Lists every V514 task with its exact declared deliverable and preserved conductor outcome.",
    "canonical_artifact_hashes": "Hashes only the declared canonical deliverable for each V514 task.",
    "same_number_alias_groups": "Discloses 5760, 5764, and 5766 same-number aliases without selecting them as canonical task evidence.",
    "outer_loop_evidence_hashes": "Hashes outer-loop development-proxy artifacts separately from conductor artifacts.",
    "artifact_selection_policy": "Must equal exact_declared_deliverable to forbid glob or mtime selection.",
    "conductor_outcomes": "Preserves task outcomes from the V514 capstone/conductor evidence boundary.",
    "blocked_task_ids": "Records precondition-blocked and gate-blocked tasks without turning them into scientific failures.",
    "gate_skipped_task_ids": "Separates Exp5760 and Exp5767 conductor gate skips from executed science.",
    "scientific_null_task_ids": "Records only executed scientific zeros, especially the ARC LOO zero.",
    "negative_result_task_ids": "Records executed measured-negative results such as Exp5759.",
    "positive_result_task_ids": "Records bounded positive V514 evidence preserved for downstream work.",
    "retired_technique_ids": "Records the technique-specific one-axis PyO3 10x retirement.",
    "archived_task_ids": "Lists exactly the V514 task denominator carried forward.",
    "research_complete_append_count": "Records that this receipt does not append or rewrite completion history.",
    "collision_scan": "Shows the Exp5769-Exp5781 namespace scan and any unowned collisions.",
    "next_task_range": "Records the destination task interval as exp5769-exp5781.",
    "next_range_collision_count": "Bare count used by downstream gates to reject occupied ids.",
    "docs_reconciled": "Records completion-ledger reconciliation mode without deleting duplicate history.",
    "research_roadmap_unchanged": "Bare boolean must remain true because roadmap activation is conductor-owned.",
    "conductor_unchanged": "Bare boolean must remain true by operator instruction.",
    "inference_substrate": "This transition uses cached local artifacts only; no LLM, solver, benchmark, or hardware run is performed.",
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed verification exits are recorded without relabeling failures.",
    "reproducibility_checksum": "Stable checksum detects artifact drift.",
    "honest_verdict": "Terminal summary starts with complete: or blocked: and does not inflate blocked work into science.",
}


def _roadmap_milestone(root: Path) -> str | None:
    value = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH).get("milestone")
    return value if isinstance(value, str) else None


def _vnext_milestone(root: Path) -> str | None:
    path = root / VNEXT_RELATIVE_PATH
    if not path.exists():
        return None
    match = re.search(r"\*\*Milestone:\*\*\s*`?([^`\n]+)`?", path.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else None


def _research_complete_blocks(root: Path) -> list[JsonDict]:
    payload = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    milestones = payload.get("milestones")
    if not isinstance(milestones, list):
        return []
    return [
        block
        for block in milestones
        if isinstance(block, dict) and block.get("id") == MILESTONE_FROM
    ]


def _task_signature(block: JsonMap) -> tuple[tuple[str, str], ...]:
    tasks = block.get("tasks")
    if not isinstance(tasks, list):
        return ()
    rows: list[tuple[str, str]] = []
    for row in tasks:
        if isinstance(row, Mapping) and isinstance(row.get("id"), str):
            deliverable = row.get("deliverable")
            rows.append((str(row["id"]), str(deliverable) if isinstance(deliverable, str) else ""))
    return tuple(rows)


def _declared_deliverable_matrix(root: Path) -> tuple[list[JsonDict], JsonDict, list[str]]:
    blocks = _research_complete_blocks(root)
    unique_signatures = {_task_signature(block) for block in blocks}
    stats: JsonDict = {
        "research_complete_milestone_from_block_count": len(blocks),
        "unique_declared_deliverable_block_count": len(unique_signatures),
        "declared_deliverables_unambiguous": bool(blocks) and len(unique_signatures) == 1,
    }
    failures: list[str] = []
    if not blocks:
        failures.append("research_complete_514_block_count=0")
    if len(unique_signatures) > 1:
        failures.append("ambiguous_research_complete_declared_task_blocks")

    selected_tasks = blocks[0].get("tasks") if blocks else []
    task_rows = selected_tasks if isinstance(selected_tasks, list) else []
    by_task: dict[str, JsonMap] = {
        str(row["id"]): row
        for row in task_rows
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    declared_ids = tuple(task_id for task_id, _deliverable in next(iter(unique_signatures), ()))
    if blocks and declared_ids != EXPECTED_TASK_IDS:
        failures.append(f"declared_task_ids_mismatch={list(declared_ids)}")

    matrix: list[JsonDict] = []
    mismatches: list[str] = []
    for task_id in EXPECTED_TASK_IDS:
        row = by_task.get(task_id, {})
        declared = row.get("deliverable")
        expected = TASK_ARTIFACT_PATHS[task_id].as_posix()
        declared_path = declared if isinstance(declared, str) else ""
        if declared_path != expected:
            mismatches.append(f"{task_id}:{declared_path or '<missing>'}!={expected}")
        matrix.append(
            {
                "task_id": task_id,
                "title": row.get("title") if isinstance(row.get("title"), str) else "",
                "declared_deliverable": declared_path or expected,
                "research_complete_result": row.get("result")
                if isinstance(row.get("result"), str)
                else "",
                "selection_policy": ARTIFACT_SELECTION_POLICY,
            }
        )
    if mismatches:
        failures.append(f"declared_deliverable_mismatch={mismatches}")
    return matrix, stats, failures


def _payload_status(payload: JsonMap, metadata: JsonMap) -> str:
    if metadata.get("exists") is False:
        return "missing"
    if metadata.get("loadable") is False:
        return "malformed"
    if payload.get("schema") == "blocked_gate_check_v1" or payload.get("blocked_at_layer"):
        return "blocked-gate"
    status = payload.get("status")
    verdict = payload.get("honest_verdict")
    if status == "blocked" or (isinstance(verdict, str) and verdict.startswith("blocked:")):
        return "blocked-precondition"
    if status == "complete" or (isinstance(verdict, str) and verdict.startswith("complete:")):
        return "complete"
    return str(status) if isinstance(status, str) and status else "unknown"


def _canonical_artifact_hashes(root: Path, matrix: Sequence[JsonMap]) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for item in matrix:
        task_id = str(item["task_id"])
        rel_path = Path(str(item["declared_deliverable"]))
        payload, metadata = _read_json_any(root / rel_path)
        rows[task_id] = {
            "path": rel_path.as_posix(),
            "present": bool(metadata.get("exists")),
            "loadable": bool(metadata.get("loadable")),
            "sha256": metadata.get("sha256"),
            "status": _payload_status(payload, metadata),
            "honest_verdict": payload.get("honest_verdict")
            if isinstance(payload.get("honest_verdict"), str)
            else "",
            "selected_by": ARTIFACT_SELECTION_POLICY,
            "error": metadata.get("error"),
        }
    return rows


def _capstone_matrix(root: Path) -> dict[str, JsonDict]:
    payload, _metadata = _read_json_any(
        root / TASK_ARTIFACT_PATHS["exp5768-v514-capstone-reconciliation"]
    )
    matrix = payload.get("task_outcome_matrix")
    if not isinstance(matrix, dict):
        return {}
    return {str(key): value for key, value in matrix.items() if isinstance(value, dict)}


def _conductor_outcomes(
    root: Path,
    canonical_artifact_hashes: Mapping[str, JsonMap],
) -> tuple[dict[str, JsonDict], list[str]]:
    capstone_rows = _capstone_matrix(root)
    outcomes: dict[str, JsonDict] = {}
    missing: list[str] = []
    for task_id in EXPECTED_TASK_IDS:
        capstone = capstone_rows.get(task_id)
        artifact = canonical_artifact_hashes[task_id]
        if capstone is None:
            if task_id != "exp5768-v514-capstone-reconciliation":
                missing.append(task_id)
            capstone = {}
        outcome = capstone.get("conductor_outcome")
        if not isinstance(outcome, str) or not outcome:
            status = str(artifact.get("status"))
            outcome = (
                "GATE_BLOCK"
                if status == "blocked-gate"
                else "FAIL"
                if status != "complete"
                else "OK"
            )
        outcomes[task_id] = {
            "outcome": outcome,
            "artifact_status": artifact.get("status"),
            "artifact_path": artifact.get("path"),
            "evidence_line": capstone.get("evidence_line")
            if isinstance(capstone.get("evidence_line"), str)
            else "",
            "gate_block_reason": capstone.get("gate_block_reason")
            if isinstance(capstone.get("gate_block_reason"), str)
            else None,
            "honest_verdict": capstone.get("honest_verdict")
            if isinstance(capstone.get("honest_verdict"), str)
            else artifact.get("honest_verdict", ""),
            "source": "results/experiment_5768_v514_capstone_reconciliation.json"
            if capstone
            else "artifact_status_fallback",
        }
    return outcomes, missing


def _same_number_alias_groups(
    root: Path,
    canonical_artifact_hashes: Mapping[str, JsonMap],
) -> tuple[dict[str, JsonDict], dict[str, JsonDict], list[str]]:
    groups: dict[str, JsonDict] = {}
    outer_hashes: dict[str, JsonDict] = {}
    missing_aliases: list[str] = []
    for number, (canonical_task_id, outer_path) in OUTER_LOOP_ALIAS_PATHS.items():
        outer_payload, outer_meta = _read_json_any(root / outer_path)
        outer_status = _payload_status(outer_payload, outer_meta)
        canonical = canonical_artifact_hashes[canonical_task_id]
        canonical_entry = {
            "task_id": canonical_task_id,
            "path": canonical["path"],
            "sha256": canonical["sha256"],
            "status": canonical["status"],
            "honest_verdict": canonical["honest_verdict"],
            "evidence_role": "v514_declared_conductor_task",
        }
        outer_entry = {
            "path": outer_path.as_posix(),
            "present": bool(outer_meta.get("exists")),
            "loadable": bool(outer_meta.get("loadable")),
            "sha256": outer_meta.get("sha256"),
            "status": outer_status,
            "honest_verdict": outer_payload.get("honest_verdict")
            if isinstance(outer_payload.get("honest_verdict"), str)
            else "",
            "evidence_role": "outer_loop_development_proxy_alias",
            "error": outer_meta.get("error"),
        }
        if outer_status in {"missing", "malformed"}:
            missing_aliases.append(number)
        groups[number] = {
            "experiment_number": number,
            "canonical": canonical_entry,
            "outer_loop": outer_entry,
            "policy": "disclose_without_conflation",
        }
        outer_hashes[outer_path.as_posix()] = outer_entry
    return groups, outer_hashes, missing_aliases


def _next_range_tokens() -> tuple[str, ...]:
    tokens: list[str] = []
    for task_id in NEXT_TASK_IDS:
        number = re.match(r"exp(\d+)", task_id)
        if number:
            tokens.append(number.group(0))
            tokens.append(f"experiment_{number.group(1)}")
        tokens.append(task_id)
    return tuple(tokens)


def _matches_next_range(text: str) -> bool:
    return any(token in text for token in _next_range_tokens())


def _repo_files(root: Path) -> list[Path]:
    rows: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in IGNORED_SCAN_DIRS)
        base = Path(dirpath)
        for filename in sorted(filenames):
            rows.append((base / filename).relative_to(root))
    return rows


def _collision_scan(root: Path) -> JsonDict:
    collisions: list[JsonDict] = []
    for rel_path in _repo_files(root):
        if rel_path in ALLOWED_CONTENT_REFERENCE_PATHS:
            continue
        rel_text = rel_path.as_posix()
        name_matches = _matches_next_range(rel_text)
        if name_matches:
            collisions.append({"path": rel_text, "kind": "preexisting_file_name"})
            continue
        path = root / rel_path
        try:
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if _matches_next_range(text):
            collisions.append({"path": rel_text, "kind": "preexisting_content_reference"})
    collisions = sorted(collisions, key=lambda row: (str(row["path"]), str(row["kind"])))
    return {
        "next_task_ids": list(NEXT_TASK_IDS),
        "allowed_reference_paths": sorted(
            path.as_posix() for path in ALLOWED_CONTENT_REFERENCE_PATHS
        ),
        "preexisting_collisions": collisions,
        "preexisting_collision_count": len(collisions),
        "collision_free": not collisions,
    }


def _git_modified(root: Path, rel_path: Path) -> bool:  # pragma: no cover - live repo check
    result = subprocess.run(
        ["git", "status", "--short", "--", rel_path.as_posix()],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    return bool(result.stdout.strip())


def _protected_files(
    root: Path,
    modification_overrides: Mapping[Path, bool] | None,
) -> dict[str, JsonDict]:
    rows: dict[str, JsonDict] = {}
    for rel_path in PROTECTED_FILE_PATHS:
        if modification_overrides is not None and rel_path in modification_overrides:
            modified = bool(modification_overrides[rel_path])
            source = "test_override"
        else:  # pragma: no cover - live artifact generation uses git status
            modified = _git_modified(root, rel_path)
            source = "git_status"
        rows[rel_path.as_posix()] = {
            "present": (root / rel_path).exists(),
            "sha256": path_sha256(root / rel_path),
            "modified_by_exp5769": modified,
            "check_source": source,
        }
    return rows


def _positive_result_task_ids(root: Path) -> list[str]:
    payload, _metadata = _read_json_any(
        root / TASK_ARTIFACT_PATHS["exp5768-v514-capstone-reconciliation"]
    )
    value = payload.get("promoted_task_ids")
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item for item in value if item in EXPECTED_TASK_IDS]
    return list(DEFAULT_POSITIVE_RESULT_TASK_IDS)


def _test_exit_codes(tests_run: Sequence[JsonMap]) -> JsonDict:
    return {str(row.get("command")): row.get("exit_code") for row in tests_run}


def _load_tests_run(path: Path | None) -> list[JsonDict]:
    if path is None:
        return [dict(row) for row in DEFAULT_TESTS_RUN]
    payload = json.loads(path.read_text(encoding="utf-8"))  # pragma: no cover - CLI convenience
    if not isinstance(payload, list):  # pragma: no cover - CLI convenience
        raise ValueError("tests-run JSON must be a list")
    return [dict(row) for row in payload]  # pragma: no cover - CLI convenience


def build_report(
    root: Path = REPO_ROOT,
    *,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
) -> JsonDict:
    declared_matrix, complete_stats, matrix_failures = _declared_deliverable_matrix(root)
    canonical_hashes = _canonical_artifact_hashes(root, declared_matrix)
    conductor_outcomes, missing_outcome_task_ids = _conductor_outcomes(root, canonical_hashes)
    alias_groups, outer_loop_hashes, missing_aliases = _same_number_alias_groups(
        root, canonical_hashes
    )
    collision_scan = _collision_scan(root)
    protected_files = _protected_files(root, modification_overrides)
    roadmap_milestone = _roadmap_milestone(root)
    vnext_milestone = _vnext_milestone(root)

    missing_or_malformed = [
        task_id
        for task_id, row in canonical_hashes.items()
        if row["status"] in {"missing", "malformed"}
    ]
    research_roadmap_unchanged = not protected_files[ROADMAP_RELATIVE_PATH.as_posix()][
        "modified_by_exp5769"
    ]
    conductor_unchanged = not protected_files[CONDUCTOR_RELATIVE_PATH.as_posix()][
        "modified_by_exp5769"
    ]

    failed_preconditions = list(matrix_failures)
    if roadmap_milestone != MILESTONE_TO:
        failed_preconditions.append(f"active_roadmap_milestone={roadmap_milestone!r}")
    if vnext_milestone != MILESTONE_TO:
        failed_preconditions.append(f"vnext_milestone={vnext_milestone!r}")
    if missing_or_malformed:
        failed_preconditions.append(
            f"missing_or_malformed_declared_deliverables={missing_or_malformed}"
        )
    if missing_outcome_task_ids:
        failed_preconditions.append(f"missing_conductor_outcomes={missing_outcome_task_ids}")
    if missing_aliases:
        failed_preconditions.append(f"missing_or_malformed_alias_groups={missing_aliases}")
    if collision_scan["preexisting_collision_count"]:
        failed_preconditions.append(
            f"next_range_collision_count={collision_scan['preexisting_collision_count']}"
        )
    if not research_roadmap_unchanged:
        failed_preconditions.append("research_roadmap_modified")
    if not conductor_unchanged:
        failed_preconditions.append("research_conductor_modified")

    status = "blocked" if failed_preconditions else "complete"
    run_rows = [dict(row) for row in (tests_run if tests_run is not None else DEFAULT_TESTS_RUN)]
    docs_mode = (
        "already_archived_preserving_duplicate_history_no_rewrite"
        if complete_stats["research_complete_milestone_from_block_count"] > 0
        else "blocked_missing_v514_completion_block_no_rewrite"
    )

    matrix_with_outcomes: list[JsonDict] = []
    for row in declared_matrix:
        task_id = str(row["task_id"])
        merged = dict(row)
        merged["canonical_artifact_path"] = canonical_hashes[task_id]["path"]
        merged["canonical_artifact_sha256"] = canonical_hashes[task_id]["sha256"]
        merged["canonical_artifact_status"] = canonical_hashes[task_id]["status"]
        merged["conductor_outcome"] = conductor_outcomes[task_id]["outcome"]
        merged["conductor_evidence_line"] = conductor_outcomes[task_id]["evidence_line"]
        matrix_with_outcomes.append(merged)

    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH.as_posix(),
        "field_principles": {},
        "preconditions_checked": {
            "active_roadmap_milestone": roadmap_milestone,
            "active_roadmap_names_milestone_from": roadmap_milestone == MILESTONE_FROM,
            "active_roadmap_names_milestone_to": roadmap_milestone == MILESTONE_TO,
            "vnext_milestone": vnext_milestone,
            "vnext_names_milestone_to": vnext_milestone == MILESTONE_TO,
            "declared_deliverable_count": len(declared_matrix),
            "canonical_artifact_count": len(canonical_hashes),
            "canonical_hash_count": sum(1 for row in canonical_hashes.values() if row["sha256"]),
            "alias_group_count": len(alias_groups),
            "outer_loop_hash_count": sum(1 for row in outer_loop_hashes.values() if row["sha256"]),
            "next_range_collision_count": collision_scan["preexisting_collision_count"],
            "research_roadmap_unchanged": research_roadmap_unchanged,
            "conductor_unchanged": conductor_unchanged,
            **complete_stats,
            "failed_preconditions": failed_preconditions,
        },
        "milestone_from": MILESTONE_FROM,
        "milestone_to": MILESTONE_TO,
        "declared_deliverable_matrix": matrix_with_outcomes,
        "canonical_artifact_hashes": canonical_hashes,
        "same_number_alias_groups": alias_groups,
        "outer_loop_evidence_hashes": outer_loop_hashes,
        "artifact_selection_policy": ARTIFACT_SELECTION_POLICY,
        "conductor_outcomes": conductor_outcomes,
        "blocked_task_ids": list(BLOCKED_TASK_IDS),
        "gate_skipped_task_ids": list(GATE_SKIPPED_TASK_IDS),
        "scientific_null_task_ids": list(SCIENTIFIC_NULL_TASK_IDS),
        "negative_result_task_ids": list(NEGATIVE_RESULT_TASK_IDS),
        "positive_result_task_ids": _positive_result_task_ids(root),
        "retired_technique_ids": list(RETIRED_TECHNIQUE_IDS),
        "archived_task_ids": list(EXPECTED_TASK_IDS),
        "research_complete_append_count": 0,
        "collision_scan": collision_scan,
        "next_task_range": NEXT_TASK_RANGE,
        "next_range_collision_count": collision_scan["preexisting_collision_count"],
        "docs_reconciled": {
            "mode": docs_mode,
            "research_complete_append_count": 0,
            "research_complete_milestone_from_block_count": complete_stats[
                "research_complete_milestone_from_block_count"
            ],
            "unique_declared_deliverable_block_count": complete_stats[
                "unique_declared_deliverable_block_count"
            ],
            "files_modified": [],
        },
        "research_roadmap_unchanged": research_roadmap_unchanged,
        "conductor_unchanged": conductor_unchanged,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": [str(row.get("command")) for row in run_rows],
        "test_exit_codes": _test_exit_codes(run_rows),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "blocked: exp5769 transition preconditions failed: " + "; ".join(failed_preconditions)
            if failed_preconditions
            else (
                "complete: archived terminal .514 evidence by exact declared deliverables "
                "into .515; same-number aliases disclosed; next_range_collision_count=0; "
                "research_complete_append_count=0"
            )
        ),
    }
    missing_principles = [field for field in artifact if field not in FIELD_PRINCIPLES]
    if missing_principles:
        raise KeyError(f"missing field principles: {missing_principles}")
    artifact["field_principles"] = {field: FIELD_PRINCIPLES[field] for field in artifact}
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def emit_report(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    tests_run: Sequence[JsonMap] | None = None,
    modification_overrides: Mapping[Path, bool] | None = None,
) -> JsonDict:
    artifact = build_report(
        root, tests_run=tests_run, modification_overrides=modification_overrides
    )
    write_json(output_path or root / RESULT_RELATIVE_PATH, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tests-run-json", type=Path, default=None)
    args = parser.parse_args(argv)
    emit_report(args.root, output_path=args.output, tests_run=_load_tests_run(args.tests_run_json))
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

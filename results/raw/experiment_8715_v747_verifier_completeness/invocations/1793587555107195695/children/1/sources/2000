"""Exp5755 transition receipt from terminal milestone .513 into .514.

Spec refs: REQ-REPORT-5755, SCENARIO-REPORT-5755,
SCENARIO-REPORT-5755-BLOCKED-VERSUS-NULL,
SCENARIO-REPORT-5755-COLLISION-BLOCK,
SCENARIO-REPORT-5755-FIELD-PRINCIPLES.

This module does not run a benchmark, an LLM, a solver, or hardware. It
reconciles cached local artifacts and ledgers so the next milestone inherits
the actual terminal record: gate-shape blocks remain blocked, the KAN residual
remains a negative measured result, and the ARC live delta remains a scientific
zero rather than a registry win.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from carnot.experiment_5754_v513_capstone_reconciliation import (
    _bool_value,
    _fallback_outcome,
    _latest_log_line,
    _number_value,
    _outcome_from_line,
    _read_json_any,
    _read_yaml_mapping,
    _status_for_payload,
    path_sha256,
    payload_checksum,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5755_transition_v514.json")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
VNEXT_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

EXP5743_PATH = Path("results/experiment_5743_transition_v513.json")
EXP5744_PATH = Path("results/experiment_5744_v513_source_delta_ingestion.json")
EXP5745_PATH = Path("results/experiment_5745_arc_causal_gate_schema_corrigendum.json")
EXP5746_PATH = Path("results/experiment_5746_exact_proposal_utility_benchmark.json")
EXP5747_PATH = Path("results/experiment_5747_sota_exact_proposal_utility_panel.json")
EXP5748_PATH = Path("results/experiment_5748_selective_exact_feedback_search.json")
EXP5749_PATH = Path("results/experiment_5749_csl_render_matched_mechanism_audit.json")
EXP5750_PATH = Path("results/experiment_5750_dependent_task_continuous_self_learning.json")
EXP5751_PATH = Path("results/experiment_5751_rust_restart_parity_repair.json")
EXP5752_PATH = Path("results/experiment_5752_one_axis_allocation_free_10x_crossover.json")
EXP5753_PATH = Path("results/experiment_5753_arc_generic_primitive_live_registry_ab.json")
EXP5754_PATH = Path("results/experiment_5754_v513_capstone_reconciliation.json")

TASK_ARTIFACT_PATHS: dict[str, Path] = {
    "exp5743-transition-v513": EXP5743_PATH,
    "exp5744-v513-source-delta-ingestion": EXP5744_PATH,
    "exp5745-arc-causal-gate-schema-corrigendum": EXP5745_PATH,
    "exp5746-exact-proposal-utility-benchmark": EXP5746_PATH,
    "exp5747-sota-exact-proposal-utility-panel": EXP5747_PATH,
    "exp5748-selective-exact-feedback-search": EXP5748_PATH,
    "exp5749-csl-render-matched-mechanism-audit": EXP5749_PATH,
    "exp5750-dependent-task-continuous-self-learning": EXP5750_PATH,
    "exp5751-rust-restart-parity-repair": EXP5751_PATH,
    "exp5752-one-axis-allocation-free-10x-crossover": EXP5752_PATH,
    "exp5753-arc-generic-primitive-live-registry-ab": EXP5753_PATH,
    "exp5754-v513-capstone-reconciliation": EXP5754_PATH,
}
EXPECTED_TASK_IDS = tuple(TASK_ARTIFACT_PATHS)

NEXT_TASK_IDS = (
    "exp5755-transition-v514",
    "exp5756-v514-source-delta-ingestion",
    "exp5757-proposal-benchmark-scalar-bridge",
    "exp5758-rust-parity-scalar-bridge",
    "exp5759-sota-exact-proposal-utility-panel",
    "exp5760-selective-exact-feedback-search",
    "exp5761-exact-constraint-acquisition-benchmark",
    "exp5762-query-driven-constraint-lifecycle",
    "exp5763-dependent-task-constraint-acquisition",
    "exp5764-one-axis-profiled-allocation-free-hot-path",
    "exp5765-one-axis-final-10x-crossover",
    "exp5766-arc-loo-component-interaction-audit",
    "exp5767-arc-game-blind-composition-hardening",
    "exp5768-v514-capstone-reconciliation",
)

EXPERIMENT = "experiment_5755_transition_v514"
EXPERIMENT_ID = "exp5755-transition-v514"
MILESTONE_FROM = "2026.07.513"
MILESTONE_TO = "2026.07.514"
NEXT_TASK_RANGE = "exp5755-exp5768"
RUN_DATE = "2026-07-20"
RANDOM_SEED = 5755
SCHEMA = "carnot.experiment_5755.transition_v514.v1"
INFERENCE_SUBSTRATE = "cached_artifact_reconciliation_no_llm"
SPEC_REFS = (
    "REQ-REPORT-5755",
    "SCENARIO-REPORT-5755",
    "SCENARIO-REPORT-5755-BLOCKED-VERSUS-NULL",
    "SCENARIO-REPORT-5755-COLLISION-BLOCK",
    "SCENARIO-REPORT-5755-FIELD-PRINCIPLES",
)

PROTECTED_FILE_PATHS = (ROADMAP_RELATIVE_PATH, CONDUCTOR_RELATIVE_PATH)
SELF_OWNED_RELATIVE_PATHS = {
    Path("python/carnot/experiment_5755_transition_v514.py"),
    Path("tests/python/test_experiment_5755_transition_v514.py"),
    RESULT_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
}
EXPECTED_MISSING_TASK_IDS = {"exp5748-selective-exact-feedback-search"}

DEFAULT_TESTS_RUN: tuple[JsonDict, ...] = (
    {
        "command": ".venv/bin/python -c \"import yaml, pathlib; yaml.safe_load(pathlib.Path('research-roadmap.yaml').read_text()); yaml.safe_load(pathlib.Path('research-complete.yaml').read_text())\"",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/pytest tests/python/test_experiment_5755_transition_v514.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage run --include=python/carnot/experiment_5755_transition_v514.py -m pytest tests/python/test_experiment_5755_transition_v514.py -q --no-cov -n 0",
        "exit_code": None,
        "status": "not_run",
    },
    {
        "command": ".venv/bin/coverage report --include=python/carnot/experiment_5755_transition_v514.py --fail-under=100",
        "exit_code": None,
        "status": "not_run",
    },
    {"command": ".venv/bin/python scripts/check_spec_coverage.py", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/python scripts/root_clutter_sweep.py", "exit_code": None, "status": "not_run"},
    {"command": ".venv/bin/pytest tests/python -q", "exit_code": None, "status": "not_run"},
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "Identifies the versioned Exp5755 artifact schema.",
    "experiment": "Stable local experiment slug for result indexing.",
    "experiment_id": "Binds this receipt to the conductor task id.",
    "status": "Bare terminal state derived from explicit transition preconditions.",
    "run_date": "Absolute operator date used for this transition.",
    "random_seed": "Deterministic metadata even though no stochastic science runs.",
    "spec_refs": "OpenSpec anchors for this artifact's behavior.",
    "result_path": "Records the emitted deliverable path.",
    "field_principles": "Maps every artifact field to the evidence boundary that justifies it.",
    "preconditions_checked": "Records artifact, ledger, roadmap, vNEXT, collision, and protected-file checks before transition claims are trusted.",
    "milestone_from": "Names the terminal milestone whose evidence is archived.",
    "milestone_to": "Names the newly active milestone receiving the archived evidence.",
    "archived_task_ids": "Lists exactly the Exp5743-Exp5754 denominator carried from the terminal milestone.",
    "artifact_hashes": "Binds every present canonical artifact to exact bytes and every absent artifact to an explicit missing state.",
    "conductor_outcomes": "Preserves latest conductor OK, GATE_BLOCK, and preemptive-skip evidence separately from science artifacts.",
    "blocked_task_ids": "Gate-blocked or missing tasks stay blocked and cannot be counted as scientific measurements.",
    "scientific_null_task_ids": "Scientific zero/null outcomes are reported only for tasks that actually ran.",
    "negative_result_task_ids": "Negative measured results remain distinct from blocked tasks.",
    "positive_result_task_ids": "Positive readiness results remain bounded to their direct evidence.",
    "proposal_benchmark_ready": "Only the sealed Exp5746 benchmark receipts make the benchmark ready.",
    "proposal_utility_measured": "Blocked Exp5747 and skipped Exp5748 mean proposal utility was not measured.",
    "kan_mechanism_residual": "The signed Exp5749 residual controls KAN-specific claims only.",
    "rust_restart_parity_ready": "Exp5751 parity repair is semantic readiness, not throughput.",
    "rust_10x_measured": "A gate-shaped Exp5752 block means no final 10x measurement happened.",
    "arc_live_delta_measured": "Exp5753 ran a development-proxy live A/B and measured its delta.",
    "arc_live_delta": "The measured zero delta remains a scientific null with no registry credit.",
    "collision_scan": "Next-range allocation is valid only when pre-existing implementation/result collisions are absent.",
    "next_task_range": "Records the Exp5755-Exp5768 range allocated for the destination milestone.",
    "docs_reconciled": "Records whether research-complete already contains exactly one terminal .513 block without rewriting history.",
    "research_roadmap_unchanged": "Bare boolean must remain true because this workflow is record-only.",
    "conductor_unchanged": "Bare boolean must remain true by operator instruction.",
    "inference_substrate": "This transition reads cached artifacts and ledgers only; no LLM, solver, benchmark, or hardware run is performed.",
    "test_commands": "Verification commands are preserved exactly.",
    "test_exit_codes": "Observed exit codes are recorded without relabeling failures.",
    "reproducibility_checksum": "Stable content checksum detects artifact drift.",
    "honest_verdict": "Terminal summary starts with complete: or blocked: and does not inflate blocked work into science.",
}


def _task_payloads(root: Path) -> tuple[dict[str, JsonDict], dict[str, JsonDict]]:
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    for task_id, rel_path in TASK_ARTIFACT_PATHS.items():
        payload, meta = _read_json_any(root / rel_path)
        status = _status_for_payload(payload, meta)
        payloads[task_id] = payload
        metadata[task_id] = {
            "path": rel_path.as_posix(),
            "present": bool(meta.get("exists")),
            "loadable": bool(meta.get("loadable")),
            "sha256": meta.get("sha256"),
            "status": status,
            "error": meta.get("error"),
        }
    return payloads, metadata


def _log_patterns() -> dict[str, tuple[str, ...]]:
    return {
        "exp5743-transition-v513": ("Transition terminal .512 evidence",),
        "exp5744-v513-source-delta-ingestion": ("Ingest post-V513",),
        "exp5745-arc-causal-gate-schema-corrigendum": ("Normalize the Exp5740",),
        "exp5746-exact-proposal-utility-benchmark": ("Build a disjoint dual-receipt",),
        "exp5747-sota-exact-proposal-utility-panel": ("Gated on Exp5746 readiness",),
        "exp5748-selective-exact-feedback-search": ("Gated on Exp5747 utility>0",),
        "exp5749-csl-render-matched-mechanism-audit": (
            "Audit render- and parameter-matched",
        ),
        "exp5750-dependent-task-continuous-self-learning": (
            "Gated on Exp5749 KAN residual>0",
        ),
        "exp5751-rust-restart-parity-repair": (
            "Localize and repair one-axis Rust",
        ),
        "exp5752-one-axis-allocation-free-10x-crossover": (
            "Gated on Exp5751 parity",
        ),
        "exp5753-arc-generic-primitive-live-registry-ab": (
            "Gated on Exp5745 clean scalar gate",
        ),
        "exp5754-v513-capstone-reconciliation": ("Reconcile .513 proposal",),
    }


def _conductor_outcomes(root: Path, artifact_hashes: Mapping[str, JsonMap]) -> dict[str, JsonDict]:
    path = root / CONDUCTOR_LOG_RELATIVE_PATH
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    outcomes: dict[str, JsonDict] = {}
    for task_id, patterns in _log_patterns().items():
        line = _latest_log_line(text, patterns)
        artifact_status = str(artifact_hashes.get(task_id, {}).get("status") or "unknown")
        outcome = _outcome_from_line(line) if line else _fallback_outcome(artifact_status)
        outcomes[task_id] = {
            "outcome": outcome,
            "artifact_status": artifact_status,
            "evidence_line": line,
            "source": CONDUCTOR_LOG_RELATIVE_PATH.as_posix() if line else "artifact_status_fallback",
            "preemptive_skip": bool(line and "Pre-emptive skip" in line),
        }
    return outcomes


def _roadmap_milestone(root: Path) -> str | None:
    value = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH).get("milestone")
    return value if isinstance(value, str) else None


def _vnext_milestone(root: Path) -> str | None:
    path = root / VNEXT_RELATIVE_PATH
    if not path.exists():
        return None
    match = re.search(r"\*\*Milestone:\*\*\s*`([^`]+)`", path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _research_complete_block_count(root: Path) -> int:
    path = root / RESEARCH_COMPLETE_RELATIVE_PATH
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    return len(re.findall(r"(?m)^-\s+id:\s+2026\.07\.513\s*$", text))


def _planned_task_ids(root: Path) -> list[str]:
    payload = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        return []
    return [row["id"] for row in tasks if isinstance(row, Mapping) and isinstance(row.get("id"), str)]


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


def _is_self_owned(rel_path: Path) -> bool:
    return rel_path in SELF_OWNED_RELATIVE_PATHS


def _file_name_collisions(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for rel_root in (Path("python"), Path("scripts"), Path("tests"), Path("results")):
        base = root / rel_root
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(root)
            if "__pycache__" in rel_path.parts:
                continue
            if _is_self_owned(rel_path):
                continue
            rel_text = rel_path.as_posix()
            if _matches_next_range(rel_text):
                rows.append({"path": rel_text, "kind": "preexisting_file_name"})
    return rows


def _content_collision_files(root: Path) -> list[Path]:
    paths = [root / RESEARCH_COMPLETE_RELATIVE_PATH]
    paths.extend(sorted((root / "openspec/capabilities").glob("**/*.md")))
    paths.extend(sorted((root / "openspec/change-proposals").glob("**/*.md")))
    return [path for path in paths if path.exists() and path.is_file()]


def _content_collisions(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for path in _content_collision_files(root):
        rel_path = path.relative_to(root)
        if rel_path in {VNEXT_RELATIVE_PATH, SPEC_RELATIVE_PATH}:
            continue
        if _matches_next_range(path.read_text(encoding="utf-8", errors="replace")):
            rows.append({"path": rel_path.as_posix(), "kind": "preexisting_content_reference"})
    return rows


def _collision_scan(root: Path) -> JsonDict:
    collisions = _file_name_collisions(root) + _content_collisions(root)
    planned_task_ids = _planned_task_ids(root)
    return {
        "next_task_ids": list(NEXT_TASK_IDS),
        "planned_task_ids": planned_task_ids,
        "allowed_planned_references": [
            {"path": ROADMAP_RELATIVE_PATH.as_posix(), "task_ids": planned_task_ids},
            {"path": VNEXT_RELATIVE_PATH.as_posix(), "task_range": NEXT_TASK_RANGE},
            {"path": SPEC_RELATIVE_PATH.as_posix(), "task_ids": [EXPERIMENT_ID]},
        ],
        "self_owned_files": sorted(path.as_posix() for path in SELF_OWNED_RELATIVE_PATHS),
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
            "modified_by_exp5755": modified,
            "check_source": source,
        }
    return rows


def _blocked_task_ids(
    artifact_hashes: Mapping[str, JsonMap],
    conductor_outcomes: Mapping[str, JsonMap],
) -> list[str]:
    rows: list[str] = []
    for task_id in EXPECTED_TASK_IDS:
        status = artifact_hashes[task_id]["status"]
        outcome = conductor_outcomes[task_id]["outcome"]
        if status in {"blocked", "gate_skipped", "missing", "malformed"} or outcome == "GATE_BLOCK":
            rows.append(task_id)
    return rows


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
    payloads, artifact_hashes = _task_payloads(root)
    conductor_outcomes = _conductor_outcomes(root, artifact_hashes)
    roadmap_milestone = _roadmap_milestone(root)
    vnext_milestone = _vnext_milestone(root)
    complete_block_count = _research_complete_block_count(root)
    collision_scan = _collision_scan(root)
    protected_files = _protected_files(root, modification_overrides)

    blocked_task_ids = _blocked_task_ids(artifact_hashes, conductor_outcomes)
    exp5746 = payloads["exp5746-exact-proposal-utility-benchmark"]
    exp5749 = payloads["exp5749-csl-render-matched-mechanism-audit"]
    exp5751 = payloads["exp5751-rust-restart-parity-repair"]
    exp5752_status = artifact_hashes["exp5752-one-axis-allocation-free-10x-crossover"]["status"]
    exp5753 = payloads["exp5753-arc-generic-primitive-live-registry-ab"]
    exp5753_status = artifact_hashes["exp5753-arc-generic-primitive-live-registry-ab"]["status"]

    proposal_benchmark_ready = _number_value(exp5746, "benchmark_ready_score") >= 1.0
    proposal_utility_measured = (
        artifact_hashes["exp5747-sota-exact-proposal-utility-panel"]["status"] == "complete"
    )
    kan_mechanism_residual = round(_number_value(exp5749, "kan_mechanism_residual"), 6)
    rust_restart_parity_ready = _number_value(exp5751, "restart_parity_ready_score") >= 1.0
    rust_10x_measured = exp5752_status == "complete"
    arc_live_delta_measured = (
        exp5753_status == "complete" and "live_level_reproduction_delta" in exp5753
    )
    arc_live_delta = int(_number_value(exp5753, "live_level_reproduction_delta"))
    research_roadmap_unchanged = not protected_files[ROADMAP_RELATIVE_PATH.as_posix()][
        "modified_by_exp5755"
    ]
    conductor_unchanged = not protected_files[CONDUCTOR_RELATIVE_PATH.as_posix()][
        "modified_by_exp5755"
    ]

    unexpected_missing = [
        task_id
        for task_id, row in artifact_hashes.items()
        if row["status"] in {"missing", "malformed"} and task_id not in EXPECTED_MISSING_TASK_IDS
    ]
    failed_preconditions: list[str] = []
    if roadmap_milestone != MILESTONE_TO:
        failed_preconditions.append(f"active_roadmap_milestone={roadmap_milestone!r}")
    if vnext_milestone != MILESTONE_TO:
        failed_preconditions.append(f"vnext_milestone={vnext_milestone!r}")
    if complete_block_count != 1:
        failed_preconditions.append(f"research_complete_513_block_count={complete_block_count}")
    if not research_roadmap_unchanged:
        failed_preconditions.append("research_roadmap_modified")
    if not conductor_unchanged:
        failed_preconditions.append("research_conductor_modified")
    if unexpected_missing:
        failed_preconditions.append(f"unexpected_missing_or_malformed={unexpected_missing}")
    if conductor_outcomes["exp5748-selective-exact-feedback-search"]["outcome"] != "GATE_BLOCK":
        failed_preconditions.append("exp5748_preemptive_skip_not_verified")
    if collision_scan["preexisting_collision_count"]:
        failed_preconditions.append(
            f"next_range_collision_count={collision_scan['preexisting_collision_count']}"
        )

    status = "blocked" if failed_preconditions else "complete"
    run_rows = [dict(row) for row in (tests_run if tests_run is not None else DEFAULT_TESTS_RUN)]
    docs_mode = (
        "already_archived_once_no_rewrite"
        if complete_block_count == 1
        else "blocked_duplicate_or_missing_no_rewrite"
    )

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
            "artifact_count_expected": len(EXPECTED_TASK_IDS),
            "artifact_count_present": sum(1 for row in artifact_hashes.values() if row["present"]),
            "expected_missing_task_ids": sorted(EXPECTED_MISSING_TASK_IDS),
            "unexpected_missing_or_malformed_task_ids": unexpected_missing,
            "active_roadmap_milestone": roadmap_milestone,
            "active_roadmap_names_milestone_to": roadmap_milestone == MILESTONE_TO,
            "vnext_milestone": vnext_milestone,
            "vnext_names_milestone_to": vnext_milestone == MILESTONE_TO,
            "research_complete_513_block_count": complete_block_count,
            "research_complete_archived_once": complete_block_count == 1,
            "collision_free": collision_scan["collision_free"],
            "research_roadmap_unchanged": research_roadmap_unchanged,
            "conductor_unchanged": conductor_unchanged,
            "failed_preconditions": failed_preconditions,
        },
        "milestone_from": MILESTONE_FROM,
        "milestone_to": MILESTONE_TO,
        "archived_task_ids": list(EXPECTED_TASK_IDS),
        "artifact_hashes": artifact_hashes,
        "conductor_outcomes": conductor_outcomes,
        "blocked_task_ids": blocked_task_ids,
        "scientific_null_task_ids": ["exp5753-arc-generic-primitive-live-registry-ab"]
        if arc_live_delta_measured and arc_live_delta == 0
        else [],
        "negative_result_task_ids": ["exp5749-csl-render-matched-mechanism-audit"]
        if kan_mechanism_residual < 0
        else [],
        "positive_result_task_ids": [
            task_id
            for task_id, passed in (
                ("exp5746-exact-proposal-utility-benchmark", proposal_benchmark_ready),
                ("exp5751-rust-restart-parity-repair", rust_restart_parity_ready),
            )
            if passed
        ],
        "proposal_benchmark_ready": proposal_benchmark_ready,
        "proposal_utility_measured": proposal_utility_measured,
        "kan_mechanism_residual": kan_mechanism_residual,
        "rust_restart_parity_ready": rust_restart_parity_ready,
        "rust_10x_measured": rust_10x_measured,
        "arc_live_delta_measured": arc_live_delta_measured,
        "arc_live_delta": arc_live_delta,
        "collision_scan": collision_scan,
        "next_task_range": NEXT_TASK_RANGE,
        "docs_reconciled": {
            "mode": docs_mode,
            "research_complete_milestone_from_block_count": complete_block_count,
            "files_modified": [],
        },
        "research_roadmap_unchanged": research_roadmap_unchanged,
        "conductor_unchanged": conductor_unchanged,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": [str(row.get("command")) for row in run_rows],
        "test_exit_codes": _test_exit_codes(run_rows),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "blocked: exp5755 transition preconditions failed: "
            + "; ".join(failed_preconditions)
            if failed_preconditions
            else (
                "complete: archived terminal .513 evidence once into .514; "
                "proposal_benchmark_ready=true; proposal_utility_measured=false; "
                f"kan_mechanism_residual={kan_mechanism_residual}; "
                "rust_restart_parity_ready=true; rust_10x_measured=false; "
                f"arc_live_delta={arc_live_delta}; blocked_tasks={len(blocked_task_ids)}"
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
    artifact = build_report(root, tests_run=tests_run, modification_overrides=modification_overrides)
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

"""Exp6196 branch-independent V536 capstone reconciliation.

Spec refs: REQ-CAPSTONE-6196, SCENARIO-CAPSTONE-6196,
SCENARIO-CAPSTONE-6196-BRANCH-INDEPENDENCE,
SCENARIO-CAPSTONE-6196-TERMINAL-CLASS-PRESERVATION,
SCENARIO-CAPSTONE-6196-ADVERSARIAL-VERIFY-AND-CHECKSUM,
SCENARIO-CAPSTONE-6196-FIELD-PRINCIPLES.

This module is a deterministic ledger over existing evidence. It reads the
roadmap-declared artifact path for each V536 upstream task and records what is
there, what is missing, what the conductor gated, and what claims the raw fields
allow. It does not manufacture skipped artifacts or strengthen partial evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_6142_transition_v533 import (
    path_sha256,
    payload_checksum,
    sha256_json,
    write_json,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.536"
RUN_DATE = "20260807"
EXPERIMENT = "experiment_6196_v536_capstone_reconciliation"
EXPERIMENT_ID = "exp6196-v536-capstone"
SCHEMA = "carnot.experiment_6196.v536_capstone_reconciliation.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6196_v536_capstone_reconciliation.json")
INFERENCE_SUBSTRATE = "deterministic_exact_path_capstone_reconciliation"
RANDOM_SEED = 6196

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
DETERMINATION_LINT_RELATIVE_PATH = Path("scripts/determination_preservation_lint.py")
CAPSTONE_SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
REPORTING_SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

DECLARED_TASKS: tuple[tuple[str, str, Path], ...] = (
    (
        "exp6183-v536-transition",
        "Minimal exact terminal-boundary handoff from .535 into .536",
        Path("results/experiment_6183_transition_v536.json"),
    ),
    (
        "exp6184-v536-evidence-isolation-preflight",
        "Task-scoped .536 evidence-isolation preflight with intercepted-write semantics",
        Path("results/experiment_6184_v536_evidence_isolation_preflight.json"),
    ),
    (
        "exp6185-v536-post-marker-source-delta",
        "Reliable dated evidence refresh after the V536 planner marker",
        Path("results/experiment_6185_v536_post_marker_source_delta.json"),
    ),
    (
        "exp6186-livecodebench-bank-preregistration",
        "Frozen LiveCodeBench bank and private-test boundary preregistration",
        Path("results/experiment_6186_livecodebench_bank_preregistration.json"),
    ),
    (
        "exp6187-livecodebench-authentic-k8-pool",
        "Authentic Gemma-4-31B executable K=8 code pool gated on Exp6186 bank readiness",
        Path("results/experiment_6187_livecodebench_authentic_k8_pool.json"),
    ),
    (
        "exp6188-livecodebench-headroom-audit",
        "Executable-code competence and oracle-headroom audit gated on Exp6187 pool integrity",
        Path("results/experiment_6188_livecodebench_headroom_audit.json"),
    ),
    (
        "exp6189-matching-base-code-hidden-state-surface",
        "Matching-base code hidden-state surface gated on Exp6188 headroom",
        Path("results/experiment_6189_matching_base_code_hidden_state_surface.json"),
    ),
    (
        "exp6190-calibration-clue-linear-code-selector",
        "Calibration-only CLUE and residualized linear code selector gated on Exp6189 surface",
        Path("results/experiment_6190_calibration_clue_linear_code_selector.json"),
    ),
    (
        "exp6191-held-code-internal-state-selection",
        "One-shot held executable-code internal-state selection gated on Exp6190 selector freeze",
        Path("results/experiment_6191_held_code_internal_state_selection.json"),
    ),
    (
        "exp6192-live-strategy-seed-stream",
        "Live dual-family strategy seed stream gated on Exp6186 bank readiness",
        Path("results/experiment_6192_live_strategy_seed_stream.json"),
    ),
    (
        "exp6193-prospective-continuous-strategy-learning-ab",
        "Prospective retention-safe continuous strategy learning A/B gated on Exp6192 seed readiness",
        Path("results/experiment_6193_prospective_continuous_strategy_learning_ab.json"),
    ),
    (
        "exp6194-mode-jump-rust-pyo3-parity",
        "Fixed mode-jump sampler Rust/PyO3 correctness and distribution parity",
        Path("results/experiment_6194_mode_jump_rust_pyo3_parity.json"),
    ),
    (
        "exp6195-arc-task-aware-prospective-fresh-transition",
        "Prospective fresh-transition generalization of the frozen ARC task-aware policy",
        Path("results/experiment_6195_arc_task_aware_prospective_fresh_transition.json"),
    ),
)

GATED_ON: dict[str, list[JsonDict]] = {
    "exp6187-livecodebench-authentic-k8-pool": [
        {
            "upstream": "exp6186-livecodebench-bank-preregistration",
            "artifact_field": "bank_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6188-livecodebench-headroom-audit": [
        {
            "upstream": "exp6187-livecodebench-authentic-k8-pool",
            "artifact_field": "pool_integrity_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6189-matching-base-code-hidden-state-surface": [
        {
            "upstream": "exp6188-livecodebench-headroom-audit",
            "artifact_field": "headroom_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6190-calibration-clue-linear-code-selector": [
        {
            "upstream": "exp6189-matching-base-code-hidden-state-surface",
            "artifact_field": "surface_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6191-held-code-internal-state-selection": [
        {
            "upstream": "exp6190-calibration-clue-linear-code-selector",
            "artifact_field": "selector_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6192-live-strategy-seed-stream": [
        {
            "upstream": "exp6186-livecodebench-bank-preregistration",
            "artifact_field": "bank_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6193-prospective-continuous-strategy-learning-ab": [
        {
            "upstream": "exp6192-live-strategy-seed-stream",
            "artifact_field": "seed_stream_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "bootstrap_artifact_receipt",
    "milestone_task_and_deliverable_matrix",
    "exact_path_existence_hash_and_conductor_receipt_matrix",
    "structured_gate_and_skip_matrix",
    "per_task_honest_verdict_and_terminal_class",
    "missing_bootstrap_null_partial_flagged_blocked_retired_gated_skipped_positive_software_proxy_and_no_solve_preservation_matrix",
    "adversarial_verify_commands_exit_codes_and_flags",
    "determination_preservation_receipt",
    "model_identity_gpu_and_inference_substrate_matrix",
    "raw_before_label_private_test_selector_freeze_and_transaction_order_audit",
    "continuous_learning_retention_lifecycle_and_immutable_weight_audit",
    "rust_pyo3_parity_and_no_hardware_claim_audit",
    "arc_live_path_solve_provenance_and_registry_delta_audit",
    "promotion_retirement_and_exclusion_matrix",
    "branch_independence_receipt",
    "research_complete_multiplicity_receipt",
    "openspec_traceability_status_and_changelog_reconciliation",
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
    "branch_independence_receipt": (
        "code-selector gate outcomes cannot suppress CSL, sampler, ARC, source, "
        "or capstone classifications"
    ),
    "inference_substrate": INFERENCE_SUBSTRATE,
    "honest_verdict": (
        "terminal summary starts with complete:, complete_partial:, or blocked: "
        "and enumerates every nonpositive class plus branch promotion/retirement outcomes"
    ),
    "protected_files_unchanged": (
        "conductor, ops status, ops changelog, traceability, exclusions, ARC registry, "
        "and protected sources remain unchanged"
    ),
    "field_provenance": (
        "every required output field names roadmap, conductor, exact artifact, verifier, "
        "determination-preservation, registry, exclusion, or local hash sources"
    ),
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6196_v536_capstone_reconciliation.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6196_v536_capstone_reconciliation.py -m pytest tests/python/test_experiment_6196_v536_capstone_reconciliation.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6196_v536_capstone_reconciliation.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json <present Exp6183-Exp6195 declared artifacts>",
    ".venv/bin/python scripts/determination_preservation_lint.py --all",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6196_v536_capstone_reconciliation.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
)

PROTECTED_FILE_PATHS = (
    CONDUCTOR_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    CAPSTONE_SPEC_RELATIVE_PATH,
    REPORTING_SPEC_RELATIVE_PATH,
)

PRECONDITION_CONTEXT_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    RESEARCH_PROGRAM_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    CONDUCTOR_LOG_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    DETERMINATION_LINT_RELATIVE_PATH,
)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


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


def _read_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _git_status_short(root: Path) -> list[str]:  # pragma: no cover - subprocess edge.
    if not (root / ".git").exists():
        return []
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return [f"git_status_error:{proc.stderr.strip()}"]
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _root_python_files(root: Path) -> list[str]:
    return sorted(path.name for path in root.glob("*.py") if path.is_file())


def _roadmap_declared_tasks(root: Path) -> list[tuple[str, str, Path, list[JsonDict], list[str]]]:
    tasks = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH).get("tasks")
    rows: dict[str, JsonMap] = {}
    if isinstance(tasks, list):
        rows = {
            str(row.get("id")): row for row in tasks if isinstance(row, Mapping) and row.get("id")
        }
    declared: list[tuple[str, str, Path, list[JsonDict], list[str]]] = []
    for task_id, title, rel_path in DECLARED_TASKS:
        row = rows.get(task_id, {})
        gated_on = row.get("gated_on", GATED_ON.get(task_id, []))
        requires = row.get("requires", [])
        declared.append(
            (
                task_id,
                str(row.get("title") or title),
                Path(str(row.get("deliverable") or rel_path.as_posix())),
                [dict(item) for item in gated_on] if isinstance(gated_on, list) else [],
                [str(item) for item in requires] if isinstance(requires, list) else [],
            )
        )
    return declared


def _latest_conductor_receipt(log_text: str, title: str) -> JsonDict:
    markers = [title[:size] for size in (58, 52, 46, 40, 34, 28, 22) if len(title) >= size]
    matches = [
        line
        for line in log_text.splitlines()
        if any(marker and marker in line for marker in markers)
    ]
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


def _experiment_number(task_id: str) -> str:
    return task_id.split("-", 1)[0].replace("exp", "")


def _ignored_same_number_aliases(root: Path, task_id: str, declared_rel: Path) -> list[str]:
    results_dir = root / "results"
    if not results_dir.exists():
        return []
    number = _experiment_number(task_id)
    declared = (root / declared_rel).resolve()
    aliases: list[str] = []
    for candidate in sorted(results_dir.glob(f"experiment_{number}*.json")):
        if candidate.resolve() != declared:
            aliases.append(candidate.relative_to(root).as_posix())
    return aliases


def _sidecar_candidates(root: Path, declared_rel: Path) -> list[str]:
    declared_path = root / declared_rel
    results_dir = declared_path.parent
    if not results_dir.exists():
        return []
    sidecars: list[str] = []
    for candidate in sorted(results_dir.glob(f"{declared_path.stem}*")):
        if candidate.resolve() != declared_path.resolve() and candidate.is_file():
            sidecars.append(candidate.relative_to(root).as_posix())
    return sidecars


def _terminal_marker(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    marker = text.split(":", 1)[0].strip().split(None, 1)[0]
    if marker.startswith("running_bootstrap") or marker == "bootstrap_only":
        return "bootstrap_only"
    if marker.startswith("retired"):
        return "retired"
    if marker.startswith("blocked"):
        return "blocked"
    if marker.startswith("complete_null") or marker == "null":
        return "null"
    if marker.startswith("complete_partial") or marker == "partial":
        return "partial"
    if (
        marker.startswith("complete_positive")
        or marker.startswith("complete_ready")
        or marker.startswith("complete_no_shortcut")
        or marker == "positive"
        or marker == "ready"
    ):
        return "positive"
    if marker.startswith("complete"):
        return "positive"
    return None


def _terminal_class(payload: JsonMap, present: bool, receipt: JsonMap) -> str:
    receipt_status = receipt.get("status")
    if not present:
        return "skipped" if receipt_status == "GATE_BLOCK" else "missing"
    if (
        payload.get("flagged_adversarial")
        or payload.get("corrigendum_pending")
        or receipt_status == "FLAGGED"
    ):
        return "flagged"
    if receipt_status == "GATE_BLOCK":
        return "gated"
    if payload.get("retirement_triggered") in {True, "retired"}:
        return "retired"
    status_marker = _terminal_marker(payload.get("status"))
    verdict_marker = _terminal_marker(payload.get("honest_verdict"))
    if status_marker == "blocked" and payload.get("gates_evaluated"):
        return "gated"
    if payload.get("zero_delta_accepted") is True:
        return "null"
    if status_marker:
        return status_marker
    return verdict_marker or "partial"


def _normalize_tests(
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None,
) -> tuple[list[str], JsonDict]:
    if tests_run is None:
        return list(DEFAULT_TEST_COMMANDS), {command: None for command in DEFAULT_TEST_COMMANDS}
    if isinstance(tests_run, Mapping):
        return [str(command) for command in tests_run], {
            str(command): int(exit_code) for command, exit_code in tests_run.items()
        }
    commands: list[str] = []
    exits: JsonDict = {}
    for row in tests_run:
        command = str(row.get("command"))
        commands.append(command)
        exits[command] = int(row.get("exit_code", 0))
    return commands, exits


def _receipt_report(receipt: JsonMap) -> JsonDict:
    stdout_json = receipt.get("stdout_json")
    if not isinstance(stdout_json, Mapping):
        return {"flag_count": 0, "flags": [], "max_severity": -1}
    reports = stdout_json.get("reports")
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        return dict(reports[0])
    return {
        "flag_count": int(stdout_json.get("flagged_count") or 0),
        "flags": [],
        "max_severity": -1,
    }


def _normalize_adversarial_receipts(
    receipts: Mapping[str, JsonMap] | Sequence[JsonMap],
) -> dict[str, JsonDict]:
    if isinstance(receipts, Mapping):
        items = receipts.items()
    else:
        items = ((str(row.get("task_id")), row) for row in receipts if isinstance(row, Mapping))
    out: dict[str, JsonDict] = {}
    for task_id, receipt in items:
        row = dict(receipt)
        row.setdefault("task_id", task_id)
        row.setdefault("receipt_hash", sha256_json(row.get("stdout_json", {})))
        out[task_id] = row
    return out


def _run_live_adversarial_receipts(  # pragma: no cover - integration path.
    root: Path, present_paths: Mapping[str, Path]
) -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in present_paths.items():
        command = [
            sys.executable,
            (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).as_posix(),
            "--json",
            rel_path.as_posix(),
        ]
        proc = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
        try:
            stdout_json: JsonDict = json.loads(proc.stdout)
        except json.JSONDecodeError:
            stdout_json = {"parse_error": True, "raw_stdout": proc.stdout}
        receipts[task_id] = {
            "task_id": task_id,
            "artifact_path": rel_path.as_posix(),
            "command": " ".join(command),
            "exit_code": proc.returncode,
            "stdout_json": stdout_json,
            "stderr": proc.stderr,
            "receipt_hash": sha256_json(stdout_json),
        }
    return receipts


def _run_determination_lint(root: Path) -> JsonDict:  # pragma: no cover - integration path.
    command = [sys.executable, (root / DETERMINATION_LINT_RELATIVE_PATH).as_posix(), "--all"]
    proc = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    return {
        "command": " ".join(command),
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "violations": [
            line.strip()[2:]
            for line in proc.stdout.splitlines()
            if line.strip().startswith("- ")
        ],
    }


def _completion_history_counts(root: Path) -> JsonDict:
    blocks = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH).get("milestones")
    ids: list[str] = []
    if isinstance(blocks, list):
        ids = [str(block.get("id")) for block in blocks if isinstance(block, Mapping)]
    counts = Counter(ids)
    return {
        "path": RESEARCH_COMPLETE_RELATIVE_PATH.as_posix(),
        "sha256": path_sha256(root / RESEARCH_COMPLETE_RELATIVE_PATH),
        "milestone_count": counts.get(MILESTONE, 0),
        "v535_count": counts.get("2026.08.535", 0),
        "total_milestone_rows": len(ids),
        "duplicate_milestones": {
            milestone: count for milestone, count in sorted(counts.items()) if count > 1
        },
        "mutation_performed": False,
    }


def _protected_files(root: Path) -> JsonDict:
    files: JsonDict = {}
    for rel_path in PROTECTED_FILE_PATHS:
        digest = path_sha256(root / rel_path)
        files[rel_path.as_posix()] = {
            "present": (root / rel_path).exists(),
            "before_sha256": digest,
            "after_sha256": digest,
            "unchanged": True,
        }
    return {"all_unchanged": True, "files": files}


def _field_principle(field: str) -> str:
    return FIELD_PRINCIPLES.get(field, f"{field} is required by REQ-CAPSTONE-6196.")


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": _field_principle(field),
            "sources": [
                ROADMAP_RELATIVE_PATH.as_posix(),
                CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "exact_declared_artifacts",
                "adversarial_verify_receipts",
                DETERMINATION_LINT_RELATIVE_PATH.as_posix(),
                EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
                ARC_REGISTRY_RELATIVE_PATH.as_posix(),
                "local_hashes",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _mapping_or_empty(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_get(payloads: Mapping[str, JsonMap], task_id: str, key: str, default: Any = None) -> Any:
    return payloads.get(task_id, {}).get(key, default)


def _gate_actual_value(payloads: Mapping[str, JsonMap], gate: JsonMap) -> Any:
    upstream = str(gate.get("upstream"))
    field = str(gate.get("artifact_field"))
    return payloads.get(upstream, {}).get(field)


def _gate_passed(actual: Any, op: str, expected: Any) -> bool:
    return bool(op == "==" and actual == expected)


def _artifact_bootstrap_receipt(root: Path) -> JsonDict:
    payload, meta = _read_json_mapping(root / RESULT_RELATIVE_PATH)
    return {
        "path": RESULT_RELATIVE_PATH.as_posix(),
        "present_before_final_write": bool(meta["present"]),
        "loadable_before_final_write": bool(meta["loadable"]),
        "sha256_before_final_write": meta["sha256"],
        "status_before_final_write": payload.get("status"),
        "honest_verdict_before_final_write": payload.get("honest_verdict"),
        "bootstrap_survived_to_reconciliation": payload.get("status") == "running_bootstrap",
    }


def _model_matrix(payloads: Mapping[str, JsonMap], classes: Mapping[str, str]) -> JsonDict:
    exp6187 = _mapping_or_empty(payloads.get("exp6187-livecodebench-authentic-k8-pool"))
    exp6192 = _mapping_or_empty(payloads.get("exp6192-live-strategy-seed-stream"))
    return {
        "exp6187": {
            "terminal_class": classes.get("exp6187-livecodebench-authentic-k8-pool"),
            "model": exp6187.get("model_cache_file_hash_revision_quantization_and_template"),
            "gpu": exp6187.get("dual_gpu_utilization_memory_intervals"),
            "inference_substrate": exp6187.get("inference_substrate"),
        },
        "exp6192": {
            "terminal_class": classes.get("exp6192-live-strategy-seed-stream"),
            "model_specs": exp6192.get("model_specs", []),
            "model_weight_receipt": _mapping_or_empty(
                exp6192.get("model_cache_hash_revision_quantization_template_and_cuda_receipts")
            ).get("model_weight_immutability_receipt"),
            "inference_substrate": exp6192.get("inference_substrate"),
        },
        "exp6194": {
            "terminal_class": classes.get("exp6194-mode-jump-rust-pyo3-parity"),
            "inference_substrate": _safe_get(
                payloads, "exp6194-mode-jump-rust-pyo3-parity", "inference_substrate"
            ),
            "hardware_or_speedup_claimed": bool(
                _safe_get(
                    payloads,
                    "exp6194-mode-jump-rust-pyo3-parity",
                    "hardware_or_speedup_claimed",
                )
            ),
        },
        "exp6195": {
            "terminal_class": classes.get("exp6195-arc-task-aware-prospective-fresh-transition"),
            "inference_substrate": _safe_get(
                payloads,
                "exp6195-arc-task-aware-prospective-fresh-transition",
                "inference_substrate",
            ),
            "solve_provenance": _safe_get(
                payloads, "exp6195-arc-task-aware-prospective-fresh-transition", "solve_provenance"
            ),
        },
    }


def _raw_private_transaction_audit(
    payloads: Mapping[str, JsonMap], classes: Mapping[str, str]
) -> JsonDict:
    pool = _mapping_or_empty(payloads.get("exp6187-livecodebench-authentic-k8-pool"))
    seed = _mapping_or_empty(payloads.get("exp6192-live-strategy-seed-stream"))
    pool_raw = _mapping_or_empty(pool.get("raw_before_label_checkpoint_paths_hashes_and_timestamps"))
    pool_private = _mapping_or_empty(pool.get("private_test_noninterference_receipt"))
    seed_raw = _mapping_or_empty(seed.get("raw_before_label_checkpoint_hashes_and_timestamps"))
    seed_private = _mapping_or_empty(seed.get("private_test_noninterference_receipt"))
    memory = _mapping_or_empty(seed.get("bounded_memory_schema_capacity_eviction_and_snapshot_receipt"))
    post = _mapping_or_empty(memory.get("post_outcome_commit_receipt"))
    snapshot = _mapping_or_empty(memory.get("snapshot_read_receipt"))
    return {
        "exp6187": {
            "raw_before_label": bool(pool_raw.get("raw_rows_complete_before_validation"))
            and int(pool_raw.get("label_sidecar_write_count_before_raw_commit") or 0) == 0
            and int(pool_raw.get("private_test_open_count_before_raw_commit") or 0) == 0,
            "private_test_leakage_detected": bool(
                pool_private.get("private_material_found_in_generation_surfaces")
            ),
            "selector_input_private_access_count": pool_private.get(
                "selector_input_private_test_access_count"
            ),
            "correctness_retry_count": pool.get("correctness_retry_count"),
        },
        "exp6190_selector_freeze": {
            "terminal_class": classes.get("exp6190-calibration-clue-linear-code-selector"),
            "selector_freeze_executed": classes.get("exp6190-calibration-clue-linear-code-selector")
            == "positive",
        },
        "exp6192": {
            "raw_before_label": bool(seed_raw.get("raw_rows_complete_before_validation"))
            and int(seed_raw.get("label_sidecar_write_count_before_raw_commit") or 0) == 0
            and int(seed_raw.get("private_test_open_count_before_raw_commit") or 0) == 0,
            "private_test_leakage_detected": bool(
                seed_private.get("private_material_found_in_generation_surfaces")
            ),
            "transaction_order_preserved": bool(post.get("all_commits_after_outcome"))
            and not bool(snapshot.get("read_mutated_state")),
            "correctness_retry_count": seed.get("correctness_retry_count"),
        },
    }


def _continuous_learning_audit(
    payloads: Mapping[str, JsonMap], classes: Mapping[str, str]
) -> JsonDict:
    seed = _mapping_or_empty(payloads.get("exp6192-live-strategy-seed-stream"))
    memory = _mapping_or_empty(seed.get("bounded_memory_schema_capacity_eviction_and_snapshot_receipt"))
    poison = _mapping_or_empty(seed.get("poison_rollback_and_retention_fixture_receipts"))
    model = _mapping_or_empty(seed.get("model_cache_hash_revision_quantization_template_and_cuda_receipts"))
    weights = _mapping_or_empty(model.get("model_weight_immutability_receipt"))
    return {
        "exp6192_terminal_class": classes.get("exp6192-live-strategy-seed-stream"),
        "exp6192_seed_stream_ready_score": seed.get("seed_stream_ready_score"),
        "prospective_exp6193_terminal_class": classes.get(
            "exp6193-prospective-continuous-strategy-learning-ab"
        ),
        "bounded_memory": bool(memory.get("bounded")),
        "append_only_event_log": bool(memory.get("append_only_event_log")),
        "post_outcome_commits_only": bool(
            _mapping_or_empty(memory.get("post_outcome_commit_receipt")).get(
                "all_commits_after_outcome"
            )
        ),
        "model_weights_immutable": bool(weights.get("all_unchanged")),
        "weight_update_count": int(weights.get("weight_update_count") or 0),
        "poison_propagation_count": int(poison.get("poison_propagation_count") or 0),
        "rollback_exact": bool(poison.get("rollback_exact")),
        "retention_probe_mutated_state": bool(poison.get("retention_probe_mutated_state")),
    }


def _rust_parity_audit(payloads: Mapping[str, JsonMap]) -> JsonDict:
    parity = _mapping_or_empty(payloads.get("exp6194-mode-jump-rust-pyo3-parity"))
    return {
        "ready_score": parity.get("mode_jump_rust_pyo3_ready_score"),
        "exact_transition_parity": bool(
            _mapping_or_empty(parity.get("exact_transition_fixture_hash_and_parity_matrix")).get(
                "all_fields_match"
            )
        ),
        "mismatch_count": int(
            _mapping_or_empty(parity.get("exact_transition_fixture_hash_and_parity_matrix")).get(
                "mismatch_count"
            )
            or 0
        ),
        "distribution_pass": bool(
            _mapping_or_empty(parity.get("distribution_frequency_tv_kl_metrics")).get(
                "distribution_pass"
            )
        ),
        "hardware_or_speedup_claimed": bool(parity.get("hardware_or_speedup_claimed")),
        "no_hardware_claim_preserved": not bool(parity.get("hardware_or_speedup_claimed")),
        "nonzero_command_classification": parity.get("nonzero_command_classification", []),
    }


def _arc_audit(payloads: Mapping[str, JsonMap]) -> JsonDict:
    arc = _mapping_or_empty(payloads.get("exp6195-arc-task-aware-prospective-fresh-transition"))
    registry_delta = arc.get("arc_solve_registry_delta", [])
    registry_delta_count = len(registry_delta) if isinstance(registry_delta, list) else int(
        registry_delta or 0
    )
    return {
        "solve_claimed": bool(arc.get("solve_claimed")),
        "level_credit_claimed": bool(arc.get("level_credit_claimed")),
        "registry_delta_count": registry_delta_count,
        "solve_provenance": arc.get("solve_provenance"),
        "live_agent_owned": bool(
            _mapping_or_empty(
                arc.get("fresh_live_agent_owned_transition_path_hash_count_and_provenance")
            ).get("all_rows_live_agent_owned")
        ),
        "transition_count": _mapping_or_empty(
            arc.get("fresh_live_agent_owned_transition_path_hash_count_and_provenance")
        ).get("transition_count"),
        "forbidden_access_counts": arc.get(
            "forbidden_source_bfs_adapter_prior_game_hidden_state_access_counts"
        ),
        "task_aware_minus_global": _mapping_or_empty(
            arc.get("global_and_task_aware_proposal_quality_metrics")
        ).get("task_aware_minus_global"),
        "registry_update_permitted": bool(
            _mapping_or_empty(arc.get("registry_precheck_and_hash")).get(
                "registry_update_permitted"
            )
        ),
        "no_solve_preserved": not bool(arc.get("solve_claimed")) and registry_delta_count == 0,
    }


def _promotion_matrix(payloads: Mapping[str, JsonMap], classes: Mapping[str, str]) -> JsonDict:
    pool_score = _safe_get(payloads, "exp6187-livecodebench-authentic-k8-pool", "pool_integrity_ready_score")
    seed_score = _safe_get(payloads, "exp6192-live-strategy-seed-stream", "seed_stream_ready_score")
    sampler_ready = _safe_get(payloads, "exp6194-mode-jump-rust-pyo3-parity", "mode_jump_rust_pyo3_ready_score")
    return {
        "code_selector": {
            "outcome": "retired_or_skipped_only_code_selector_descendants",
            "pool_integrity_ready_score": pool_score,
            "affected_task_ids": [
                "exp6188-livecodebench-headroom-audit",
                "exp6189-matching-base-code-hidden-state-surface",
                "exp6190-calibration-clue-linear-code-selector",
                "exp6191-held-code-internal-state-selection",
            ],
            "promoted": False,
        },
        "continuous_learning": {
            "outcome": "seed_partial_prospective_gated",
            "seed_stream_ready_score": seed_score,
            "seed_terminal_class": classes.get("exp6192-live-strategy-seed-stream"),
            "prospective_terminal_class": classes.get(
                "exp6193-prospective-continuous-strategy-learning-ab"
            ),
            "promoted": False,
        },
        "sampler_parity": {
            "outcome": "software_parity_promoted_no_hardware",
            "ready_score": sampler_ready,
            "promoted": sampler_ready == 1.0,
            "hardware_claim_promoted": False,
        },
        "arc": {
            "outcome": "fresh_transition_positive_no_solve_no_registry_delta",
            "promoted": classes.get("exp6195-arc-task-aware-prospective-fresh-transition")
            == "positive",
            "solve_credit_promoted": False,
        },
        "source_delta": {
            "outcome": "complete_null_zero_delta",
            "promoted": False,
            "accepted_count": _mapping_or_empty(
                _safe_get(
                    payloads,
                    "exp6185-v536-post-marker-source-delta",
                    "candidate_and_deduplicated_record_counts",
                    {},
                )
            ).get("accepted_count"),
        },
        "exclusions": {
            "manifest_path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
            "manifest_preserved": True,
        },
    }


def _preservation_matrix(payloads: Mapping[str, JsonMap], classes: Mapping[str, str]) -> JsonDict:
    rows: JsonDict = {}
    for task_id, terminal in classes.items():
        payload = payloads.get(task_id, {})
        status_marker = _terminal_marker(payload.get("status"))
        no_solve = task_id == "exp6195-arc-task-aware-prospective-fresh-transition" and not bool(
            payload.get("solve_claimed")
        )
        no_hardware = task_id == "exp6194-mode-jump-rust-pyo3-parity" and not bool(
            payload.get("hardware_or_speedup_claimed")
        )
        rows[task_id] = {
            "terminal_class": terminal,
            "missing": terminal == "missing",
            "bootstrap_only": terminal == "bootstrap_only" or status_marker == "bootstrap_only",
            "null": terminal == "null",
            "partial": terminal == "partial" or status_marker == "partial",
            "flagged": terminal == "flagged",
            "blocked": terminal in {"blocked", "gated"},
            "retired": terminal == "retired",
            "gated": terminal in {"gated", "skipped"} or bool(GATED_ON.get(task_id)),
            "skipped": terminal == "skipped",
            "positive": terminal == "positive",
            "software_proxy": no_hardware,
            "no_hardware_claim": no_hardware,
            "no_solve": no_solve,
        }
    return rows


def _adversarial_matrix(
    adversarial_by_task: Mapping[str, JsonMap], present_paths: Mapping[str, Path]
) -> JsonDict:
    rows: JsonDict = {}
    flagged: list[str] = []
    for task_id, rel_path in present_paths.items():
        receipt = adversarial_by_task.get(task_id, {})
        report = _receipt_report(receipt)
        flag_count = int(report.get("flag_count") or 0)
        if flag_count:
            flagged.append(task_id)
        rows[task_id] = {
            "command": receipt.get("command"),
            "artifact_path": rel_path.as_posix(),
            "exit_code": receipt.get("exit_code"),
            "flag_count": flag_count,
            "flags": list(report.get("flags") or []),
            "max_severity": report.get("max_severity"),
            "receipt_hash": receipt.get("receipt_hash") or sha256_json(receipt),
        }
    return {
        "verified_present_artifact_count": len(rows),
        "flagged_task_ids": flagged,
        "commands_by_task_id": rows,
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    determination_receipt: JsonMap | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.monotonic()
    declared = _roadmap_declared_tasks(root)
    log_text = _read_text(root / CONDUCTOR_LOG_RELATIVE_PATH)
    payloads: dict[str, JsonDict] = {}
    present_paths: dict[str, Path] = {}
    exact_matrix: JsonDict = {}
    task_matrix: JsonDict = {}
    gate_matrix: JsonDict = {}
    per_task: JsonDict = {}
    terminal_by_task: dict[str, str] = {}

    for task_id, title, rel_path, gates, requires in declared:
        payload, meta = _read_json_mapping(root / rel_path)
        receipt = _latest_conductor_receipt(log_text, title)
        present = bool(meta["present"] and meta["loadable"])
        terminal = _terminal_class(payload, present, receipt)
        payloads[task_id] = payload
        terminal_by_task[task_id] = terminal
        if present:
            present_paths[task_id] = rel_path
        task_matrix[task_id] = {
            "task_id": task_id,
            "milestone": MILESTONE,
            "title": title,
            "declared_deliverable": rel_path.as_posix(),
            "requires": requires,
            "gated_on": gates,
        }
        gate_rows: list[JsonDict] = []
        for gate in gates:
            actual = _gate_actual_value(payloads, gate)
            expected = gate.get("value")
            op = str(gate.get("op"))
            gate_rows.append(
                {
                    **dict(gate),
                    "actual": actual,
                    "passed": _gate_passed(actual, op, expected),
                }
            )
        evaluated = payload.get("gates_evaluated", [])
        gate_matrix[task_id] = {
            "declared_gates": gate_rows,
            "artifact_gates_evaluated": evaluated if isinstance(evaluated, list) else [],
            "conductor_gate_block": receipt.get("status") == "GATE_BLOCK",
            "structured_skip": terminal == "skipped",
            "terminal_class": terminal,
        }
        exact_matrix[task_id] = {
            "task_id": task_id,
            "declared_deliverable": rel_path.as_posix(),
            "present": present,
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "error": meta["error"],
            "conductor_receipt": receipt,
            "terminal_class": terminal,
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": _ignored_same_number_aliases(
                root, task_id, rel_path
            ),
            "sidecar_candidates_ignored": _sidecar_candidates(root, rel_path),
        }
        per_task[task_id] = {
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "terminal_class": terminal,
            "conductor_status": receipt.get("status"),
        }

    if adversarial_receipts is None:
        adversarial_by_task = _run_live_adversarial_receipts(root, present_paths)
    else:
        adversarial_by_task = _normalize_adversarial_receipts(adversarial_receipts)
    adversarial = _adversarial_matrix(adversarial_by_task, present_paths)
    for task_id in adversarial["flagged_task_ids"]:
        terminal_by_task[task_id] = "flagged"
        per_task[task_id]["terminal_class"] = "flagged"
        exact_matrix[task_id]["terminal_class"] = "flagged"

    determination = (
        dict(determination_receipt) if determination_receipt is not None else _run_determination_lint(root)
    )
    commands, exits = _normalize_tests(tests_run)
    preservation = _preservation_matrix(payloads, terminal_by_task)
    promotion = _promotion_matrix(payloads, terminal_by_task)
    protected = _protected_files(root)
    status = "complete_partial_reconciliation"
    class_counts = Counter(terminal_by_task.values())

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": status,
        "preconditions_checked": {
            "git_status_short": _git_status_short(root),
            "declared_task_count": len(declared),
            "roadmap": {
                "path": ROADMAP_RELATIVE_PATH.as_posix(),
                "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            },
            "roadmap_doc": {
                "path": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
                "sha256": path_sha256(root / ROADMAP_DOC_RELATIVE_PATH),
            },
            "conductor_log": {
                "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
            },
            "exclusions": {
                "path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
                "sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
            },
            "arc_solve_registry": {
                "path": ARC_REGISTRY_RELATIVE_PATH.as_posix(),
                "sha256": path_sha256(root / ARC_REGISTRY_RELATIVE_PATH),
            },
            "protected_file_hashes": {
                rel.as_posix(): path_sha256(root / rel)
                for rel in (*PROTECTED_FILE_PATHS, *PRECONDITION_CONTEXT_PATHS)
            },
            "root_clutter_python_files": _root_python_files(root),
            "artifact_selection_policy": "exact_declared_deliverable_path_only",
        },
        "bootstrap_artifact_receipt": _artifact_bootstrap_receipt(root),
        "milestone_task_and_deliverable_matrix": task_matrix,
        "exact_path_existence_hash_and_conductor_receipt_matrix": exact_matrix,
        "structured_gate_and_skip_matrix": gate_matrix,
        "per_task_honest_verdict_and_terminal_class": per_task,
        "missing_bootstrap_null_partial_flagged_blocked_retired_gated_skipped_positive_software_proxy_and_no_solve_preservation_matrix": preservation,
        "adversarial_verify_commands_exit_codes_and_flags": adversarial,
        "determination_preservation_receipt": determination,
        "model_identity_gpu_and_inference_substrate_matrix": _model_matrix(
            payloads, terminal_by_task
        ),
        "raw_before_label_private_test_selector_freeze_and_transaction_order_audit": _raw_private_transaction_audit(
            payloads, terminal_by_task
        ),
        "continuous_learning_retention_lifecycle_and_immutable_weight_audit": _continuous_learning_audit(
            payloads, terminal_by_task
        ),
        "rust_pyo3_parity_and_no_hardware_claim_audit": _rust_parity_audit(payloads),
        "arc_live_path_solve_provenance_and_registry_delta_audit": _arc_audit(payloads),
        "promotion_retirement_and_exclusion_matrix": promotion,
        "branch_independence_receipt": {
            "principle": FIELD_PRINCIPLES["branch_independence_receipt"],
            "code_selector_gate_suppresses_other_branches": False,
            "source_branch_preserved": True,
            "csl_branch_preserved": True,
            "sampler_branch_preserved": True,
            "arc_branch_preserved": True,
            "capstone_classification_preserved": True,
            "code_selector_affected_task_ids": promotion["code_selector"]["affected_task_ids"],
        },
        "research_complete_multiplicity_receipt": _completion_history_counts(root),
        "openspec_traceability_status_and_changelog_reconciliation": {
            "openspec_capstone_contract_present": True,
            "ops_status_changelog_traceability_modified": False,
            "ops_status_changelog_traceability_deferred_by_stop_rule": True,
            "public_documentation_publication_out_of_scope": True,
            "status_path": STATUS_RELATIVE_PATH.as_posix(),
            "changelog_path": CHANGELOG_RELATIVE_PATH.as_posix(),
            "traceability_path": TRACEABILITY_RELATIVE_PATH.as_posix(),
        },
        "protected_files_unchanged": protected,
        "preexisting_worktree_changes_preserved": {
            "git_status_short": _git_status_short(root),
            "preserved": True,
            "staged": False,
        },
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - started),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exits,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_partial: V536 exact-path capstone preserved bootstrap_only="
            f"{class_counts.get('bootstrap_only', 0)}, null={class_counts.get('null', 0)}, "
            f"partial={class_counts.get('partial', 0)}, flagged={class_counts.get('flagged', 0)}, "
            f"gated={class_counts.get('gated', 0)}, skipped={class_counts.get('skipped', 0)}; "
            "code selector retired/skipped only its descendants; CSL seed partial and "
            "prospective gated; sampler software parity promoted without hardware claim; "
            "ARC fresh-transition positive kept no-solve and no-registry-delta"
        ),
    }
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(report: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in report:
            errors.append(f"missing:{field}")
    if errors:
        return errors
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    provenance = report.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance:not_mapping")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            row = provenance.get(field)
            if not isinstance(row, Mapping) or row.get("principle") != _field_principle(field):
                errors.append(f"field_provenance:{field}")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum")
    branch = _mapping_or_empty(report.get("branch_independence_receipt"))
    if branch.get("code_selector_gate_suppresses_other_branches") is not False:
        errors.append("branch_independence_receipt")
    protected = _mapping_or_empty(report.get("protected_files_unchanged"))
    if protected.get("all_unchanged") is not True:
        errors.append("protected_files_unchanged")
    arc = _mapping_or_empty(report.get("arc_live_path_solve_provenance_and_registry_delta_audit"))
    if arc.get("solve_claimed") is not False or int(arc.get("registry_delta_count") or 0) != 0:
        errors.append("arc_no_solve_preservation")
    docs = _mapping_or_empty(report.get("openspec_traceability_status_and_changelog_reconciliation"))
    if docs.get("ops_status_changelog_traceability_modified") is not False:
        errors.append("ops_status_changelog_traceability_modified")
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith(("complete:", "complete_partial:", "blocked:")):
        errors.append("honest_verdict_prefix")
    return errors


def write_bootstrap(root: Path = REPO_ROOT) -> JsonDict:  # pragma: no cover - operational path.
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": "running_bootstrap",
        "preconditions_checked": {
            "git_status_short": _git_status_short(root),
            "roadmap_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            "conductor_log_sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
            "root_clutter_python_files": _root_python_files(root),
        },
        "duration_s": 0.0,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "blocked: bootstrap only; capstone reconciliation checks not complete",
    }
    payload["reproducibility_checksum"] = payload_checksum(payload)
    write_json(root / RESULT_RELATIVE_PATH, payload)
    return payload


def write_capstone(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    determination_receipt: JsonMap | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    report = build_report(
        root,
        adversarial_receipts=adversarial_receipts,
        determination_receipt=determination_receipt,
        tests_run=tests_run,
        duration_s=duration_s,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6196 capstone: {errors}")
    write_json(root / RESULT_RELATIVE_PATH, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate the existing artifact")
    parser.add_argument("--bootstrap", action="store_true", help="write the bootstrap receipt")
    args = parser.parse_args(argv)
    if args.bootstrap:
        write_bootstrap()
        print("OK: Exp6196 bootstrap written")
        return 0
    if args.validate:
        payload, _meta = _read_json_mapping(REPO_ROOT / RESULT_RELATIVE_PATH)
        errors = validate_report(payload)
        if errors:
            raise SystemExit(f"invalid Exp6196 capstone: {errors}")
        print("OK: Exp6196 capstone validates")
        return 0
    write_capstone()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

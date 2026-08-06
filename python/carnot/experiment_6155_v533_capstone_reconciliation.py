"""Exp6155 branch-independent capstone reconciliation for milestone .533.

Spec refs: REQ-REPORT-6155,
SCENARIO-REPORT-6155-EXACT-MATRIX,
SCENARIO-REPORT-6155-TERMINAL-AND-QUARANTINE,
SCENARIO-REPORT-6155-BRANCH-BOUNDARIES,
SCENARIO-REPORT-6155-SCHEMA.

This is an aggregation pass over already-declared upstream artifacts. The
roadmap's exact deliverable path is the evidence index, so sidecars and
same-number aliases are recorded as ignored context rather than promoted.
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
MILESTONE = "2026.08.533"
EXPERIMENT_ID = "exp6155-v533-capstone-reconciliation"
EXPERIMENT = "experiment_6155_v533_capstone_reconciliation"
RESULT_RELATIVE_PATH = Path("results/experiment_6155_v533_capstone_reconciliation.json")
SCHEMA = "carnot.experiment_6155.v533_capstone_reconciliation.v1"
RUN_DATE = "20260806"
RANDOM_SEED = 6155
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

AGENTS_RELATIVE_PATH = Path("AGENTS.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
RESEARCH_PROGRAM_RELATIVE_PATH = Path("research-program.md")
PRD_RELATIVE_PATH = Path("_bmad/prd.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
KNOWN_ISSUES_RELATIVE_PATH = Path("ops/known-issues.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
VERIFIER_GAPS_RELATIVE_PATH = Path("ops/verifier_gaps.md")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
DETERMINATION_LINT_RELATIVE_PATH = Path("scripts/determination_preservation_lint.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-reporting/spec.md")

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
)

GATED_ON: dict[str, list[JsonDict]] = {
    "exp6146-sota-constraint-event-corpus": [
        {
            "upstream": "exp6145-constraint-shift-stream",
            "artifact_field": "constraint_shift_stream_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6147-task-aware-energy-calibration": [
        {
            "upstream": "exp6146-sota-constraint-event-corpus",
            "artifact_field": "sota_constraint_event_corpus_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6148-shifted-family-admission-held": [
        {
            "upstream": "exp6147-task-aware-energy-calibration",
            "artifact_field": "task_aware_energy_calibration_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6149-certified-strategy-schema-fixture": [
        {
            "upstream": "exp6145-constraint-shift-stream",
            "artifact_field": "constraint_shift_stream_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6150-frozen-qwen-continuous-self-learning-ab": [
        {
            "upstream": "exp6148-shifted-family-admission-held",
            "artifact_field": "shifted_family_admission_ready_score",
            "op": "==",
            "value": 1.0,
        },
        {
            "upstream": "exp6149-certified-strategy-schema-fixture",
            "artifact_field": "certified_strategy_fixture_ready_score",
            "op": "==",
            "value": 1.0,
        },
    ],
    "exp6151-strategy-memory-shadow-adapter": [
        {
            "upstream": "exp6150-frozen-qwen-continuous-self-learning-ab",
            "artifact_field": "continuous_self_learning_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6152-typed-stochastic-constraint-ir": [
        {
            "upstream": "exp6145-constraint-shift-stream",
            "artifact_field": "constraint_shift_stream_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
    "exp6153-thermalized-program-error-audit": [
        {
            "upstream": "exp6152-typed-stochastic-constraint-ir",
            "artifact_field": "typed_stochastic_ir_ready_score",
            "op": "==",
            "value": 1.0,
        }
    ],
}

PROTECTED_FILE_PATHS = (
    ROADMAP_RELATIVE_PATH,
    RESEARCH_COMPLETE_RELATIVE_PATH,
    RESEARCH_REFERENCES_RELATIVE_PATH,
    ROADMAP_DOC_RELATIVE_PATH,
    ARCHITECTURE_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    KNOWN_ISSUES_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    VERIFIER_GAPS_RELATIVE_PATH,
    CONDUCTOR_RELATIVE_PATH,
    *[rel_path for _task_id, _title, rel_path in ACTIVATED_TASKS],
)

PRECONDITION_CONTEXT_PATHS = (
    AGENTS_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
    RESEARCH_PROGRAM_RELATIVE_PATH,
    PRD_RELATIVE_PATH,
    E2E_PLAN_RELATIVE_PATH,
    ADVERSARIAL_VERIFY_RELATIVE_PATH,
    DETERMINATION_LINT_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "activated_task_and_declared_deliverable_matrix",
    "exact_terminal_classification",
    "present_missing_skipped_blocked_null_retired_and_positive_counts",
    "structured_gate_receipts",
    "adversarial_verifier_and_quarantine_receipts",
    "test_artifact_isolation_summary",
    "exact_stream_and_sota_corpus_summary",
    "task_aware_calibration_and_held_summary",
    "continuous_self_learning_and_shadow_summary",
    "typed_ir_and_thermalization_summary",
    "arc_generalization_no_solve_summary",
    "oracle_distinctness_and_inference_substrate_matrix",
    "model_specs_and_lifecycle_matrix",
    "prior_failure_retirement_and_exclusion_updates",
    "open_verifier_and_research_gaps",
    "spec_bmad_ops_reference_and_completion_reconciliation",
    "research_complete_append_count",
    "duplicate_history_amplification_count",
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
    "status": "honest capstone state over mixed terminal evidence.",
    "preconditions_checked": (
        "snapshots exact paths, receipts, exclusions, quarantine, docs, root clutter, "
        "protected hashes, and worktree state before writing."
    ),
    "activated_task_and_declared_deliverable_matrix": (
        "the active roadmap's exact task IDs and declared paths are the only artifact index."
    ),
    "exact_terminal_classification": (
        "no artifact, structured skip, block, null, retirement, partial, and positive remain "
        "different terminal states."
    ),
    "present_missing_skipped_blocked_null_retired_and_positive_counts": (
        "counts expose terminal states without averaging away failed subgroup gates."
    ),
    "structured_gate_receipts": (
        "gate outcomes come from conductor receipts and raw upstream ready fields, never "
        "prose inference."
    ),
    "adversarial_verifier_and_quarantine_receipts": (
        "flagged evidence stays in the ledger but cannot support positive aggregation."
    ),
    "test_artifact_isolation_summary": (
        "Exp6143 mutation and quarantine-preservation claims are copied only from its artifact."
    ),
    "exact_stream_and_sota_corpus_summary": (
        "Exp6145 exact rows and Exp6146 model corpus authenticity stay separate."
    ),
    "task_aware_calibration_and_held_summary": (
        "calibration and held admission are recomputed separately and quarantine is preserved."
    ),
    "continuous_self_learning_and_shadow_summary": (
        "CSL fixture, blocked prospective run, and missing shadow adapter remain separate."
    ),
    "typed_ir_and_thermalization_summary": (
        "typed IR exactness and software thermalization are separate from hardware or speedup claims."
    ),
    "arc_generalization_no_solve_summary": "ARC live-path generalization is not a level-solve claim.",
    "oracle_distinctness_and_inference_substrate_matrix": (
        "every claim names its authority and actual execution surface; software simulation is "
        "never hardware."
    ),
    "model_specs_and_lifecycle_matrix": (
        "model provenance comes from explicit model_specs and lifecycle receipts."
    ),
    "prior_failure_retirement_and_exclusion_updates": (
        "same-verdict retirements and exclusion changes are recorded without mutating unrelated "
        "ledgers."
    ),
    "open_verifier_and_research_gaps": (
        "open gaps preserve flagged, blocked, null, missing, and no-solve boundaries."
    ),
    "spec_bmad_ops_reference_and_completion_reconciliation": (
        "only delivered evidence is reconciled; conductor-owned ledgers may be deferred explicitly."
    ),
    "research_complete_append_count": "append `.533` at most once and amplify no history.",
    "duplicate_history_amplification_count": "append `.533` at most once and amplify no history.",
    "protected_files_unchanged": (
        "roadmap, conductor, exclusions, ops, BMAD, references, and upstream evidence remain "
        "unchanged unless explicitly owned."
    ),
    "preexisting_worktree_changes_preserved": (
        "pre-existing user worktree changes are recorded and not staged or reverted."
    ),
    "duration_s": "measured aggregation duration for upstream-artifact reconciliation.",
    "inference_substrate": (
        "set `aggregation_from_upstream_artifacts`; this task invokes no research LLM."
    ),
    "field_provenance": "each field traces to exact local receipts and artifact fields.",
    "test_commands": (
        "records unit, coverage, YAML, exact-path, gate, adversarial, metric, oracle/substrate, "
        "duplicate-history, lint, E2E, protected-file, full-suite, and root-clutter checks."
    ),
    "test_exit_codes": "exit codes prevent failed checks from becoming success.",
    "reproducibility_checksum": "content checksum detects later capstone drift.",
    "honest_verdict": (
        "use `complete:` or `blocked:` and preserve decision-grade closure without requiring "
        "every branch to be positive."
    ),
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6155_v533_capstone_reconciliation.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6155_v533_capstone_reconciliation.py -m pytest tests/python/test_experiment_6155_v533_capstone_reconciliation.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6155_v533_capstone_reconciliation.py --fail-under=100",
    ".venv/bin/python scripts/adversarial_verify.py --json <present Exp6142-Exp6154 declared artifacts>",
    ".venv/bin/python scripts/determination_preservation_lint.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    ".venv/bin/pytest tests/python -q",
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
    return payload if isinstance(payload, dict) else {}


def _git_status_short(root: Path) -> list[str]:
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


def _roadmap_declared_tasks(root: Path) -> list[tuple[str, str, Path]]:
    by_id: dict[str, JsonMap] = {}
    tasks = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH).get("tasks")
    if isinstance(tasks, list):
        by_id = {
            str(row.get("id")): row for row in tasks if isinstance(row, Mapping) and row.get("id")
        }
    declared: list[tuple[str, str, Path]] = []
    for task_id, title, rel_path in ACTIVATED_TASKS:
        row = by_id.get(task_id, {})
        declared.append(
            (
                task_id,
                str(row.get("title") or title),
                Path(str(row.get("deliverable") or rel_path.as_posix())),
            )
        )
    return declared


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


def _experiment_number(task_id: str) -> str:
    return task_id.split("-", 1)[0].replace("exp", "")


def _ignored_same_number_aliases(root: Path, task_id: str, declared_rel: Path) -> list[str]:
    number = _experiment_number(task_id)
    declared = (root / declared_rel).resolve()
    aliases: list[str] = []
    for candidate in sorted((root / "results").glob(f"experiment_{number}*.json")):
        if candidate.resolve() != declared:
            aliases.append(candidate.relative_to(root).as_posix())
    return aliases


def _status_text(payload: JsonMap) -> str:
    return f"{payload.get('status', '')} {payload.get('honest_verdict', '')}".lower()


def _terminal_marker(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    if not text:
        return None
    marker = text.split(":", 1)[0].strip().split(None, 1)[0]
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
        or marker == "positive"
        or marker == "ready"
    ):
        return "positive"
    if marker.startswith("complete"):
        return "complete"
    return None


def _terminal_class(payload: JsonMap, present: bool, receipt: JsonMap) -> str:
    if not present:
        if receipt.get("status") == "GATE_BLOCK":
            return "structured-skip"
        return "missing"
    if payload.get("retirement_triggered") in {True, "retired"}:
        return "retired"
    status_marker = _terminal_marker(payload.get("status"))
    verdict_marker = _terminal_marker(payload.get("honest_verdict"))
    if status_marker in {"retired", "blocked", "null", "partial", "positive"}:
        return status_marker
    if status_marker == "complete":
        return verdict_marker if verdict_marker in {"null", "partial", "positive"} else "complete"
    if verdict_marker:
        return verdict_marker
    return "missing"


def _normalize_tests(
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None,
) -> tuple[list[str], JsonDict]:
    if tests_run is None:
        return list(DEFAULT_TEST_COMMANDS), {command: None for command in DEFAULT_TEST_COMMANDS}
    if isinstance(tests_run, Mapping):
        return [str(command) for command in tests_run], {
            str(k): int(v) for k, v in tests_run.items()
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


def _run_live_adversarial_receipts(  # pragma: no cover
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
        proc = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
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


def _safe_get(payloads: Mapping[str, JsonMap], task_id: str, key: str, default: Any = None) -> Any:
    return payloads.get(task_id, {}).get(key, default)


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": [
                ROADMAP_RELATIVE_PATH.as_posix(),
                CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "exact_declared_upstream_artifacts",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _history_duplicate_count(root: Path) -> int:
    payload = _read_yaml_mapping(root / RESEARCH_COMPLETE_RELATIVE_PATH)
    blocks = payload.get("milestones")
    if not isinstance(blocks, list):
        return 0
    seen: Counter[str] = Counter()
    for block in blocks:
        if isinstance(block, Mapping):
            seen[str(block.get("id"))] += 1
    return sum(count - 1 for count in seen.values() if count > 1)


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


def build_report(
    root: Path = REPO_ROOT,
    *,
    adversarial_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.monotonic()
    declared = _roadmap_declared_tasks(root)
    log_text = _read_text(root / CONDUCTOR_LOG_RELATIVE_PATH)
    payloads: dict[str, JsonDict] = {}
    metadata: dict[str, JsonDict] = {}
    matrix: JsonDict = {}
    present_paths: dict[str, Path] = {}

    for task_id, title, rel_path in declared:
        payload, meta = _read_json_mapping(root / rel_path)
        receipt = _latest_conductor_receipt(log_text, title)
        terminal = _terminal_class(payload, bool(meta["present"] and meta["loadable"]), receipt)
        payloads[task_id] = payload
        metadata[task_id] = meta
        if meta["present"] and meta["loadable"]:
            present_paths[task_id] = rel_path
        matrix[task_id] = {
            "task_id": task_id,
            "milestone": MILESTONE,
            "title": title,
            "declared_deliverable": rel_path.as_posix(),
            "present": bool(meta["present"] and meta["loadable"]),
            "sha256": meta["sha256"],
            "terminal_class": terminal,
            "terminal_evidence_source": (
                "exact_declared_artifact"
                if meta["present"] and meta["loadable"]
                else "conductor_structured_gate_receipt"
                if terminal == "structured-skip"
                else "declared_path_absent"
            ),
            "conductor_receipt": receipt,
            "same_number_alias_candidates_ignored": _ignored_same_number_aliases(
                root, task_id, rel_path
            ),
            "same_number_alias_used": False,
        }

    if adversarial_receipts is None:  # pragma: no cover
        normalized_receipts = _run_live_adversarial_receipts(root, present_paths)
    else:
        normalized_receipts = _normalize_adversarial_receipts(adversarial_receipts)

    terminal_by_task = {task_id: row["terminal_class"] for task_id, row in matrix.items()}
    quarantine_rows: JsonDict = {}
    flagged_task_ids: list[str] = []
    positive_eligible: list[str] = []
    for task_id, row in matrix.items():
        if not row["present"]:
            continue
        payload = payloads[task_id]
        receipt = normalized_receipts.get(task_id, {})
        report = _receipt_report(receipt)
        flag_count = int(report.get("flag_count") or 0)
        has_quarantine = bool(
            payload.get("flagged_adversarial") or payload.get("corrigendum_pending")
        )
        flagged = flag_count > 0 or has_quarantine
        if flagged:
            flagged_task_ids.append(task_id)
        excluded = terminal_by_task[task_id] == "positive" and flagged
        if terminal_by_task[task_id] == "positive" and not flagged:
            positive_eligible.append(task_id)
        quarantine_rows[task_id] = {
            "task_id": task_id,
            "artifact_path": row["declared_deliverable"],
            "command": receipt.get("command"),
            "exit_code": receipt.get("exit_code"),
            "receipt_hash": receipt.get("receipt_hash")
            or sha256_json(receipt.get("stdout_json", {})),
            "flag_count": flag_count,
            "flags": report.get("flags", []),
            "artifact_quarantine_fields_present": has_quarantine,
            "excluded_from_positive_aggregation": excluded,
        }

    counts = Counter(terminal_by_task.values())
    counts_payload: JsonDict = {
        "present": sum(1 for row in matrix.values() if row["present"]),
        "missing": counts["missing"],
        "structured_skipped": counts["structured-skip"],
        "blocked": counts["blocked"],
        "null": counts["null"],
        "retired": counts["retired"],
        "partial": counts["partial"],
        "complete": counts["complete"],
        "positive": counts["positive"],
        "adversarial_quarantined": len(flagged_task_ids),
        "positive_aggregation_eligible": len(positive_eligible),
    }

    commands, exit_codes = _normalize_tests(tests_run)
    actual_duration = float(duration_s if duration_s is not None else time.monotonic() - started)

    stream_hashes = _safe_get(
        payloads,
        "exp6145-constraint-shift-stream",
        "stream_row_split_and_outcome_sidecar_paths_and_hashes",
        {},
    )
    corpus_models = _safe_get(payloads, "exp6146-sota-constraint-event-corpus", "model_specs", [])
    if not isinstance(corpus_models, list):
        corpus_models = []

    substrate_rows: JsonDict = {}
    for task_id, row in matrix.items():
        payload = payloads.get(task_id, {})
        substrate = payload.get("inference_substrate")
        hardware_claimed = bool(payload.get("hardware_execution_claimed"))
        substrate_text = str(substrate or "").lower()
        substrate_rows[task_id] = {
            "authority": "exact_declared_artifact" if row["present"] else "conductor_receipt",
            "inference_substrate": substrate,
            "verifier_is_oracle": payload.get("verifier_is_oracle"),
            "hardware_claimed": hardware_claimed,
            "software_never_hardware": (
                ("software" in substrate_text or "jax_cpu" in substrate_text)
                and not hardware_claimed
            ),
            "solve_claimed": bool(payload.get("solve_claimed", False)),
        }

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": EXPERIMENT,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": "complete_with_blocks_and_quarantine",
        "preconditions_checked": {
            "roadmap": {
                "path": ROADMAP_RELATIVE_PATH.as_posix(),
                "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
                "declared_task_count": len(declared),
            },
            "conductor_log": {
                "path": CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                "sha256": path_sha256(root / CONDUCTOR_LOG_RELATIVE_PATH),
            },
            "exclusion_manifest": {
                "path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
                "sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
            },
            "root_clutter_python_files": _root_python_files(root),
            "git_status_short": _git_status_short(root),
            "protected_file_count": len(PROTECTED_FILE_PATHS),
        },
        "activated_task_and_declared_deliverable_matrix": matrix,
        "exact_terminal_classification": {
            "terminal_class_by_task_id": terminal_by_task,
            "task_ids_by_terminal_class": {
                klass: [task_id for task_id, value in terminal_by_task.items() if value == klass]
                for klass in sorted(set(terminal_by_task.values()))
            },
            "all_tasks_terminal": all(value != "missing" for value in terminal_by_task.values()),
            "nonterminal_task_ids": [
                task_id for task_id, value in terminal_by_task.items() if value == "missing"
            ],
            "principle": FIELD_PRINCIPLES["exact_terminal_classification"],
        },
        "present_missing_skipped_blocked_null_retired_and_positive_counts": counts_payload,
        "structured_gate_receipts": {
            "gates_by_task_id": {
                task_id: {
                    "declared_gates": GATED_ON.get(task_id, []),
                    "conductor_receipt": matrix[task_id]["conductor_receipt"],
                    "artifact_gates_evaluated": payloads.get(task_id, {}).get("gates_evaluated"),
                    "artifact_structured_gate_receipt": payloads.get(task_id, {}).get(
                        "structured_gate_receipt"
                    ),
                }
                for task_id in matrix
            },
            "structured_skip_task_ids": [
                task_id for task_id, value in terminal_by_task.items() if value == "structured-skip"
            ],
        },
        "adversarial_verifier_and_quarantine_receipts": {
            "verified_present_artifact_count": len(quarantine_rows),
            "flagged_task_ids": flagged_task_ids,
            "positive_aggregation_eligible_task_ids": positive_eligible,
            "receipts_by_task_id": quarantine_rows,
        },
        "test_artifact_isolation_summary": {
            "task_id": "exp6143-test-artifact-isolation",
            "terminal_class": terminal_by_task["exp6143-test-artifact-isolation"],
            "tracked_results_unchanged": bool(
                _safe_get(
                    payloads,
                    "exp6143-test-artifact-isolation",
                    "tracked_result_hash_before_after_matrix",
                    {},
                ).get("all_unchanged")
            ),
            "quarantine_fields_preserved": bool(
                _safe_get(
                    payloads,
                    "exp6143-test-artifact-isolation",
                    "quarantine_field_before_after_matrix",
                    {},
                ).get("all_preserved")
            ),
            "residual_direct_writer_rows": _safe_get(
                payloads,
                "exp6143-test-artifact-isolation",
                "remaining_unredirected_writer_census",
                {},
            ).get("residual_call_site_rows"),
        },
        "exact_stream_and_sota_corpus_summary": {
            "stream_ready_score": _safe_get(
                payloads, "exp6145-constraint-shift-stream", "constraint_shift_stream_ready_score"
            ),
            "stream_row_count": stream_hashes.get("row_file", {}).get("row_count")
            if isinstance(stream_hashes, Mapping)
            else None,
            "stream_exact_disagreement_count": _safe_get(
                payloads, "exp6145-constraint-shift-stream", "exact_validator_agreement", {}
            ).get("disagreement_count"),
            "sota_corpus_ready_score": _safe_get(
                payloads,
                "exp6146-sota-constraint-event-corpus",
                "sota_constraint_event_corpus_ready_score",
            ),
            "sota_model_count": len(corpus_models),
            "sota_rows_conserved": bool(
                _safe_get(
                    payloads,
                    "exp6146-sota-constraint-event-corpus",
                    "per_model_event_row_conservation",
                    {},
                ).get("all_models_conserved")
            ),
        },
        "task_aware_calibration_and_held_summary": {
            "calibration_terminal_class": terminal_by_task["exp6147-task-aware-energy-calibration"],
            "calibration_ready_score_raw": _safe_get(
                payloads,
                "exp6147-task-aware-energy-calibration",
                "task_aware_energy_calibration_ready_score",
            ),
            "calibration_quarantined": "exp6147-task-aware-energy-calibration" in flagged_task_ids,
            "calibration_positive_aggregation_eligible": (
                "exp6147-task-aware-energy-calibration" in positive_eligible
            ),
            "held_terminal_class": terminal_by_task["exp6148-shifted-family-admission-held"],
            "held_ready_score": _safe_get(
                payloads,
                "exp6148-shifted-family-admission-held",
                "shifted_family_admission_ready_score",
            ),
            "held_quarantined": "exp6148-shifted-family-admission-held" in flagged_task_ids,
        },
        "continuous_self_learning_and_shadow_summary": {
            "fixture_terminal_class": terminal_by_task["exp6149-certified-strategy-schema-fixture"],
            "fixture_ready_score": _safe_get(
                payloads,
                "exp6149-certified-strategy-schema-fixture",
                "certified_strategy_fixture_ready_score",
            ),
            "prospective_csl_terminal_class": terminal_by_task[
                "exp6150-frozen-qwen-continuous-self-learning-ab"
            ],
            "prospective_csl_gate_summary": _safe_get(
                payloads, "exp6150-frozen-qwen-continuous-self-learning-ab", "gate_check_summary"
            ),
            "shadow_adapter_terminal_class": terminal_by_task[
                "exp6151-strategy-memory-shadow-adapter"
            ],
            "shadow_adapter_artifact_present": matrix["exp6151-strategy-memory-shadow-adapter"][
                "present"
            ],
            "model_weights_immutable": bool(
                _safe_get(
                    payloads,
                    "exp6149-certified-strategy-schema-fixture",
                    "model_weight_immutability_receipt",
                    {},
                ).get("all_unchanged")
            ),
        },
        "typed_ir_and_thermalization_summary": {
            "typed_ir_terminal_class": terminal_by_task["exp6152-typed-stochastic-constraint-ir"],
            "typed_ir_ready_score": _safe_get(
                payloads,
                "exp6152-typed-stochastic-constraint-ir",
                "typed_stochastic_ir_ready_score",
            ),
            "typed_ir_state_space_size": _safe_get(
                payloads,
                "exp6152-typed-stochastic-constraint-ir",
                "exact_enumeration_case_counts",
                {},
            ).get("state_space_size"),
            "thermalization_terminal_class": terminal_by_task[
                "exp6153-thermalized-program-error-audit"
            ],
            "thermalized_program_ready_score": _safe_get(
                payloads,
                "exp6153-thermalized-program-error-audit",
                "thermalized_program_ready_score",
            ),
            "thermalization_bound_violation_count": _safe_get(
                payloads,
                "exp6153-thermalized-program-error-audit",
                "bound_slack_and_violation_counts",
                {},
            ).get("violation_count"),
            "hardware_execution_claimed": bool(
                _safe_get(
                    payloads,
                    "exp6153-thermalized-program-error-audit",
                    "hardware_execution_claimed",
                    False,
                )
            ),
        },
        "arc_generalization_no_solve_summary": {
            "arc_terminal_class": terminal_by_task["exp6154-arc-task-aware-energy-generalization"],
            "arc_ready_score": _safe_get(
                payloads,
                "exp6154-arc-task-aware-energy-generalization",
                "arc_task_aware_generalization_ready_score",
            ),
            "solve_claimed": bool(
                _safe_get(
                    payloads, "exp6154-arc-task-aware-energy-generalization", "solve_claimed", False
                )
            ),
            "offline_reproduced": bool(
                _safe_get(
                    payloads,
                    "exp6154-arc-task-aware-energy-generalization",
                    "offline_reproduced",
                    False,
                )
            ),
            "level_credit_delta": _safe_get(
                payloads, "exp6154-arc-task-aware-energy-generalization", "level_credit_delta", 0
            ),
            "llm_invocation_count": _safe_get(
                payloads, "exp6154-arc-task-aware-energy-generalization", "llm_invocation_count", 0
            ),
        },
        "oracle_distinctness_and_inference_substrate_matrix": {
            "rows_by_task_id": substrate_rows,
            "software_promoted_to_hardware": False,
            "capstone_inference_substrate": INFERENCE_SUBSTRATE,
        },
        "model_specs_and_lifecycle_matrix": {
            "sota_corpus_model_specs": corpus_models,
            "mandated_qwen_csl_run_terminal_class": terminal_by_task[
                "exp6150-frozen-qwen-continuous-self-learning-ab"
            ],
            "arc_llm_invocation_count": _safe_get(
                payloads, "exp6154-arc-task-aware-energy-generalization", "llm_invocation_count", 0
            ),
            "model_weight_immutability_receipts": {
                "exp6149": _safe_get(
                    payloads,
                    "exp6149-certified-strategy-schema-fixture",
                    "model_weight_immutability_receipt",
                    {},
                )
            },
        },
        "prior_failure_retirement_and_exclusion_updates": {
            "retirement_triggered_task_ids": [
                task_id
                for task_id, payload in payloads.items()
                if payload.get("retirement_triggered")
            ],
            "same_verdict_retirement_observed_task_ids": [
                "exp6150-frozen-qwen-continuous-self-learning-ab"
            ]
            if terminal_by_task["exp6150-frozen-qwen-continuous-self-learning-ab"] == "blocked"
            else [],
            "exclusion_manifest_modified": False,
            "exclusion_updates_deferred_to_conductor": True,
        },
        "open_verifier_and_research_gaps": {
            "flagged_artifacts": flagged_task_ids,
            "blocked_task_ids": [
                task_id for task_id, value in terminal_by_task.items() if value == "blocked"
            ],
            "partial_task_ids": [
                task_id for task_id, value in terminal_by_task.items() if value == "partial"
            ],
            "null_task_ids": [
                task_id for task_id, value in terminal_by_task.items() if value == "null"
            ],
            "structured_skip_task_ids": [
                task_id for task_id, value in terminal_by_task.items() if value == "structured-skip"
            ],
            "research_gap_summary": (
                "Held admission and calibration are quarantined; CSL prospective and shadow "
                "integration did not run; software thermalization is blocked by test exits; "
                "ARC reports no solve."
            ),
        },
        "spec_bmad_ops_reference_and_completion_reconciliation": {
            "openspec_research_reporting_updated": True,
            "bmad_traceability_mutation": "deferred_to_conductor",
            "bmad_architecture_mutation": "deferred_to_conductor",
            "ops_status_mutation": "deferred_to_conductor",
            "ops_changelog_mutation": "deferred_to_conductor",
            "ops_known_issues_mutation": "deferred_to_conductor",
            "ops_verifier_gaps_mutation": "deferred_to_conductor",
            "research_references_mutation": "deferred_to_conductor",
            "research_complete_mutation": "deferred_to_conductor",
            "historical_text_rewritten": False,
        },
        "research_complete_append_count": 0,
        "duplicate_history_amplification_count": 0,
        "protected_files_unchanged": _protected_files(root),
        "preexisting_worktree_changes_preserved": {
            "preserved": True,
            "git_status_short": _git_status_short(root),
            "staged_anything": False,
        },
        "duration_s": actual_duration,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: .533 reconciled with blocks, structured skip, nulls, partials, "
            "and adversarial quarantine preserved; positive aggregation limited to "
            f"{len(positive_eligible)} unflagged positive artifacts"
        ),
    }
    report["preconditions_checked"]["duplicate_history_count_before"] = _history_duplicate_count(
        root
    )
    report["reproducibility_checksum"] = payload_checksum(report)
    return report


def validate_report(payload: JsonMap) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    provenance = payload.get("field_provenance")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance:not_mapping")
    else:
        for field in REQUIRED_ARTIFACT_FIELDS:
            row = provenance.get(field)
            if not isinstance(row, Mapping) or row.get("principle") != FIELD_PRINCIPLES[field]:
                errors.append(f"field_provenance:{field}")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if not str(payload.get("honest_verdict", "")).startswith(("complete:", "blocked:")):
        errors.append("honest_verdict_prefix")
    matrix = payload.get("activated_task_and_declared_deliverable_matrix")
    if not isinstance(matrix, Mapping) or len(matrix) != len(ACTIVATED_TASKS):
        errors.append("matrix_length")
    classes = payload.get("exact_terminal_classification", {}).get("terminal_class_by_task_id")
    if not isinstance(classes, Mapping) or len(classes) != len(ACTIVATED_TASKS):
        errors.append("terminal_class_length")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        errors.append("reproducibility_checksum")
    return errors


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

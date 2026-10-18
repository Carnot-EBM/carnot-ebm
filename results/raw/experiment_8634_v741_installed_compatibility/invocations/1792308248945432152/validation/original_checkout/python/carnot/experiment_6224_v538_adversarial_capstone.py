"""Exp6224 V538 exact-path adversarial capstone.

Spec refs: REQ-CAPSTONE-6224, SCENARIO-CAPSTONE-6224,
SCENARIO-CAPSTONE-6224-EXACT-PATH,
SCENARIO-CAPSTONE-6224-BRANCH-INDEPENDENCE,
SCENARIO-CAPSTONE-6224-GATEMATE,
SCENARIO-CAPSTONE-6224-ARC-REGISTRY,
SCENARIO-CAPSTONE-6224-FIELD-PRINCIPLES.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot import experiment_6210_v537_adversarial_capstone as v537
from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import (
    NONTERMINAL_CLASSES,
    canonical_json,
    classify_artifact_path,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
CommandRunner = Callable[[tuple[str, ...], Path], JsonDict]
Validator = Callable[[JsonMap], list[str]]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.538"
EXPERIMENT_ID = "exp6224-v538-adversarial-capstone"
SCHEMA = "carnot.experiment_6224.v538_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6224_v538_adversarial_capstone.json")
INFERENCE_SUBSTRATE = "deterministic_exact_path_v538_capstone_reconciliation"

ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
CONDUCTOR_RELATIVE_PATH = Path("scripts/research_conductor.py")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/capstone/spec.md")
ARC_REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
CLASSIFIER_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
SUMMARY_RELATIVE_PATH = Path("scripts/summarize_artifact.py")
PUBLICATION_GATE_RELATIVE_PATH = Path("scripts/publication_gate.py")
GATEMATE_RECEIPT_RELATIVE_PATH = Path(
    "results/experiment_6199_gatemate_terminal_action_audit_v537.json"
)

V538_UPSTREAM_RANGE = range(6211, 6224)
ARC_TASK_IDS = (
    "exp6213-arc-object-delta-perception-wiring",
    "exp6214-arc-object-delta-heldout-ab",
    "exp6215-arc-trajectory-transfer-ab",
    "exp6216-arc-budget-aware-search-ab",
    "exp6217-arc-gemma31-think-ab",
    "exp6218-arc-admissible-lever-portfolio-heldout",
)
GGUF_RUNTIME_TASK_IDS = ("exp6212-three-family-gguf-runtime-recovery",)
CSL_TASK_IDS = ("exp6219-two-timescale-constraint-csl",)
SAMPLER_TASK_IDS = ("exp6220-mode-jump-runtime-ab",)
PHASE_D_TASK_IDS = (
    "exp6221-three-family-code-transport-canary-v3",
    "exp6222-livecodebench-k8-pool-v3",
    "exp6223-livecodebench-headroom-v3",
)

BRANCH_SCORE_RULES: dict[str, dict[str, dict[str, Any]]] = {
    "gguf_runtime": {
        "exp6212-three-family-gguf-runtime-recovery": {
            "three_family_runtime_ready_score": 1,
            "gemma_4_31b_runtime_ready_score": 1,
        }
    },
    "continuous_self_learning": {
        "exp6219-two-timescale-constraint-csl": {
            "continuous_self_learning_ready_score": 1,
        }
    },
    "sampler_runtime": {
        "exp6220-mode-jump-runtime-ab": {
            "sampler_runtime_ready_score": 1,
        }
    },
    "phase_d": {
        "exp6221-three-family-code-transport-canary-v3": {
            "phase_d_transport_ready_score": 1,
        },
        "exp6222-livecodebench-k8-pool-v3": {
            "pool_integrity_ready_score": 1,
        },
        "exp6223-livecodebench-headroom-v3": {
            "headroom_ready_score": 1,
        },
    },
}

UNAUTHORIZED_HARDWARE_CLAIM_KEYS = (
    "speed",
    "power",
    "energy",
    "terminal",
    "terminal_hardware",
    "tsu",
    "kona",
    "hardware_speedup",
    "hardware_execution",
    "fpga",
    "board_command",
)

PROTECTED_RELATIVE_PATHS = (
    CONDUCTOR_RELATIVE_PATH,
    ARC_REGISTRY_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    STATUS_RELATIVE_PATH,
    CHANGELOG_RELATIVE_PATH,
    TRACEABILITY_RELATIVE_PATH,
    ROADMAP_RELATIVE_PATH,
    CODEX_RELATIVE_PATH,
    CLAUDE_RELATIVE_PATH,
)

DEFAULT_COMMANDS = (
    ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
    ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
    ".venv/bin/python scripts/publication_gate.py --json",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6224_v538_adversarial_capstone.py",
    ".venv/bin/ruff check python/carnot/experiment_6224_v538_adversarial_capstone.py tests/python/test_experiment_6224_v538_adversarial_capstone.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6224_v538_adversarial_capstone.py tests/python/test_experiment_6224_v538_adversarial_capstone.py",
    ".venv/bin/pytest tests/python/test_experiment_6224_v538_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6224_v538_adversarial_capstone.py -m pytest tests/python/test_experiment_6224_v538_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6224_v538_adversarial_capstone.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "declared_task_ids_and_deliverables",
    "exact_artifact_paths_hashes_and_terminal_classifications",
    "missing_nonterminal_blocked_skipped_null_retired_and_flagged_counts",
    "conductor_receipt_override_count",
    "adversarial_verification_results",
    "arc_registry_hash_before_after",
    "arc_solve_claim_count",
    "arc_level_credit_delta",
    "arc_lever_and_portfolio_eligibility",
    "gguf_runtime_eligibility",
    "continuous_self_learning_eligibility",
    "sampler_runtime_eligibility",
    "phase_d_transport_pool_and_headroom_eligibility",
    "gate_cascade_receipts",
    "gatemate_cached_state_and_new_receipt_count",
    "hardware_claim_eligibility",
    "publication_gate_snapshot",
    "exclusion_manifest_and_prior_failure_reconciliation",
    "spec_trace_status_changelog_reconciliation",
    "protected_files_unchanged",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "terminal only after exact path classification, verifier replay, zero-count checks, and checksum validation finish",
    "declared_task_ids_and_deliverables": "fixed Exp6211-Exp6223 denominator from the active roadmap exact paths",
    "exact_artifact_paths_hashes_and_terminal_classifications": "Exp6197 classifier output, path hash, load state, and conductor receipt state stay adjacent",
    "missing_nonterminal_blocked_skipped_null_retired_and_flagged_counts": "negative classes stay disjoint before branch summaries",
    "conductor_receipt_override_count": "completion receipts never promote an artifact",
    "adversarial_verification_results": "each present exact artifact has a replayed verifier receipt or an injected test receipt",
    "arc_registry_hash_before_after": "registry bytes are content-addressed before and after the run",
    "arc_solve_claim_count": "bare zero prevents hidden public-game solve banking",
    "arc_level_credit_delta": "bare zero prevents level-credit inflation",
    "arc_lever_and_portfolio_eligibility": "ARC wiring, lever A/Bs, think A/B, and portfolio close only from exact no-solve artifacts",
    "gguf_runtime_eligibility": "runtime promotion requires dense and three-family readiness from Exp6212",
    "continuous_self_learning_eligibility": "CSL eligibility is independent from Phase-D and sampler gate cascades",
    "sampler_runtime_eligibility": "sampler runtime eligibility uses Exp6220 quality, state, timing, fallback, and hardware-claim gates",
    "phase_d_transport_pool_and_headroom_eligibility": "code transport, pool, and headroom require the exact Exp6221-Exp6223 cascade",
    "gate_cascade_receipts": "declared roadmap gates are replayed from raw upstream artifact fields",
    "gatemate_cached_state_and_new_receipt_count": "cached hardware state remains blocked without a newer dated physical receipt",
    "hardware_claim_eligibility": "physical hardware promotion is false unless authenticated workload and receipt gates pass",
    "publication_gate_snapshot": "G1-G4 publication state is imported from the stable gate script",
    "exclusion_manifest_and_prior_failure_reconciliation": "exclusion lint and prior-failure receipts stay visible without deleting history",
    "spec_trace_status_changelog_reconciliation": "OpenSpec is updated here, while ops/status, ops/changelog, and traceability edits are deferred by the operator stop rule",
    "protected_files_unchanged": "conductor, ops ledgers, traceability, ARC registry, and exclusion manifest hashes are compared",
    "inference_substrate": "deterministic_exact_path_v538_capstone_reconciliation",
    "verifier_is_oracle": "bare false because this capstone verifies evidence discipline, not benchmark answers",
    "field_provenance": "each required field names roadmap, exact artifacts, verifier receipts, registry, publication gate, GateMate receipt, exclusions, or local hashes",
    "field_principles": "required field purpose is emitted next to the measured value",
    "test_commands": "focused, coverage, spec, capstone, adversarial, and suite commands are replayable",
    "test_exit_codes": "observed exits are recorded without laundering failures",
    "duration_s": "wall time for deterministic aggregation is reported without padding",
    "reproducibility_checksum": "the normalized report is content-addressed",
    "honest_verdict": "terminal summary starts with complete: and names every blocked or missing boundary without strengthening claims",
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


def _read_yaml_mapping(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _roadmap_declared_tasks(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    roadmap = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    raw_tasks = roadmap.get("tasks")
    tasks = (
        [dict(row) for row in raw_tasks if isinstance(row, Mapping)]
        if isinstance(raw_tasks, list)
        else []
    )
    by_id = {str(row.get("id")): row for row in tasks if row.get("id")}
    capstone = by_id.get(EXPERIMENT_ID, {})
    requires = capstone.get("requires")
    if isinstance(requires, list):
        wanted = [str(task_id) for task_id in requires]
    else:
        wanted = [
            str(row.get("id"))
            for row in tasks
            if (number := v537._experiment_number(str(row.get("id")))) in V538_UPSTREAM_RANGE
        ]
    declared: list[JsonDict] = []
    for task_id in wanted:
        number = v537._experiment_number(task_id)
        if number not in V538_UPSTREAM_RANGE:
            continue
        row = by_id.get(task_id, {})
        declared.append(
            {
                "task_id": task_id,
                "title": str(row.get("title") or task_id),
                "track": str(row.get("track") or ""),
                "deliverable": Path(str(row.get("deliverable") or "")),
                "requires": [str(item) for item in row.get("requires", [])]
                if isinstance(row.get("requires"), list)
                else [],
                "gated_on": [dict(item) for item in row.get("gated_on", [])]
                if isinstance(row.get("gated_on"), list)
                else [],
                "required_fields": v537._required_fields_from_prompt(str(row.get("prompt") or "")),
                "prior_failures": [dict(item) for item in row.get("prior_failures", [])]
                if isinstance(row.get("prior_failures"), list)
                else [],
            }
        )
    return declared, roadmap, dict(capstone) if isinstance(capstone, Mapping) else {}


def _argv(command: str) -> tuple[str, ...]:
    return tuple(command.split())


def _run_command(argv: tuple[str, ...], root: Path) -> JsonDict:  # pragma: no cover - shell edge.
    command = " ".join(argv)
    try:
        proc = subprocess.run(argv, cwd=root, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return {
            "command": command,
            "exit_code": 127,
            "classification": "command_not_found",
            "error": str(exc),
        }
    receipt: JsonDict = {
        "command": command,
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if command.endswith("publication_gate.py --json"):
        try:
            receipt["stdout_json"] = json.loads(proc.stdout)
        except json.JSONDecodeError:
            receipt["stdout_json"] = {"parse_error": True}
    return receipt


def _run_commands(
    root: Path, commands: Sequence[str], runner: CommandRunner
) -> list[JsonDict]:  # pragma: no cover - shell edge.
    return [runner(_argv(command), root) for command in commands]


def _command_receipts(
    root: Path,
    command_receipts: Sequence[JsonMap] | None,
    runner: CommandRunner,
) -> list[JsonDict]:
    if command_receipts is not None:
        rows: list[JsonDict] = []
        for row in command_receipts:
            normalized = dict(row)
            normalized.setdefault("command", "")
            normalized.setdefault("exit_code", 0)
            normalized.setdefault(
                "classification",
                "passed"
                if int(normalized.get("exit_code") or 0) == 0
                else f"nonzero_exit_{normalized.get('exit_code')}",
            )
            rows.append(normalized)
        return rows
    return _run_commands(root, DEFAULT_COMMANDS, runner)


def _normalize_test_commands(
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None,
    command_rows: Sequence[JsonMap],
) -> tuple[list[str], JsonDict]:
    commands: list[str] = []
    exits: JsonDict = {}
    if tests_run is not None:
        base_commands, base_exits = v537._normalize_tests(tests_run)
        commands.extend(base_commands)
        exits.update(base_exits)
    for row in command_rows:
        command = str(row.get("command") or "")
        if not command:
            continue
        if command not in commands:
            commands.append(command)
        exits[command] = int(row.get("exit_code") or 0)
    return commands, exits


def _publication_snapshot(command_rows: Sequence[JsonMap]) -> JsonDict:
    for row in command_rows:
        if str(row.get("command") or "").endswith("publication_gate.py --json"):
            payload = row.get("stdout_json")
            if isinstance(payload, Mapping):
                return dict(payload)
            raw = row.get("stdout_tail")
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, Mapping):
                    return dict(parsed)
    return {
        "paper_ready": False,
        "gates": {},
        "unmet_gates": ["publication_gate_not_run"],
        "note": "publication_gate.py --json receipt was absent or unparseable",
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": FIELD_PRINCIPLES[field],
            "sources": [
                "REQ-CAPSTONE-6224",
                ROADMAP_RELATIVE_PATH.as_posix(),
                CLASSIFIER_RELATIVE_PATH.as_posix(),
                ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
                PUBLICATION_GATE_RELATIVE_PATH.as_posix(),
                ARC_REGISTRY_RELATIVE_PATH.as_posix(),
                GATEMATE_RECEIPT_RELATIVE_PATH.as_posix(),
                "exact_declared_artifacts",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _real_field_items(value: Any, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    if any(part in {"field_principles", "field_provenance"} for part in path):
        return []
    if isinstance(value, Mapping):
        out: list[tuple[tuple[str, ...], Any]] = []
        for key, child in value.items():
            out.extend(_real_field_items(child, (*path, str(key))))
        return out
    if isinstance(value, list):
        out = []
        for index, child in enumerate(value):
            out.extend(_real_field_items(child, (*path, str(index))))
        return out
    return [(path, value)]


def _positive_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return 0


def _arc_claim_counts(payloads: Mapping[str, JsonMap]) -> tuple[int, int]:
    solves = 0
    credit = 0
    for payload in payloads.values():
        for path, value in _real_field_items(payload):
            key = path[-1] if path else ""
            if key in {"solve_claimed", "arc_solve_claimed"} and value is True:
                solves += 1
            elif key in {"solve_claim_count", "arc_solve_claim_count"}:
                solves += _positive_int(value)
            elif key in {"level_credit_delta", "arc_level_credit_delta"}:
                credit += _positive_int(value)
    return int(solves), int(credit)


def _hardware_claim_counts(payloads: Mapping[str, JsonMap]) -> JsonDict:
    counts = {key: 0 for key in UNAUTHORIZED_HARDWARE_CLAIM_KEYS}
    for payload in payloads.values():
        for path, value in _real_field_items(payload):
            key = path[-1] if path else ""
            parent = path[-2] if len(path) > 1 else ""
            if parent == "speed_power_energy_terminal_tsu_kona_claim_counts" and key in counts:
                counts[key] += _positive_int(value)
            elif key in {"hardware_claim_count", "unauthorized_hardware_claim_count"}:
                counts["hardware_speedup"] += _positive_int(value)
            elif key in {"hardware_speedup_claim", "speedup_claimed"} and value is True:
                counts["hardware_speedup"] += 1
            elif (
                key in {"hardware_execution_claimed", "hardware_execution_authenticated"}
                and value is True
            ):
                counts["hardware_execution"] += 1
    return {key: int(value) for key, value in counts.items()}


def _protected_hashes(root: Path) -> JsonDict:
    return {rel.as_posix(): path_sha256(root / rel) for rel in PROTECTED_RELATIVE_PATHS}


def _protected_unchanged(before: JsonMap, root: Path) -> JsonDict:
    after = _protected_hashes(root)
    rows = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {
        "unchanged": all(row["unchanged"] for row in rows.values()),
        "paths": rows,
    }


def _task_blockers(
    task_id: str,
    task_rows: Mapping[str, JsonMap],
    gates: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    score_rules: Mapping[str, Any] | None = None,
) -> list[str]:
    row = task_rows.get(task_id, {})
    gate_row = gates.get(task_id, {})
    blockers: list[str] = []
    terminal_class = str(row.get("terminal_class"))
    if terminal_class in set(NONTERMINAL_CLASSES) | {
        "blocked",
        "skipped",
        "retired",
        "flagged",
        "null",
    }:
        blockers.append(terminal_class)
    if int(row.get("critical_adversarial_flag_count") or 0) > 0:
        blockers.append("critical_adversarial_flag")
    if row.get("missing_required_fields"):
        blockers.append("missing_required_field")
    if row.get("unclassified_nonzero_commands"):
        blockers.append("unclassified_nonzero_command")
    declared_gates = gate_row.get("declared_gates")
    if isinstance(declared_gates, list) and any(
        isinstance(gate, Mapping) and gate.get("passed") is False for gate in declared_gates
    ):
        blockers.append("failed_gate")
    payload = payloads.get(task_id, {})
    for field, expected in (score_rules or {}).items():
        actual = payload.get(field)
        if actual != expected:
            blockers.append(f"{field}={actual}")
    return blockers


def _branch_eligibility(
    branch: str,
    task_ids: Sequence[str],
    task_rows: Mapping[str, JsonMap],
    gates: Mapping[str, JsonMap],
    payloads: Mapping[str, JsonMap],
    score_rules: Mapping[str, Mapping[str, Any]] | None = None,
) -> JsonDict:
    blockers: list[str] = []
    per_task: JsonDict = {}
    for task_id in task_ids:
        reasons = _task_blockers(
            task_id,
            task_rows,
            gates,
            payloads,
            (score_rules or {}).get(task_id, {}),
        )
        per_task[task_id] = {
            "terminal_class": task_rows.get(task_id, {}).get("terminal_class"),
            "blocking_reasons": reasons,
        }
        blockers.extend(f"{task_id}:{reason}" for reason in reasons)
    return {
        "branch": branch,
        "eligible": not blockers,
        "task_ids": list(task_ids),
        "per_task": per_task,
        "blocking_reasons": blockers,
        "reason": "eligible_from_terminal_clean_exact_artifacts"
        if not blockers
        else "; ".join(blockers),
    }


def _gate_receipts(
    declared: Sequence[JsonMap],
    payloads: Mapping[str, JsonMap],
    task_rows: Mapping[str, JsonMap],
    conductor_receipts: Mapping[str, JsonMap],
) -> JsonDict:
    receipts: JsonDict = {}
    for row in declared:
        task_id = str(row["task_id"])
        gate_rows: list[JsonDict] = []
        for gate in row["gated_on"]:
            actual = v537._gate_actual_value(payloads, gate)
            expected = gate.get("value")
            op = str(gate.get("op"))
            upstream = str(gate.get("upstream"))
            gate_rows.append(
                {
                    **gate,
                    "actual": actual,
                    "passed": v537._gate_passed(actual, op, expected),
                    "upstream_terminal_class": task_rows.get(upstream, {}).get("terminal_class"),
                    "reason": None
                    if actual is not None
                    else f"upstream artifact field unavailable for {upstream}.{gate.get('artifact_field')}",
                }
            )
        receipts[task_id] = {
            "declared_gates": gate_rows,
            "artifact_gates_evaluated": payloads.get(task_id, {}).get("gates_evaluated", []),
            "conductor_gate_block": conductor_receipts.get(task_id, {}).get("status")
            == "GATE_BLOCK",
            "terminal_class": task_rows.get(task_id, {}).get("terminal_class"),
        }
    return receipts


def _gatemate_state(root: Path) -> JsonDict:
    payload, meta = v537._read_json_mapping(root / GATEMATE_RECEIPT_RELATIVE_PATH)
    receipt = payload.get("current_dated_operator_receipt") if isinstance(payload, Mapping) else {}
    attempts = (
        payload.get("detect_attempt_count_command_stdout_stderr_exit_code")
        if isinstance(payload, Mapping)
        else {}
    )
    return {
        "exp6199_path": GATEMATE_RECEIPT_RELATIVE_PATH.as_posix(),
        "exp6199_sha256": meta["sha256"],
        "exp6199_present": meta["present"],
        "exp6199_loadable": meta["loadable"],
        "exp6199_status": payload.get("status"),
        "exp6199_honest_verdict": payload.get("honest_verdict"),
        "cached_current_dated_operator_receipt": receipt if isinstance(receipt, Mapping) else {},
        "physical_state_changed": payload.get("physical_state_changed"),
        "new_receipt_count": 0,
        "new_receipts": [],
        "board_command_count": 0,
        "detect_attempt_count": attempts.get("attempt_count")
        if isinstance(attempts, Mapping)
        else None,
        "boundary_imported_from_exp6199": True,
    }


def _spec_reconciliation(root: Path) -> JsonDict:
    return {
        "openspec_req_capstone_6224_present": "REQ-CAPSTONE-6224"
        in v537._read_text(root / SPEC_RELATIVE_PATH),
        "spec_path": SPEC_RELATIVE_PATH.as_posix(),
        "traceability_path": TRACEABILITY_RELATIVE_PATH.as_posix(),
        "status_path": STATUS_RELATIVE_PATH.as_posix(),
        "changelog_path": CHANGELOG_RELATIVE_PATH.as_posix(),
        "ops_status_changelog_traceability_modified": False,
        "ops_status_changelog_traceability_deferred_by_stop_rule": True,
        "path_hashes": {
            rel.as_posix(): path_sha256(root / rel)
            for rel in (
                SPEC_RELATIVE_PATH,
                TRACEABILITY_RELATIVE_PATH,
                STATUS_RELATIVE_PATH,
                CHANGELOG_RELATIVE_PATH,
            )
        },
    }


def _exclusion_and_prior(
    root: Path,
    declared: Sequence[JsonMap],
    capstone_row: JsonMap,
    task_rows: Mapping[str, JsonMap],
    command_rows: Sequence[JsonMap],
) -> JsonDict:
    command_index = {str(row.get("command") or ""): dict(row) for row in command_rows}
    return {
        "exclusion_manifest_path": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        "exclusion_manifest_sha256": path_sha256(root / EXCLUSION_MANIFEST_RELATIVE_PATH),
        "exclusion_lint_command": command_index.get(
            ".venv/bin/python scripts/exclusion_manifest_lint.py research-roadmap.yaml",
            {},
        ),
        "prior_failure_lint_command": command_index.get(
            ".venv/bin/python scripts/validate_prior_failures.py research-roadmap.yaml",
            {},
        ),
        "roadmap_gate_audit_command": command_index.get(
            ".venv/bin/python scripts/audit_roadmap_gates.py research-roadmap.yaml",
            {},
        ),
        "prior_failure_retirement_actions": v537._prior_failure_actions(
            declared, capstone_row, task_rows
        ),
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    verifier_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    command_receipts: Sequence[JsonMap] | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.monotonic()
    protected_before = _protected_hashes(root)
    registry_before = path_sha256(root / ARC_REGISTRY_RELATIVE_PATH)
    declared, roadmap, capstone_row = _roadmap_declared_tasks(root)
    log_text = v537._read_text(root / CONDUCTOR_LOG_RELATIVE_PATH)

    payloads: dict[str, JsonDict] = {}
    present_paths: dict[str, Path] = {}
    conductor_receipts: JsonDict = {}
    classifier_results: JsonDict = {}
    exact_rows: JsonDict = {}

    for row in declared:
        task_id = str(row["task_id"])
        rel_path = Path(row["deliverable"])
        payload, meta = v537._read_json_mapping(root / rel_path)
        payloads[task_id] = payload
        receipt = v537._latest_conductor_receipt(log_text, str(row["title"]))
        conductor_receipts[task_id] = receipt
        classified = classify_artifact_path(root / rel_path, conductor_receipt=receipt)
        classifier_results[task_id] = classified.to_dict()
        if meta["present"] and meta["loadable"]:
            present_paths[task_id] = rel_path
        exact_rows[task_id] = {
            "task_id": task_id,
            "title": row["title"],
            "track": row["track"],
            "declared_deliverable": rel_path.as_posix(),
            "present": bool(meta["present"]),
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "error": meta["error"],
            "classifier_class": classified.classification,
            "classifier_terminal": classified.terminal,
            "classification": classified.classification,
            "reason": classified.reason,
            "receipt_override_attempted": classified.receipt_override_attempted,
            "receipt_overrode": classified.receipt_overrode,
            "conductor_receipt_status": classified.conductor_receipt_status,
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": v537._same_number_aliases(
                root, task_id, rel_path
            ),
            "required_fields": row["required_fields"],
        }

    raw_receipts = (
        v537._run_artifact_verifiers(root, present_paths)
        if verifier_receipts is None
        else verifier_receipts
    )
    normalized_receipts = v537._normalize_verifier_receipts(raw_receipts)

    task_rows: JsonDict = {}
    for row in declared:
        task_id = str(row["task_id"])
        payload = payloads.get(task_id, {})
        exact = exact_rows[task_id]
        critical = v537._critical_flag_count(normalized_receipts.get(task_id, {}))
        terminal_class = str(exact["classifier_class"])
        if critical or payload.get("flagged_adversarial") is True:
            terminal_class = "flagged"
        exact["critical_adversarial_flag_count"] = critical
        exact["flag_count"] = v537._flag_count(normalized_receipts.get(task_id, {}))
        exact["classification"] = terminal_class
        missing_fields = (
            v537._missing_required_fields(payload, row["required_fields"]) if payload else []
        )
        task_rows[task_id] = {
            "task_id": task_id,
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "classifier_class": exact["classifier_class"],
            "classifier_terminal": exact["classifier_terminal"],
            "terminal_class": terminal_class,
            "present": exact["present"],
            "loadable": exact["loadable"],
            "critical_adversarial_flag_count": critical,
            "flag_count": exact["flag_count"],
            "missing_required_fields": missing_fields,
            "unclassified_nonzero_commands": v537._unclassified_nonzero_commands(payload),
        }

    gates = _gate_receipts(declared, payloads, task_rows, conductor_receipts)
    command_rows = _command_receipts(root, command_receipts, command_runner)
    test_commands, test_exits = _normalize_test_commands(tests_run, command_rows)
    registry_after = path_sha256(root / ARC_REGISTRY_RELATIVE_PATH)
    class_counts = Counter(str(row["terminal_class"]) for row in task_rows.values())
    counts: JsonDict = {
        "missing": class_counts.get("missing", 0),
        "nonterminal": sum(class_counts.get(name, 0) for name in NONTERMINAL_CLASSES),
        "blocked": class_counts.get("blocked", 0),
        "skipped": class_counts.get("skipped", 0),
        "null": class_counts.get("null", 0),
        "retired": class_counts.get("retired", 0),
        "flagged": class_counts.get("flagged", 0),
        "classification_counts": dict(sorted(class_counts.items())),
    }
    solve_count, level_credit = _arc_claim_counts(payloads)
    gatemate = _gatemate_state(root)
    hardware_counts = _hardware_claim_counts(
        {**payloads, "exp6199": v537._read_json_mapping(root / GATEMATE_RECEIPT_RELATIVE_PATH)[0]}
    )
    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": date,
        "status": "complete",
        "declared_task_ids_and_deliverables": {
            "milestone": MILESTONE,
            "roadmap_path": ROADMAP_RELATIVE_PATH.as_posix(),
            "roadmap_sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
            "roadmap_task_count": len(roadmap.get("tasks", []))
            if isinstance(roadmap.get("tasks"), list)
            else 0,
            "task_count": len(declared),
            "task_ids": [str(row["task_id"]) for row in declared],
            "tasks": [
                {
                    "task_id": row["task_id"],
                    "track": row["track"],
                    "deliverable": Path(row["deliverable"]).as_posix(),
                    "requires": row["requires"],
                    "gated_on": row["gated_on"],
                }
                for row in declared
            ],
        },
        "exact_artifact_paths_hashes_and_terminal_classifications": exact_rows,
        "missing_nonterminal_blocked_skipped_null_retired_and_flagged_counts": counts,
        "conductor_receipt_override_count": sum(
            1 for row in exact_rows.values() if row.get("receipt_overrode") is True
        ),
        "adversarial_verification_results": {
            "artifact_receipts": normalized_receipts,
            "artifact_receipt_count": len(normalized_receipts),
            "flag_count": sum(
                int(row.get("flag_count") or 0) for row in normalized_receipts.values()
            ),
            "critical_flag_count": sum(
                int(row.get("critical_flag_count") or 0) for row in normalized_receipts.values()
            ),
            "command_receipts": command_rows,
        },
        "arc_registry_hash_before_after": {
            "path": ARC_REGISTRY_RELATIVE_PATH.as_posix(),
            "before_sha256": registry_before,
            "after_sha256": registry_after,
            "unchanged": registry_before == registry_after,
        },
        "arc_solve_claim_count": int(solve_count),
        "arc_level_credit_delta": int(level_credit),
        "arc_lever_and_portfolio_eligibility": _branch_eligibility(
            "arc_lever_and_portfolio", ARC_TASK_IDS, task_rows, gates, payloads
        ),
        "gguf_runtime_eligibility": _branch_eligibility(
            "gguf_runtime",
            GGUF_RUNTIME_TASK_IDS,
            task_rows,
            gates,
            payloads,
            BRANCH_SCORE_RULES["gguf_runtime"],
        ),
        "continuous_self_learning_eligibility": _branch_eligibility(
            "continuous_self_learning",
            CSL_TASK_IDS,
            task_rows,
            gates,
            payloads,
            BRANCH_SCORE_RULES["continuous_self_learning"],
        ),
        "sampler_runtime_eligibility": _branch_eligibility(
            "sampler_runtime",
            SAMPLER_TASK_IDS,
            task_rows,
            gates,
            payloads,
            BRANCH_SCORE_RULES["sampler_runtime"],
        ),
        "phase_d_transport_pool_and_headroom_eligibility": _branch_eligibility(
            "phase_d_transport_pool_and_headroom",
            PHASE_D_TASK_IDS,
            task_rows,
            gates,
            payloads,
            BRANCH_SCORE_RULES["phase_d"],
        ),
        "gate_cascade_receipts": gates,
        "gatemate_cached_state_and_new_receipt_count": gatemate,
        "hardware_claim_eligibility": {
            "eligible": False,
            "unauthorized_hardware_claim_counts": hardware_counts,
            "blocking_reasons": ["gatemate_new_dated_physical_receipt_missing"]
            if gatemate["new_receipt_count"] == 0
            else ["authenticated_workload_receipt_missing"],
        },
        "publication_gate_snapshot": _publication_snapshot(command_rows),
        "exclusion_manifest_and_prior_failure_reconciliation": _exclusion_and_prior(
            root, declared, capstone_row, task_rows, command_rows
        ),
        "spec_trace_status_changelog_reconciliation": _spec_reconciliation(root),
        "protected_files_unchanged": _protected_unchanged(protected_before, root),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": test_commands,
        "test_exit_codes": test_exits,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - started),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: V538 exact-path capstone preserved missing="
            f"{counts['missing']}, nonterminal={counts['nonterminal']}, "
            f"blocked={counts['blocked']}, skipped={counts['skipped']}, "
            f"null={counts['null']}, retired={counts['retired']}, flagged={counts['flagged']}; "
            "branch eligibility is separated; conductor receipts, ARC credit, and hardware claims remain zero"
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
    if report.get("status") != "complete":
        errors.append("status")
    if report.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if report.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for zero_field in (
        "conductor_receipt_override_count",
        "arc_solve_claim_count",
        "arc_level_credit_delta",
    ):
        if type(report.get(zero_field)) is not int or report.get(zero_field) != 0:
            errors.append(zero_field)
    registry = report.get("arc_registry_hash_before_after")
    if not isinstance(registry, Mapping) or registry.get("unchanged") is not True:
        errors.append("arc_registry_hash_before_after")
    hardware = report.get("hardware_claim_eligibility")
    counts = (
        hardware.get("unauthorized_hardware_claim_counts")
        if isinstance(hardware, Mapping)
        else None
    )
    if not isinstance(counts, Mapping) or any(
        type(value) is not int or value != 0 for value in counts.values()
    ):
        errors.append("hardware_claim_eligibility")
    protected = report.get("protected_files_unchanged")
    if not isinstance(protected, Mapping) or protected.get("unchanged") is not True:
        errors.append("protected_files_unchanged")
    docs = report.get("spec_trace_status_changelog_reconciliation")
    if (
        not isinstance(docs, Mapping)
        or docs.get("ops_status_changelog_traceability_modified") is not False
    ):
        errors.append("spec_trace_status_changelog_reconciliation")
    provenance = report.get("field_provenance")
    principles = report.get("field_principles")
    if not isinstance(provenance, Mapping):
        errors.append("field_provenance:not_mapping")
    if not isinstance(principles, Mapping):
        errors.append("field_principles:not_mapping")
    if isinstance(provenance, Mapping) and isinstance(principles, Mapping):
        for field in REQUIRED_ARTIFACT_FIELDS:
            row = provenance.get(field)
            if not principles.get(field):
                errors.append(f"field_principles:{field}")
            if not isinstance(row, Mapping) or row.get("principle") != principles.get(field):
                errors.append(f"field_provenance:{field}")
    if not str(report.get("honest_verdict") or "").startswith("complete:"):
        errors.append("honest_verdict")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum")
    return errors


def write_capstone(
    root: Path = REPO_ROOT,
    *,
    date: str,
    verifier_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    command_receipts: Sequence[JsonMap] | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
    env: Mapping[str, str] | None = None,
    validator: Validator = validate_report,
) -> JsonDict:
    report = build_report(
        root,
        date=date,
        verifier_receipts=verifier_receipts,
        command_receipts=command_receipts,
        tests_run=tests_run,
        command_runner=command_runner,
        duration_s=duration_s,
    )
    errors = validator(report)
    if errors:
        raise ValueError(f"invalid Exp6224 capstone: {errors}")
    atomic_write_json(RESULT_RELATIVE_PATH, report, root=root, env=env, sort_keys=False)
    return report


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    report = write_capstone(REPO_ROOT, date=args.date)
    print(
        json.dumps(
            {
                "path": RESULT_RELATIVE_PATH.as_posix(),
                "checksum": report["reproducibility_checksum"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    sys.exit(main())

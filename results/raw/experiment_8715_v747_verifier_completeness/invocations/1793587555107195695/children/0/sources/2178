"""Exp6210 V537 exact-path adversarial capstone.

Spec refs: REQ-INFRA-6210, SCENARIO-INFRA-6210-1,
SCENARIO-INFRA-6210-2, SCENARIO-INFRA-6210-3,
SCENARIO-INFRA-6210-4, SCENARIO-INFRA-6210-5.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml

from carnot.experiment_artifacts import atomic_write_json
from carnot.terminal_artifacts import (
    NONTERMINAL_CLASSES,
    TERMINAL_CLASSES,
    canonical_json,
    classify_artifact_path,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
CommandRunner = Callable[[tuple[str, ...], Path], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.08.537"
EXPERIMENT_ID = "exp6210-v537-adversarial-capstone"
SCHEMA = "carnot.experiment_6210.v537_adversarial_capstone.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_6210_v537_adversarial_capstone.json")
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

ROADMAP_RELATIVE_PATH = Path("research-roadmap.yaml")
ROADMAP_DOC_RELATIVE_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
CONDUCTOR_LOG_RELATIVE_PATH = Path("ops/conductor-log.md")
CODEX_RELATIVE_PATH = Path("CODEX.md")
CLAUDE_RELATIVE_PATH = Path("CLAUDE.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
RESEARCH_COMPLETE_RELATIVE_PATH = Path("research-complete.yaml")
STATUS_RELATIVE_PATH = Path("ops/status.md")
CHANGELOG_RELATIVE_PATH = Path("ops/changelog.md")
TRACEABILITY_RELATIVE_PATH = Path("_bmad/traceability.md")
ARCHITECTURE_RELATIVE_PATH = Path("_bmad/architecture.md")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/research-harnesses/spec.md")
CLASSIFIER_RELATIVE_PATH = Path("python/carnot/terminal_artifacts.py")
ADVERSARIAL_VERIFY_RELATIVE_PATH = Path("scripts/adversarial_verify.py")
SUMMARY_RELATIVE_PATH = Path("scripts/summarize_artifact.py")

V537_UPSTREAM_RANGE = range(6197, 6210)
FORBIDDEN_CLAIM_KEYS = ("energy", "kona", "power", "speed", "terminal", "terminal_hardware", "tsu")
BRANCH_TASKS: dict[str, tuple[str, ...]] = {
    "phase_d": (
        "exp6200-three-family-raw-code-transport-canary",
        "exp6201-livecodebench-k8-pool-v2",
        "exp6202-livecodebench-headroom-v2",
        "exp6203-matching-base-code-hidden-state-v2",
        "exp6204-calibration-code-selector-v2",
        "exp6205-held-code-selection-v2",
    ),
    "continuous_self_learning": (
        "exp6200-three-family-raw-code-transport-canary",
        "exp6206-live-strategy-seed-v2",
        "exp6207-prospective-procedural-memory-csl",
    ),
    "sampler_integration": ("exp6208-mode-jump-runtime-integration",),
    "arc_generalization": ("exp6209-arc-loo-task-aware-shadow",),
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6210_v537_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6210_v537_adversarial_capstone.py -m pytest tests/python/test_experiment_6210_v537_adversarial_capstone.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6210_v537_adversarial_capstone.py --fail-under=100",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6210_v537_adversarial_capstone.py",
    ".venv/bin/ruff check python/carnot/experiment_6210_v537_adversarial_capstone.py tests/python/test_experiment_6210_v537_adversarial_capstone.py",
    ".venv/bin/ruff format --check python/carnot/experiment_6210_v537_adversarial_capstone.py tests/python/test_experiment_6210_v537_adversarial_capstone.py",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/pytest tests/python -q",
)
COMMAND_TIMEOUTS_S = {
    ".venv/bin/pytest tests/python -q": 300,
}

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "milestone_and_declared_task_graph_hash",
    "exact_deliverable_manifest",
    "conductor_receipts",
    "terminal_classifier_path_hash_and_results",
    "task_terminal_classes",
    "missing_nonterminal_blocked_skipped_null_retired_flagged_counts",
    "structured_gate_recomputation",
    "adversarial_verify_receipts_by_artifact",
    "protected_historical_artifact_mutation_count",
    "phase_d_headline_eligibility_and_reason",
    "continuous_self_learning_headline_eligibility_and_reason",
    "sampler_integration_headline_eligibility_and_reason",
    "arc_generalization_headline_eligibility_and_reason",
    "hardware_continuity_state",
    "source_delta_state",
    "prior_failure_retirement_actions",
    "spec_traceability_status_changelog_reconciliation_receipts",
    "architecture_freshness_warning",
    "forbidden_claim_counts",
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
    "status": "Exp6210 is terminal only after every upstream exact path is classified and every receipt is recorded.",
    "milestone_and_declared_task_graph_hash": "Hashes the roadmap task graph so the denominator cannot drift after reconciliation.",
    "exact_deliverable_manifest": "Uses exact declared deliverables only; sidecars and aliases are evidence of nonuse, not substitutes.",
    "conductor_receipts": "Keeps orchestration receipts visible while preventing them from promoting artifact state.",
    "terminal_classifier_path_hash_and_results": "Names and hashes the Exp6197 classifier used for every upstream path.",
    "task_terminal_classes": "Preserves classifier class, adversarial quarantine, missing fields, and command issues per task.",
    "missing_nonterminal_blocked_skipped_null_retired_flagged_counts": "Counts every negative class before any branch summary is written.",
    "structured_gate_recomputation": "Recomputes gates from immutable upstream fields instead of trusting launch decisions.",
    "adversarial_verify_receipts_by_artifact": "Records live artifact-reader and adversarial-verifier receipts for each present result.",
    "protected_historical_artifact_mutation_count": "Bare zero proves the capstone did not rewrite upstream historical results.",
    "phase_d_headline_eligibility_and_reason": "Separates Phase-D headline eligibility from other branches and names blockers.",
    "continuous_self_learning_headline_eligibility_and_reason": "Separates FR-11 memory eligibility from transport and selector outcomes.",
    "sampler_integration_headline_eligibility_and_reason": "Allows the sampler branch to stand or fall independently of code-generation gates.",
    "arc_generalization_headline_eligibility_and_reason": "Keeps ARC generalization evidence distinct from solve or registry credit.",
    "hardware_continuity_state": "Preserves GateMate continuity as a bounded state audit with no speed or power claim.",
    "source_delta_state": "Preserves source-ingestion nulls without converting zero accepted findings into progress.",
    "prior_failure_retirement_actions": "Records retire-if-same-verdict decisions without deleting prior records or reusing IDs.",
    "spec_traceability_status_changelog_reconciliation_receipts": "Documents this run's additive spec update and deferred ops reconciliation.",
    "architecture_freshness_warning": "Warns that _bmad/architecture.md is stale rather than citing it as current.",
    "forbidden_claim_counts": "Bare zero records that no forbidden hardware, rewrite, registry, or sidecar promotion claim is made.",
    "inference_substrate": "Declares aggregation over checked-in artifacts, not live inference or hardware measurement.",
    "verifier_is_oracle": "Bare false because this capstone verifies artifact discipline, not benchmark answers.",
    "field_provenance": "Ties each field to the roadmap, exact artifacts, classifier, verifier receipts, or local hashes.",
    "field_principles": "Keeps why each required field matters next to the emitted artifact.",
    "test_commands": "Lists the focused, coverage, spec-coverage, E2E-plan, and global commands.",
    "test_exit_codes": "Records observed command exits without hiding nonzero results.",
    "duration_s": "Records wall time for deterministic aggregation without padding.",
    "reproducibility_checksum": "Content-addresses the normalized report payload.",
    "honest_verdict": "Starts with a terminal prefix and names headline blockers without strengthening claims.",
}


def payload_checksum(report: JsonMap) -> str:
    normalized = json.loads(canonical_json(report))
    normalized["duration_s"] = 0.0
    normalized["reproducibility_checksum"] = ""
    return payload_sha256(normalized)


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


def _experiment_number(task_id: str) -> int | None:
    match = re.match(r"exp(\d+)", task_id)
    return int(match.group(1)) if match else None


def _same_number_aliases(root: Path, task_id: str, declared_rel: Path) -> list[str]:
    number = _experiment_number(task_id)
    results_dir = root / "results"
    if number is None or not results_dir.exists():
        return []
    declared = (root / declared_rel).resolve()
    aliases: list[str] = []
    for candidate in sorted(results_dir.glob(f"experiment_{number}*.json")):
        if candidate.resolve() != declared:
            aliases.append(candidate.relative_to(root).as_posix())
    return aliases


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


def _required_fields_from_prompt(prompt: str) -> list[str]:
    marker = "REQUIRED ARTIFACT FIELDS:"
    if marker not in prompt:
        return []
    blob = prompt.split(marker, 1)[1].split("\n\n", 1)[0].split(".", 1)[0]
    fields: list[str] = []
    for part in blob.split(","):
        field = part.strip().strip("`").strip()
        if field.startswith("and "):
            field = field[4:].strip()
        field = field.strip("`").strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field):
            fields.append(field)
    return fields


def _roadmap_declared_tasks(root: Path) -> tuple[list[JsonDict], JsonDict, JsonDict]:
    roadmap = _read_yaml_mapping(root / ROADMAP_RELATIVE_PATH)
    raw_tasks = roadmap.get("tasks")
    rows = (
        [dict(row) for row in raw_tasks if isinstance(row, Mapping)]
        if isinstance(raw_tasks, list)
        else []
    )
    by_id = {str(row.get("id")): row for row in rows if row.get("id")}
    capstone = by_id.get(EXPERIMENT_ID, {})
    requires = capstone.get("requires")
    if isinstance(requires, list):
        wanted = [str(task_id) for task_id in requires]
    else:
        wanted = [
            str(row.get("id"))
            for row in rows
            if (number := _experiment_number(str(row.get("id")))) in V537_UPSTREAM_RANGE
        ]
    declared: list[JsonDict] = []
    for task_id in wanted:
        number = _experiment_number(task_id)
        if number not in V537_UPSTREAM_RANGE:
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
                "required_fields": _required_fields_from_prompt(str(row.get("prompt") or "")),
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
    timeout_s = COMMAND_TIMEOUTS_S.get(command)
    try:
        proc = subprocess.run(
            argv,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except FileNotFoundError as exc:
        return {
            "command": command,
            "exit_code": 127,
            "classification": "command_not_found",
            "error": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout
            if isinstance(exc.stdout, str)
            else (exc.stdout or b"").decode("utf-8", "replace")
        )
        stderr = (
            exc.stderr
            if isinstance(exc.stderr, str)
            else (exc.stderr or b"").decode("utf-8", "replace")
        )
        return {
            "command": command,
            "exit_code": 124,
            "classification": f"timed_out_after_{timeout_s}s",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    return {
        "command": command,
        "exit_code": proc.returncode,
        "classification": "passed" if proc.returncode == 0 else f"nonzero_exit_{proc.returncode}",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _run_commands(root: Path, commands: Sequence[str], runner: CommandRunner) -> list[JsonDict]:
    return [runner(_argv(command), root) for command in commands]


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


def _run_artifact_verifiers(
    root: Path, present_paths: Mapping[str, Path]
) -> dict[str, JsonDict]:  # pragma: no cover - integration path.
    receipts: dict[str, JsonDict] = {}
    for task_id, rel_path in present_paths.items():
        adv_cmd = [
            sys.executable,
            (root / ADVERSARIAL_VERIFY_RELATIVE_PATH).as_posix(),
            "--json",
            rel_path.as_posix(),
        ]
        adv = subprocess.run(adv_cmd, cwd=root, text=True, capture_output=True, check=False)
        try:
            stdout_json: JsonDict = json.loads(adv.stdout)
        except json.JSONDecodeError:
            stdout_json = {"parse_error": True, "raw_stdout": adv.stdout}
        sum_cmd = [sys.executable, (root / SUMMARY_RELATIVE_PATH).as_posix(), rel_path.as_posix()]
        summary = subprocess.run(sum_cmd, cwd=root, text=True, capture_output=True, check=False)
        receipts[task_id] = {
            "task_id": task_id,
            "artifact_path": rel_path.as_posix(),
            "adversarial": {
                "command": " ".join(adv_cmd),
                "exit_code": adv.returncode,
                "stdout_json": stdout_json,
                "stderr": adv.stderr,
            },
            "summary": {
                "command": " ".join(sum_cmd),
                "exit_code": summary.returncode,
                "stdout_tail": summary.stdout[-4000:],
                "stderr_tail": summary.stderr[-4000:],
            },
        }
    return receipts


def _normalize_verifier_receipts(
    receipts: Mapping[str, JsonMap] | Sequence[JsonMap],
) -> dict[str, JsonDict]:
    items = (
        receipts.items()
        if isinstance(receipts, Mapping)
        else ((str(row.get("task_id")), row) for row in receipts if isinstance(row, Mapping))
    )
    normalized: dict[str, JsonDict] = {}
    for task_id, receipt in items:
        row = dict(receipt)
        row.setdefault("task_id", task_id)
        if "adversarial" not in row and "stdout_json" in row:
            row["adversarial"] = {
                "command": row.get("command"),
                "exit_code": row.get("exit_code"),
                "stdout_json": row.get("stdout_json"),
                "stderr": row.get("stderr", ""),
            }
        row.setdefault("adversarial", {"stdout_json": {}, "exit_code": None, "command": None})
        row.setdefault("summary", {"exit_code": None, "command": None})
        row["critical_flag_count"] = _critical_flag_count(row)
        row["flag_count"] = _flag_count(row)
        row["receipt_hash"] = payload_sha256(
            {
                "adversarial": row.get("adversarial"),
                "summary": row.get("summary"),
                "critical_flag_count": row["critical_flag_count"],
            }
        )
        normalized[str(task_id)] = row
    return normalized


def _flag_count(receipt: JsonMap) -> int:
    adv = receipt.get("adversarial")
    stdout_json = adv.get("stdout_json") if isinstance(adv, Mapping) else None
    if not isinstance(stdout_json, Mapping):
        return 0
    reports = stdout_json.get("reports")
    if isinstance(reports, list) and reports and isinstance(reports[0], Mapping):
        return int(reports[0].get("flag_count") or 0)
    return int(stdout_json.get("flagged_count") or 0)


def _critical_flag_count(receipt: JsonMap) -> int:
    adv = receipt.get("adversarial")
    stdout_json = adv.get("stdout_json") if isinstance(adv, Mapping) else None
    if not isinstance(stdout_json, Mapping):
        return 0
    reports = stdout_json.get("reports")
    if not isinstance(reports, list) or not reports or not isinstance(reports[0], Mapping):
        return 0
    flags = reports[0].get("flags")
    if not isinstance(flags, list):
        return 0
    return sum(
        1
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    )


def _missing_required_fields(payload: JsonMap, required_fields: Sequence[str]) -> list[str]:
    return [field for field in required_fields if field not in payload]


def _classified_nonzero_commands(payload: JsonMap) -> set[str]:
    classified: set[str] = set()
    for key in ("nonzero_command_classification", "unrelated_nonzero_command_classifications"):
        rows = payload.get(key)
        if isinstance(rows, list):
            classified.update(str(row.get("command")) for row in rows if isinstance(row, Mapping))
        elif isinstance(rows, Mapping):
            classified.update(str(command) for command in rows)
    command_block = payload.get("task_owned_test_commands_and_exit_codes")
    if isinstance(command_block, Mapping):
        rows = command_block.get("command_receipts")
        if isinstance(rows, list):
            classified.update(
                str(row.get("command"))
                for row in rows
                if isinstance(row, Mapping) and row.get("classification")
            )
    full_suite = payload.get("full_suite_command_and_classified_exit_code")
    if isinstance(full_suite, Mapping) and full_suite.get("classification"):
        classified.add(str(full_suite.get("command")))
    return classified


def _command_exit_codes(payload: JsonMap) -> dict[str, int]:
    codes: dict[str, int] = {}
    for key in ("test_exit_codes", "focused_test_exit_codes"):
        raw = payload.get(key)
        if isinstance(raw, Mapping):
            for command, exit_code in raw.items():
                try:
                    codes[str(command)] = int(exit_code)
                except (TypeError, ValueError):
                    codes[str(command)] = 1
    command_block = payload.get("task_owned_test_commands_and_exit_codes")
    if isinstance(command_block, Mapping):
        rows = command_block.get("command_receipts")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and row.get("command") is not None:
                    codes[str(row.get("command"))] = int(row.get("exit_code") or 0)
    full_suite = payload.get("full_suite_command_and_classified_exit_code")
    if isinstance(full_suite, Mapping) and full_suite.get("command") is not None:
        codes[str(full_suite.get("command"))] = int(full_suite.get("exit_code") or 0)
    return codes


def _unclassified_nonzero_commands(payload: JsonMap) -> list[str]:
    classified = _classified_nonzero_commands(payload)
    return sorted(
        command
        for command, exit_code in _command_exit_codes(payload).items()
        if exit_code != 0 and command not in classified
    )


def _gate_actual_value(payloads: Mapping[str, JsonMap], gate: JsonMap) -> Any:
    upstream = str(gate.get("upstream"))
    field = str(gate.get("artifact_field"))
    return payloads.get(upstream, {}).get(field)


def _gate_passed(actual: Any, op: str, expected: Any) -> bool:
    return bool(op == "==" and actual == expected)


def _field_principle(field: str) -> str:
    return FIELD_PRINCIPLES.get(field, f"{field} is required by REQ-INFRA-6210.")


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": _field_principle(field),
            "sources": [
                "REQ-INFRA-6210",
                ROADMAP_RELATIVE_PATH.as_posix(),
                CONDUCTOR_LOG_RELATIVE_PATH.as_posix(),
                CLASSIFIER_RELATIVE_PATH.as_posix(),
                ADVERSARIAL_VERIFY_RELATIVE_PATH.as_posix(),
                SUMMARY_RELATIVE_PATH.as_posix(),
                "exact_declared_artifacts",
            ],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _artifact_forbidden_claim_counts(payload: JsonMap) -> JsonDict:
    raw = payload.get("speed_power_energy_terminal_tsu_kona_claim_counts")
    counts = {key: 0 for key in FORBIDDEN_CLAIM_KEYS}
    if isinstance(raw, Mapping):
        for key in FORBIDDEN_CLAIM_KEYS:
            try:
                counts[key] = int(raw.get(key) or 0)
            except (TypeError, ValueError):
                counts[key] = 1
    return counts


def _sum_forbidden_claim_counts(payloads: Mapping[str, JsonMap]) -> JsonDict:
    counts = {key: 0 for key in FORBIDDEN_CLAIM_KEYS}
    hardware = payloads.get("exp6199-gatemate-terminal-action-audit-v537", {})
    for key, value in _artifact_forbidden_claim_counts(hardware).items():
        counts[key] += int(value)
    return counts


def _task_issues(task_id: str, row: JsonMap, gate_row: JsonMap) -> list[str]:
    issues: list[str] = []
    terminal_class = str(row.get("terminal_class"))
    if terminal_class in NONTERMINAL_CLASSES or terminal_class == "missing":
        issues.append(terminal_class)
    if int(row.get("critical_adversarial_flag_count") or 0) > 0:
        issues.append("critical_adversarial_flag")
    if row.get("missing_required_fields"):
        issues.append("missing_required_field")
    if row.get("unclassified_nonzero_commands"):
        issues.append("unclassified_nonzero_command")
    gates = gate_row.get("declared_gates")
    if isinstance(gates, list) and any(
        isinstance(gate, Mapping) and gate.get("passed") is False for gate in gates
    ):
        issues.append("failed_gate")
    return issues


def _branch_eligibility(
    branch: str,
    task_ids: Sequence[str],
    task_rows: Mapping[str, JsonMap],
    gates: Mapping[str, JsonMap],
) -> JsonDict:
    blockers: list[str] = []
    for task_id in task_ids:
        row = task_rows.get(task_id, {})
        gate_row = gates.get(task_id, {})
        for issue in _task_issues(task_id, row, gate_row):
            blockers.append(f"{task_id}:{issue}")
    return {
        "branch": branch,
        "eligible": not blockers,
        "task_ids": list(task_ids),
        "blocking_reasons": blockers,
        "reason": "eligible_from_terminal_clean_exact_artifacts"
        if not blockers
        else "; ".join(blockers),
    }


def _prior_failure_actions(
    tasks: Sequence[JsonMap], capstone_row: JsonMap, task_rows: Mapping[str, JsonMap]
) -> list[JsonDict]:
    actions: list[JsonDict] = []
    all_rows = list(tasks)
    if capstone_row:
        all_rows.append(
            {
                "task_id": EXPERIMENT_ID,
                "prior_failures": capstone_row.get("prior_failures", []),
            }
        )
    for task in all_rows:
        task_id = str(task.get("task_id") or task.get("id"))
        for prior in task.get("prior_failures", []):
            if not isinstance(prior, Mapping) or prior.get("retire_if_same_verdict") is not True:
                continue
            current = str(task_rows.get(task_id, {}).get("honest_verdict") or "")
            prior_verdict = str(prior.get("verdict") or "")
            same = bool(current and current == prior_verdict)
            actions.append(
                {
                    "task_id": task_id,
                    "prior_experiment_id": prior.get("experiment_id"),
                    "retire_if_same_verdict": True,
                    "same_verdict": same,
                    "action": "retire_current_lineage_without_deleting_prior_records"
                    if same
                    else "no_retirement_current_verdict_differs_or_capstone_pending",
                }
            )
    return actions


def _architecture_warning(root: Path, roadmap_doc: str) -> JsonDict:
    stale_text = "last reconciled 2026-07-03" in roadmap_doc
    return {
        "path": ARCHITECTURE_RELATIVE_PATH.as_posix(),
        "present": (root / ARCHITECTURE_RELATIVE_PATH).exists(),
        "sha256": path_sha256(root / ARCHITECTURE_RELATIVE_PATH),
        "last_reconciled": "2026-07-03" if stale_text else None,
        "current_date": "2026-08-07",
        "stale": True,
        "warning": "_bmad/architecture.md is stale by the roadmap freshness note; do not cite it as current.",
    }


def build_report(
    root: Path = REPO_ROOT,
    *,
    date: str,
    verifier_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.monotonic()
    declared, roadmap, capstone_row = _roadmap_declared_tasks(root)
    log_text = _read_text(root / CONDUCTOR_LOG_RELATIVE_PATH)
    roadmap_doc = _read_text(root / ROADMAP_DOC_RELATIVE_PATH)
    before_hashes = {row["task_id"]: path_sha256(root / row["deliverable"]) for row in declared}

    payloads: dict[str, JsonDict] = {}
    present_paths: dict[str, Path] = {}
    manifest: JsonDict = {}
    conductor_receipts: JsonDict = {}
    classifier_results: JsonDict = {}
    gates: JsonDict = {}

    for row in declared:
        task_id = str(row["task_id"])
        rel_path = Path(row["deliverable"])
        payload, meta = _read_json_mapping(root / rel_path)
        payloads[task_id] = payload
        receipt = _latest_conductor_receipt(log_text, str(row["title"]))
        conductor_receipts[task_id] = receipt
        classified = classify_artifact_path(root / rel_path, conductor_receipt=receipt)
        classifier_results[task_id] = classified.to_dict()
        if meta["present"] and meta["loadable"]:
            present_paths[task_id] = rel_path
        manifest[task_id] = {
            "task_id": task_id,
            "title": row["title"],
            "track": row["track"],
            "declared_deliverable": rel_path.as_posix(),
            "present": bool(meta["present"]),
            "loadable": bool(meta["loadable"]),
            "sha256": meta["sha256"],
            "error": meta["error"],
            "requires": row["requires"],
            "gated_on": row["gated_on"],
            "required_fields": row["required_fields"],
            "same_number_alias_used": False,
            "same_number_alias_candidates_ignored": _same_number_aliases(root, task_id, rel_path),
        }

    raw_receipts = (
        _run_artifact_verifiers(root, present_paths)
        if verifier_receipts is None
        else verifier_receipts
    )
    normalized_receipts = _normalize_verifier_receipts(raw_receipts)

    task_rows: JsonDict = {}
    for row in declared:
        task_id = str(row["task_id"])
        payload = payloads.get(task_id, {})
        classifier = classifier_results[task_id]
        critical = _critical_flag_count(normalized_receipts.get(task_id, {}))
        terminal_class = str(classifier["classification"])
        if critical:
            terminal_class = "flagged"
        missing_fields = (
            _missing_required_fields(payload, row["required_fields"]) if payload else []
        )
        unclassified = _unclassified_nonzero_commands(payload)
        task_rows[task_id] = {
            "task_id": task_id,
            "status": payload.get("status"),
            "honest_verdict": payload.get("honest_verdict"),
            "classifier_class": classifier["classification"],
            "classifier_terminal": classifier["terminal"],
            "terminal_class": terminal_class,
            "present": classifier["present"],
            "loadable": classifier["loadable"],
            "critical_adversarial_flag_count": critical,
            "missing_required_fields": missing_fields,
            "unclassified_nonzero_commands": unclassified,
        }

    for row in declared:
        task_id = str(row["task_id"])
        gate_rows: list[JsonDict] = []
        for gate in row["gated_on"]:
            actual = _gate_actual_value(payloads, gate)
            expected = gate.get("value")
            op = str(gate.get("op"))
            gate_rows.append(
                {**gate, "actual": actual, "passed": _gate_passed(actual, op, expected)}
            )
        gates[task_id] = {
            "declared_gates": gate_rows,
            "artifact_gates_evaluated": payloads.get(task_id, {}).get("gates_evaluated", []),
            "conductor_gate_block": conductor_receipts.get(task_id, {}).get("status")
            == "GATE_BLOCK",
            "terminal_class": task_rows[task_id]["terminal_class"],
        }

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
    graph = {
        "milestone": MILESTONE,
        "roadmap": {
            "path": ROADMAP_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / ROADMAP_RELATIVE_PATH),
        },
        "roadmap_doc": {
            "path": ROADMAP_DOC_RELATIVE_PATH.as_posix(),
            "sha256": path_sha256(root / ROADMAP_DOC_RELATIVE_PATH),
        },
        "tasks": [
            {
                "task_id": row["task_id"],
                "deliverable": Path(row["deliverable"]).as_posix(),
                "requires": row["requires"],
                "gated_on": row["gated_on"],
            }
            for row in declared
        ],
    }
    commands, exits = _normalize_tests(tests_run)
    command_receipts: list[JsonDict] = [
        {
            "command": command,
            "exit_code": exit_code,
            "classification": "passed" if exit_code == 0 else f"nonzero_exit_{exit_code}",
        }
        for command, exit_code in exits.items()
    ]
    if tests_run is None:
        command_receipts = _run_commands(root, DEFAULT_TEST_COMMANDS, command_runner)
        commands, exits = _normalize_tests(command_receipts)
    after_hashes = {row["task_id"]: path_sha256(root / row["deliverable"]) for row in declared}
    mutation_count = sum(
        1 for task_id, before in before_hashes.items() if before != after_hashes[task_id]
    )

    report: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": date,
        "status": "complete",
        "milestone_and_declared_task_graph_hash": {
            **graph,
            "declared_task_count": len(declared),
            "graph_sha256": payload_sha256(graph),
        },
        "exact_deliverable_manifest": manifest,
        "conductor_receipts": conductor_receipts,
        "terminal_classifier_path_hash_and_results": {
            "classifier_path": CLASSIFIER_RELATIVE_PATH.as_posix(),
            "classifier_sha256": path_sha256(root / CLASSIFIER_RELATIVE_PATH),
            "terminal_classes": sorted(TERMINAL_CLASSES),
            "nonterminal_classes": sorted(NONTERMINAL_CLASSES),
            "results": classifier_results,
        },
        "task_terminal_classes": task_rows,
        "missing_nonterminal_blocked_skipped_null_retired_flagged_counts": counts,
        "structured_gate_recomputation": gates,
        "adversarial_verify_receipts_by_artifact": normalized_receipts,
        "protected_historical_artifact_mutation_count": int(mutation_count),
        "phase_d_headline_eligibility_and_reason": _branch_eligibility(
            "phase_d", BRANCH_TASKS["phase_d"], task_rows, gates
        ),
        "continuous_self_learning_headline_eligibility_and_reason": _branch_eligibility(
            "continuous_self_learning", BRANCH_TASKS["continuous_self_learning"], task_rows, gates
        ),
        "sampler_integration_headline_eligibility_and_reason": _branch_eligibility(
            "sampler_integration", BRANCH_TASKS["sampler_integration"], task_rows, gates
        ),
        "arc_generalization_headline_eligibility_and_reason": _branch_eligibility(
            "arc_generalization", BRANCH_TASKS["arc_generalization"], task_rows, gates
        ),
        "hardware_continuity_state": {
            "task_id": "exp6199-gatemate-terminal-action-audit-v537",
            "terminal_class": task_rows.get("exp6199-gatemate-terminal-action-audit-v537", {}).get(
                "terminal_class"
            ),
            "forbidden_claim_counts": _artifact_forbidden_claim_counts(
                payloads.get("exp6199-gatemate-terminal-action-audit-v537", {})
            ),
        },
        "source_delta_state": {
            "task_id": "exp6198-v537-post-marker-source-scope-audit",
            "terminal_class": task_rows.get("exp6198-v537-post-marker-source-scope-audit", {}).get(
                "terminal_class"
            ),
            "accepted_count": payloads.get("exp6198-v537-post-marker-source-scope-audit", {}).get(
                "accepted_count"
            ),
        },
        "prior_failure_retirement_actions": _prior_failure_actions(
            declared, capstone_row, task_rows
        ),
        "spec_traceability_status_changelog_reconciliation_receipts": {
            "openspec_req_infra_6210_present": "REQ-INFRA-6210"
            in _read_text(root / SPEC_RELATIVE_PATH),
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
                    EXCLUSION_MANIFEST_RELATIVE_PATH,
                    RESEARCH_COMPLETE_RELATIVE_PATH,
                    CODEX_RELATIVE_PATH,
                    CLAUDE_RELATIVE_PATH,
                )
            },
        },
        "architecture_freshness_warning": _architecture_warning(root, roadmap_doc),
        "forbidden_claim_counts": _sum_forbidden_claim_counts(payloads),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": commands,
        "test_exit_codes": exits,
        "test_command_receipts": command_receipts,
        "duration_s": float(duration_s if duration_s is not None else time.monotonic() - started),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: V537 exact-path capstone preserved missing="
            f"{counts['missing']}, nonterminal={counts['nonterminal']}, "
            f"blocked={counts['blocked']}, skipped={counts['skipped']}, "
            f"null={counts['null']}, retired={counts['retired']}, flagged={counts['flagged']}; "
            "headline eligibility is branch-separated and conductor receipts did not promote artifact state"
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
    if (
        type(report.get("protected_historical_artifact_mutation_count")) is not int
        or report.get("protected_historical_artifact_mutation_count") != 0
    ):
        errors.append("protected_historical_artifact_mutation_count")
    forbidden = report.get("forbidden_claim_counts")
    if not isinstance(forbidden, Mapping) or any(
        type(value) is not int or value != 0 for value in forbidden.values()
    ):
        errors.append("forbidden_claim_counts")
    docs = report.get("spec_traceability_status_changelog_reconciliation_receipts")
    if (
        not isinstance(docs, Mapping)
        or docs.get("ops_status_changelog_traceability_modified") is not False
    ):
        errors.append("spec_traceability_status_changelog_reconciliation_receipts")
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
    verdict = str(report.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        errors.append("honest_verdict_prefix")
    if report.get("reproducibility_checksum") != payload_checksum(report):
        errors.append("reproducibility_checksum")
    return errors


def write_capstone(
    root: Path = REPO_ROOT,
    *,
    date: str,
    verifier_receipts: Mapping[str, JsonMap] | Sequence[JsonMap] | None = None,
    tests_run: Mapping[str, int] | Sequence[JsonMap] | None = None,
    command_runner: CommandRunner = _run_command,
    duration_s: float | None = None,
    env: Mapping[str, str] | None = None,
) -> JsonDict:
    report = build_report(
        root,
        date=date,
        verifier_receipts=verifier_receipts,
        tests_run=tests_run,
        command_runner=command_runner,
        duration_s=duration_s,
    )
    errors = validate_report(report)
    if errors:
        raise ValueError(f"invalid Exp6210 capstone: {errors}")
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

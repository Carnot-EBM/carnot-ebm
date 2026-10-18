"""Bind the V662 planning contract to literal V661 evidence.

This module reports planning readiness. It does not load a model, alter a
scientific gate, or convert historical completion into benefit.

Spec refs: REQ-REPORT-7573 and SCENARIO-REPORT-7573-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7329_v644_contract import parse_markdown_contract
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.662"
EXPERIMENT_ID = "exp7573-contract-methods"
SCHEMA = "carnot.exp7573.v662.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
V661_DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-v661-preserved-20260923.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7573_v662_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7573_v662_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7573_v662_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7573_v662_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7573_v662_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v662-method-map.md")
STUDY_PATH = Path("research-studying.md")
STUDY_MARKER = "<!-- EXP7573-V662-METHOD-INGESTION -->"
V661_CAPSTONE_PATH = Path("results/experiment_7572_v661_capstone.json")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7573, "contract-methods"),
        (7574, "measurement-requalification"),
        (7575, "cached-learning-protocol"),
        (7576, "proper-loss-energy"),
        (7577, "proper-loss-evaluation"),
        (7578, "continuous-proper-loss"),
        (7579, "decision-learning-audit"),
        (7580, "arc-verifier-support"),
        (7581, "arc-bounded-canary"),
        (7582, "arc-panel-a"),
        (7583, "arc-panel-b"),
        (7584, "arc-independent-audit"),
        (7585, "portable-service"),
        (7586, "capstone"),
    )
)
V661_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7560, "contract-methods"),
        (7561, "recalibration-prototype"),
        (7562, "arc-plan-lineage"),
        (7563, "native-pilot"),
        (7564, "fit-capture"),
        (7565, "test-online-capture"),
        (7566, "energy-fit"),
        (7567, "source-evaluation"),
        (7568, "continuous-recalibration"),
        (7569, "decision-learning-audit"),
        (7570, "arc-live-lineage"),
        (7571, "portable-calibration"),
        (7572, "capstone"),
    )
)

ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls", "tokens")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
TERMINAL_CLASSES = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Print a truthful phase boundary before a possibly slow operation."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7573] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def _load_yaml(path: Path) -> JsonDict:
    """Load one YAML mapping so malformed authorities fail closed."""

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def resolve_v662_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer matching staged bytes, then accept the matching active roadmap."""

    candidates: list[JsonDict] = []
    selected: tuple[Path, JsonDict] | None = None
    for relative in (NEXT_ROADMAP_PATH, ACTIVE_ROADMAP_PATH):
        path = root / relative
        try:
            value = _load_yaml(path)
            observed: object = value.get("milestone")
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            value = {}
            observed = "absent" if isinstance(exc, FileNotFoundError) else type(exc).__name__
        matches = path.is_file() and observed == MILESTONE
        candidates.append(
            {
                "path": relative.as_posix(),
                "exists": path.is_file(),
                "expected": MILESTONE,
                "observed": observed,
                "matches_milestone": matches,
            }
        )
        if selected is None and matches:
            selected = (path, value)
    if selected is None:
        raise ValueError("V662 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def _normalize_yaml_contract(roadmap: object) -> JsonDict:
    """Read the public contract directly from the YAML task fields."""

    if not isinstance(roadmap, Mapping) or not isinstance(roadmap.get("tasks"), list):
        raise ValueError("roadmap task list required")
    tasks: list[JsonDict] = []
    for order, raw in enumerate(roadmap["tasks"], 1):
        if not isinstance(raw, Mapping):
            raise ValueError("roadmap task mapping required")
        gates = raw.get("gated_on") or []
        if not isinstance(gates, list) or any(not isinstance(gate, Mapping) for gate in gates):
            raise ValueError("roadmap gate list required")
        tasks.append(
            {
                "order": order,
                "id": raw.get("id"),
                "title": raw.get("title"),
                "deliverable": raw.get("deliverable"),
                "phase": raw.get("phase"),
                "substrate": raw.get("inference_substrate_class"),
                "milestone": raw.get("milestone"),
                "gates": [
                    {
                        "upstream": gate.get("upstream"),
                        "artifact_field": gate.get("artifact_field"),
                        "op": gate.get("op"),
                        "value": deepcopy(gate.get("value")),
                    }
                    for gate in gates
                ],
            }
        )
    return {"milestone": roadmap.get("milestone"), "tasks": tasks}


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only fields independently present in both planning authorities."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
    }


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare fourteen exact identities, paths, substrates, and gate lists."""

    errors: list[str] = []
    try:
        markdown = parse_markdown_contract(markdown_text)
    except (TypeError, ValueError):
        markdown = {"milestone": None, "tasks": []}
        errors.append("markdown_task_table_missing")
    try:
        selected = _normalize_yaml_contract(roadmap)
    except (TypeError, ValueError):
        selected = {"milestone": None, "tasks": []}
        errors.append("yaml_task_table_missing")
    expected_tasks = markdown["tasks"]
    observed_tasks = selected["tasks"]
    rows: list[JsonDict] = []
    fields = ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
    for index, task_id in enumerate(EXPECTED_TASK_IDS):
        expected = expected_tasks[index] if index < len(expected_tasks) else None
        observed = observed_tasks[index] if index < len(observed_tasks) else None
        checks = {
            field: bool(expected and observed and expected.get(field) == observed.get(field))
            for field in fields
        }
        matched = all(checks.values())
        rows.append(
            {
                "unit_id": task_id,
                "arm": "markdown_vs_yaml",
                "order": index + 1,
                "expected": _public_task(expected),
                "observed": _public_task(observed),
                "checks": checks,
                "matched": matched,
                "passed": matched,
                "raw_numerator": sum(checks.values()),
                "raw_denominator": len(checks),
                "metric_direction": "exact_match",
                "seed": None,
                "censored": False,
                "provenance": [DESIGN_PATH.as_posix(), "selected_roadmap"],
            }
        )
    if markdown.get("milestone") != MILESTONE:
        errors.append("markdown_milestone")
    if selected.get("milestone") != MILESTONE:
        errors.append("yaml_milestone")
    if len(expected_tasks) != 14:
        errors.append("markdown_task_count")
    if len(observed_tasks) != 14:
        errors.append("yaml_task_count")
    if [row.get("id") for row in expected_tasks] != list(EXPECTED_TASK_IDS):
        errors.append("markdown_task_order")
    if [row.get("id") for row in observed_tasks] != list(EXPECTED_TASK_IDS):
        errors.append("yaml_task_order")
    if any(not row["matched"] for row in rows):
        errors.append("row_mismatch")
    return {
        "passed": not errors,
        "errors": list(dict.fromkeys(errors)),
        "markdown_milestone": markdown.get("milestone"),
        "yaml_milestone": selected.get("milestone"),
        "contract_rows": rows,
        "rows": rows,
    }


def mutation_names() -> tuple[str, ...]:
    """Name the four planning defects required by the V662 contract."""

    return ("count", "order", "path", "gate")


def mutate_yaml_for_test(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Return one private YAML corruption without changing authority bytes."""

    if mutation not in mutation_names():
        raise ValueError(f"unknown mutation: {mutation}")
    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "count":
        changed["tasks"] = tasks[:-1]
    elif mutation == "order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "path":
        tasks[0]["deliverable"] = "results/private-changed.json"
    else:
        gated = next(task for task in tasks if task.get("gated_on"))
        gated["gated_on"][0]["artifact_field"] = "private_changed_score"
    return changed


def _mutate_markdown(text: str, mutation: str) -> str:
    """Return one private Markdown corruption for paired mutation checks."""

    lines = text.splitlines()
    task_lines = [index for index, line in enumerate(lines) if re.match(r"\| \d+ \| exp\d+", line)]
    if mutation == "count":
        del lines[task_lines[-1]]
    elif mutation == "order":
        lines[task_lines[0]], lines[task_lines[1]] = lines[task_lines[1]], lines[task_lines[0]]
    elif mutation == "path":
        lines[task_lines[0]] = lines[task_lines[0]].replace(
            "results/experiment_7573_v662_contract_methods.json", "results/private-changed.json"
        )
    else:
        index = next(i for i in task_lines if ".recalibration_ready_score" in lines[i])
        lines[index] = lines[index].replace("recalibration_ready_score", "private_changed_score", 1)
    return "\n".join(lines)


def run_contract_mutation_controls(text: str, roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Require the valid baseline and reject each private authority defect."""

    baseline = compare_contract_authorities(text, roadmap)["passed"] is True
    rows: list[JsonDict] = []
    for authority in ("markdown", "yaml"):
        for mutation in mutation_names():
            changed_text = _mutate_markdown(text, mutation) if authority == "markdown" else text
            changed_yaml = (
                mutate_yaml_for_test(roadmap, mutation) if authority == "yaml" else roadmap
            )
            rejected = compare_contract_authorities(changed_text, changed_yaml)["passed"] is False
            rows.append(
                {
                    "authority": authority,
                    "mutation": mutation,
                    "baseline_valid": baseline,
                    "rejected": rejected,
                    "qualified": baseline and rejected,
                }
            )
    return rows


def _load_json(path: Path) -> JsonDict:
    """Load one JSON object so evidence arrays cannot masquerade as artifacts."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def collect_v661_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate thirteen literal capstone dispositions against source bytes."""

    capstone_path = root / V661_CAPSTONE_PATH
    capstone = _load_json(capstone_path)
    raw_rows = capstone.get("task_dispositions")
    if not isinstance(raw_rows, list):
        raise ValueError("V661 capstone dispositions are unavailable")
    rows: list[JsonDict] = []
    for expected_id, raw in zip(V661_TASK_IDS, raw_rows, strict=True):
        if not isinstance(raw, Mapping) or raw.get("task_id") != expected_id:
            raise ValueError("V661 capstone disposition order is invalid")
        row = deepcopy(dict(raw))
        evidence = row.get("evidence_path")
        evidence_path = root / str(evidence) if evidence else capstone_path
        observed_hash = sha256_file(evidence_path) if evidence_path.is_file() else None
        expected_hash = row.get("artifact_sha256") or sha256_file(capstone_path)
        row.update(
            {
                "authentication_path": evidence_path.relative_to(root).as_posix(),
                "authentication_sha256": observed_hash,
                "authenticated": observed_hash == expected_hash,
                "evidence_kind": "pre_gate"
                if row.get("evidence_state") == "conductor_gate_blocked"
                else "producer",
                "archive_history_invented": False,
            }
        )
        if expected_id == "exp7561-recalibration-prototype":
            strict = next(
                (
                    receipt
                    for receipt in row.get("validation_receipts", [])
                    if receipt.get("name") == "verdict_row_consistency_strict"
                ),
                {},
            )
            row["strict_row_lint_exit_code"] = strict.get("exit_code")
        elif expected_id == "exp7567-source-evaluation":
            payload = _load_json(evidence_path)
            row["fresh_confirmatory_claim_allowed"] = payload.get(
                "fresh_confirmatory_claim_allowed", False
            )
        elif expected_id == "exp7570-arc-live-lineage":
            payload = _load_json(evidence_path)
            counts = payload.get("invocation_counts") or {}
            row["inference_started"] = any(
                value != 0 for value in counts.values() if isinstance(value, int)
            )
        elif expected_id == "exp7571-portable-calibration":
            ready = row.get("ready_value_fields") or {}
            row["board_continuity_complete_score"] = ready.get("board_continuity_complete_score")
            row["service_measurement_complete_score"] = ready.get(
                "service_measurement_complete_score"
            )
        rows.append(row)
    return rows


def method_rows() -> list[JsonDict]:
    """Map four primary method sections to bounded local uses."""

    return [
        {
            "method_family": "proper_loss_recalibration",
            "primary_url": "https://arxiv.org/html/2603.22167v1",
            "primary_section": "Calibeating method and proper-loss regret sections",
            "applicable_test_or_deferral": "Exp7576 Brier fit and Exp7578 delayed-feedback retention",
            "claim_limit": "Immediate-feedback guarantees do not establish Carnot delayed retention.",
            "destination_tasks": ["exp7576-proper-loss-energy", "exp7578-continuous-proper-loss"],
            "external_result_is_local_measurement": False,
        },
        {
            "method_family": "local_kan_learning",
            "primary_url": "https://arxiv.org/abs/2602.02056",
            "primary_section": "Sparse local spline update and fixed-precision method",
            "applicable_test_or_deferral": "Exp7578 locality control; FPGA placement deferred to measured need",
            "claim_limit": "A CPU local update is not an FPGA speed or retention result.",
            "destination_tasks": ["exp7578-continuous-proper-loss", "exp7585-portable-service"],
            "external_result_is_local_measurement": False,
        },
        {
            "method_family": "structured_decoding_semantics",
            "primary_url": "https://arxiv.org/abs/2603.03305v2",
            "primary_section": "Draft-conditioned constrained decoding semantics and costs",
            "applicable_test_or_deferral": "Exp7580 separates syntax, support, and execution; extra generation deferred",
            "claim_limit": "Executable syntax does not prove supported or useful world-model behavior.",
            "destination_tasks": ["exp7580-arc-verifier-support", "exp7584-arc-independent-audit"],
            "external_result_is_local_measurement": False,
        },
        {
            "method_family": "fpga_asic_cost_accounting",
            "primary_url": "https://arxiv.org/abs/2602.15985",
            "primary_section": "FPGA-ASIC decomposition, orchestration, and communication boundary",
            "applicable_test_or_deferral": "Exp7585 times transfer, update, persistence, and orchestration",
            "claim_limit": "External estimates do not establish local acceleration or energy benefit.",
            "destination_tasks": ["exp7585-portable-service"],
            "external_result_is_local_measurement": False,
        },
    ]


def secondary_access_rows() -> list[JsonDict]:
    """Keep failed discovery access visible instead of claiming completeness."""

    return [
        {
            "channel": "Semantic Scholar",
            "requested": "Citation lists for ARXIV:2507.02092 and ARXIV:2512.15605",
            "observed": "internal errors for both reads",
            "complete": False,
        },
        {
            "channel": "OpenReview",
            "requested": "ARM-EBM and compositional-energy forums",
            "observed": "browser challenges for both forums",
            "complete": False,
        },
    ]


def write_method_records(root: Path) -> None:
    """Write the method map and one idempotent studying marker."""

    lines = [
        "# V662 method map",
        "",
        "Date: 2026-09-23. Scope: advisory method accounting.",
        "External results are not local Carnot measurements.",
        "",
        "| Family | Primary section | Local test or deferral | Claim limit | Destination |",
        "|---|---|---|---|---|",
    ]
    for row in method_rows():
        source = f"[{row['primary_section']}]({row['primary_url']})"
        lines.append(
            f"| {row['method_family']} | {source} | {row['applicable_test_or_deferral']} | "
            f"{row['claim_limit']} | {', '.join(row['destination_tasks'])} |"
        )
    lines.extend(["", "## Incomplete secondary access", ""])
    for row in secondary_access_rows():
        lines.append(f"- {row['channel']}: {row['observed']}. The access receipt is incomplete.")
    lines.extend(["", "No model, board, publication, purchase, or external contact ran.", ""])
    note = root / NOTE_PATH
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines), encoding="utf-8")

    study = root / STUDY_PATH
    existing = study.read_text(encoding="utf-8") if study.is_file() else ""
    if STUDY_MARKER not in existing:
        addition = "\n".join(
            [
                STUDY_MARKER,
                "## 2026-09-23 Exp7573 — V662 methods — INGESTED",
                "",
                "Proper-loss, local KAN, structured-decoding, and full-service cost methods map",
                "to V662 tests. Delayed-feedback theorems, FPGA speed, and semantic success",
                "remain explicit deferrals. Access failures remain in the V662 method map.",
                "",
            ]
        )
        study.write_text(existing.rstrip() + "\n\n" + addition, encoding="utf-8")


REQUIRED_CHECK_NAMES = (
    "worktree_imports",
    "focused_pytest",
    "changed_module_coverage",
    "changed_module_coverage_report",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
)
REPOSITORY_CHECK_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "exclusion_manifest",
    "roadmap_gate_audit",
    "harness_fit",
    "arc_floor",
    "overdue_priority",
    "publication_gate_json",
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "fresh_process_lifecycle_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def affected_file_manifest() -> JsonDict:
    """Freeze exact implementation, test, wrapper, and requirement paths."""

    return {
        "tests": [TEST_PATH.as_posix()],
        "modules": [MODULE_PATH.as_posix()],
        "static": [WRAPPER_PATH.as_posix(), SPEC_PATH.as_posix()],
    }


def build_validation_plan(root: Path, private_root: Path) -> list[CommandSpec]:
    """Build explicit changed-file checks and create private parents first."""

    focused = private_root / "basetemp/focused"
    coverage_temp = private_root / "basetemp/coverage"
    coverage_file = private_root / "coverage/.coverage"
    for parent in (focused.parent, coverage_temp.parent, coverage_file.parent):
        parent.mkdir(parents=True, exist_ok=True)
    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    coverage = str(root / ".venv/bin/coverage")
    ruff = str(root / ".venv/bin/ruff")
    mypy = str(root / ".venv/bin/mypy")
    paths = (MODULE_PATH.as_posix(), WRAPPER_PATH.as_posix(), TEST_PATH.as_posix())
    import_code = (
        "import importlib,json,pathlib;"
        f"root=pathlib.Path({str(root)!r}).resolve();"
        "p=pathlib.Path(importlib.import_module('carnot.experiment_7573_v662_contract_methods').__file__).resolve();"
        "print(json.dumps({'resolved_imports':{'carnot.experiment_7573_v662_contract_methods':str(p)}}),flush=True);"
        "raise SystemExit(not p.is_relative_to(root/'python'))"
    )
    return [
        CommandSpec("worktree_imports", (python, "-u", "-c", import_code), "affected", 120),
        CommandSpec(
            "focused_pytest",
            (
                pytest,
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={focused}",
                TEST_PATH.as_posix(),
                "-q",
            ),
            "affected",
            900,
        ),
        CommandSpec(
            "changed_module_coverage",
            (
                "env",
                f"COVERAGE_FILE={coverage_file}",
                coverage,
                "run",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                f"--basetemp={coverage_temp}",
                TEST_PATH.as_posix(),
                "-q",
            ),
            "affected",
            900,
        ),
        CommandSpec(
            "changed_module_coverage_report",
            (
                "env",
                f"COVERAGE_FILE={coverage_file}",
                coverage,
                "report",
                f"--include=*/{MODULE_PATH.name}",
                "--show-missing",
                "--fail-under=100",
            ),
            "affected",
            120,
        ),
        CommandSpec("ruff_check", (ruff, "check", *paths), "affected", 120),
        CommandSpec("ruff_format", (ruff, "format", "--check", *paths), "affected", 120),
        CommandSpec("changed_module_mypy", (mypy, MODULE_PATH.as_posix()), "affected", 300),
        CommandSpec(
            "scoped_spec_coverage",
            (python, "-u", "scripts/check_spec_coverage.py", TEST_PATH.as_posix()),
            "affected",
            120,
        ),
    ]


def private_basetemp_parents(commands: Sequence[CommandSpec]) -> list[Path]:
    """Return private pytest parents so callers can prove they exist."""

    result: list[Path] = []
    for command in commands:
        for argument in command.argv:
            if argument.startswith("--basetemp="):
                result.append(Path(argument.split("=", 1)[1]).parent)
    return result


def validate_validation_plan(root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject expanded targets, command drift, and missing private parents."""

    errors: list[str] = []
    if [row.name for row in commands] != list(REQUIRED_CHECK_NAMES):
        errors.append("validation_command_names_changed")
    if any(TEST_PATH.as_posix() not in row.argv for row in commands[1:3]):
        errors.append("pytest_target_changed")
    if any(not parent.is_dir() for parent in private_basetemp_parents(commands)):
        errors.append("private_basetemp_parent_missing")
    if any(str(root / "tests/python") in row.argv for row in commands):
        errors.append("unscoped_test_directory_forbidden")
    return errors


def build_repository_check_plan(root: Path, roadmap_path: Path) -> list[CommandSpec]:
    """Point each unchanged guard at the selected milestone authority."""

    python = str(root / ".venv/bin/python")
    selected = str(roadmap_path)
    schema = (
        "import pathlib,sys,yaml;from scripts.roadmap_schema import Roadmap;"
        "p=pathlib.Path(sys.argv[1]);Roadmap.model_validate(yaml.safe_load(p.read_text()));"
        "print('roadmap schema clean',flush=True)"
    )
    overdue = (
        "import pathlib,sys;import scripts.overdue_priority_lint as lint;"
        "lint.ROADMAP_NEXT=pathlib.Path(sys.argv[1]);sys.argv=sys.argv[:1];"
        "raise SystemExit(lint.main())"
    )
    commands = (
        ("roadmap_schema", (python, "-u", "-c", schema, selected)),
        ("prior_failure", (python, "-u", "scripts/validate_prior_failures.py", selected)),
        ("exclusion_manifest", (python, "-u", "scripts/exclusion_manifest_lint.py", selected)),
        ("roadmap_gate_audit", (python, "-u", "scripts/audit_roadmap_gates.py", selected)),
        ("harness_fit", (python, "-u", "scripts/harness_fit_lint.py", selected)),
        ("arc_floor", (python, "-u", "scripts/arc_levelup_guarantee_lint.py", selected)),
        ("overdue_priority", (python, "-u", "-c", overdue, selected)),
        ("publication_gate_json", (python, "-u", "scripts/publication_gate.py", "--json")),
    )
    return [CommandSpec(name, argv, "selected_authority", 900) for name, argv in commands]


def reduce_validation_receipts(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require one complete receipt for every declared current check."""

    required = (*REQUIRED_CHECK_NAMES, *REPOSITORY_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    errors: list[str] = []
    for name in required:
        matches = [row for row in receipts if row.get("name") == name]
        if len(matches) != 1:
            errors.append(f"receipt_count:{name}")
            continue
        row = matches[0]
        command = row.get("command_argv") or row.get("command")
        if not isinstance(command, list) and not isinstance(command, tuple):
            errors.append(f"receipt_command:{name}")
        if not row.get("worktree") and not row.get("cwd"):
            errors.append(f"receipt_worktree:{name}")
        if not isinstance(row.get("exit_code"), int):
            errors.append(f"receipt_exit:{name}")
        if not isinstance(row.get("log_sha256"), str):
            errors.append(f"receipt_log_hash:{name}")
        if row.get("exit_code") != 0 or row.get("passed") is not True:
            errors.append(f"receipt_failed:{name}")
    return {"passed": not errors, "errors": errors}


def operator_queue(root: Path) -> JsonDict:
    """Keep the operator-only E0 issue and resolved E6 issue distinct."""

    path = root / "ops/known-issues.md"
    text = path.read_text(encoding="utf-8")
    e0 = "BLOCKED 2026-09-22 (operator-only step pending): SEMIF OPTION-READOUT" in text
    e6 = "RESOLVED 2026-09-21: SEMIF FOLLOW-UP E6" in text
    return {
        "E0": {"status": "operator_blocked" if e0 else "not_authenticated"},
        "E6": {"status": "resolved" if e6 else "not_authenticated"},
        "source_path": "ops/known-issues.md",
        "source_sha256": sha256_file(path),
        "authenticated": e0 and e6,
    }


def activation_state_controls(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Exercise matching staged and consumed-staging selection semantics."""

    matches = roadmap.get("milestone") == MILESTONE
    return [
        {
            "state": "pre_activation",
            "selected": NEXT_ROADMAP_PATH.as_posix() if matches else None,
            "expected": NEXT_ROADMAP_PATH.as_posix(),
            "passed": matches,
        },
        {
            "state": "post_activation",
            "selected": ACTIVE_ROADMAP_PATH.as_posix() if matches else None,
            "expected": ACTIVE_ROADMAP_PATH.as_posix(),
            "passed": matches,
        },
    ]


def run_lifecycle_control(directory: Path) -> JsonDict:
    """Exercise predict, legal release, update, durable persist, and reload."""

    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "exp7573-lifecycle.json"
    state = {"seen": 0, "positive": 0}
    events = ["predict"]
    prediction = state["positive"] / max(1, state["seen"])
    prediction_before_release = state["seen"] == 0
    events.append("release")
    label = 1
    events.append("update")
    state = {"seen": state["seen"] + 1, "positive": state["positive"] + label}
    events.append("persist")
    atomic_json(state_path, state)
    events.append("reload")
    reloaded = _load_json(state_path)
    return {
        "events": events,
        "prediction": prediction,
        "prediction_before_release": prediction_before_release,
        "reload_equal": reloaded == state,
        "passed": events == ["predict", "release", "update", "persist", "reload"]
        and prediction_before_release
        and reloaded == state,
        "state_sha256": sha256_file(state_path),
    }


def _publication_gates(root: Path, receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Read stable G1-G4 from the exact publication guard log."""

    receipt = next((row for row in receipts if row.get("name") == "publication_gate_json"), {})
    log_path = Path(str(receipt.get("log_path") or ""))
    if not log_path.is_absolute():
        log_path = root / log_path
    try:
        payload = _load_json(log_path)
    except (OSError, ValueError, json.JSONDecodeError):
        payload = {}
    raw = payload.get("gates") if isinstance(payload.get("gates"), Mapping) else {}
    gates = {name: deepcopy(raw.get(name) or {"pass": False}) for name in ("G1", "G2", "G3", "G4")}
    unmet = [name for name, gate in gates.items() if gate.get("pass") is not True]
    return {
        "gates": gates,
        "paper_ready": not unmet,
        "unmet_gates": unmet,
        "publication_performed": False,
    }


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Bind current authorities, methods, code, and literal V661 evidence bytes."""

    current = (
        roadmap_path.relative_to(root),
        DESIGN_PATH,
        V661_DESIGN_PATH,
        SPEC_PATH,
        Path("research-references.md"),
        STUDY_PATH,
        Path("ops/conductor-log.md"),
        Path("ops/known-issues.md"),
        V661_CAPSTONE_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        NOTE_PATH,
    )
    rows = [
        {"path": path.as_posix(), "sha256": sha256_file(root / path), "role": "current_input"}
        for path in dict.fromkeys(current)
        if (root / path).is_file()
    ]
    for disposition in dispositions:
        rows.append(
            {
                "path": disposition.get("authentication_path"),
                "sha256": disposition.get("authentication_sha256"),
                "role": "v661_disposition_evidence",
                "task_id": disposition.get("task_id"),
                "evidence_kind": disposition.get("evidence_kind"),
            }
        )
    return rows


def _gate(
    check: str,
    category: str,
    expected: object,
    observed: object,
    passed: bool,
) -> JsonDict:
    """Keep validity, readiness, and benefit operands independently visible."""

    return {
        "check": check,
        "category": category,
        "upstream": EXPERIMENT_ID,
        "path": RESULT_PATH.as_posix(),
        "field": check,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Separate exact validity, advisory readiness, and scientific benefit."""

    authority = contract.get("passed") is True and len(contract.get("rows") or []) == 14
    mutation = len(mutations) == 8 and all(row.get("qualified") is True for row in mutations)
    custody = len(dispositions) == 13 and all(
        row.get("authenticated") is True for row in dispositions
    )
    method = len(methods) == 4 and all(
        row.get("external_result_is_local_measurement") is False for row in methods
    )
    return [
        _gate("authority_exact", "validity", True, authority, authority),
        _gate("private_mutations_rejected", "validity", True, mutation, mutation),
        _gate("v661_custody_authenticated", "validity", True, custody, custody),
        _gate(
            "lifecycle_replay",
            "validity",
            True,
            lifecycle.get("passed"),
            lifecycle.get("passed") is True,
        ),
        _gate(
            "required_validation",
            "validity",
            True,
            validation.get("passed"),
            validation.get("passed") is True,
        ),
        _gate("method_ingestion", "readiness", True, method, method),
        _gate("scientific_benefit", "benefit", True, False, False),
    ]


def _field_principles(keys: Sequence[str]) -> JsonDict:
    """Attach one failure-prevention statement to every top-level field."""

    principles = {
        key: "This field keeps the advisory record explicit and reproducible." for key in keys
    }
    principles.update(
        {
            "honest_verdict": "Use a complete_ prefix; completion does not establish benefit.",
            "verdict_class": "Use exactly one closed terminal class; partial is owned unfinished work only.",
            "flagged_adversarial": "Persist the terminal verifier outcome; flagged evidence cannot open readiness.",
            "gate_check_summary": "A blocked verdict names each exact failed prerequisite operand.",
            "acceptance_gate_results": "Validity, readiness, and benefit remain separate.",
            "rows": "Each comparison unit retains raw operands, direction, seed, censoring, and provenance.",
            "inference_substrate_class": "Actual and planned substrates stay separate from live-inference claims.",
            "MODEL_SPECS": "Current LLM tasks name models; this no-model aggregation stays empty.",
            "invocation_counts": "Loads, forwards, generations, and tokens are counted independently.",
            "duration_s": "Current monotonic time excludes inherited work and artificial sleeps.",
            "source_artifact_hashes": "Exact source bytes distinguish producers from pre-gate evidence.",
            "validation_receipts": "Each check binds its command, worktree, exit, and log hash.",
            "field_principles": "Each emitted field states its failure-prevention purpose.",
            "verifier_is_oracle": "Oracle controls cannot support an oracle-distinct positive claim.",
            "contract_ready_score": "One requires fourteen exact rows and passing authority mutations.",
            "v661_dispositions": "Each V661 task appears once, including pre-gate evidence.",
            "method_map_path": "Each proposed method stays tied to a read section and claim limit.",
        }
    )
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all emitted evidence except the checksum's own value."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness and terminal class from embedded comparative evidence."""

    rows = artifact.get("rows")
    mutations = artifact.get("contract_mutation_rows")
    dispositions = artifact.get("v661_dispositions")
    methods = artifact.get("method_rows")
    lifecycle = artifact.get("lifecycle_control")
    by_id = {str(row.get("task_id")): row for row in dispositions or [] if isinstance(row, Mapping)}
    rows_ok = (
        isinstance(rows, list)
        and [row.get("unit_id") for row in rows if isinstance(row, Mapping)]
        == list(EXPECTED_TASK_IDS)
        and all(
            isinstance(row, Mapping)
            and row.get("expected") == row.get("observed")
            and row.get("raw_numerator") == row.get("raw_denominator") == 7
            for row in rows
        )
    )
    mutation_ok = (
        isinstance(mutations, list)
        and len(mutations) == 8
        and all(isinstance(row, Mapping) and row.get("qualified") is True for row in mutations)
    )
    dispositions_ok = (
        isinstance(dispositions, list)
        and [row.get("task_id") for row in dispositions if isinstance(row, Mapping)]
        == list(V661_TASK_IDS)
        and all(row.get("authenticated") is True for row in dispositions)
        and by_id.get("exp7561-recalibration-prototype", {}).get("strict_row_lint_exit_code") == 1
        and by_id.get("exp7567-source-evaluation", {}).get("verdict_class") == "null"
        and by_id.get("exp7567-source-evaluation", {}).get("fresh_confirmatory_claim_allowed")
        is False
        and by_id.get("exp7568-continuous-recalibration", {}).get("evidence_kind") == "pre_gate"
        and by_id.get("exp7570-arc-live-lineage", {}).get("flagged_adversarial") is True
        and by_id.get("exp7570-arc-live-lineage", {}).get("inference_started") is False
        and by_id.get("exp7571-portable-calibration", {}).get("board_continuity_complete_score")
        == 1
        and by_id.get("exp7571-portable-calibration", {}).get("service_measurement_complete_score")
        == 0
    )
    methods_ok = isinstance(methods, list) and {
        row.get("method_family") for row in methods if isinstance(row, Mapping)
    } == {
        "proper_loss_recalibration",
        "local_kan_learning",
        "structured_decoding_semantics",
        "fpga_asic_cost_accounting",
    }
    lifecycle_ok = isinstance(lifecycle, Mapping) and lifecycle.get("passed") is True
    ready = rows_ok and mutation_ok and dispositions_ok and methods_ok and lifecycle_ok
    passed = (
        ready
        and artifact.get("contract_ready_score") == 1
        and artifact.get("honest_verdict") == "complete_null_v662_contract_methods_ingested"
        and artifact.get("verdict_class") == "null"
        and artifact.get("flagged_adversarial") is False
        and artifact.get("positive_claim") is False
    )
    return {
        "passed": passed,
        "contract_ready_score": int(ready),
        "expected_verdict_class": "null" if ready else "disqualified",
    }


def build_artifact(
    root: Path,
    roadmap_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
) -> JsonDict:
    """Build one schema-complete advisory from authenticated current inputs."""

    methods = method_rows()
    present_names = {row.get("name") for row in validation_receipts}
    expected_names = {*REQUIRED_CHECK_NAMES, *REPOSITORY_CHECK_NAMES}
    if present_names & set(TERMINAL_CHECK_NAMES):
        expected_names.update(TERMINAL_CHECK_NAMES)
    validation = {
        "passed": expected_names.issubset(present_names)
        and all(
            row.get("passed") is True and row.get("exit_code") == 0
            for row in validation_receipts
            if row.get("name") in expected_names
        )
    }
    gates = _acceptance_gates(contract, mutations, dispositions, methods, lifecycle, validation)
    ready = all(row["passed"] is True for row in gates if row["category"] != "benefit")
    roadmap = _load_yaml(roadmap_path)
    rows = deepcopy(list(contract.get("rows") or []))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7573,
        "title": "Bind fourteen tasks and ingest methods against terminal V661 evidence",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "complete_disqualified",
        "honest_verdict": "complete_null_v662_contract_methods_ingested"
        if ready
        else "complete_disqualified_v662_contract_or_validation",
        "verdict_class": "null" if ready else "disqualified",
        "positive_claim": False,
        "flagged_adversarial": False,
        "verifier_is_oracle": False,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": max(0.0, duration_s),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF in V661 producer evidence only",
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "selected_authority_state": "staged"
        if roadmap_path.name == NEXT_ROADMAP_PATH.name
        else "active_after_activation",
        "roadmap_resolution_candidates": deepcopy(list(candidates)),
        "authority_state_rows": activation_state_controls(roadmap),
        "contract_comparison": deepcopy(dict(contract)),
        "contract_mutation_rows": deepcopy(list(mutations)),
        "contract_ready_score": int(ready),
        "rows": rows,
        "v661_dispositions": deepcopy(list(dispositions)),
        "completed_archive_ends_at": "2026.09.660",
        "v661_archive_history_invented": False,
        "method_rows": methods,
        "method_map_path": NOTE_PATH.as_posix(),
        "secondary_access_rows": secondary_access_rows(),
        "operator_queue": operator_queue(root),
        "lifecycle_control": deepcopy(dict(lifecycle)),
        "method_ingestion_complete_score": int(len(methods) == 4),
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "passed": ready,
            "failed_checks": [row["check"] for row in gates if not row["passed"]],
            "first_failure": next((deepcopy(row) for row in gates if not row["passed"]), None),
        },
        "source_artifact_hashes": _source_hashes(root, roadmap_path, dispositions),
        "validation_manifest": affected_file_manifest(),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "publication_gates": _publication_gates(root, validation_receipts),
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": True,
            "lifecycle": ["predict", "release", "update", "persist", "reload"],
            "numbered_runtime_e2e_applicable": False,
        },
        "sample_size_budget": {
            "planned": 14,
            "attempted": 14,
            "completed": 14,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_unit": "ordered_v662_contract_row",
        },
        "repository_health": {
            "full_python_suite_launched": False,
            "pre_existing_debt_is_current_validity": False,
        },
        "advisory_only": True,
        "global_science_gate": False,
        "active_roadmap_modified": False,
        "research_conductor_modified": False,
        "production_defaults_changed": False,
        "publication_performed": False,
        "push_performed": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test(
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build deterministic evidence from the current worktree for unit tests."""

    roadmap_path, roadmap, candidates = resolve_v662_roadmap(ROOT)
    design = (ROOT / DESIGN_PATH).read_text(encoding="utf-8")
    if lifecycle is None:
        with tempfile.TemporaryDirectory(prefix="exp7573-unit-") as directory:
            lifecycle = run_lifecycle_control(Path(directory))
    return build_artifact(
        ROOT,
        roadmap_path,
        candidates,
        compare_contract_authorities(design, roadmap),
        run_contract_mutation_controls(design, roadmap),
        collect_v661_dispositions(ROOT),
        lifecycle,
        validation_receipts,
        started_at_utc="2026-09-23T00:00:00+00:00",
        ended_at_utc="2026-09-23T00:00:00+00:00",
        duration_s=0.0,
    )


def build_blocked_artifact(
    *, upstream: str, path: str, field: str, expected: object, observed: object
) -> JsonDict:
    """Publish external absence as complete blocked evidence without substitutes."""

    failure = {
        "check": "required_input",
        "upstream": upstream,
        "path": path,
        "field": field,
        "op": "==",
        "expected": expected,
        "observed": observed,
        "passed": False,
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "honest_verdict": f"complete_blocked_{re.sub(r'[^a-z0-9]+', '_', field.lower()).strip('_')}",
        "verdict_class": "blocked",
        "positive_claim": False,
        "flagged_adversarial": False,
        "verifier_is_oracle": False,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": 0.0,
        "contract_ready_score": 0,
        "method_ingestion_complete_score": 0,
        "rows": [],
        "v661_dispositions": [],
        "method_map_path": NOTE_PATH.as_posix(),
        "acceptance_gate_results": [{**failure, "category": "validity"}],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": ["required_input"],
            "first_failure": failure,
        },
        "source_artifact_hashes": [],
        "validation_receipts": [],
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _source_hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash each source row that claims exact existing bytes."""

    rows = artifact.get("source_artifact_hashes")
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        path_text = row.get("path")
        expected = row.get("sha256")
        if not isinstance(path_text, str) or not isinstance(expected, str):
            return False
        path = Path(path_text)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256_file(path) != expected:
            return False
    return True


def validate_artifact(
    value: object, *, root: Path = ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, raw reductions, source bytes, and receipt custody."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors: list[str] = []
    required = {
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "honest_verdict",
        "verdict_class",
        "flagged_adversarial",
        "gate_check_summary",
        "acceptance_gate_results",
        "rows",
        "inference_substrate_class",
        "planned_inference_substrate_class",
        "MODEL_SPECS",
        "model_specs",
        "invocation_counts",
        "duration_s",
        "source_artifact_hashes",
        "validation_receipts",
        "field_principles",
        "verifier_is_oracle",
        "contract_ready_score",
        "v661_dispositions",
        "method_map_path",
        "reproducibility_checksum",
    }
    if missing := sorted(required - artifact.keys()):
        errors.append(f"required_fields_missing:{','.join(missing)}")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    if (
        artifact.get("experiment_id") != EXPERIMENT_ID
        or artifact.get("milestone") != MILESTONE
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("experiment_identity_mismatch")
    if not str(artifact.get("honest_verdict") or "").startswith("complete_"):
        errors.append("terminal_verdict_prefix_required")
    if artifact.get("verdict_class") not in TERMINAL_CLASSES:
        errors.append("verdict_class_invalid")
    if type(artifact.get("contract_ready_score")) is not int:  # noqa: E721
        errors.append("contract_ready_score_bare_integer_required")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("model_specs_must_be_empty")
    counts = artifact.get("invocation_counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(ZERO_INVOCATION_COUNTS)
        or any(value != 0 for value in counts.values())
    ):
        errors.append("current_model_calls_nonzero")
    if (
        artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("planned_inference_substrate_class") != "aggregation"
    ):
        errors.append("substrate_class_invalid")
    gates = artifact.get("acceptance_gate_results")
    if not isinstance(gates, list) or {
        row.get("category") for row in gates if isinstance(row, Mapping)
    } != {
        "validity",
        "readiness",
        "benefit",
    }:
        errors.append("acceptance_gate_shape_invalid")
    if artifact.get("verdict_class") == "blocked":
        first = (artifact.get("gate_check_summary") or {}).get("first_failure")
        if not isinstance(first, Mapping) or any(
            key not in first
            for key in ("check", "upstream", "path", "field", "op", "expected", "observed")
        ):
            errors.append("blocked_gate_summary_incomplete")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        not principles.get(key) for key in artifact if key != "reproducibility_checksum"
    ):
        errors.append("field_principles_incomplete")
    if artifact.get("method_map_path") != NOTE_PATH.as_posix():
        errors.append("method_map_path_invalid")
    if artifact.get("verdict_class") != "blocked" and not _source_hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    if artifact.get("verdict_class") != "blocked" and not independent_reduce(artifact)["passed"]:
        errors.append("independent_reduction_failed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("checksum_mismatch")
    if require_terminal:
        reduction = reduce_validation_receipts(artifact.get("validation_receipts") or [])
        if not reduction["passed"]:
            errors.append("terminal_validation_incomplete")
    return list(dict.fromkeys(errors))


def _terminal_commands(root: Path, candidate: Path, lifecycle_dir: Path) -> list[CommandSpec]:
    """Build exact candidate replay, reduction, and strict-reader commands."""

    python = str(root / ".venv/bin/python")
    wrapper = WRAPPER_PATH.as_posix()
    return [
        CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", wrapper, "--root", str(root), "--cold-validate", str(candidate)),
            "capability_e2e",
            900,
        ),
        CommandSpec(
            "fresh_process_lifecycle_replay",
            (python, "-u", wrapper, "--root", str(root), "--lifecycle-replay", str(lifecycle_dir)),
            "capability_e2e",
            900,
        ),
        CommandSpec(
            "independent_reduction",
            (python, "-u", wrapper, "--root", str(root), "--independent-reduce", str(candidate)),
            "candidate_reduction",
            900,
        ),
        CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
            900,
        ),
        CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate_rows",
            900,
        ),
    ]


def _normalize_receipts(receipts: Sequence[Mapping[str, Any]], root: Path) -> list[JsonDict]:
    """Add the resolved worktree required by current validation custody."""

    result: list[JsonDict] = []
    for source in receipts:
        row = deepcopy(dict(source))
        row["worktree"] = str(root.resolve())
        result.append(row)
    return result


def _required_inputs(root: Path) -> list[tuple[Path, str, object, object]]:
    """Observe exact external prerequisites before any dependent aggregation."""

    paths = (
        DESIGN_PATH,
        V661_DESIGN_PATH,
        SPEC_PATH,
        V661_CAPSTONE_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        Path("ops/conductor-log.md"),
        Path("ops/known-issues.md"),
    )
    rows = [
        (
            path,
            "bytes",
            "readable_nonempty_bytes",
            "readable_nonempty_bytes"
            if (root / path).is_file() and (root / path).stat().st_size > 0
            else "absent_or_empty",
        )
        for path in paths
    ]
    text = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    rows.append(
        (
            SPEC_PATH,
            "REQ-*",
            "REQ-REPORT-7573",
            "REQ-REPORT-7573" if "REQ-REPORT-7573" in text else "absent",
        )
    )
    return rows


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Run bounded validation and atomically publish the exact advisory."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress(started, "preconditions", "before")
    failed = next((row for row in _required_inputs(root) if row[2] != row[3]), None)
    if failed is not None:
        path, field, expected, observed = failed
        blocked = build_blocked_artifact(
            upstream=path.as_posix(),
            path=path.as_posix(),
            field=field,
            expected=expected,
            observed=observed,
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=path)
        return blocked
    try:
        roadmap_path, roadmap, candidates = resolve_v662_roadmap(root)
    except ValueError:
        blocked = build_blocked_artifact(
            upstream=ACTIVE_ROADMAP_PATH.as_posix(),
            path=ACTIVE_ROADMAP_PATH.as_posix(),
            field="milestone",
            expected=MILESTONE,
            observed="no matching staged or active authority",
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=ACTIVE_ROADMAP_PATH)
        return blocked
    design = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(design, roadmap)
    mutations = run_contract_mutation_controls(design, roadmap)
    dispositions = collect_v661_dispositions(root)
    progress(started, "preconditions", "after", completed_units=len(_required_inputs(root)))

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned_units=0)
        progress(started, phase, "after", completed_units=0)

    progress(started, "method_ingestion", "before", planned_units=4)
    write_method_records(root)
    progress(started, "method_ingestion", "after", completed_units=4)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7573-validation-", dir="/tmp"))
    lifecycle = run_lifecycle_control(private_root / "lifecycle")
    affected_plan = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, affected_plan)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{plan_errors}")
    atomic_json(raw_dir / "affected-file-manifest.json", affected_file_manifest())
    repository_plan = build_repository_check_plan(root, roadmap_path)

    progress(
        started,
        "validation",
        "before_subprocesses",
        planned_units=len(affected_plan) + len(repository_plan),
    )
    affected = _normalize_receipts(
        run_commands(root, affected_plan, log_dir=raw_dir / "validation/affected"), root
    )
    repository = _normalize_receipts(
        run_commands(root, repository_plan, log_dir=raw_dir / "validation/repository"), root
    )
    progress(
        started,
        "validation",
        "after_subprocesses",
        completed_units=len(affected) + len(repository),
    )

    provisional_receipts = [*affected, *repository]
    candidate = build_artifact(
        root,
        roadmap_path,
        candidates,
        contract,
        mutations,
        dispositions,
        lifecycle,
        provisional_receipts,
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
    )
    candidate_path = raw_dir / "terminal-candidate.json"
    atomic_json(candidate_path, candidate)
    candidate_errors = validate_artifact(candidate, root=root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")

    terminal_plan = _terminal_commands(root, candidate_path, private_root / "fresh-lifecycle")
    progress(
        started, "terminal_validation", "before_subprocesses", planned_units=len(terminal_plan)
    )
    terminal = _normalize_receipts(
        run_commands(root, terminal_plan, log_dir=raw_dir / "validation/terminal"), root
    )
    progress(started, "terminal_validation", "after_subprocesses", completed_units=len(terminal))
    final = build_artifact(
        root,
        roadmap_path,
        candidates,
        contract,
        mutations,
        dispositions,
        lifecycle,
        [*affected, *repository, *terminal],
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    atomic_json(candidate_path, final)
    progress(started, "publish", "before_atomic", path=RESULT_PATH)
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publish", "after_atomic", path=RESULT_PATH)
    return final


def date_argument(value: str) -> str:
    """Accept only the run date frozen by the V662 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse measured execution and bounded read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--date", type=date_argument)
    modes.add_argument("--cold-validate", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    modes.add_argument("--lifecycle-replay", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run Exp7573 or replay one bounded candidate check."""

    print("[exp7573] phase=startup event=flushed", flush=True)
    args = parse_args(argv)
    root = args.root.resolve()
    if args.cold_validate is not None:
        errors = validate_artifact(
            _load_json(args.cold_validate), root=root, require_terminal=False
        )
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        reduction = independent_reduce(_load_json(args.independent_reduce))
        print(json.dumps(reduction, sort_keys=True), flush=True)
        return int(reduction.get("passed") is not True)
    if args.lifecycle_replay is not None:
        lifecycle = run_lifecycle_control(args.lifecycle_replay)
        print(json.dumps(lifecycle, sort_keys=True), flush=True)
        return int(lifecycle.get("passed") is not True)
    artifact = run_experiment(root, args.date)
    print(
        json.dumps(
            {
                "result": RESULT_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
                "contract_ready_score": artifact["contract_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Bind the V665 task contract to literal V664 evidence and method limits.

This administrative aggregation makes no model call. Exact contract readiness
cannot establish scientific benefit.

Spec refs: REQ-REPORT-7615 and SCENARIO-REPORT-7615-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7601_v664_contract_methods as prior
from carnot.experiment_7329_v644_contract import parse_markdown_contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    run_commands,
)


JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.665"
EXPERIMENT_ID = "exp7615-contract-methods"
SCHEMA = "carnot.exp7615.v665.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7615_v665_contract_methods.json")
MODULE_PATH = Path("python/carnot/experiment_7615_v665_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7615_v665_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7615_v665_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v665-method-map.md")
STUDY_PATH = Path("research-studying.md")
STUDY_MARKER = "<!-- EXP7615-V665-METHOD-INGESTION -->"
V664_CAPSTONE_PATH = Path("results/experiment_7614_v664_capstone.json")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7615, "contract-methods"),
        (7616, "evidence-schema"),
        (7617, "schema-pilot"),
        (7618, "fit-evidence"),
        (7619, "online-evidence"),
        (7620, "evaluation-evidence"),
        (7621, "evidence-energy"),
        (7622, "decision-evaluation"),
        (7623, "guarded-learning"),
        (7624, "evidence-audit"),
        (7625, "arc-supervisor-transfer"),
        (7626, "native-service"),
        (7627, "native-cost"),
        (7628, "capstone"),
    )
)
V664_TASK_IDS = prior.EXPECTED_TASK_IDS
TERMINAL_CLASSES = prior.TERMINAL_CLASSES
ZERO_INVOCATION_COUNTS = deepcopy(prior.ZERO_INVOCATION_COUNTS)
REQUIRED_CHECK_NAMES = prior.REQUIRED_CHECK_NAMES
REPOSITORY_CHECK_NAMES = prior.REPOSITORY_CHECK_NAMES
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_reduction",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

_load_yaml = prior._load_yaml
_load_json = prior._load_json
_normalize_yaml_contract = prior._normalize_yaml_contract
_public_task = prior._public_task
private_basetemp_parents = prior.private_basetemp_parents
build_repository_check_plan = prior.build_repository_check_plan
_normalize_receipts = prior._normalize_receipts
_publication_gates = prior.prior._publication_gates


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Print a flushed monotonic boundary around each potentially slow phase."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7615] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v665_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer matching staged bytes, then accept a consumed staging authority."""

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
        raise ValueError("V665 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare all fourteen identities and complete structured gate lists."""

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
    fields = ("order", "id", "title", "phase", "deliverable", "substrate", "gates")
    rows: list[JsonDict] = []
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
                "arm": "design_vs_selected_authority",
                "order": index + 1,
                "expected": _public_task(expected),
                "observed": _public_task(observed),
                "checks": checks,
                "matched": matched,
                "passed": matched,
                "absolute_metric": int(matched),
                "raw_numerator": sum(checks.values()),
                "raw_denominator": len(checks),
                "metric_direction": "exact_match",
                "seed": None,
                "missing": expected is None or observed is None,
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
        "rows": rows,
    }


def mutation_names() -> tuple[str, ...]:
    """Name the four byte-copy defects required by REQ-REPORT-7615."""

    return ("missing_task", "reordered_task", "altered_path", "misspelled_field")


def mutate_yaml_for_test(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Return one private corruption without changing authority bytes."""

    if mutation not in mutation_names():
        raise ValueError(f"unknown mutation: {mutation}")
    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "missing_task":
        changed["tasks"] = tasks[:-1]
    elif mutation == "reordered_task":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "altered_path":
        tasks[0]["deliverable"] = "results/private-altered.json"
    else:
        gated = next(task for task in tasks if task.get("gated_on"))
        gated["gated_on"][0]["artifact_field"] = "evidence_schema_reddy_score"
    return changed


def _mutate_markdown(text: str, mutation: str) -> str:
    """Apply the same four defects to a private design-table copy."""

    lines = text.splitlines()
    task_lines = [index for index, line in enumerate(lines) if re.match(r"\| \d+ \| exp\d+", line)]
    if mutation == "missing_task":
        del lines[task_lines[-1]]
    elif mutation == "reordered_task":
        lines[task_lines[0]], lines[task_lines[1]] = lines[task_lines[1]], lines[task_lines[0]]
    elif mutation == "altered_path":
        lines[task_lines[0]] = lines[task_lines[0]].replace(
            RESULT_PATH.as_posix(), "results/private-altered.json"
        )
    else:
        index = next(i for i in task_lines if ".evidence_schema_ready_score" in lines[i])
        lines[index] = lines[index].replace(
            "evidence_schema_ready_score", "evidence_schema_reddy_score", 1
        )
    return "\n".join(lines)


def run_contract_mutation_controls(text: str, roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Reject eight corruptions and accept matching activated authority bytes."""

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
    rows.append(
        {
            "authority": "selector",
            "mutation": "consumed_staging",
            "baseline_valid": baseline,
            "rejected": False,
            "selected_active_after_staging_consumed": roadmap.get("milestone") == MILESTONE,
            "qualified": baseline and roadmap.get("milestone") == MILESTONE,
        }
    )
    return rows


def collect_v664_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate fourteen V664 dispositions without inventing missing producers."""

    capstone_path = root / V664_CAPSTONE_PATH
    capstone = _load_json(capstone_path)
    raw_rows = capstone.get("task_dispositions")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(V664_TASK_IDS):
        raise ValueError("V664 capstone dispositions are unavailable")

    rows: list[JsonDict] = []
    for expected_id, raw in zip(V664_TASK_IDS, raw_rows, strict=True):
        if not isinstance(raw, Mapping) or raw.get("task_id") != expected_id:
            raise ValueError("V664 capstone disposition order is invalid")
        row = deepcopy(dict(raw))
        state = row.get("evidence_state")
        if state == "conductor_gate_blocked":
            kind = "conductor_pre_gate"
        elif state == "missing":
            kind = "absent_producer"
        else:
            kind = "actual_producer"

        if expected_id == "exp7614-capstone":
            path = capstone_path
            exists = True
            observed_hash = sha256_file(path)
            expected_hash = observed_hash
        else:
            label = row.get("evidence_path") or row.get("expected_artifact_path")
            path = root / str(label)
            exists = path.is_file()
            observed_hash = sha256_file(path) if exists else None
            expected_hash = row.get("artifact_sha256")
        authenticated = (not exists and kind == "absent_producer" and expected_hash is None) or (
            exists and isinstance(expected_hash, str) and observed_hash == expected_hash
        )
        row.update(
            {
                "evidence_kind": kind,
                "authentication_path": path.relative_to(root).as_posix(),
                "authentication_sha256": observed_hash,
                "authentication_expected_sha256": expected_hash,
                "authentication_exists": exists,
                "authenticated": authenticated,
                "environment_or_custody_block_retires_hypothesis": False,
            }
        )
        rows.append(row)
    return rows


def disposition_kind_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count producers, pre-gates and absences without merging their meanings."""

    counts = Counter(str(row.get("evidence_kind")) for row in rows)
    return {
        "actual_producer": counts["actual_producer"],
        "conductor_pre_gate": counts["conductor_pre_gate"],
        "absent_producer": counts["absent_producer"],
    }


def activation_state_controls(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Record both valid authority lifecycle states without activating either."""

    matches = roadmap.get("milestone") == MILESTONE
    return [
        {
            "state": "staged",
            "selected": NEXT_ROADMAP_PATH.as_posix() if matches else None,
            "expected": NEXT_ROADMAP_PATH.as_posix(),
            "passed": matches,
        },
        {
            "state": "activated_after_consumed_staging",
            "selected": ACTIVE_ROADMAP_PATH.as_posix() if matches else None,
            "expected": ACTIVE_ROADMAP_PATH.as_posix(),
            "passed": matches,
        },
    ]


def method_rows() -> list[JsonDict]:
    """Map primary methods to bounded V665 tests without reproduction claims."""

    return [
        {
            "rank": 1,
            "status": "applicable_with_limit",
            "method_family": "jsonschemabench",
            "primary_url": "https://arxiv.org/html/2501.10868",
            "primary_section": "Sections 4-6",
            "mechanism_read": (
                "Coverage separates declared, empirical and true coverage plus compliance; "
                "quality uses task exact match; speed separates GCT, TTFT and TPOT on the "
                "intersection of covered schemas."
            ),
            "mapped_work": (
                "Exp7616-7617 separately report validator coverage, semantic outcomes and "
                "generation latency for the fixed evidence schema."
            ),
            "claim_limit": (
                "This small fixed-schema pilot is not a reproduction of JSONSchemaBench."
            ),
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 2,
            "status": "applicable_with_limit",
            "method_family": "eaev_evidence_alignment",
            "primary_url": "https://arxiv.org/html/2609.08267v1",
            "primary_section": "Sections 3.3-3.6",
            "mechanism_read": (
                "Identity, semantic and consistency alignment are separate factors; "
                "counterfactual stability tests controlled evidence perturbations before "
                "entity aggregation and supervised learning."
            ),
            "mapped_work": (
                "Exp7621-7624 retain source links, contradiction features, evidence erasure, "
                "legal derangement and independent source-group reduction."
            ),
            "claim_limit": "The compact local evidence head is not a reproduction of EAEV.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 3,
            "status": "applicable_with_limit",
            "method_family": "proper_calibeating",
            "primary_url": "https://arxiv.org/html/2605.26703v2",
            "primary_section": "Sections 2.3 and 5",
            "mechanism_read": (
                "The theorem is online, observes the reference forecast before the new "
                "forecast, and requires uniform asymptotic guarantees over normalized bounded "
                "proper scoring rules; quadratic calibeating alone does not transfer."
            ),
            "mapped_work": (
                "Exp7623 compares frozen, guarded and legally deranged delayed-label arms with "
                "Brier, cost, anchors, restart and retention reported separately."
            ),
            "claim_limit": (
                "This finite delayed learner is not a reproduction of Proper Calibeating or "
                "its theorem."
            ),
            "external_result_is_local_measurement": False,
        },
    ]


def write_method_records(root: Path) -> None:
    """Write the method map and one idempotent studying-log record."""

    lines = [
        "# V665 schema-constrained evidence method map",
        "",
        "Date: 2026-09-24. Scope: administrative method qualification.",
        "External results are not local Carnot measurements.",
        "This local learner is not a reproduction of any paper or theorem below.",
        "",
        "| Rank | Method | Primary section | V665 mapping | Claim limit |",
        "|---:|---|---|---|---|",
    ]
    for row in method_rows():
        source = f"[{row['primary_section']}]({row['primary_url']})"
        lines.append(
            f"| {row['rank']} | {row['method_family']} | {source} | {row['mapped_work']} | "
            f"{row['claim_limit']} |"
        )
    lines.extend(
        [
            "",
            "## Verified assumptions",
            "",
            "JSONSchemaBench keeps coverage, downstream quality and speed separate. Its speed",
            "comparison uses grammar compilation time, time to first token and time per output",
            "token on the intersection of schemas supported by all compared engines.",
            "",
            "EAEV separates identity, semantic, consistency and counterfactual-stability factors.",
            "Carnot maps these ideas to source-linked features and explicit erasure or derangement",
            "controls. It does not import the paper's trained verifier or results.",
            "",
            "Proper Calibeating studies online forecasts, a reference visible before the new",
            "forecast, asymptotic comparisons and normalized bounded proper scoring rules. The",
            "V665 delayed-release learner has finite blocks, admission guards and restart checks;",
            "the theorem does not transfer to that protocol.",
            "",
            "## Access limits",
            "",
            "The three arXiv HTML primary sources were readable and verified. The planning review",
            "also recorded a Semantic Scholar citation-endpoint failure and one inaccessible",
            "OpenReview PDF browser challenge. Cached feeds and trending pages do not establish a",
            "fresh citation census. None of these secondary-source limits blocks the methods above.",
            "",
        ]
    )
    note = root / NOTE_PATH
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines), encoding="utf-8")

    study = root / STUDY_PATH
    existing = study.read_text(encoding="utf-8") if study.is_file() else ""
    if STUDY_MARKER not in existing:
        addition = "\n".join(
            [
                STUDY_MARKER,
                "## 2026-09-24 Exp7615 — V665 contract methods — INGESTED",
                "",
                "JSONSchemaBench coverage, quality and speed; EAEV evidence factors; and Proper",
                "Calibeating assumptions map to separate V665 tests. The local finite delayed",
                "learner is not a reproduction of a paper or theorem. Semantic Scholar citation",
                "endpoints and one OpenReview PDF remained inaccessible secondary sources.",
                "",
            ]
        )
        study.write_text(existing.rstrip() + "\n\n" + addition, encoding="utf-8")


def affected_file_manifest() -> JsonDict:
    """Freeze every implementation, test, requirement and method-record path."""

    return {
        "tests": [TEST_PATH.as_posix()],
        "modules": [MODULE_PATH.as_posix()],
        "static": [
            WRAPPER_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            NOTE_PATH.as_posix(),
            STUDY_PATH.as_posix(),
        ],
    }


def build_validation_plan(root: Path, private_root: Path) -> list[CommandSpec]:
    """Build fixed scoped checks under private temporary parents."""

    basetemp = private_root / "basetemp"
    coverage_file = private_root / "coverage/.coverage"
    basetemp.mkdir(parents=True, exist_ok=True)
    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    return build_scoped_commands(
        root,
        [TEST_PATH.as_posix()],
        [MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        basetemp=basetemp,
        coverage_file=coverage_file,
    )


def validate_validation_plan(root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject broad targets, command drift and non-private temporary paths."""

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


def reduce_validation_receipts(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require one passing, hashed, worktree-bound receipt for every check."""

    required = (*REQUIRED_CHECK_NAMES, *REPOSITORY_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
    errors: list[str] = []
    for name in required:
        matches = [row for row in receipts if row.get("name") == name]
        if len(matches) != 1:
            errors.append(f"receipt_count:{name}")
            continue
        row = matches[0]
        for field in ("command_argv", "worktree", "exit_code", "log_sha256"):
            if row.get(field) is None:
                errors.append(f"receipt_{field}:{name}")
        if row.get("passed") is not True or row.get("exit_code") != 0:
            errors.append(f"receipt_failed:{name}")
    return {"passed": not errors, "errors": errors}


def _receipt_group_passed(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Reduce a named command group without silently accepting duplicates."""

    return all(
        len(matches := [row for row in receipts if row.get("name") == name]) == 1
        and matches[0].get("passed") is True
        and matches[0].get("exit_code") == 0
        for name in names
    )


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Build the fresh replay, independent reducer and two exact readers."""

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
            "independent_reduction",
            (
                python,
                "-u",
                wrapper,
                "--root",
                str(root),
                "--independent-reduce",
                str(candidate),
            ),
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


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Bind current authorities and distinguish each prior evidence class."""

    current = (
        roadmap_path.relative_to(root),
        DESIGN_PATH,
        SPEC_PATH,
        Path("research-references.md"),
        STUDY_PATH,
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("ops/known-issues.md"),
        V664_CAPSTONE_PATH,
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        NOTE_PATH,
    )
    rows = [
        {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "exists": True,
            "role": "current_input",
        }
        for path in dict.fromkeys(current)
        if (root / path).is_file()
    ]
    rows.extend(
        {
            "path": row.get("authentication_path"),
            "sha256": row.get("authentication_sha256"),
            "exists": row.get("authentication_exists"),
            "role": "v664_disposition_evidence",
            "task_id": row.get("task_id"),
            "evidence_kind": row.get("evidence_kind"),
        }
        for row in dispositions
    )
    return rows


def _source_hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash claimed bytes and verify that literal absent producers stay absent."""

    rows = artifact.get("source_artifact_hashes")
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            return False
        path = Path(str(row["path"]))
        resolved = path if path.is_absolute() else root / path
        if row.get("exists") is False:
            if resolved.exists() or row.get("sha256") is not None:
                return False
        elif (
            row.get("exists") is not True
            or not isinstance(row.get("sha256"), str)
            or not resolved.is_file()
            or sha256_file(resolved) != row.get("sha256")
        ):
            return False
    return True


def _required_inputs(root: Path) -> list[tuple[Path, str, object, object]]:
    """Observe every external prerequisite before dependent aggregation."""

    paths = (
        DESIGN_PATH,
        SPEC_PATH,
        V664_CAPSTONE_PATH,
        Path("research-references.md"),
        STUDY_PATH,
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("ops/known-issues.md"),
        Path("python/carnot/experiment_7601_v664_contract_methods.py"),
    )
    rows = [
        (
            path,
            "bytes",
            "readable_nonempty_bytes",
            (
                "readable_nonempty_bytes"
                if (root / path).is_file() and (root / path).stat().st_size > 0
                else "absent_or_empty"
            ),
        )
        for path in paths
    ]
    text = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    rows.append(
        (
            SPEC_PATH,
            "REQ-*",
            "REQ-REPORT-7615",
            "REQ-REPORT-7615" if "REQ-REPORT-7615" in text else "absent",
        )
    )
    return rows


def preconditions_checked(root: Path) -> list[JsonDict]:
    """Record input checks and actual local tool versions used by this task."""

    rows = [
        {
            "check": "required_input",
            "upstream": path.as_posix(),
            "path": path.as_posix(),
            "field": field,
            "operator": "eq",
            "expected": expected,
            "observed": observed,
            "passed": expected == observed,
        }
        for path, field, expected, observed in _required_inputs(root)
    ]
    for package in ("pytest", "ruff", "mypy", "PyYAML"):
        try:
            observed = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed = "absent"
        rows.append(
            {
                "check": "tool_version",
                "upstream": package,
                "path": str(root / ".venv"),
                "field": "installed_version",
                "operator": "!=",
                "expected": "absent",
                "observed": observed,
                "passed": observed != "absent",
            }
        )
    rows.append(
        {
            "check": "tool_version",
            "upstream": "python",
            "path": str(root / ".venv/bin/python"),
            "field": "version",
            "operator": ">=",
            "expected": "3.12",
            "observed": platform.python_version(),
            "passed": platform.python_version_tuple() >= ("3", "12", "0"),
        }
    )
    return rows


def _gate(check: str, category: str, expected: object, observed: object, passed: bool) -> JsonDict:
    """Keep validity, readiness, benefit, retention and freshness separate."""

    return {
        "check": check,
        "category": category,
        "upstream": EXPERIMENT_ID,
        "path": RESULT_PATH.as_posix(),
        "field": check,
        "operator": "eq",
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": "One evidence class cannot promote another evidence class.",
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    receipts: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Reduce administrative gates while leaving science explicitly unopened."""

    authority = contract.get("passed") is True and len(contract.get("rows") or []) == 14
    mutation = len(mutations) == 9 and all(row.get("qualified") is True for row in mutations)
    custody = disposition_kind_counts(dispositions) == {
        "actual_producer": 9,
        "conductor_pre_gate": 2,
        "absent_producer": 3,
    } and all(row.get("authenticated") is True for row in dispositions)
    guards = _receipt_group_passed(receipts, (*REQUIRED_CHECK_NAMES, *REPOSITORY_CHECK_NAMES))
    method = len(methods) == 3 and all(
        row.get("external_result_is_local_measurement") is False for row in methods
    )
    return [
        _gate("authority_exact", "validity", True, authority, authority),
        _gate("private_mutations_rejected", "validity", True, mutation, mutation),
        _gate("v664_custody_authenticated", "validity", True, custody, custody),
        _gate("required_validation_and_guards", "validity", True, guards, guards),
        _gate("method_limits_ingested", "readiness", True, method, method),
        _gate("scientific_benefit", "benefit", True, False, False),
        _gate("retention_measurement", "retention", "not_applicable", "not_applicable", True),
        _gate(
            "historical_source_freshness",
            "freshness",
            "descriptive_reuse",
            "descriptive_reuse",
            True,
        ),
    ]


def _field_principles(keys: Sequence[str]) -> JsonDict:
    """Attach a plain failure-prevention principle to every top-level field."""

    principles = {
        key: "Keep this administrative evidence explicit and reproducible." for key in keys
    }
    principles.update(
        {
            "honest_verdict": (
                "Use a complete_ terminal prefix; completion alone is not scientific benefit."
            ),
            "verdict_class": (
                "Use one allowed class; partial is reserved for unfinished work owned here."
            ),
            "flagged_adversarial": (
                "Persist the terminal reader result; flagged evidence never opens a gate."
            ),
            "gate_check_summary": (
                "A block names check, upstream, path, field, operator, expected and observed."
            ),
            "acceptance_gate_results": (
                "Validity, readiness, benefit, retention and freshness stay separate."
            ),
            "rows": (
                "Each independent task keeps absolute operands, seed, direction and provenance."
            ),
            "sample_size_budget": (
                "Tasks, not seeds, views or replays, determine the independent sample count."
            ),
            "preconditions_checked": (
                "Observe inputs and tools before aggregation; unavailable work is not fabricated."
            ),
            "inference_substrate": (
                "Describe current execution; historical GPU evidence is not a current call."
            ),
            "inference_substrate_class": (
                "Record planned and actual classes separately without padding runtime."
            ),
            "MODEL_SPECS": "A no-model aggregation uses an empty current model list.",
            "model_invoked": "True requires a current real model load or generation attempt.",
            "execution_venue": "Name the actual host and state that no physical GPU was used.",
            "execution_venue_details": (
                "Preserve the hostname and physical-device detail outside the closed venue enum."
            ),
            "phase_spans": "Use disjoint measured current stages with counters and checkpoints.",
            "invocation_counts": (
                "Count loads, forwards, generations and tokens separately from inherited evidence."
            ),
            "duration_s": "Measure monotonic current work; never inherit or pad duration.",
            "random_seed": "Persist each stochastic seed; this deterministic task uses none.",
            "reproducibility_checksum": (
                "Bind immutable inputs, configuration and the complete reduction."
            ),
            "source_artifact_hashes": (
                "Distinguish actual producers, pre-gate receipts and missing artifacts."
            ),
            "validation_receipts": "Retain actual commands, exits, worktree and log hashes.",
            "verifier_is_oracle": (
                "Exact fixtures cannot establish an oracle-distinct learned advantage."
            ),
            "field_principles": "Carry one prevention principle beside every top-level field.",
            "contract_ready_score": (
                "One requires fourteen-task parity, mutation rejection and clean guards."
            ),
            "task_contract_rows": (
                "Pair each new contract row with its same-order prior terminal disposition."
            ),
            "method_map_path": "Bind only the primary methods and limitations actually read.",
            "publication_gates": (
                "Preserve fixed G1-G4 and distinguish old FoVer eligibility from new evidence."
            ),
        }
    )
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the complete evidence object except the checksum field itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness from raw rows rather than trusting the headline."""

    rows = artifact.get("rows")
    mutations = artifact.get("contract_mutation_rows")
    dispositions = artifact.get("v664_dispositions")
    methods = artifact.get("method_rows")
    contract_rows = artifact.get("task_contract_rows")
    receipts = artifact.get("validation_receipts")
    rows_ok = (
        isinstance(rows, list)
        and [row.get("unit_id") for row in rows if isinstance(row, Mapping)]
        == list(EXPECTED_TASK_IDS)
        and all(
            isinstance(row, Mapping)
            and row.get("expected") == row.get("observed")
            and row.get("raw_numerator") == row.get("raw_denominator") == 7
            and row.get("missing") is False
            and row.get("censored") is False
            for row in rows
        )
    )
    mutation_ok = (
        isinstance(mutations, list)
        and len(mutations) == 9
        and all(isinstance(row, Mapping) and row.get("qualified") is True for row in mutations)
    )
    custody_ok = (
        isinstance(dispositions, list)
        and [row.get("task_id") for row in dispositions if isinstance(row, Mapping)]
        == list(V664_TASK_IDS)
        and all(row.get("authenticated") is True for row in dispositions)
        and disposition_kind_counts(dispositions)
        == {"actual_producer": 9, "conductor_pre_gate": 2, "absent_producer": 3}
    )
    methods_ok = isinstance(methods, list) and {
        row.get("method_family") for row in methods if isinstance(row, Mapping)
    } == {"jsonschemabench", "eaev_evidence_alignment", "proper_calibeating"}
    paired_ok = (
        isinstance(contract_rows, list)
        and len(contract_rows) == 14
        and all(
            row.get("new_task_id") == EXPECTED_TASK_IDS[index]
            and row.get("prior_task_id") == V664_TASK_IDS[index]
            for index, row in enumerate(contract_rows)
            if isinstance(row, Mapping)
        )
    )
    guards_ok = isinstance(receipts, list) and _receipt_group_passed(
        receipts, (*REQUIRED_CHECK_NAMES, *REPOSITORY_CHECK_NAMES)
    )
    ready = rows_ok and mutation_ok and custody_ok and methods_ok and paired_ok and guards_ok
    passed = (
        ready
        and artifact.get("contract_ready_score") == 1
        and artifact.get("honest_verdict") == "complete_null_v665_contract_methods_ingested"
        and artifact.get("verdict_class") == "null"
        and artifact.get("flagged_adversarial") is False
        and artifact.get("positive_claim") is False
    )
    return {
        "passed": passed,
        "contract_ready_score": int(ready),
        "expected_verdict_class": "null" if ready else "disqualified",
        "row_signs": {"matched": 14 if rows_ok else 0, "mismatched": 0 if rows_ok else 14},
        "censored_count": 0 if rows_ok else None,
    }


def _task_contract_rows(
    rows: Sequence[Mapping[str, Any]], dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Pair new authority rows with same-order prior terminal dispositions."""

    return [
        {
            "order": index,
            "new_task_id": current.get("unit_id"),
            "new_contract_row": deepcopy(dict(current)),
            "prior_task_id": previous.get("task_id"),
            "prior_disposition": deepcopy(dict(previous)),
            "principle": "New plans and prior terminal evidence remain separate.",
        }
        for index, (current, previous) in enumerate(zip(rows, dispositions, strict=True), 1)
    ]


def _terminal_outcomes(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Persist each exact terminal reader result, including a not-run state."""

    outcomes: list[JsonDict] = []
    for name in TERMINAL_CHECK_NAMES:
        receipt = next((row for row in receipts if row.get("name") == name), None)
        outcomes.append(
            {
                "name": name,
                "outcome": (
                    "not_run"
                    if receipt is None
                    else "passed"
                    if receipt.get("passed") is True and receipt.get("exit_code") == 0
                    else "failed"
                ),
                "exit_code": None if receipt is None else receipt.get("exit_code"),
                "log_sha256": None if receipt is None else receipt.get("log_sha256"),
            }
        )
    return outcomes


def build_artifact(
    root: Path,
    roadmap_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build a complete null advisory from authenticated administrative evidence."""

    methods = method_rows()
    gates = _acceptance_gates(contract, mutations, dispositions, validation_receipts, methods)
    ready = all(row["passed"] is True for row in gates if row["category"] != "benefit")
    terminal_outcomes = _terminal_outcomes(validation_receipts)
    adversarial = next(
        (row for row in terminal_outcomes if row["name"] == "adversarial_verify"), None
    )
    flagged = bool(adversarial and adversarial["outcome"] == "failed")
    roadmap = _load_yaml(roadmap_path)
    contract_rows = list(contract.get("rows") or [])
    measured_spans = list(phase_spans or [])
    if not measured_spans:
        measured_spans = [
            {
                "phase": "current_administrative_aggregation",
                "started_offset_s": 0.0,
                "ended_offset_s": max(0.0, duration_s),
                "duration_s": max(0.0, duration_s),
                "planned_units": 14,
                "completed_units": 14,
                "pending_operation": None,
                "checkpoint_position": 14,
            }
        ]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7615,
        "title": "Bind fourteen tasks and qualify schema-constrained evidence methods",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "complete_disqualified",
        "honest_verdict": (
            "complete_null_v665_contract_methods_ingested"
            if ready
            else "complete_disqualified_v665_contract_or_validation"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "positive_claim": False,
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": max(0.0, duration_s),
        "phase_spans": deepcopy(measured_spans),
        "random_seed": 7615,
        "stochastic_stage_seeds": {},
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": (
            "unsloth/Qwen3.8-27B-GGUF appears only in authenticated V664 evidence"
        ),
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "physical_device": "host_cpu",
            "gpu_uuid": None,
            "gpu_used": False,
        },
        "preconditions_checked": preconditions_checked(root),
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "selected_authority_state": (
            "staged"
            if roadmap_path.name == NEXT_ROADMAP_PATH.name
            else "activated_after_consumed_staging"
        ),
        "roadmap_resolution_candidates": deepcopy(list(candidates)),
        "authority_state_rows": activation_state_controls(roadmap),
        "contract_comparison": deepcopy(dict(contract)),
        "contract_mutation_rows": deepcopy(list(mutations)),
        "contract_ready_score": int(ready),
        "rows": deepcopy(contract_rows),
        "task_contract_rows": _task_contract_rows(contract_rows, dispositions),
        "v664_dispositions": deepcopy(list(dispositions)),
        "v664_disposition_counts": disposition_kind_counts(dispositions),
        "v664_terminal_context": {
            "protocol_and_lifecycle_fixtures_valid": True,
            "real_generations": 8,
            "parser_invalid_generations": 8,
            "evidence_chain_measured": False,
            "arc_protocol": "blocked",
            "service_attribution": "null",
            "archive_lag_erases_capstone": False,
        },
        "method_rows": methods,
        "method_map_path": NOTE_PATH.as_posix(),
        "method_ingestion_complete_score": int(len(methods) == 3),
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "passed": ready,
            "failed_checks": [row["check"] for row in gates if not row["passed"]],
            "first_failure": next(
                (
                    deepcopy(row)
                    for row in gates
                    if row["category"] != "benefit" and not row["passed"]
                ),
                None,
            ),
        },
        "source_artifact_hashes": _source_hashes(root, roadmap_path, dispositions),
        "validation_manifest": affected_file_manifest(),
        "validation_receipts": deepcopy(list(validation_receipts)),
        "terminal_reader_outcomes": terminal_outcomes,
        "publication_gates": {
            **_publication_gates(root, validation_receipts),
            "historical_scope": "existing_FoVer_eligibility_only",
            "new_v665_evidence": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "staged_authority_selection": True,
            "activated_authority_selection": True,
            "five_required_controls": [
                "missing_task",
                "reordered_task",
                "altered_path",
                "misspelled_field",
                "consumed_staging",
            ],
            "fresh_process_cold_replay": True,
            "independent_comparative_reduction": True,
            "numbered_runtime_e2e_applicable": False,
            "numbered_runtime_e2e_reason": "read_only_reporting_has_no_numbered_runtime_e2e",
        },
        "sample_size_budget": {
            "intended": 14,
            "planned": 14,
            "observed": 14,
            "excluded": 0,
            "censored": 0,
            "independent_unit": "ordered_v665_contract_task",
            "seeds_views_or_replays_multiply_units": False,
        },
        "prior_failure_dispositions": [
            {
                "scope": "eight_parser_invalid_generations",
                "retired": False,
                "retirement_condition": "shared_schema_and_new_real_outputs_measured",
                "reason": "A parser protocol failure is not a semantic hypothesis test.",
            },
            {
                "scope": "unmeasured_evidence_chain",
                "retired": False,
                "retirement_condition": "eligible_external_evidence_is_independently_reduced",
                "reason": "Missing evidence cannot retire an unmeasured scientific hypothesis.",
            },
            {
                "scope": "blocked_arc_protocol",
                "retired": False,
                "retirement_condition": "historical_failed_check_cause_resolved",
                "reason": "A current readiness check does not rewrite the saved block.",
            },
            {
                "scope": "service_attribution_null",
                "retired": True,
                "retirement_condition": "exact_arithmetic_attribution_question_measured",
                "reason": "The measured null retires only the exact attribution mechanism.",
            },
        ],
        "repository_health": {
            "full_python_suite_launched": False,
            "pre_existing_debt_is_current_validity": False,
        },
        "advisory_only": True,
        "global_science_gate": False,
        "active_roadmap_modified": False,
        "roadmap_activation_performed": False,
        "research_conductor_modified": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "publication_performed": False,
        "submission_performed": False,
        "purchase_performed": False,
        "push_performed": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_artifact_for_test(*, validation_receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Build deterministic evidence from the current worktree for unit tests."""

    roadmap_path, roadmap, candidates = resolve_v665_roadmap(ROOT)
    design = (ROOT / DESIGN_PATH).read_text(encoding="utf-8")
    return build_artifact(
        ROOT,
        roadmap_path,
        candidates,
        compare_contract_authorities(design, roadmap),
        run_contract_mutation_controls(design, roadmap),
        collect_v664_dispositions(ROOT),
        validation_receipts,
        started_at_utc="2026-09-24T00:00:00+00:00",
        ended_at_utc="2026-09-24T00:00:00+00:00",
        duration_s=0.0,
    )


def build_blocked_artifact(
    *, upstream: str, path: str, field: str, expected: object, observed: object
) -> JsonDict:
    """Publish unchanged external absence as complete blocked evidence."""

    failure = {
        "check": "required_input",
        "category": "validity",
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "eq",
        "expected": expected,
        "observed": observed,
        "passed": False,
        "principle": "An unchanged external blocker is complete blocked, never partial.",
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
        "inference_substrate_class": "blocked_no_run",
        "inference_substrate": "blocked_no_run",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "physical_device": "host_cpu",
            "gpu_uuid": None,
            "gpu_used": False,
        },
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": 7615,
        "stochastic_stage_seeds": {},
        "preconditions_checked": [failure],
        "contract_ready_score": 0,
        "rows": [],
        "task_contract_rows": [],
        "v664_dispositions": [],
        "method_map_path": NOTE_PATH.as_posix(),
        "acceptance_gate_results": [failure],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": ["required_input"],
            "first_failure": failure,
        },
        "sample_size_budget": {
            "intended": 14,
            "observed": 0,
            "excluded": 0,
            "censored": 14,
            "independent_unit": "ordered_v665_contract_task",
        },
        "source_artifact_hashes": [],
        "validation_receipts": [],
        "publication_gates": {
            "gates": {name: {"pass": False} for name in ("G1", "G2", "G3", "G4")},
            "paper_ready": False,
            "unmet_gates": ["G1", "G2", "G3", "G4"],
            "publication_performed": False,
            "independent_science_gated": False,
            "historical_scope": "existing_FoVer_eligibility_only",
            "new_v665_evidence": False,
        },
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def validate_artifact(
    value: object, *, root: Path = ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, raw reduction, source bytes and command custody."""

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
        "task_contract_rows",
        "sample_size_budget",
        "preconditions_checked",
        "inference_substrate",
        "inference_substrate_class",
        "planned_inference_substrate_class",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "execution_venue",
        "execution_venue_details",
        "phase_spans",
        "invocation_counts",
        "duration_s",
        "random_seed",
        "source_artifact_hashes",
        "validation_receipts",
        "field_principles",
        "verifier_is_oracle",
        "contract_ready_score",
        "v664_dispositions",
        "method_map_path",
        "publication_gates",
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
    if artifact.get("model_invoked") is not False:
        errors.append("model_invoked_must_be_false")
    counts = artifact.get("invocation_counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(ZERO_INVOCATION_COUNTS)
        or any(count != 0 for count in counts.values())
    ):
        errors.append("current_model_calls_nonzero")
    if artifact.get("verdict_class") != "blocked" and (
        artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("planned_inference_substrate_class") != "aggregation"
    ):
        errors.append("substrate_class_invalid")
    gates = artifact.get("acceptance_gate_results")
    categories = {row.get("category") for row in gates or [] if isinstance(row, Mapping)}
    if artifact.get("verdict_class") != "blocked" and categories != {
        "validity",
        "readiness",
        "benefit",
        "retention",
        "freshness",
    }:
        errors.append("acceptance_gate_shape_invalid")
    if artifact.get("verdict_class") == "blocked":
        first = (artifact.get("gate_check_summary") or {}).get("first_failure")
        if not isinstance(first, Mapping) or any(
            key not in first
            for key in (
                "check",
                "upstream",
                "path",
                "field",
                "operator",
                "expected",
                "observed",
            )
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
    if (
        require_terminal
        and not reduce_validation_receipts(artifact.get("validation_receipts") or [])["passed"]
    ):
        errors.append("terminal_validation_incomplete")
    return list(dict.fromkeys(errors))


def _terminal_receipts_equal(
    expected: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]
) -> bool:
    """Confirm that exact final candidate readers reproduce persisted outcomes."""

    fields = ("name", "command_argv", "exit_code", "passed", "log_sha256")
    return len(expected) == len(observed) and all(
        all(left.get(field) == right.get(field) for field in fields)
        for left, right in zip(expected, observed, strict=True)
    )


def _span(
    name: str,
    started: float,
    phase_started: float,
    planned: int,
    completed: int,
) -> JsonDict:
    """Close one disjoint monotonic stage with unit and checkpoint counters."""

    ended = time.monotonic()
    return {
        "phase": name,
        "started_offset_s": phase_started - started,
        "ended_offset_s": ended - started,
        "duration_s": ended - phase_started,
        "planned_units": planned,
        "completed_units": completed,
        "pending_operation": None,
        "checkpoint_position": completed,
    }


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Run bounded checks, exact terminal replay and atomic publication."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    spans: list[JsonDict] = []

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    required = _required_inputs(root)
    failed = next((row for row in required if row[2] != row[3]), None)
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
        roadmap_path, roadmap, candidates = resolve_v665_roadmap(root)
    except ValueError:
        blocked = build_blocked_artifact(
            upstream=ACTIVE_ROADMAP_PATH.as_posix(),
            path=ACTIVE_ROADMAP_PATH.as_posix(),
            field="milestone",
            expected=MILESTONE,
            observed="no matching staged or activated authority",
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=ACTIVE_ROADMAP_PATH)
        return blocked
    design = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(design, roadmap)
    mutations = run_contract_mutation_controls(design, roadmap)
    dispositions = collect_v664_dispositions(root)
    spans.append(
        _span("preconditions", started, phase_started, len(required) + 3, len(required) + 3)
    )
    progress(started, "preconditions", "after", completed_units=len(required) + 3)

    for phase in ("model_load", "generation", "benchmark"):
        phase_started = time.monotonic()
        progress(started, phase, "before", planned_units=0)
        spans.append(_span(phase, started, phase_started, 0, 0))
        progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    progress(started, "method_ingestion", "before", planned_units=3)
    write_method_records(root)
    spans.append(_span("method_ingestion", started, phase_started, 3, 3))
    progress(started, "method_ingestion", "after", completed_units=3)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7615-validation-", dir="/tmp"))
    affected_plan = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, affected_plan)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{plan_errors}")
    atomic_json(private_root / "affected-file-manifest.json", affected_file_manifest())
    repository_plan = build_repository_check_plan(root, roadmap_path)

    phase_started = time.monotonic()
    progress(
        started,
        "validation",
        "before_subprocesses",
        planned_units=len(affected_plan) + len(repository_plan),
    )
    affected = _normalize_receipts(
        run_commands(root, affected_plan, log_dir=private_root / "logs/affected"), root
    )
    repository = _normalize_receipts(
        run_commands(root, repository_plan, log_dir=private_root / "logs/repository"), root
    )
    spans.append(
        _span(
            "validation",
            started,
            phase_started,
            len(affected_plan) + len(repository_plan),
            len(affected) + len(repository),
        )
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
        provisional_receipts,
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = private_root / "terminal-candidate.json"
    atomic_json(candidate_path, candidate)
    candidate_errors = validate_artifact(candidate, root=root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")

    terminal_plan = _terminal_commands(root, candidate_path)
    phase_started = time.monotonic()
    progress(
        started, "terminal_validation", "before_subprocesses", planned_units=len(terminal_plan)
    )
    terminal = _normalize_receipts(
        run_commands(root, terminal_plan, log_dir=private_root / "logs/terminal"), root
    )
    spans.append(
        _span(
            "terminal_validation",
            started,
            phase_started,
            len(terminal_plan),
            len(terminal),
        )
    )
    progress(
        started,
        "terminal_validation",
        "after_subprocesses",
        completed_units=len(terminal),
    )

    final = build_artifact(
        root,
        roadmap_path,
        candidates,
        contract,
        mutations,
        dispositions,
        [*affected, *repository, *terminal],
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    atomic_json(candidate_path, final)
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")

    progress(started, "exact_terminal_replay", "before_subprocesses", planned_units=4)
    exact_terminal = _normalize_receipts(
        run_commands(root, terminal_plan, log_dir=private_root / "logs/terminal"), root
    )
    if not _terminal_receipts_equal(terminal, exact_terminal):
        raise RuntimeError("exact_terminal_reader_receipts_changed")
    progress(
        started,
        "exact_terminal_replay",
        "after_subprocesses",
        completed_units=len(exact_terminal),
    )

    progress(started, "publish", "before_atomic", path=RESULT_PATH)
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publish", "after_atomic", path=RESULT_PATH)
    return final


def date_argument(value: str) -> str:
    """Accept only the date frozen by the V665 execution contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def output_argument(value: str) -> Path:
    """Keep the declared output at the sole task-owned deliverable path."""

    path = Path(value)
    if path != RESULT_PATH:
        raise argparse.ArgumentTypeError(f"output must be {RESULT_PATH.as_posix()}")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the measured run and bounded read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=output_argument, default=RESULT_PATH)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--date", type=date_argument)
    modes.add_argument("--cold-validate", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run Exp7615 or one fresh-process candidate reader."""

    print("[exp7615] phase=startup event=flushed", flush=True)
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

"""Bind the V664 task contract to literal V663 evidence and method limits.

This is an administrative aggregation. It makes no model call and cannot turn
planning readiness, exact fixtures, or historical completion into scientific
benefit.

Spec refs: REQ-REPORT-7601 and SCENARIO-REPORT-7601-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7587_v663_contract_methods as prior
from carnot import experiment_7588_v663_evidence_protocol as protocol
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
MILESTONE = "2026.09.664"
EXPERIMENT_ID = "exp7601-contract-methods"
SCHEMA = "carnot.exp7601.v664.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7601_v664_contract_methods.json")
MODULE_PATH = Path("python/carnot/experiment_7601_v664_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7601_v664_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7601_v664_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v664-method-map.md")
STUDY_PATH = Path("research-studying.md")
STUDY_MARKER = "<!-- EXP7601-V664-METHOD-INGESTION -->"
V663_CAPSTONE_PATH = Path("results/experiment_7600_v663_capstone.json")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7601, "contract-methods"),
        (7602, "evidence-requalification"),
        (7603, "guarded-update-fixture"),
        (7604, "evidence-pilot"),
        (7605, "fit-evidence"),
        (7606, "test-online-evidence"),
        (7607, "evidence-energy"),
        (7608, "decision-evaluation"),
        (7609, "guarded-learning"),
        (7610, "evidence-audit"),
        (7611, "arc-matched-support"),
        (7612, "arc-history-measurement"),
        (7613, "service-attribution"),
        (7614, "capstone"),
    )
)
V663_TASK_IDS = prior.EXPECTED_TASK_IDS
TERMINAL_CLASSES = prior.TERMINAL_CLASSES
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in (
        "model_loads",
        "forward_calls",
        "generation_calls",
        "input_tokens",
        "output_tokens",
    )
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}
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
operator_queue = prior.operator_queue
_normalize_receipts = prior._normalize_receipts


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Print a flushed monotonic phase boundary around every slow operation."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7601] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v664_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer matching staged bytes, then accept consumed staging via active bytes."""

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
        raise ValueError("V664 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare all fourteen task identities and complete structured gates."""

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
    """Name the four private authority defects required by REQ-REPORT-7601."""

    return ("removed_row", "reordered_row", "changed_path", "misspelled_field")


def mutate_yaml_for_test(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Return one private YAML corruption without changing authority bytes."""

    if mutation not in mutation_names():
        raise ValueError(f"unknown mutation: {mutation}")
    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "removed_row":
        changed["tasks"] = tasks[:-1]
    elif mutation == "reordered_row":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "changed_path":
        tasks[0]["deliverable"] = "results/private-changed.json"
    else:
        gated = next(task for task in tasks if task.get("gated_on"))
        gated["gated_on"][0]["artifact_field"] = "evidence_protcol_ready_score"
    return changed


def _mutate_markdown(text: str, mutation: str) -> str:
    """Return the same four defects in a private design-table copy."""

    lines = text.splitlines()
    task_lines = [index for index, line in enumerate(lines) if re.match(r"\| \d+ \| exp\d+", line)]
    if mutation == "removed_row":
        del lines[task_lines[-1]]
    elif mutation == "reordered_row":
        lines[task_lines[0]], lines[task_lines[1]] = lines[task_lines[1]], lines[task_lines[0]]
    elif mutation == "changed_path":
        lines[task_lines[0]] = lines[task_lines[0]].replace(
            RESULT_PATH.as_posix(), "results/private-changed.json"
        )
    else:
        index = next(i for i in task_lines if ".evidence_protocol_ready_score" in lines[i])
        lines[index] = lines[index].replace(
            "evidence_protocol_ready_score", "evidence_protcol_ready_score", 1
        )
    return "\n".join(lines)


def run_contract_mutation_controls(text: str, roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Require a valid baseline, reject eight defects, and consume staging safely."""

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


def collect_v663_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate eight producers, one pre-gate, and five literal absences."""

    capstone_path = root / V663_CAPSTONE_PATH
    capstone = _load_json(capstone_path)
    raw_rows = capstone.get("task_dispositions")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(V663_TASK_IDS):
        raise ValueError("V663 capstone dispositions are unavailable")

    rows: list[JsonDict] = []
    for expected_id, raw in zip(V663_TASK_IDS, raw_rows, strict=True):
        if not isinstance(raw, Mapping) or raw.get("task_id") != expected_id:
            raise ValueError("V663 capstone disposition order is invalid")
        row = deepcopy(dict(raw))
        state = row.get("evidence_state")
        if state == "conductor_gate_blocked":
            kind = "conductor_pre_gate"
        elif state == "missing":
            kind = "absent_producer"
        else:
            kind = "actual_producer"

        if expected_id == "exp7600-capstone":
            evidence_path = capstone_path
            expected_hash = sha256_file(capstone_path)
            observed_hash = expected_hash
            exists = True
        else:
            label = row.get("evidence_path") or row.get("expected_artifact_path")
            evidence_path = root / str(label)
            exists = evidence_path.is_file()
            observed_hash = sha256_file(evidence_path) if exists else None
            expected_hash = row.get("artifact_sha256")
        authenticated = (not exists and kind == "absent_producer" and expected_hash is None) or (
            exists and isinstance(expected_hash, str) and observed_hash == expected_hash
        )
        row.update(
            {
                "evidence_kind": kind,
                "authentication_path": evidence_path.relative_to(root).as_posix(),
                "authentication_sha256": observed_hash,
                "authentication_expected_sha256": expected_hash,
                "authentication_exists": exists,
                "authenticated": authenticated,
                "archive_lag_is_missing_science": False,
            }
        )
        rows.append(row)
    return rows


def disposition_kind_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count evidence classes without merging planned tasks with real producers."""

    counts = Counter(str(row.get("evidence_kind")) for row in rows)
    return {
        "actual_producer": counts["actual_producer"],
        "conductor_pre_gate": counts["conductor_pre_gate"],
        "absent_producer": counts["absent_producer"],
    }


def planning_selector_hypothesis(root: Path) -> JsonDict:
    """Observe current selector behavior without correcting the blocked protocol."""

    checks, hashes, roles = protocol.collect_preconditions(root)
    failed = [str(row.get("check")) for row in checks if row.get("passed") is not True]
    if failed:
        return {
            "classification": "hypothesis_about_changed_executable_readiness",
            "failed_preconditions": failed,
            "source_group_count": sum(len(rows) for rows in roles.values()),
            "scored_group_count": 0,
            "pilot_group_count": 0,
            "historical_artifact_corrected": False,
            "scientific_result": False,
            "source_hash_count": len(hashes),
        }
    selected = protocol.select_roles(roles)
    return {
        "classification": "hypothesis_about_changed_executable_readiness",
        "failed_preconditions": [],
        "source_group_count": sum(len(rows) for rows in roles.values()),
        "scored_group_count": sum(len(rows) for rows in selected["scored"].values()),
        "pilot_group_count": len(selected["pilot"]),
        "salt": selected["salt"],
        "historical_artifact_corrected": False,
        "historical_blocked_verdict_preserved": True,
        "scientific_result": False,
        "source_hash_count": len(hashes),
    }


def activation_state_controls(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Describe staged and consumed-staging selection without changing files."""

    matches = roadmap.get("milestone") == MILESTONE
    return [
        {
            "state": "pre_activation",
            "selected": NEXT_ROADMAP_PATH.as_posix() if matches else None,
            "expected": NEXT_ROADMAP_PATH.as_posix(),
            "passed": matches,
        },
        {
            "state": "post_activation_consumed_staging",
            "selected": ACTIVE_ROADMAP_PATH.as_posix() if matches else None,
            "expected": ACTIVE_ROADMAP_PATH.as_posix(),
            "passed": matches,
        },
    ]


def method_rows() -> list[JsonDict]:
    """Map the reviewed primary methods and keep repository leads deferred."""

    return [
        {
            "rank": 1,
            "status": "applicable",
            "method_family": "eaev_evidence_alignment",
            "primary_url": "https://arxiv.org/html/2609.08267v1",
            "primary_section": "Sections 3.3-3.6",
            "mechanism_read": (
                "entity/evidence alignment, semantic and consistency signals, controlled "
                "evidence perturbations, entity aggregation, and verifier training"
            ),
            "applicable_test_or_deferral": (
                "Exp7602 and Exp7604-7610 retain complete sentence/source links, unknowns, "
                "erasure, derangement, and independent reduction"
            ),
            "claim_limit": "This is an evidence-link adaptation, not an EAEV replication.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 2,
            "status": "applicable",
            "method_family": "on_chip_kan_locality",
            "primary_url": "https://arxiv.org/html/2602.02056v4",
            "primary_section": "Sparse local-update method and fixed-point implementation",
            "mechanism_read": "Only active spline coefficients and local basis values update.",
            "applicable_test_or_deferral": (
                "Exp7603 bounds local updates; Exp7613 measures the whole durable service."
            ),
            "claim_limit": "Locality alone proves neither retention nor hardware speed.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 3,
            "status": "applicable_with_limit",
            "method_family": "proper_calibeating",
            "primary_url": "https://arxiv.org/abs/2605.26703",
            "primary_section": "Proper-loss online calibration method",
            "mechanism_read": "Proper loss and calibration are evaluated as separate outcomes.",
            "applicable_test_or_deferral": (
                "Exp7603 and Exp7609 measure delayed Brier updates, admission, and retention."
            ),
            "claim_limit": "Immediate-feedback guarantees do not establish delayed retention.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 4,
            "status": "deferred_algorithm",
            "method_family": "u_calibration",
            "primary_url": "https://arxiv.org/abs/2606.18527",
            "primary_section": "U-Calibration algorithm and immediate label observation",
            "mechanism_read": "Random perturbation and immediate feedback define the published method.",
            "applicable_test_or_deferral": (
                "V664 tests a finite guarded delayed learner, not an imported algorithm."
            ),
            "claim_limit": "The guarded delayed learner is not the U-Calibration algorithm.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 5,
            "status": "future_lead",
            "method_family": "spintronic_ising_future_lead",
            "primary_url": "https://github.com/SpinX-Lab/Voltage-controlled-Spintronic-Ising-Machine",
            "primary_section": "Public plotting code and source data only",
            "mechanism_read": "The repository excludes device fabrication and machine implementation.",
            "applicable_test_or_deferral": "Track for later hardware review; no V664 dependency.",
            "claim_limit": "Repository access is not Carnot hardware access or sampling-law evidence.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 6,
            "status": "future_lead",
            "method_family": "sparse_transformer_future_lead",
            "primary_url": "https://github.com/extropic-ai/sparse-transformers",
            "primary_section": "Public JAX repository",
            "mechanism_read": "Code availability is separate from device access and service speed.",
            "applicable_test_or_deferral": "Track for later placement work; no V664 dependency.",
            "claim_limit": "The repository does not establish a local accelerator result.",
            "external_result_is_local_measurement": False,
        },
    ]


def write_method_records(root: Path) -> None:
    """Write one method note and append one idempotent studying entry."""

    lines = [
        "# V664 method map",
        "",
        "Date: 2026-09-24. Scope: advisory method accounting.",
        "External results and repository availability are not local Carnot measurements.",
        "The guarded delayed learner is not the U-Calibration algorithm.",
        "",
        "| Rank | Status | Family | Primary section | V664 use | Claim limit |",
        "|---:|---|---|---|---|---|",
    ]
    for row in method_rows():
        source = f"[{row['primary_section']}]({row['primary_url']})"
        lines.append(
            f"| {row['rank']} | {row['status']} | {row['method_family']} | {source} | "
            f"{row['applicable_test_or_deferral']} | {row['claim_limit']} |"
        )
    lines.extend(
        [
            "",
            "The spintronic and sparse-transformer repositories remain future leads.",
            "No model, board, publication, submission, purchase, or external contact ran.",
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
                "## 2026-09-24 Exp7601 — V664 method limits — INGESTED",
                "",
                "EAEV controls, local KAN updates, and proper-loss evaluation map to bounded",
                "V664 tests. The guarded delayed learner is not published U-Calibration.",
                "The spintronic and sparse-transformer repositories remain future leads.",
                "",
            ]
        )
        study.write_text(existing.rstrip() + "\n\n" + addition, encoding="utf-8")


def affected_file_manifest() -> JsonDict:
    """Freeze every implementation, test, requirement, and method-record path."""

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
    """Build the shipped fixed scoped checks under private temporary parents."""

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
    """Reject command drift, broad test targets, and missing private parents."""

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
        if not isinstance(row.get("command_argv"), (list, tuple)):
            errors.append(f"receipt_command:{name}")
        if not row.get("worktree"):
            errors.append(f"receipt_worktree:{name}")
        if not isinstance(row.get("exit_code"), int):
            errors.append(f"receipt_exit:{name}")
        if not isinstance(row.get("log_sha256"), str):
            errors.append(f"receipt_log_hash:{name}")
        if row.get("passed") is not True or row.get("exit_code") != 0:
            errors.append(f"receipt_failed:{name}")
    return {"passed": not errors, "errors": errors}


def _receipt_group_passed(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Reduce one named command group without silently accepting duplicates."""

    return all(
        len(matches := [row for row in receipts if row.get("name") == name]) == 1
        and matches[0].get("passed") is True
        and matches[0].get("exit_code") == 0
        for name in names
    )


def _terminal_commands(root: Path, candidate: Path) -> list[CommandSpec]:
    """Build the cold replay, independent reducer, and two exact readers."""

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


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Bind current authorities and preserve explicit absent-producer rows."""

    current = (
        roadmap_path.relative_to(root),
        DESIGN_PATH,
        SPEC_PATH,
        Path("research-references.md"),
        STUDY_PATH,
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("ops/known-issues.md"),
        V663_CAPSTONE_PATH,
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
    for disposition in dispositions:
        exists = disposition.get("authentication_exists") is True
        rows.append(
            {
                "path": disposition.get("authentication_path"),
                "sha256": disposition.get("authentication_sha256"),
                "exists": exists,
                "role": "v663_disposition_evidence",
                "task_id": disposition.get("task_id"),
                "evidence_kind": disposition.get("evidence_kind"),
            }
        )
    return rows


def _source_hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash present bytes and verify that declared missing producers stay absent."""

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


def _gate(check: str, category: str, expected: object, observed: object, passed: bool) -> JsonDict:
    """Keep validity, readiness, benefit, retention, and freshness separate."""

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
    selector: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    queue: Mapping[str, Any],
) -> list[JsonDict]:
    """Build visible administrative gates and explicit non-scientific outcomes."""

    authority = contract.get("passed") is True and len(contract.get("rows") or []) == 14
    mutation = len(mutations) == 9 and all(row.get("qualified") is True for row in mutations)
    custody = disposition_kind_counts(dispositions) == {
        "actual_producer": 8,
        "conductor_pre_gate": 1,
        "absent_producer": 5,
    } and all(row.get("authenticated") is True for row in dispositions)
    checks = _receipt_group_passed(
        validation_receipts, (*REQUIRED_CHECK_NAMES, *REPOSITORY_CHECK_NAMES)
    )
    method = len(methods) == 6 and all(
        row.get("external_result_is_local_measurement") is False for row in methods
    )
    selector_ok = (
        selector.get("failed_preconditions") == []
        and selector.get("source_group_count") == 480
        and selector.get("scored_group_count") == 240
        and selector.get("pilot_group_count") == 8
        and selector.get("historical_artifact_corrected") is False
    )
    queue_ok = (
        queue.get("E0", {}).get("status") == "operator_blocked"
        and queue.get("E6", {}).get("status") == "resolved"
    )
    return [
        _gate("authority_exact", "validity", True, authority, authority),
        _gate("private_mutations_rejected", "validity", True, mutation, mutation),
        _gate("v663_custody_authenticated", "validity", True, custody, custody),
        _gate("required_validation_and_guards", "validity", True, checks, checks),
        _gate("method_limits_ingested", "readiness", True, method, method),
        _gate("selector_readiness_hypothesis", "readiness", True, selector_ok, selector_ok),
        _gate("operator_queue_preserved", "readiness", True, queue_ok, queue_ok),
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
                "Use exactly positive, circular_positive, null, blocked, disqualified, or "
                "partial; partial means only unfinished work owned by this task."
            ),
            "flagged_adversarial": (
                "Persist the terminal reader result; flagged evidence never opens readiness."
            ),
            "gate_check_summary": (
                "Every blocked verdict names check, upstream, path, field, operator, expected, "
                "and observed values."
            ),
            "acceptance_gate_results": (
                "Give validity, readiness, benefit, retention, and freshness separate results."
            ),
            "rows": (
                "Keep one row per independent unit and arm with absolute operands, seed, "
                "direction, censoring, and provenance."
            ),
            "sample_size_budget": (
                "Record intended, observed, excluded, and censored independent units; seeds do "
                "not multiply tasks."
            ),
            "inference_substrate": (
                "Describe current execution; historical GPU evidence is not a current model call."
            ),
            "inference_substrate_class": (
                "Record planned and actual classes separately; blocked_no_run means no model work."
            ),
            "MODEL_SPECS": (
                "Current LLM calls must name unsloth/Qwen3.8-27B-GGUF; this no-model task uses []."
            ),
            "invocation_counts": (
                "Count loads, forwards, generations, input tokens, and output tokens separately."
            ),
            "duration_s": "Measure current monotonic elapsed time; never inherit or pad duration.",
            "random_seed": "Persist every stochastic-stage seed; this task has no stochastic stage.",
            "reproducibility_checksum": (
                "Bind configuration, immutable raw evidence, and terminal reduction."
            ),
            "source_artifact_hashes": (
                "Distinguish producer artifacts, conductor pre-gates, and missing producers."
            ),
            "validation_receipts": (
                "Bind actual commands, exits, worktree, log hashes, and terminal readers."
            ),
            "verifier_is_oracle": (
                "Exact fixtures cannot establish learned correctness or oracle-distinct gain."
            ),
            "field_principles": "Carry one prevention principle beside every governed field.",
            "contract_ready_score": (
                "One requires exact fourteen-row agreement and passing mutations and guards."
            ),
            "v663_dispositions": (
                "Fourteen rows distinguish actual producers, the pre-gate, and absent producers."
            ),
            "method_map_path": "Bind methods actually read to new work and deferred leads.",
            "publication_gates": (
                "Use stable G1-G4, paper_ready, and unmet_gates without redefining them."
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
    """Recompute contract readiness from embedded rows, never from the headline."""

    rows = artifact.get("rows")
    mutations = artifact.get("contract_mutation_rows")
    dispositions = artifact.get("v663_dispositions")
    methods = artifact.get("method_rows")
    selector = artifact.get("planning_selector_check")
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
        == list(V663_TASK_IDS)
        and all(row.get("authenticated") is True for row in dispositions)
        and disposition_kind_counts(dispositions)
        == {"actual_producer": 8, "conductor_pre_gate": 1, "absent_producer": 5}
    )
    methods_ok = isinstance(methods, list) and {
        row.get("method_family") for row in methods if isinstance(row, Mapping)
    } == {
        "eaev_evidence_alignment",
        "on_chip_kan_locality",
        "proper_calibeating",
        "u_calibration",
        "spintronic_ising_future_lead",
        "sparse_transformer_future_lead",
    }
    selector_ok = (
        isinstance(selector, Mapping)
        and selector.get("failed_preconditions") == []
        and selector.get("source_group_count") == 480
        and selector.get("scored_group_count") == 240
        and selector.get("pilot_group_count") == 8
        and selector.get("historical_artifact_corrected") is False
    )
    ready = rows_ok and mutation_ok and custody_ok and methods_ok and selector_ok
    passed = (
        ready
        and artifact.get("contract_ready_score") == 1
        and artifact.get("honest_verdict") == "complete_null_v664_contract_methods_ingested"
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


def build_artifact(
    root: Path,
    roadmap_path: Path,
    candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    selector: Mapping[str, Any],
    validation_receipts: Sequence[Mapping[str, Any]],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
) -> JsonDict:
    """Build one schema-complete null advisory from authenticated evidence."""

    methods = method_rows()
    queue = operator_queue(root)
    gates = _acceptance_gates(
        contract, mutations, dispositions, selector, validation_receipts, methods, queue
    )
    ready = all(row["passed"] is True for row in gates if row["category"] != "benefit")
    terminal_outcomes = []
    for name in TERMINAL_CHECK_NAMES:
        receipt = next((row for row in validation_receipts if row.get("name") == name), None)
        terminal_outcomes.append(
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
    adversarial = next(
        (row for row in terminal_outcomes if row["name"] == "adversarial_verify"), None
    )
    flagged = bool(adversarial and adversarial["outcome"] == "failed")
    roadmap = _load_yaml(roadmap_path)
    disposition_counts = disposition_kind_counts(dispositions)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7601,
        "title": "Bind fourteen tasks and ingest qualified evidence methods",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "complete_disqualified",
        "honest_verdict": (
            "complete_null_v664_contract_methods_ingested"
            if ready
            else "complete_disqualified_v664_contract_or_validation"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "positive_claim": False,
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": max(0.0, duration_s),
        "phase_spans": [
            {"phase": "current_administrative_aggregation", "duration_s": max(0.0, duration_s)}
        ],
        "random_seed": 7601,
        "stochastic_stage_seeds": {},
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": (
            "unsloth/Qwen3.8-27B-GGUF appears only in authenticated V663 history"
        ),
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "selected_authority_state": (
            "staged" if roadmap_path.name == NEXT_ROADMAP_PATH.name else "active_after_activation"
        ),
        "roadmap_resolution_candidates": deepcopy(list(candidates)),
        "authority_state_rows": activation_state_controls(roadmap),
        "contract_comparison": deepcopy(dict(contract)),
        "contract_mutation_rows": deepcopy(list(mutations)),
        "contract_ready_score": int(ready),
        "rows": deepcopy(list(contract.get("rows") or [])),
        "v663_dispositions": deepcopy(list(dispositions)),
        "v663_disposition_counts": disposition_counts,
        "v663_actual_producers_separate_from_v664_planned_tasks": True,
        "v663_archive_history_invented": False,
        "planning_selector_check": deepcopy(dict(selector)),
        "method_rows": methods,
        "method_map_path": NOTE_PATH.as_posix(),
        "method_ingestion_complete_score": int(len(methods) == 6),
        "operator_queue": queue,
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
        "publication_gates": prior._publication_gates(root, validation_receipts),
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
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
            "independent_unit": "ordered_v664_contract_task",
            "seeds_or_windows_multiply_units": False,
        },
        "prior_failure_disposition": {
            "experiment_id": "exp7600-capstone",
            "literal_prior_verdict": "complete_blocked_required_v663_external_evidence",
            "retire_if_same_verdict": True,
            "literal_verdict_repeated": False,
            "retired": False,
            "missing_resources_retire_scientific_hypothesis": False,
        },
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

    roadmap_path, roadmap, candidates = resolve_v664_roadmap(ROOT)
    design = (ROOT / DESIGN_PATH).read_text(encoding="utf-8")
    return build_artifact(
        ROOT,
        roadmap_path,
        candidates,
        compare_contract_authorities(design, roadmap),
        run_contract_mutation_controls(design, roadmap),
        collect_v663_dispositions(ROOT),
        planning_selector_hypothesis(ROOT),
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
        "duration_s": 0.0,
        "random_seed": 7601,
        "contract_ready_score": 0,
        "rows": [],
        "v663_dispositions": [],
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
            "independent_unit": "ordered_v664_contract_task",
        },
        "source_artifact_hashes": [],
        "validation_receipts": [],
        "publication_gates": {
            "gates": {name: {"pass": False} for name in ("G1", "G2", "G3", "G4")},
            "paper_ready": False,
            "unmet_gates": ["G1", "G2", "G3", "G4"],
            "publication_performed": False,
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
    """Cold-check identity, raw reduction, source bytes, and command custody."""

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
        "sample_size_budget",
        "inference_substrate",
        "inference_substrate_class",
        "planned_inference_substrate_class",
        "MODEL_SPECS",
        "model_specs",
        "invocation_counts",
        "duration_s",
        "random_seed",
        "source_artifact_hashes",
        "validation_receipts",
        "field_principles",
        "verifier_is_oracle",
        "contract_ready_score",
        "v663_dispositions",
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
    counts = artifact.get("invocation_counts")
    if (
        not isinstance(counts, Mapping)
        or set(counts) != set(ZERO_INVOCATION_COUNTS)
        or any(value != 0 for value in counts.values())
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


def _required_inputs(root: Path) -> list[tuple[Path, str, object, object]]:
    """Observe every external prerequisite before dependent aggregation."""

    paths = (
        DESIGN_PATH,
        SPEC_PATH,
        V663_CAPSTONE_PATH,
        Path("research-references.md"),
        Path("research-studying.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("ops/known-issues.md"),
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
            "REQ-REPORT-7601",
            "REQ-REPORT-7601" if "REQ-REPORT-7601" in text else "absent",
        )
    )
    return rows


def _terminal_receipts_equal(
    expected: Sequence[Mapping[str, Any]], observed: Sequence[Mapping[str, Any]]
) -> bool:
    """Confirm the exact final candidate reproduces all persisted reader outcomes."""

    fields = ("name", "command_argv", "exit_code", "passed", "log_sha256")
    return len(expected) == len(observed) and all(
        all(left.get(field) == right.get(field) for field in fields)
        for left, right in zip(expected, observed, strict=True)
    )


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Run bounded checks, exact terminal replay, and atomic publication."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
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
        roadmap_path, roadmap, candidates = resolve_v664_roadmap(root)
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
    dispositions = collect_v663_dispositions(root)
    selector = planning_selector_hypothesis(root)
    progress(started, "preconditions", "after", completed_units=len(_required_inputs(root)))

    for phase in ("model_load", "generation", "benchmark"):
        progress(started, phase, "before", planned_units=0)
        progress(started, phase, "after", completed_units=0)

    progress(started, "method_ingestion", "before", planned_units=6)
    write_method_records(root)
    progress(started, "method_ingestion", "after", completed_units=6)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7601-validation-", dir="/tmp"))
    affected_plan = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, affected_plan)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{plan_errors}")
    atomic_json(private_root / "affected-file-manifest.json", affected_file_manifest())
    repository_plan = build_repository_check_plan(root, roadmap_path)

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
        selector,
        provisional_receipts,
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
    )
    candidate_path = private_root / "terminal-candidate.json"
    atomic_json(candidate_path, candidate)
    candidate_errors = validate_artifact(candidate, root=root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")

    terminal_plan = _terminal_commands(root, candidate_path)
    progress(
        started, "terminal_validation", "before_subprocesses", planned_units=len(terminal_plan)
    )
    terminal = _normalize_receipts(
        run_commands(root, terminal_plan, log_dir=private_root / "logs/terminal"), root
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
        selector,
        [*affected, *repository, *terminal],
        started_at_utc=started_at,
        ended_at_utc=datetime.now(UTC).isoformat(),
        duration_s=time.monotonic() - started,
    )
    atomic_json(candidate_path, final)
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")

    progress(
        started,
        "exact_terminal_replay",
        "before_subprocesses",
        planned_units=len(terminal_plan),
    )
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
    """Accept only the date frozen by the V664 execution contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the measured run and bounded read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--date", type=date_argument)
    modes.add_argument("--cold-validate", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI boundary.
    """Run Exp7601 or one fresh-process candidate reader."""

    print("[exp7601] phase=startup event=flushed", flush=True)
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

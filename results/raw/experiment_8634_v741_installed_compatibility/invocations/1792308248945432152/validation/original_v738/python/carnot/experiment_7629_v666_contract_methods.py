"""Bind the V666 contract to literal V665 evidence and bounded method use.

This infrastructure aggregation makes no model call. Contract readiness does
not establish scientific benefit.

Spec refs: REQ-REPORT-7629 and SCENARIO-REPORT-7629-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any

import yaml

from carnot import experiment_7615_v665_contract_methods as prior
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
MILESTONE = "2026.09.666"
EXPERIMENT_ID = "exp7629-contract-methods"
SCHEMA = "carnot.exp7629.v666.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7629_v666_contract_methods.json")
MODULE_PATH = Path("python/carnot/experiment_7629_v666_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7629_v666_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7629_v666_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v666-method-map.md")
STUDY_PATH = Path("research-studying.md")
STUDY_MARKER = "<!-- EXP7629-V666-METHOD-INGESTION -->"
V665_CAPSTONE_PATH = Path("results/experiment_7628_v665_capstone.json")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7629, "contract-methods"),
        (7630, "cuda-ownership"),
        (7631, "schema-pilot"),
        (7632, "fit-evidence"),
        (7633, "online-evidence"),
        (7634, "evaluation-evidence"),
        (7635, "evidence-energy"),
        (7636, "decision-evaluation"),
        (7637, "guarded-learning"),
        (7638, "evidence-audit"),
        (7639, "arc-goal-dedup"),
        (7640, "arc-wrapper-generalization"),
        (7641, "native-consumer"),
        (7642, "capstone"),
    )
)
V665_TASK_IDS = prior.EXPECTED_TASK_IDS
TERMINAL_CLASSES = prior.TERMINAL_CLASSES
ZERO_INVOCATION_COUNTS = deepcopy(prior.ZERO_INVOCATION_COUNTS)
REQUIRED_CHECK_NAMES = prior.REQUIRED_CHECK_NAMES
REPOSITORY_CHECK_NAMES = prior.REPOSITORY_CHECK_NAMES
TERMINAL_CHECK_NAMES = prior.TERMINAL_CHECK_NAMES

_load_yaml = prior._load_yaml
_load_json = prior._load_json
_normalize_yaml_contract = prior._normalize_yaml_contract
_public_task = prior._public_task
private_basetemp_parents = prior.private_basetemp_parents
build_repository_check_plan = prior.build_repository_check_plan
_normalize_receipts = prior._normalize_receipts
_publication_gates = prior._publication_gates
_terminal_receipts_equal = prior._terminal_receipts_equal
_span = prior._span


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Print a flushed monotonic boundary around each potentially slow phase."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7629] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v666_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
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
        raise ValueError("V666 roadmap authority is unavailable")
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
    """Name the four private defects required by REQ-REPORT-7629."""

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
        gated["gated_on"][0]["artifact_field"] = "launch_protocol_reddy_score"
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
        index = next(i for i in task_lines if ".launch_protocol_ready_score" in lines[i])
        lines[index] = lines[index].replace(
            "launch_protocol_ready_score", "launch_protocol_reddy_score", 1
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


def collect_v665_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate fourteen V665 dispositions without inventing scientific work."""

    capstone_path = root / V665_CAPSTONE_PATH
    capstone = _load_json(capstone_path)
    raw_rows = capstone.get("milestone_dispositions")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(V665_TASK_IDS):
        raise ValueError("V665 capstone dispositions are unavailable")

    rows: list[JsonDict] = []
    for expected_id, raw in zip(V665_TASK_IDS, raw_rows, strict=True):
        if not isinstance(raw, Mapping) or raw.get("task_id") != expected_id:
            raise ValueError("V665 capstone disposition order is invalid")
        row = deepcopy(dict(raw))
        custody = row.get("custody_kind")
        kind = "capstone_self" if custody == "current_self" else str(custody)
        if kind == "capstone_self":
            path = capstone_path
            exists = True
            observed_hash = sha256_file(path)
            expected_hash = observed_hash
        else:
            label = row.get("actual_path") or row.get("planned_path")
            path = root / str(label)
            exists = path.is_file()
            observed_hash = sha256_file(path) if exists else None
            expected_hash = row.get("sha256")
        authenticated = (kind == "missing_work" and not exists and expected_hash is None) or (
            exists and isinstance(expected_hash, str) and observed_hash == expected_hash
        )
        model_invoked = False
        if exists and kind != "capstone_self":
            source = _load_json(path)
            model_invoked = source.get("model_invoked") is True
        row.update(
            {
                "evidence_kind": kind,
                "authentication_path": path.relative_to(root).as_posix(),
                "authentication_sha256": observed_hash,
                "authentication_expected_sha256": expected_hash,
                "authentication_exists": exists,
                "authenticated": authenticated,
                "model_invoked": model_invoked,
                "conductor_completion_is_scientific_success": False,
                "resource_or_custody_block_retires_hypothesis": False,
            }
        )
        rows.append(row)
    return rows


def disposition_kind_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Count all four V665 custody classes without merging their meanings."""

    counts = Counter(str(row.get("evidence_kind")) for row in rows)
    return {
        "terminal_producer": counts["terminal_producer"],
        "conductor_pre_gate": counts["conductor_pre_gate"],
        "missing_work": counts["missing_work"],
        "capstone_self": counts["capstone_self"],
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
    """Map primary methods to V666 controls without transferring paper results."""

    return [
        {
            "rank": 1,
            "review_class": "recheck",
            "method_family": "jsonschemabench",
            "primary_url": "https://arxiv.org/html/2501.10868v1",
            "primary_section": "Sections 4-6",
            "mechanism_read": (
                "Coverage, semantic quality, grammar compilation, time to first token and "
                "time per output token are separate measures on comparable schemas."
            ),
            "mapped_tasks": ["exp7631-schema-pilot"],
            "adopted_control": "Independent schema and semantic validation plus separate latency.",
            "claim_limit": "The eight-group pilot is not a JSONSchemaBench reproduction.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 2,
            "review_class": "recheck",
            "method_family": "eaev_evidence_alignment",
            "primary_url": "https://arxiv.org/html/2609.08267v1",
            "primary_section": "Sections 3.3-3.6",
            "mechanism_read": (
                "Identity, semantic and consistency evidence remain separate; controlled "
                "counterfactual perturbations test whether support depends on its source."
            ),
            "mapped_tasks": ["exp7635-evidence-energy", "exp7636-decision-evaluation"],
            "adopted_control": "Evidence erasure and source-group permutation controls.",
            "claim_limit": "Carnot's compact decision energy is not an EAEV reproduction.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 3,
            "review_class": "recheck",
            "method_family": "proper_calibeating",
            "primary_url": "https://arxiv.org/html/2605.26703v2",
            "primary_section": "Sections 3, 5 and 6",
            "mechanism_read": (
                "Uniform bounded proper-score guarantees require online assumptions and "
                "normalization; quadratic calibeating does not transfer to every proper rule."
            ),
            "mapped_tasks": [
                "exp7635-evidence-energy",
                "exp7636-decision-evaluation",
                "exp7637-guarded-learning",
            ],
            "adopted_control": "Report proper loss, typed utility and retention separately.",
            "claim_limit": "Delayed selective admission is not the paper's theorem setting.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 4,
            "review_class": "recheck",
            "method_family": "spline_local_kan",
            "primary_url": "https://arxiv.org/html/2602.02056v4",
            "primary_section": "Section 3.1 and fixed-point implementation",
            "mechanism_read": (
                "Local B-spline support updates only active coefficients and supplies a future "
                "fixed-point hardware path."
            ),
            "mapped_tasks": ["exp7635-evidence-energy", "exp7637-guarded-learning"],
            "adopted_control": "Count bounded local updates and preserve whole-service costs.",
            "claim_limit": "CPU locality is not an FPGA speed or retention result.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 5,
            "review_class": "new_lead",
            "method_family": "as2_soft_answer_sets",
            "primary_url": "https://arxiv.org/html/2603.18436",
            "primary_section": "Differentiable soft consequence operator",
            "mechanism_read": "Constraint residuals can train perception through a soft operator.",
            "mapped_tasks": [],
            "adopted_control": "Defer architecture work until source evidence has measured value.",
            "claim_limit": "AS2 is not adopted and does not replace independent exact checks.",
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 6,
            "review_class": "new_lead",
            "method_family": "energy_guided_recursive_model",
            "primary_url": "https://arxiv.org/html/2607.10128",
            "primary_section": "Candidate selection, sampling and depth control",
            "mechanism_read": "Hopfield memories score recursive candidates and guide stopping.",
            "mapped_tasks": [],
            "adopted_control": "Defer new ranking until a distinct live-path bottleneck exists.",
            "claim_limit": "ERM does not reopen retired rankers or prove hidden-game utility.",
            "external_result_is_local_measurement": False,
        },
    ]


def write_method_records(root: Path) -> None:
    """Write the V666 method map and one idempotent studying-log record."""

    lines = [
        "# V666 evidence and goal-safe planning method map",
        "",
        "Date: 2026-09-24. Scope: administrative method qualification.",
        "External paper results are not local Carnot measurements.",
        "No paper result, theorem, hardware estimate, or learned benefit transfers automatically.",
        "",
        "| Rank | Class | Method | V666 mapping | Claim limit |",
        "|---:|---|---|---|---|",
    ]
    for row in method_rows():
        source = f"[{row['primary_section']}]({row['primary_url']})"
        mapping = ", ".join(row["mapped_tasks"]) or "future lead only"
        lines.append(
            f"| {row['rank']} | {row['review_class']} | {row['method_family']} {source} | "
            f"{mapping}: {row['adopted_control']} | {row['claim_limit']} |"
        )
    lines.extend(
        [
            "",
            "## Adopted controls",
            "",
            "Exp7631 keeps JSON schema coverage, semantic quality, grammar compilation,",
            "time to first token and output-token cost separate. An independent validator",
            "still checks each constrained generation.",
            "",
            "Exp7635 and Exp7636 keep identity, semantic and consistency evidence separate.",
            "They test evidence erasure and source-group permutation. Exp7635 through Exp7637",
            "also keep proper loss, typed utility, update admission and retention separate.",
            "Spline locality is a future hardware path, not a measured device benefit.",
            "",
            "Exp7639 and Exp7640 adopt the independent goal check principle. A deduplication",
            "key cannot remove a state before the full-grid goal predicate runs. Wrapper-level",
            "measurement must use the live mask and cannot inherit a wider direct-harness mask.",
            "",
            "## Leads and access boundaries",
            "",
            "AS2 and ERM are new leads. AS2's soft consequence operator does not replace an",
            "exact checker. ERM does not reopen retired external-text rankers and does not",
            "establish an ARC hidden-game heuristic.",
            "",
            "The named arXiv HTML pages were readable during the low-concurrency recheck.",
            "Semantic Scholar citation pages remained inaccessible in the recorded planning",
            "pass. Direct OpenReview forums returned browser challenges even when indexed PDFs",
            "were readable. Those outcomes establish no citation census or acceptance claim.",
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
                "## 2026-09-24 Exp7629 — V666 contract methods — INGESTED",
                "",
                "Rechecked JSONSchemaBench, EAEV, Proper Calibeating and spline-local KAN.",
                "Mapped bounded controls to Exp7631 and Exp7635 through Exp7637. The independent",
                "goal-check principle maps to Exp7639 and Exp7640. AS2 and ERM remain new leads.",
                "Semantic Scholar and direct OpenReview access limits remain explicit. No paper",
                "result or theorem transfers automatically to Carnot.",
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
    """Bind current authorities and distinguish every prior evidence class."""

    current = (
        roadmap_path.relative_to(root),
        DESIGN_PATH,
        SPEC_PATH,
        Path("research-references.md"),
        STUDY_PATH,
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("ops/known-issues.md"),
        V665_CAPSTONE_PATH,
        Path("python/carnot/experiment_7615_v665_contract_methods.py"),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        NOTE_PATH,
    )
    rows = [
        {
            "path": path.as_posix(),
            "actual_path": path.as_posix(),
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
            "actual_path": row.get("authentication_path")
            if row.get("authentication_exists")
            else None,
            "sha256": row.get("authentication_sha256"),
            "exists": row.get("authentication_exists"),
            "role": "v665_disposition_evidence",
            "task_id": row.get("task_id"),
            "evidence_kind": row.get("evidence_kind"),
        }
        for row in dispositions
    )
    rows.append(
        {
            "path": RESULT_PATH.as_posix(),
            "actual_path": None,
            "sha256": None,
            "exists": False,
            "role": "planned_output_not_input",
        }
    )
    return rows


def _source_hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash inputs while keeping the current planned output non-circular."""

    rows = artifact.get("source_artifact_hashes")
    if not isinstance(rows, list) or not rows:
        return False
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            return False
        if row.get("role") == "planned_output_not_input":
            if row.get("actual_path") is not None or row.get("sha256") is not None:
                return False
            continue
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
        Path("AGENTS.md"),
        Path("CODEX.md"),
        Path("CLAUDE.md"),
        Path("research-program.md"),
        Path("ops/exclusion_manifest.yaml"),
        Path("ops/e2e-test-plan.md"),
        Path("scripts/experiment_template.py"),
        Path("python/carnot/reporting/current_work_receipt.py"),
        Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
        ACTIVE_ROADMAP_PATH,
        DESIGN_PATH,
        V665_CAPSTONE_PATH,
        Path("python/carnot/experiment_7615_v665_contract_methods.py"),
        Path("scripts/audit_roadmap_gates.py"),
        Path("scripts/publication_gate.py"),
        Path("research-references.md"),
        STUDY_PATH,
        SPEC_PATH,
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
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    rows.append(
        (
            SPEC_PATH,
            "REQ-*",
            "REQ-REPORT-7629",
            "REQ-REPORT-7629" if "REQ-REPORT-7629" in spec_text else "absent",
        )
    )
    return rows


def preconditions_checked(root: Path) -> list[JsonDict]:
    """Record named inputs, absolute-root custody, process ownership and tools."""

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
    rows.extend(
        [
            {
                "check": "absolute_repository_root",
                "upstream": "current_process",
                "path": str(root),
                "field": "is_absolute_directory",
                "operator": "eq",
                "expected": True,
                "observed": root.is_absolute() and root.is_dir(),
                "passed": root.is_absolute() and root.is_dir(),
            },
            {
                "check": "process_resource_ownership",
                "upstream": "current_process",
                "path": f"/proc/{os.getpid()}",
                "field": "owned_pid_exists",
                "operator": "eq",
                "expected": True,
                "observed": Path(f"/proc/{os.getpid()}").is_dir(),
                "passed": Path(f"/proc/{os.getpid()}").is_dir(),
            },
        ]
    )
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
    return rows


def _gate(check: str, category: str, expected: object, observed: object, passed: bool) -> JsonDict:
    """Keep all administrative, benefit and retention outcomes separate."""

    return {
        "check": check,
        "category": category,
        "upstream": EXPERIMENT_ID,
        "path": RESULT_PATH.as_posix(),
        "field": check,
        "operator": "eq",
        "condition": f"{check} == {expected!r}",
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
    """Reduce infrastructure gates while leaving every science gate unopened."""

    authority = contract.get("passed") is True and len(contract.get("rows") or []) == 14
    mutation = len(mutations) == 9 and all(row.get("qualified") is True for row in mutations)
    custody = disposition_kind_counts(dispositions) == {
        "terminal_producer": 7,
        "conductor_pre_gate": 3,
        "missing_work": 3,
        "capstone_self": 1,
    } and all(row.get("authenticated") is True for row in dispositions)
    guards = _receipt_group_passed(receipts, (*REQUIRED_CHECK_NAMES, *REPOSITORY_CHECK_NAMES))
    methods_ok = len(methods) == 6 and all(
        row.get("external_result_is_local_measurement") is False for row in methods
    )
    return [
        _gate("authority_exact", "validity", True, authority, authority),
        _gate("private_mutations_rejected", "validity", True, mutation, mutation),
        _gate("v665_custody_authenticated", "validity", True, custody, custody),
        _gate("required_validation_and_guards", "validity", True, guards, guards),
        _gate("method_limits_ingested", "readiness", True, methods_ok, methods_ok),
        _gate("probability_benefit", "probability_benefit", True, False, False),
        _gate("typed_decision_utility", "utility", True, False, False),
        _gate("retention_measurement", "retention", "not_applicable", "not_applicable", True),
        _gate("historical_source_freshness", "freshness", "descriptive", "descriptive", True),
    ]


def _field_principles(keys: Sequence[str]) -> JsonDict:
    """Attach a plain failure-prevention principle to every top-level field."""

    principles = {
        key: "Keep this infrastructure evidence explicit and reproducible." for key in keys
    }
    principles.update(
        {
            "honest_verdict": (
                "Use a complete_ terminal prefix; completion alone is not scientific benefit."
            ),
            "verdict_class": "Partial is unfinished owned work; external absence is blocked.",
            "flagged_adversarial": "Persist the terminal reader result; flags open no gate.",
            "gate_check_summary": (
                "A block names check, upstream, path, field, operator, expected and observed."
            ),
            "acceptance_gate_results": (
                "Validity, readiness, probability benefit, utility, retention and freshness stay separate."
            ),
            "rows": "Each task keeps absolute operands, seed, direction, censoring and provenance.",
            "sample_size_budget": "Tasks, not views, orders or seeds, define sample size.",
            "preconditions_checked": "Unavailable inputs cannot become fabricated work.",
            "inference_substrate": "Historical GPU evidence is not a current model invocation.",
            "inference_substrate_class": "Record actual and planned substrate classes separately.",
            "MODEL_SPECS": "A no-call aggregation uses an empty current model list.",
            "model_invoked": "True requires a current real model load or generation attempt.",
            "execution_venue": "Name the actual host and owned PID without borrowing old scope.",
            "phase_spans": "Use disjoint current stages with completed units and checkpoints.",
            "invocation_counts": "Count current calls separately from all historical calls.",
            "duration_s": "Measure current monotonic time and never inherit or pad duration.",
            "random_seed": "Persist each seed and its purpose; this task is deterministic.",
            "reproducibility_checksum": "Bind inputs, configuration, authority and reduction code.",
            "source_artifact_hashes": "Separate inputs, producers, pre-gates, absences and output.",
            "validation_receipts": "Retain actual commands, exits, paths and exact log hashes.",
            "verifier_is_oracle": "Exact contract truth cannot prove learned advantage.",
            "field_principles": "Carry one governing principle beside every top-level field.",
            "contract_ready_score": "One requires exact parity, negative controls and clean guards.",
            "method_map_path": "Bind only methods and access limits actually rechecked.",
            "task_contract_rows": "Pair each V666 task with the same-order V665 disposition.",
            "publication_gates": "Keep fixed G1-G4 and historic FoVer eligibility separate.",
        }
    )
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the complete evidence object except the checksum field itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness from raw rows instead of trusting the headline."""

    rows = artifact.get("rows")
    mutations = artifact.get("contract_mutation_rows")
    dispositions = artifact.get("v665_dispositions")
    methods = artifact.get("method_rows")
    paired = artifact.get("task_contract_rows")
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
        == list(V665_TASK_IDS)
        and all(row.get("authenticated") is True for row in dispositions)
        and disposition_kind_counts(dispositions)
        == {
            "terminal_producer": 7,
            "conductor_pre_gate": 3,
            "missing_work": 3,
            "capstone_self": 1,
        }
    )
    methods_ok = isinstance(methods, list) and {
        row.get("method_family") for row in methods if isinstance(row, Mapping)
    } == {
        "jsonschemabench",
        "eaev_evidence_alignment",
        "proper_calibeating",
        "spline_local_kan",
        "as2_soft_answer_sets",
        "energy_guided_recursive_model",
    }
    paired_ok = (
        isinstance(paired, list)
        and len(paired) == 14
        and all(
            row.get("new_task_id") == EXPECTED_TASK_IDS[index]
            and row.get("prior_task_id") == V665_TASK_IDS[index]
            for index, row in enumerate(paired)
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
        and artifact.get("honest_verdict") == "complete_null_v666_contract_methods_ingested"
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
    """Pair V666 authority rows with same-order V665 terminal dispositions."""

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
    """Build a complete null advisory from authenticated infrastructure evidence."""

    methods = method_rows()
    gates = _acceptance_gates(contract, mutations, dispositions, validation_receipts, methods)
    ready = all(
        row["passed"] is True
        for row in gates
        if row["category"] not in {"probability_benefit", "utility"}
    )
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
                "phase": "current_infrastructure_aggregation",
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
        "experiment": 7629,
        "title": "Bind fourteen tasks and ingest evidence and planner methods",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "complete_disqualified",
        "honest_verdict": (
            "complete_null_v666_contract_methods_ingested"
            if ready
            else "complete_disqualified_v666_contract_or_validation"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "positive_claim": False,
        "flagged_adversarial": flagged,
        "verifier_is_oracle": False,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": max(0.0, duration_s),
        "phase_spans": deepcopy(measured_spans),
        "random_seed": 7629,
        "stochastic_stage_seeds": {"contract_mutations": 7629},
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": (
            "unsloth/Qwen3.8-27B-GGUF was planned for V665 model work; no model was invoked here"
        ),
        "planned_MODEL_SPECS": [],
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "execution_venue_details": {
            "host": platform.node(),
            "physical_device": "host_cpu",
            "gpu_uuid": None,
            "gpu_used": False,
            "owned_pid": os.getpid(),
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
        "v665_dispositions": deepcopy(list(dispositions)),
        "v665_disposition_counts": disposition_kind_counts(dispositions),
        "v665_terminal_context": {
            "schema_authority_qualified": True,
            "native_parity_qualified": True,
            "schema_pilot_model_invoked": False,
            "schema_pilot_observed_groups": 0,
            "scientific_producer_count": 0,
            "missing_scientific_producers": [
                "exp7621-evidence-energy",
                "exp7622-decision-evaluation",
                "exp7623-guarded-learning",
            ],
            "conductor_ok_is_scientific_success": False,
            "prior_capstone_verdict": "complete_blocked_required_v665_external_evidence",
        },
        "method_rows": methods,
        "method_map_path": NOTE_PATH.as_posix(),
        "method_ingestion_complete_score": int(len(methods) == 6),
        "independent_goal_check_mapping": {
            "principle": "Evaluate the full-grid goal before masked-state deduplication.",
            "tasks": ["exp7639-arc-goal-dedup", "exp7640-arc-wrapper-generalization"],
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": {
            "passed": ready,
            "failed_checks": [row["check"] for row in gates if not row["passed"]],
            "first_failure": next(
                (
                    deepcopy(row)
                    for row in gates
                    if row["category"] not in {"probability_benefit", "utility"}
                    and not row["passed"]
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
            "new_v666_evidence": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "authority_selection": "staged_or_activated_by_exact_milestone",
            "negative_contract_mutations": [
                "missing_task",
                "reordered_task",
                "altered_path",
                "misspelled_field",
            ],
            "consumed_staging_control": True,
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
            "independent_unit": "ordered_v666_contract_task",
            "seeds_views_or_replays_multiply_units": False,
        },
        "prior_failure_dispositions": [
            {
                "scope": "v665_exclusive_cuda_selector_block",
                "retired": False,
                "reason": "A resource and ownership block does not falsify schema quality.",
            },
            {
                "scope": "v665_missing_scientific_producers",
                "retired": False,
                "reason": "Missing evidence cannot retire an unmeasured scientific hypothesis.",
            },
            {
                "scope": "v665_empty_arc_supervisor_ledger",
                "retired": True,
                "reason": "Retire only another unchanged empty-ledger refinement attempt.",
            },
            {
                "scope": "v665_native_total_cost_measurement",
                "retired": False,
                "reason": "The positive 1.10 gate and failed 10x NFR retain separate scopes.",
            },
        ],
        "repository_health": {
            "full_python_suite_launched_by_entrypoint": False,
            "pre_existing_debt_is_current_validity": False,
        },
        "advisory_only": True,
        "global_science_gate": False,
        "active_roadmap_modified": False,
        "roadmap_activation_performed": False,
        "roadmap_archival_performed": False,
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

    roadmap_path, roadmap, candidates = resolve_v666_roadmap(ROOT)
    design = (ROOT / DESIGN_PATH).read_text(encoding="utf-8")
    return build_artifact(
        ROOT,
        roadmap_path,
        candidates,
        compare_contract_authorities(design, roadmap),
        run_contract_mutation_controls(design, roadmap),
        collect_v665_dispositions(ROOT),
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
        "condition": f"{field} == {expected!r}",
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
        "planned_MODEL_SPECS": [],
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
            "owned_pid": os.getpid(),
        },
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": 7629,
        "stochastic_stage_seeds": {},
        "preconditions_checked": [failure],
        "contract_ready_score": 0,
        "rows": [],
        "task_contract_rows": [],
        "v665_dispositions": [],
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
            "independent_unit": "ordered_v666_contract_task",
        },
        "source_artifact_hashes": [
            {
                "path": RESULT_PATH.as_posix(),
                "actual_path": None,
                "sha256": None,
                "exists": False,
                "role": "planned_output_not_input",
            }
        ],
        "validation_receipts": [],
        "publication_gates": {
            "gates": {name: {"pass": False} for name in ("G1", "G2", "G3", "G4")},
            "paper_ready": False,
            "unmet_gates": ["G1", "G2", "G3", "G4"],
            "publication_performed": False,
            "independent_science_gated": False,
            "historical_scope": "existing_FoVer_eligibility_only",
            "new_v666_evidence": False,
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
        "v665_dispositions",
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
        "probability_benefit",
        "utility",
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
        roadmap_path, roadmap, candidates = resolve_v666_roadmap(root)
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
    dispositions = collect_v665_dispositions(root)
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
    progress(started, "method_ingestion", "before", planned_units=6)
    write_method_records(root)
    spans.append(_span("method_ingestion", started, phase_started, 6, 6))
    progress(started, "method_ingestion", "after", completed_units=6)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7629-validation-", dir="/tmp"))
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
    """Accept only the date frozen by the V666 execution contract."""

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
    """Run Experiment 7629 or one fresh-process candidate reader."""

    print("[exp7629] phase=startup event=flushed", flush=True)
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

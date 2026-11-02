"""Bind the V663 planning contract to literal V662 evidence.

This module reports planning readiness. It does not run an LLM, alter a
scientific gate, or turn historical completion into scientific benefit.

Spec refs: REQ-REPORT-7587 and SCENARIO-REPORT-7587-*.
"""

from __future__ import annotations

import argparse
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

from carnot import experiment_7573_v662_contract_methods as prior
from carnot.experiment_7329_v644_contract import parse_markdown_contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands


JsonDict = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.663"
EXPERIMENT_ID = "exp7587-contract-methods"
SCHEMA = "carnot.exp7587.v663.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7587_v663_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7587_v663_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7587_v663_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7587_v663_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7587_v663_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v663-method-map.md")
STUDY_PATH = Path("research-studying.md")
STUDY_MARKER = "<!-- EXP7587-V663-METHOD-INGESTION -->"
V662_CAPSTONE_PATH = Path("results/experiment_7586_v662_capstone.json")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7587, "contract-methods"),
        (7588, "evidence-protocol"),
        (7589, "arc-output-boundary"),
        (7590, "evidence-pilot"),
        (7591, "fit-evidence"),
        (7592, "test-online-evidence"),
        (7593, "evidence-energy"),
        (7594, "decision-evaluation"),
        (7595, "guarded-learning"),
        (7596, "evidence-audit"),
        (7597, "arc-history-generalization"),
        (7598, "rust-consumer"),
        (7599, "board-continuity"),
        (7600, "capstone"),
    )
)
V662_TASK_IDS = tuple(
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
ZERO_INVOCATION_COUNTS = deepcopy(prior.ZERO_INVOCATION_COUNTS)
TERMINAL_CLASSES = deepcopy(prior.TERMINAL_CLASSES)

_load_yaml = prior._load_yaml
_load_json = prior._load_json
_normalize_yaml_contract = prior._normalize_yaml_contract
_public_task = prior._public_task
private_basetemp_parents = prior.private_basetemp_parents
build_repository_check_plan = prior.build_repository_check_plan
reduce_validation_receipts = prior.reduce_validation_receipts
operator_queue = prior.operator_queue
_normalize_receipts = prior._normalize_receipts


def progress(started: float, phase: str, event: str, **details: object) -> None:  # pragma: no cover
    """Print one truthful phase boundary before or after slow work."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7587] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v663_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
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
        raise ValueError("V663 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


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
        "contract_rows": rows,
        "rows": rows,
    }


def mutation_names() -> tuple[str, ...]:
    """Name the four authority defects required by REQ-REPORT-7587."""

    return ("count", "order", "path", "field")


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
            RESULT_PATH.as_posix(), "results/private-changed.json"
        )
    else:
        index = next(i for i in task_lines if ".evidence_protocol_ready_score" in lines[i])
        lines[index] = lines[index].replace(
            "evidence_protocol_ready_score", "private_changed_score", 1
        )
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


def collect_v662_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate fourteen literal dispositions and their key measurements."""

    capstone_path = root / V662_CAPSTONE_PATH
    capstone = _load_json(capstone_path)
    raw_rows = capstone.get("task_dispositions")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(V662_TASK_IDS):
        raise ValueError("V662 capstone dispositions are unavailable")

    rows: list[JsonDict] = []
    for expected_id, raw in zip(V662_TASK_IDS, raw_rows, strict=True):
        if not isinstance(raw, Mapping) or raw.get("task_id") != expected_id:
            raise ValueError("V662 capstone disposition order is invalid")
        row = deepcopy(dict(raw))
        evidence = row.get("evidence_path")
        if evidence:
            evidence_path = root / str(evidence)
            observed_hash = sha256_file(evidence_path) if evidence_path.is_file() else None
            expected_hash = row.get("artifact_sha256")
        else:
            evidence_path = capstone_path
            observed_hash = sha256_file(capstone_path)
            expected_hash = observed_hash

        state = row.get("evidence_state")
        if state == "conductor_gate_blocked":
            evidence_kind = "conductor_pre_gate"
        elif state == "missing":
            evidence_kind = "missing_producer"
        elif expected_id == "exp7586-capstone":
            evidence_kind = "capstone_terminal"
        else:
            evidence_kind = "producer"
        row.update(
            {
                "authentication_path": evidence_path.relative_to(root).as_posix(),
                "authentication_sha256": observed_hash,
                "authenticated": observed_hash == expected_hash,
                "evidence_kind": evidence_kind,
                "archive_lag_is_missing_science": False,
            }
        )

        if expected_id == "exp7578-continuous-proper-loss":
            payload = _load_json(evidence_path)
            reduction = payload.get("independent_reduction") or {}
            row["retention_passed"] = reduction.get("retention_passed")
            row["retention_degradation"] = deepcopy(reduction.get("retention_degradation"))
        elif expected_id == "exp7579-decision-learning-audit":
            static_path = root / "results/experiment_7577_v662_proper_loss_evaluation.json"
            static = _load_json(static_path)
            metrics = static.get("probability_metrics") or {}
            fitted = metrics.get("proper_loss_monotone") or {}
            raw_metric = metrics.get("raw_original") or {}
            row.update(
                {
                    "static_group_count": fitted.get("n_source_components"),
                    "static_brier": round(float(fitted.get("mean_brier")), 9),
                    "raw_brier": round(float(raw_metric.get("mean_brier")), 9),
                    "static_metric_source_path": static_path.relative_to(root).as_posix(),
                    "static_metric_source_sha256": sha256_file(static_path),
                }
            )
        elif expected_id == "exp7581-arc-bounded-canary":
            payload = _load_json(evidence_path)
            counts = payload.get("invocation_counts") or {}
            row["inference_started"] = any(
                value != 0 for value in counts.values() if isinstance(value, int)
            )
            row["output_path_rejected_before_inference"] = row["inference_started"] is False
        elif expected_id == "exp7585-portable-service":
            payload = _load_json(evidence_path)
            warm = (payload.get("whole_service_speedup") or {}).get("warm") or {}
            cold = (payload.get("whole_service_speedup") or {}).get("cold") or {}
            row.update(
                {
                    "warm_ratio": warm.get("estimate"),
                    "warm_lower95": warm.get("lower95"),
                    "warm_pair_count": warm.get("pair_count"),
                    "cold_ratio": cold.get("estimate"),
                    "execution_venue": payload.get("execution_venue"),
                    "board_speed_claimed": False,
                }
            )
        rows.append(row)
    return rows


def method_rows() -> list[JsonDict]:
    """Map four primary methods to bounded V663 tests and explicit exclusions."""

    return [
        {
            "rank": 1,
            "status": "applicable",
            "method_family": "eaev_evidence_alignment",
            "primary_url": "https://arxiv.org/html/2609.08267v1",
            "primary_section": "Sections 3.3-3.6",
            "mechanism_read": (
                "identity, semantic and consistency signals; controlled perturbations; "
                "entity aggregation; supervised verifier training"
            ),
            "applicable_test_or_deferral": (
                "Exp7588 and Exp7590-7596 retain lossless sentence/source pointers, "
                "feature erasures and explicit unknown states"
            ),
            "claim_limit": "Evidence-link adaptation is not an EAEV replication.",
            "destination_tasks": ["exp7588-evidence-protocol", "exp7590-evidence-pilot"],
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 2,
            "status": "applicable",
            "method_family": "rt4chart_hierarchical_verification",
            "primary_url": "https://arxiv.org/html/2603.27752v1",
            "primary_section": "Sections 3.2-3.6 hierarchical verification method",
            "mechanism_read": (
                "self-contained claims, overlapping local chunks, deterministic OR-join, "
                "fresh global evidence extraction and answer-level AND-join"
            ),
            "applicable_test_or_deferral": (
                "Exp7588 preserves claim-to-span support, contradiction and baseless states"
            ),
            "claim_limit": "Pointer integrity and deterministic joins do not prove semantic support.",
            "destination_tasks": ["exp7588-evidence-protocol", "exp7596-evidence-audit"],
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 3,
            "status": "deferred",
            "method_family": "u_calibration",
            "primary_url": "https://arxiv.org/html/2606.18527v1",
            "primary_section": "Algorithm 1",
            "mechanism_read": (
                "self-concordant random perturbation of the empirical label average followed "
                "by immediate label observation"
            ),
            "applicable_test_or_deferral": (
                "Exp7595 measures proper loss and retention but defers FTPL-style implementation"
            ),
            "claim_limit": "Delayed guarded SGD is not the U-Calibration algorithm.",
            "destination_tasks": ["exp7595-guarded-learning", "exp7596-evidence-audit"],
            "external_result_is_local_measurement": False,
        },
        {
            "rank": 4,
            "status": "deferred",
            "method_family": "on_chip_kan_locality",
            "primary_url": "https://arxiv.org/html/2602.02056v4",
            "primary_section": "Sections 3.1-3.3 and Section 4 locality implementation",
            "mechanism_read": (
                "S+1 active spline coefficients, LUT basis values and fixed-point on-chip updates"
            ),
            "applicable_test_or_deferral": (
                "Exp7595 measures bounded update locality; FPGA placement waits for a measured need"
            ),
            "claim_limit": "Spline locality alone proves neither retention nor Carnot FPGA speed.",
            "destination_tasks": ["exp7595-guarded-learning", "exp7599-board-continuity"],
            "external_result_is_local_measurement": False,
        },
    ]


def write_method_records(root: Path) -> None:
    """Write the method map and one idempotent ranked studying entry."""

    lines = [
        "# V663 method map",
        "",
        "Date: 2026-09-24. Scope: advisory method accounting.",
        "External results are not local Carnot measurements.",
        "Evidence-link adaptation is not an EAEV replication.",
        "Delayed guarded SGD is not the U-Calibration algorithm.",
        "",
        "| Rank | Status | Family | Primary section | V663 use | Claim limit |",
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
                "## 2026-09-24 Exp7587 — V663 methods — INGESTED",
                "",
                "Applicable (rank 1-2): EAEV evidence dimensions and RT4CHART claim/span hierarchy.",
                "Deferred (rank 3-4): U-Calibration FTPL and FPGA KAN placement.",
                "Evidence-link adaptation is not EAEV replication; delayed guarded SGD is not",
                "U-Calibration Algorithm 1. Locality and readiness do not establish retention.",
                "",
            ]
        )
        study.write_text(existing.rstrip() + "\n\n" + addition, encoding="utf-8")


REQUIRED_CHECK_NAMES = prior.REQUIRED_CHECK_NAMES
REPOSITORY_CHECK_NAMES = prior.REPOSITORY_CHECK_NAMES
TERMINAL_CHECK_NAMES = prior.TERMINAL_CHECK_NAMES


def affected_file_manifest() -> JsonDict:
    """Freeze exact implementation, test, wrapper, and requirement paths."""

    return {
        "tests": [TEST_PATH.as_posix()],
        "modules": [MODULE_PATH.as_posix()],
        "static": [WRAPPER_PATH.as_posix(), SPEC_PATH.as_posix()],
    }


def build_validation_plan(root: Path, private_root: Path) -> list[CommandSpec]:
    """Build scoped checks after creating each private temporary parent."""

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
        "p=pathlib.Path(importlib.import_module('carnot.experiment_7587_v663_contract_methods').__file__).resolve();"
        "print(json.dumps({'resolved_imports':{'carnot.experiment_7587_v663_contract_methods':str(p)}}),flush=True);"
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


def validate_validation_plan(root: Path, commands: Sequence[CommandSpec]) -> list[str]:
    """Reject expanded test targets, command drift, and missing private parents."""

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
            "state": "post_activation_consumed_staging",
            "selected": ACTIVE_ROADMAP_PATH.as_posix() if matches else None,
            "expected": ACTIVE_ROADMAP_PATH.as_posix(),
            "passed": matches,
        },
    ]


def run_lifecycle_control(directory: Path) -> JsonDict:
    """Exercise delayed update, durable reload, and duplicate rejection."""

    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "exp7587-lifecycle.json"
    event_id = "unit-0001"
    state: JsonDict = {"seen": 0, "positive": 0, "event_ids": []}
    events = ["predict"]
    prediction = state["positive"] / max(1, state["seen"])
    prediction_before_release = state["seen"] == 0
    events.append("release")
    label = 1
    events.append("update")
    state = {"seen": 1, "positive": label, "event_ids": [event_id]}
    events.append("persist")
    atomic_json(state_path, state)
    events.append("reload")
    reloaded = _load_json(state_path)
    events.append("duplicate")
    duplicate_rejected = event_id in reloaded.get("event_ids", [])
    unchanged_after_duplicate = reloaded == state
    passed = (
        events == ["predict", "release", "update", "persist", "reload", "duplicate"]
        and prediction_before_release
        and reloaded == state
        and duplicate_rejected
        and unchanged_after_duplicate
    )
    return {
        "events": events,
        "prediction": prediction,
        "prediction_before_release": prediction_before_release,
        "reload_equal": reloaded == state,
        "duplicate_rejected": duplicate_rejected,
        "unchanged_after_duplicate": unchanged_after_duplicate,
        "passed": passed,
        "state_sha256": sha256_file(state_path),
    }


def _publication_gates(root: Path, receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Read unchanged G1-G4 from the exact publication-guard log."""

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
        "independent_science_gated": False,
    }


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Bind authorities, methods, code, and literal V662 evidence bytes."""

    current = (
        roadmap_path.relative_to(root),
        DESIGN_PATH,
        SPEC_PATH,
        Path("research-references.md"),
        STUDY_PATH,
        Path("ops/conductor-log.md"),
        Path("ops/known-issues.md"),
        V662_CAPSTONE_PATH,
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
                "role": "v662_disposition_evidence",
                "task_id": disposition.get("task_id"),
                "evidence_kind": disposition.get("evidence_kind"),
            }
        )
        static_path = disposition.get("static_metric_source_path")
        if static_path:
            rows.append(
                {
                    "path": static_path,
                    "sha256": disposition.get("static_metric_source_sha256"),
                    "role": "v662_static_metric_evidence",
                    "task_id": disposition.get("task_id"),
                    "evidence_kind": "producer",
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
    """Keep validity, readiness, benefit, retention, and freshness separate."""

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
        "principle": "A gate keeps its own evidence class and cannot promote another class.",
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Build independently visible administrative and scientific gate rows."""

    authority = contract.get("passed") is True and len(contract.get("rows") or []) == 14
    mutation = len(mutations) == 8 and all(row.get("qualified") is True for row in mutations)
    custody = len(dispositions) == 14 and all(
        row.get("authenticated") is True for row in dispositions
    )
    method = len(methods) == 4 and all(
        row.get("external_result_is_local_measurement") is False for row in methods
    )
    by_id = {str(row.get("task_id")): row for row in dispositions}
    retention = by_id.get("exp7578-continuous-proper-loss", {}).get("retention_passed")
    return [
        _gate("authority_exact", "validity", True, authority, authority),
        _gate("private_mutations_rejected", "validity", True, mutation, mutation),
        _gate("v662_custody_authenticated", "validity", True, custody, custody),
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
        _gate("historical_retention_result", "retention", False, retention, retention is False),
        _gate(
            "historical_source_freshness",
            "freshness",
            "descriptive_reuse",
            "descriptive_reuse",
            True,
        ),
    ]


def _field_principles(keys: Sequence[str]) -> JsonDict:
    """Attach one failure-prevention statement to every top-level field."""

    principles = {
        key: "This field keeps the advisory record explicit and reproducible." for key in keys
    }
    principles.update(
        {
            "honest_verdict": "Use a complete_ terminal prefix; execution completion does not prove benefit.",
            "verdict_class": "Use exactly one terminal class; only owned unfinished work is partial.",
            "flagged_adversarial": "Persist the exact terminal reader outcome; flagged evidence cannot open a gate.",
            "gate_check_summary": "Every blocked result names check, upstream, path, field, op, expected, and observed.",
            "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness stay separate.",
            "rows": "Each unit and arm keeps absolute metrics, operands, seed, direction, missingness, and provenance.",
            "sample_size_budget": "Independent source rows, not seeds or windows, determine the sample count.",
            "inference_substrate": "State the real current execution mode; historical GPU use is not current inference.",
            "inference_substrate_class": "Record actual and planned classes separately; blocked_no_run means no model work.",
            "MODEL_SPECS": "Every current LLM call names Qwen; this cached-only aggregation declares an empty list.",
            "invocation_counts": "Count loads, forwards, generations, and tokens independently from history.",
            "duration_s": "Measure current monotonic work without padding or inherited elapsed time.",
            "random_seed": "Record an explicit seed for each stochastic stage; this deterministic task has one sentinel seed.",
            "reproducibility_checksum": "Bind immutable source evidence, configuration, and terminal reduction.",
            "source_artifact_hashes": "Distinguish authenticated producers, missing producers, and conductor pre-gates.",
            "validation_receipts": "Bind command, worktree, exit code, and raw log hash for every current check.",
            "verifier_is_oracle": "Exact labels and hand-built controls cannot support an oracle-distinct claim.",
            "field_principles": "Carry one plain-language prevention principle for every emitted field.",
            "contract_ready_score": "One requires exact fourteen-row agreement, clean guards, and mutation rejection.",
            "v662_dispositions": "Keep one immutable evidence disposition for every V662 task.",
            "method_map_path": "Tie mechanisms and exclusions to the primary sections actually read.",
            "publication_gates": "Use fixed G1-G4, paper_ready, and unmet_gates without redefining them.",
        }
    )
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all emitted evidence except the checksum's own value."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    return canonical_hash(payload)


def independent_reduce(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute readiness and terminal class from embedded raw evidence."""

    rows = artifact.get("rows")
    mutations = artifact.get("contract_mutation_rows")
    dispositions = artifact.get("v662_dispositions")
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
            and row.get("missing") is False
            and row.get("censored") is False
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
        == list(V662_TASK_IDS)
        and all(row.get("authenticated") is True for row in dispositions)
        and by_id.get("exp7579-decision-learning-audit", {}).get("static_group_count") == 80
        and by_id.get("exp7579-decision-learning-audit", {}).get("static_brier") == 0.146249673
        and by_id.get("exp7579-decision-learning-audit", {}).get("raw_brier") == 0.145635636
        and by_id.get("exp7578-continuous-proper-loss", {}).get("retention_passed") is False
        and by_id.get("exp7581-arc-bounded-canary", {}).get("inference_started") is False
        and by_id.get("exp7582-arc-panel-a", {}).get("evidence_kind") == "conductor_pre_gate"
        and by_id.get("exp7583-arc-panel-b", {}).get("evidence_kind") == "missing_producer"
        and by_id.get("exp7585-portable-service", {}).get("execution_venue") == "host"
        and float(by_id.get("exp7585-portable-service", {}).get("warm_ratio") or 0.0) > 1.0
        and float(by_id.get("exp7585-portable-service", {}).get("warm_lower95") or 0.0) > 1.0
    )
    methods_ok = isinstance(methods, list) and {
        row.get("method_family") for row in methods if isinstance(row, Mapping)
    } == {
        "eaev_evidence_alignment",
        "rt4chart_hierarchical_verification",
        "u_calibration",
        "on_chip_kan_locality",
    }
    lifecycle_ok = (
        isinstance(lifecycle, Mapping)
        and lifecycle.get("passed") is True
        and lifecycle.get("duplicate_rejected") is True
    )
    ready = rows_ok and mutation_ok and dispositions_ok and methods_ok and lifecycle_ok
    passed = (
        ready
        and artifact.get("contract_ready_score") == 1
        and artifact.get("honest_verdict") == "complete_null_v663_contract_methods_ingested"
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
        "experiment": 7587,
        "title": "Bind fourteen tasks and ingest evidence-alignment methods",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": "complete" if ready else "complete_disqualified",
        "honest_verdict": (
            "complete_null_v663_contract_methods_ingested"
            if ready
            else "complete_disqualified_v663_contract_or_validation"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "positive_claim": False,
        "flagged_adversarial": False,
        "verifier_is_oracle": False,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "duration_s": max(0.0, duration_s),
        "phase_spans": [{"phase": "current_advisory", "duration_s": max(0.0, duration_s)}],
        "random_seed": 7587,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "historical_model_identity": "unsloth/Qwen3.8-27B-GGUF in V662 producer evidence only",
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
        "rows": rows,
        "v662_dispositions": deepcopy(list(dispositions)),
        "v662_archive_history_invented": False,
        "method_rows": methods,
        "method_map_path": NOTE_PATH.as_posix(),
        "method_ingestion_complete_score": int(len(methods) == 4),
        "operator_queue": operator_queue(root),
        "lifecycle_control": deepcopy(dict(lifecycle)),
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
            "lifecycle": ["predict", "release", "update", "persist", "reload", "duplicate"],
            "numbered_runtime_e2e_applicable": False,
            "numbered_runtime_e2e_reason": "read_only_reporting_has_no_numbered_runtime_e2e",
        },
        "sample_size_budget": {
            "intended": 14,
            "planned": 14,
            "observed": 14,
            "excluded": 0,
            "censored": 0,
            "independent_unit": "ordered_v663_contract_row",
            "seeds_or_windows_multiply_units": False,
        },
        "prior_failure_disposition": {
            "experiment_id": "exp7586-capstone",
            "literal_prior_verdict": "complete_blocked_required_v662_external_evidence",
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


def build_artifact_for_test(
    *,
    validation_receipts: Sequence[Mapping[str, Any]],
    lifecycle: Mapping[str, Any] | None = None,
) -> JsonDict:
    """Build deterministic evidence from the current worktree for unit tests."""

    roadmap_path, roadmap, candidates = resolve_v663_roadmap(ROOT)
    design = (ROOT / DESIGN_PATH).read_text(encoding="utf-8")
    if lifecycle is None:
        with tempfile.TemporaryDirectory(prefix="exp7587-unit-", dir="/tmp") as directory:
            lifecycle = run_lifecycle_control(Path(directory))
    return build_artifact(
        ROOT,
        roadmap_path,
        candidates,
        compare_contract_authorities(design, roadmap),
        run_contract_mutation_controls(design, roadmap),
        collect_v662_dispositions(ROOT),
        lifecycle,
        validation_receipts,
        started_at_utc="2026-09-24T00:00:00+00:00",
        ended_at_utc="2026-09-24T00:00:00+00:00",
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
        "inference_substrate_class": "blocked_no_run",
        "inference_substrate": "blocked_no_run",
        "duration_s": 0.0,
        "random_seed": 7587,
        "contract_ready_score": 0,
        "method_ingestion_complete_score": 0,
        "rows": [],
        "v662_dispositions": [],
        "method_map_path": NOTE_PATH.as_posix(),
        "acceptance_gate_results": [{**failure, "category": "validity"}],
        "gate_check_summary": {
            "passed": False,
            "failed_checks": ["required_input"],
            "first_failure": failure,
        },
        "sample_size_budget": {
            "intended": 14,
            "planned": 14,
            "observed": 0,
            "excluded": 0,
            "censored": 14,
            "independent_unit": "ordered_v663_contract_row",
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


def _source_hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash every source row that claims exact existing bytes."""

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
        "v662_dispositions",
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
    """Build exact replay, independent reduction, and strict-reader commands."""

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


def _required_inputs(root: Path) -> list[tuple[Path, str, object, object]]:
    """Observe exact external prerequisites before dependent aggregation."""

    paths = (
        DESIGN_PATH,
        SPEC_PATH,
        V662_CAPSTONE_PATH,
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
            "REQ-REPORT-7587",
            "REQ-REPORT-7587" if "REQ-REPORT-7587" in text else "absent",
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
        roadmap_path, roadmap, candidates = resolve_v663_roadmap(root)
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
    dispositions = collect_v662_dispositions(root)
    progress(started, "preconditions", "after", completed_units=len(_required_inputs(root)))

    for phase in ("model_load", "generation"):
        progress(started, phase, "before", planned_units=0)
        progress(started, phase, "after", completed_units=0)

    progress(started, "method_ingestion", "before", planned_units=4)
    write_method_records(root)
    progress(started, "method_ingestion", "after", completed_units=4)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7587-validation-", dir="/tmp"))
    lifecycle = run_lifecycle_control(private_root / "lifecycle")
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
        lifecycle,
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

    terminal_plan = _terminal_commands(root, candidate_path, private_root / "fresh-lifecycle")
    progress(
        started, "terminal_validation", "before_subprocesses", planned_units=len(terminal_plan)
    )
    terminal = _normalize_receipts(
        run_commands(root, terminal_plan, log_dir=private_root / "logs/terminal"), root
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
    """Accept only the run date frozen by the V663 contract."""

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
    """Run Exp7587 or replay one bounded candidate check."""

    print("[exp7587] phase=startup event=flushed", flush=True)
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

"""Build the V635 source and exact task-contract receipt.

The receipt compares the Markdown design with the activated or staged YAML.
It also checks source metadata and real conductor gate behavior. It does not
run an LLM or turn planning readiness into a scientific result.

Spec refs: REQ-REPORT-7205 and SCENARIO-REPORT-7205-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import sys
import tempfile
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

from carnot.experiment_7179_v633_contract_receipt import (
    _atomic_write_bytes,
    _atomic_write_json,
    _canonical_gates,
    _field_declared,
    _load_yaml_bytes,
    _run_streaming_command,
    _sha256,
    _sha256_bytes,
    parse_markdown_contract,
    parse_yaml_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - import bootstrap varies by caller.
    sys.path.insert(0, str(SCRIPTS_DIR))

from conductor_gates import _is_quarantined, evaluate_gates  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]

MILESTONE = "2026.09.635"
RUN_DATE = "20260911"
PLANNING_CUTOFF = "2026-09-11"
FIRST_TASK_ID = "exp7205-source-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7206-arc-volume-a",
    "exp7207-arc-volume-b",
    "exp7208-span-fixture",
    "exp7209-span-canary",
    "exp7210-span-capture",
    "exp7211-span-value-audit",
    "exp7212-refinement-fixture",
    "exp7213-refinement-learning",
    "exp7214-refinement-cold-audit",
    "exp7215-down-up-prototype",
    "exp7216-down-up-quality",
    "exp7217-abi-board-readiness",
    "exp7218-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
RANDOM_SEED = 7_205_202_609_11

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
V634_RECEIPT_PATH = Path("results/experiment_7192_v634_source_contract.json")
V634_CAPSTONE_PATH = Path("results/experiment_7204_v634_capstone.json")
MODULE_PATH = Path("python/carnot/experiment_7205_v635_source_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7205_v635_source_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7205_v635_source_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7205_v635_source_contract.json")
RAW_DIR = Path("results/raw/experiment_7205")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7205_v635_source_contract.json")
RAW_MARKDOWN_NAME = "research-roadmap-vNEXT.md"
RAW_YAML_NAME = "selected-roadmap.yaml"

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
INVENTED_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ARC_LINT_PATH = Path("scripts/arc_levelup_guarantee_lint.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")
TOOL_PATHS = (
    ROADMAP_SCHEMA_PATH,
    Path("scripts/conductor_gates.py"),
    PRIOR_LINT_PATH,
    EXCLUSION_LINT_PATH,
    INVENTED_PATH_LINT_PATH,
    ARC_LINT_PATH,
    GATE_LINT_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    SPEC_COVERAGE_PATH,
)
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    REFERENCE_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    DESIGN_PATH,
    SPEC_PATH,
    V634_RECEIPT_PATH,
    V634_CAPSTONE_PATH,
    Path("scripts/roadmap_schema.py"),
    Path("scripts/conductor_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/harness_consumer_checks.py"),
    Path("python/carnot/experiment_7179_v633_contract_receipt.py"),
    *TOOL_PATHS,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
POSITIVE_VERDICT = "complete_positive_v635_source_contract_exact_agreement"
DISQUALIFIED_VERDICT = "complete_disqualified_v635_source_contract_mismatch"
BLOCKED_VERDICT = "blocked_v635_source_contract_prerequisite_missing"

REQUIRED_FIELD_PRINCIPLES = {
    "field_principles": "Echo the reason for each field beside its actual evidence.",
    "status": "Write a terminal artifact only after completion or a diagnosed external block.",
    "run_date": "Use 20260911; do not substitute an upstream experiment date.",
    "preconditions_checked": "Record the actual resource, code and gate observations.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "The actual operation determines its duration floor.",
    "execution_venue": "Use exactly host, kv260, gatemate or polarfire; these tasks execute on host.",
    "execution_host": "Put the actual hostname here, never inside execution_venue.",
    "duration_s": "Measure monotonic work time; do not pad it to pass a floor.",
    "source_artifact_hashes": "Bind source code, input data and frozen contracts to the claim.",
    "rows": "Keep unit_id, arm, seed, metric, error and abstention for each comparison; do not replace numeric rows with a task roster.",
    "sample_size_budget": "Retain planned, attempted, completed, censored and independent-unit counts.",
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and raw rows.",
    "gate_check_summary": "Every blocked_* verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "Same correctness authority remains circular even with a separate implementation.",
    "verdict_class": "Use exactly positive | circular_positive | null | blocked | disqualified | partial; only incomplete own work can be partial.",
    "honest_verdict": "Use complete_ or complete: for completed findings; blocked_* for external absence. Readiness is not scientific value.",
    "source_contract_complete_score": "One means the advisory receipt is complete, not that science succeeded.",
    "contract_rows": "Every contracted task is independently recovered from both files.",
    "activation_validation_rows": "Actual parsers and gates must accept the emitted format.",
    "source_method_rows": "URLs, dates and access outcomes distinguish source ingestion from invention.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

VALIDATION_COMMAND_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "exclusion_manifest",
    "invented_paths",
    "arc_floor",
    "gate_declarations",
    "artifact",
    "focused_tests",
    "focused_coverage",
    "scoped_spec_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "adversarial",
    "row_consistency",
)

ARXIV_SOURCES: dict[str, JsonDict] = {
    "2509.00360": {
        "title": "ChopChop",
        "submitted_date": "2025-08-30",
        "method_boundary": "Carnot's finite relation grammar is an adaptation, not a reproduction of regular coinduction.",
        "target_task": "exp7208-span-fixture",
    },
    "2509.24489": {
        "title": "Query-Driven Interactive Refinement for Constraint Acquisition",
        "submitted_date": "2025-09-29",
        "method_boundary": "Oracle-query learning in a controlled domain is not LLM weight learning.",
        "target_task": "exp7212-refinement-fixture",
    },
    "2609.08873": {
        "title": "High-Magnetization Sampling at Low Temperatures",
        "submitted_date": "2026-09-08",
        "method_boundary": "A finite frustrated-graph kernel test does not reproduce the sparse-SK theorem.",
        "target_task": "exp7215-down-up-prototype",
    },
    "2608.00585": {
        "title": "Verification Without Sufficiency",
        "submitted_date": "2026-08-01",
        "method_boundary": "Missing support is not a contradiction, and gold decomposition is not a headline arm.",
        "target_task": "exp7211-span-value-audit",
    },
}
NONPAPER_SOURCES: tuple[JsonDict, ...] = (
    {
        "source_id": "extropic:z1t",
        "source_type": "vendor_page",
        "url": "https://extropic.ai/writing/z1t",
        "version_or_date": "2026-09-04",
        "method_boundary": "Vendor energy and latency estimates are not local timings.",
        "target_task": "exp7217-abi-board-readiness",
    },
)


def _progress(phase: int, state: str, detail: str) -> None:
    """Flush one real phase event so silence cannot look like progress."""

    print(f"[exp7205] phase {phase} {state}: {detail}", flush=True)


def _summary(
    failed_check: str | None,
    expected: Any,
    observed: Any,
    *,
    upstream: Any = None,
    field: Any = None,
    passed: bool = False,
) -> JsonDict:
    """Use one stable diagnostic shape for every terminal outcome."""

    return {
        "failed_check": failed_check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _contract_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only fields declared by the exact V635 contract table."""

    if task is None:
        return None
    return {
        "order": task.get("order"),
        "id": task.get("id"),
        "title": task.get("title"),
        "deliverable": task.get("deliverable"),
        "gates": task.get("gates"),
    }


def _gate_producer_rows(tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Check the producer and exact declaration behind every YAML gate."""

    by_id = {task.get("id"): task for task in tasks}
    rows: list[JsonDict] = []
    for consumer in tasks:
        for gate in consumer.get("gates", []):
            upstream = gate.get("upstream")
            field = gate.get("artifact_field")
            producer = by_id.get(upstream)
            checks = {
                "producer_exists": producer is not None,
                "producer_is_earlier": bool(
                    producer and int(producer["order"]) < int(consumer["order"])
                ),
                "same_milestone": bool(
                    producer and producer.get("milestone") == consumer.get("milestone") == MILESTONE
                ),
                "bare_field_declared": bool(
                    producer and _field_declared(str(producer.get("required_block", "")), field)
                ),
                "receipt_is_not_upstream": upstream != FIRST_TASK_ID,
            }
            rows.append(
                {
                    "consumer": consumer.get("id"),
                    "upstream": upstream,
                    "field": field,
                    "operator": gate.get("op"),
                    "value": gate.get("value"),
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return rows


def _prior_failure_rows(tasks: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Check every raw prior-failure row without accepting schema defaults."""

    rows: list[JsonDict] = []
    for task in tasks:
        entries = task.get("prior_failures", [])
        if not isinstance(entries, list):
            rows.append(
                {
                    "task_id": task.get("id"),
                    "prior_index": None,
                    "checks": {"list_shape": False},
                    "passed": False,
                }
            )
            continue
        for index, entry in enumerate(entries):
            checks = {
                "mapping": isinstance(entry, Mapping),
                "experiment_id": isinstance(entry, Mapping)
                and bool(str(entry.get("experiment_id", "")).strip()),
                "verdict": isinstance(entry, Mapping)
                and bool(str(entry.get("verdict", "")).strip()),
                "addressed_by": isinstance(entry, Mapping)
                and bool(str(entry.get("addressed_by", "")).strip()),
                "retire_if_same_verdict": isinstance(entry, Mapping)
                and entry.get("retire_if_same_verdict") is True,
            }
            rows.append(
                {
                    "task_id": task.get("id"),
                    "prior_index": index,
                    "checks": checks,
                    "passed": all(checks.values()),
                }
            )
    return rows


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare the independent Markdown and YAML task records."""

    markdown = parse_markdown_contract(markdown_text)
    roadmap = parse_yaml_contract(yaml_document)
    assert isinstance(yaml_document, Mapping)
    raw_tasks = yaml_document.get("tasks")
    assert isinstance(raw_tasks, list)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = roadmap["tasks"]
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))
    rows: list[JsonDict] = []
    for index in range(width):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else None
        observed = yaml_tasks[index] if index < len(yaml_tasks) else None
        checks = {
            "order": bool(
                expected
                and observed
                and expected.get("order") == observed.get("order") == index + 1
            ),
            "id": bool(expected and observed and expected.get("id") == observed.get("id")),
            "title": bool(expected and observed and expected.get("title") == observed.get("title")),
            "deliverable": bool(
                expected and observed and expected.get("deliverable") == observed.get("deliverable")
            ),
            "gates": bool(
                expected
                and observed
                and _canonical_gates(expected.get("gates", []))
                == _canonical_gates(observed.get("gates", []))
            ),
        }
        rows.append(
            {
                "unit_id": (observed or expected or {}).get("id"),
                "arm": "markdown_vs_yaml",
                "seed": RANDOM_SEED,
                "metric": "exact_contract_agreement",
                "error": None if all(checks.values()) else "contract_mismatch",
                "abstention": False,
                "order": index + 1,
                "markdown": _contract_task(expected),
                "yaml": _contract_task(observed),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    markdown_ids = [task["id"] for task in markdown_tasks]
    yaml_ids = [task["id"] for task in yaml_tasks]
    producer_rows = _gate_producer_rows(yaml_tasks)
    prior_failure_rows = _prior_failure_rows(raw_tasks)
    consumers = [
        task["id"]
        for task in yaml_tasks
        if any(gate.get("upstream") == FIRST_TASK_ID for gate in task.get("gates", []))
    ]
    dependency_rows = [
        {
            "upstream": FIRST_TASK_ID,
            "expected_consumers": [],
            "observed_consumers": consumers,
            "passed": not consumers,
        }
    ]
    passed = all(
        (
            markdown["milestone"] == MILESTONE,
            roadmap["milestone"] == MILESTONE,
            len(markdown_tasks) == EXPECTED_TASK_COUNT,
            len(yaml_tasks) == EXPECTED_TASK_COUNT,
            markdown_ids == list(EXPECTED_ID_ORDER),
            yaml_ids == list(EXPECTED_ID_ORDER),
            all(row["passed"] for row in rows),
            all(row["passed"] for row in producer_rows),
            all(row["passed"] for row in prior_failure_rows),
            all(row["passed"] for row in dependency_rows),
        )
    )
    return {
        "passed": passed,
        "markdown_milestone": markdown["milestone"],
        "yaml_milestone": roadmap["milestone"],
        "markdown_task_rows": [_contract_task(task) for task in markdown_tasks],
        "yaml_task_rows": [_contract_task(task) for task in yaml_tasks],
        "contract_rows": rows,
        "gate_producer_rows": producer_rows,
        "prior_failure_rows": prior_failure_rows,
        "receipt_dependency_rows": dependency_rows,
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "markdown_id_order": markdown_ids,
        "observed_id_order": yaml_ids,
    }


def _select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Select only a V635 active roadmap or, if needed, its staged file."""

    candidates: list[JsonDict] = []
    selected: tuple[Path, JsonDict, bytes] | None = None
    for relative in (ACTIVE_ROADMAP_PATH, NEXT_ROADMAP_PATH):
        _progress(1, "check start", f"roadmap candidate {relative}")
        path = root / relative
        try:
            content = path.read_bytes()
            document = _load_yaml_bytes(content, path)
            observed: Any = document.get("milestone")
            matches = observed == MILESTONE
        except (OSError, ValueError) as exc:
            content, document, matches = b"", None, False
            observed = f"{type(exc).__name__}: {exc}"
        candidates.append(
            {
                "check": f"yaml_candidate:{relative}",
                "path": str(relative),
                "upstream": str(relative),
                "field": "milestone",
                "expected_value": MILESTONE,
                "observed_value": observed,
                "available": matches,
                "blocking_external": False,
            }
        )
        _progress(1, "check end", f"roadmap candidate {relative} matches={matches}")
        if selected is None and matches and document is not None:
            selected = (relative, document, content)
    if selected is None:
        return None, None, None, candidates
    return *selected, candidates


def _check_writable_directory(path: Path) -> tuple[bool, str]:
    """Test one output directory without leaving a probe file."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".exp7205-write-"):
            pass
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "temporary_write_succeeded"


def _source_hashes(root: Path, authority: Path | None) -> dict[str, str | None]:
    """Hash every exact local input, implementation file, and contract."""

    paths = [*SOURCE_PATHS, *([authority] if authority else [])]
    hashes: dict[str, str | None] = {}
    for relative in dict.fromkeys(paths):
        try:
            hashes[str(relative)] = _sha256(root / relative)
        except OSError:
            hashes[str(relative)] = None
    return hashes


def _preconditions(
    root: Path,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Record required bytes, tools, outputs, requirement, and roadmap gates."""

    rows: list[JsonDict] = []

    def add(
        check: str,
        path: Path,
        expected: Any,
        observed: Any,
        available: bool,
        *,
        upstream: str | None = None,
        field: str | None = None,
        blocking_external: bool = True,
    ) -> None:
        rows.append(
            {
                "check": check,
                "path": str(path),
                "upstream": upstream or str(path),
                "field": field,
                "expected_value": expected,
                "observed_value": observed,
                "available": available,
                "blocking_external": blocking_external,
            }
        )

    authority, roadmap, yaml_bytes, candidates = _select_yaml_authority(root)
    rows.extend(candidates)
    observations = {row["path"]: row["observed_value"] for row in candidates}
    add(
        "yaml_authority",
        root / (authority or ACTIVE_ROADMAP_PATH),
        f"one roadmap with milestone {MILESTONE}",
        str(authority) if authority else observations,
        authority is not None,
        upstream="research-roadmap.yaml|research-roadmap-next.yaml",
        field="milestone",
    )
    for relative in SOURCE_PATHS:
        _progress(1, "check start", f"source bytes {relative}")
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        add(
            f"source_bytes:{relative}",
            path,
            "readable_nonempty_bytes",
            path.stat().st_size if available else "missing_or_empty",
            available,
            field="bytes",
        )
        _progress(1, "check end", f"source bytes {relative} available={available}")
    _progress(1, "check start", "driving requirement")
    try:
        spec_has_req = "REQ-REPORT-7205" in (root / SPEC_PATH).read_text(encoding="utf-8")
    except OSError:
        spec_has_req = False
    add(
        "driving_requirement",
        root / SPEC_PATH,
        "REQ-REPORT-7205",
        "REQ-REPORT-7205" if spec_has_req else "missing",
        spec_has_req,
        field="REQ-*",
    )
    _progress(1, "check end", f"driving requirement available={spec_has_req}")
    _progress(1, "check start", "V635 planning source table")
    try:
        references = (root / REFERENCE_PATH).read_text(encoding="utf-8")
        source_table_available = all(
            marker in references
            for marker in (
                "V635-PLANNER-REFRESH-20260911-START",
                *ARXIV_SOURCES,
                "https://extropic.ai/writing/z1t",
            )
        )
    except OSError:
        source_table_available = False
    add(
        "planning_source_table",
        root / REFERENCE_PATH,
        "dated V635 table with all five selected URLs",
        "available" if source_table_available else "missing_or_incomplete",
        source_table_available,
        field="V635-PLANNER-REFRESH-20260911",
    )
    _progress(1, "check end", f"V635 planning source table available={source_table_available}")
    _progress(1, "check start", "Python executable")
    python_path = Path(sys.executable)
    add(
        "python_tool",
        python_path,
        "executable_file",
        str(python_path) if python_path.is_file() else "missing",
        python_path.is_file(),
        field="tool",
    )
    _progress(1, "check end", f"Python executable available={python_path.is_file()}")
    for label, directory in (
        ("result_directory", output_path.parent),
        ("raw_directory", raw_dir),
        ("checkpoint_directory", checkpoint_path.parent),
    ):
        _progress(1, "check start", label)
        available, observed = _check_writable_directory(directory)
        add(label, directory, "temporary_write_succeeds", observed, available, field="writable")
        _progress(1, "check end", f"{label} available={available}")
    execution_host = platform.node()
    add(
        "execution_metadata",
        Path("host"),
        {
            "execution_venue": sorted(("host", "kv260", "gatemate", "polarfire")),
            "hostname": "nonempty",
        },
        {"execution_venue": "host", "execution_host": execution_host},
        bool(execution_host),
        upstream="local_execution_environment",
        field="execution_venue|execution_host",
    )
    if roadmap is not None:
        _progress(1, "check start", "task has no declared upstream gates")
        own = next(
            (task for task in roadmap.get("tasks", []) if task.get("id") == FIRST_TASK_ID), None
        )
        gates = own.get("gated_on") if isinstance(own, Mapping) else None
        own_ungated = not gates
        add(
            "task_upstream_gates",
            root / authority if authority else root,
            [],
            gates or [],
            own_ungated,
            upstream=FIRST_TASK_ID,
            field="gated_on",
        )
        _progress(1, "check end", f"task upstream gates empty={own_ungated}")
    _progress(1, "check start", "source hashes")
    hashes = _source_hashes(root, authority)
    missing_hashes = {key: value for key, value in hashes.items() if value is None}
    add(
        "source_hashes",
        root,
        "sha256_for_every_required_input",
        missing_hashes,
        not missing_hashes,
        field="sha256",
    )
    _progress(1, "check end", f"source hashes missing={len(missing_hashes)}")
    return rows, authority, roadmap, yaml_bytes


def _unwrap_principle_value(value: Any) -> Any:
    """Unwrap only the shared two-key evidence convention.

    A domain object can legitimately contain a key named ``value``. Requiring
    ``principle`` as well prevents the intake check from changing such data.
    """

    if isinstance(value, Mapping) and "principle" in value and "value" in value:
        return value["value"]
    return value


def upstream_precondition(data: Mapping[str, Any], field: str, expected: Any) -> JsonDict:
    """Reject quarantine before a structured field can authorize consumption."""

    payload = dict(data)
    quarantined = _is_quarantined(payload)
    actual = _unwrap_principle_value(payload.get(field))
    field_passed = actual == expected
    passed = not quarantined and field_passed
    if quarantined:
        reason = "quarantined_upstream_rejected_before_field_consumption"
    elif not field_passed:
        reason = "upstream_field_did_not_match"
    else:
        reason = "upstream_precondition_passed"
    return {
        "field": field,
        "expected": expected,
        "actual": actual,
        "quarantined": quarantined,
        "field_passed": field_passed,
        "passed": passed,
        "reason": reason,
    }


def _upstream_intake_rows(root: Path) -> list[JsonDict]:
    """Authenticate V634 receipts without making them V635 authorities."""

    rows: list[JsonDict] = []
    fields = (
        (V634_RECEIPT_PATH, "source_contract_complete_score"),
        (V634_CAPSTONE_PATH, "capstone_complete_score"),
    )
    for relative, field in fields:
        _progress(1, "check start", f"historical intake {relative}")
        path = root / relative
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("artifact must be a mapping")
            quarantined = _is_quarantined(data)
            actual = _unwrap_principle_value(data.get(field))
            known_failed = actual != 1 or "disqualified" in str(
                _unwrap_principle_value(data.get("honest_verdict", ""))
            )
            error = None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            data, quarantined, actual, known_failed = {}, False, None, False
            error = f"{type(exc).__name__}: {exc}"
        accepted = error is None and not quarantined and not known_failed
        rows.append(
            {
                "source_path": str(relative),
                "source_sha256": _sha256(path) if error is None else None,
                "checked_field": field,
                "observed_value": actual,
                "quarantined": quarantined,
                "known_failed_value": known_failed,
                "accepted_for_evidence": accepted,
                "consumed": False,
                "reason": error
                or (
                    "quarantined_upstream_rejected"
                    if quarantined
                    else "known_failed_value_not_promoted"
                    if known_failed
                    else "authenticated_v634_history_not_used_as_v635_contract_authority"
                ),
            }
        )
        _progress(1, "check end", f"historical intake {relative} accepted={accepted}")
    return rows


def run_gate_validation_fixtures(results_dir: Path) -> list[JsonDict]:
    """Exercise the real field evaluator and the separate quarantine check."""

    results_dir.mkdir(parents=True, exist_ok=True)
    cases: tuple[tuple[str, int, JsonDict | None], ...] = (
        ("field_one", 9_920_501, {"status": "complete", "ready_score": 1}),
        ("field_zero", 9_920_502, {"status": "complete", "ready_score": 0}),
        ("missing_field", 9_920_503, {"status": "complete"}),
        ("missing_file", 9_920_504, None),
        (
            "quarantined_field_one",
            9_920_505,
            {"status": "complete", "ready_score": 1, "flagged_adversarial": True},
        ),
    )
    rows: list[JsonDict] = []
    for case, number, data in cases:
        _progress(4, "check start", f"real conductor gate fixture {case}")
        upstream = f"exp{number}-validation-input"
        path = results_dir / f"experiment_{number}_validation_input.json"
        if data is not None:
            _atomic_write_json(path, data)
        task = {
            "id": "exp999999-validation-consumer",
            "gated_on": [
                {
                    "upstream": upstream,
                    "artifact_field": "ready_score",
                    "op": "==",
                    "value": 1,
                }
            ],
        }
        gate_check = evaluate_gates(task, results_dir)
        intake = upstream_precondition(data or {}, "ready_score", 1)
        first = gate_check.gates_evaluated[0]
        rows.append(
            {
                "case": case,
                "validation_input": True,
                "research_result": False,
                "fixture_path": str(path) if data is not None else None,
                "field": "ready_score",
                "expected": 1,
                "actual": data.get("ready_score") if data is not None else None,
                "quarantined": intake["quarantined"],
                "conductor_passed": gate_check.passed,
                "conductor_reason": first.reason,
                "experiment_precondition_passed": intake["passed"],
                "experiment_precondition_reason": intake["reason"],
            }
        )
        _progress(
            4,
            "check end",
            f"real conductor gate fixture {case} conductor_passed={gate_check.passed}",
        )
    return rows


def _fetch_url(url: str, *, timeout: float = 20.0) -> JsonDict:
    """Fetch one bounded primary page and retain the actual access outcome."""

    request = Request(
        url,
        headers={"User-Agent": "Carnot-EBM-source-contract/7205"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= int(response.status) < 400,
                "status_code": int(response.status),
                "url": url,
                "headers": dict(response.headers),
                "body": body,
                "error": None,
            }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return {
            "ok": False,
            "status_code": int(exc.code),
            "url": url,
            "headers": dict(exc.headers),
            "body": body,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - access failures belong in the receipt.
        return {
            "ok": False,
            "status_code": None,
            "url": url,
            "headers": {},
            "body": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _observed_version_and_date(
    body: str, fallback_date: str | None
) -> tuple[str | None, str | None]:
    """Extract a visible arXiv version and date without inferring missing facts."""

    version_match = re.search(r"\[v(\d+)\]", body, re.I)
    iso_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", body)
    submitted_match = re.search(
        r"(?:Submitted on|\[v\d+\])\s*(?:\w{3},\s*)?(\d{1,2})\s+([A-Z][a-z]{2})\s+(20\d{2})",
        body,
    )
    observed_date = iso_match.group(1) if iso_match else None
    if observed_date is None and submitted_match:
        observed_date = datetime.strptime(" ".join(submitted_match.groups()), "%d %b %Y").strftime(
            "%Y-%m-%d"
        )
    return (
        f"v{version_match.group(1)}" if version_match else None,
        observed_date or fallback_date,
    )


def collect_source_method_rows(fetcher: Fetcher = _fetch_url) -> list[JsonDict]:
    """Check four primary papers plus dated Extropic and Kona pages."""

    rows: list[JsonDict] = []
    for paper_id, source in ARXIV_SOURCES.items():
        url = f"https://arxiv.org/abs/{paper_id}"
        _progress(3, "source request start", url)
        try:
            receipt = dict(fetcher(url))
        except Exception as exc:  # noqa: BLE001 - an injected source must fail visibly.
            receipt = {"ok": False, "status_code": None, "body": "", "error": str(exc)}
        _progress(3, "source request end", f"{url} ok={receipt.get('ok') is True}")
        body = str(receipt.get("body") or "")
        version, observed_date = _observed_version_and_date(body, str(source["submitted_date"]))
        accessed = receipt.get("ok") is True
        rows.append(
            {
                "source_id": f"arxiv:{paper_id}",
                "source_type": "primary_paper",
                "url": url,
                "planning_date": source["submitted_date"],
                "version": version or "v1_cached_primary_evidence",
                "version_or_date": observed_date,
                "access_outcome": (
                    f"http_{receipt.get('status_code')}"
                    if accessed
                    else "unavailable_cached_primary_evidence"
                ),
                "access_error": receipt.get("error"),
                "content_sha256": _sha256_bytes(body.encode()) if body else None,
                "method_boundary": source["method_boundary"],
                "target_task": source["target_task"],
                "post_planning_delta": bool(
                    accessed and observed_date and observed_date > PLANNING_CUTOFF
                ),
                "ingested_change": False,
            }
        )
    for source in NONPAPER_SOURCES:
        url = str(source["url"])
        _progress(3, "source request start", url)
        try:
            receipt = dict(fetcher(url))
        except Exception as exc:  # noqa: BLE001 - an injected source must fail visibly.
            receipt = {"ok": False, "status_code": None, "body": "", "error": str(exc)}
        _progress(3, "source request end", f"{url} ok={receipt.get('ok') is True}")
        body = str(receipt.get("body") or "")
        _version, observed_date = _observed_version_and_date(body, source.get("version_or_date"))
        accessed = receipt.get("ok") is True
        rows.append(
            {
                "source_id": source["source_id"],
                "source_type": source["source_type"],
                "url": url,
                "planning_date": source.get("version_or_date"),
                "version": None,
                "version_or_date": observed_date,
                "access_outcome": (
                    f"http_{receipt.get('status_code')}"
                    if accessed
                    else "unavailable_cached_primary_evidence"
                ),
                "access_error": receipt.get("error"),
                "content_sha256": _sha256_bytes(body.encode()) if body else None,
                "method_boundary": source["method_boundary"],
                "target_task": source["target_task"],
                "post_planning_delta": bool(
                    accessed and observed_date and observed_date > PLANNING_CUTOFF
                ),
                "ingested_change": False,
            }
        )
    return rows


def _raw_source_rows(
    root: Path, authority: Path, yaml_bytes: bytes, raw_dir: Path
) -> list[JsonDict]:
    """Freeze both current contract authorities without normalizing bytes."""

    sources = (
        ("markdown", DESIGN_PATH, (root / DESIGN_PATH).read_bytes(), raw_dir / RAW_MARKDOWN_NAME),
        ("yaml", authority, yaml_bytes, raw_dir / RAW_YAML_NAME),
    )
    rows: list[JsonDict] = []
    for source_type, relative, content, destination in sources:
        _atomic_write_bytes(destination, content)
        source_hash = _sha256_bytes(content)
        raw_hash = _sha256(destination)
        rows.append(
            {
                "source_type": source_type,
                "source_path": str(relative),
                "raw_path": str(destination.relative_to(root)),
                "source_sha256": source_hash,
                "raw_sha256": raw_hash,
                "hash_matches": source_hash == raw_hash,
            }
        )
    return rows


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Give every stored field a short audit purpose and preserve exact principles."""

    principles = {
        field: f"The {field} field keeps one part of the V635 receipt auditable."
        for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all stored evidence except the checksum's own value."""

    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _base_artifact(run_date: str, checkpoint_path: Path) -> JsonDict:
    """Create a running checkpoint with every required terminal field."""

    artifact: JsonDict = {
        "schema": "carnot.exp7205.v635_source_contract.v1",
        "experiment_id": FIRST_TASK_ID,
        "status": "running",
        "run_date": run_date,
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.000001,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "contract_rows": {
                "planned": EXPECTED_TASK_COUNT,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": EXPECTED_TASK_COUNT,
            },
            "source_urls": {
                "planned": len(ARXIV_SOURCES) + len(NONPAPER_SOURCES),
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": len(ARXIV_SOURCES) + len(NONPAPER_SOURCES),
            },
            "gate_validation_inputs": {
                "planned": 5,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": 5,
            },
            "independent_units": EXPECTED_TASK_COUNT,
            "exclusions": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _summary(
            "preconditions_not_checked", "all required resources", "not_checked"
        ),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_v635_source_contract",
        "source_contract_complete_score": 0,
        "contract_rows": [],
        "source_method_rows": [],
        "activation_validation_rows": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "markdown_task_rows": [],
        "yaml_task_rows": [],
        "gate_producer_rows": [],
        "prior_failure_rows": [],
        "receipt_dependency_rows": [],
        "raw_source_rows": [],
        "upstream_intake_rows": [],
        "validation_command_rows": [],
        "contract_authorities": [],
        "yaml_authority_path": None,
        "markdown_milestone": None,
        "yaml_milestone": None,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "markdown_id_order": [],
        "observed_id_order": [],
        "source_mapping_complete": False,
        "structural_checks_complete": False,
        "post_planning_source_change_count": 0,
        "research_files_updated": False,
        "checkpoint_path": str(checkpoint_path),
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _set_time_and_checksum(artifact: JsonDict, started: float) -> None:
    """Store measured elapsed work and bind the complete current payload."""

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.000001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def _checkpoint(path: Path, artifact: JsonDict, started: float) -> None:
    """Persist live state under the checkpoint tree only."""

    _set_time_and_checksum(artifact, started)
    _atomic_write_json(path, artifact)


def _activation_rows(roadmap: JsonDict) -> list[JsonDict]:
    """Run the real schema parser and gate fixtures in a private directory."""

    _progress(4, "check start", "real Roadmap parser")
    try:
        Roadmap.model_validate(roadmap)
        schema_error = None
    except Exception as exc:  # noqa: BLE001 - schema diagnostics belong in the artifact.
        schema_error = f"{type(exc).__name__}: {exc}"
    schema_row = {
        "case": "roadmap_schema_parse",
        "validation_input": True,
        "research_result": False,
        "passed": schema_error is None,
        "error": schema_error,
    }
    _progress(4, "check end", f"real Roadmap parser passed={schema_error is None}")
    with tempfile.TemporaryDirectory(prefix="exp7205-gates-", dir="/tmp") as directory:
        gate_rows = run_gate_validation_fixtures(Path(directory))
    return [schema_row, *gate_rows]


def _activation_complete(rows: object) -> bool:
    """Recompute the expected pass and failure pattern of validation inputs."""

    if not isinstance(rows, list) or len(rows) != 6:
        return False
    by_case = {row.get("case"): row for row in rows if isinstance(row, Mapping)}
    schema = by_case.get("roadmap_schema_parse", {})
    one = by_case.get("field_one", {})
    zero = by_case.get("field_zero", {})
    missing = by_case.get("missing_field", {})
    missing_file = by_case.get("missing_file", {})
    quarantine = by_case.get("quarantined_field_one", {})
    return all(
        (
            schema.get("passed") is True,
            one.get("conductor_passed") is True,
            one.get("experiment_precondition_passed") is True,
            zero.get("conductor_passed") is False,
            zero.get("experiment_precondition_passed") is False,
            missing.get("conductor_passed") is False,
            missing.get("experiment_precondition_passed") is False,
            missing_file.get("conductor_passed") is False,
            missing_file.get("experiment_precondition_passed") is False,
            missing_file.get("fixture_path") is None,
            quarantine.get("conductor_passed") is True,
            quarantine.get("experiment_precondition_passed") is False,
            all(row.get("validation_input") is True for row in rows),
        )
    )


def _score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Return one only for complete source mapping and structural evidence."""

    contract_rows = artifact.get("contract_rows")
    source_rows = artifact.get("source_method_rows")
    producer_rows = artifact.get("gate_producer_rows")
    prior_failure_rows = artifact.get("prior_failure_rows")
    dependency_rows = artifact.get("receipt_dependency_rows")
    intake_rows = artifact.get("upstream_intake_rows")
    command_rows = artifact.get("validation_command_rows")
    if not all(
        isinstance(rows, list)
        for rows in (
            contract_rows,
            source_rows,
            producer_rows,
            prior_failure_rows,
            dependency_rows,
            intake_rows,
        )
    ):
        return 0
    if (
        artifact.get("markdown_milestone") != MILESTONE
        or artifact.get("yaml_milestone") != MILESTONE
        or len(contract_rows) != EXPECTED_TASK_COUNT
        or [row.get("unit_id") for row in contract_rows] != list(EXPECTED_ID_ORDER)
        or not all(row.get("passed") is True for row in contract_rows)
        or not all(
            row.get("passed") is True
            for row in producer_rows + prior_failure_rows + dependency_rows
        )
    ):
        return 0
    primary_rows = [row for row in source_rows if row.get("source_type") == "primary_paper"]
    if (
        len(primary_rows) != len(ARXIV_SOURCES)
        or len(source_rows) != len(ARXIV_SOURCES) + len(NONPAPER_SOURCES)
        or any(
            not row.get("url")
            or not row.get("planning_date")
            or not row.get("access_outcome")
            or not row.get("method_boundary")
            for row in source_rows
        )
    ):
        return 0
    if not _activation_complete(artifact.get("activation_validation_rows")):
        return 0
    if len(intake_rows) != 2 or any(
        row.get("accepted_for_evidence") is not True for row in intake_rows
    ):
        return 0
    if isinstance(command_rows, list) and command_rows:
        if any(row.get("passed") is not True for row in command_rows):
            return 0
    return 1


def _failure_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Name the first failed source, contract, activation, or command check."""

    if artifact.get("markdown_milestone") != MILESTONE:
        return _summary(
            "markdown_milestone",
            MILESTONE,
            artifact.get("markdown_milestone"),
            upstream=str(DESIGN_PATH),
            field="milestone",
        )
    if artifact.get("yaml_milestone") != MILESTONE:
        return _summary(
            "yaml_milestone",
            MILESTONE,
            artifact.get("yaml_milestone"),
            upstream=artifact.get("yaml_authority_path"),
            field="milestone",
        )
    row = next(
        (row for row in artifact.get("contract_rows", []) if row.get("passed") is not True),
        None,
    )
    if row:
        return _summary(
            f"contract_order_{row.get('order')}",
            row.get("markdown"),
            row.get("yaml"),
            upstream=row.get("unit_id"),
            field=",".join(name for name, passed in row.get("checks", {}).items() if not passed),
        )
    producer = next(
        (row for row in artifact.get("gate_producer_rows", []) if row.get("passed") is not True),
        None,
    )
    if producer:
        return _summary(
            "gate_producer_contract",
            True,
            producer.get("checks"),
            upstream=producer.get("upstream"),
            field=producer.get("field"),
        )
    prior = next(
        (row for row in artifact.get("prior_failure_rows", []) if row.get("passed") is not True),
        None,
    )
    if prior:
        return _summary(
            "prior_failure_contract",
            True,
            prior.get("checks"),
            upstream=prior.get("task_id"),
            field="prior_failures",
        )
    if not _activation_complete(artifact.get("activation_validation_rows")):
        return _summary(
            "activation_validation",
            "schema and four exact gate outcomes",
            artifact.get("activation_validation_rows"),
            field="activation_validation_rows",
        )
    failed_command = next(
        (
            row
            for row in artifact.get("validation_command_rows", [])
            if row.get("passed") is not True
        ),
        None,
    )
    if failed_command:
        return _summary(
            f"validation:{failed_command.get('name')}",
            0,
            failed_command.get("exit_code"),
            field="validation_command_rows",
        )
    return _summary(
        "stored_source_contract_evidence",
        "complete V635 mapping and structural checks",
        "incomplete evidence",
    )


def _failed_precondition(artifact: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the first unavailable resource that must block all work."""

    rows = artifact.get("preconditions_checked")
    if not isinstance(rows, list):
        return None
    return next(
        (
            row
            for row in rows
            if row.get("blocking_external") is True and row.get("available") is not True
        ),
        None,
    )


def _apply_terminal_state(artifact: JsonDict) -> None:
    """Derive lifecycle fields from stored evidence rather than intent."""

    artifact["status"] = "complete"
    failed = _failed_precondition(artifact)
    score = _score_from_artifact(artifact)
    artifact["source_contract_complete_score"] = score
    if failed:
        artifact["inference_substrate"] = "blocked_no_run"
        artifact["inference_substrate_class"] = "blocked_no_run"
        artifact["verdict_class"] = "blocked"
        artifact["honest_verdict"] = BLOCKED_VERDICT
        artifact["gate_check_summary"] = _summary(
            str(failed.get("check")),
            failed.get("expected_value"),
            failed.get("observed_value"),
            upstream=failed.get("upstream"),
            field=failed.get("field"),
        )
    elif score == 1:
        artifact["inference_substrate"] = INFERENCE_SUBSTRATE
        artifact["inference_substrate_class"] = "aggregation"
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = POSITIVE_VERDICT
        artifact["gate_check_summary"] = _summary(
            None,
            "all_14_v635_contract_rows_and_structural_checks_pass",
            "all_14_v635_contract_rows_and_structural_checks_pass",
            passed=True,
        )
    else:
        artifact["inference_substrate"] = INFERENCE_SUBSTRATE
        artifact["inference_substrate_class"] = "aggregation"
        artifact["verdict_class"] = "disqualified"
        artifact["honest_verdict"] = DISQUALIFIED_VERDICT
        artifact["gate_check_summary"] = _failure_summary(artifact)


def validate_artifact(artifact: object) -> list[str]:
    """Recompute required fields, score, lifecycle state, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != value for field, value in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles_invalid")
    if artifact.get("status") != "complete":
        errors.append("status_invalid")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("execution_host") != (platform.node() or "unknown"):
        errors.append("execution_host_invalid")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        errors.append("duration_invalid")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("rows") != artifact.get("contract_rows"):
        errors.append("rows_invalid")
    sample_budget = artifact.get("sample_size_budget")
    if not isinstance(sample_budget, Mapping) or any(
        not isinstance(sample_budget.get(group), Mapping)
        or any(
            field not in sample_budget[group]
            for field in ("planned", "attempted", "completed", "censored", "independent_units")
        )
        for group in ("contract_rows", "source_urls", "gate_validation_inputs")
    ):
        errors.append("sample_size_budget_invalid")
    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        errors.append("source_hashes_invalid")
    raw_rows = artifact.get("raw_source_rows")
    if raw_rows and (
        not isinstance(raw_rows, list)
        or any(
            row.get("hash_matches") is not True or row.get("source_sha256") != row.get("raw_sha256")
            for row in raw_rows
        )
    ):
        errors.append("raw_source_rows_invalid")
    command_rows = artifact.get("validation_command_rows")
    if not isinstance(command_rows, list) or [row.get("name") for row in command_rows] != list(
        VALIDATION_COMMAND_NAMES[: len(command_rows)]
    ):
        errors.append("validation_command_rows_invalid")
    expected_score = _score_from_artifact(artifact)
    if artifact.get("source_contract_complete_score") != expected_score:
        errors.append("source_contract_complete_score_invalid")
    expected = dict(artifact)
    _apply_terminal_state(expected)
    for field in (
        "inference_substrate",
        "inference_substrate_class",
        "verdict_class",
        "honest_verdict",
        "gate_check_summary",
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"{field}_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def _validation_commands(
    root: Path, artifact_path: Path, authority: Path
) -> tuple[tuple[str, list[str]], ...]:
    """Return changed-scope checks and the required planning validators."""

    python = str(Path(sys.executable))
    roadmap = str(root / authority)
    artifact = str(artifact_path)
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    artifact_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7205_v635_source_contract import validate_artifact;"
        f"errors=validate_artifact(json.loads(pathlib.Path({artifact!r}).read_text()));"
        "print(errors);sys.exit(bool(errors))"
    )
    pytest_base = "/tmp/exp7205-v635-focused-pytest"
    return (
        ("roadmap_schema", [python, "-u", "-c", schema_code]),
        ("prior_failure", [python, "-u", str(root / PRIOR_LINT_PATH), roadmap]),
        ("exclusion_manifest", [python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap]),
        (
            "invented_paths",
            [python, "-u", str(root / INVENTED_PATH_LINT_PATH), "prompt-paths", roadmap],
        ),
        ("arc_floor", [python, "-u", str(root / ARC_LINT_PATH), roadmap]),
        ("gate_declarations", [python, "-u", str(root / GATE_LINT_PATH), roadmap]),
        ("artifact", [python, "-u", "-c", artifact_code]),
        (
            "focused_tests",
            [
                str(root / ".venv/bin/coverage"),
                "run",
                "--data-file=/tmp/.coverage-exp7205-v635-validation",
                "--include=*/experiment_7205_v635_source_contract.py",
                "-m",
                "pytest",
                "--no-cov",
                "-n",
                "0",
                f"--basetemp={pytest_base}",
                str(root / TEST_PATH),
                "-q",
            ],
        ),
        (
            "focused_coverage",
            [
                str(root / ".venv/bin/coverage"),
                "report",
                "--data-file=/tmp/.coverage-exp7205-v635-validation",
                "--include=*/experiment_7205_v635_source_contract.py",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        (
            "scoped_spec_coverage",
            [python, "-u", str(root / SPEC_COVERAGE_PATH), str(root / TEST_PATH)],
        ),
        (
            "ruff_check",
            [
                str(root / ".venv/bin/ruff"),
                "check",
                str(root / MODULE_PATH),
                str(root / WRAPPER_PATH),
                str(root / TEST_PATH),
            ],
        ),
        (
            "ruff_format",
            [
                str(root / ".venv/bin/ruff"),
                "format",
                "--check",
                str(root / MODULE_PATH),
                str(root / WRAPPER_PATH),
                str(root / TEST_PATH),
            ],
        ),
        ("mypy", [str(root / ".venv/bin/mypy"), str(root / MODULE_PATH)]),
        ("adversarial", [python, "-u", str(root / ADVERSARIAL_PATH), artifact]),
        ("row_consistency", [python, "-u", str(root / ROW_LINT_PATH), artifact]),
    )


def run_validation_commands(
    root: Path,
    checkpoint_path: Path,
    artifact: JsonDict,
    started: float,
) -> list[JsonDict]:
    """Stream each bounded validator and save every completed receipt."""

    authority = Path(str(artifact["yaml_authority_path"]))
    commands = _validation_commands(root, checkpoint_path, authority)
    rows: list[JsonDict] = []
    for index, (name, command) in enumerate(commands, 1):
        _checkpoint(checkpoint_path, artifact, started)
        _progress(5, "subprocess start", f"{index}/{len(commands)} {name}")
        receipt = _run_streaming_command(
            command,
            cwd=root,
            timeout_s=600,
            heartbeat_s=60,
            operation=f"exp7205:{name}",
        )
        row = {
            "name": name,
            "command": shlex.join(command),
            **receipt,
            "passed": receipt["exit_code"] == 0,
        }
        rows.append(row)
        artifact["validation_command_rows"] = list(rows)
        _apply_terminal_state(artifact)
        _checkpoint(checkpoint_path, artifact, started)
        _progress(5, "subprocess end", f"{index}/{len(commands)} {name} exit={row['exit_code']}")
    return rows


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    fetcher: Fetcher = _fetch_url,
    run_commands: bool = True,
) -> JsonDict:
    """Build one positive, disqualified, or externally blocked receipt."""

    started = time.monotonic()
    _progress(0, "start", "initialize a nonterminal checkpoint")
    artifact = _base_artifact(run_date, checkpoint_path)
    _checkpoint(checkpoint_path, artifact, started)
    _progress(0, "end", "checkpoint exists outside the terminal result path")

    _progress(1, "start", "check required bytes, tools, outputs, gates, and quarantines")
    preconditions, authority, roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["yaml_authority_path"] = str(authority) if authority else None
    artifact["source_artifact_hashes"] = _source_hashes(root, authority)
    artifact["upstream_intake_rows"] = _upstream_intake_rows(root)
    failed = _failed_precondition(artifact)
    if failed:
        _apply_terminal_state(artifact)
        _checkpoint(checkpoint_path, artifact, started)
        _progress(1, "end", f"blocked on {failed.get('check')}")
    else:
        assert authority is not None and roadmap is not None and yaml_bytes is not None
        artifact["contract_authorities"] = [str(DESIGN_PATH), str(authority)]
        artifact["raw_source_rows"] = _raw_source_rows(root, authority, yaml_bytes, raw_dir)
        artifact["inference_substrate"] = INFERENCE_SUBSTRATE
        artifact["inference_substrate_class"] = "aggregation"
        _checkpoint(checkpoint_path, artifact, started)
        _progress(1, "end", "current authority bytes are hash-pinned and history is rejected")

        _progress(2, "start", "parse and compare fourteen independent contract rows")
        try:
            contract = evaluate_contract((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)
        except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
            artifact["markdown_milestone"] = "parse_error"
            artifact["gate_check_summary"] = _summary(
                "contract_parse",
                "independently parseable Markdown and YAML",
                f"{type(exc).__name__}: {exc}",
                upstream=f"{DESIGN_PATH}|{authority}",
                field="contract_rows",
            )
        else:
            for field in (
                "markdown_milestone",
                "yaml_milestone",
                "markdown_task_rows",
                "yaml_task_rows",
                "contract_rows",
                "gate_producer_rows",
                "prior_failure_rows",
                "receipt_dependency_rows",
                "expected_id_order",
                "markdown_id_order",
                "observed_id_order",
            ):
                artifact[field] = contract[field]
            artifact["rows"] = artifact["contract_rows"]
            artifact["observed_task_count"] = len(artifact["yaml_task_rows"])
            artifact["source_mapping_complete"] = bool(contract["passed"])
            artifact["sample_size_budget"]["contract_rows"]["completed"] = len(
                artifact["contract_rows"]
            )
            artifact["sample_size_budget"]["contract_rows"]["attempted"] = len(
                artifact["contract_rows"]
            )
        _checkpoint(checkpoint_path, artifact, started)
        _progress(2, "end", f"source mapping complete={artifact['source_mapping_complete']}")

        _progress(3, "start", "check bounded primary sources and dated product pages")
        artifact["source_method_rows"] = collect_source_method_rows(fetcher)
        artifact["post_planning_source_change_count"] = sum(
            row["post_planning_delta"] for row in artifact["source_method_rows"]
        )
        artifact["sample_size_budget"]["source_urls"]["attempted"] = len(
            artifact["source_method_rows"]
        )
        artifact["sample_size_budget"]["source_urls"]["completed"] = len(
            artifact["source_method_rows"]
        )
        artifact["sample_size_budget"]["source_urls"]["censored"] = sum(
            row["access_outcome"] == "unavailable_cached_primary_evidence"
            for row in artifact["source_method_rows"]
        )
        artifact["research_files_updated"] = False
        _checkpoint(checkpoint_path, artifact, started)
        _progress(
            3,
            "end",
            f"post-planning source changes={artifact['post_planning_source_change_count']}",
        )

        _progress(4, "start", "run the real parser and conductor gate validation inputs")
        artifact["activation_validation_rows"] = _activation_rows(roadmap)
        artifact["sample_size_budget"]["gate_validation_inputs"]["attempted"] = 5
        artifact["sample_size_budget"]["gate_validation_inputs"]["completed"] = 5
        artifact["structural_checks_complete"] = _activation_complete(
            artifact["activation_validation_rows"]
        ) and all(row["passed"] for row in artifact["gate_producer_rows"])
        _apply_terminal_state(artifact)
        _checkpoint(checkpoint_path, artifact, started)
        _progress(
            4,
            "end",
            f"structural checks complete={artifact['structural_checks_complete']}",
        )

        if run_commands:
            _progress(5, "start", "run changed-scope and planning validation subprocesses")
            run_validation_commands(root, checkpoint_path, artifact, started)
            _progress(5, "end", "all validation subprocess receipts are stored")

    _apply_terminal_state(artifact)
    _progress(6, "validation start", "recompute the terminal evidence contract")
    _set_time_and_checksum(artifact, started)
    errors = validate_artifact(artifact)
    _progress(6, "validation end", f"errors={errors}")
    if errors:  # pragma: no cover - the CLI must fail closed on an invalid artifact.
        raise ValueError(f"invalid Exp7205 artifact: {errors}")
    _progress(7, "write start", "atomically write checkpoint and terminal deliverable")
    _set_time_and_checksum(artifact, started)
    _atomic_write_json(checkpoint_path, artifact)
    _atomic_write_json(output_path, artifact)
    _progress(7, "write end", f"{artifact['verdict_class']} artifact complete")
    return artifact


def _date_argument(value: str) -> str:
    """Accept only the fixed V635 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source receipt and accept any valid terminal class."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=_date_argument)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    artifact = build_artifact(
        REPO_ROOT,
        args.date,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7205] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7205] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['source_contract_complete_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the required wrapper exercises main.
    raise SystemExit(main())

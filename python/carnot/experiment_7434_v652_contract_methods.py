"""Audit the V652 contract and ingest three bounded method families.

This host-only audit reads exact authorities, historical receipts, and at most
six public paper pages. It performs no model inference or energy-head training.
Its contract score is advisory, so a failed audit cannot block later science.

Spec refs: REQ-REPORT-7434 and SCENARIO-REPORT-7434-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

import yaml

from carnot.experiment_7329_v644_contract import (
    _field_declared,
    parse_markdown_contract,
    parse_yaml_contract,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from scripts.conductor_gates import evaluate_gates


JsonDict = dict[str, Any]
SourceFetcher = Callable[[Mapping[str, str]], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.652"
EXPERIMENT_ID = "exp7434-contract-methods"
SCHEMA = "carnot.exp7434.v652.contract_methods.v1"
RANDOM_SEED = {"contract_mutations": 7_434_652_01, "gate_controls": 7_434_652_02}

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7434_v652_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7434_v652_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7434_v652_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7434_v652_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7434_v652_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v652-method-ingestion.md")
STUDY_PATH = Path("research-studying.md")
CAPSTONE_PATH = Path("results/experiment_7433_v651_capstone.json")
COMPLETE_PATH = Path("research-complete.yaml")
KNOWN_ISSUES_PATH = Path("ops/known-issues.md")

EXPECTED_TASK_IDS = (
    "exp7434-contract-methods",
    "exp7435-round-breaker",
    "exp7436-selection-protocol",
    "exp7437-span-protocol",
    "exp7438-mixture-prototype",
    "exp7439-certified-decisions",
    "exp7440-mixture-learning",
    "exp7441-decision-audit",
    "exp7442-span-capture",
    "exp7443-span-audit",
    "exp7444-arc-supervisor-evidence",
    "exp7445-hardware-envelope",
    "exp7446-capstone",
)

SELECTED_SOURCES = (
    {
        "source_id": "joint_certificate",
        "version": "arXiv:2606.08517v1 (2026-06-07)",
        "url": "https://arxiv.org/abs/2606.08517v1",
        "kind": "primary_paper",
    },
    {
        "source_id": "expert_aggregation",
        "version": "arXiv:2607.20239v1 (2026-07-22)",
        "url": "https://arxiv.org/abs/2607.20239v1",
        "kind": "primary_paper",
    },
    {
        "source_id": "selective_risk",
        "version": "arXiv:2603.24704v1 (2026-03-25)",
        "url": "https://arxiv.org/abs/2603.24704v1",
        "kind": "primary_paper",
    },
    {
        "source_id": "crane_representation",
        "version": "arXiv:2502.09061 (2025)",
        "url": "https://arxiv.org/abs/2502.09061",
        "kind": "primary_paper",
    },
    {
        "source_id": "kan_locality",
        "version": "arXiv:2602.02056 (2026-02)",
        "url": "https://arxiv.org/abs/2602.02056",
        "kind": "primary_paper",
    },
    {
        "source_id": "kan_forgetting",
        "version": "arXiv:2511.12828 (2025-11)",
        "url": "https://arxiv.org/abs/2511.12828",
        "kind": "primary_paper",
    },
)

SOURCE_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/known-issues.md"),
    Path("scripts/experiment_template.py"),
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/conductor_gates.py"),
    Path("scripts/sweep_clusters.py"),
    Path("scripts/sweep_semscholar.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7329_v644_contract.py"),
    Path("python/carnot/experiment_7421_v651_contract_ingestion.py"),
    CAPSTONE_PATH,
    COMPLETE_PATH,
    Path("research-references.md"),
    STUDY_PATH,
    ROADMAP_PATH,
    DESIGN_PATH,
    SPEC_PATH,
    NOTE_PATH,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def utc_now() -> str:  # pragma: no cover - authentic runtime boundary.
    """Return one real UTC boundary in a stable text form."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public E2E progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush every phase boundary with measured monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7434] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact input bytes so later edits cannot inherit this receipt."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_yaml(path: Path) -> JsonDict:
    """Load one YAML mapping and reject prose or sequence-shaped input."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML mapping required: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def load_json(path: Path) -> JsonDict:
    """Load one JSON mapping from an authenticated local path."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON mapping required: {path}")
    return value


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only fields declared independently by both authorities."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Check that each structured gate reads an earlier declared field."""

    tasks = roadmap.get("tasks") if isinstance(roadmap.get("tasks"), list) else []
    positions = {
        task.get("id"): (index, task)
        for index, task in enumerate(tasks)
        if isinstance(task, Mapping)
    }
    result: dict[str, list[JsonDict]] = {}
    for consumer_index, consumer in enumerate(tasks):
        if not isinstance(consumer, Mapping):
            continue
        rows: list[JsonDict] = []
        for gate in consumer.get("gated_on") or []:
            upstream = gate.get("upstream")
            producer_entry = positions.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            declared = _field_declared(str(producer.get("prompt", "")), gate.get("artifact_field"))
            rows.append(
                {
                    "upstream": upstream,
                    "artifact_field": gate.get("artifact_field"),
                    "producer_precedes_consumer": producer_index is not None
                    and producer_index < consumer_index,
                    "declared_verbatim": declared,
                    "passed": producer_index is not None
                    and producer_index < consumer_index
                    and declared,
                }
            )
        result[str(consumer.get("id"))] = rows
    return result


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare generic parser results under the exact V652 contract."""

    try:
        markdown = parse_markdown_contract(markdown_text)
        parsed_yaml = parse_yaml_contract(roadmap)
    except (TypeError, ValueError) as exc:
        return {
            "passed": False,
            "errors": [f"parse_error:{exc}"],
            "markdown_milestone": None,
            "yaml_milestone": roadmap.get("milestone") if isinstance(roadmap, Mapping) else None,
            "markdown_task_rows": [],
            "yaml_task_rows": [],
            "contract_rows": [],
        }

    roadmap_mapping = roadmap if isinstance(roadmap, Mapping) else {}
    declarations = _producer_declarations(roadmap_mapping)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    width = max(len(EXPECTED_TASK_IDS), len(markdown_tasks), len(yaml_tasks))
    rows: list[JsonDict] = []
    raw_tasks = roadmap_mapping.get("tasks")
    raw_tasks = raw_tasks if isinstance(raw_tasks, list) else []
    for index in range(width):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else None
        observed = yaml_tasks[index] if index < len(yaml_tasks) else None
        task_id = str((observed or expected or {}).get("id") or f"missing-{index + 1}")
        declaration_rows = declarations.get(task_id, [])
        checks = {
            field: bool(expected and observed and expected.get(field) == observed.get(field))
            for field in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
        }
        checks["producer_fields"] = all(row["passed"] for row in declaration_rows)
        failures = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "unit_id": task_id,
                "arm": "markdown_vs_yaml",
                "order": index + 1,
                "markdown": _public_task(expected),
                "yaml": _public_task(observed),
                "producer_declarations": declaration_rows,
                "checks": checks,
                "failures": failures,
                "passed": not failures,
            }
        )

    markdown_ids = [task["id"] for task in markdown_tasks]
    yaml_ids = [task["id"] for task in yaml_tasks]
    errors: list[str] = []
    if markdown["milestone"] != MILESTONE:
        errors.append("markdown_milestone")
    if parsed_yaml["milestone"] != MILESTONE:
        errors.append("yaml_milestone")
    if markdown_ids != list(EXPECTED_TASK_IDS):
        errors.append("markdown_task_order")
    if yaml_ids != list(EXPECTED_TASK_IDS):
        errors.append("yaml_task_order")
    if len(markdown_tasks) != len(EXPECTED_TASK_IDS):
        errors.append("markdown_task_count")
    if len(yaml_tasks) != len(EXPECTED_TASK_IDS):
        errors.append("yaml_task_count")
    if any(not row["passed"] for row in rows):
        errors.append("row_mismatch")
    return {
        "passed": not errors,
        "errors": errors,
        "markdown_milestone": markdown["milestone"],
        "yaml_milestone": parsed_yaml["milestone"],
        "markdown_task_rows": [_public_task(task) for task in markdown_tasks],
        "yaml_task_rows": [_public_task(task) for task in yaml_tasks],
        "contract_rows": rows,
    }


def load_contract_audit(root: Path) -> JsonDict:
    """Read the two active authorities without repairing either one."""

    return compare_contract_authorities(
        (root / DESIGN_PATH).read_text(encoding="utf-8"),
        load_yaml(root / ROADMAP_PATH),
    )


def _table_rows(markdown: str) -> tuple[list[str], list[int]]:
    """Locate only task rows inside the first exact contract table."""

    lines = markdown.splitlines()
    indices = [
        index for index, line in enumerate(lines) if re.match(r"^\|\s*\d+\s*\|\s*exp\d+", line)
    ]
    return lines, indices


def _mutate_markdown(markdown: str, mutation: str) -> str:
    """Change one Markdown authority field while leaving YAML untouched."""

    if mutation == "milestone":
        return markdown.replace("**Milestone:** `2026.09.652`", "**Milestone:** `2026.09.650`", 1)
    lines, indices = _table_rows(markdown)
    if mutation == "count":
        del lines[indices[-1]]
    elif mutation == "order":
        lines[indices[0]], lines[indices[1]] = lines[indices[1]], lines[indices[0]]
    else:
        target = indices[5] if mutation == "gates" else indices[0]
        cells = [cell.strip() for cell in lines[target].strip().strip("|").split("|")]
        columns = {
            "title": 2,
            "phase": 3,
            "deliverable": 4,
            "substrate": 5,
            "gates": 6,
        }
        position = columns[mutation]
        replacements = {
            "title": "changed title",
            "phase": "9",
            "deliverable": "results/changed.json",
            "substrate": "no_model_load",
            "gates": "None",
        }
        cells[position] = replacements[mutation]
        lines[target] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines) + "\n"


def _mutate_yaml(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Change one YAML authority field while leaving Markdown untouched."""

    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "milestone":
        changed["milestone"] = "2026.09.650"
    elif mutation == "count":
        changed["tasks"] = tasks[:-1]
    elif mutation == "order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "title":
        tasks[0]["title"] = "changed title"
    elif mutation == "deliverable":
        tasks[0]["deliverable"] = "results/changed.json"
    elif mutation == "phase":
        tasks[0]["phase"] = 9
    elif mutation == "substrate":
        tasks[0]["prompt"] = re.sub(
            r"inference_substrate_class=aggregation",
            "inference_substrate_class=no_model_load",
            tasks[0]["prompt"],
            count=1,
        )
    elif mutation == "gates":
        consumer = next(task for task in tasks if task.get("gated_on"))
        consumer["gated_on"] = deepcopy(consumer["gated_on"])
        consumer["gated_on"][0]["value"] = 0
    return changed


def run_contract_mutation_controls(
    markdown_text: str, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Mutate every compared field in each authority independently."""

    mutations = (
        "milestone",
        "count",
        "order",
        "title",
        "deliverable",
        "phase",
        "substrate",
        "gates",
    )
    rows: list[JsonDict] = []
    for authority in ("markdown", "yaml"):
        for mutation in mutations:
            candidate_markdown = (
                _mutate_markdown(markdown_text, mutation)
                if authority == "markdown"
                else markdown_text
            )
            candidate_yaml = (
                _mutate_yaml(roadmap, mutation) if authority == "yaml" else deepcopy(dict(roadmap))
            )
            comparison = compare_contract_authorities(candidate_markdown, candidate_yaml)
            if authority == "markdown":
                other = parse_yaml_contract(roadmap)
            else:
                other = parse_markdown_contract(markdown_text)
            other_readable = other["milestone"] == MILESTONE and len(other["tasks"]) == 13
            rows.append(
                {
                    "unit_id": f"{authority}:{mutation}",
                    "arm": "contract_mutation",
                    "authority": authority,
                    "mutation": mutation,
                    "rejected": comparison["passed"] is False,
                    "other_authority_readable": other_readable,
                    "errors": comparison["errors"],
                }
            )
    return rows


def independent_branch_schedule(contract: Mapping[str, Any]) -> list[JsonDict]:
    """Show that this advisory audit is not a root gate for science."""

    contract_passed = contract.get("passed") is True
    return [
        {
            "unit_id": task_id,
            "arm": "independent_branch_schedule",
            "task_id": task_id,
            "contract_audit_passed": contract_passed,
            "scheduled_independently": True,
            "contract_failure_effect": "none",
        }
        for task_id in EXPECTED_TASK_IDS[1:]
    ]


def run_conductor_gate_controls(private_root: Path) -> list[JsonDict]:
    """Exercise the shipped conductor reader on eight private evidence shapes."""

    cases: tuple[tuple[str, JsonDict | None, str, Any], ...] = (
        ("passing", {"status": "complete", "ready": 1}, "==", 1),
        ("zero", {"status": "complete", "ready": 0}, "==", 1),
        ("missing_file", None, "==", 1),
        ("missing_field", {"status": "complete"}, "==", 1),
        ("none", {"status": "complete", "ready": None}, "==", 1),
        ("wrong_type", {"status": "complete", "ready": "one"}, ">", 0),
        (
            "disqualified_class",
            {"status": "complete", "verdict_class": "disqualified"},
            "in",
            ["null", "positive"],
        ),
        (
            "flagged_evidence",
            {"status": "complete", "ready": 1, "flagged_adversarial": True},
            "==",
            1,
        ),
    )
    rows: list[JsonDict] = []
    for index, (name, payload, operator, expected) in enumerate(cases, 1):
        results_dir = private_root / f"{index:02d}_{name}"
        results_dir.mkdir(parents=True, exist_ok=True)
        task_id = f"exp98{index:04d}-{name.replace('_', '-')}"
        if payload is not None:
            artifact = results_dir / f"experiment_98{index:04d}_{name}.json"
            artifact.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        field = "verdict_class" if name == "disqualified_class" else "ready"
        check = evaluate_gates(
            {
                "gated_on": [
                    {
                        "upstream": task_id,
                        "artifact_field": field,
                        "op": operator,
                        "value": expected,
                    }
                ]
            },
            results_dir,
        )
        gate = check.gates_evaluated[0]
        flagged = bool(payload and payload.get("flagged_adversarial"))
        disqualified = bool(payload and payload.get("verdict_class") == "disqualified")
        rows.append(
            {
                "unit_id": name,
                "arm": "real_conductor_gate_reader",
                "case": name,
                "reader_passed": check.passed,
                "evidence_flagged": flagged,
                "evidence_disqualified": disqualified,
                "admissible": check.passed and not flagged and not disqualified,
                "upstream": gate.upstream,
                "artifact_field": gate.artifact_field,
                "operator": gate.op,
                "expected": deepcopy(gate.expected),
                "observed": deepcopy(gate.actual),
                "reason": gate.reason,
                "artifact_path": gate.artifact_path,
                "artifact_sha256": gate.artifact_sha256,
            }
        )
    return rows


def collect_prior_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate all thirteen V651 outcomes without relabeling them."""

    capstone = load_json(root / CAPSTONE_PATH)
    source_rows = capstone.get("task_dispositions")
    if not isinstance(source_rows, list):
        raise ValueError("V651 capstone task_dispositions list required")
    rows: list[JsonDict] = []
    for source in source_rows:
        if not isinstance(source, Mapping):
            continue
        declared = Path(str(source.get("declared_path")))
        observed = Path(str(source.get("observed_path")))
        path = root / observed
        available = path.is_file()
        payload = load_json(path) if available else {}
        observed_verdict = payload.get("honest_verdict")
        observed_flag = bool(payload.get("flagged_adversarial", False))
        expected_flag = bool(source.get("flagged_adversarial", False))
        authenticated = bool(
            available
            and observed_verdict == source.get("honest_verdict")
            and observed_flag == expected_flag
        )
        rows.append(
            {
                "unit_id": str(source.get("task_id")),
                "arm": "v651_disposition_authentication",
                "task_id": source.get("task_id"),
                "declared_path": declared.as_posix(),
                "declared_path_exists": (root / declared).is_file(),
                "observed_path": observed.as_posix(),
                "observed_path_exists": available,
                "source_kind": source.get("source_kind"),
                "sha256": sha256_file(path) if available else None,
                "original_status": payload.get("status"),
                "original_honest_verdict": source.get("honest_verdict"),
                "original_verdict_class": source.get("verdict_class"),
                "flagged_adversarial": expected_flag,
                "raw_rows_available": source.get("raw_rows_available"),
                "authenticated": authenticated,
                "evidence_scope": "historical_model_receipts",
                "counted_as_current_invocation": False,
            }
        )
    return rows


def completion_archive_state(root: Path) -> JsonDict:
    """Separate the V652 planning observation from the later archive update."""

    design = (root / DESIGN_PATH).read_text(encoding="utf-8")
    planning_statement = "The archive still\nended at V650 when planning began."
    planning_recorded = planning_statement in design
    document = load_yaml(root / COMPLETE_PATH)
    milestones = document.get("milestones")
    milestones = milestones if isinstance(milestones, list) else []
    current = (
        str(milestones[-1].get("id"))
        if milestones and isinstance(milestones[-1], Mapping)
        else None
    )
    return {
        "path": COMPLETE_PATH.as_posix(),
        "planning_authority_path": DESIGN_PATH.as_posix(),
        "planning_latest_milestone": "2026.09.650" if planning_recorded else None,
        "planning_lag_recorded": planning_recorded,
        "current_latest_milestone": current,
        "history_rewritten": False,
        "principle": "Keep the dated planning observation and the later archive state distinct.",
    }


def _direct_fetch(source: Mapping[str, str]) -> JsonDict:  # pragma: no cover - network boundary.
    """Read a small primary-page prefix and preserve access failure as data."""

    request = urllib.request.Request(
        source["url"],
        headers={"User-Agent": "carnot-v652-contract-audit/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            prefix = response.read(4096)
            return {
                "access_state": "ok",
                "http_status": int(response.status),
                "detail": f"read_prefix_bytes={len(prefix)}",
            }
    except urllib.error.HTTPError as exc:
        return {"access_state": "failed", "http_status": exc.code, "detail": str(exc)}
    except (OSError, urllib.error.URLError) as exc:
        return {"access_state": "failed", "http_status": None, "detail": str(exc)}


def check_selected_sources(fetcher: SourceFetcher = _direct_fetch) -> list[JsonDict]:
    """Check the six frozen primary sources once and retain every outcome."""

    rows: list[JsonDict] = []
    for source in SELECTED_SOURCES:
        outcome = fetcher(source)
        rows.append(
            {
                "unit_id": source["source_id"],
                "arm": "primary_source_access",
                **deepcopy(dict(source)),
                "access_state": outcome.get("access_state", "failed"),
                "http_status": outcome.get("http_status"),
                "detail": str(outcome.get("detail", "no detail"))[:500],
            }
        )
    return rows


def _check_one_source(source: Mapping[str, str], fetcher: SourceFetcher) -> JsonDict:
    """Normalize one bounded source request for the measured serial loop."""

    outcome = fetcher(source)
    return {
        "unit_id": source["source_id"],
        "arm": "primary_source_access",
        **deepcopy(dict(source)),
        "access_state": outcome.get("access_state", "failed"),
        "http_status": outcome.get("http_status"),
        "detail": str(outcome.get("detail", "no detail"))[:500],
    }


def method_mapping_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Map three paper ideas to exact V652 tasks, controls, and limits."""

    by_id = {str(row.get("source_id")): row for row in source_rows}

    def sources(*identifiers: str) -> list[JsonDict]:
        return [
            {
                "source_id": identifier,
                "version": by_id.get(identifier, {}).get("version"),
                "url": by_id.get(identifier, {}).get("url"),
                "access_state": by_id.get(identifier, {}).get("access_state"),
            }
            for identifier in identifiers
        ]

    return [
        {
            "unit_id": "joint_certification",
            "arm": "method_mapping",
            "method": "joint_certification",
            "primary_sources": sources("joint_certificate", "selective_risk"),
            "assumptions": [
                "The tuning split and certification split are disjoint within the reused corpus.",
                "Binary harm uses exact bounds and empty selected sets have undefined selected risk.",
            ],
            "task_hooks": [
                "exp7436-selection-protocol",
                "exp7439-certified-decisions",
                "exp7441-decision-audit",
            ],
            "code_hooks": ["selection protocol", "independent certification reducer"],
            "controls": ["old fixed thresholds", "per-head tuned policies", "all-escalate"],
            "deferrals": ["fresh deployment certificate", "generic inequality sweep"],
        },
        {
            "unit_id": "expert_aggregation",
            "arm": "method_mapping",
            "method": "expert_aggregation",
            "primary_sources": sources("expert_aggregation", "kan_locality", "kan_forgetting"),
            "assumptions": [
                "Predictions commit before delayed labels arrive.",
                "The empirical fixed-share replay inherits no unstated Bayesian guarantee.",
            ],
            "task_hooks": [
                "exp7438-mixture-prototype",
                "exp7440-mixture-learning",
                "exp7441-decision-audit",
            ],
            "code_hooks": ["frozen/adaptive expert bank", "prediction-time loss ledger"],
            "controls": ["equal weights", "frozen experts", "shuffled labels", "no feedback"],
            "deferrals": ["unchanged proof-memory trials", "generator training"],
        },
        {
            "unit_id": "output_representation",
            "arm": "method_mapping",
            "method": "output_representation",
            "primary_sources": sources("crane_representation"),
            "assumptions": [
                "A complete compact span is not a semantic certificate.",
                "Literal reconstruction must reject truncation and invalid offsets.",
            ],
            "task_hooks": [
                "exp7437-span-protocol",
                "exp7442-span-capture",
                "exp7443-span-audit",
            ],
            "code_hooks": ["compact span parser", "literal reconstruction", "semantic audit"],
            "controls": ["failed verbose canary", "source qualifier pairs", "invalid offsets"],
            "deferrals": ["finite-choice transport rerun", "grammar-is-truth claim"],
        },
    ]


def mandatory_priority_rows(root: Path) -> list[JsonDict]:
    """Record the two dated priorities without claiming either is closed here."""

    text = (root / KNOWN_ISSUES_PATH).read_text(encoding="utf-8")
    breaker = "NEW 2026-09-19: AUTORESEARCH CIRCUIT BREAKER" in text
    size_gate = "NEW 2026-09-18: GITHUB PACK-SIZE INCIDENT" in text
    return [
        {
            "unit_id": "2026-09-19-autoresearch-breaker",
            "arm": "mandatory_priority",
            "date": "2026-09-19",
            "priority": "autoresearch cumulative rejection breaker",
            "source_path": KNOWN_ISSUES_PATH.as_posix(),
            "source_present": breaker,
            "owner": "exp7435-round-breaker",
            "state": "assigned_next_task",
        },
        {
            "unit_id": "2026-09-18-conductor-size-gate",
            "arm": "mandatory_priority",
            "date": "2026-09-18",
            "priority": "hard conductor commit-size gate",
            "source_path": KNOWN_ISSUES_PATH.as_posix(),
            "source_present": size_gate,
            "owner": None,
            "state": "deferred_by_explicit_file_prohibition",
            "prohibited_path": "scripts/research_conductor.py",
            "unwired_helper_closes_priority": False,
        },
    ]


def collect_preconditions(root: Path, priors: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Name each exact resource and identity before dependent reduction."""

    rows: list[JsonDict] = []
    prerequisite_paths = [
        path
        for path in SOURCE_INPUT_PATHS
        if path not in {NOTE_PATH, MODULE_PATH, WRAPPER_PATH, TEST_PATH}
    ]
    for relative in prerequisite_paths:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative.as_posix()}",
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "field": "bytes",
                "expected": "readable_nonempty_bytes",
                "observed": "readable_nonempty_bytes" if available else None,
                "sha256": sha256_file(path) if available else None,
                "passed": available,
            }
        )
    spec = (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    rows.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": SPEC_PATH.as_posix(),
                "path": SPEC_PATH.as_posix(),
                "field": "REQ-*",
                "expected": "REQ-REPORT-7434",
                "observed": "REQ-REPORT-7434" if "REQ-REPORT-7434" in spec else None,
                "passed": "REQ-REPORT-7434" in spec,
            },
            {
                "check": "roadmap_identity",
                "upstream": ROADMAP_PATH.as_posix(),
                "path": ROADMAP_PATH.as_posix(),
                "field": "milestone",
                "expected": MILESTONE,
                "observed": load_yaml(root / ROADMAP_PATH).get("milestone"),
                "passed": load_yaml(root / ROADMAP_PATH).get("milestone") == MILESTONE,
            },
            {
                "check": "v651_disposition_authentication",
                "upstream": CAPSTONE_PATH.as_posix(),
                "path": CAPSTONE_PATH.as_posix(),
                "field": "authenticated_count",
                "expected": 13,
                "observed": sum(row.get("authenticated") is True for row in priors),
                "passed": len(priors) == 13
                and all(row.get("authenticated") is True for row in priors),
            },
        ]
    )
    return rows


def _source_hashes(root: Path, priors: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    """Bind current protocol bytes and typed historical sidecars."""

    hashes: dict[str, JsonDict] = {}
    for relative in SOURCE_INPUT_PATHS:
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "evidence_type": "current_protocol_or_input_bytes",
            }
    for row in priors:
        path = str(row.get("observed_path"))
        if path and (root / path).is_file():
            hashes[f"historical:{row.get('task_id')}"] = {
                "path": path,
                "sha256": sha256_file(root / path),
                "evidence_type": "historical_model_receipts",
                "original_verdict_class": row.get("original_verdict_class"),
                "flagged_adversarial": row.get("flagged_adversarial"),
                "counted_as_current_invocation": False,
            }
    return hashes


def _receipts_pass(receipts: object, names: Sequence[str]) -> bool:
    """Require one successful, untimed receipt for every exact check name."""

    if not isinstance(receipts, list):
        return False
    counts = Counter(str(row.get("name")) for row in receipts if isinstance(row, Mapping))
    return all(
        counts[name] == 1
        and any(
            row.get("name") == name
            and row.get("passed") is True
            and row.get("exit_code") == 0
            and row.get("timed_out") is False
            for row in receipts
            if isinstance(row, Mapping)
        )
        for name in names
    )


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Keep every gate operand plain and independently reviewable."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    priors: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    method_rows: Sequence[Mapping[str, Any]],
    priorities: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Reduce validity checks without turning method access into a benefit gate."""

    return [
        _gate(
            "exact_contract",
            "contract_validity",
            "==",
            True,
            contract.get("passed"),
            contract.get("passed") is True,
            "Both authorities must agree on all thirteen rows.",
        ),
        _gate(
            "independent_contract_mutations",
            "contract_validity",
            "==",
            16,
            sum(
                row.get("rejected") is True and row.get("other_authority_readable") is True
                for row in mutation_rows
            ),
            len(mutation_rows) == 16
            and all(
                row.get("rejected") is True and row.get("other_authority_readable") is True
                for row in mutation_rows
            ),
            "Each authority must fail closed without corrupting the other parser input.",
        ),
        _gate(
            "real_conductor_gate_controls",
            "gate_reader_validity",
            "==",
            8,
            len(gate_rows),
            len(gate_rows) == 8
            and [row.get("case") for row in gate_rows]
            == [
                "passing",
                "zero",
                "missing_file",
                "missing_field",
                "none",
                "wrong_type",
                "disqualified_class",
                "flagged_evidence",
            ]
            and sum(row.get("admissible") is True for row in gate_rows) == 1,
            "The real reader must expose each failure shape and flagged evidence stays inadmissible.",
        ),
        _gate(
            "v651_dispositions",
            "historical_evidence_validity",
            "==",
            13,
            sum(row.get("authenticated") is True for row in priors),
            len(priors) == 13 and all(row.get("authenticated") is True for row in priors),
            "Historical verdicts and flags must match their exact source bytes.",
        ),
        _gate(
            "bounded_method_ingestion",
            "method_accounting",
            "==",
            {"sources": 6, "methods": 3},
            {"sources": len(source_rows), "methods": len(method_rows)},
            len(source_rows) == 6 and len(method_rows) == 3,
            "Access failure is data; it does not erase a dated source decision.",
        ),
        _gate(
            "dated_priorities",
            "operations_accounting",
            "==",
            2,
            sum(row.get("source_present") is True for row in priorities),
            len(priorities) == 2 and all(row.get("source_present") is True for row in priorities),
            "Assigned and explicitly deferred priorities must remain distinct.",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            "==",
            True,
            validation.get("required_checks_passed"),
            validation.get("required_checks_passed") is True,
            "Only the frozen affected manifest controls current implementation validity.",
        ),
        _gate(
            "terminal_validation",
            "required_validation",
            "==",
            True,
            validation.get("terminal_validation_passed"),
            validation.get("terminal_validation_passed") is True,
            "Cold replay and unchanged terminal readers must pass before publication.",
        ),
    ]


def _failure_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name exact operands for every failed required gate."""

    failures = [
        {
            "upstream": row.get("check"),
            "path": RESULT_PATH.as_posix()
            if row.get("category") == "required_validation"
            else "authenticated_inputs",
            "check": row.get("check"),
            "field": "passed",
            "operator": row.get("operator"),
            "expected": deepcopy(row.get("expected")),
            "observed": deepcopy(row.get("observed")),
            "category": row.get("category"),
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {
        "passed": not failures,
        "failure_count": len(failures),
        "first_failure": deepcopy(failures[0]) if failures else None,
        "failures": failures,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain top-level intent separately from plain field values."""

    required = {
        "schema": "Use a versioned plain top-level schema with experiment_id, milestone and terminal status.",
        "run_date": "Use 20260919; record actual UTC start/end and monotonic duration.",
        "preconditions_checked": "Name each actual resource, path, identity and observed prerequisite before dependent work.",
        "MODEL_SPECS": "Name unsloth/Qwen3.8-27B-GGUF for every current LLM; use [] when none is invoked.",
        "model_invoked": "Current attempted model use is distinct from archived or scripted model-shaped data.",
        "invocation_counts": "Reconcile current loads and generations: attempted, completed, failed, cancelled and in-flight.",
        "inference_substrate": "Use a truthful string; keep device details in inference_substrate_details.",
        "inference_substrate_class": "Declare actual current compute; the duration floor follows this class.",
        "execution_venue": "Use host; record CPU, CUDA and external-device identity separately.",
        "duration_s": "Measure current work, separating model, computation, cold start and validation time.",
        "phase_spans": "Bind phase times, flushed progress events and resumed checkpoints.",
        "random_seed": "Freeze every fitting, sampling and resampling seed; null only when inapplicable.",
        "reproducibility_checksum": "Bind code, protocol, input bytes, row shards and exact validation scope.",
        "source_artifact_hashes": "Preserve source identity, original classes and flags through typed sidecars.",
        "rows": "Keep per-unit metrics for each arm, source group, seed and condition, including failures and unstarted units.",
        "sample_size_budget": "Separate planned, attempted, completed, failed, censored and unstarted independent units with a stopping rule.",
        "acceptance_gate_results": "Each gate names check, category, operator, expected, observed, passed and principle; validity differs from benefit.",
        "gate_check_summary": "Every blocked_* names exact upstream, path, check, field, expected and observed; distinguish missing, None and zero.",
        "verifier_is_oracle": "True when scoring authority is the deployed verifier; synthetic or exact-oracle benefit is circular.",
        "honest_verdict": "Completed findings start complete_; unchanged absent prerequisites use blocked_* with a specific cause.",
        "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. partial is only unfinished OWN retryable work.",
        "flagged_adversarial": "Preserve critical findings; flagged science cannot supply readiness.",
        "validation_receipts": "Record actual scoped argv/environment, exit codes, durations and hashed logs, including adversarial_verify.",
        "field_principles": "Explain field intent separately; gate scalars remain plain numbers and booleans.",
        "promotion_score": "Always zero; this milestone authorizes no automatic rollout, publication or generator-weight updates.",
        "contract_ready_score": "One only for the exact thirteen-row contract; this is advisory accounting.",
        "task_contract_rows": "Each independently parsed row makes count and ordering reviewable.",
        "method_mapping_rows": "Bind primary URLs, assumptions, code hooks, controls and deferrals.",
        "prior_dispositions": "Keep the original thirteen outcomes and any missing-path diagnosis.",
    }
    return {
        field: required.get(
            field, "Keep this field plain, reviewable, and bound by the artifact checksum."
        )
        for field in fields
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the checksum slot itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide deterministic disjoint spans for pure reducer tests."""

    return [
        {
            "phase": phase,
            "start_s": float(index),
            "end_s": float(index + 1),
            "duration_s": 1.0,
            "completed_units": 0,
            "heartbeat_count": 0,
            "checkpoint": "test_fixture",
        }
        for index, phase in enumerate(
            (
                "preconditions",
                "model_load",
                "generation",
                "contract",
                "gates",
                "sources",
                "validation",
            )
        )
    ]


def _terminal_state(
    preconditions_passed: bool, validation_passed: bool, contract_ready: bool
) -> tuple[str, str, str]:
    """Classify the receipt without treating advisory science as a gate."""

    if not preconditions_passed:
        return (
            "blocked_missing_authenticated_input",
            "blocked_missing_authenticated_input",
            "blocked",
        )
    if not validation_passed:
        return (
            "complete_disqualified_required_validation",
            "complete_disqualified_required_validation",
            "disqualified",
        )
    if not contract_ready:
        return (
            "complete_disqualified_contract_authority",
            "complete_disqualified_contract_authority",
            "disqualified",
        )
    return (
        "complete_advisory_contract_and_methods",
        "complete_null_exact_contract_methods_ingested",
        "null",
    )


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
    priors: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one complete advisory record from independently reducible rows."""

    methods = method_mapping_rows(source_rows)
    priorities = mandatory_priority_rows(root)
    branches = independent_branch_schedule(contract)
    preconditions = collect_preconditions(root, priors)
    gates = _acceptance_gates(
        contract,
        mutation_rows,
        gate_rows,
        priors,
        source_rows,
        methods,
        priorities,
        validation,
    )
    preconditions_passed = all(row.get("passed") is True for row in preconditions)
    validation_passed = all(
        row.get("passed") is True for row in gates if row.get("category") == "required_validation"
    )
    contract_ready = int(contract.get("passed") is True)
    status, honest_verdict, verdict_class = _terminal_state(
        preconditions_passed, validation_passed, bool(contract_ready)
    )

    rows = [
        *[
            deepcopy(dict(row)) | {"row_kind": "task_contract"}
            for row in contract.get("contract_rows", [])
        ],
        *[deepcopy(dict(row)) | {"row_kind": "contract_mutation"} for row in mutation_rows],
        *[deepcopy(dict(row)) | {"row_kind": "conductor_gate_control"} for row in gate_rows],
        *[deepcopy(dict(row)) | {"row_kind": "prior_disposition"} for row in priors],
        *[deepcopy(dict(row)) | {"row_kind": "source_access"} for row in source_rows],
        *[deepcopy(dict(row)) | {"row_kind": "method_mapping"} for row in methods],
        *[deepcopy(dict(row)) | {"row_kind": "mandatory_priority"} for row in priorities],
        *[deepcopy(dict(row)) | {"row_kind": "branch_schedule"} for row in branches],
    ]
    receipts = [deepcopy(dict(row)) for row in validation.get("validation_receipts", [])]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7434,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "status": status,
        "title": "Bind the thirteen-task contract and ingest methods against V651 evidence",
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_exact_declared_artifacts",
        "inference_substrate_details": {
            "cpu": "host process",
            "cuda": "not used",
            "external_device": "none",
            "network_request_limit": 6,
            "current_llm_operations": 0,
        },
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration_s,
        "duration_components_s": {
            "model": 0.0,
            "computation": sum(
                float(row.get("duration_s", 0.0))
                for row in phase_spans
                if row.get("phase") not in {"validation", "terminal_validation"}
            ),
            "cold_start": 0.0,
            "validation": sum(
                float(row.get("duration_s", 0.0))
                for row in phase_spans
                if row.get("phase") in {"validation", "terminal_validation"}
            ),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_hashes(root, priors),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows),
            "completed": len(rows),
            "failed": sum(row.get("access_state") == "failed" for row in source_rows),
            "censored": 0,
            "unstarted": 0,
            "independent_groups": {
                "contract_slots": len(contract.get("contract_rows", [])),
                "contract_mutations": len(mutation_rows),
                "gate_controls": len(gate_rows),
                "v651_dispositions": len(priors),
                "source_endpoints": len(source_rows),
                "method_mappings": len(methods),
                "dated_priorities": len(priorities),
                "independent_branches": len(branches),
            },
            "stop_rule": "Stop after thirteen contract rows, sixteen authority mutations, eight gate controls, thirteen prior dispositions, six source checks, three method mappings, two priorities, and twelve branch decisions.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _failure_summary(gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "field_principles": {},
        "promotion_score": 0,
        "contract_ready_score": contract_ready,
        "method_ingestion_complete_score": int(len(source_rows) == 6 and len(methods) == 3),
        "task_contract_rows": deepcopy(list(contract.get("contract_rows", []))),
        "contract_comparison": deepcopy(dict(contract)),
        "contract_mutation_rows": [deepcopy(dict(row)) for row in mutation_rows],
        "conductor_gate_control_rows": [deepcopy(dict(row)) for row in gate_rows],
        "branch_schedule_rows": branches,
        "method_mapping_rows": methods,
        "source_access_rows": [deepcopy(dict(row)) for row in source_rows],
        "prior_dispositions": [deepcopy(dict(row)) for row in priors],
        "completion_archive_state": completion_archive_state(root),
        "mandatory_priority_rows": priorities,
        "small_ebm_training": {
            "performed": False,
            "attempted": 0,
            "completed": 0,
            "principle": "This audit reads archived fitting receipts but trains no current energy head.",
        },
        "archived_model_event_sidecars": [
            {
                "task_id": row.get("task_id"),
                "path": row.get("observed_path"),
                "sha256": row.get("sha256"),
                "scope": "historical_model_receipts",
                "counted_as_current_invocation": False,
            }
            for row in priors
        ],
        "deferred_ideas": [
            "generator training",
            "unchanged proof-memory trials",
            "generic Ising sweeps",
            "conductor commit-size gate because scripts/research_conductor.py is prohibited",
        ],
        "numbered_e2e_applicable": [],
        "capability_e2e": ["declared entrypoint", "fresh-process cold artifact replay"],
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _gate_controls_valid(rows: object) -> bool:
    """Reduce the eight stored reader controls without rerunning private files."""

    if not isinstance(rows, list):
        return False
    cases = [row.get("case") for row in rows if isinstance(row, Mapping)]
    return cases == [
        "passing",
        "zero",
        "missing_file",
        "missing_field",
        "none",
        "wrong_type",
        "disqualified_class",
        "flagged_evidence",
    ] and [row.get("admissible") for row in rows] == [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def independent_reduce(artifact: Mapping[str, Any], *, require_terminal: bool = True) -> JsonDict:
    """Recompute terminal accounting from raw rows and exact receipts."""

    contract_rows = artifact.get("task_contract_rows")
    contract_ready = bool(
        isinstance(contract_rows, list)
        and len(contract_rows) == 13
        and [row.get("unit_id") for row in contract_rows if isinstance(row, Mapping)]
        == list(EXPECTED_TASK_IDS)
        and all(row.get("passed") is True for row in contract_rows if isinstance(row, Mapping))
    )
    mutation_rows = artifact.get("contract_mutation_rows")
    mutations_valid = bool(
        isinstance(mutation_rows, list)
        and len(mutation_rows) == 16
        and all(
            row.get("rejected") is True and row.get("other_authority_readable") is True
            for row in mutation_rows
            if isinstance(row, Mapping)
        )
    )
    priors = artifact.get("prior_dispositions")
    priors_valid = bool(
        isinstance(priors, list)
        and len(priors) == 13
        and all(row.get("authenticated") is True for row in priors if isinstance(row, Mapping))
    )
    sources = artifact.get("source_access_rows")
    methods = artifact.get("method_mapping_rows")
    methods_valid = bool(
        isinstance(sources, list)
        and len(sources) == 6
        and isinstance(methods, list)
        and len(methods) == 3
    )
    priorities = artifact.get("mandatory_priority_rows")
    priorities_valid = bool(
        isinstance(priorities, list)
        and len(priorities) == 2
        and all(row.get("source_present") is True for row in priorities if isinstance(row, Mapping))
    )
    receipts = artifact.get("validation_receipts")
    affected_valid = _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    terminal_valid = (
        True if not require_terminal else _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    )
    required_valid = bool(
        mutations_valid
        and _gate_controls_valid(artifact.get("conductor_gate_control_rows"))
        and priors_valid
        and methods_valid
        and priorities_valid
        and affected_valid
        and terminal_valid
    )
    preconditions = artifact.get("preconditions_checked")
    prerequisites_valid = bool(
        isinstance(preconditions, list)
        and preconditions
        and all(row.get("passed") is True for row in preconditions if isinstance(row, Mapping))
    )
    status, honest, verdict = _terminal_state(prerequisites_valid, required_valid, contract_ready)
    return {
        "contract_ready_score": int(contract_ready),
        "method_ingestion_complete_score": int(methods_valid),
        "required_checks_passed": required_valid,
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
    }


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash every declared local source and typed historical receipt."""

    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return False
    expected = _source_hashes(root, collect_prior_dispositions(root))
    return hashes == expected


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, raw reductions, sources, scores, and checksum."""

    if not isinstance(value, Mapping):
        return ["artifact_mapping_required"]
    artifact = dict(value)
    errors: list[str] = []
    if (
        artifact.get("schema"),
        artifact.get("experiment_id"),
        artifact.get("milestone"),
        artifact.get("run_date"),
    ) != (SCHEMA, EXPERIMENT_ID, MILESTONE, RUN_DATE):
        errors.append("identity_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("model_contract_invalid")

    contract = load_contract_audit(root)
    if artifact.get("task_contract_rows") != contract.get("contract_rows"):
        errors.append("task_contract_rows_mismatch")
    if artifact.get("contract_comparison") != contract:
        errors.append("contract_comparison_mismatch")
    expected_mutations = run_contract_mutation_controls(
        (root / DESIGN_PATH).read_text(encoding="utf-8"),
        load_yaml(root / ROADMAP_PATH),
    )
    if artifact.get("contract_mutation_rows") != expected_mutations:
        errors.append("contract_mutations_invalid")
    if not _gate_controls_valid(artifact.get("conductor_gate_control_rows")):
        errors.append("gate_controls_invalid")

    expected_priors = collect_prior_dispositions(root)
    if artifact.get("prior_dispositions") != expected_priors:
        errors.append("prior_dispositions_mismatch")
    sources = artifact.get("source_access_rows")
    source_ids = [row["source_id"] for row in SELECTED_SOURCES]
    if (
        not isinstance(sources, list)
        or [row.get("source_id") for row in sources if isinstance(row, Mapping)] != source_ids
    ):
        errors.append("source_access_rows_invalid")
        sources = []
    expected_methods = method_mapping_rows(sources)
    if artifact.get("method_mapping_rows") != expected_methods:
        errors.append("method_mapping_rows_mismatch")
    if artifact.get("mandatory_priority_rows") != mandatory_priority_rows(root):
        errors.append("priority_rows_mismatch")
    if artifact.get("completion_archive_state") != completion_archive_state(root):
        errors.append("completion_archive_mismatch")
    if artifact.get("branch_schedule_rows") != independent_branch_schedule(contract):
        errors.append("branch_schedule_mismatch")

    receipts = artifact.get("validation_receipts")
    if not _receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES):
        errors.append("affected_validation_invalid")
    if require_terminal and not _receipts_pass(receipts, TERMINAL_CHECK_NAMES):
        errors.append("terminal_validation_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_score_nonzero")

    reduction = independent_reduce(artifact, require_terminal=require_terminal)
    for key, error in (
        ("contract_ready_score", "contract_score_mismatch"),
        ("method_ingestion_complete_score", "method_score_mismatch"),
        ("status", "status_mismatch"),
        ("honest_verdict", "honest_verdict_mismatch"),
        ("verdict_class", "verdict_class_mismatch"),
    ):
        if artifact.get(key) != reduction[key]:
            errors.append(error)
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 file-scoped command plan for this experiment."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets and command drift before a child starts."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _phase_span(  # pragma: no cover - authentic runtime timing boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    units: int,
    checkpoint: str,
) -> JsonDict:
    """Close one measured phase and identify its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "heartbeat_count": 0,
        "checkpoint": checkpoint,
    }


def _terminal_commands(root: Path, candidate: Path) -> list[PlannedCommand]:
    """Build cold replay, raw reduction, and unchanged terminal readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7434_v652_contract_methods import validate_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "e=validate_artifact(v,require_terminal=False);"
        "print(json.dumps({'errors':e},sort_keys=True),flush=True);"
        "raise SystemExit(bool(e))"
    )
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (
                python,
                "-u",
                WRAPPER_PATH.as_posix(),
                "--date",
                RUN_DATE,
                "--validate",
                str(candidate),
            ),
            "candidate_capability_e2e",
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", "-c", reducer, str(candidate)),
            "candidate_raw_reduction",
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(candidate),
            ),
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def _write_method_records(root: Path, source_rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the durable V652 note and one idempotent studying-ledger entry."""

    methods = method_mapping_rows(source_rows)
    access_lines = [
        f"- `{row['version']}`: `{row['access_state']}` ({row.get('http_status')}); {row['url']}"
        for row in source_rows
    ]
    method_lines = [
        f"- **{row['method']}** -> {', '.join(row['task_hooks'])}. "
        f"Controls: {', '.join(row['controls'])}. Deferrals: {', '.join(row['deferrals'])}."
        for row in methods
    ]
    note = "\n".join(
        [
            "# V652 method ingestion",
            "",
            "Date: 2026-09-19. Scope: advisory contract accounting and bounded source ingestion.",
            "External findings are method inputs, not Carnot measurements.",
            "",
            "## Primary access receipts",
            "",
            *access_lines,
            "",
            "## Method-to-roster mapping",
            "",
            *method_lines,
            "",
            "## Preserved limits",
            "",
            "Generator training, unchanged proof-memory trials, and generic Ising sweeps remain deferred.",
            "Exp7435 owns the autoresearch breaker. The conductor size gate remains deferred because",
            "this task cannot modify `scripts/research_conductor.py`. No unwired size helper closes it.",
            "",
        ]
    )
    (root / NOTE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (root / NOTE_PATH).write_text(note, encoding="utf-8")

    marker = "<!-- EXP7434-V652-METHOD-INGESTION -->"
    study_path = root / STUDY_PATH
    studying = study_path.read_text(encoding="utf-8")
    if marker not in studying:
        addition = "\n".join(
            [
                "",
                marker,
                "## 2026-09-19 Exp7434 — V652 methods — INGESTED",
                "",
                "Joint certification, expert aggregation, and compact output representation were",
                "mapped to the V652 roster. Exact access receipts and limits are in",
                "`docs/research-notes/v652-method-ingestion.md`. External claims are not Carnot",
                "measurements. Generator training, unchanged proof-memory trials, and generic",
                "Ising sweeps remain deferred.",
                "",
            ]
        )
        study_path.write_text(studying.rstrip() + "\n" + addition, encoding="utf-8")


def run_experiment(  # pragma: no cover - exercised by the declared capability E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Measure inputs, run scoped checks, and atomically publish the receipt."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    priors = collect_prior_dispositions(root)
    preconditions = collect_preconditions(root, priors)
    if not all(row["passed"] for row in preconditions):
        failed = next(row for row in preconditions if not row["passed"])
        raise RuntimeError(
            "blocked_missing_authenticated_input:"
            f"{failed['upstream']}:{failed['field']}:{failed['observed']}"
        )
    spans.append(
        _phase_span(
            "preconditions",
            phase_started,
            started,
            units=len(preconditions),
            checkpoint="authenticated_inputs",
        )
    )
    progress(started, "preconditions", "after", completed_units=len(preconditions))

    for phase in ("model_load", "generation"):
        phase_started = time.monotonic()
        progress(started, phase, "before", completed_units=0)
        spans.append(
            _phase_span(
                phase,
                phase_started,
                started,
                units=0,
                checkpoint="no_current_llm_work",
            )
        )
        progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    progress(started, "contract", "before")
    roadmap = load_yaml(root / ROADMAP_PATH)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(markdown, roadmap)
    mutations = run_contract_mutation_controls(markdown, roadmap)
    spans.append(
        _phase_span(
            "contract",
            phase_started,
            started,
            units=len(contract.get("contract_rows", [])) + len(mutations),
            checkpoint="authorities_and_mutations_reduced",
        )
    )
    progress(
        started,
        "contract",
        "after",
        completed_units=len(contract.get("contract_rows", [])) + len(mutations),
    )

    phase_started = time.monotonic()
    progress(started, "gate_controls", "before", planned_units=8)
    private_gates = Path(tempfile.mkdtemp(prefix="carnot-exp7434-gates-", dir="/tmp"))
    gate_rows = run_conductor_gate_controls(private_gates)
    spans.append(
        _phase_span(
            "gate_controls",
            phase_started,
            started,
            units=len(gate_rows),
            checkpoint="real_conductor_reader_controls",
        )
    )
    progress(started, "gate_controls", "after", completed_units=len(gate_rows))

    phase_started = time.monotonic()
    progress(started, "source_access", "before", planned_units=len(SELECTED_SOURCES))
    source_rows: list[JsonDict] = []
    for index, source in enumerate(SELECTED_SOURCES, 1):
        progress(
            started,
            "source_access",
            "before_request",
            completed_units=index - 1,
            pending=source["source_id"],
        )
        source_rows.append(_check_one_source(source, _direct_fetch))
        progress(
            started,
            "source_access",
            "after_request",
            completed_units=index,
            source=source["source_id"],
        )
    _write_method_records(root, source_rows)
    spans.append(
        _phase_span(
            "source_access",
            phase_started,
            started,
            units=len(source_rows),
            checkpoint="source_rows_and_method_note",
        )
    )
    progress(started, "source_access", "after", completed_units=len(source_rows))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7434-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{plan_errors}")
    phase_started = time.monotonic()
    progress(started, "validation", "before_subprocesses", planned_units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "validation",
            phase_started,
            started,
            units=len(affected),
            checkpoint="affected_checks_complete",
        )
    )
    progress(started, "validation", "after_subprocesses", passed=reduced["passed"])

    candidate = build_artifact(
        root,
        contract,
        mutations,
        gate_rows,
        priors,
        source_rows,
        {
            "validation_receipts": affected,
            "required_checks_passed": reduced["passed"],
            "terminal_validation_passed": True,
        },
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    candidate_errors = validate_artifact(candidate, root=root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")

    terminal_commands = _terminal_commands(root, candidate_path)
    phase_started = time.monotonic()
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        planned_units=len(terminal_commands),
    )
    terminal = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    spans.append(
        _phase_span(
            "terminal_validation",
            phase_started,
            started,
            units=len(terminal),
            checkpoint="cold_replay_and_strict_readers_complete",
        )
    )
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        root,
        contract,
        mutations,
        gate_rows,
        priors,
        source_rows,
        {
            "validation_receipts": [*affected, *terminal],
            "required_checks_passed": reduced["passed"],
            "terminal_validation_passed": terminal_passed,
        },
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic", path=RESULT_PATH.as_posix())
    atomic_json(candidate_path, final)
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publish", "after_atomic", path=RESULT_PATH.as_posix())
    return final


def date_argument(value: str) -> str:
    """Accept only the execution date frozen by the V652 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse normal execution and cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date_argument)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(  # pragma: no cover - exercised by the declared capability E2E.
    argv: Sequence[str] | None = None,
) -> int:
    """Run V652 contract ingestion or validate one measured candidate."""

    print("[exp7434] phase=startup event=flushed", flush=True)
    args = parse_args(argv)
    if args.validate is not None:
        errors = validate_artifact(load_json(args.validate), require_terminal=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    artifact = run_experiment(REPO_ROOT, args.date)
    print(
        json.dumps(
            {
                "artifact": RESULT_PATH.as_posix(),
                "status": artifact["status"],
                "contract_ready_score": artifact["contract_ready_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI boundary.
    raise SystemExit(main())

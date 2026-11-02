"""Bind the V658 contract without rehabilitating disqualified V657 evidence.

This advisory report performs aggregation only. It preserves the static audit
failure, the disqualified capstone, and the independently qualified online null.

Spec refs: REQ-REPORT-7516 and SCENARIO-REPORT-7516-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import time
from typing import Any

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
from carnot.experiment_7503_v657_contract_methods import (
    canonical_hash,
    load_json,
    load_yaml,
    receipts_pass,
    receipts_recorded,
    sha256_file,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json
from scripts import overdue_priority_lint


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.658"
EXPERIMENT_ID = "exp7516-contract-methods"
SCHEMA = "carnot.exp7516.v658.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7516_v658_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7516_v658_contract_methods")
FULL_SUITE_ATTEMPT_PATH = RAW_DIR / "full_python_suite_attempt.json"
MODULE_PATH = Path("python/carnot/experiment_7516_v658_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7516_v658_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7516_v658_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v658-method-map.md")
STUDY_PATH = Path("research-studying.md")
STRICT_LOG_PATH = Path(
    "results/raw/experiment_7508_v657_static_audit/validation/terminal/"
    "03_verdict_row_consistency_strict/00_verdict_row_consistency_strict.log"
)
E6_LIVE_PATH = Path("results/experiment_7491_e6_timed_live_profile.json")
E6_COST_PATH = Path("results/experiment_7492_e6_timed_cost_profile.json")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7516, "contract-methods"),
        (7517, "source-protocol"),
        (7518, "source-pilot"),
        (7519, "source-fit-capture"),
        (7520, "source-eval-capture"),
        (7521, "consistency-energy"),
        (7522, "source-evaluation"),
        (7523, "count-memory"),
        (7524, "count-online"),
        (7525, "decision-audit"),
        (7526, "arc-eligibility"),
        (7527, "arc-opportunities"),
        (7528, "service-boundary"),
        (7529, "capstone"),
    )
)

V657_ARTIFACTS = (
    (
        "exp7503-contract-methods",
        "exp7503-contract-methods",
        "results/experiment_7503_v657_contract_methods.json",
    ),
    (
        "exp7504-evidence-interface",
        "exp7504-v657-evidence-interface",
        "results/experiment_7504_v657_evidence_interface.json",
    ),
    (
        "exp7505-energy-fit",
        "exp7505-v657-energy-fit",
        "results/experiment_7505_v657_energy_fit.json",
    ),
    (
        "exp7506-causal-prototype",
        "exp7506-v657-causal-prototype",
        "results/experiment_7506_v657_causal_prototype.json",
    ),
    (
        "exp7507-static-evaluation",
        "exp7507-v657-static-evaluation",
        "results/experiment_7507_v657_static_evaluation.json",
    ),
    (
        "exp7508-static-audit",
        "exp7508-v657-static-audit",
        "results/experiment_7508_v657_static_audit.json",
    ),
    (
        "exp7509-causal-online",
        "exp7509-v657-causal-online",
        "results/experiment_7509_v657_causal_online.json",
    ),
    (
        "exp7510-causal-audit",
        "exp7510-v657-causal-audit",
        "results/experiment_7510_v657_causal_audit.json",
    ),
    (
        "exp7511-arc-evidence-recovery",
        "exp7511-arc-evidence-recovery",
        "results/experiment_7511_v657_arc_evidence_recovery.json",
    ),
    (
        "exp7512-arc-opportunity",
        "exp7512-arc-opportunity",
        "results/experiment_7512_v657_arc_opportunity.json",
    ),
    (
        "exp7513-placement-continuity",
        "exp7513-v657-placement-continuity",
        "results/experiment_7513_v657_placement_continuity.json",
    ),
    (
        "exp7514-service-trace",
        "exp7514-v657-service-trace",
        "results/experiment_7514_v657_service_trace.json",
    ),
    ("exp7515-capstone", "exp7515-capstone", "results/experiment_7515_v657_capstone.json"),
)

ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}

SOURCE_INPUT_PATHS = tuple(
    dict.fromkeys(
        (
            Path("AGENTS.md"),
            Path("CLAUDE.md"),
            Path("CODEX.md"),
            Path("research-program.md"),
            Path("ops/exclusion_manifest.yaml"),
            Path("ops/e2e-test-plan.md"),
            Path("scripts/experiment_template.py"),
            Path("python/carnot/reporting/current_work_receipt.py"),
            Path("python/carnot/experiment_7358_v646_validation_contract.py"),
            Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
            Path("python/carnot/experiment_7503_v657_contract_methods.py"),
            Path("python/carnot/experiment_7515_v657_capstone.py"),
            Path("research-references.md"),
            STUDY_PATH,
            Path("ops/known-issues.md"),
            Path("scripts/roadmap_schema.py"),
            Path("scripts/validate_prior_failures.py"),
            Path("scripts/audit_roadmap_gates.py"),
            Path("scripts/exclusion_manifest_lint.py"),
            Path("scripts/arc_levelup_guarantee_lint.py"),
            Path("scripts/harness_consumer_checks.py"),
            Path("scripts/overdue_priority_lint.py"),
            DESIGN_PATH,
            SPEC_PATH,
            STRICT_LOG_PATH,
            E6_LIVE_PATH,
            E6_COST_PATH,
            *(Path(path) for _task, _identity, path in V657_ARTIFACTS),
        )
    )
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
REQUIRED_REPOSITORY_CHECK_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "roadmap_gate_audit",
    "exclusion_manifest",
    "arc_floor",
    "valid_null_guard",
)
BASELINE_CHECK_NAMES = ("harness_boundary", "overdue_priority", "full_python_suite")
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

METHOD_SPECS: tuple[JsonDict, ...] = (
    {
        "method_family": "consistency",
        "primary_url": "https://arxiv.org/html/2606.08158",
        "source_revision": "arXiv:2606.08158v1 (2026-06-06)",
        "source_section": "Section 2.1, constrained training objective",
        "adaptation": "Constrain the two semantic option orders while retaining unconstrained and same-information controls.",
        "counterexample": "Absent or mismatched source text is not label-preserving and cannot inherit the original label.",
        "task_mapping": ["exp7521-consistency-energy", "exp7522-source-evaluation"],
    },
    {
        "method_family": "calibration",
        "primary_url": "https://arxiv.org/html/2609.11446v1",
        "source_revision": "arXiv:2609.11446v1 (2026-09-10)",
        "source_section": "Sections 3-4 and Appendix B split protocol",
        "adaptation": "Freeze probability calibration before selecting the accept, reject, or escalate cost policy.",
        "counterexample": "Marginal calibration does not guarantee risk for each selected subgroup.",
        "task_mapping": [
            "exp7517-source-protocol",
            "exp7521-consistency-energy",
            "exp7522-source-evaluation",
        ],
    },
    {
        "method_family": "online_proper_loss",
        "primary_url": "https://arxiv.org/html/2607.19689v1",
        "source_revision": "arXiv:2607.19689v1 (2026-07-22)",
        "source_section": "Section 4 and Section 7.2 quadratic audit primitive",
        "adaptation": "Compare chronological excess Brier loss with frozen, prevalence, and legally shuffled forecasts.",
        "counterexample": "A count memory is not the Blackwell algorithm and delayed partial feedback inherits no theorem.",
        "task_mapping": ["exp7523-count-memory", "exp7524-count-online"],
    },
    {
        "method_family": "thermodynamic_co_design",
        "primary_url": "https://extropic.ai/writing/z1t",
        "source_revision": "Extropic Z1T (2026-09-04)",
        "source_section": "Porting Transformers, disaggregated inference, and hardware encoding",
        "adaptation": "Preserve sparse operation shape and the complete FPGA-host service boundary.",
        "counterexample": "Vendor estimates and a fast local kernel are not measured Carnot board or service speedups.",
        "task_mapping": ["exp7528-service-boundary"],
    },
)


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    """Return one aware UTC timestamp for a durable boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush phase and subprocess boundaries so long work stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7516] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v658_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer staged V658 bytes, then accept the matching active roadmap."""

    candidates: list[JsonDict] = []
    selected: tuple[Path, JsonDict] | None = None
    for relative in (NEXT_ROADMAP_PATH, ACTIVE_ROADMAP_PATH):
        path = root / relative
        try:
            value = load_yaml(path)
            observed: Any = value.get("milestone")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            value = {}
            observed = None if isinstance(exc, FileNotFoundError) else f"{type(exc).__name__}:{exc}"
        matches = observed == MILESTONE
        candidates.append(
            {
                "path": relative.as_posix(),
                "exists": path.is_file(),
                "expected": MILESTONE,
                "observed": observed,
                "matches_milestone": matches,
                "principle": "Milestone matching prevents stale authority selection.",
            }
        )
        if selected is None and matches:
            selected = (path, value)
    if selected is None:
        raise ValueError("V658 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Retain only fields that both independent authorities declare."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _parse_design(markdown_text: str) -> JsonDict:
    """Adapt the V658 ``Title`` header to the shared exact-title reader."""

    normalized = markdown_text.replace(
        "| Order | Task ID | Title |", "| Order | Task ID | Exact title |", 1
    )
    normalized = normalized.replace("| Structured gates |", "| Structured gate |", 1)
    return parse_markdown_contract(normalized)


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Check that every structured gate consumes an earlier declared field."""

    raw = roadmap.get("tasks")
    tasks = raw if isinstance(raw, list) else []
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
        gates = consumer.get("gated_on") or []
        for gate in gates if isinstance(gates, list) else []:
            if not isinstance(gate, Mapping):
                continue
            upstream = gate.get("upstream")
            producer_entry = positions.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            field = gate.get("artifact_field")
            precedes = producer_index is not None and producer_index < consumer_index
            declared = _field_declared(str(producer.get("prompt", "")), field)
            rows.append(
                {
                    "upstream": upstream,
                    "artifact_field": field,
                    "producer_precedes_consumer": precedes,
                    "declared_verbatim": declared,
                    "passed": precedes and declared,
                    "principle": "A gate cannot consume an undeclared or later field.",
                }
            )
        result[str(consumer.get("id"))] = rows
    return result


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare fourteen ordered identities and every executable public field."""

    try:
        markdown = _parse_design(markdown_text)
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
    declarations = _producer_declarations(roadmap if isinstance(roadmap, Mapping) else {})
    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    width = max(len(EXPECTED_TASK_IDS), len(markdown_tasks), len(yaml_tasks))
    rows: list[JsonDict] = []
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
                "principle": "Exact paired fields prevent a readable but different task.",
            }
        )
    errors: list[str] = []
    markdown_ids = [task["id"] for task in markdown_tasks]
    yaml_ids = [task["id"] for task in yaml_tasks]
    if markdown["milestone"] != MILESTONE:
        errors.append("markdown_milestone")
    if parsed_yaml["milestone"] != MILESTONE:
        errors.append("yaml_milestone")
    if markdown_ids != list(EXPECTED_TASK_IDS):
        errors.append("markdown_task_order")
    if yaml_ids != list(EXPECTED_TASK_IDS):
        errors.append("yaml_task_order")
    if len(markdown_tasks) != 14:
        errors.append("markdown_task_count")
    if len(yaml_tasks) != 14:
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


def _table_rows(markdown: str) -> tuple[list[str], list[int]]:
    """Locate the numbered rows in the exact task contract table."""

    lines = markdown.splitlines()
    indices = [
        index for index, line in enumerate(lines) if re.match(r"^\|\s*\d+\s*\|\s*exp\d+", line)
    ]
    return lines, indices


def _mutate_markdown(markdown: str, mutation: str) -> str:
    """Plant one private Markdown defect without changing the real design."""

    if mutation == "milestone":
        return re.sub(r"2026\.09\.658", "2026.09.657", markdown, count=1)
    lines, indices = _table_rows(markdown)
    if mutation == "count":
        del lines[indices[-1]]
    elif mutation == "order":
        lines[indices[0]], lines[indices[1]] = lines[indices[1]], lines[indices[0]]
    elif mutation == "gate_field":
        lines[indices[2]] = lines[indices[2]].replace(
            "source_protocol_ready_score", "changed_score", 1
        )
    else:
        cells = [cell.strip() for cell in lines[indices[0]].strip().strip("|").split("|")]
        position = {"id": 1, "title": 2, "path": 4}[mutation]
        cells[position] = {
            "id": "exp9999-changed",
            "title": "Changed title",
            "path": "results/changed.json",
        }[mutation]
        lines[indices[0]] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines) + "\n"


def _mutate_yaml(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Plant one private YAML defect without changing the active roadmap."""

    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "milestone":
        changed["milestone"] = "2026.09.657"
    elif mutation == "count":
        changed["tasks"] = tasks[:-1]
    elif mutation == "order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "id":
        tasks[0]["id"] = "exp9999-changed"
    elif mutation == "title":
        tasks[0]["title"] = "Changed title"
    elif mutation == "path":
        tasks[0]["deliverable"] = "results/changed.json"
    elif mutation == "gate_field":
        tasks[2]["gated_on"][0]["artifact_field"] = "changed_score"
    return changed


def run_contract_mutation_controls(
    markdown_text: str, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Require seven defects in each independent authority to fail."""

    mutations = ("count", "order", "id", "title", "path", "gate_field", "milestone")
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
            other = (
                parse_yaml_contract(roadmap)
                if authority == "markdown"
                else _parse_design(markdown_text)
            )
            rows.append(
                {
                    "unit_id": f"{authority}:{mutation}",
                    "arm": "contract_mutation",
                    "authority": authority,
                    "mutation": mutation,
                    "rejected": comparison["passed"] is False,
                    "other_authority_readable": (
                        other["milestone"] == MILESTONE and len(other["tasks"]) == 14
                    ),
                    "errors": comparison["errors"],
                    "principle": "A planted defect must fail without changing its peer.",
                }
            )
    return rows


def _compact_failed_receipts(payload: Mapping[str, Any]) -> list[JsonDict]:
    """Retain failed validation names and exits without copying large log tails."""

    return [
        {
            "name": row.get("name"),
            "exit_code": row.get("exit_code"),
            "log_path": row.get("log_path"),
            "log_sha256": row.get("log_sha256"),
            "required": row.get("required"),
        }
        for row in payload.get("validation_receipts") or []
        if isinstance(row, Mapping) and row.get("passed") is not True
    ]


def collect_v657_dispositions(root: Path) -> list[JsonDict]:
    """Inventory all V657 terminal bytes while preserving original limits."""

    strict_text = (root / STRICT_LOG_PATH).read_text(encoding="utf-8")
    advisory_count = strict_text.count("NO_HEADROOM_MAJORITY")
    blocking_count = 0 if "0 blocking, 8 advisory" in strict_text else None
    rows: list[JsonDict] = []
    for order, (task_id, expected_identity, path_text) in enumerate(V657_ARTIFACTS, start=1):
        path = root / path_text
        present = path.is_file()
        payload = load_json(path) if present else {}
        receipts = payload.get("validation_receipts") or []
        strict_receipts = [
            row
            for row in receipts
            if isinstance(row, Mapping) and row.get("name") == "verdict_row_consistency_strict"
        ]
        special_ok = True
        if task_id == "exp7508-static-audit":
            special_ok = (
                payload.get("verdict_class") == "disqualified"
                and payload.get("flagged_adversarial") is True
                and advisory_count == 8
                and blocking_count == 0
                and len(strict_receipts) == 1
                and strict_receipts[0].get("exit_code") not in (None, 0)
            )
        elif task_id == "exp7515-capstone":
            special_ok = (
                payload.get("verdict_class") == "disqualified"
                and payload.get("flagged_adversarial") is True
            )
        elif task_id in {"exp7509-causal-online", "exp7510-causal-audit"}:
            special_ok = (
                payload.get("verdict_class") == "null"
                and payload.get("flagged_adversarial") is False
            )
        identity_ok = (
            present
            and payload.get("experiment_id") == expected_identity
            and payload.get("milestone") == "2026.09.657"
            and str(payload.get("honest_verdict", "")).startswith("complete_")
            and payload.get("verdict_class")
            in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
        )
        strict_exit = strict_receipts[0].get("exit_code") if len(strict_receipts) == 1 else None
        rows.append(
            {
                "unit_id": task_id,
                "arm": "v657_custody",
                "order": order,
                "task_id": task_id,
                "expected_path": path_text,
                "producer_present": present,
                "source_sha256": sha256_file(path) if present else None,
                "original_experiment_id": payload.get("experiment_id"),
                "original_honest_verdict": payload.get("honest_verdict"),
                "original_verdict_class": payload.get("verdict_class"),
                "original_flagged_adversarial": payload.get("flagged_adversarial"),
                "failed_validation_receipts": _compact_failed_receipts(payload),
                "strict_reader_exit_code": strict_exit
                if task_id == "exp7508-static-audit"
                else None,
                "strict_no_headroom_advisories": advisory_count
                if task_id == "exp7508-static-audit"
                else None,
                "strict_blocking_findings": blocking_count
                if task_id == "exp7508-static-audit"
                else None,
                "signed_cost_difference_is_headroom_metric": False
                if task_id == "exp7508-static-audit"
                else None,
                "scientific_result_established": payload.get("verdict_class")
                not in {"blocked", "disqualified", "partial"},
                "authenticated": bool(identity_ok and special_ok),
                "failures": []
                if identity_ok and special_ok
                else ["v657_custody_authentication_failed"],
                "principle": "Literal prior verdicts prevent completion from rehabilitating invalid evidence.",
            }
        )
    return rows


def valid_null_fixture() -> JsonDict:
    """Build equal-cost rows that state why no decision headroom exists."""

    rows = [
        {
            "unit_id": f"valid-null-{index}",
            "arm_a_cost": cost,
            "arm_b_cost": cost,
            "no_headroom": True,
            "positive_claim": False,
            "headroom_explanation": "Both absolute arm costs are equal on this factual row.",
            "disposition": "complete_null_equal_absolute_costs",
        }
        for index, cost in enumerate((0.2, 0.4, 0.8), start=1)
    ]
    return {
        "schema": "carnot.v658.valid_null_fixture.v1",
        "honest_verdict": "complete_null_valid_no_headroom_fixture",
        "verdict_class": "null",
        "positive_claim": False,
        "rows": rows,
    }


def validate_valid_null_fixture(value: object) -> list[str]:
    """Require absolute costs and factual null context on every fixture row."""

    if not isinstance(value, Mapping):
        return ["fixture_mapping_required"]
    errors: list[str] = []
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        return ["fixture_rows_required"]
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"row_mapping_required:{index}")
            continue
        costs = (row.get("arm_a_cost"), row.get("arm_b_cost"))
        if any(not isinstance(cost, (int, float)) or isinstance(cost, bool) for cost in costs):
            errors.append(f"row_absolute_costs_missing:{index}")
        if row.get("no_headroom") is not True or costs[0] != costs[1]:
            errors.append(f"row_no_headroom_invalid:{index}")
        if row.get("positive_claim") is not False or not row.get("headroom_explanation"):
            errors.append(f"row_null_context_invalid:{index}")
    return errors


def method_rows() -> list[JsonDict]:
    """Return four source-grounded adaptations without importing claims."""

    return [
        {
            "unit_id": row["method_family"],
            "arm": "method_ingestion",
            **deepcopy(row),
            "access_status": "primary_method_section_read_20260922",
            "external_claim_is_carnot_measurement": False,
            "failures": [],
            "principle": "A primary method can define a control without becoming local evidence.",
        }
        for row in METHOD_SPECS
    ]


def write_method_records(root: Path) -> None:
    """Write the bounded map and append one idempotent studying marker."""

    lines = [
        "# V658 method map",
        "",
        "Date: 2026-09-22. Scope: advisory method accounting.",
        "Paper results are not local Carnot results.",
        "",
        "| Family | Primary source and section | Exact adaptation | Counterexample | Destination task |",
        "|---|---|---|---|---|",
    ]
    for row in method_rows():
        source = f"[{row['source_revision']}]({row['primary_url']}), {row['source_section']}"
        lines.append(
            f"| {row['method_family']} | {source} | {row['adaptation']} | "
            f"{row['counterexample']} | {', '.join(row['task_mapping'])} |"
        )
    lines.extend(
        [
            "",
            "The primary method sections were read in a bounded low-concurrency pass.",
            "Source access succeeded for these four rows. No benchmark number is a local result.",
            "",
        ]
    )
    note = root / NOTE_PATH
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines), encoding="utf-8")

    marker = "<!-- EXP7516-V658-METHOD-INGESTION -->"
    study = root / STUDY_PATH
    text = study.read_text(encoding="utf-8")
    if marker not in text:
        addition = "\n".join(
            [
                "",
                marker,
                "## 2026-09-22 Exp7516 — V658 methods — INGESTED",
                "",
                "Consistency, probability calibration, online proper loss, and thermodynamic",
                "co-design map to separate V658 tasks. External results remain source context.",
                "Exact sections, counterexamples, and destinations are in",
                "`docs/research-notes/v658-method-map.md`.",
                "",
            ]
        )
        study.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


def priority_disposition(root: Path, roadmap_path: Path) -> JsonDict:
    """Record every parsed priority and each real overdue violation."""

    known = (root / "ops/known-issues.md").read_text(encoding="utf-8")
    complete = (root / "research-complete.yaml").read_text(encoding="utf-8")
    roadmap = roadmap_path.read_text(encoding="utf-8")
    priorities = overdue_priority_lint._parse_priorities(known)
    violations = overdue_priority_lint._check(priorities, complete, roadmap)
    return {
        "unit_id": "overdue_priority",
        "arm": "repository_health",
        "source_path": "ops/known-issues.md",
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "parsed_priority_count": len(priorities),
        "open_priorities": [
            {"filed_date": row.filed_date.isoformat(), "title": row.title, "slug": row.slug}
            for row in priorities
        ],
        "unresolved_count": len(violations),
        "unresolved_priorities": violations,
        "affects_contract_authority_match": False,
        "required_current_validation": False,
        "principle": "Open priorities stay visible without changing the active roadmap.",
    }


def e6_timed_profile(root: Path) -> JsonDict:
    """Retain exact E6 evidence separately from V657 coarse stage timers."""

    live = load_json(root / E6_LIVE_PATH)
    cost = load_json(root / E6_COST_PATH)
    decision = cost["decision_point_profile"]
    induction = decision["decision_points"]["induction_and_generation"]
    return {
        "unit_id": "outer_loop_e6_timed_profile",
        "arm": "historical_timing_evidence",
        "scope": "historical",
        "live_path": E6_LIVE_PATH.as_posix(),
        "live_sha256": sha256_file(root / E6_LIVE_PATH),
        "cost_path": E6_COST_PATH.as_posix(),
        "cost_sha256": sha256_file(root / E6_COST_PATH),
        "fully_timed_complete_episodes": cost["sample_size"]["fully_timed_complete_episodes"],
        "fully_timed_games": cost["sample_size"]["fully_timed_games"],
        "induction_generation_wall_fraction": induction["wall_fraction"],
        "induction_generation_tokens": induction["tokens"],
        "historical_capture_duration_s": live["duration_s"],
        "separate_from_v657_coarse_timers": True,
        "v657_coarse_timer_paths": [
            "results/experiment_7512_v657_arc_opportunity.json",
            "results/experiment_7514_v657_service_trace.json",
        ],
        "principle": "Exclusive E6 spans cannot be pooled with unresolved coarse stage labels.",
    }


def _precondition(check: str, path: str, field: str, expected: Any, observed: Any) -> JsonDict:
    """Record one exact prerequisite value and read-only owner."""

    return {
        "check": check,
        "upstream": path,
        "path": path,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected,
        "ownership": "read_only_input",
        "principle": "An exact prerequisite prevents reduction over missing or different bytes.",
    }


def collect_preconditions(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Authenticate named files, special dispositions, and method prerequisites."""

    rows: list[JsonDict] = []
    for relative in SOURCE_INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        row = _precondition(
            f"source_bytes:{relative.as_posix()}",
            relative.as_posix(),
            "bytes",
            "readable_nonempty_bytes",
            "readable_nonempty_bytes" if available else None,
        )
        row["sha256"] = sha256_file(path) if available else None
        rows.append(row)
    by_id = {str(row.get("task_id")): row for row in dispositions}
    roadmap = load_yaml(roadmap_path)
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    timed = e6_timed_profile(root)
    rows.extend(
        [
            _precondition(
                "driving_requirement",
                SPEC_PATH.as_posix(),
                "REQ-*",
                "REQ-REPORT-7516",
                "REQ-REPORT-7516" if "REQ-REPORT-7516" in spec else None,
            ),
            _precondition(
                "roadmap_identity",
                roadmap_path.relative_to(root).as_posix(),
                "milestone",
                MILESTONE,
                roadmap.get("milestone"),
            ),
            _precondition(
                "v657_artifact_count",
                "results/experiment_75*_v657_*.json",
                "present",
                13,
                len(dispositions),
            ),
            _precondition(
                "v657_custody",
                "results/experiment_75*_v657_*.json",
                "authenticated",
                13,
                sum(row.get("authenticated") is True for row in dispositions),
            ),
            _precondition(
                "exp7508_strict_advisories",
                STRICT_LOG_PATH.as_posix(),
                "NO_HEADROOM_MAJORITY",
                8,
                by_id.get("exp7508-static-audit", {}).get("strict_no_headroom_advisories"),
            ),
            _precondition(
                "exp7508_disqualified",
                V657_ARTIFACTS[5][2],
                "verdict_class",
                "disqualified",
                by_id.get("exp7508-static-audit", {}).get("original_verdict_class"),
            ),
            _precondition(
                "exp7515_disqualified",
                V657_ARTIFACTS[-1][2],
                "verdict_class",
                "disqualified",
                by_id.get("exp7515-capstone", {}).get("original_verdict_class"),
            ),
            _precondition(
                "online_null_qualified",
                V657_ARTIFACTS[6][2],
                "verdict_class",
                "null",
                by_id.get("exp7509-causal-online", {}).get("original_verdict_class"),
            ),
            _precondition(
                "valid_null_shape",
                "private_fixture",
                "validation_errors",
                [],
                validate_valid_null_fixture(valid_null_fixture()),
            ),
            _precondition(
                "e6_timed_episodes",
                E6_COST_PATH.as_posix(),
                "fully_timed_complete_episodes",
                36,
                timed["fully_timed_complete_episodes"],
            ),
            _precondition(
                "host_identity", "/proc/cpuinfo", "machine_nonempty", True, bool(platform.machine())
            ),
        ]
    )
    return rows


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Bind current protocol files and original V657 bytes separately."""

    current = (
        *SOURCE_INPUT_PATHS,
        roadmap_path.relative_to(root),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        NOTE_PATH,
    )
    hashes: dict[str, JsonDict] = {}
    for relative in dict.fromkeys(current):
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "evidence_type": "current_protocol_or_input_bytes",
            }
    for row in dispositions:
        path_text = row.get("expected_path")
        if isinstance(path_text, str) and (root / path_text).is_file():
            hashes[f"historical:{row.get('task_id')}"] = {
                "path": path_text,
                "sha256": sha256_file(root / path_text),
                "evidence_type": "historical_terminal_bytes",
                "original_verdict_class": row.get("original_verdict_class"),
                "flagged_adversarial": row.get("original_flagged_adversarial"),
                "counted_as_current_invocation": False,
            }
    return hashes


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the shared scoped plan for the frozen affected manifest."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def build_repository_check_plan(
    root: Path, roadmap_path: Path, fixture_path: Path
) -> list[PlannedCommand]:
    """Build unchanged guards and separate pre-existing repository health."""

    python = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    selected = str(roadmap_path)
    schema_code = (
        "import pathlib,sys,yaml;from scripts.roadmap_schema import Roadmap;"
        "p=pathlib.Path(sys.argv[1]);Roadmap.model_validate(yaml.safe_load(p.read_text()));"
        "print('roadmap schema clean',flush=True)"
    )
    overdue_code = (
        "import pathlib,sys;import scripts.overdue_priority_lint as lint;"
        "lint.ROADMAP_NEXT=pathlib.Path(sys.argv[1]);sys.argv=sys.argv[:1];"
        "raise SystemExit(lint.main())"
    )
    specs = (
        ("roadmap_schema", (python, "-u", "-c", schema_code, selected), "selected_roadmap", True),
        (
            "prior_failure",
            (python, "-u", "scripts/validate_prior_failures.py", selected),
            "selected_roadmap",
            True,
        ),
        (
            "roadmap_gate_audit",
            (python, "-u", "scripts/audit_roadmap_gates.py", selected),
            "selected_roadmap",
            True,
        ),
        (
            "exclusion_manifest",
            (python, "-u", "scripts/exclusion_manifest_lint.py", selected),
            "selected_roadmap",
            True,
        ),
        (
            "arc_floor",
            (python, "-u", "scripts/arc_levelup_guarantee_lint.py", selected),
            "selected_roadmap",
            True,
        ),
        (
            "valid_null_guard",
            (
                python,
                "-u",
                "scripts/verdict_row_consistency_lint.py",
                "--strict",
                str(fixture_path),
            ),
            "private_guard_fixture",
            True,
        ),
        (
            "harness_boundary",
            (python, "-u", "scripts/harness_consumer_checks.py", "prompt-paths", selected),
            "repository_health",
            False,
        ),
        (
            "overdue_priority",
            (python, "-u", "-c", overdue_code, selected),
            "repository_health",
            False,
        ),
        ("full_python_suite", (pytest, "tests/python", "-q"), "repository_health", False),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                name, argv, scope, 1800.0 if name == "full_python_suite" else 900.0
            ),
            "required_validation" if required else "diagnostic_repository_health",
            required,
        )
        for name, argv, scope, required in specs
    ]


def _gate(
    check: str, gate_type: str, expected: Any, observed: Any, passed: bool, principle: str
) -> JsonDict:
    """Attach exact operands and the failure that each gate prevents."""

    return {
        "check": check,
        "gate_type": gate_type,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": "==",
        "passed": passed,
        "principle": principle,
    }


def terminal_state(
    preconditions_passed: bool, validation_passed: bool, contract_ready: bool
) -> tuple[str, str, str]:
    """Keep external absence, invalid evidence, and an advisory null distinct."""

    if not preconditions_passed:
        value = "complete_blocked_external_v658_contract_input"
        return value, value, "blocked"
    if not validation_passed or not contract_ready:
        value = "complete_disqualified_v658_contract_or_validation"
        return value, value, "disqualified"
    return (
        "complete_advisory_v658_contract_and_methods",
        "complete_null_v658_contract_methods_ingested",
        "null",
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed check with its exact expected and observed values."""

    failed = [
        {
            "check": row.get("check"),
            "upstream": "current_exp7516_reduction",
            "field_path": f"acceptance_gate_results.{row.get('check')}",
            "expected": deepcopy(row.get("expected")),
            "observed": deepcopy(row.get("observed")),
            "op": row.get("op"),
            "principle": row.get("principle"),
        }
        for row in gates
        if row.get("passed") is not True
    ]
    return {"passed": not failed, "failed_count": len(failed), "failed_checks": failed}


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    priority: Mapping[str, Any],
    timed: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep validity, readiness, source support, and benefit independent."""

    preconditions_ok = all(row.get("passed") is True for row in preconditions)
    return [
        _gate(
            "preconditions_authenticated",
            "validity",
            True,
            preconditions_ok,
            preconditions_ok,
            "Favorable rows cannot excuse missing prerequisite bytes.",
        ),
        _gate(
            "exact_v658_contract",
            "validity",
            True,
            contract.get("passed") is True,
            contract.get("passed") is True,
            "Paired authorities prevent task and gate drift.",
        ),
        _gate(
            "private_contract_mutations_rejected",
            "validity",
            14,
            sum(row.get("rejected") is True for row in mutations),
            len(mutations) == 14 and all(row.get("rejected") is True for row in mutations),
            "Every named authority mutation must be detected.",
        ),
        _gate(
            "v657_custody_authenticated",
            "validity",
            13,
            sum(row.get("authenticated") is True for row in dispositions),
            len(dispositions) == 13
            and all(row.get("authenticated") is True for row in dispositions),
            "Literal custody prevents disqualified evidence from becoming valid.",
        ),
        _gate(
            "valid_null_shape_qualified",
            "validity",
            [],
            validate_valid_null_fixture(valid_null_fixture()),
            not validate_valid_null_fixture(valid_null_fixture()),
            "Absolute arm costs and factual context prevent signed-delta headroom claims.",
        ),
        _gate(
            "affected_validation",
            "validity",
            True,
            validation.get("affected_checks_passed") is True,
            validation.get("affected_checks_passed") is True,
            "Scoped checks prevent unrelated suite state from laundering changed code.",
        ),
        _gate(
            "required_repository_guards",
            "validity",
            True,
            validation.get("repository_checks_passed") is True,
            validation.get("repository_checks_passed") is True,
            "Schema, prior-failure, gate, exclusion, ARC, and null-shape guards retain policy.",
        ),
        _gate(
            "terminal_readers",
            "validity",
            True,
            validation.get("terminal_checks_passed") is True,
            validation.get("terminal_checks_passed") is True,
            "Fresh readers prevent the producer from serving as its own oracle.",
        ),
        _gate(
            "method_rows_source_grounded",
            "support",
            4,
            len(methods),
            len(methods) == 4
            and all(row.get("external_claim_is_carnot_measurement") is False for row in methods),
            "External method results cannot become local measurements.",
        ),
        _gate(
            "open_priorities_recorded",
            "support",
            True,
            priority.get("parsed_priority_count") == len(priority.get("open_priorities") or []),
            priority.get("parsed_priority_count") == len(priority.get("open_priorities") or []),
            "Every parsed priority remains visible even when its lint is diagnostic.",
        ),
        _gate(
            "e6_profile_separate",
            "support",
            True,
            timed.get("separate_from_v657_coarse_timers") is True,
            timed.get("separate_from_v657_coarse_timers") is True,
            "Exclusive E6 evidence cannot absorb V657 coarse timers.",
        ),
        _gate(
            "scientific_benefit_not_claimed",
            "benefit",
            False,
            False,
            True,
            "Advisory readiness cannot become a scientific benefit claim.",
        ),
    ]


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain how every emitted top-level field prevents a failure."""

    required = {
        "schema": "Version, experiment_id and milestone bind the terminal reader contract.",
        "run_date": "The frozen date and measured clocks distinguish this work from history.",
        "preconditions_checked": "Exact resources and hashes prevent invented readiness.",
        "MODEL_SPECS": "An empty list states that this aggregation loaded no model.",
        "model_specs": "The lowercase mirror prevents a reader from inferring a model.",
        "model_invoked": "Current calls remain separate from historical provenance.",
        "invocation_counts": "Attempted and terminal call states reconcile to zero.",
        "inference_substrate_class": "The aggregation class prevents a false runtime floor.",
        "inference_substrate": "Canonical aggregation spelling prevents substrate ambiguity.",
        "execution_venue": "Host reduction stays distinct from GPU and board evidence.",
        "duration_s": "Measured current elapsed work is not historical capture time.",
        "phase_spans": "Named boundaries expose unfinished or silent work.",
        "random_seed": "Frozen bookkeeping seeds cannot multiply scientific support.",
        "reproducibility_checksum": "The checksum binds settings, rows, sources, and receipts.",
        "source_artifact_hashes": "Exact bytes retain exposure history and adversarial flags.",
        "rows": "Every contract, custody, method, fixture, priority, and timing unit stays reducible.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, and unstarted stay distinct.",
        "acceptance_gate_results": "Each check gives operands, outcome, and prevented failure.",
        "gate_check_summary": "Blocked or invalid work names exact failed operands.",
        "honest_verdict": "A complete terminal prefix prevents retry-state ambiguity.",
        "verdict_class": "The closed class separates null, blocked, and disqualified evidence.",
        "verifier_is_oracle": "No oracle fixture supports a positive advisory claim.",
        "flagged_adversarial": "Actual guard findings cannot be cleared to open a gate.",
        "validation_receipts": "Exact commands, exits, log hashes, and required status remain auditable.",
        "field_principles": "Every field states the reporting failure it prevents.",
        "contract_ready_score": "Bare readiness covers exact authorities and mutations, not science.",
        "method_ingestion_complete_score": "Bare method completion remains independent of benefit.",
        "prior_dispositions": "Valid null, exposed data, and disqualified evidence stay separate.",
        "task_contract_rows": "Exactly fourteen ordered identities and gates form the contract.",
    }
    return {
        field: required.get(field, f"The {field} field preserves exact scope and meaning.")
        for field in fields
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete stable record except the checksum's own value."""

    payload = deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide named zero-work spans for deterministic reducer tests."""

    return [
        {
            "phase": phase,
            "start_s": 0.0,
            "end_s": 0.0,
            "duration_s": 0.0,
            "completed_units": 0,
            "heartbeat_count": 0,
            "checkpoint": "unit_test_fixture",
            "principle": "A named zero-work span prevents invented runtime evidence.",
        }
        for phase in (
            "preconditions",
            "model_load",
            "generation",
            "contract",
            "method_ingestion",
            "validation",
            "terminal_validation",
        )
    ]


def build_artifact(
    root: Path,
    roadmap_path: Path,
    roadmap_candidates: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one reducible advisory record from exact inputs and receipts."""

    contract_copy = deepcopy(dict(contract))
    mutations = deepcopy(list(mutation_rows))
    prior = deepcopy(list(dispositions))
    methods = method_rows()
    preconditions = collect_preconditions(root, roadmap_path, prior)
    priority = priority_disposition(root, roadmap_path)
    timed = e6_timed_profile(root)
    fixture = valid_null_fixture()
    receipts = deepcopy(list(validation.get("validation_receipts") or []))
    gates = _acceptance_gates(
        contract_copy, mutations, prior, methods, preconditions, validation, priority, timed
    )
    preconditions_ok = all(row.get("passed") is True for row in preconditions)
    validity_ok = all(
        row.get("passed") is True for row in gates if row.get("gate_type") == "validity"
    )
    contract_ready = bool(preconditions_ok and validity_ok and contract_copy.get("passed"))
    status, honest, verdict = terminal_state(preconditions_ok, validity_ok, contract_ready)
    rows = [
        *deepcopy(contract_copy.get("contract_rows") or []),
        *mutations,
        *prior,
        *deepcopy(methods),
        deepcopy(priority),
        deepcopy(timed),
        {
            "unit_id": "valid_null_shape",
            "arm": "guard_qualification",
            "fixture": fixture,
            "failures": validate_valid_null_fixture(fixture),
            "principle": "A valid null records absolute arm costs before later measurement.",
        },
    ]
    duration_s = max(0.0, (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000)
    validation_duration = sum(float(row.get("duration_s", 0.0)) for row in receipts)
    baseline = [row for row in receipts if row.get("name") in BASELINE_CHECK_NAMES]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7516,
        "title": "Bind fourteen tasks and qualify the V657 scientific limits",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": status,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "process_identity": {"pid": os.getpid(), "owner": "current_process"},
        "duration_s": duration_s,
        "clock_identity": {"utc": "datetime.now(UTC)", "monotonic": "time.monotonic_ns"},
        "phase_spans": deepcopy(list(phase_spans)),
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "device_identity": {
            "machine": platform.machine(),
            "node": platform.node(),
            "processor": platform.processor(),
            "cuda_used": False,
        },
        "duration_components_s": {
            "authoring": 0.0,
            "computation": max(0.0, duration_s - validation_duration),
            "validation": validation_duration,
            "historical_capture": timed["historical_capture_duration_s"],
            "total_current": duration_s,
        },
        "random_seed": {
            "selection": None,
            "fitting": None,
            "arrival": None,
            "bootstrap": None,
            "contract_mutations": 7_516_658_01,
            "order": 7_516_658_02,
            "explanation": "This deterministic aggregation makes no stochastic estimate.",
        },
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "selected_authority_state": "staged"
        if roadmap_path.name.endswith("-next.yaml")
        else "active",
        "roadmap_resolution_candidates": deepcopy(list(roadmap_candidates)),
        "contract_comparison": contract_copy,
        "task_contract_rows": deepcopy(contract_copy.get("contract_rows") or []),
        "contract_mutation_rows": mutations,
        "prior_dispositions": prior,
        "method_rows": deepcopy(methods),
        "valid_null_fixture_qualification": fixture,
        "priority_disposition": deepcopy(priority),
        "e6_timed_profile": deepcopy(timed),
        "source_artifact_hashes": _source_hashes(root, roadmap_path, prior),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows),
            "complete": len(rows),
            "completed": len(rows),
            "excluded": 0,
            "failed": 2,
            "censored": 8,
            "unstarted": 0,
            "independent_unit": "contract_mutation_disposition_method_fixture_priority_or_timing_row",
            "counting_rule": "Prior failures and advisories describe custody and do not remove rows.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "contract_ready_score": int(contract_ready),
        "method_ingestion_complete_score": int(
            len(methods) == 4
            and all(
                row.get("primary_url")
                and row.get("source_section")
                and row.get("adaptation")
                and row.get("counterexample")
                and row.get("task_mapping")
                for row in methods
            )
        ),
        "honest_verdict": honest,
        "verdict_class": verdict,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "repository_health": {
            "baseline_receipts": baseline,
            "failed_baseline_checks": [
                row.get("name") for row in baseline if not row.get("passed")
            ],
            "affects_required_scoped_validation": False,
        },
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "documentation_paths": [
                NOTE_PATH.as_posix(),
                STUDY_PATH.as_posix(),
                SPEC_PATH.as_posix(),
            ],
            "full_python_suite_run_once": any(
                row.get("name") == "full_python_suite" for row in receipts
            ),
            "numbered_runtime_e2e_applicable": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": True,
            "numbered_runtime_e2e_applicable": False,
        },
        "advisory_only": True,
        "later_science_tasks_gate_on_exp7516": False,
        "roadmap_activation_performed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_modified": False,
        "publication_performed": False,
        "submission_performed": False,
        "push_performed": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any], *, require_terminal: bool = True) -> JsonDict:
    """Recompute readiness and terminal class from embedded raw rows."""

    contract = artifact.get("contract_comparison")
    mutations = artifact.get("contract_mutation_rows")
    dispositions = artifact.get("prior_dispositions")
    methods = artifact.get("method_rows")
    preconditions = artifact.get("preconditions_checked")
    receipts = artifact.get("validation_receipts")
    contract_ok = isinstance(contract, Mapping) and contract.get("passed") is True
    mutations_ok = (
        isinstance(mutations, list)
        and len(mutations) == 14
        and all(isinstance(row, Mapping) and row.get("rejected") is True for row in mutations)
    )
    dispositions_ok = (
        isinstance(dispositions, list)
        and len(dispositions) == 13
        and all(
            isinstance(row, Mapping) and row.get("authenticated") is True for row in dispositions
        )
    )
    methods_ok = (
        isinstance(methods, list)
        and len(methods) == 4
        and all(
            isinstance(row, Mapping) and row.get("external_claim_is_carnot_measurement") is False
            for row in methods
        )
    )
    preconditions_ok = (
        isinstance(preconditions, list)
        and bool(preconditions)
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in preconditions)
    )
    fixture_ok = not validate_valid_null_fixture(artifact.get("valid_null_fixture_qualification"))
    affected_ok = receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    repository_ok = receipts_pass(receipts, REQUIRED_REPOSITORY_CHECK_NAMES)
    terminal_ok = receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    validation_ok = affected_ok and repository_ok and terminal_ok
    ready = (
        contract_ok
        and mutations_ok
        and dispositions_ok
        and preconditions_ok
        and fixture_ok
        and validation_ok
    )
    status, honest, verdict = terminal_state(preconditions_ok, validation_ok, ready)
    return {
        "contract_ready_score": int(ready),
        "method_ingestion_complete_score": int(methods_ok),
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
    }


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash every declared source from the selected current worktree."""

    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return False
    try:
        roadmap_path, _roadmap, _candidates = resolve_v658_roadmap(root)
        expected = _source_hashes(root, roadmap_path, collect_v657_dispositions(root))
    except (OSError, ValueError, json.JSONDecodeError, KeyError):
        return False
    return hashes == expected


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, custody, rows, hashes, receipts, and reduction."""

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
        or artifact.get("model_specs") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
        or artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("model_contract_invalid")
    try:
        roadmap_path, roadmap, _candidates = resolve_v658_roadmap(root)
        markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
        contract = compare_contract_authorities(markdown, roadmap)
        mutations = run_contract_mutation_controls(markdown, roadmap)
        dispositions = collect_v657_dispositions(root)
        priority = priority_disposition(root, roadmap_path)
        timed = e6_timed_profile(root)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        return [*errors, f"source_reduction_failed:{type(exc).__name__}"]
    if artifact.get("task_contract_rows") != contract.get("contract_rows"):
        errors.append("task_contract_rows_mismatch")
    if artifact.get("contract_comparison") != contract:
        errors.append("contract_comparison_mismatch")
    if artifact.get("contract_mutation_rows") != mutations:
        errors.append("contract_mutations_invalid")
    if artifact.get("prior_dispositions") != dispositions:
        errors.append("prior_dispositions_mismatch")
    if artifact.get("method_rows") != method_rows():
        errors.append("method_rows_mismatch")
    if artifact.get("valid_null_fixture_qualification") != valid_null_fixture():
        errors.append("valid_null_fixture_mismatch")
    if artifact.get("priority_disposition") != priority:
        errors.append("priority_disposition_mismatch")
    if artifact.get("e6_timed_profile") != timed:
        errors.append("e6_timed_profile_mismatch")
    expected_rows = [
        *deepcopy(contract.get("contract_rows") or []),
        *deepcopy(mutations),
        *deepcopy(dispositions),
        *method_rows(),
        deepcopy(priority),
        deepcopy(timed),
        {
            "unit_id": "valid_null_shape",
            "arm": "guard_qualification",
            "fixture": valid_null_fixture(),
            "failures": [],
            "principle": "A valid null records absolute arm costs before later measurement.",
        },
    ]
    if artifact.get("rows") != expected_rows:
        errors.append("rows_mismatch")
    receipts = artifact.get("validation_receipts")
    if not receipts_recorded(receipts, validation_scope.REQUIRED_CHECK_NAMES):
        errors.append("affected_validation_invalid")
    if not receipts_recorded(receipts, REQUIRED_REPOSITORY_CHECK_NAMES):
        errors.append("repository_validation_invalid")
    if not receipts_recorded(receipts, BASELINE_CHECK_NAMES):
        errors.append("repository_health_invalid")
    if require_terminal and not receipts_recorded(receipts, TERMINAL_CHECK_NAMES):
        errors.append("terminal_validation_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
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
    if (
        not isinstance(principles, Mapping)
        or set(principles) != set(artifact)
        or any(not principle for principle in principles.values())
    ):
        errors.append("field_principles_invalid")
    gates = artifact.get("acceptance_gate_results")
    if (
        not isinstance(gates, list)
        or not gates
        or any(
            not isinstance(row, Mapping)
            or not row.get("principle")
            or row.get("gate_type") not in {"validity", "support", "benefit"}
            or not {"expected", "observed", "op", "passed"}.issubset(row)
            for row in gates
        )
    ):
        errors.append("gate_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _phase_span(
    phase: str, phase_started: float, run_started: float, *, units: int, checkpoint: str
) -> JsonDict:  # pragma: no cover - measured runtime boundary.
    """Close one real phase and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "start_s": phase_started - run_started,
        "end_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "heartbeat_count": 0,
        "checkpoint": checkpoint,
        "principle": "A measured span exposes unfinished work and prevents duration padding.",
    }


def _terminal_commands(root: Path, candidate: Path) -> list[PlannedCommand]:
    """Build entrypoint replay, cold reduction, and unchanged strict readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7516_v658_contract_methods import validate_artifact;"
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
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate_row_consistency",
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def _blocked_artifact(
    missing: Path,
    field: str,
    expected: Any,
    observed: Any,
    started_at: str,
    started_ns: int,
) -> JsonDict:  # pragma: no cover - external absence boundary.
    """Publish exact external absence without inventing dependent rows."""

    ended_ns = time.monotonic_ns()
    reason = re.sub(r"[^a-z0-9]+", "_", missing.name.lower()).strip("_") or "prerequisite"
    gate = _gate(
        "external_prerequisite_available",
        "validity",
        expected,
        observed,
        False,
        "A missing external input blocks dependent reduction.",
    )
    verdict = f"complete_blocked_missing_{reason}"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7516,
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": verdict,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "process_identity": {"pid": os.getpid(), "owner": "current_process"},
        "duration_s": max(0.0, (ended_ns - started_ns) / 1_000_000_000),
        "phase_spans": [],
        "preconditions_checked": [
            _precondition(
                "external_prerequisite_available", missing.as_posix(), field, expected, observed
            )
        ],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_components_s": {"authoring": 0.0, "computation": 0.0, "validation": 0.0},
        "random_seed": {},
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned": 0,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "passed": False,
            "failed_count": 1,
            "failed_checks": [
                {
                    "check": gate["check"],
                    "upstream": missing.as_posix(),
                    "field_path": field,
                    "expected": deepcopy(expected),
                    "observed": deepcopy(observed),
                    "op": "==",
                    "principle": gate["principle"],
                }
            ],
        },
        "honest_verdict": verdict,
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "contract_ready_score": 0,
        "method_ingestion_complete_score": 0,
        "task_contract_rows": [],
        "prior_dispositions": [],
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Run bounded checks and atomically publish only validated terminal bytes."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_ns = time.monotonic_ns()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    missing = next((path for path in SOURCE_INPUT_PATHS if not (root / path).is_file()), None)
    if missing is not None:
        blocked = _blocked_artifact(
            missing, "bytes", "readable_nonempty_bytes", None, started_at, started_ns
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=missing.as_posix())
        return blocked
    try:
        roadmap_path, roadmap, roadmap_candidates = resolve_v658_roadmap(root)
        dispositions = collect_v657_dispositions(root)
        preconditions = collect_preconditions(root, roadmap_path, dispositions)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        missing_path = Path(f"authority:{type(exc).__name__}")
        blocked = _blocked_artifact(missing_path, "readable", True, False, started_at, started_ns)
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", error=type(exc).__name__)
        return blocked
    if not all(row["passed"] for row in preconditions):
        failed = next(row for row in preconditions if not row["passed"])
        blocked = _blocked_artifact(
            Path(str(failed["path"])),
            str(failed["field"]),
            failed["expected"],
            failed["observed"],
            started_at,
            started_ns,
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=failed["path"])
        return blocked
    spans.append(
        _phase_span(
            "preconditions", phase_started, started, units=len(preconditions), checkpoint="inputs"
        )
    )
    progress(started, "preconditions", "after", completed_units=len(preconditions))

    for phase in ("model_load", "generation"):
        phase_started = time.monotonic()
        progress(started, phase, "before", completed_units=0)
        spans.append(
            _phase_span(phase, phase_started, started, units=0, checkpoint="no_current_llm_work")
        )
        progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    progress(started, "contract", "before")
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(markdown, roadmap)
    mutations = run_contract_mutation_controls(markdown, roadmap)
    contract_units = len(contract.get("contract_rows", [])) + len(mutations) + len(dispositions)
    spans.append(
        _phase_span(
            "contract",
            phase_started,
            started,
            units=contract_units,
            checkpoint="contract_custody_and_null_shape",
        )
    )
    progress(started, "contract", "after", completed_units=contract_units)

    phase_started = time.monotonic()
    progress(started, "method_ingestion", "before", planned_units=len(METHOD_SPECS))
    write_method_records(root)
    spans.append(
        _phase_span(
            "method_ingestion",
            phase_started,
            started,
            units=len(METHOD_SPECS),
            checkpoint="method_note_and_study_marker",
        )
    )
    progress(started, "method_ingestion", "after", completed_units=len(METHOD_SPECS))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7516-validation-", dir="/tmp"))
    fixture_path = private_root / "valid-null-fixture.json"
    atomic_json(fixture_path, valid_null_fixture())
    affected_plan = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, affected_plan)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{plan_errors}")
    atomic_json(
        raw_dir / "affected_validation_manifest.json",
        {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "documentation_paths": [
                NOTE_PATH.as_posix(),
                STUDY_PATH.as_posix(),
                SPEC_PATH.as_posix(),
            ],
        },
    )

    phase_started = time.monotonic()
    repository_plan = build_repository_check_plan(root, roadmap_path, fixture_path)
    prior_full_suite = root / FULL_SUITE_ATTEMPT_PATH
    if prior_full_suite.is_file():
        repository_plan = [row for row in repository_plan if row.spec.name != "full_python_suite"]
    progress(
        started,
        "validation",
        "before_subprocesses",
        planned_units=len(affected_plan) + len(repository_plan),
    )
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in affected_plan],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    repository = run_categorized_commands(
        root, repository_plan, log_dir=raw_dir / "validation/repository", heartbeat_s=60.0
    )
    if prior_full_suite.is_file():
        repository.append(load_json(prior_full_suite))
    repository_passed = receipts_pass(repository, REQUIRED_REPOSITORY_CHECK_NAMES)
    full_suite_passed = receipts_pass(repository, ("full_python_suite",))
    spans.append(
        _phase_span(
            "validation",
            phase_started,
            started,
            units=len(affected) + len(repository),
            checkpoint="scoped_repository_and_full_suite_checks",
        )
    )
    progress(
        started,
        "validation",
        "after_subprocesses",
        affected=affected_reduction["passed"],
        repository=repository_passed,
        full_python_suite=full_suite_passed,
    )
    candidate = build_artifact(
        root,
        roadmap_path,
        roadmap_candidates,
        contract,
        mutations,
        dispositions,
        {
            "validation_receipts": [*affected, *repository],
            "affected_checks_passed": affected_reduction["passed"],
            "repository_checks_passed": repository_passed,
            "terminal_checks_passed": True,
        },
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    candidate_errors = validate_artifact(candidate, root=root, require_terminal=False)
    if candidate_errors:
        raise RuntimeError(f"candidate_invalid:{candidate_errors}")

    terminal_plan = _terminal_commands(root, candidate_path)
    phase_started = time.monotonic()
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        planned_units=len(terminal_plan),
    )
    terminal = run_categorized_commands(
        root, terminal_plan, log_dir=raw_dir / "validation/terminal", heartbeat_s=60.0
    )
    terminal_passed = receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    spans.append(
        _phase_span(
            "terminal_validation",
            phase_started,
            started,
            units=len(terminal),
            checkpoint="entrypoint_replay_reduction_and_strict_readers",
        )
    )
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        root,
        roadmap_path,
        roadmap_candidates,
        contract,
        mutations,
        dispositions,
        {
            "validation_receipts": [*affected, *repository, *terminal],
            "affected_checks_passed": affected_reduction["passed"],
            "repository_checks_passed": repository_passed,
            "terminal_checks_passed": terminal_passed,
        },
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
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
    """Accept only the date frozen by the V658 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse measured-run and read-only cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date_argument)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - capability E2E.
    """Run V658 contract ingestion or validate a measured candidate."""

    print("[exp7516] phase=startup event=flushed", flush=True)
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
                "method_ingestion_complete_score": artifact["method_ingestion_complete_score"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - module execution boundary.
    raise SystemExit(main())

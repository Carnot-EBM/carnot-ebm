"""Bind the complete V660 advisory contract and dated method limits.

This module performs aggregation only. It preserves prior failures and absence
instead of turning planning readiness into a scientific benefit claim.

Spec refs: REQ-REPORT-7546 and SCENARIO-REPORT-7546-*.
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

from carnot.experiment_7329_v644_contract import parse_markdown_contract, parse_yaml_contract
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
from scripts import publication_gate


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.660"
EXPERIMENT_ID = "exp7546-contract-methods"
SCHEMA = "carnot.exp7546.v660.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7546_v660_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7546_v660_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7546_v660_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7546_v660_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7546_v660_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v660-method-map.md")
STUDY_PATH = Path("research-studying.md")

EXPECTED_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7546, "contract-methods"),
        (7547, "count-stream"),
        (7548, "capture-runner"),
        (7549, "count-learning"),
        (7550, "count-audit"),
        (7551, "native-pilot"),
        (7552, "fit-capture"),
        (7553, "test-capture"),
        (7554, "energy-fit"),
        (7555, "source-evaluation"),
        (7556, "arc-corrected-custody"),
        (7557, "arc-generalization"),
        (7558, "service-boundary"),
        (7559, "capstone"),
    )
)
V659_TASK_IDS = tuple(
    f"exp{number}-{slug}"
    for number, slug in (
        (7532, "contract-methods"),
        (7533, "tool-protocol"),
        (7534, "count-memory"),
        (7535, "native-pilot"),
        (7536, "fit-capture"),
        (7537, "eval-capture"),
        (7538, "energy-fit"),
        (7539, "static-evaluation"),
    )
)
V659_PATHS = (
    Path("results/experiment_7532_v659_contract_methods.json"),
    Path("results/experiment_7533_v659_tool_protocol.json"),
    Path("results/experiment_7534_v659_count_memory.json"),
    Path("results/experiment_7535_v659_native_pilot.json"),
    Path("results/experiment_7536_v659_fit_capture.json"),
    Path("results/experiment_7537_v659_eval_capture.json"),
    Path("results/experiment_7538_v659_energy_fit.json"),
    Path("results/experiment_7539_v659_source_evaluation.json"),
)
DIAGNOSTIC_PATHS = {
    4: Path("results/experiment_7536_fit_capture.json"),
    5: Path("results/experiment_7537_eval_capture.json"),
}
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}

SOURCE_INPUT_PATHS = (
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
    Path("python/carnot/experiment_7532_v659_contract_methods.py"),
    Path("scripts/roadmap_schema.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/arc_levelup_guarantee_lint.py"),
    Path("scripts/harness_consumer_checks.py"),
    Path("scripts/overdue_priority_lint.py"),
    Path("scripts/publication_gate.py"),
    DESIGN_PATH,
    SPEC_PATH,
    Path("research-references.md"),
    STUDY_PATH,
    Path("ops/known-issues.md"),
    *V659_PATHS[:4],
    *DIAGNOSTIC_PATHS.values(),
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
    "overdue_priority",
    "publication_gate_json",
)
REPOSITORY_DIAGNOSTIC_CHECK_NAMES = ("harness_fit", "full_python_suite")
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

METHOD_SPECS: tuple[JsonDict, ...] = (
    {
        "method_family": "safe_source_ablation",
        "primary_url": "https://arxiv.org/html/2609.16646v1",
        "source_revision": "arXiv:2609.16646v1 (2026-09-15)",
        "source_section": "Method sections on SAFE contrastive visual and vision-ablated probes",
        "adaptation": "Compare original, absent, and same-role mismatched tool evidence readouts.",
        "counterexample": "Visual ablation results do not prove textual factuality; absent evidence is not a false answer.",
        "task_mapping": ["exp7551-native-pilot", "exp7554-energy-fit", "exp7555-source-evaluation"],
    },
    {
        "method_family": "adversarial_world_model_sequences",
        "primary_url": "https://arxiv.org/abs/2602.05903",
        "source_revision": "arXiv:2602.05903 (2026-02-05)",
        "source_section": "Method sections on valid adversarial sequences and board-state probes",
        "adaptation": "Join induced predictions, executed plans, actions, and progress in runtime-observed ARC sequences.",
        "counterexample": "Chess sequence failures do not prove ARC transfer or that a probe causally controls actions.",
        "task_mapping": ["exp7556-arc-corrected-custody", "exp7557-arc-generalization"],
    },
    {
        "method_family": "online_proper_loss_recalibration",
        "primary_url": "https://arxiv.org/html/2607.19689v1",
        "source_revision": "arXiv:2607.19689v1 (2026-07-22)",
        "source_section": "Sections 2, 4, and 7.2 on proper-loss comparisons",
        "adaptation": "Compare delayed local count updates with the starting predictor and frozen, global, and legal shuffled controls.",
        "counterexample": "A conjugate count learner does not inherit the paper's algorithm or regret guarantee.",
        "task_mapping": ["exp7547-count-stream", "exp7549-count-learning", "exp7550-count-audit"],
    },
    {
        "method_family": "continual_calibration",
        "primary_url": "https://arxiv.org/abs/2604.23987",
        "source_revision": "arXiv:2604.23987 (2026-04)",
        "source_section": "Method sections on continual calibration and retained uncertainty",
        "adaptation": "Track retained Brier loss, action cost, and coverage at frozen arrival checkpoints.",
        "counterexample": "Sequential classification results do not guarantee delayed cached Carnot streams.",
        "task_mapping": ["exp7549-count-learning", "exp7550-count-audit"],
    },
)


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    """Return one aware UTC timestamp for a durable boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase boundary so a bounded task stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7546] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v660_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer matching staged V660 bytes, then accept the active roadmap."""

    candidates: list[JsonDict] = []
    selected: tuple[Path, JsonDict] | None = None
    for relative in (NEXT_ROADMAP_PATH, ACTIVE_ROADMAP_PATH):
        path = root / relative
        try:
            value = load_yaml(path)
            observed: Any = value.get("milestone")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            value = {}
            observed = None if isinstance(exc, FileNotFoundError) else type(exc).__name__
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
        raise ValueError("V660 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Retain only fields declared by both planning authorities."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Name whether each gate consumes an earlier declared producer field."""

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
        for gate in consumer.get("gated_on") or []:
            if not isinstance(gate, Mapping):
                continue
            producer = positions.get(gate.get("upstream"))
            producer_index = producer[0] if producer else None
            prompt = str(producer[1].get("prompt", "")) if producer else ""
            field = gate.get("artifact_field")
            declared = isinstance(field, str) and field in prompt
            precedes = producer_index is not None and producer_index < consumer_index
            rows.append(
                {
                    "upstream": gate.get("upstream"),
                    "artifact_field": field,
                    "producer_precedes_consumer": precedes,
                    "declared_verbatim": declared,
                    "passed": precedes and declared,
                    "principle": "A gate cannot consume an undeclared or later field.",
                }
            )
        result[str(consumer.get("id"))] = rows
    return result


def _parse_design(markdown_text: str) -> JsonDict:
    """Parse the independent task table from the V660 design."""

    normalized = markdown_text.replace(
        "| Order | Task ID | Title |", "| Order | Task ID | Exact title |", 1
    ).replace("| Structured gates |", "| Structured gate |", 1)
    return parse_markdown_contract(normalized)


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare all fourteen task identities and structured gates."""

    errors: list[str] = []
    try:
        parsed_yaml = parse_yaml_contract(roadmap)
    except (TypeError, ValueError) as exc:
        return {
            "passed": False,
            "errors": [f"parse_error:{exc}"],
            "markdown_milestone": None,
            "yaml_milestone": None,
            "markdown_task_rows": [],
            "yaml_task_rows": [],
            "contract_rows": [],
        }
    try:
        markdown = _parse_design(markdown_text)
    except (TypeError, ValueError):
        milestone = re.search(r"\*\*Milestone:\*\*\s*([^\s]+)", markdown_text)
        markdown = {"milestone": milestone.group(1) if milestone else None, "tasks": []}
        errors.append("markdown_task_table_missing")

    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    declarations = _producer_declarations(roadmap if isinstance(roadmap, Mapping) else {})
    rows: list[JsonDict] = []
    for index, expected_id in enumerate(EXPECTED_TASK_IDS):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else None
        observed = yaml_tasks[index] if index < len(yaml_tasks) else None
        checks = {
            field: bool(expected and observed and expected.get(field) == observed.get(field))
            for field in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
        }
        task_id = str((observed or expected or {}).get("id") or expected_id)
        declaration_rows = declarations.get(task_id, [])
        checks["producer_fields"] = bool(observed) and all(
            row["passed"] for row in declaration_rows
        )
        failures = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "unit_id": task_id,
                "arm": "markdown_vs_yaml",
                "order": index + 1,
                "expected_task_id": expected_id,
                "markdown": _public_task(expected),
                "yaml": _public_task(observed),
                "producer_declarations": declaration_rows,
                "checks": checks,
                "failures": failures,
                "passed": not failures,
                "principle": "Exact paired fields prevent a readable but different task.",
            }
        )
    if markdown.get("milestone") != MILESTONE:
        errors.append("markdown_milestone")
    if parsed_yaml.get("milestone") != MILESTONE:
        errors.append("yaml_milestone")
    if len(markdown_tasks) != 14:
        errors.append("markdown_task_count")
    if len(yaml_tasks) != 14:
        errors.append("yaml_task_count")
    if [row.get("id") for row in markdown_tasks] != list(EXPECTED_TASK_IDS):
        errors.append("markdown_task_order")
    if [row.get("id") for row in yaml_tasks] != list(EXPECTED_TASK_IDS):
        errors.append("yaml_task_order")
    if any(not row["passed"] for row in rows):
        errors.append("row_mismatch")
    return {
        "passed": not errors,
        "errors": list(dict.fromkeys(errors)),
        "markdown_milestone": markdown.get("milestone"),
        "yaml_milestone": parsed_yaml.get("milestone"),
        "markdown_task_rows": [_public_task(task) for task in markdown_tasks],
        "yaml_task_rows": [_public_task(task) for task in yaml_tasks],
        "contract_rows": rows,
    }


def mutation_names() -> tuple[str, ...]:
    """Return every registered private planning-authority defect."""

    return ("count", "order", "id", "title", "path", "gate_field", "milestone", "substrate")


def _mutate_yaml(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Plant one private YAML defect without changing repository bytes."""

    changed = deepcopy(dict(roadmap))
    tasks = changed.get("tasks", [])
    if mutation == "milestone":
        changed["milestone"] = "2026.09.659"
    elif mutation == "count":
        changed["tasks"] = tasks[:-1]
    elif mutation == "order" and len(tasks) > 1:
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif tasks:
        if mutation == "id":
            tasks[0]["id"] = "exp9999-changed"
        elif mutation == "title":
            tasks[0]["title"] = "Changed title"
        elif mutation == "path":
            tasks[0]["deliverable"] = "results/changed.json"
        elif mutation == "substrate":
            tasks[0]["prompt"] = str(tasks[0].get("prompt", "")).replace(
                "inference_substrate_class=aggregation",
                "inference_substrate_class=changed_substrate",
                1,
            )
        elif mutation == "gate_field":
            gated = next((task for task in tasks if task.get("gated_on")), None)
            if gated is not None:
                gated["gated_on"][0]["artifact_field"] = "changed_score"
    return changed


def _mutate_markdown(markdown_text: str, mutation: str) -> str:
    """Plant one private design-table defect without writing a file."""

    changed = markdown_text
    if mutation == "milestone":
        return changed.replace("**Milestone:** 2026.09.660", "**Milestone:** 2026.09.659", 1)
    lines = changed.splitlines()
    if mutation == "count":
        lines = [line for line in lines if not line.startswith("| 14 | exp7559-capstone |")]
        return "\n".join(lines)
    if mutation == "order":
        first = next(i for i, line in enumerate(lines) if line.startswith("| 1 | exp7546-"))
        second = next(i for i, line in enumerate(lines) if line.startswith("| 2 | exp7547-"))
        lines[first], lines[second] = lines[second], lines[first]
        return "\n".join(lines)
    replacements = {
        "id": ("| 1 | exp7546-contract-methods |", "| 1 | exp9999-changed |"),
        "title": (
            "Bind fourteen complete authorities and ingest the new method limits",
            "Changed title",
        ),
        "path": (
            "results/experiment_7546_v660_contract_methods.json",
            "results/changed.json",
        ),
        "gate_field": ("cached_stream_ready_score", "changed_score"),
        "substrate": ("| aggregation | none |", "| changed_substrate | none |"),
    }
    old, new = replacements[mutation]
    return changed.replace(old, new, 1)


def run_contract_mutation_controls(
    markdown_text: str, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Require the real baseline, then reject defects in either authority."""

    baseline = compare_contract_authorities(markdown_text, roadmap)
    rows: list[JsonDict] = []
    for authority in ("markdown", "yaml"):
        for mutation in mutation_names():
            if baseline["passed"]:
                comparison = (
                    compare_contract_authorities(_mutate_markdown(markdown_text, mutation), roadmap)
                    if authority == "markdown"
                    else compare_contract_authorities(
                        markdown_text, _mutate_yaml(roadmap, mutation)
                    )
                )
                rejected = comparison["passed"] is False
                errors = comparison["errors"]
            else:
                rejected = False
                errors = ["baseline_authorities_incomplete"]
            rows.append(
                {
                    "unit_id": f"{authority}:{mutation}",
                    "arm": "contract_mutation",
                    "authority": authority,
                    "mutation": mutation,
                    "baseline_valid": baseline["passed"],
                    "rejected": rejected,
                    "qualified": bool(baseline["passed"] and rejected),
                    "observed_errors": errors,
                    "principle": "A passing baseline makes each planted authority defect meaningful.",
                }
            )
    return rows


def collect_v659_dispositions(root: Path) -> list[JsonDict]:
    """Preserve four artifacts, two diagnostics, and two absent outputs."""

    rows: list[JsonDict] = []
    for index, (task_id, scheduled) in enumerate(zip(V659_TASK_IDS, V659_PATHS, strict=True)):
        scheduled_path = root / scheduled
        producer_present = scheduled_path.is_file()
        diagnostic = DIAGNOSTIC_PATHS.get(index)
        evidence = scheduled if producer_present else diagnostic
        evidence_path = root / evidence if evidence is not None else None
        payload = (
            load_json(evidence_path)
            if evidence_path is not None and evidence_path.is_file()
            else {}
        )
        invocation_counts = payload.get("invocation_counts")
        current_calls = (
            sum(value for value in invocation_counts.values() if isinstance(value, int))
            if isinstance(invocation_counts, Mapping)
            else None
        )
        evidence_kind = (
            "terminal_artifact"
            if producer_present
            else "pre_gate_diagnostic"
            if diagnostic is not None and evidence_path is not None and evidence_path.is_file()
            else "absent_scheduled_output"
        )
        expected_absent = index >= 4
        authenticated = bool(
            (producer_present is (not expected_absent))
            and (
                index >= 6
                or (
                    evidence_path is not None
                    and evidence_path.is_file()
                    and isinstance(payload.get("honest_verdict"), str)
                )
            )
        )
        if index in (4, 5):
            authenticated = bool(
                authenticated
                and payload.get("failed_upstream") == "exp7535-native-pilot"
                and payload.get("failed_field") == "native_tool_ready_score"
                and payload.get("failed_expected") == 1
                and payload.get("failed_observed") == 0
            )
        row: JsonDict = {
            "unit_id": task_id,
            "arm": "v659_custody",
            "order": index + 1,
            "task_id": task_id,
            "scheduled_path": scheduled.as_posix(),
            "producer_present": producer_present,
            "evidence_kind": evidence_kind,
            "evidence_path": evidence.as_posix() if evidence is not None else None,
            "source_sha256": sha256_file(evidence_path)
            if evidence_path is not None and evidence_path.is_file()
            else None,
            "original_honest_verdict": payload.get("honest_verdict"),
            "original_verdict_class": payload.get("verdict_class"),
            "historical_flagged_adversarial": payload.get("flagged_adversarial"),
            "historical_execution_venue": payload.get("execution_venue"),
            "historical_model_invoked": payload.get("model_invoked"),
            "current_invocation_count": current_calls,
            "failed_upstream": payload.get("failed_upstream"),
            "failed_field": payload.get("failed_field"),
            "failed_expected": payload.get("failed_expected"),
            "failed_observed": payload.get("failed_observed"),
            "missing_is_zero_metric": False,
            "authenticated": authenticated,
            "principle": "A missing producer is not a measured zero or a completed archive row.",
        }
        rows.append(row)
    return rows


def unissued_promises() -> list[JsonDict]:
    """Keep the six V659 prose promises separate from issued tasks."""

    slugs = (
        "online-learning",
        "decision-audit",
        "arc-harness-parity",
        "arc-induction",
        "service-boundary",
        "capstone",
    )
    return [
        {
            "unit_id": f"exp{number}",
            "arm": "v659_unissued_promise",
            "task_id": f"exp{number}",
            "promised_slug": slug,
            "issued": False,
            "producer_present": False,
            "archive_entry_created": False,
            "missing_is_zero_metric": False,
            "principle": "A prose promise is not an issued experiment or completed ledger entry.",
        }
        for number, slug in zip(range(7540, 7546), slugs, strict=True)
    ]


def method_rows() -> list[JsonDict]:
    """Return four primary method sections without importing paper claims."""

    return [
        {
            "unit_id": row["method_family"],
            "arm": "method_ingestion",
            **deepcopy(row),
            "access_status": "accessed_20260922",
            "access_failure": None,
            "external_claim_is_carnot_measurement": False,
            "principle": "A primary method can define a control without becoming local evidence.",
        }
        for row in METHOD_SPECS
    ]


def secondary_access_rows() -> list[JsonDict]:
    """Preserve incomplete secondary access instead of claiming coverage."""

    return [
        {
            "channel": "Semantic Scholar",
            "requested": "EBT and ARM-EBM citation lists",
            "observed": "HTTP 429",
            "complete": False,
            "principle": "Rate-limited citation access cannot support an exhaustive review claim.",
        },
        {
            "channel": "OpenReview",
            "requested": "2026 EBM and reasoning method text",
            "observed": "browser challenge",
            "complete": False,
            "principle": "A challenged page is a lead, not a method section that was read.",
        },
    ]


def write_method_records(root: Path) -> None:
    """Write the bounded V660 method map and one idempotent study marker."""

    lines = [
        "# V660 method map",
        "",
        "Date: 2026-09-23. Scope: advisory method accounting.",
        "External results are not local Carnot results.",
        "",
        "| Family | Primary source and section | Adaptation | Counterexample | Destination | Access |",
        "|---|---|---|---|---|---|",
    ]
    for row in method_rows():
        source = f"[{row['source_revision']}]({row['primary_url']}), {row['source_section']}"
        lines.append(
            f"| {row['method_family']} | {source} | {row['adaptation']} | "
            f"{row['counterexample']} | {', '.join(row['task_mapping'])} | {row['access_status']} |"
        )
    lines.extend(
        [
            "",
            "## Incomplete secondary access",
            "",
            "- Semantic Scholar citation requests returned HTTP 429. Citation coverage is incomplete.",
            "- OpenReview direct retrieval returned a browser challenge. It remains a lead, not a read method.",
            "",
            "No model, generation, benchmark, publication, or external contact occurred.",
            "",
        ]
    )
    note = root / NOTE_PATH
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines), encoding="utf-8")

    marker = "<!-- EXP7546-V660-METHOD-INGESTION -->"
    study = root / STUDY_PATH
    text = study.read_text(encoding="utf-8")
    if marker not in text:
        addition = "\n".join(
            [
                "",
                marker,
                "## 2026-09-23 Exp7546 — V660 methods — INGESTED",
                "",
                "SAFE source ablation, adversarial world-model sequences, proper-loss",
                "recalibration, and continual calibration map to separate V660 tasks.",
                "External results remain source context. Semantic Scholar HTTP 429 and",
                "the OpenReview browser challenge keep secondary access incomplete.",
                "See `docs/research-notes/v660-method-map.md` for limits and destinations.",
                "",
            ]
        )
        study.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


def operator_queue(root: Path) -> JsonDict:
    """Preserve E0 as operator-blocked and E6 as resolved."""

    text = (root / "ops/known-issues.md").read_text(encoding="utf-8")
    e0 = "BLOCKED 2026-09-22 (operator-only step pending): SEMIF OPTION-READOUT" in text
    e6 = "RESOLVED 2026-09-21: SEMIF FOLLOW-UP E6" in text
    return {
        "unit_id": "operator_queue",
        "arm": "repository_state",
        "source_path": "ops/known-issues.md",
        "source_sha256": sha256_file(root / "ops/known-issues.md"),
        "e0_status": "operator_blocked" if e0 else "not_authenticated",
        "e6_status": "resolved" if e6 else "not_authenticated",
        "authenticated": e0 and e6,
        "principle": "Operator-only and resolved work cannot be silently reclassified.",
    }


def _precondition(check: str, path: str, field: str, expected: Any, observed: Any) -> JsonDict:
    """Record one exact prerequisite and its read-only owner."""

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
    root: Path,
    roadmap_path: Path,
    contract: Mapping[str, Any],
    dispositions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Authenticate source, specification, custody, and host resources."""

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
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    roadmap = load_yaml(roadmap_path)
    rows.extend(
        [
            _precondition(
                "driving_requirement",
                SPEC_PATH.as_posix(),
                "REQ-*",
                "REQ-REPORT-7546",
                "REQ-REPORT-7546" if "REQ-REPORT-7546" in spec else None,
            ),
            _precondition(
                "roadmap_identity",
                roadmap_path.relative_to(root).as_posix(),
                "milestone",
                MILESTONE,
                roadmap.get("milestone"),
            ),
            _precondition(
                "design_task_count",
                DESIGN_PATH.as_posix(),
                "exact_ordered_task_count",
                14,
                len(contract.get("markdown_task_rows") or []),
            ),
            _precondition(
                "yaml_task_count",
                roadmap_path.relative_to(root).as_posix(),
                "exact_ordered_task_count",
                14,
                len(contract.get("yaml_task_rows") or []),
            ),
            _precondition(
                "v659_disposition_custody",
                "results/experiment_7532..7539",
                "authenticated_count",
                8,
                sum(row.get("authenticated") is True for row in dispositions),
            ),
            _precondition(
                "host_aggregation_resource",
                "/proc/cpuinfo",
                "machine_nonempty",
                True,
                bool(platform.machine()),
            ),
            _precondition(
                "gpu_required_for_current_work", "current_exp7546", "gpu_required", False, False
            ),
        ]
    )
    return rows


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Bind current protocol bytes and present historical evidence."""

    paths = (
        *SOURCE_INPUT_PATHS,
        roadmap_path.relative_to(root),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        NOTE_PATH,
    )
    hashes: dict[str, JsonDict] = {}
    for relative in dict.fromkeys(paths):
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "evidence_type": "current_protocol_or_input_bytes",
            }
    for row in dispositions:
        path_text = row.get("evidence_path")
        if isinstance(path_text, str) and (root / path_text).is_file():
            hashes[f"historical:{row.get('task_id')}"] = {
                "path": path_text,
                "sha256": sha256_file(root / path_text),
                "evidence_type": "historical_terminal_or_diagnostic_bytes",
                "original_honest_verdict": row.get("original_honest_verdict"),
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


def build_repository_check_plan(root: Path, roadmap_path: Path) -> list[PlannedCommand]:
    """Run unchanged policy checks and one separate full-suite diagnostic."""

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
        ("roadmap_schema", (python, "-u", "-c", schema_code, selected), True, 900.0),
        (
            "prior_failure",
            (python, "-u", "scripts/validate_prior_failures.py", selected),
            True,
            900.0,
        ),
        (
            "roadmap_gate_audit",
            (python, "-u", "scripts/audit_roadmap_gates.py", selected),
            True,
            900.0,
        ),
        (
            "exclusion_manifest",
            (python, "-u", "scripts/exclusion_manifest_lint.py", selected),
            True,
            900.0,
        ),
        (
            "arc_floor",
            (python, "-u", "scripts/arc_levelup_guarantee_lint.py", selected),
            True,
            900.0,
        ),
        ("overdue_priority", (python, "-u", "-c", overdue_code, selected), True, 900.0),
        (
            "publication_gate_json",
            (python, "-u", "scripts/publication_gate.py", "--json"),
            True,
            900.0,
        ),
        (
            "harness_fit",
            (python, "-u", "scripts/harness_consumer_checks.py", "prompt-paths", selected),
            False,
            900.0,
        ),
        ("full_python_suite", (pytest, "tests/python", "-q"), False, 4200.0),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                name,
                argv,
                "selected_roadmap" if name != "full_python_suite" else "repository_health",
                timeout_s,
            ),
            "required_validation" if required else "diagnostic_repository_health",
            required,
        )
        for name, argv, required, timeout_s in specs
    ]


def prior_full_suite_receipt(root: Path) -> JsonDict | None:
    """Reuse the recorded full-suite attempt instead of running it twice."""

    path = root / RESULT_PATH
    if not path.is_file():
        return None
    payload = load_json(path)
    for row in payload.get("validation_receipts") or []:
        if isinstance(row, Mapping) and row.get("name") == "full_python_suite":
            return deepcopy(dict(row))
    return None


def _gate(
    check: str,
    gate_type: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    *,
    upstream: str = "current_exp7546_reduction",
    field_path: str | None = None,
) -> JsonDict:
    """Attach exact operands and the failure prevented by a gate."""

    return {
        "check": check,
        "gate_type": gate_type,
        "condition": f"{field_path or check} == expected",
        "upstream": upstream,
        "field_path": field_path or check,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": "==",
        "passed": passed,
        "principle": principle,
    }


def terminal_state(external_ready: bool, validation_passed: bool) -> tuple[str, str, str]:
    """Keep external absence, invalid work, and advisory completion distinct."""

    if not external_ready:
        value = "complete_blocked_incomplete_v660_authorities"
        return value, value, "blocked"
    if not validation_passed:
        value = "complete_disqualified_v660_contract_or_validation"
        return value, value, "disqualified"
    return (
        "complete_advisory_v660_contract_and_methods",
        "complete_null_v660_contract_methods_ingested",
        "null",
    )


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed check with its exact upstream operands."""

    failed = [
        {
            "check": row.get("check"),
            "upstream": row.get("upstream"),
            "field_path": row.get("field_path"),
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
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep readiness, validity, support, and benefit independent."""

    contract_ok = contract.get("passed") is True
    mutation_ok = len(mutations) == 2 * len(mutation_names()) and all(
        row.get("qualified") is True for row in mutations
    )
    return [
        _gate(
            "exact_v660_contract",
            "readiness",
            {"count": 14, "ids": list(EXPECTED_TASK_IDS)},
            {
                "markdown_count": len(contract.get("markdown_task_rows") or []),
                "yaml_count": len(contract.get("yaml_task_rows") or []),
                "errors": contract.get("errors"),
            },
            contract_ok,
            "Both complete authorities are required before contract readiness.",
            upstream=f"{DESIGN_PATH.as_posix()} + {ACTIVE_ROADMAP_PATH.as_posix()}",
            field_path="tasks.count_order_and_public_fields",
        ),
        _gate(
            "private_contract_mutations",
            "validity",
            2 * len(mutation_names()),
            sum(row.get("qualified") is True for row in mutations),
            mutation_ok,
            "Private defects prove the comparison rejects contract drift.",
            field_path="contract_mutation_rows.qualified",
        ),
        _gate(
            "v659_custody",
            "validity",
            8,
            sum(row.get("authenticated") is True for row in dispositions),
            len(dispositions) == 8
            and all(row.get("authenticated") is True for row in dispositions),
            "Prior failures, diagnostics, and absence cannot be rehabilitated.",
            upstream="results/experiment_7532..7539",
            field_path="prior_dispositions.authenticated",
        ),
        _gate(
            "affected_validation",
            "validity",
            True,
            validation.get("affected_checks_passed") is True,
            validation.get("affected_checks_passed") is True,
            "Scoped checks prevent unrelated repository state from laundering changed code.",
        ),
        _gate(
            "repository_guards",
            "validity",
            True,
            validation.get("repository_checks_passed") is True,
            validation.get("repository_checks_passed") is True,
            "Unchanged policy guards retain their original rules.",
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
            {"minimum": 3, "maximum": 5},
            len(methods),
            3 <= len(methods) <= 5
            and all(row.get("external_claim_is_carnot_measurement") is False for row in methods),
            "External method results cannot become local measurements.",
        ),
        _gate(
            "scientific_benefit_not_claimed",
            "benefit",
            False,
            False,
            True,
            "An advisory contract cannot become a scientific benefit claim.",
        ),
    ]


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain how every top-level field prevents a reporting failure."""

    required = {
        "preconditions_checked": "Exact paths and observations prevent fabricated fallback inputs.",
        "MODEL_SPECS": "An empty list states that this aggregation loaded no model.",
        "model_specs": "The lowercase mirror prevents a reader from inferring a model.",
        "model_invoked": "Current calls stay separate from historical model evidence.",
        "inference_substrate_class": "The aggregation class prevents a false runtime floor.",
        "inference_substrate": "Canonical aggregation spelling prevents substrate ambiguity.",
        "execution_venue": "Legal host reduction stays distinct from historical host_cpu data.",
        "duration_s": "Monotonic current work remains separate from historical duration.",
        "random_seed": "Frozen bookkeeping seeds cannot multiply scientific support.",
        "reproducibility_checksum": "The checksum binds code, inputs, rows, and receipts.",
        "rows": "Every task, disposition, promise, method, and access row stays reducible.",
        "sample_size_budget": "Planned, attempted, completed, failed, censored, and unstarted stay distinct.",
        "acceptance_gate_results": "Each gate retains operands, outcome, and prevented failure.",
        "gate_check_summary": "Blocked or invalid work names exact upstream fields and observations.",
        "honest_verdict": "A complete terminal prefix prevents retry-state ambiguity.",
        "verdict_class": "The closed class separates blocked, invalid, and valid null work.",
        "verifier_is_oracle": "No oracle fixture supports a positive advisory claim.",
        "flagged_adversarial": "The historical Exp7532 flag remains visible.",
        "validation_receipts": "Commands, exits, log hashes, cold replay, and reductions remain auditable.",
        "field_principles": "Every emitted field states the failure it prevents.",
        "contract_ready_score": "Bare one requires exact agreement and rejected mutations.",
        "method_ingestion_complete_score": "Bare one requires primary methods and access limits.",
        "prior_dispositions": "All eight actual V659 outcomes preserve literal evidence and absence.",
        "publication_gates": "Stable G1-G4 replace a redefinable blocker count without publishing.",
    }
    return {
        field: required.get(field, f"The {field} field preserves exact scope and meaning.")
        for field in fields
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the stable artifact except the checksum's own value."""

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
    """Build one complete advisory record from exact raw inputs."""

    contract_copy = deepcopy(dict(contract))
    mutations = deepcopy(list(mutation_rows))
    prior = deepcopy(list(dispositions))
    promises = unissued_promises()
    methods = method_rows()
    access = secondary_access_rows()
    queue = operator_queue(root)
    preconditions = collect_preconditions(root, roadmap_path, contract_copy, prior)
    receipts = deepcopy(list(validation.get("validation_receipts") or []))
    gates = _acceptance_gates(contract_copy, mutations, prior, methods, validation)
    external_ready = contract_copy.get("passed") is True
    validation_ok = all(
        row.get("passed") is True
        for row in gates
        if row.get("gate_type") in {"validity", "readiness"}
    )
    status, honest, verdict = terminal_state(external_ready, validation_ok)
    rows = [
        *deepcopy(contract_copy.get("contract_rows") or []),
        *mutations,
        *prior,
        *deepcopy(promises),
        *deepcopy(methods),
        *deepcopy(access),
        deepcopy(queue),
    ]
    duration_s = max(0.0, (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000)
    validation_duration = sum(float(row.get("duration_s", 0.0)) for row in receipts)
    baseline = [row for row in receipts if row.get("name") in REPOSITORY_DIAGNOSTIC_CHECK_NAMES]
    contract_ready = bool(external_ready and validation_ok)
    methods_ready = bool(
        3 <= len(methods) <= 5
        and all(
            row.get("primary_url")
            and row.get("source_section")
            and row.get("adaptation")
            and row.get("counterexample")
            and row.get("task_mapping")
            and row.get("access_status") == "accessed_20260922"
            for row in methods
        )
        and all(row.get("complete") is False for row in access)
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7546,
        "title": "Bind fourteen complete authorities and ingest the new method limits",
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
        "historical_invocations_counted_as_current": False,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "device_identity": {
            "machine": platform.machine(),
            "node": platform.node(),
            "processor": platform.processor(),
            "cuda_used": False,
            "gpu_required": False,
        },
        "duration_components_s": {
            "authoring": 0.0,
            "computation": max(0.0, duration_s - validation_duration),
            "validation": validation_duration,
            "historical": 0.0,
            "total_current": duration_s,
        },
        "random_seed": {
            "sampling": None,
            "fitting": None,
            "ordering": None,
            "bootstrap": None,
            "contract_mutations": 7_546_660_01,
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
        "unissued_promises": deepcopy(promises),
        "completed_ledger_end_at_planning": "2026.09.658",
        "v659_archive_entries_invented": False,
        "method_rows": deepcopy(methods),
        "secondary_access_rows": deepcopy(access),
        "operator_queue": deepcopy(queue),
        "source_artifact_hashes": _source_hashes(root, roadmap_path, prior),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows) - len(promises),
            "complete": len(rows) - len(promises),
            "completed": len(rows) - len(promises),
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": len(promises),
            "independent_unit": "contract_mutation_disposition_promise_method_access_or_queue_row",
            "counting_rule": "Six prose promises remain unstarted; absent producer metrics are not zeros.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "contract_ready_score": int(contract_ready),
        "method_ingestion_complete_score": int(methods_ready),
        "positive_claim": False,
        "no_headroom": {
            "applicable": False,
            "observed": None,
            "reason": "This advisory task did not measure a scientific effect.",
        },
        "honest_verdict": honest,
        "verdict_class": verdict,
        "verifier_is_oracle": False,
        "flagged_adversarial": any(
            row.get("historical_flagged_adversarial") is True for row in prior
        ),
        "validation_receipts": receipts,
        "repository_health": {
            "baseline_receipts": baseline,
            "failed_baseline_checks": [
                row.get("name") for row in baseline if row.get("passed") is not True
            ],
            "affects_required_scoped_validation": False,
        },
        "publication_gates": publication_gate.evaluate(),
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
            "fresh_process_cold_replay": receipts_pass(receipts, TERMINAL_CHECK_NAMES[:2]),
            "learning_predict_release_update_persist_reload_applicable": False,
            "numbered_runtime_e2e_applicable": False,
            "llm_off_real_environment_smoke_applicable": False,
        },
        "advisory_only": True,
        "global_science_gate": False,
        "roadmap_activation_performed": False,
        "roadmap_archive_performed": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
        "research_conductor_modified": False,
        "publication_performed": False,
        "external_contact_performed": False,
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
    access = artifact.get("secondary_access_rows")
    receipts = artifact.get("validation_receipts")
    contract_ok = isinstance(contract, Mapping) and contract.get("passed") is True
    mutation_ok = (
        isinstance(mutations, list)
        and len(mutations) == 2 * len(mutation_names())
        and all(isinstance(row, Mapping) and row.get("qualified") is True for row in mutations)
    )
    dispositions_ok = (
        isinstance(dispositions, list)
        and len(dispositions) == 8
        and all(
            isinstance(row, Mapping) and row.get("authenticated") is True for row in dispositions
        )
    )
    methods_ok = (
        isinstance(methods, list)
        and 3 <= len(methods) <= 5
        and all(
            isinstance(row, Mapping)
            and row.get("external_claim_is_carnot_measurement") is False
            and row.get("access_status") == "accessed_20260922"
            for row in methods
        )
        and isinstance(access, list)
        and len(access) == 2
        and all(isinstance(row, Mapping) and row.get("complete") is False for row in access)
    )
    affected_ok = receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    repository_ok = receipts_pass(receipts, REQUIRED_REPOSITORY_CHECK_NAMES)
    terminal_ok = receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    validation_ok = (
        mutation_ok and dispositions_ok and affected_ok and repository_ok and terminal_ok
    )
    status, honest, verdict = terminal_state(contract_ok, validation_ok)
    return {
        "contract_ready_score": int(contract_ok and validation_ok),
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
        roadmap_path, _roadmap, _candidates = resolve_v660_roadmap(root)
        expected = _source_hashes(root, roadmap_path, collect_v659_dispositions(root))
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
        _roadmap_path, roadmap, _candidates = resolve_v660_roadmap(root)
        markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
        contract = compare_contract_authorities(markdown, roadmap)
        mutations = run_contract_mutation_controls(markdown, roadmap)
        dispositions = collect_v659_dispositions(root)
        promises = unissued_promises()
        queue = operator_queue(root)
    except (OSError, ValueError, json.JSONDecodeError, KeyError, IndexError) as exc:
        return [*errors, f"source_reduction_failed:{type(exc).__name__}"]
    if artifact.get("task_contract_rows") != contract.get("contract_rows"):
        errors.append("task_contract_rows_mismatch")
    if artifact.get("contract_comparison") != contract:
        errors.append("contract_comparison_mismatch")
    if artifact.get("contract_mutation_rows") != mutations:
        errors.append("contract_mutations_invalid")
    if artifact.get("prior_dispositions") != dispositions:
        errors.append("prior_dispositions_mismatch")
    if artifact.get("unissued_promises") != promises:
        errors.append("unissued_promises_mismatch")
    if artifact.get("method_rows") != method_rows():
        errors.append("method_rows_mismatch")
    if artifact.get("secondary_access_rows") != secondary_access_rows():
        errors.append("secondary_access_rows_mismatch")
    if artifact.get("operator_queue") != queue:
        errors.append("operator_queue_mismatch")
    expected_rows = [
        *deepcopy(contract.get("contract_rows") or []),
        *deepcopy(mutations),
        *deepcopy(dispositions),
        *deepcopy(promises),
        *method_rows(),
        *secondary_access_rows(),
        deepcopy(queue),
    ]
    if artifact.get("rows") != expected_rows:
        errors.append("rows_mismatch")
    receipts = artifact.get("validation_receipts")
    if not receipts_recorded(receipts, validation_scope.REQUIRED_CHECK_NAMES):
        errors.append("affected_validation_invalid")
    if not receipts_recorded(receipts, REQUIRED_REPOSITORY_CHECK_NAMES):
        errors.append("repository_validation_invalid")
    if not receipts_recorded(receipts, REPOSITORY_DIAGNOSTIC_CHECK_NAMES):
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
    if artifact.get("positive_claim") is not False:
        errors.append("positive_claim_invalid")
    if artifact.get("publication_gates") != publication_gate.evaluate():
        errors.append("publication_gates_mismatch")
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
            or row.get("gate_type") not in {"validity", "readiness", "support", "benefit"}
            or not {"condition", "expected", "observed", "op", "passed"}.issubset(row)
            for row in gates
        )
    ):
        errors.append("gate_principles_invalid")
    if artifact.get("gate_check_summary") != _gate_summary(
        gates if isinstance(gates, list) else []
    ):
        errors.append("gate_summary_mismatch")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_blocked_artifact(
    missing: Path,
    field: str,
    expected: Any,
    observed: Any,
    started_at: str,
    started_ns: int,
) -> JsonDict:
    """Build a schema-complete external-absence receipt without fallback."""

    ended_ns = time.monotonic_ns()
    reason = re.sub(r"[^a-z0-9]+", "_", missing.name.lower()).strip("_") or "prerequisite"
    verdict = f"complete_blocked_missing_{reason}"
    gate = _gate(
        "external_prerequisite_available",
        "validity",
        expected,
        observed,
        False,
        "A missing external input blocks dependent reduction.",
        upstream=missing.as_posix(),
        field_path=field,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "status": verdict,
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "duration_s": max(0.0, (ended_ns - started_ns) / 1_000_000_000),
        "preconditions_checked": [
            _precondition(gate["check"], missing.as_posix(), field, expected, observed)
        ],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "rows": [],
        "sample_size_budget": {
            "planned": 0,
            "attempted": 0,
            "complete": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": _gate_summary([gate]),
        "honest_verdict": verdict,
        "verdict_class": "blocked",
        "positive_claim": False,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "contract_ready_score": 0,
        "method_ingestion_complete_score": 0,
        "task_contract_rows": [],
        "prior_dispositions": [],
        "unissued_promises": unissued_promises(),
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


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
        "from carnot.experiment_7546_v660_contract_methods import independent_reduce,validate_artifact;"
        "p=pathlib.Path(sys.argv[1]);v=json.loads(p.read_text());"
        "r=independent_reduce(v,require_terminal=False);e=validate_artifact(v,require_terminal=False);"
        "print(json.dumps({'reduction':r,'errors':e},sort_keys=True),flush=True);"
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


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Run bounded checks and atomically publish validated terminal bytes."""

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
        blocked = build_blocked_artifact(
            missing, "bytes", "readable_nonempty_bytes", None, started_at, started_ns
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=missing.as_posix())
        return blocked
    roadmap_path, roadmap, roadmap_candidates = resolve_v660_roadmap(root)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(markdown, roadmap)
    mutations = run_contract_mutation_controls(markdown, roadmap)
    dispositions = collect_v659_dispositions(root)
    preconditions = collect_preconditions(root, roadmap_path, contract, dispositions)
    spans.append(
        _phase_span(
            "preconditions",
            phase_started,
            started,
            units=len(preconditions),
            checkpoint="inputs_resources_and_authorities_observed",
        )
    )
    progress(
        started,
        "preconditions",
        "after",
        completed_units=len(preconditions),
        failed_units=sum(row["passed"] is not True for row in preconditions),
    )

    for phase in ("model_load", "generation"):
        phase_started = time.monotonic()
        progress(started, phase, "before", completed_units=0)
        spans.append(
            _phase_span(phase, phase_started, started, units=0, checkpoint="no_current_llm_work")
        )
        progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    progress(started, "contract", "before")
    contract_units = len(contract["contract_rows"]) + len(mutations) + len(dispositions)
    spans.append(
        _phase_span(
            "contract",
            phase_started,
            started,
            units=contract_units,
            checkpoint="authorities_mutations_and_v659_custody_reduced",
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

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7546-validation-", dir="/tmp"))
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
    repository_plan = build_repository_check_plan(root, roadmap_path)
    prior_full_suite = prior_full_suite_receipt(root)
    if prior_full_suite is not None:
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
        root,
        repository_plan,
        log_dir=raw_dir / "validation/repository",
        heartbeat_s=60.0,
    )
    if prior_full_suite is not None:
        repository.append(prior_full_suite)
    repository_passed = receipts_pass(repository, REQUIRED_REPOSITORY_CHECK_NAMES)
    full_suite_passed = receipts_pass(repository, ("full_python_suite",))
    spans.append(
        _phase_span(
            "validation",
            phase_started,
            started,
            units=len(affected) + len(repository),
            checkpoint="scoped_repository_publication_and_full_suite_checks",
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
        root,
        terminal_plan,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
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
    """Accept only the date frozen by the V660 contract."""

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
    """Run V660 contract ingestion or validate a measured candidate."""

    print("[exp7546] phase=startup event=flushed", flush=True)
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

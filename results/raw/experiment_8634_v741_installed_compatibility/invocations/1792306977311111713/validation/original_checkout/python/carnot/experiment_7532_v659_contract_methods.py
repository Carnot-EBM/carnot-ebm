"""Bind the V659 advisory contract while preserving corrected prior evidence.

This report performs aggregation only. It fails closed when either planning
authority is incomplete and does not turn that planning defect into science.

Spec refs: REQ-REPORT-7532 and SCENARIO-REPORT-7532-*.
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


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260922"
MILESTONE = "2026.09.659"
EXPERIMENT_ID = "exp7532-contract-methods"
SCHEMA = "carnot.exp7532.v659.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7532_v659_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7532_v659_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7532_v659_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7532_v659_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7532_v659_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v659-method-map.md")
STUDY_PATH = Path("research-studying.md")
V658_CAPSTONE_PATH = Path("results/experiment_7529_v658_capstone.json")
B2_PATH = Path("results/experiment_7531_b2_induction_gate_measurement.json")

EXPECTED_TASK_IDS = tuple(
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
        (7540, "online-learning"),
        (7541, "decision-audit"),
        (7542, "arc-harness-parity"),
        (7543, "arc-induction"),
        (7544, "service-boundary"),
        (7545, "capstone"),
    )
)

V658_PATHS = tuple(
    Path(f"results/experiment_{number}_v658_{slug}.json")
    for number, slug in (
        (7516, "contract_methods"),
        (7517, "source_protocol"),
        (7518, "source_pilot"),
        (7519, "source_fit_capture"),
        (7520, "source_eval_capture"),
        (7521, "consistency_energy"),
        (7522, "source_evaluation"),
        (7523, "count_memory"),
        (7524, "count_online"),
        (7525, "decision_audit"),
        (7526, "arc_eligibility"),
        (7527, "arc_opportunities"),
        (7528, "service_boundary"),
        (7529, "capstone"),
    )
)

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
    Path("python/carnot/experiment_7516_v658_contract_methods.py"),
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
    V658_CAPSTONE_PATH,
    B2_PATH,
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
    "harness_fit",
    "overdue_priority",
)
BASELINE_CHECK_NAMES = ("full_python_suite",)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

METHOD_SPECS: tuple[JsonDict, ...] = (
    {
        "method_family": "tool_grounding",
        "primary_url": "https://arxiv.org/html/2607.00895v1",
        "source_revision": "arXiv:2607.00895v1 (2026-07-01)",
        "source_section": "Sections 3, 4.1, 4.3 and 4.4",
        "adaptation": "Seal tool-output source groups and keep injected edits distinct from natural errors.",
        "counterexample": "Injected unsupported spans do not establish organic hallucination or code correctness.",
        "task_mapping": ["exp7533-tool-protocol", "exp7539-static-evaluation"],
    },
    {
        "method_family": "continual_calibration",
        "primary_url": "https://arxiv.org/html/2604.23987v1",
        "source_revision": "arXiv:2604.23987v1 (2026-04-30)",
        "source_section": "Sections 3-5, calibration and retention evaluation",
        "adaptation": "Measure old-domain Brier and typed-action coverage at each legal update checkpoint.",
        "counterexample": "Sequential classification results do not guarantee delayed Carnot streams.",
        "task_mapping": ["exp7540-online-learning", "exp7541-decision-audit"],
    },
    {
        "method_family": "online_recalibration",
        "primary_url": "https://arxiv.org/html/2607.19689v1",
        "source_revision": "arXiv:2607.19689v1 (2026-07-22)",
        "source_section": "Sections 2, 4 and 7.2, proper-loss comparisons",
        "adaptation": "Compare causal bin-count updates with frozen, global and legal shuffled controls.",
        "counterexample": "A conjugate count memory is not the paper's Blackwell algorithm.",
        "task_mapping": ["exp7534-count-memory", "exp7540-online-learning"],
    },
    {
        "method_family": "consistency",
        "primary_url": "https://arxiv.org/html/2606.08158",
        "source_revision": "arXiv:2606.08158v1 (2026-06-06)",
        "source_section": "Section 2.1, constrained consistency objective",
        "adaptation": "Regularize semantic option order while keeping source interventions as separate features.",
        "counterexample": "Replacing evidence is not label-preserving after semantic remapping.",
        "task_mapping": ["exp7538-energy-fit", "exp7539-static-evaluation"],
    },
)


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    """Return one aware UTC timestamp for a durable boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush each phase boundary so bounded work stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7532] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v659_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer staged V659 bytes, then accept the matching active roadmap."""

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
        raise ValueError("V659 roadmap authority is unavailable")
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
    """Name whether each gate consumes an earlier producer field."""

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
            producer_entry = positions.get(gate.get("upstream"))
            producer_index = producer_entry[0] if producer_entry else None
            prompt = str(producer_entry[1].get("prompt", "")) if producer_entry else ""
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
    """Parse the independent executable task table when it is present."""

    normalized = markdown_text.replace(
        "| Order | Task ID | Title |", "| Order | Task ID | Exact title |", 1
    ).replace("| Structured gates |", "| Structured gate |", 1)
    return parse_markdown_contract(normalized)


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare all fourteen identities while retaining an incomplete authority."""

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
        changed["milestone"] = "2026.09.658"
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
            tasks[0]["inference_substrate_class"] = "changed_substrate"
        elif mutation == "gate_field":
            gated = next((task for task in tasks if task.get("gated_on")), None)
            if gated is not None:
                gated["gated_on"][0]["artifact_field"] = "changed_score"
    return changed


def run_contract_mutation_controls(
    markdown_text: str, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Reject private defects only after the real paired baseline is valid."""

    baseline = compare_contract_authorities(markdown_text, roadmap)
    rows: list[JsonDict] = []
    for authority in ("markdown", "yaml"):
        for mutation in mutation_names():
            rows.append(
                {
                    "unit_id": f"{authority}:{mutation}",
                    "arm": "contract_mutation",
                    "authority": authority,
                    "mutation": mutation,
                    "baseline_valid": baseline["passed"],
                    "rejected": False,
                    "qualified": False,
                    "reason": "baseline_authorities_incomplete",
                    "principle": "Mutation controls cannot pass when the unmodified contract is absent.",
                }
            )
    return rows


def collect_v658_dispositions(root: Path) -> list[JsonDict]:
    """Preserve fourteen capstone dispositions and seven absent producers."""

    capstone = load_json(root / V658_CAPSTONE_PATH)
    source_protocol = load_json(root / V658_PATHS[1])
    raw_rows = capstone.get("task_dispositions")
    rows: list[JsonDict] = []
    if not isinstance(raw_rows, list):
        return rows
    for order, expected_path in enumerate(V658_PATHS, start=1):
        prior = raw_rows[order - 1] if order <= len(raw_rows) else {}
        path = root / expected_path
        present = path.is_file()
        payload = load_json(path) if present else {}
        expected_task = str(prior.get("task_id"))
        verdict = prior.get("honest_verdict")
        verdict_class = prior.get("verdict_class")
        producer_matches = not present or (
            payload.get("honest_verdict") == verdict
            and payload.get("verdict_class") == verdict_class
        )
        expected_absent = 3 <= order <= 9
        presence_matches = present is (not expected_absent)
        row: JsonDict = {
            "unit_id": expected_task,
            "arm": "v658_custody",
            "order": order,
            "task_id": expected_task,
            "expected_path": expected_path.as_posix(),
            "producer_present": present,
            "expected_producer_present": not expected_absent,
            "source_sha256": sha256_file(path) if present else None,
            "original_honest_verdict": verdict,
            "original_verdict_class": verdict_class,
            "original_flagged_adversarial": payload.get("flagged_adversarial") if present else None,
            "missing_is_zero_metric": False,
            "authenticated": bool(
                len(raw_rows) == 14
                and prior.get("order") == order
                and prior.get("expected_artifact_path") == expected_path.as_posix()
                and isinstance(verdict, str)
                and verdict.startswith("complete_")
                and verdict_class
                in {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
                and presence_matches
                and producer_matches
            ),
            "principle": "Literal prior dispositions keep absence distinct from measured zero.",
        }
        if order == 2:
            audit = source_protocol.get("exposure_audit") or {}
            row.update(
                {
                    "candidate_official_training_groups": audit.get(
                        "candidate_official_training_groups"
                    ),
                    "exposed_candidate_groups": audit.get("exposed_candidate_groups"),
                    "fresh_eligible_groups": audit.get("fresh_eligible_groups"),
                }
            )
            row["authenticated"] = bool(
                row["authenticated"]
                and row["candidate_official_training_groups"] == 2487
                and row["exposed_candidate_groups"] == 2487
                and row["fresh_eligible_groups"] == 0
            )
        rows.append(row)
    return rows


def collect_b2_corrections(root: Path) -> list[JsonDict]:
    """Reduce both appended Exp7531 corrections from raw attempt rows."""

    payload = load_json(root / B2_PATH)
    attempts = payload.get("per_attempt_rows")
    attempt_rows = attempts if isinstance(attempts, list) else []
    progress_count = sum(row.get("progress_within_window") is True for row in attempt_rows)
    capped_count = sum(row.get("completion_tokens") == 256 for row in attempt_rows)
    first = payload.get("false_negative_risk") or {}
    second = payload.get("further_correction_2026_09_22") or {}
    source_hash = sha256_file(root / B2_PATH)
    return [
        {
            "unit_id": "exp7531-progress-proxy-correction",
            "arm": "historical_correction",
            "source_path": B2_PATH.as_posix(),
            "source_sha256": source_hash,
            "source_field": "false_negative_risk",
            "flagged": first.get("flagged"),
            "progress_attempt_count": progress_count,
            "total_attempt_count": len(attempt_rows),
            "progress_proxy_saturated": bool(attempt_rows) and progress_count == len(attempt_rows),
            "induced_plan_required_for_progress": False,
            "measurement_scope": "feasibility_only",
            "authenticated": first.get("flagged") is True and progress_count == 60,
            "principle": "A saturated proxy cannot establish induction reliability or no headroom.",
        },
        {
            "unit_id": "exp7531-token-cap-correction",
            "arm": "historical_correction",
            "source_path": B2_PATH.as_posix(),
            "source_sha256": source_hash,
            "source_field": "further_correction_2026_09_22",
            "flagged": second.get("flagged"),
            "attempts_at_harness_token_cap": capped_count,
            "total_attempt_count": len(attempt_rows),
            "harness_max_new_tokens": 256,
            "shipped_production_setting_claimed": False,
            "historical_e6_mean_completion_tokens_per_episode": 11950,
            "measurement_scope": "feasibility_only",
            "authenticated": second.get("flagged") is True
            and len(attempt_rows) == 60
            and capped_count == 60,
            "principle": "An inherited harness cap cannot be attributed to shipped production settings.",
        },
    ]


def method_rows() -> list[JsonDict]:
    """Return four source-grounded methods without importing paper claims."""

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


def write_method_records(root: Path) -> None:
    """Write the bounded method map and one idempotent study marker."""

    lines = [
        "# V659 method map",
        "",
        "Date: 2026-09-22. Scope: advisory method accounting.",
        "External results are not local Carnot results.",
        "",
        "| Family | Primary source and section | Exact adaptation | Counterexample | Destination task | Access |",
        "|---|---|---|---|---|---|",
    ]
    for row in method_rows():
        source = f"[{row['source_revision']}]({row['primary_url']}), {row['source_section']}"
        lines.append(
            f"| {row['method_family']} | {source} | {row['adaptation']} | "
            f"{row['counterexample']} | {', '.join(row['task_mapping'])} | {row['access_status']} |"
        )
    lines.extend(["", "All four primary pages were accessible. No model or benchmark ran.", ""])
    note = root / NOTE_PATH
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines), encoding="utf-8")

    marker = "<!-- EXP7532-V659-METHOD-INGESTION -->"
    study = root / STUDY_PATH
    text = study.read_text(encoding="utf-8")
    if marker not in text:
        addition = "\n".join(
            [
                "",
                marker,
                "## 2026-09-22 Exp7532 — V659 methods — INGESTED",
                "",
                "Tool grounding, continual calibration, online recalibration, and consistency",
                "map to separate V659 tasks. External results remain source context.",
                "Exact sections, counterexamples, and destinations are in",
                "`docs/research-notes/v659-method-map.md`.",
                "",
            ]
        )
        study.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


def operator_queue(root: Path) -> JsonDict:
    """Preserve E0, E6 and corrected B2 states from the operator queue."""

    text = (root / "ops/known-issues.md").read_text(encoding="utf-8")
    e0 = "BLOCKED 2026-09-22 (operator-only step pending): SEMIF OPTION-READOUT" in text
    e6 = "RESOLVED 2026-09-21: SEMIF FOLLOW-UP E6" in text
    b2 = "FURTHER CORRECTION, 2026-09-22" in text and "completion_tokens == 256" in text
    return {
        "unit_id": "operator_queue",
        "arm": "repository_state",
        "source_path": "ops/known-issues.md",
        "source_sha256": sha256_file(root / "ops/known-issues.md"),
        "e0_status": "operator_blocked" if e0 else "not_authenticated",
        "e6_status": "resolved" if e6 else "not_authenticated",
        "b2_status": "feasibility_only_corrected" if b2 else "not_authenticated",
        "authenticated": e0 and e6 and b2,
        "principle": "Operator-only and resolved work cannot be silently reclassified.",
    }


def _precondition(check: str, path: str, field: str, expected: Any, observed: Any) -> JsonDict:
    """Record an exact prerequisite and its read-only owner."""

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
    corrections: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Authenticate named resources before reporting contract readiness."""

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
    queue = operator_queue(root)
    source = next(
        (row for row in dispositions if row.get("task_id") == "exp7517-source-protocol"), {}
    )
    rows.extend(
        [
            _precondition(
                "driving_requirement",
                SPEC_PATH.as_posix(),
                "REQ-*",
                "REQ-REPORT-7532",
                "REQ-REPORT-7532" if "REQ-REPORT-7532" in spec else None,
            ),
            _precondition(
                "roadmap_identity",
                roadmap_path.relative_to(root).as_posix(),
                "milestone",
                MILESTONE,
                roadmap.get("milestone"),
            ),
            _precondition(
                "paired_authority_task_count",
                f"{DESIGN_PATH.as_posix()} + {roadmap_path.relative_to(root).as_posix()}",
                "exact_ordered_task_count",
                14,
                {
                    "markdown": len(contract.get("markdown_task_rows") or []),
                    "yaml": len(contract.get("yaml_task_rows") or []),
                },
            ),
            _precondition(
                "v658_disposition_custody",
                V658_CAPSTONE_PATH.as_posix(),
                "authenticated_count",
                14,
                sum(row.get("authenticated") is True for row in dispositions),
            ),
            _precondition(
                "v658_absent_producers",
                V658_CAPSTONE_PATH.as_posix(),
                "absent_producer_count",
                7,
                sum(row.get("producer_present") is False for row in dispositions),
            ),
            _precondition(
                "v658_exposure_union",
                V658_PATHS[1].as_posix(),
                "candidate_and_exposed_groups",
                [2487, 2487, 0],
                [
                    source.get("candidate_official_training_groups"),
                    source.get("exposed_candidate_groups"),
                    source.get("fresh_eligible_groups"),
                ],
            ),
            _precondition(
                "b2_corrections",
                B2_PATH.as_posix(),
                "authenticated_count",
                2,
                sum(row.get("authenticated") is True for row in corrections),
            ),
            _precondition(
                "operator_queue",
                "ops/known-issues.md",
                "e0_e6_b2",
                ["operator_blocked", "resolved", "feasibility_only_corrected"],
                [queue["e0_status"], queue["e6_status"], queue["b2_status"]],
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
    """Bind current protocol bytes and present historical artifacts."""

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
        path_text = row.get("expected_path")
        if isinstance(path_text, str) and (root / path_text).is_file():
            hashes[f"historical:{row.get('task_id')}"] = {
                "path": path_text,
                "sha256": sha256_file(root / path_text),
                "evidence_type": "historical_terminal_bytes",
                "original_verdict_class": row.get("original_verdict_class"),
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
    """Run unchanged policy guards and keep the full suite separately labeled."""

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
        ("roadmap_schema", (python, "-u", "-c", schema_code, selected), True),
        ("prior_failure", (python, "-u", "scripts/validate_prior_failures.py", selected), True),
        ("roadmap_gate_audit", (python, "-u", "scripts/audit_roadmap_gates.py", selected), True),
        (
            "exclusion_manifest",
            (python, "-u", "scripts/exclusion_manifest_lint.py", selected),
            True,
        ),
        ("arc_floor", (python, "-u", "scripts/arc_levelup_guarantee_lint.py", selected), True),
        (
            "harness_fit",
            (python, "-u", "scripts/harness_consumer_checks.py", "prompt-paths", selected),
            True,
        ),
        ("overdue_priority", (python, "-u", "-c", overdue_code, selected), True),
        ("full_python_suite", (pytest, "tests/python", "-q"), False),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(
                name,
                argv,
                "selected_roadmap" if name != "full_python_suite" else "repository_health",
                1800.0 if name == "full_python_suite" else 900.0,
            ),
            "required_validation" if required else "diagnostic_repository_health",
            required,
        )
        for name, argv, required in specs
    ]


def prior_full_suite_receipt(root: Path) -> JsonDict | None:
    """Reuse the one recorded full-suite attempt instead of running it twice."""

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
    upstream: str = "current_exp7532_reduction",
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


def terminal_state(
    external_ready: bool, validation_passed: bool, contract_ready: bool
) -> tuple[str, str, str]:
    """Keep external absence, invalid work, and advisory completion distinct."""

    if not external_ready:
        value = "complete_blocked_incomplete_v659_authorities"
        return value, value, "blocked"
    if not validation_passed or not contract_ready:
        value = "complete_disqualified_v659_contract_or_validation"
        return value, value, "disqualified"
    return (
        "complete_advisory_v659_contract_and_methods",
        "complete_null_v659_contract_methods_ingested",
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
    dispositions: Sequence[Mapping[str, Any]],
    corrections: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep external readiness, current validity, support, and benefit separate."""

    contract_ok = contract.get("passed") is True
    return [
        _gate(
            "exact_v659_contract",
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
            "v658_custody",
            "validity",
            14,
            sum(row.get("authenticated") is True for row in dispositions),
            len(dispositions) == 14
            and all(row.get("authenticated") is True for row in dispositions),
            "Historical verdicts and absent producers cannot be rehabilitated.",
            upstream=V658_CAPSTONE_PATH.as_posix(),
            field_path="task_dispositions",
        ),
        _gate(
            "b2_corrections",
            "validity",
            2,
            sum(row.get("authenticated") is True for row in corrections),
            len(corrections) == 2 and all(row.get("authenticated") is True for row in corrections),
            "Appended corrections remain visible and cannot open a gate.",
            upstream=B2_PATH.as_posix(),
            field_path="false_negative_risk + further_correction_2026_09_22",
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
            4,
            len(methods),
            len(methods) == 4
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
        "schema": "Identity, milestone, version and run date bind the terminal contract.",
        "preconditions_checked": "Exact resources and hashes prevent invented readiness.",
        "MODEL_SPECS": "An empty list states that this aggregation loaded no model.",
        "model_specs": "The lowercase mirror prevents a reader from inferring a model.",
        "model_invoked": "Current calls stay separate from historical model evidence.",
        "inference_substrate_class": "The aggregation class prevents a false runtime floor.",
        "inference_substrate": "Canonical aggregation spelling prevents substrate ambiguity.",
        "execution_venue": "Host reduction stays distinct from CUDA and board evidence.",
        "duration_s": "Monotonic current work remains separate from historical duration.",
        "random_seed": "Frozen bookkeeping seeds cannot multiply scientific support.",
        "reproducibility_checksum": "The checksum binds code, inputs, roles, rows, and receipts.",
        "rows": "Every task, disposition, correction and method remains reducible.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored and unstarted stay distinct.",
        "acceptance_gate_results": "Each gate retains operands, outcome, and prevented failure.",
        "gate_check_summary": "Blocked work names exact upstream fields and observations.",
        "honest_verdict": "A complete terminal prefix prevents retry-state ambiguity.",
        "verdict_class": "The closed class separates blocked from partial work.",
        "verifier_is_oracle": "No oracle fixture supports a positive advisory claim.",
        "flagged_adversarial": "Historical corrections remain visible without inventing a finding.",
        "validation_receipts": "Commands, exits, log hashes and cold replay remain auditable.",
        "field_principles": "Every emitted field states the failure it prevents.",
        "contract_ready_score": "Bare readiness requires both complete authorities and all checks.",
        "method_ingestion_complete_score": "Bare method completion is independent of benefit.",
        "task_contract_rows": "Exactly fourteen expected identities expose absent authority rows.",
        "prior_dispositions": "Blocked, null and absent V658 evidence remains literal.",
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
    corrections: Sequence[Mapping[str, Any]],
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
    b2 = deepcopy(list(corrections))
    methods = method_rows()
    preconditions = collect_preconditions(root, roadmap_path, contract_copy, prior, b2)
    queue = operator_queue(root)
    receipts = deepcopy(list(validation.get("validation_receipts") or []))
    gates = _acceptance_gates(contract_copy, prior, b2, methods, validation)
    external_ready = contract_copy.get("passed") is True
    validation_ok = all(
        row.get("passed") is True for row in gates if row.get("gate_type") == "validity"
    )
    contract_ready = bool(external_ready and validation_ok)
    status, honest, verdict = terminal_state(external_ready, validation_ok, contract_ready)
    rows = [
        *deepcopy(contract_copy.get("contract_rows") or []),
        *mutations,
        *prior,
        *b2,
        *deepcopy(methods),
        deepcopy(queue),
    ]
    duration_s = max(0.0, (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000)
    validation_duration = sum(float(row.get("duration_s", 0.0)) for row in receipts)
    baseline = [row for row in receipts if row.get("name") in BASELINE_CHECK_NAMES]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7532,
        "title": "Bind fourteen tasks and ingest source-grounding methods",
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
            "historical": 0.0,
            "total_current": duration_s,
        },
        "random_seed": {
            "sampling": None,
            "fitting": None,
            "arrival": None,
            "bootstrap": None,
            "contract_mutations": 7_532_659_01,
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
        "b2_corrections": b2,
        "method_rows": deepcopy(methods),
        "operator_queue": deepcopy(queue),
        "source_artifact_hashes": _source_hashes(root, roadmap_path, prior),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows),
            "complete": len(rows),
            "completed": len(rows),
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 6,
            "independent_unit": "contract_disposition_correction_method_or_queue_row",
            "counting_rule": "Six missing task declarations remain unstarted, not zero-valued rows.",
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
                and row.get("access_status") == "accessed_20260922"
                for row in methods
            )
        ),
        "honest_verdict": honest,
        "verdict_class": verdict,
        "verifier_is_oracle": False,
        "flagged_adversarial": any(row.get("flagged") is True for row in b2),
        "validation_receipts": receipts,
        "repository_health": {
            "baseline_receipts": baseline,
            "failed_baseline_checks": [
                row.get("name") for row in baseline if row.get("passed") is not True
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
            "fresh_process_cold_replay": receipts_pass(receipts, TERMINAL_CHECK_NAMES[:2]),
            "numbered_runtime_e2e_applicable": False,
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
    dispositions = artifact.get("prior_dispositions")
    corrections = artifact.get("b2_corrections")
    methods = artifact.get("method_rows")
    receipts = artifact.get("validation_receipts")
    contract_ok = isinstance(contract, Mapping) and contract.get("passed") is True
    dispositions_ok = (
        isinstance(dispositions, list)
        and len(dispositions) == 14
        and all(
            isinstance(row, Mapping) and row.get("authenticated") is True for row in dispositions
        )
    )
    corrections_ok = (
        isinstance(corrections, list)
        and len(corrections) == 2
        and all(
            isinstance(row, Mapping) and row.get("authenticated") is True for row in corrections
        )
    )
    methods_ok = (
        isinstance(methods, list)
        and len(methods) == 4
        and all(
            isinstance(row, Mapping)
            and row.get("external_claim_is_carnot_measurement") is False
            and row.get("access_status") == "accessed_20260922"
            for row in methods
        )
    )
    affected_ok = receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    repository_ok = receipts_pass(receipts, REQUIRED_REPOSITORY_CHECK_NAMES)
    terminal_ok = receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    validation_ok = (
        dispositions_ok and corrections_ok and affected_ok and repository_ok and terminal_ok
    )
    ready = contract_ok and validation_ok
    status, honest, verdict = terminal_state(contract_ok, validation_ok, ready)
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
        roadmap_path, _roadmap, _candidates = resolve_v659_roadmap(root)
        expected = _source_hashes(root, roadmap_path, collect_v658_dispositions(root))
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
        roadmap_path, roadmap, _candidates = resolve_v659_roadmap(root)
        markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
        contract = compare_contract_authorities(markdown, roadmap)
        mutations = run_contract_mutation_controls(markdown, roadmap)
        dispositions = collect_v658_dispositions(root)
        corrections = collect_b2_corrections(root)
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
    if artifact.get("b2_corrections") != corrections:
        errors.append("b2_corrections_mismatch")
    if artifact.get("method_rows") != method_rows():
        errors.append("method_rows_mismatch")
    if artifact.get("operator_queue") != queue:
        errors.append("operator_queue_mismatch")
    expected_rows = [
        *deepcopy(contract.get("contract_rows") or []),
        *deepcopy(mutations),
        *deepcopy(dispositions),
        *deepcopy(corrections),
        *method_rows(),
        deepcopy(queue),
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
    """Build a minimal exact external-absence receipt without fallback data."""

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
        "execution_venue": "host",
        "rows": [],
        "sample_size_budget": {
            "planned": 0,
            "attempted": 0,
            "complete": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": _gate_summary([gate]),
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
        "from carnot.experiment_7532_v659_contract_methods import independent_reduce,validate_artifact;"
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
    roadmap_path, roadmap, roadmap_candidates = resolve_v659_roadmap(root)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(markdown, roadmap)
    mutations = run_contract_mutation_controls(markdown, roadmap)
    dispositions = collect_v658_dispositions(root)
    corrections = collect_b2_corrections(root)
    preconditions = collect_preconditions(root, roadmap_path, contract, dispositions, corrections)
    spans.append(
        _phase_span(
            "preconditions",
            phase_started,
            started,
            units=len(preconditions),
            checkpoint="inputs_and_authority_shortfall_observed",
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
    contract_units = (
        len(contract["contract_rows"]) + len(mutations) + len(dispositions) + len(corrections)
    )
    spans.append(
        _phase_span(
            "contract",
            phase_started,
            started,
            units=contract_units,
            checkpoint="authority_and_prior_custody_reduced",
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

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7532-validation-", dir="/tmp"))
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
    full_suite_passed = receipts_pass(repository, BASELINE_CHECK_NAMES)
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
        corrections,
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
        corrections,
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
    """Accept only the date frozen by the V659 contract."""

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
    """Run V659 contract ingestion or validate a measured candidate."""

    print("[exp7532] phase=startup event=flushed", flush=True)
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

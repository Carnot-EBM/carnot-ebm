"""Bind the activated V661 contract and preserve bounded method limits.

The report aggregates existing evidence. It does not load a model or measure a
scientific effect. Spec refs: REQ-REPORT-7560 and SCENARIO-REPORT-7560-*.
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
import shutil
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
from carnot.experiment_7503_v657_contract_methods import load_json, load_yaml, receipts_pass
from carnot.experiment_7559_v660_capstone import parse_conductor_statuses
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260923"
MILESTONE = "2026.09.661"
EXPERIMENT_ID = "exp7560-contract-methods"
SCHEMA = "carnot.exp7560.v661.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
V660_DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-v660-preserved-20260923.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7560_v661_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7560_v661_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7560_v661_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7560_v661_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7560_v661_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v661-method-map.md")
STUDY_PATH = Path("research-studying.md")
STUDY_MARKER = "<!-- EXP7560-V661-METHOD-INGESTION -->"

EXPECTED_TASK_IDS = tuple(
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
V660_TASK_IDS = tuple(
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
V660_PRODUCER_PATHS = {
    "exp7546-contract-methods": Path("results/experiment_7546_v660_contract_methods.json"),
    "exp7547-count-stream": Path("results/experiment_7547_v660_count_stream.json"),
    "exp7548-capture-runner": Path("results/experiment_7548_v660_capture_runner.json"),
    "exp7549-count-learning": Path("results/experiment_7549_v660_count_learning.json"),
    "exp7550-count-audit": Path("results/experiment_7550_v660_count_audit.json"),
    "exp7556-arc-corrected-custody": Path(
        "results/experiment_7556_v660_arc_corrected_custody.json"
    ),
    "exp7557-arc-generalization": Path("results/experiment_7557_v660_arc_generalization.json"),
    "exp7558-service-boundary": Path("results/experiment_7558_v660_service_boundary.json"),
    "exp7559-capstone": Path("results/experiment_7559_v660_capstone.json"),
}
ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}

METHOD_SPECS: tuple[JsonDict, ...] = (
    {
        "method_family": "calibeating_proper_loss",
        "primary_url": "https://arxiv.org/html/2603.22167v1",
        "source_revision": "arXiv:2603.22167v1 (2026-03-23)",
        "source_section": "Method sections connecting online recalibration and proper-loss regret",
        "adaptation": "Optimize a constrained probability map and retain the starting forecast.",
        "limit": "Delayed Carnot feedback does not inherit the paper's rates.",
        "local_test_or_deferral": "exp7561 shift, no-shift, chronology, and restart tests",
        "task_mapping": ["exp7561-recalibration-prototype", "exp7568-continuous-recalibration"],
    },
    {
        "method_family": "proper_scoring_limits",
        "primary_url": "https://arxiv.org/abs/2605.26703",
        "source_revision": "arXiv:2605.26703 (2026-05-26)",
        "source_section": "Sections separating calibration, calibeating, and scoring rules",
        "adaptation": "Measure Brier, log loss, decision cost, and retention separately.",
        "limit": "One improved proper score cannot prove useful typed decisions.",
        "local_test_or_deferral": "exp7568 comparator and retention gates; exp7569 audit",
        "task_mapping": ["exp7568-continuous-recalibration", "exp7569-decision-learning-audit"],
    },
    {
        "method_family": "textual_source_interventions",
        "primary_url": "https://arxiv.org/abs/2607.00895",
        "source_revision": "arXiv:2607.00895 (2026-07)",
        "source_section": "Method sections on grounded evidence and controlled source changes",
        "adaptation": "Compare original, absent, and same-role mismatched tool evidence.",
        "limit": "Visual SAFE probes do not prove textual transfer or source truth.",
        "local_test_or_deferral": "exp7563 transport controls and exp7567 held-out source tests",
        "task_mapping": ["exp7563-native-pilot", "exp7567-source-evaluation"],
    },
    {
        "method_family": "fpga_asic_service_cost",
        "primary_url": "https://arxiv.org/abs/2602.15985v2",
        "source_revision": "arXiv:2602.15985v2 (2026-09-04 revision)",
        "source_section": "Revision sections on orchestration, capacity, and memory movement",
        "adaptation": "Measure compute, movement, persistence, and acknowledgement together.",
        "limit": "External device results are not Carnot hardware acceleration.",
        "local_test_or_deferral": "exp7571 parity and full-service benchmark; FPGA work deferred",
        "task_mapping": ["exp7571-portable-calibration"],
    },
)

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
    Path("python/carnot/experiment_7546_v660_contract_methods.py"),
    Path("python/carnot/experiment_7559_v660_capstone.py"),
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/harness_fit_lint.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("research-references.md"),
    STUDY_PATH,
    Path("ops/known-issues.md"),
    DESIGN_PATH,
    V660_DESIGN_PATH,
    SPEC_PATH,
    Path("ops/conductor-log.md"),
    Path("results/experiment_7551_native_pilot.json"),
    *V660_PRODUCER_PATHS.values(),
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
    "exclusion_manifest",
    "roadmap_gate_audit",
    "arc_floor",
    "harness_fit",
    "overdue_priority",
    "publication_gate_json",
)
REQUIRED_CHECK_NAMES = (*validation_scope.REQUIRED_CHECK_NAMES, *REQUIRED_REPOSITORY_CHECK_NAMES)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    """Return an aware timestamp for a durable experiment boundary."""

    return datetime.now(UTC).isoformat()


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush a phase boundary so long validation remains observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7560] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v661_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer matching staged bytes, then accept the matching active roadmap."""

    candidates: list[JsonDict] = []
    selected: tuple[Path, JsonDict] | None = None
    for relative in (NEXT_ROADMAP_PATH, ACTIVE_ROADMAP_PATH):
        path = root / relative
        try:
            value = load_yaml(path)
            observed: Any = value.get("milestone")
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
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
                "principle": "Milestone matching prevents stale authority selection.",
            }
        )
        if selected is None and matches:
            selected = (path, value)
    if selected is None:
        raise ValueError("V661 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Retain only the task fields shared by both planning authorities."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Check that every gate reads an earlier producer field named in its prompt."""

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
    """Parse the exact task table from the preserved V661 design."""

    normalized = markdown_text.replace(
        "| Order | Task ID | Title |", "| Order | Task ID | Exact title |", 1
    ).replace("| Structured gates |", "| Structured gate |", 1)
    return parse_markdown_contract(normalized)


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare all thirteen identities, paths, substrates, and gate lists."""

    errors: list[str] = []
    try:
        parsed_yaml = parse_yaml_contract(roadmap)
    except (TypeError, ValueError) as exc:
        return {
            "passed": False,
            "errors": [f"parse_error:{exc}"],
            "markdown_milestone": None,
            "yaml_milestone": None,
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
    if len(markdown_tasks) != 13:
        errors.append("markdown_task_count")
    if len(yaml_tasks) != 13:
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
    """Return each private planning-authority defect tested in both sources."""

    return ("count", "order", "id", "title", "path", "gate_field", "milestone", "substrate")


def _mutate_yaml(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Plant one private YAML defect without changing repository bytes."""

    changed = deepcopy(dict(roadmap))
    tasks = changed.get("tasks", [])
    if mutation == "milestone":
        changed["milestone"] = "2026.09.660"
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


def mutate_yaml_for_test(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Expose a copied private mutation so tests never edit authority bytes."""

    if mutation not in mutation_names():
        raise ValueError(f"unknown mutation: {mutation}")
    return _mutate_yaml(roadmap, mutation)


def _mutate_markdown(markdown_text: str, mutation: str) -> str:
    """Plant one private design-table defect without writing a file."""

    if mutation == "milestone":
        return markdown_text.replace("**Milestone:** 2026.09.661", "**Milestone:** 2026.09.660", 1)
    lines = markdown_text.splitlines()
    if mutation == "count":
        return "\n".join(line for line in lines if not line.startswith("| 13 | exp7572-capstone |"))
    if mutation == "order":
        first = next(i for i, line in enumerate(lines) if line.startswith("| 1 | exp7560-"))
        second = next(i for i, line in enumerate(lines) if line.startswith("| 2 | exp7561-"))
        lines[first], lines[second] = lines[second], lines[first]
        return "\n".join(lines)
    replacements = {
        "id": ("| 1 | exp7560-contract-methods |", "| 1 | exp9999-changed |"),
        "title": (
            "Bind thirteen tasks and qualify methods against terminal V660 evidence",
            "Changed title",
        ),
        "path": (
            "results/experiment_7560_v661_contract_methods.json",
            "results/changed.json",
        ),
        "gate_field": ("native_tool_ready_score", "changed_score"),
        "substrate": ("| aggregation | none |", "| changed_substrate | none |"),
    }
    old, new = replacements[mutation]
    return markdown_text.replace(old, new, 1)


def run_contract_mutation_controls(
    markdown_text: str, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Require a valid baseline, then reject each private authority defect."""

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
                    "principle": "A passing baseline makes each planted defect meaningful.",
                }
            )
    return rows


def _v660_tasks(root: Path) -> list[JsonDict]:
    """Read the fourteen-task table from the preserved V660 design."""

    parsed = parse_markdown_contract((root / V660_DESIGN_PATH).read_text(encoding="utf-8"))
    tasks = parsed.get("tasks")
    if not isinstance(tasks, list) or [row.get("id") for row in tasks] != list(V660_TASK_IDS):
        raise ValueError("preserved V660 design task table is incomplete")
    return [dict(row) for row in tasks]


def _capstone_disposition_index(root: Path) -> dict[str, JsonDict]:
    """Read the prior independent reduction without replacing original bytes."""

    payload = load_json(root / V660_PRODUCER_PATHS["exp7559-capstone"])
    rows = payload.get("task_dispositions")
    if not isinstance(rows, list):
        raise ValueError("V660 capstone dispositions are unavailable")
    return {
        str(row.get("task_id")): dict(row)
        for row in rows
        if isinstance(row, Mapping) and row.get("task_id")
    }


def collect_v660_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate nine V660 producers and five distinct source absences."""

    tasks = _v660_tasks(root)
    prior = _capstone_disposition_index(root)
    conductor = parse_conductor_statuses(root, tasks)
    diagnostic_path = Path("results/experiment_7551_native_pilot.json")
    rows: list[JsonDict] = []
    for order, task in enumerate(tasks, 1):
        task_id = str(task["id"])
        scheduled = Path(str(task["deliverable"]))
        producer_path = V660_PRODUCER_PATHS.get(task_id)
        producer_present = producer_path is not None and (root / producer_path).is_file()
        source = prior.get(task_id, {})
        receipt = conductor.get(task_id)
        evidence_kind = "producer_artifact"
        evidence_path: Path | None = producer_path
        payload: JsonDict = {}
        if producer_present and producer_path is not None:
            payload = load_json(root / producer_path)
        elif task_id == "exp7551-native-pilot":
            evidence_kind = "pre_gate_diagnostic"
            evidence_path = diagnostic_path
            payload = load_json(root / diagnostic_path)
        else:
            evidence_kind = "conductor_gate_block"
            evidence_path = Path("ops/conductor-log.md")

        source_verdict = payload.get("honest_verdict") or source.get("honest_verdict")
        source_class = payload.get("verdict_class") or source.get("verdict_class")
        flagged = payload.get("flagged_adversarial")
        if not isinstance(flagged, bool):
            flagged = source.get("flagged_adversarial") is True
        diagnostic_ok = task_id != "exp7551-native-pilot" or (
            payload.get("status") == "blocked"
            and payload.get("honest_verdict") == "blocked_gate_check_failed"
        )
        conductor_ok = isinstance(receipt, Mapping) and receipt.get("status") in {
            "OK",
            "FLAGGED",
            "GATE_BLOCK",
        }
        expected_absent = task_id not in V660_PRODUCER_PATHS
        absence_ok = (root / scheduled).is_file() is not expected_absent
        authenticated = bool(
            conductor_ok
            and absence_ok
            and diagnostic_ok
            and evidence_path is not None
            and (root / evidence_path).is_file()
        )
        rows.append(
            {
                "unit_id": task_id,
                "arm": "v660_custody",
                "order": order,
                "task_id": task_id,
                "scheduled_path": scheduled.as_posix(),
                "producer_present": producer_present,
                "evidence_kind": evidence_kind,
                "evidence_path": evidence_path.as_posix() if evidence_path else None,
                "source_sha256": sha256_file(root / evidence_path) if evidence_path else None,
                "honest_verdict": source_verdict,
                "verdict_class": source_class or "blocked",
                "flagged_adversarial": flagged,
                "successful_pilot": False if task_id == "exp7551-native-pilot" else None,
                "missing_is_zero_metric": False,
                "conductor_receipt": deepcopy(dict(receipt)) if receipt else None,
                "authenticated": authenticated,
                "principle": "Original bytes and conductor state keep absence distinct from null.",
            }
        )
    return rows


def method_rows() -> list[JsonDict]:
    """Return four reviewed method sections without importing paper claims."""

    return [
        {
            "unit_id": row["method_family"],
            "arm": "method_ingestion",
            **deepcopy(row),
            "access_status": "reviewed_20260923",
            "access_failure": None,
            "external_result_is_local_measurement": False,
            "principle": "A paper can define a control without becoming local evidence.",
        }
        for row in METHOD_SPECS
    ]


def secondary_access_rows() -> list[JsonDict]:
    """Preserve failed secondary reads instead of claiming complete coverage."""

    return [
        {
            "unit_id": "semantic_scholar",
            "channel": "Semantic Scholar",
            "requested": "Citation lists for ARXIV:2507.02092 and ARXIV:2512.15605",
            "observed": "both direct reads failed",
            "complete": False,
            "principle": "Failed citation reads cannot support a complete census.",
        },
        {
            "unit_id": "hugging_face_feed",
            "channel": "Hugging Face",
            "requested": "dated 2026-09-23 paper feed",
            "observed": "feed failed to load",
            "complete": False,
            "principle": "A failed discovery feed cannot become a primary-source claim.",
        },
        {
            "unit_id": "thrml_repository",
            "channel": "GitHub",
            "requested": "direct THRML repository read",
            "observed": "direct read failed",
            "complete": False,
            "principle": "An unavailable repository cannot support a release claim.",
        },
    ]


def retired_mechanisms() -> list[JsonDict]:
    """Keep unchanged failed mechanisms closed until their stated evidence changes."""

    names = (
        "generic_external_text_scorer",
        "unchanged_importance_anchor",
        "four_expert_mixture",
        "compact_generated_span_extraction",
        "induction_budget_raise",
        "public_game_resolving",
        "binary_sampler_sweep",
    )
    return [
        {
            "unit_id": name,
            "arm": "retired_mechanism",
            "mechanism": name,
            "status": "retired_or_deferred",
            "reopened": False,
            "principle": "Unchanged failed scope cannot be revived by a new plan name.",
        }
        for name in names
    ]


def write_method_records(root: Path) -> None:
    """Write the V661 method map and one idempotent study marker."""

    lines = [
        "# V661 method map",
        "",
        "Date: 2026-09-23. Scope: advisory method accounting.",
        "External results are not local Carnot results.",
        "",
        "| Family | Primary section | Adaptation | Limit | Test or deferral | Destination |",
        "|---|---|---|---|---|---|",
    ]
    for row in method_rows():
        source = f"[{row['source_revision']}]({row['primary_url']}), {row['source_section']}"
        lines.append(
            f"| {row['method_family']} | {source} | {row['adaptation']} | "
            f"{row['limit']} | {row['local_test_or_deferral']} | "
            f"{', '.join(row['task_mapping'])} |"
        )
    lines.extend(["", "## Incomplete secondary access", ""])
    for row in secondary_access_rows():
        lines.append(f"- {row['channel']}: {row['observed']}. {row['principle']}")
    lines.extend(
        [
            "",
            "## Retired or deferred mechanisms",
            "",
            "The generic text scorer, importance anchor, four-expert mixture, generated-span",
            "extraction, induction-budget raise, public-game replay, and binary sampler sweep",
            "remain closed. No model, publication, contact, purchase, or hardware change occurred.",
            "",
        ]
    )
    note = root / NOTE_PATH
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("\n".join(lines), encoding="utf-8")

    study = root / STUDY_PATH
    text = study.read_text(encoding="utf-8") if study.is_file() else ""
    if STUDY_MARKER not in text:
        addition = "\n".join(
            [
                "",
                STUDY_MARKER,
                "## 2026-09-23 Exp7560 — V661 methods — INGESTED",
                "",
                "Proper-loss recalibration, scoring-rule limits, textual source changes,",
                "and full-service hardware costs map to separate V661 tests. External",
                "results remain context. Access failures and retired mechanisms remain",
                "explicit in `docs/research-notes/v661-method-map.md`.",
                "",
            ]
        )
        study.write_text(text.rstrip() + "\n" + addition, encoding="utf-8")


def operator_queue(root: Path) -> JsonDict:
    """Preserve E0 as operator-blocked and E6 as resolved."""

    path = root / "ops/known-issues.md"
    text = path.read_text(encoding="utf-8")
    e0 = "BLOCKED 2026-09-22 (operator-only step pending): SEMIF OPTION-READOUT" in text
    e6 = "RESOLVED 2026-09-21: SEMIF FOLLOW-UP E6" in text
    return {
        "unit_id": "operator_queue",
        "arm": "repository_state",
        "source_path": "ops/known-issues.md",
        "source_sha256": sha256_file(path),
        "E0": {"status": "operator_blocked" if e0 else "not_authenticated"},
        "E6": {"status": "resolved" if e6 else "not_authenticated"},
        "authenticated": e0 and e6,
        "principle": "Operator-only and resolved work cannot be silently reclassified.",
    }


def _precondition(
    check: str, upstream: str, path: str, field: str, expected: Any, observed: Any
) -> JsonDict:
    """Record an exact prerequisite before dependent aggregation starts."""

    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": "==",
        "passed": observed == expected,
        "principle": "Exact observations prevent reduction over fallback data.",
    }


def collect_preconditions(root: Path, roadmap_path: Path) -> list[JsonDict]:
    """Check required bytes, requirement identity, selected authority, and resources."""

    rows: list[JsonDict] = []
    for relative in SOURCE_INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        row = _precondition(
            f"source_bytes:{relative.as_posix()}",
            relative.as_posix(),
            relative.as_posix(),
            "bytes",
            "readable_nonempty_bytes",
            "readable_nonempty_bytes" if available else "absent_or_empty",
        )
        row["sha256"] = sha256_file(path) if available else None
        rows.append(row)
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    rows.append(
        _precondition(
            "driving_requirement",
            SPEC_PATH.as_posix(),
            SPEC_PATH.as_posix(),
            "REQ-*",
            "REQ-REPORT-7560",
            "REQ-REPORT-7560" if "REQ-REPORT-7560" in spec_text else "absent",
        )
    )
    try:
        milestone = load_yaml(roadmap_path).get("milestone")
    except (OSError, ValueError, json.JSONDecodeError):
        milestone = "unreadable"
    rows.append(
        _precondition(
            "selected_roadmap_authority",
            roadmap_path.name,
            roadmap_path.relative_to(root).as_posix(),
            "milestone",
            MILESTONE,
            milestone,
        )
    )
    disk = shutil.disk_usage(root)
    resources = {
        "cpu_count": os.cpu_count(),
        "free_bytes": disk.free,
        "gpu_required": False,
    }
    rows.append(
        {
            "check": "aggregation_resource",
            "upstream": "current_host",
            "path": str(root),
            "field": "cpu_and_storage",
            "expected": "cpu_count>0 and free_bytes>0",
            "observed": resources,
            "op": "all",
            "passed": bool((os.cpu_count() or 0) > 0 and disk.free > 0),
            "principle": "Aggregation requires readable storage and one CPU.",
        }
    )
    return rows


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Bind current authorities, code, methods, and all historical evidence bytes."""

    paths = [
        *SOURCE_INPUT_PATHS,
        roadmap_path.relative_to(root),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        NOTE_PATH,
    ]
    rows = [
        {
            "path": relative.as_posix(),
            "sha256": sha256_file(root / relative),
            "input_role": "current_authority_code_or_method_record",
        }
        for relative in dict.fromkeys(paths)
        if (root / relative).is_file()
    ]
    for row in dispositions:
        rows.append(
            {
                "path": row.get("evidence_path"),
                "sha256": row.get("source_sha256"),
                "input_role": "historical_v660_producer_or_diagnostic",
                "task_id": row.get("task_id"),
                "producer_present": row.get("producer_present"),
                "honest_verdict": row.get("honest_verdict"),
                "verdict_class": row.get("verdict_class"),
                "flagged_adversarial": row.get("flagged_adversarial"),
            }
        )
    return rows


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build scoped tests, coverage, lint, format, type, and spec checks."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject expanded targets, command drift, and missing private parents."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def build_repository_check_plan(root: Path, roadmap_path: Path) -> list[PlannedCommand]:
    """Run each unchanged roadmap guard against the selected authority."""

    python = str(root / ".venv/bin/python")
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
        ("roadmap_schema", (python, "-u", "-c", schema_code, selected)),
        ("prior_failure", (python, "-u", "scripts/validate_prior_failures.py", selected)),
        ("exclusion_manifest", (python, "-u", "scripts/exclusion_manifest_lint.py", selected)),
        ("roadmap_gate_audit", (python, "-u", "scripts/audit_roadmap_gates.py", selected)),
        ("arc_floor", (python, "-u", "scripts/arc_levelup_guarantee_lint.py", selected)),
        ("harness_fit", (python, "-u", "scripts/harness_fit_lint.py", selected)),
        ("overdue_priority", (python, "-u", "-c", overdue_code, selected)),
        ("publication_gate_json", (python, "-u", "scripts/publication_gate.py", "--json")),
    )
    return [
        PlannedCommand(
            validation_scope.CommandSpec(name, argv, "selected_roadmap", 900.0),
            "required_validation",
            True,
        )
        for name, argv in specs
    ]


def _publication_from_receipts(root: Path, receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Read G1-G4 from the exact validation log instead of rerunning the command."""

    receipt = next((row for row in receipts if row.get("name") == "publication_gate_json"), None)
    parsed: Mapping[str, Any] = {}
    if receipt is not None:
        log_path = Path(str(receipt.get("log_path") or ""))
        if not log_path.is_absolute():
            log_path = root / log_path
        try:
            value = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(value, Mapping):
                parsed = value
        except (OSError, json.JSONDecodeError):
            parsed = {}
    raw = parsed.get("gates")
    gates = raw if isinstance(raw, Mapping) else {}
    normalized = {
        name: deepcopy(gates.get(name) or {"pass": False}) for name in ("G1", "G2", "G3", "G4")
    }
    unmet = [name for name, gate in normalized.items() if gate.get("pass") is not True]
    return {
        "gates": normalized,
        "paper_ready": not unmet,
        "unmet_gates": unmet,
        "command_argv": deepcopy(receipt.get("command_argv")) if receipt else [],
        "exit_code": receipt.get("exit_code") if receipt else None,
        "stdout_sha256": receipt.get("log_sha256") if receipt else None,
        "publication_performed": False,
    }


def activation_state_controls(roadmap: Mapping[str, Any]) -> list[JsonDict]:
    """Exercise selection semantics for staged-present and staged-consumed states."""

    matches = roadmap.get("milestone") == MILESTONE
    return [
        {
            "unit_id": "staged_present",
            "arm": "authority_state",
            "staged_exists": True,
            "staged_milestone": roadmap.get("milestone"),
            "active_exists": True,
            "selected": NEXT_ROADMAP_PATH.as_posix() if matches else None,
            "expected": NEXT_ROADMAP_PATH.as_posix(),
            "qualified": matches,
            "principle": "A matching staged file has priority before activation.",
        },
        {
            "unit_id": "staging_consumed",
            "arm": "authority_state",
            "staged_exists": False,
            "staged_milestone": None,
            "active_exists": True,
            "selected": ACTIVE_ROADMAP_PATH.as_posix() if matches else None,
            "expected": ACTIVE_ROADMAP_PATH.as_posix(),
            "qualified": matches,
            "principle": "A consumed staged path selects the matching active authority.",
        },
    ]


def _gate(
    check: str,
    category: str,
    upstream: str,
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
    op: str = "==",
) -> JsonDict:
    """Attach exact operands and a failure-prevention principle to one gate."""

    return {
        "check": check,
        "category": category,
        "upstream": upstream,
        "path": path,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    states: Sequence[Mapping[str, Any]],
    prior: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep advisory validity and readiness separate from scientific benefit."""

    contract_ok = contract.get("passed") is True and len(contract.get("contract_rows") or []) == 13
    mutations_ok = len(mutations) == 2 * len(mutation_names()) and all(
        row.get("qualified") is True for row in mutations
    )
    states_ok = len(states) == 2 and all(row.get("qualified") is True for row in states)
    prior_ok = (
        len(prior) == 14
        and sum(row.get("producer_present") is True for row in prior) == 9
        and sum(row.get("producer_present") is False for row in prior) == 5
        and all(row.get("authenticated") is True for row in prior)
    )
    methods_ok = 3 <= len(methods) <= 5 and all(
        row.get("local_test_or_deferral")
        and row.get("external_result_is_local_measurement") is False
        for row in methods
    )
    affected_ok = validation.get("affected_checks_passed") is True
    repository_ok = validation.get("repository_checks_passed") is True
    terminal_ok = validation.get("terminal_checks_passed") is True
    return [
        _gate(
            "exact_v661_contract",
            "readiness",
            "V661 design and selected roadmap",
            DESIGN_PATH.as_posix(),
            "thirteen_complete_rows",
            True,
            contract_ok,
            contract_ok,
            "Incomplete or different planning authorities cannot open readiness.",
        ),
        _gate(
            "activation_states",
            "readiness",
            "private authority-state controls",
            "private fixtures",
            "qualified_states",
            2,
            sum(row.get("qualified") is True for row in states),
            states_ok,
            "Activation must not turn a consumed staging path into a defect.",
        ),
        _gate(
            "private_contract_mutations",
            "validity",
            "current contract reducer",
            "private fixtures",
            "rejected_mutations",
            2 * len(mutation_names()),
            sum(row.get("qualified") is True for row in mutations),
            mutations_ok,
            "Private drift must fail before the real contract can pass.",
        ),
        _gate(
            "v660_custody",
            "validity",
            "preserved V660 evidence",
            V660_DESIGN_PATH.as_posix(),
            "authenticated_dispositions",
            {"total": 14, "producers": 9, "absences": 5},
            {
                "total": len(prior),
                "producers": sum(row.get("producer_present") is True for row in prior),
                "absences": sum(row.get("producer_present") is False for row in prior),
            },
            prior_ok,
            "Missing source work cannot become a null or successful pilot.",
        ),
        _gate(
            "method_ingestion",
            "readiness",
            "V661 primary-source review",
            "research-references.md",
            "bounded_method_rows",
            "3..5 with test or deferral",
            len(methods),
            methods_ok,
            "External methods need a local test or a named deferral.",
        ),
        _gate(
            "affected_validation",
            "validity",
            "current scoped validation",
            TEST_PATH.as_posix(),
            "required_checks_passed",
            True,
            affected_ok,
            affected_ok,
            "Invalid changed code cannot support readiness.",
        ),
        _gate(
            "selected_roadmap_guards",
            "validity",
            "unchanged repository guards",
            "selected_roadmap_path",
            "all_required_guard_exits",
            True,
            repository_ok,
            repository_ok,
            "Every guard must inspect the authority that actually exists.",
        ),
        _gate(
            "terminal_readers",
            "validity",
            "fresh-process terminal validation",
            "measured terminal candidate",
            "all_terminal_readers_passed",
            True,
            terminal_ok,
            terminal_ok,
            "A producer cannot be its own final verifier.",
        ),
        _gate(
            "scientific_benefit_not_claimed",
            "benefit",
            "advisory scope",
            RESULT_PATH.as_posix(),
            "positive_claim",
            False,
            False,
            True,
            "Contract completion cannot substitute for empirical benefit.",
        ),
    ]


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain every failed check and the first exact upstream operand."""

    failures = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failures,
        "failed_count": len(failures),
        "failed_checks": failures,
        "first_failure": failures[0] if failures else None,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain how every emitted top-level field prevents a reporting failure."""

    specific = {
        "experiment_id": "Bind the exact task, milestone, and frozen run date.",
        "preconditions_checked": "Record inputs and resources before measurement.",
        "MODEL_SPECS": "An empty list states that aggregation loaded no model.",
        "model_specs": "The lowercase mirror prevents inferred model work.",
        "model_invoked": "Typed zero calls keep historical work separate.",
        "inference_substrate_class": "Aggregation prevents a false model runtime floor.",
        "inference_substrate": "Canonical spelling prevents substrate ambiguity.",
        "execution_venue": "Legal host venue stays separate from device identity.",
        "duration_s": "Monotonic current time stays separate from historical time.",
        "random_seed": "Frozen bookkeeping seeds cannot multiply support.",
        "reproducibility_checksum": "The checksum binds sources, settings, rows, and receipts.",
        "rows": "Every compared task retains its absolute contract disposition.",
        "sample_size_budget": "Attempted, completed, absent, and unstarted stay distinct.",
        "acceptance_gate_results": "Every gate retains operands, category, and outcome.",
        "gate_check_summary": "A failure names upstream, path, field, expected, and observed.",
        "honest_verdict": "A complete terminal prefix prevents retry ambiguity.",
        "verdict_class": "The closed class separates a valid advisory null from failure.",
        "verifier_is_oracle": "No fixture result becomes an empirical claim.",
        "flagged_adversarial": "Current status stays separate from the preserved Exp7546 flag.",
        "validation_receipts": "Exact commands, exits, and log hashes remain auditable.",
        "field_principles": "Each emitted field names the failure it prevents.",
        "contract_ready_score": "Bare one requires both authority states and rejected drift.",
        "method_ingestion_complete_score": "Bare one requires reviewed methods and limits.",
        "selected_roadmap_path": "The selected authority must exist and match V661.",
        "prior_dispositions": "Fourteen V660 rows retain nine producers and five absences.",
        "publication_gates": "Stable G1-G4 remain read-only and separate from this task.",
    }
    return {
        field: specific.get(field, f"The {field} field preserves exact scope and meaning.")
        for field in fields
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all emitted evidence except the checksum's own value."""

    payload = deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def _publication_for_artifact(root: Path, validation: Mapping[str, Any]) -> JsonDict:
    """Use the current guard receipt, with prior stable G1-G4 only for unit construction."""

    supplied = validation.get("publication_gates")
    if isinstance(supplied, Mapping):
        result = deepcopy(dict(supplied))
    else:
        prior = load_json(root / V660_PRODUCER_PATHS["exp7559-capstone"])
        value = prior.get("publication_gates")
        result = deepcopy(dict(value)) if isinstance(value, Mapping) else {}
    result["publication_performed"] = False
    return result


def _normalized_validation(validation: Mapping[str, Any]) -> JsonDict:
    """Accept explicit phase reductions while keeping unit fixtures compact."""

    required = validation.get("required_checks_passed") is True
    terminal = validation.get("terminal_validation_passed") is True
    return {
        **deepcopy(dict(validation)),
        "affected_checks_passed": validation.get("affected_checks_passed", required) is True,
        "repository_checks_passed": validation.get("repository_checks_passed", required) is True,
        "terminal_checks_passed": validation.get("terminal_checks_passed", terminal or required)
        is True,
    }


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
    """Build one complete advisory record from authenticated current inputs."""

    roadmap = load_yaml(roadmap_path)
    contract_copy = deepcopy(dict(contract))
    mutations = deepcopy(list(mutation_rows))
    prior = deepcopy(list(dispositions))
    methods = method_rows()
    access = secondary_access_rows()
    retired = retired_mechanisms()
    states = activation_state_controls(roadmap)
    normalized = _normalized_validation(validation)
    receipts = deepcopy(list(normalized.get("validation_receipts") or []))
    gates = _acceptance_gates(contract_copy, mutations, states, prior, methods, normalized)
    valid = all(row["passed"] is True for row in gates if row["category"] != "benefit")
    methods_ready = 3 <= len(methods) <= 5 and all(
        row.get("local_test_or_deferral")
        and row.get("external_result_is_local_measurement") is False
        for row in methods
    )
    duration_s = max(0.0, (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000)
    validation_duration = sum(float(row.get("duration_s", 0.0)) for row in receipts)
    if valid:
        status = "complete"
        honest = "complete_null_v661_contract_methods_ingested"
        verdict = "null"
    else:
        status = "complete_disqualified"
        honest = "complete_disqualified_v661_contract_or_validation"
        verdict = "disqualified"
    rows = deepcopy(list(contract_copy.get("contract_rows") or []))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7560,
        "title": "Bind thirteen tasks and qualify methods against terminal V660 evidence",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "positive_claim": False,
        "flagged_adversarial": False,
        "verifier_is_oracle": False,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": duration_s,
        "process_identity": {"pid": os.getpid(), "owner": "current_process"},
        "clock_identity": {"utc": "datetime.now(UTC)", "monotonic": "time.monotonic_ns"},
        "phase_spans": deepcopy(list(phase_spans)),
        "duration_components_s": {
            "aggregation": max(0.0, duration_s - validation_duration),
            "validation": validation_duration,
            "historical": 0.0,
            "total_current": duration_s,
        },
        "preconditions_checked": collect_preconditions(root, roadmap_path),
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "historical_model_receipts_counted_as_current": False,
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "device_identity": {
            "machine": platform.machine(),
            "node": platform.node(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "gpu_required": False,
            "cuda_used": False,
        },
        "random_seed": {
            "model": None,
            "fitting": None,
            "ordering": None,
            "bootstrap": None,
            "contract_mutations": 7_560_661_01,
            "explanation": "Deterministic aggregation makes no stochastic estimate.",
        },
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "selected_authority_state": "staged"
        if roadmap_path.name == NEXT_ROADMAP_PATH.name
        else "active_after_activation",
        "roadmap_resolution_candidates": deepcopy(list(roadmap_candidates)),
        "authority_state_rows": states,
        "contract_comparison": contract_copy,
        "contract_mutation_rows": mutations,
        "rows": rows,
        "prior_dispositions": prior,
        "completed_ledger_ends_at": "2026.09.659",
        "v660_archive_entry_invented": False,
        "method_rows": deepcopy(methods),
        "secondary_access_rows": deepcopy(access),
        "retired_mechanisms": deepcopy(retired),
        "operator_queue": operator_queue(root),
        "source_artifact_hashes": _source_hashes(root, roadmap_path, prior),
        "sample_size_budget": {
            "planned": 13,
            "attempted": 13,
            "completed": 13,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 0,
            "independent_unit": "ordered_v661_task_contract_row",
            "historical_v660": {
                "planned": 14,
                "producer_artifacts": 9,
                "source_absences": 5,
            },
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "contract_ready_score": int(valid),
        "method_ingestion_complete_score": int(methods_ready),
        "validation_receipts": receipts,
        "repository_health": {
            "status": "preserved_unrelated_debt_not_retested",
            "known_issues_path": "ops/known-issues.md",
            "known_issues_sha256": sha256_file(root / "ops/known-issues.md"),
            "full_python_suite_launched": False,
            "affects_required_scoped_validation": False,
        },
        "publication_gates": _publication_for_artifact(root, normalized),
        "validation_manifest": {
            "experiment_id": VALIDATION_MANIFEST.experiment_id,
            "test_paths": list(VALIDATION_MANIFEST.test_paths),
            "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
            "static_paths": list(VALIDATION_MANIFEST.static_paths),
            "documentation_paths": [
                SPEC_PATH.as_posix(),
                NOTE_PATH.as_posix(),
                STUDY_PATH.as_posix(),
            ],
            "full_python_suite_applicable": False,
            "numbered_runtime_e2e_applicable": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "cold_replay_required": True,
            "cold_replay_passed": receipts_pass(receipts, TERMINAL_CHECK_NAMES[:2]),
            "read_only_reporting": True,
            "numbered_runtime_e2e_applicable": False,
            "learning_lifecycle_applicable": False,
            "arc_runtime_smoke_applicable": False,
        },
        "advisory_only": True,
        "global_science_gate": False,
        "roadmap_activation_performed": False,
        "roadmap_archive_performed": False,
        "active_roadmap_modified": False,
        "research_conductor_modified": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
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


def _test_phase_spans() -> list[JsonDict]:
    """Create deterministic named spans for pure unit artifact construction."""

    return [
        {
            "phase": phase,
            "started_elapsed_s": 0.0,
            "ended_elapsed_s": 0.0,
            "duration_s": 0.0,
            "completed_units": units,
            "checkpoint": checkpoint,
        }
        for phase, units, checkpoint in (
            ("preconditions", len(SOURCE_INPUT_PATHS) + 3, "inputs_observed"),
            ("model_load", 0, "no_current_model_load"),
            ("generation", 0, "no_current_generation"),
            ("contract", 43, "contract_mutations_and_v660_custody"),
            ("method_ingestion", 4, "four_primary_method_sections"),
            ("validation", len(REQUIRED_CHECK_NAMES), "scoped_checks_and_guards"),
            ("terminal_validation", 4, "fresh_process_readers"),
            ("write", 1, "atomic_terminal_artifact"),
        )
    ]


def build_artifact_for_test(*, validation: Mapping[str, Any]) -> JsonDict:
    """Build deterministic evidence from the current worktree for unit tests."""

    roadmap_path, roadmap, candidates = resolve_v661_roadmap(REPO_ROOT)
    design = (REPO_ROOT / DESIGN_PATH).read_text(encoding="utf-8")
    return build_artifact(
        REPO_ROOT,
        roadmap_path,
        candidates,
        compare_contract_authorities(design, roadmap),
        run_contract_mutation_controls(design, roadmap),
        collect_v660_dispositions(REPO_ROOT),
        validation,
        started_at_utc="2026-09-23T00:00:00+00:00",
        ended_at_utc="2026-09-23T00:00:00+00:00",
        started_monotonic_ns=0,
        ended_monotonic_ns=0,
        phase_spans=_test_phase_spans(),
    )


def independent_reduce(artifact: Mapping[str, Any], *, require_terminal: bool = True) -> JsonDict:
    """Recompute readiness and terminal class from embedded comparative evidence."""

    contract = artifact.get("contract_comparison")
    rows = artifact.get("rows")
    mutations = artifact.get("contract_mutation_rows")
    states = artifact.get("authority_state_rows")
    prior = artifact.get("prior_dispositions")
    methods = artifact.get("method_rows")
    access = artifact.get("secondary_access_rows")
    receipts = artifact.get("validation_receipts")
    contract_ok = (
        isinstance(contract, Mapping)
        and contract.get("passed") is True
        and isinstance(rows, list)
        and [row.get("unit_id") for row in rows if isinstance(row, Mapping)]
        == list(EXPECTED_TASK_IDS)
        and all(isinstance(row, Mapping) and row.get("passed") is True for row in rows)
    )
    mutations_ok = (
        isinstance(mutations, list)
        and len(mutations) == 2 * len(mutation_names())
        and all(isinstance(row, Mapping) and row.get("qualified") is True for row in mutations)
    )
    states_ok = (
        isinstance(states, list)
        and len(states) == 2
        and all(isinstance(row, Mapping) and row.get("qualified") is True for row in states)
    )
    prior_by_id = {str(row.get("task_id")): row for row in prior or [] if isinstance(row, Mapping)}
    prior_ok = (
        isinstance(prior, list)
        and [row.get("task_id") for row in prior if isinstance(row, Mapping)] == list(V660_TASK_IDS)
        and all(isinstance(row, Mapping) and row.get("authenticated") is True for row in prior)
        and sum(row.get("producer_present") is True for row in prior if isinstance(row, Mapping))
        == 9
        and sum(row.get("producer_present") is False for row in prior if isinstance(row, Mapping))
        == 5
        and prior_by_id.get("exp7546-contract-methods", {}).get("flagged_adversarial") is True
        and prior_by_id.get("exp7548-capture-runner", {}).get("verdict_class") == "blocked"
        and prior_by_id.get("exp7549-count-learning", {}).get("verdict_class") == "null"
        and prior_by_id.get("exp7550-count-audit", {}).get("verdict_class") == "null"
        and prior_by_id.get("exp7551-native-pilot", {}).get("evidence_kind")
        == "pre_gate_diagnostic"
        and prior_by_id.get("exp7551-native-pilot", {}).get("successful_pilot") is False
    )
    method_families = {
        row.get("method_family") for row in methods or [] if isinstance(row, Mapping)
    }
    methods_ok = (
        isinstance(methods, list)
        and 3 <= len(methods) <= 5
        and method_families
        == {
            "calibeating_proper_loss",
            "proper_scoring_limits",
            "textual_source_interventions",
            "fpga_asic_service_cost",
        }
        and all(
            isinstance(row, Mapping)
            and row.get("local_test_or_deferral")
            and row.get("external_result_is_local_measurement") is False
            for row in methods
        )
        and isinstance(access, list)
        and all(isinstance(row, Mapping) and row.get("complete") is False for row in access)
    )
    preconditions = artifact.get("preconditions_checked")
    preconditions_ok = isinstance(preconditions, list) and all(
        isinstance(row, Mapping) and row.get("passed") is True for row in preconditions
    )
    zero_calls = artifact.get("invocation_counts") == ZERO_INVOCATION_COUNTS
    affected_ok = receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    repository_ok = receipts_pass(receipts, REQUIRED_REPOSITORY_CHECK_NAMES)
    terminal_ok = receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    valid = all(
        (
            contract_ok,
            mutations_ok,
            states_ok,
            prior_ok,
            methods_ok,
            preconditions_ok,
            zero_calls,
            affected_ok,
            repository_ok,
            terminal_ok,
        )
    )
    expected = {
        "contract_ready_score": int(valid),
        "method_ingestion_complete_score": int(methods_ok),
        "honest_verdict": "complete_null_v661_contract_methods_ingested"
        if valid
        else "complete_disqualified_v661_contract_or_validation",
        "verdict_class": "null" if valid else "disqualified",
    }
    matches = all(artifact.get(key) == value for key, value in expected.items())
    return {
        "passed": valid and matches,
        "contract_ok": contract_ok,
        "mutations_ok": mutations_ok,
        "activation_states_ok": states_ok,
        "prior_dispositions_ok": prior_ok,
        "methods_ok": methods_ok,
        "preconditions_ok": preconditions_ok,
        "zero_current_calls": zero_calls,
        "affected_validation_ok": affected_ok,
        "repository_guards_ok": repository_ok,
        "terminal_readers_ok": terminal_ok,
        "expected": expected,
        "artifact_matches": matches,
    }


def _source_hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash every source row that claims existing bytes."""

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
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, custody, reductions, principles, and checksum."""

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
        "preconditions_checked",
        "MODEL_SPECS",
        "model_specs",
        "model_invoked",
        "invocation_counts",
        "inference_substrate_class",
        "inference_substrate",
        "execution_venue",
        "duration_s",
        "random_seed",
        "reproducibility_checksum",
        "rows",
        "sample_size_budget",
        "acceptance_gate_results",
        "gate_check_summary",
        "verifier_is_oracle",
        "flagged_adversarial",
        "validation_receipts",
        "field_principles",
        "contract_ready_score",
        "method_ingestion_complete_score",
        "selected_roadmap_path",
        "prior_dispositions",
        "publication_gates",
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
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    for field in ("contract_ready_score", "method_ingestion_complete_score"):
        if type(artifact.get(field)) is not int or artifact.get(field) not in (0, 1):
            errors.append(f"{field}_bare_integer_required")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("model_specs_must_be_empty")
    if (
        artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("current_model_calls_nonzero")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("substrate_class_invalid")
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("substrate_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    selected_text = artifact.get("selected_roadmap_path")
    selected = root / str(selected_text)
    if (
        selected_text not in {ACTIVE_ROADMAP_PATH.as_posix(), NEXT_ROADMAP_PATH.as_posix()}
        or not selected.is_file()
    ):
        errors.append("selected_roadmap_path_invalid")
    else:
        try:
            if load_yaml(selected).get("milestone") != MILESTONE:
                errors.append("selected_roadmap_milestone_invalid")
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("selected_roadmap_milestone_invalid")
    gates = artifact.get("acceptance_gate_results")
    if (
        not isinstance(gates, list)
        or not gates
        or any(
            not isinstance(row, Mapping)
            or not all(
                key in row
                for key in ("expected", "observed", "op", "passed", "category", "principle")
            )
            for row in gates
        )
    ):
        errors.append("acceptance_gate_shape_invalid")
    publication = artifact.get("publication_gates")
    if not isinstance(publication, Mapping) or set((publication.get("gates") or {})) != {
        "G1",
        "G2",
        "G3",
        "G4",
    }:
        errors.append("publication_gates_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_incomplete")
    elif any(not isinstance(value, str) or not value for value in principles.values()):
        errors.append("field_principles_invalid")
    if not _source_hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    reduction = independent_reduce(artifact, require_terminal=require_terminal)
    if reduction.get("passed") is not True:
        errors.append("independent_reduction_failed")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("checksum_mismatch")
    if len(json.dumps(artifact, sort_keys=True).encode("utf-8")) >= 20 * 1024 * 1024:
        errors.append("terminal_artifact_size_limit")
    return list(dict.fromkeys(errors))


def build_blocked_artifact(missing: str, *, field: str, expected: Any, observed: Any) -> JsonDict:
    """Build a complete external-input block without fallback evidence."""

    failure = {
        "check": "required_input",
        "upstream": missing,
        "path": missing,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": "==",
        "passed": False,
    }
    gate = {
        **deepcopy(failure),
        "category": "external_precondition",
        "principle": "Missing external bytes cannot be replaced by fallback data.",
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7560,
        "title": "Bind thirteen tasks and qualify methods against terminal V660 evidence",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": "complete_blocked",
        "honest_verdict": "complete_blocked_required_input_unavailable",
        "verdict_class": "blocked",
        "positive_claim": False,
        "flagged_adversarial": False,
        "verifier_is_oracle": False,
        "preconditions_checked": [gate],
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "planned_inference_substrate_class": "aggregation",
        "inference_substrate_class": "aggregation",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "execution_venue": "host",
        "device_identity": {"gpu_required": False, "cuda_used": False},
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": {"model": None, "fitting": None, "ordering": None, "bootstrap": None},
        "selected_roadmap_path": missing,
        "rows": [],
        "prior_dispositions": [],
        "method_rows": [],
        "source_artifact_hashes": [],
        "sample_size_budget": {
            "planned": 13,
            "attempted": 0,
            "completed": 0,
            "excluded": 0,
            "failed": 0,
            "censored": 0,
            "unstarted": 13,
            "independent_unit": "ordered_v661_task_contract_row",
        },
        "acceptance_gate_results": [gate],
        "gate_check_summary": {
            "passed": False,
            "failed_count": 1,
            "failed_checks": [failure],
            "first_failure": failure,
        },
        "contract_ready_score": 0,
        "method_ingestion_complete_score": 0,
        "validation_receipts": [],
        "publication_gates": {
            "gates": {name: {"pass": False} for name in ("G1", "G2", "G3", "G4")},
            "paper_ready": False,
            "unmet_gates": ["reader_not_run"],
            "publication_performed": False,
        },
        "advisory_only": True,
        "publication_performed": False,
        "generator_weights_changed": False,
        "field_principles": {},
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _phase_span(
    phase: str, phase_started: float, run_started: float, *, units: int, checkpoint: str
) -> JsonDict:
    """Close one monotonic phase and name its durable checkpoint."""

    ended = time.monotonic()
    return {
        "phase": phase,
        "started_elapsed_s": phase_started - run_started,
        "ended_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint": checkpoint,
    }


def _terminal_commands(root: Path, candidate: Path) -> list[PlannedCommand]:
    """Build entrypoint replay, independent reduction, and strict readers."""

    python = str(root / ".venv/bin/python")
    specs = (
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--cold-validate", str(candidate)),
            "candidate_capability_e2e",
            900.0,
        ),
        validation_scope.CommandSpec(
            "independent_cold_reducer",
            (python, "-u", WRAPPER_PATH.as_posix(), "--independent-reduce", str(candidate)),
            "candidate_raw_reduction",
            900.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "candidate_safety",
            900.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "candidate_row_consistency",
            900.0,
        ),
    )
    return [PlannedCommand(spec, "required_validation", True) for spec in specs]


def run_experiment(root: Path, run_date: str) -> JsonDict:  # pragma: no cover - capability E2E.
    """Run bounded checks and atomically publish the validated advisory."""

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
            missing.as_posix(),
            field="bytes",
            expected="readable_nonempty_bytes",
            observed="absent",
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=missing.as_posix())
        return blocked
    try:
        roadmap_path, roadmap, candidates = resolve_v661_roadmap(root)
    except ValueError:
        blocked = build_blocked_artifact(
            ACTIVE_ROADMAP_PATH.as_posix(),
            field="milestone",
            expected=MILESTONE,
            observed="no matching staged or active authority",
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=ACTIVE_ROADMAP_PATH)
        return blocked
    preconditions = collect_preconditions(root, roadmap_path)
    failed_precondition = next(
        (row for row in preconditions if row.get("passed") is not True), None
    )
    if failed_precondition is not None:
        blocked = build_blocked_artifact(
            str(failed_precondition["upstream"]),
            field=str(failed_precondition["field"]),
            expected=failed_precondition["expected"],
            observed=failed_precondition["observed"],
        )
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=failed_precondition["path"])
        return blocked
    design = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(design, roadmap)
    mutations = run_contract_mutation_controls(design, roadmap)
    dispositions = collect_v660_dispositions(root)
    spans.append(
        _phase_span(
            "preconditions",
            phase_started,
            started,
            units=len(preconditions),
            checkpoint="inputs_resources_and_authority_observed",
        )
    )
    progress(started, "preconditions", "after", completed_units=len(preconditions))

    for phase in ("model_load", "generation"):
        phase_started = time.monotonic()
        progress(started, phase, "before", completed_units=0)
        spans.append(
            _phase_span(phase, phase_started, started, units=0, checkpoint="no_current_model_work")
        )
        progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    progress(started, "contract", "before")
    contract_units = len(contract.get("contract_rows") or []) + len(mutations) + len(dispositions)
    spans.append(
        _phase_span(
            "contract",
            phase_started,
            started,
            units=contract_units,
            checkpoint="v661_contract_and_v660_custody_reduced",
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
            checkpoint="method_note_and_study_marker_written",
        )
    )
    progress(started, "method_ingestion", "after", completed_units=len(METHOD_SPECS))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7560-validation-", dir="/tmp"))
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
                SPEC_PATH.as_posix(),
                NOTE_PATH.as_posix(),
                STUDY_PATH.as_posix(),
            ],
        },
    )

    phase_started = time.monotonic()
    repository_plan = build_repository_check_plan(root, roadmap_path)
    progress(
        started,
        "validation",
        "before_subprocesses",
        planned_units=len(affected_plan) + len(repository_plan),
    )
    affected = run_categorized_commands(
        root,
        [PlannedCommand(spec, "required_validation", True) for spec in affected_plan],
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
    repository_passed = receipts_pass(repository, REQUIRED_REPOSITORY_CHECK_NAMES)
    publication = _publication_from_receipts(root, repository)
    spans.append(
        _phase_span(
            "validation",
            phase_started,
            started,
            units=len(affected) + len(repository),
            checkpoint="scoped_checks_and_selected_authority_guards",
        )
    )
    progress(
        started,
        "validation",
        "after_subprocesses",
        affected=affected_reduction["passed"],
        repository=repository_passed,
    )

    validation = {
        "validation_receipts": [*affected, *repository],
        "affected_checks_passed": affected_reduction["passed"],
        "repository_checks_passed": repository_passed,
        "terminal_checks_passed": True,
        "publication_gates": publication,
    }
    candidate = build_artifact(
        root,
        roadmap_path,
        candidates,
        contract,
        mutations,
        dispositions,
        validation,
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
        candidates,
        contract,
        mutations,
        dispositions,
        {
            "validation_receipts": [*affected, *repository, *terminal],
            "affected_checks_passed": affected_reduction["passed"],
            "repository_checks_passed": repository_passed,
            "terminal_checks_passed": terminal_passed,
            "publication_gates": publication,
        },
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        started_monotonic_ns=started_ns,
        ended_monotonic_ns=time.monotonic_ns(),
        phase_spans=spans,
    )
    errors = validate_artifact(final, root=root, require_terminal=True)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic", path=RESULT_PATH.as_posix())
    atomic_json(candidate_path, final)
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publish", "after_atomic", path=RESULT_PATH.as_posix())
    return final


def date_argument(value: str) -> str:
    """Accept only the run date frozen by the V661 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse measured execution and the two read-only replay modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--date", type=date_argument)
    modes.add_argument("--cold-validate", type=Path)
    modes.add_argument("--independent-reduce", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - capability E2E.
    """Run Exp7560 or validate its candidate without new model work."""

    print("[exp7560] phase=startup event=flushed", flush=True)
    args = parse_args(argv)
    if args.cold_validate is not None:
        errors = validate_artifact(load_json(args.cold_validate), require_terminal=False)
        print(json.dumps({"errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    if args.independent_reduce is not None:
        value = load_json(args.independent_reduce)
        reduction = independent_reduce(value, require_terminal=False)
        print(json.dumps(reduction, sort_keys=True), flush=True)
        return int(reduction.get("passed") is not True)
    artifact = run_experiment(REPO_ROOT, args.date)
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

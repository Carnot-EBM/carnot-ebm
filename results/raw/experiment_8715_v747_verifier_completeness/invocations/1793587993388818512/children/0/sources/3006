"""Bind the V657 contract to exact V656 custody and source-grounded methods.

This advisory report performs no model work and makes no scientific benefit
claim. It preserves missing producer states instead of turning raw completion
or conductor outcomes into scientific verdicts.

Spec refs: REQ-REPORT-7503 and SCENARIO-REPORT-7503-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from carnot.experiment_7489_v656_contract_methods import (
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
MILESTONE = "2026.09.657"
EXPERIMENT_ID = "exp7503-contract-methods"
SCHEMA = "carnot.exp7503.v657.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7503_v657_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7503_v657_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7503_v657_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7503_v657_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7503_v657_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v657-method-map.md")
STUDY_PATH = Path("research-studying.md")
CAPSTONE_PATH = Path("results/experiment_7502_v656_capstone.json")
CONDUCTOR_LOG_PATH = Path("ops/conductor-log.md")

EXPECTED_TASK_IDS = (
    "exp7503-contract-methods",
    "exp7504-evidence-interface",
    "exp7505-energy-fit",
    "exp7506-causal-prototype",
    "exp7507-static-evaluation",
    "exp7508-static-audit",
    "exp7509-causal-online",
    "exp7510-causal-audit",
    "exp7511-arc-evidence-recovery",
    "exp7512-arc-opportunity",
    "exp7513-placement-continuity",
    "exp7514-service-trace",
    "exp7515-capstone",
)

ZERO_INVOCATION_COUNTS = {
    f"{operation}_{state}": 0
    for operation in ("model_loads", "forward_calls", "generation_calls")
    for state in ("attempted", "completed", "failed", "cancelled", "in_flight")
}

V656_PATHS = (
    ("exp7489-contract-methods", "results/experiment_7489_v656_contract_methods.json"),
    ("exp7490-historical-audit", "results/experiment_7490_v656_historical_audit.json"),
    ("exp7491-window-protocol", "results/experiment_7491_v656_window_protocol.json"),
    ("exp7492-window-pilot", "results/experiment_7492_v656_window_pilot.json"),
    ("exp7493-window-fit-capture", "results/experiment_7493_v656_window_fit_capture.json"),
    ("exp7494-window-eval-capture", "results/experiment_7494_v656_window_eval_capture.json"),
    ("exp7495-window-calibration", "results/experiment_7495_v656_window_calibration.json"),
    ("exp7496-causal-update-fixture", "results/experiment_7496_v656_causal_update_fixture.json"),
    ("exp7497-causal-online-learning", "results/experiment_7497_v656_causal_online_learning.json"),
    ("exp7498-independent-audit", "results/experiment_7498_v656_independent_audit.json"),
    ("exp7499-arc-panel-b", "results/experiment_7499_v656_arc_panel_b.json"),
    ("exp7500-arc-opportunity-audit", "results/experiment_7500_v656_arc_opportunity_audit.json"),
    ("exp7501-service-placement", "results/experiment_7501_v656_service_placement.json"),
    ("exp7502-capstone", "results/experiment_7502_v656_capstone.json"),
)

CONDUCTOR_STATUSES = {
    "exp7495-window-calibration": "FAIL",
    "exp7496-causal-update-fixture": "FAIL",
    "exp7497-causal-online-learning": "GATE_BLOCK",
    "exp7499-arc-panel-b": "FAIL",
    "exp7501-service-placement": "GATE_BLOCK",
}

PANEL_B_RAW_PATHS = {
    "session": Path("results/raw/experiment_7499_v656_arc_panel_b/live_session.json"),
    "checkpoint": Path("results/checkpoints/experiment_7499_v656_arc_panel_b.json"),
    "candidate": Path(
        "results/raw/experiment_7499_v656_arc_panel_b/measured_terminal_candidate.json"
    ),
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
    Path("python/carnot/experiment_7489_v656_contract_methods.py"),
    Path("python/carnot/experiment_7502_v656_capstone.py"),
    CAPSTONE_PATH,
    Path("ops/known-issues.md"),
    Path("ops/north-star.md"),
    Path("research-references.md"),
    STUDY_PATH,
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/arc_levelup_guarantee_lint.py"),
    DESIGN_PATH,
    SPEC_PATH,
    CONDUCTOR_LOG_PATH,
    *PANEL_B_RAW_PATHS.values(),
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)
REQUIRED_REPOSITORY_CHECK_NAMES = (
    "roadmap_schema",
    "roadmap_gate_audit",
    "exclusion_manifest",
    "arc_floor",
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)

METHOD_SPECS: tuple[JsonDict, ...] = (
    {
        "method": "binary_scoring_token_expectation",
        "primary_url": "https://arxiv.org/html/2607.05391",
        "source_revision": "arXiv:2607.05391v2 (2026-07-07)",
        "source_section": "Section 3.2, equation 3.1",
        "adaptation": "Use the expectation over the two existing semantic option logits.",
        "counterexample": "A sharp binary score can still be miscalibrated against source labels.",
        "task_mapping": ["exp7504-evidence-interface", "exp7507-static-evaluation"],
    },
    {
        "method": "equal_detector_access",
        "primary_url": "https://arxiv.org/html/2606.06959",
        "source_revision": "arXiv:2606.06959v1 (2026-06-08)",
        "source_section": "Sections 2.1-2.2 and detector implementation appendix",
        "adaptation": "Ledger equal source, window, option-order, label, and metric access.",
        "counterexample": "Equal prompts do not help if one detector receives hidden labels.",
        "task_mapping": ["exp7504-evidence-interface", "exp7508-static-audit"],
    },
    {
        "method": "aligned_vs_shuffled_information",
        "primary_url": "https://arxiv.org/html/2606.26476",
        "source_revision": "arXiv:2606.26476v1 (2026-06-25)",
        "source_section": "Section 4.4, aligned-versus-shuffled five-arm suite",
        "adaptation": "Shuffle only labels available within the same release batch.",
        "counterexample": "A cross-batch shuffle can import future labels and fake causality.",
        "task_mapping": ["exp7506-causal-prototype", "exp7509-causal-online"],
    },
    {
        "method": "kan_retention",
        "primary_url": "https://arxiv.org/html/2511.12828",
        "source_revision": "arXiv:2511.12828v1 (2025-11-17)",
        "source_section": "Bounded Retention, Lemma 1 and Theorem 1",
        "adaptation": "Measure local-support overlap and a frozen held-out retention set.",
        "counterexample": "Local splines still forget when later supports overlap earlier supports.",
        "task_mapping": ["exp7506-causal-prototype", "exp7509-causal-online"],
    },
    {
        "method": "whole_service_hardware_accounting",
        "primary_url": "https://arxiv.org/html/2602.15985",
        "source_revision": "arXiv:2602.15985v2 (2026-09-04)",
        "source_section": "Sections 2.3, 4.2, and 5.3.2 end-to-end TTS",
        "adaptation": "Count conversion, state, transfer, update, fsync, and acknowledgement.",
        "counterexample": "A fast kernel can leave the complete service slower after orchestration.",
        "task_mapping": ["exp7513-placement-continuity", "exp7514-service-trace"],
    },
)


def utc_now() -> str:  # pragma: no cover - measured runtime boundary.
    """Return one aware UTC boundary for a durable run receipt."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush a phase boundary so a long child operation stays observable."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7503] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def resolve_v657_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer a staged V657 authority, then use the matching active roadmap."""

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
                "principle": "Milestone matching prevents a stale roadmap from becoming authority.",
            }
        )
        if selected is None and matches:
            selected = (path, value)
    if selected is None:
        raise ValueError("V657 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Retain only fields independently declared by both authorities."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Confirm each gate consumes a field declared by an earlier producer."""

    raw_tasks = roadmap.get("tasks")
    tasks = raw_tasks if isinstance(raw_tasks, list) else []
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
    """Compare all thirteen identities, paths, substrates, and gate triples."""

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
    if len(markdown_tasks) != 13:
        errors.append("markdown_task_count")
    if len(yaml_tasks) != 13:
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
    """Plant one defect in a private Markdown authority copy."""

    if mutation == "milestone":
        return re.sub(r"2026\.09\.657", "2026.09.656", markdown, count=1)
    lines, indices = _table_rows(markdown)
    if mutation == "count":
        del lines[indices[-1]]
    elif mutation == "order":
        lines[indices[0]], lines[indices[1]] = lines[indices[1]], lines[indices[0]]
    elif mutation == "gate_field":
        lines[indices[2]] = lines[indices[2]].replace("evidence_ready_score", "changed_score", 1)
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
    """Plant one defect in a private YAML authority copy."""

    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "milestone":
        changed["milestone"] = "2026.09.656"
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
    """Require seven private defects in each authority to fail comparison."""

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
                else parse_markdown_contract(markdown_text)
            )
            rows.append(
                {
                    "unit_id": f"{authority}:{mutation}",
                    "arm": "contract_mutation",
                    "authority": authority,
                    "mutation": mutation,
                    "rejected": comparison["passed"] is False,
                    "other_authority_readable": (
                        other["milestone"] == MILESTONE and len(other["tasks"]) == 13
                    ),
                    "errors": comparison["errors"],
                    "principle": "A planted defect must fail without changing its peer.",
                }
            )
    return rows


def _capstone_source_rows(capstone: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Index the capstone's retained source rows by task identity."""

    result: dict[str, JsonDict] = {}
    for row in capstone.get("source_artifact_hashes") or []:
        if isinstance(row, Mapping) and isinstance(row.get("task_id"), str):
            result[str(row["task_id"])] = dict(row)
    return result


def _raw_panel_b_custody(root: Path) -> tuple[dict[str, JsonDict], int, float, bool]:
    """Hash the raw session, checkpoint, and candidate as distinct evidence."""

    hashes = {
        role: {
            "path": path.as_posix(),
            "sha256": sha256_file(root / path),
            "bytes": (root / path).stat().st_size,
            "scope": "raw_unpromoted_evidence",
        }
        for role, path in PANEL_B_RAW_PATHS.items()
    }
    session = load_json(root / PANEL_B_RAW_PATHS["session"])
    episodes = session.get("episodes")
    completed = len(episodes) if isinstance(episodes, list) else 0
    duration = float(session.get("duration_s", 0.0))
    clean_child = session.get("error") is None and session.get("timed_out") is False
    return hashes, completed, duration, clean_child


_CONDUCTOR_TITLE_FRAGMENTS = {
    "exp7495-window-calibration": "Test calibrated energy decisions against controls",
    "exp7496-causal-update-fixture": "Qualify Brier updates and chronology-preserving",
    "exp7497-causal-online-learning": "Measure whether correctly paired delayed feedback",
    "exp7499-arc-panel-b": "Complete withheld-game live ARC panel B",
    "exp7501-service-placement": "Measure durable feedback service and retain honest",
}


def collect_v656_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate nine terminal paths and five absent producer paths."""

    capstone = load_json(root / CAPSTONE_PATH)
    retained = _capstone_source_rows(capstone)
    conductor_log = (root / CONDUCTOR_LOG_PATH).read_text(encoding="utf-8")
    raw_hashes, raw_episodes, raw_duration, raw_child_clean = _raw_panel_b_custody(root)
    rows: list[JsonDict] = []
    for order, (task_id, path_text) in enumerate(V656_PATHS, start=1):
        path = root / path_text
        present = path.is_file()
        payload = load_json(path) if present else {}
        digest = sha256_file(path) if present else None
        retained_row = retained.get(task_id, {})
        conductor_status = CONDUCTOR_STATUSES.get(task_id, "OK")
        title_fragment = _CONDUCTOR_TITLE_FRAGMENTS.get(task_id)
        status_authenticated = (
            True
            if title_fragment is None
            else any(
                title_fragment in line and f"| {conductor_status} |" in line
                for line in conductor_log.splitlines()
            )
        )
        if task_id == "exp7502-capstone":
            byte_authenticated = (
                present
                and payload.get("experiment_id") == "exp7502-capstone"
                and payload.get("milestone") == "2026.09.656"
            )
        elif present:
            byte_authenticated = (
                retained_row.get("evidence_class") == "terminal"
                and retained_row.get("sha256") == digest
                and payload.get("milestone") == "2026.09.656"
            )
        else:
            byte_authenticated = (
                retained_row.get("evidence_class") == "missing"
                and retained_row.get("sha256") is None
            )
        raw_only = task_id == "exp7499-arc-panel-b"
        raw_authenticated = not raw_only or (
            raw_episodes == 18
            and raw_duration == 3284.8582281540002
            and raw_child_clean
            and len(raw_hashes) == 3
        )
        rows.append(
            {
                "unit_id": task_id,
                "arm": "v656_custody",
                "order": order,
                "task_id": task_id,
                "expected_path": path_text,
                "producer_present": present,
                "source_sha256": digest,
                "custody_state": (
                    "terminal"
                    if present
                    else "raw_only_unpromoted_candidate"
                    if raw_only
                    else "conductor_only_absent"
                ),
                "conductor_status": conductor_status,
                "conductor_status_authenticated": status_authenticated,
                "original_status": payload.get("status") if present else None,
                "original_honest_verdict": payload.get("honest_verdict") if present else None,
                "original_verdict_class": payload.get("verdict_class") if present else None,
                "original_flagged_adversarial": (
                    bool(payload.get("flagged_adversarial", False)) if present else False
                ),
                "completed_raw_episodes": raw_episodes if raw_only else None,
                "raw_duration_s": raw_duration if raw_only else None,
                "raw_child_clean": raw_child_clean if raw_only else None,
                "raw_custody_hashes": deepcopy(raw_hashes) if raw_only else {},
                "scientific_result_established": present,
                "candidate_promoted": False if raw_only else None,
                "authenticated": bool(
                    byte_authenticated and status_authenticated and raw_authenticated
                ),
                "failures": []
                if byte_authenticated and status_authenticated and raw_authenticated
                else ["custody_authentication_failed"],
                "principle": (
                    "Terminal, conductor-only, and raw-only evidence must remain different states."
                ),
            }
        )
    return rows


def method_rows() -> list[JsonDict]:
    """Return five source-grounded adaptations without importing paper claims."""

    return [
        {
            "unit_id": row["method"],
            "arm": "method_ingestion",
            **deepcopy(row),
            "access_status": "primary_html_read_20260922",
            "external_claim_is_carnot_measurement": False,
            "failures": [],
            "principle": "A source method can guide a control without becoming a local result.",
        }
        for row in METHOD_SPECS
    ]


def write_method_records(root: Path) -> None:
    """Write one bounded note and append one idempotent study marker."""

    methods = method_rows()
    lines = [
        "# V657 method map",
        "",
        "Date: 2026-09-22. Scope: advisory method accounting.",
        "Paper results are not local Carnot results.",
        "",
        "| Method | Primary source and section | Adaptation | Counterexample | Named task |",
        "|---|---|---|---|---|",
    ]
    for row in methods:
        source = f"[{row['source_revision']}]({row['primary_url']}), {row['source_section']}"
        lines.append(
            f"| {row['method']} | {source} | {row['adaptation']} | "
            f"{row['counterexample']} | {', '.join(row['task_mapping'])} |"
        )
    lines.extend(
        [
            "",
            "The primary HTML sections were read with bounded sequential requests.",
            "No paper benchmark number is treated as a local measurement.",
            "",
        ]
    )
    note_path = root / NOTE_PATH
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines), encoding="utf-8")

    marker = "<!-- EXP7503-V657-METHOD-INGESTION -->"
    study_path = root / STUDY_PATH
    studying = study_path.read_text(encoding="utf-8")
    if marker not in studying:
        addition = "\n".join(
            [
                "",
                marker,
                "## 2026-09-22 Exp7503 — V657 methods — INGESTED",
                "",
                "Binary scoring-token expectation, equal detector access, legal aligned-versus-",
                "shuffled information, KAN retention, and whole-service hardware accounting",
                "map to V657. Paper results are not local Carnot results. Exact sections,",
                "counterexamples, and task mappings are in",
                "`docs/research-notes/v657-method-map.md`.",
                "",
            ]
        )
        study_path.write_text(studying.rstrip() + "\n" + addition, encoding="utf-8")


def priority_disposition(root: Path, roadmap_path: Path) -> JsonDict:
    """Record every real overdue-priority violation without hiding its exit."""

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
        "unresolved_count": len(violations),
        "unresolved_priorities": violations,
        "affects_contract_authority_match": False,
        "required_current_validation": False,
        "principle": "A real unresolved priority remains visible without rewriting the active roadmap.",
    }


def _precondition(check: str, path: str, field: str, expected: Any, observed: Any) -> JsonDict:
    """Record one exact prerequisite value and its read-only ownership."""

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
    """Authenticate listed inputs, the driving requirement, and custody counts."""

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
                "REQ-REPORT-7503",
                "REQ-REPORT-7503" if "REQ-REPORT-7503" in spec else None,
            ),
            _precondition(
                "roadmap_identity",
                roadmap_path.relative_to(root).as_posix(),
                "milestone",
                MILESTONE,
                roadmap.get("milestone"),
            ),
            _precondition(
                "v656_disposition_count",
                CAPSTONE_PATH.as_posix(),
                "dispositions",
                14,
                len(dispositions),
            ),
            _precondition(
                "v656_terminal_count",
                CAPSTONE_PATH.as_posix(),
                "producer_present",
                9,
                sum(row.get("producer_present") is True for row in dispositions),
            ),
            _precondition(
                "v656_absent_count",
                CAPSTONE_PATH.as_posix(),
                "producer_absent",
                5,
                sum(row.get("producer_present") is False for row in dispositions),
            ),
            _precondition(
                "silent_authoring_diagnosis",
                "ops/known-issues.md",
                "dated_elapsed_seconds",
                "2026-09-21:937",
                "2026-09-21:937"
                if "stream went silent for 937 s" in (root / "ops/known-issues.md").read_text()
                else None,
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
    """Bind protocol, implementation, terminal, and raw evidence bytes."""

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
                "counted_as_current_invocation": False,
            }
    return hashes


def build_repository_check_plan(
    root: Path, roadmap_path: Path
) -> list[validation_scope.CommandSpec]:
    """Build unchanged schema, gate, exclusion, ARC, and priority checks."""

    python = str(root / ".venv/bin/python")
    selected = str(roadmap_path)
    schema_code = (
        "import pathlib,sys,yaml;"
        "from scripts.roadmap_schema import Roadmap;"
        "p=pathlib.Path(sys.argv[1]);"
        "Roadmap.model_validate(yaml.safe_load(p.read_text()));"
        "print('roadmap schema clean',flush=True)"
    )
    overdue_code = (
        "import pathlib,sys;"
        "import scripts.overdue_priority_lint as lint;"
        "lint.ROADMAP_NEXT=pathlib.Path(sys.argv[1]);"
        "sys.argv=sys.argv[:1];"
        "raise SystemExit(lint.main())"
    )
    return [
        validation_scope.CommandSpec(
            "roadmap_schema", (python, "-u", "-c", schema_code, selected), "selected_roadmap"
        ),
        validation_scope.CommandSpec(
            "roadmap_gate_audit",
            (python, "-u", "scripts/audit_roadmap_gates.py", selected),
            "selected_roadmap",
        ),
        validation_scope.CommandSpec(
            "exclusion_manifest",
            (python, "-u", "scripts/exclusion_manifest_lint.py", selected),
            "selected_roadmap",
        ),
        validation_scope.CommandSpec(
            "arc_floor",
            (python, "-u", "scripts/arc_levelup_guarantee_lint.py", selected),
            "selected_roadmap",
        ),
        validation_scope.CommandSpec(
            "overdue_priority", (python, "-u", "-c", overdue_code, selected), "repository_health"
        ),
    ]


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the shared scoped validation plan for the affected manifest."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _gate(
    check: str,
    gate_type: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Attach exact operands and the prevented failure to one check."""

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
    """Keep external absence, invalid work, and a valid advisory null distinct."""

    if not preconditions_passed:
        return (
            "complete_blocked_external_v657_contract_input",
            "complete_blocked_external_v657_contract_input",
            "blocked",
        )
    if not validation_passed or not contract_ready:
        return (
            "complete_disqualified_v657_contract_or_validation",
            "complete_disqualified_v657_contract_or_validation",
            "disqualified",
        )
    return (
        "complete_advisory_v657_contract_and_methods",
        "complete_null_v657_contract_methods_ingested",
        "null",
    )


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    priority: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep validity, readiness, method intake, and benefit independent."""

    return [
        _gate(
            "preconditions_authenticated",
            "validity",
            True,
            all(row.get("passed") is True for row in preconditions),
            all(row.get("passed") is True for row in preconditions),
            "Favorable rows cannot excuse missing prerequisite bytes.",
        ),
        _gate(
            "exact_v657_contract",
            "validity",
            True,
            contract.get("passed") is True,
            contract.get("passed") is True,
            "Paired authorities prevent task identity and gate drift.",
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
            "v656_custody_authenticated",
            "validity",
            14,
            sum(row.get("authenticated") is True for row in dispositions),
            len(dispositions) == 14
            and all(row.get("authenticated") is True for row in dispositions),
            "Nine terminal and five absent paths prevent completion inflation.",
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
            "Schema, prior-failure, exclusion, gate, and ARC guards retain policy.",
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
            5,
            len(methods),
            len(methods) == 5
            and all(row.get("external_claim_is_carnot_measurement") is False for row in methods),
            "Paper results cannot become local measurements through ingestion.",
        ),
        _gate(
            "overdue_priority_observed",
            "support",
            True,
            priority.get("unresolved_count") == len(priority.get("unresolved_priorities") or []),
            priority.get("unresolved_count") == len(priority.get("unresolved_priorities") or []),
            "Repository health remains visible without mutating the active roadmap.",
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
    """Explain how each top-level field prevents a reporting failure."""

    required = {
        "schema": "Version, experiment identity, milestone, and terminal status prevent reader drift.",
        "run_date": "The frozen date and measured clocks distinguish this run from cited history.",
        "preconditions_checked": "Exact paths, bytes, values, and ownership prevent invented inputs.",
        "MODEL_SPECS": "An empty list states that this aggregation loaded no current model.",
        "model_specs": "The lowercase mirror prevents schema readers from inferring a model.",
        "model_invoked": "Current calls remain separate from historical model provenance.",
        "invocation_counts": "Every attempted, terminal, cancelled, and in-flight call reconciles to zero.",
        "inference_substrate": "The canonical aggregation spelling prevents substrate ambiguity.",
        "inference_substrate_class": "The aggregation class prevents invented duration floors.",
        "execution_venue": "Current host reduction stays distinct from archived board evidence.",
        "duration_s": "Measured elapsed work separates authoring, validation, and historical capture.",
        "phase_spans": "Flushed boundaries and checkpoints expose unfinished or silent work.",
        "random_seed": "Frozen role seeds do not multiply the sample size of deterministic accounting.",
        "reproducibility_checksum": "The checksum binds source, settings, custody, and validation scope.",
        "source_artifact_hashes": "Exact hashes preserve original bytes and raw exposure history.",
        "rows": "Per-unit failures and censoring permit independent reduction.",
        "sample_size_budget": "Planned, attempted, complete, failed, censored, and unstarted stay distinct.",
        "acceptance_gate_results": "Each gate records operands, operator, outcome, and prevented failure.",
        "gate_check_summary": "A blocked or invalid result names each exact failed check.",
        "honest_verdict": "A complete terminal prefix prevents retry and retirement drift.",
        "verdict_class": "The closed class keeps external absence blocked and invalid work disqualified.",
        "verifier_is_oracle": "No oracle fixture supports a positive claim in this advisory report.",
        "flagged_adversarial": "Actual guard findings cannot be cleared to open a gate.",
        "validation_receipts": "Commands, exits, hashes, scope, and baseline status remain auditable.",
        "field_principles": "Every emitted field states the failure it prevents.",
        "contract_ready_score": "Bare contract readiness covers authority and custody, not science.",
        "method_ingestion_complete_score": "Bare method completion is independent of benefit.",
        "task_contract_rows": "Exactly thirteen ordered task identities prevent authority divergence.",
        "prior_dispositions": "Missing producers and raw-only work remain different custody states.",
        "method_rows": "Sources, adaptations, counterexamples, and named tasks bound paper use.",
    }
    return {
        field: required.get(
            field, f"The {field} field preserves exact scope and prevents reinterpretation."
        )
        for field in fields
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name each failed gate with its exact operands and current owner."""

    failed = [
        {
            "check": row.get("check"),
            "upstream": "current_exp7503_reduction",
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


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete artifact except the checksum's own value."""

    payload = deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide named zero-work spans for deterministic pure reducer tests."""

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
    receipts = deepcopy(list(validation.get("validation_receipts") or []))
    gates = _acceptance_gates(
        contract_copy, mutations, prior, methods, preconditions, validation, priority
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
    ]
    duration_s = max(0.0, (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000)
    validation_duration = sum(float(row.get("duration_s", 0.0)) for row in receipts)
    panel_b = next(row for row in prior if row.get("task_id") == "exp7499-arc-panel-b")
    absent = sum(row.get("producer_present") is False for row in prior)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7503,
        "title": "Bind thirteen tasks and ingest methods against terminal V656 evidence",
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
            "historical_capture": float(panel_b.get("raw_duration_s") or 0.0),
            "total_current": duration_s,
        },
        "small_ebm_training": {"performed": False, "parameter_count": 0},
        "random_seed": {
            "fitting": None,
            "arrival": None,
            "audit": 7_503_657_03,
            "bootstrap": None,
            "contract_mutations": 7_503_657_01,
            "order": 7_503_657_02,
            "explanation": "This deterministic aggregation makes no stochastic estimate.",
        },
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "roadmap_resolution_candidates": deepcopy(list(roadmap_candidates)),
        "contract_comparison": contract_copy,
        "task_contract_rows": deepcopy(contract_copy.get("contract_rows") or []),
        "contract_mutation_rows": mutations,
        "prior_dispositions": prior,
        "method_rows": deepcopy(methods),
        "priority_disposition": deepcopy(priority),
        "source_artifact_hashes": _source_hashes(root, roadmap_path, prior),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows) - absent,
            "complete": len(rows) - absent,
            "completed": len(rows) - absent,
            "failed": 2,
            "excluded": absent,
            "censored": 1,
            "unstarted": 4,
            "independent_unit": "contract_mutation_disposition_method_or_priority_row",
            "counting_rule": "Failure and censoring describe custody and can overlap exclusion.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "contract_ready_score": int(contract_ready),
        "method_ingestion_complete_score": int(
            len(methods) == 5
            and all(
                row.get("source_section")
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
            "full_python_suite_run": False,
            "numbered_runtime_e2e_applicable": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": True,
            "numbered_runtime_e2e_applicable": False,
        },
        "advisory_only": True,
        "later_science_tasks_gate_on_exp7503": False,
        "roadmap_activated": False,
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
        and len(dispositions) == 14
        and all(
            isinstance(row, Mapping) and row.get("authenticated") is True for row in dispositions
        )
    )
    methods_ok = (
        isinstance(methods, list)
        and len(methods) == 5
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
    affected_ok = receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    repository_ok = receipts_pass(receipts, REQUIRED_REPOSITORY_CHECK_NAMES)
    terminal_ok = receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    validation_ok = affected_ok and repository_ok and terminal_ok
    ready = contract_ok and mutations_ok and dispositions_ok and preconditions_ok and validation_ok
    status, honest, verdict = terminal_state(preconditions_ok, validation_ok, ready)
    return {
        "contract_ready_score": int(ready),
        "method_ingestion_complete_score": int(methods_ok),
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
    }


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash all declared sources from the selected current worktree."""

    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return False
    try:
        roadmap_path, _roadmap, _candidates = resolve_v657_roadmap(root)
        expected = _source_hashes(root, roadmap_path, collect_v656_dispositions(root))
    except (OSError, ValueError, json.JSONDecodeError):
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
    roadmap_path, roadmap, _candidates = resolve_v657_roadmap(root)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    contract = compare_contract_authorities(markdown, roadmap)
    mutations = run_contract_mutation_controls(markdown, roadmap)
    dispositions = collect_v656_dispositions(root)
    priority = priority_disposition(root, roadmap_path)
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
    if artifact.get("priority_disposition") != priority:
        errors.append("priority_disposition_mismatch")
    expected_rows = [
        *deepcopy(contract.get("contract_rows") or []),
        *deepcopy(mutations),
        *deepcopy(dispositions),
        *method_rows(),
        deepcopy(priority),
    ]
    if artifact.get("rows") != expected_rows:
        errors.append("rows_mismatch")
    receipts = artifact.get("validation_receipts")
    if not receipts_recorded(receipts, validation_scope.REQUIRED_CHECK_NAMES):
        errors.append("affected_validation_invalid")
    if not receipts_recorded(receipts, REQUIRED_REPOSITORY_CHECK_NAMES):
        errors.append("repository_validation_invalid")
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


def _phase_span(  # pragma: no cover - measured runtime boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    units: int,
    checkpoint: str,
) -> JsonDict:
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


def _terminal_commands(  # pragma: no cover - capability E2E child plan.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build entrypoint replay, cold reduction, and strict reader checks."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7503_v657_contract_methods import validate_artifact;"
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


def _external_block_artifact(  # pragma: no cover - external absence boundary.
    missing: Path, started_at: str, started_ns: int
) -> JsonDict:
    """Emit the exact missing prerequisite without inventing dependent rows."""

    ended_ns = time.monotonic_ns()
    gate = _gate(
        "external_prerequisite_available",
        "validity",
        "readable_nonempty_bytes",
        None,
        False,
        "A missing external input blocks dependent reduction.",
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7503,
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": f"complete_blocked_missing_{missing.name}",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": ended_ns,
        "process_identity": {"pid": os.getpid(), "owner": "current_process"},
        "duration_s": max(0.0, (ended_ns - started_ns) / 1_000_000_000),
        "phase_spans": [],
        "preconditions_checked": [
            _precondition(
                "external_prerequisite_available",
                missing.as_posix(),
                "bytes",
                "readable_nonempty_bytes",
                None,
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
        "gate_check_summary": _gate_summary([gate]),
        "honest_verdict": f"complete_blocked_missing_{missing.name}",
        "verdict_class": "blocked",
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": [],
        "field_principles": {},
        "contract_ready_score": 0,
        "method_ingestion_complete_score": 0,
        "task_contract_rows": [],
        "prior_dispositions": [],
        "method_rows": [],
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def run_experiment(  # pragma: no cover - exercised through capability E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Run scoped checks and atomically publish only validated terminal bytes."""

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
    missing = next(
        (path for path in SOURCE_INPUT_PATHS if not (root / path).is_file()),
        None,
    )
    if missing is not None:
        blocked = _external_block_artifact(missing, started_at, started_ns)
        atomic_json(root / RESULT_PATH, blocked)
        progress(started, "preconditions", "after_blocked", path=missing.as_posix())
        return blocked
    roadmap_path, roadmap, roadmap_candidates = resolve_v657_roadmap(root)
    dispositions = collect_v656_dispositions(root)
    preconditions = collect_preconditions(root, roadmap_path, dispositions)
    if not all(row["passed"] for row in preconditions):
        failed = next(row for row in preconditions if not row["passed"])
        blocked = _external_block_artifact(Path(str(failed["path"])), started_at, started_ns)
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
            checkpoint="contract_and_custody_rows",
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

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7503-validation-", dir="/tmp"))
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
    progress(started, "validation", "before_subprocesses", planned_units=len(affected_plan) + 5)
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in affected_plan],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    affected_reduction = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    repository_plan = build_repository_check_plan(root, roadmap_path)
    repository = run_categorized_commands(
        root,
        [
            PlannedCommand(
                command,
                "required_validation"
                if command.name in REQUIRED_REPOSITORY_CHECK_NAMES
                else "repository_health",
                command.name in REQUIRED_REPOSITORY_CHECK_NAMES,
            )
            for command in repository_plan
        ],
        log_dir=raw_dir / "validation/repository",
        heartbeat_s=60.0,
    )
    repository_passed = receipts_pass(repository, REQUIRED_REPOSITORY_CHECK_NAMES)
    spans.append(
        _phase_span(
            "validation",
            phase_started,
            started,
            units=len(affected) + len(repository),
            checkpoint="scoped_and_repository_checks",
        )
    )
    progress(
        started,
        "validation",
        "after_subprocesses",
        affected=affected_reduction["passed"],
        repository=repository_passed,
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
        started, "terminal_validation", "before_subprocesses", planned_units=len(terminal_plan)
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
            checkpoint="cold_replay_and_strict_readers",
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
    """Accept only the date frozen by the V657 task contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the measured run and read-only cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date_argument)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(  # pragma: no cover - exercised by capability E2E.
    argv: Sequence[str] | None = None,
) -> int:
    """Run V657 contract ingestion or validate one measured candidate."""

    print("[exp7503] phase=startup event=flushed", flush=True)
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

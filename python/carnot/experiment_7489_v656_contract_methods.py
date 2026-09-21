"""Bind the V656 task contract to the terminal V655 evidence.

This report is advisory. It authenticates planning and historical bytes, but it
does not make a later science task depend on this report. It performs no model
load, forward, generation, training, board operation, or roadmap activation.

Spec refs: REQ-REPORT-7489 and SCENARIO-REPORT-7489-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
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
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json
from scripts import overdue_priority_lint


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.656"
EXPERIMENT_ID = "exp7489-contract-methods"
SCHEMA = "carnot.exp7489.v656.contract_methods.v1"

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7489_v656_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7489_v656_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7489_v656_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7489_v656_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7489_v656_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v656-method-ingestion.md")
STUDY_PATH = Path("research-studying.md")
CAPSTONE_PATH = Path("results/experiment_7488_v655_capstone.json")
EXP7435_PATH = Path("results/experiment_7435_v652_round_breaker.json")

EXPECTED_TASK_IDS = (
    "exp7489-contract-methods",
    "exp7490-historical-audit",
    "exp7491-window-protocol",
    "exp7492-window-pilot",
    "exp7493-window-fit-capture",
    "exp7494-window-eval-capture",
    "exp7495-window-calibration",
    "exp7496-causal-update-fixture",
    "exp7497-causal-online-learning",
    "exp7498-independent-audit",
    "exp7499-arc-panel-b",
    "exp7500-arc-opportunity-audit",
    "exp7501-service-placement",
    "exp7502-capstone",
)

V655_TASK_IDS = (
    "exp7475-contract-methods",
    "exp7476-option-qualification",
    "exp7477-native-readout-pilot",
    "exp7478-arc-interval-protocol",
    "exp7479-source-fit-capture",
    "exp7480-source-eval-capture",
    "exp7481-typed-calibration",
    "exp7482-importance-anchor",
    "exp7483-continuous-learning",
    "exp7484-decision-audit",
    "exp7485-arc-cost-panel-a",
    "exp7486-arc-cost-panel-b",
    "exp7487-learning-placement",
    "exp7488-capstone",
)

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "forward_calls_attempted": 0,
    "forward_calls_completed": 0,
    "forward_calls_failed": 0,
    "forward_calls_cancelled": 0,
    "forward_calls_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

RANDOM_SEED = {
    "role": 656_001,
    "optimizer": None,
    "audit": 7_489_656_03,
    "order": 7_489_656_02,
    "interval": None,
    "contract_mutations": 7_489_656_01,
    "deterministic_null_seed_explanation": (
        "No stochastic scientific estimate is made; null seeds mark inapplicable roles."
    ),
}

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
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/arc_levelup_guarantee_lint.py"),
    Path("scripts/overdue_priority_lint.py"),
    Path("scripts/retro_timing_fallback.py"),
    Path("scripts/autoresearch_conductor_round.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("research-references.md"),
    STUDY_PATH,
    DESIGN_PATH,
    SPEC_PATH,
    CAPSTONE_PATH,
    EXP7435_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REPOSITORY_CHECK_NAMES = (
    "roadmap_schema",
    "roadmap_gate_audit",
    "exclusion_manifest",
    "arc_floor",
    "overdue_priority",
)

TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return one measured UTC boundary for the public run receipt."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush a phase boundary so long validation never appears inactive."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7489] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a changed upstream file cannot inherit an old claim."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash stable JSON bytes so cold replay detects any evidence change."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_yaml(path: Path) -> JsonDict:
    """Load one YAML mapping and reject lists or malformed authority bytes."""

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML mapping required: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"YAML mapping required: {path}")
    return value


def load_json(path: Path) -> JsonDict:
    """Load one JSON mapping without normalizing historical evidence."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON mapping required: {path}")
    return value


def resolve_v656_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer a matching staged V656 roadmap, then the matching active roadmap."""

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
                "principle": "Milestone selection prevents a stale roadmap from becoming authority.",
            }
        )
        if selected is None and matches:
            selected = (path, value)
    if selected is None:
        raise ValueError("V656 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def _parse_v656_markdown(text: str) -> JsonDict:
    """Normalize only the V656 table labels before the shared parser runs."""

    normalized = re.sub(
        r"(?m)^\| Order \| Task ID \| Title \| Phase \| Deliverable \| Substrate class \| Structured gates \|$",
        "| Order | Task ID | Exact title | Phase | Deliverable | Substrate class | Structured gate |",
        text,
        count=1,
    )
    return parse_markdown_contract(normalized)


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Retain only fields declared independently by both contract authorities."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Check that every consumed gate field is declared by an earlier producer."""

    raw_tasks = roadmap.get("tasks")
    tasks = raw_tasks if isinstance(raw_tasks, list) else []
    positions = {
        task.get("id"): (index, task)
        for index, task in enumerate(tasks)
        if isinstance(task, Mapping)
    }
    result: dict[str, list[JsonDict]] = {}
    for consumer_index, consumer in enumerate(tasks):
        if not isinstance(consumer, Mapping):  # pragma: no cover - parser defense.
            continue
        declarations: list[JsonDict] = []
        for gate in consumer.get("gated_on") or []:
            upstream = gate.get("upstream")
            producer_entry = positions.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            field = gate.get("artifact_field")
            declared = _field_declared(str(producer.get("prompt", "")), field)
            precedes = producer_index is not None and producer_index < consumer_index
            declarations.append(
                {
                    "upstream": upstream,
                    "artifact_field": field,
                    "producer_precedes_consumer": precedes,
                    "declared_verbatim": declared,
                    "passed": precedes and declared,
                    "principle": (
                        "A gate cannot depend on an undeclared field or a later producer."
                    ),
                }
            )
        result[str(consumer.get("id"))] = declarations
    return result


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare order, identity, paths, substrates, and complete gate triples."""

    try:
        markdown = _parse_v656_markdown(markdown_text)
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
                "principle": (
                    "Exact paired fields prevent a readable but different task from passing."
                ),
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


def _table_rows(markdown: str) -> tuple[list[str], list[int]]:
    """Locate only numbered rows in the exact task contract table."""

    lines = markdown.splitlines()
    indices = [
        index for index, line in enumerate(lines) if re.match(r"^\|\s*\d+\s*\|\s*exp\d+", line)
    ]
    return lines, indices


def _mutate_markdown(markdown: str, mutation: str) -> str:
    """Change one private Markdown authority field without touching the source."""

    if mutation == "milestone":
        return re.sub(
            r"(?m)^\*\*Milestone:\*\*\s*`?2026\.09\.656`?\s*$",
            "**Milestone:** 2026.09.655",
            markdown,
            count=1,
        )
    lines, indices = _table_rows(markdown)
    if mutation == "count":
        del lines[indices[-1]]
    elif mutation == "order":
        lines[indices[0]], lines[indices[1]] = lines[indices[1]], lines[indices[0]]
    elif mutation == "gate_field":
        gate_index = indices[3]
        lines[gate_index] = lines[gate_index].replace(
            "window_protocol_ready_score", "changed_ready_score", 1
        )
    else:
        cells = [cell.strip() for cell in lines[indices[0]].strip().strip("|").split("|")]
        position = {"id": 1, "path": 4}[mutation]
        cells[position] = {"id": "exp9999-changed", "path": "results/changed.json"}[mutation]
        lines[indices[0]] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines) + "\n"


def _mutate_yaml(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Change one private YAML authority field without touching the source."""

    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "milestone":
        changed["milestone"] = "2026.09.655"
    elif mutation == "count":
        changed["tasks"] = tasks[:-1]
    elif mutation == "order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "id":
        tasks[0]["id"] = "exp9999-changed"
    elif mutation == "path":
        tasks[0]["deliverable"] = "results/changed.json"
    elif mutation == "gate_field":
        tasks[3]["gated_on"][0]["artifact_field"] = "changed_ready_score"
    return changed


def run_contract_mutation_controls(
    markdown_text: str, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Reject six distinct changes in each private authority copy."""

    mutations = ("count", "id", "order", "path", "gate_field", "milestone")
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
                else _parse_v656_markdown(markdown_text)
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
                    "principle": (
                        "A planted contract defect must fail without corrupting its peer."
                    ),
                }
            )
    return rows


def _failed_receipts(source: Mapping[str, Any]) -> list[JsonDict]:
    """Preserve every failed required receipt from its original producer."""

    return [
        {
            "name": receipt.get("name"),
            "exit_code": receipt.get("exit_code"),
            "passed": receipt.get("passed"),
            "log_sha256": receipt.get("log_sha256"),
        }
        for receipt in source.get("validation_receipts") or []
        if isinstance(receipt, Mapping)
        and (receipt.get("passed") is False or receipt.get("exit_code") not in {0, None})
    ]


def collect_v655_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate thirteen producer files and the explicitly absent panel B."""

    capstone_path = root / CAPSTONE_PATH
    capstone = load_json(capstone_path)
    source_rows = capstone.get("task_dispositions")
    if not isinstance(source_rows, list) or len(source_rows) != 14:  # pragma: no cover
        raise ValueError("V655 capstone must contain fourteen task_dispositions")
    if [row.get("task_id") for row in source_rows if isinstance(row, Mapping)] != list(
        V655_TASK_IDS
    ):  # pragma: no cover
        raise ValueError("V655 capstone task order changed")

    capstone_hash = sha256_file(capstone_path)
    rows: list[JsonDict] = []
    for capstone_row in source_rows:
        assert isinstance(capstone_row, Mapping)
        task_id = str(capstone_row.get("task_id"))
        expected_path = str(capstone_row.get("expected_path"))
        if task_id == "exp7488-capstone":
            observed_path: str | None = CAPSTONE_PATH.as_posix()
            source_kind = "disqualified_capstone"
        else:
            found = capstone_row.get("found_path")
            observed_path = str(found) if isinstance(found, str) else None
            source_kind = "producer" if observed_path else "absent_producer"
        source_path = root / observed_path if observed_path else None
        producer_present = bool(source_path and source_path.is_file())
        payload = load_json(source_path) if producer_present and source_path is not None else {}
        source_hash = (
            sha256_file(source_path) if producer_present and source_path is not None else None
        )

        if task_id == "exp7488-capstone":
            hash_matches = source_hash == capstone_hash
        elif producer_present:
            hash_matches = source_hash == capstone_row.get("source_sha256")
        else:
            hash_matches = (
                task_id == "exp7486-arc-cost-panel-b" and not (root / expected_path).exists()
            )
        identity_matches = (
            payload.get("experiment_id") == capstone_row.get("raw_experiment_id")
            and payload.get("milestone") == "2026.09.655"
            if producer_present
            else task_id == "exp7486-arc-cost-panel-b"
        )
        status_matches = (
            payload.get("status") == capstone_row.get("raw_status")
            if producer_present
            else capstone_row.get("raw_status") is None
        )
        flag_matches = (
            bool(payload.get("flagged_adversarial", False))
            == bool(capstone_row.get("flagged_adversarial", False))
            if producer_present
            else capstone_row.get("flagged_adversarial") is False
        )
        authenticated = bool(
            capstone_hash and hash_matches and identity_matches and status_matches and flag_matches
        )
        failed_receipts = _failed_receipts(payload)
        rows.append(
            {
                "unit_id": task_id,
                "arm": "v655_disposition_authentication",
                "order": capstone_row.get("order"),
                "task_id": task_id,
                "expected_path": expected_path,
                "observed_path": observed_path,
                "source_kind": source_kind,
                "producer_present": producer_present,
                "source_sha256": source_hash,
                "capstone_sha256": capstone_hash,
                "evidence_state": (
                    "absent"
                    if task_id == "exp7486-arc-cost-panel-b"
                    else capstone_row.get("evidence_state")
                ),
                "original_status": capstone_row.get("raw_status"),
                "original_honest_verdict": capstone_row.get("honest_verdict"),
                "producer_verdict_class": payload.get("verdict_class"),
                "original_verdict_class": capstone_row.get("verdict_class"),
                "flagged_adversarial": bool(capstone_row.get("flagged_adversarial", False)),
                "required_validation_passed": capstone_row.get("required_validation_passed"),
                "failed_validation_receipts": failed_receipts,
                "historical_validation_failures": list(
                    capstone_row.get("validation_failures") or []
                ),
                "original_inference_substrate": payload.get("inference_substrate"),
                "authenticated": authenticated,
                "evidence_scope": "historical_model_receipts",
                "counted_as_current_invocation": False,
                "principle": (
                    "Exact source bytes preserve failed, absent, and disqualified history."
                ),
            }
        )
    return rows


METHOD_SPECS: tuple[JsonDict, ...] = (
    {
        "method": "response_granularity",
        "source_id": "hall_detect",
        "primary_url": "https://arxiv.org/html/2608.05823v1",
        "source_revision": "arXiv:2608.05823v1 (2026-08-06)",
        "reusable_component": (
            "Compare whole-response and lossless response-window features with the same source."
        ),
        "limitation": (
            "This does not reproduce HallDetect's extractor or encoder, and its ablation is confounded."
        ),
        "task_mapping": ["exp7491-window-protocol", "exp7495-window-calibration"],
    },
    {
        "method": "evidence_alignment",
        "source_id": "input_side_alignment",
        "primary_url": "https://arxiv.org/abs/2608.15804",
        "source_revision": "arXiv:2608.15804 (2026-08-16)",
        "reusable_component": (
            "Preserve response offsets, full source evidence, and a trace to evaluated text."
        ),
        "limitation": "The separate masked-token encoder is not trained or adopted.",
        "task_mapping": ["exp7491-window-protocol", "exp7498-independent-audit"],
    },
    {
        "method": "corpus_leakage_controls",
        "source_id": "parallax",
        "primary_url": "https://arxiv.org/abs/2605.17028",
        "source_revision": "arXiv:2605.17028 (2026-05-16)",
        "reusable_component": (
            "Audit label polarity, source access, annotation leakage, and corpus roles before fitting."
        ),
        "limitation": (
            "Corpus shortcut findings do not establish Carnot detection or probability quality."
        ),
        "task_mapping": ["exp7491-window-protocol", "exp7498-independent-audit"],
    },
    {
        "method": "budgeted_feedback",
        "source_id": "limited_feedback",
        "primary_url": "https://arxiv.org/abs/2609.05820",
        "source_revision": "arXiv:2609.05820 (2026-09-05)",
        "reusable_component": (
            "Freeze feedback opportunities and delay; compare paired labels to shuffled and intercept controls."
        ),
        "limitation": "The component experiment inherits no routing-regret theorem.",
        "task_mapping": [
            "exp7496-causal-update-fixture",
            "exp7497-causal-online-learning",
        ],
    },
    {
        "method": "kan_support_limits",
        "source_id": "kan_forgetting",
        "primary_url": "https://arxiv.org/abs/2511.12828",
        "source_revision": "arXiv:2511.12828 (2025-11; rechecked 2026-09-21)",
        "reusable_component": (
            "Use frozen, simple-online, and permuted-feedback controls for local spline updates."
        ),
        "limitation": "Local spline support alone does not guarantee retained performance.",
        "task_mapping": [
            "exp7496-causal-update-fixture",
            "exp7497-causal-online-learning",
        ],
    },
    {
        "method": "on_chip_locality",
        "source_id": "on_chip_spline_learning",
        "primary_url": "https://arxiv.org/abs/2602.02056",
        "source_revision": "arXiv:2602.02056 (2026-02; later revisions rechecked)",
        "reusable_component": (
            "Count active coefficients and state bytes before measuring the complete CPU service."
        ),
        "limitation": (
            "CPU arithmetic is not FPGA speed; prefill and durable acknowledgement remain in cost."
        ),
        "task_mapping": ["exp7501-service-placement"],
    },
)


def method_rows() -> list[JsonDict]:
    """Return six reviewed methods without turning paper results into measurements."""

    return [
        {
            "unit_id": row["method"],
            "arm": "method_ingestion",
            **deepcopy(row),
            "external_claim_is_carnot_measurement": False,
            "principle": (
                "A method can guide a control without becoming a local scientific result."
            ),
        }
        for row in METHOD_SPECS
    ]


def write_method_records(root: Path) -> None:
    """Write one bounded note and append one idempotent studying marker."""

    methods = method_rows()
    method_lines = [
        f"| {row['method']} | {row['source_revision']} | {row['reusable_component']} | "
        f"{row['limitation']} | {', '.join(row['task_mapping'])} |"
        for row in methods
    ]
    note = "\n".join(
        [
            "# V656 method ingestion",
            "",
            "Date: 2026-09-21. Scope: advisory contract and method accounting.",
            "External paper results are not Carnot measurements.",
            "",
            "| Method | Paper revision | Reusable component | Limitation | Task mapping |",
            "|---|---|---|---|---|",
            *method_lines,
            "",
            "The rows reuse bounded design components only. They do not reproduce full",
            "paper systems, establish local benefit, or authorize model and hardware claims.",
            "",
        ]
    )
    note_path = root / NOTE_PATH
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note, encoding="utf-8")

    marker = "<!-- EXP7489-V656-METHOD-INGESTION -->"
    study_path = root / STUDY_PATH
    studying = study_path.read_text(encoding="utf-8")
    if marker not in studying:
        addition = "\n".join(
            [
                "",
                marker,
                "## 2026-09-21 Exp7489 — V656 methods — INGESTED",
                "",
                "Response granularity, evidence alignment, corpus leakage controls,",
                "budgeted feedback, KAN support limits, and on-chip locality map to V656.",
                "External paper results are not Carnot measurements. Exact revisions,",
                "limits, and task mappings are in",
                "`docs/research-notes/v656-method-ingestion.md`.",
                "",
            ]
        )
        study_path.write_text(studying.rstrip() + "\n" + addition, encoding="utf-8")


def priority_disposition(root: Path) -> JsonDict:
    """Separate the parser's cross-heading slug from actual call-site evidence."""

    known_text = (root / "ops/known-issues.md").read_text(encoding="utf-8")
    priorities = overdue_priority_lint._parse_priorities(known_text)
    priority = next(row for row in priorities if "AUTORESEARCH CIRCUIT BREAKER" in row.title)
    exp7435 = load_json(root / EXP7435_PATH)
    _roadmap_path, roadmap, _candidates = resolve_v656_roadmap(root)
    roadmap_text = yaml.safe_dump(roadmap, sort_keys=False)
    conductor_text = (root / "scripts/research_conductor.py").read_text(encoding="utf-8")
    resolved_heading = bool(
        re.search(
            r"(?m)^### RESOLVED 2026-07-14 .*retro_timing_fallback\.py.*$",
            known_text,
        )
    )
    callsite_present = (
        "build_retro_timing_fallback(current, repo_root=PROJECT_ROOT)" in conductor_text
    )
    return {
        "unit_id": "overdue_priority_parse",
        "arm": "priority_disposition",
        "priority_title": priority.title,
        "priority_filed_date": priority.filed_date.isoformat(),
        "priority_source_path": "ops/known-issues.md",
        "addressed_marker_present": "ADDRESSED 2026-09-19 (Exp7435" in priority.raw_block,
        "exp7435_evidence_path": EXP7435_PATH.as_posix(),
        "exp7435_sha256": sha256_file(root / EXP7435_PATH),
        "exp7435_ready_score": exp7435.get("breaker_recovery_ready_score"),
        "parser_slug": priority.slug,
        "slug_present_in_selected_roadmap": overdue_priority_lint._slug_present_in_roadmap_next(
            roadmap_text, priority.slug
        ),
        "slug_match_proves_wiring": False,
        "timing_fallback": {
            "known_issue_heading": (
                "RESOLVED 2026-07-14: WIRE scripts/retro_timing_fallback.py INTO THE CONDUCTOR"
            ),
            "known_issue_state": "resolved" if resolved_heading else "unresolved",
            "known_issue_path": "ops/known-issues.md",
            "callsite_path": "scripts/research_conductor.py",
            "callsite_present": callsite_present,
            "current_task_modified_conductor": False,
            "historical_backfill_optional": True,
        },
        "principle": (
            "A parser slug and green lint cannot replace direct evidence at the named call site."
        ),
    }


def unresolved_obligations(root: Path) -> list[JsonDict]:
    """Keep the parser defect and the current conductor-edit prohibition explicit."""

    priority = priority_disposition(root)
    timing = priority["timing_fallback"]
    return [
        {
            "unit_id": "overdue_priority_cross_heading_association",
            "arm": "operator_obligation",
            "obligation_id": "overdue_priority_cross_heading_association",
            "state": "unresolved_parser_scope",
            "scope": "overdue_priority_lint priority-block parsing",
            "evidence_path": "ops/known-issues.md",
            "observed": {
                "priority_title": priority["priority_title"],
                "associated_slug": priority["parser_slug"],
                "circuit_breaker_addressed": priority["addressed_marker_present"],
            },
            "prohibited_action": "Do not manufacture a V656 implementation task from the slug.",
            "principle": (
                "A false cross-heading association must remain visible even when lint is green."
            ),
        },
        {
            "unit_id": "conductor_modification_scope",
            "arm": "operator_obligation",
            "obligation_id": "conductor_modification_scope",
            "state": "user_forbidden_for_current_task",
            "scope": "retro timing fallback call site",
            "evidence_path": timing["callsite_path"],
            "observed": {
                "known_issue_state": timing["known_issue_state"],
                "callsite_present": timing["callsite_present"],
                "historical_backfill_optional": timing["historical_backfill_optional"],
            },
            "prohibited_path": "scripts/research_conductor.py",
            "prohibited_action": "No conductor edit is authorized by Exp7489.",
            "principle": (
                "A reporting task must not turn parser ambiguity into an unauthorized conductor edit."
            ),
        },
    ]


def build_repository_check_plan(
    root: Path, roadmap_path: Path
) -> list[validation_scope.CommandSpec]:
    """Build the five unchanged repository guards for the selected V656 YAML."""

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
            "overdue_priority",
            (python, "-u", "-c", overdue_code, selected),
            "selected_roadmap",
        ),
    ]


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the shared file-scoped validation plan for this exact module."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets, missing private parents, and command drift."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def receipts_pass(receipts: object, names: Sequence[str]) -> bool:
    """Require one successful and bounded receipt for every exact command."""

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


def receipts_recorded(receipts: object, names: Sequence[str]) -> bool:
    """Require one complete receipt per command while preserving real failures."""

    if not isinstance(receipts, list):
        return False
    counts = Counter(str(row.get("name")) for row in receipts if isinstance(row, Mapping))
    return all(
        counts[name] == 1
        and any(
            row.get("name") == name
            and isinstance(row.get("exit_code"), int)
            and isinstance(row.get("passed"), bool)
            and isinstance(row.get("timed_out"), bool)
            for row in receipts
            if isinstance(row, Mapping)
        )
        for name in names
    )


def _precondition(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Attach source identity and ownership to one exact prerequisite."""

    return {
        "check": check,
        "upstream": upstream,
        "path": upstream,
        "field": field,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": observed == expected,
        "ownership": "read_only_input",
        "principle": "An exact prerequisite prevents a favorable reduction over the wrong bytes.",
    }


def collect_preconditions(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Record readable inputs, the driving requirement, and original failure flags."""

    rows: list[JsonDict] = []
    for relative in SOURCE_INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                **_precondition(
                    f"source_bytes:{relative.as_posix()}",
                    relative.as_posix(),
                    "bytes",
                    "readable_nonempty_bytes",
                    "readable_nonempty_bytes" if available else None,
                ),
                "sha256": sha256_file(path) if available else None,
            }
        )
    roadmap = load_yaml(roadmap_path)
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    by_id = {str(row.get("task_id")): row for row in dispositions}
    rows.extend(
        [
            _precondition(
                "driving_requirement",
                SPEC_PATH.as_posix(),
                "REQ-*",
                "REQ-REPORT-7489",
                "REQ-REPORT-7489" if "REQ-REPORT-7489" in spec else None,
            ),
            _precondition(
                "roadmap_identity",
                roadmap_path.relative_to(root).as_posix(),
                "milestone",
                MILESTONE,
                roadmap.get("milestone"),
            ),
            _precondition(
                "v655_disposition_count",
                CAPSTONE_PATH.as_posix(),
                "task_dispositions",
                14,
                len(dispositions),
            ),
            _precondition(
                "v655_producer_count",
                CAPSTONE_PATH.as_posix(),
                "producer_present",
                13,
                sum(row.get("producer_present") is True for row in dispositions),
            ),
            _precondition(
                "exp7475_failed_receipt",
                str(by_id["exp7475-contract-methods"].get("observed_path")),
                "failed_validation_receipts.name",
                ["overdue_priority"],
                [
                    row.get("name")
                    for row in by_id["exp7475-contract-methods"].get(
                        "failed_validation_receipts", []
                    )
                ],
            ),
            _precondition(
                "exp7484_noncanonical_substrate",
                str(by_id["exp7484-decision-audit"].get("observed_path")),
                "inference_substrate",
                "aggregation_from_hash_bound_raw_rows_and_frozen_numeric_checkpoints",
                by_id["exp7484-decision-audit"].get("original_inference_substrate"),
            ),
            _precondition(
                "exp7486_absent",
                str(by_id["exp7486-arc-cost-panel-b"].get("expected_path")),
                "producer_present",
                False,
                by_id["exp7486-arc-cost-panel-b"].get("producer_present"),
            ),
            _precondition(
                "exp7488_disqualified",
                CAPSTONE_PATH.as_posix(),
                "verdict_class",
                "disqualified",
                by_id["exp7488-capstone"].get("original_verdict_class"),
            ),
            _precondition(
                "host_identity",
                "/proc/cpuinfo",
                "machine_nonempty",
                True,
                bool(platform.machine()),
            ),
        ]
    )
    return rows


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Bind current protocol files and original V655 producer bytes separately."""

    hashes: dict[str, JsonDict] = {}
    current_paths = (
        *SOURCE_INPUT_PATHS,
        roadmap_path.relative_to(root),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
        NOTE_PATH,
    )
    for relative in current_paths:
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "evidence_type": "current_protocol_or_input_bytes",
                "principle": "Current inputs are rehashed so later edits cannot inherit this receipt.",
            }
    for row in dispositions:
        path_text = row.get("observed_path")
        if isinstance(path_text, str) and (root / path_text).is_file():
            hashes[f"historical:{row.get('task_id')}"] = {
                "path": path_text,
                "sha256": sha256_file(root / path_text),
                "evidence_type": "historical_model_receipts",
                "original_verdict_class": row.get("original_verdict_class"),
                "flagged_adversarial": row.get("flagged_adversarial"),
                "counted_as_current_invocation": False,
                "principle": "Historical bytes retain their original verdict and never become current calls.",
            }
    return hashes


def _gate(
    check: str,
    gate_type: str,
    expected: Any,
    observed: Any,
    op: str,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Give every acceptance check a category and prevented failure mode."""

    return {
        "check": check,
        "gate_type": gate_type,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "op": op,
        "passed": passed,
        "principle": principle,
    }


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Keep validity, support, and benefit claims independent."""

    return [
        _gate(
            "preconditions_authenticated",
            "validity",
            True,
            all(row.get("passed") is True for row in preconditions),
            "==",
            all(row.get("passed") is True for row in preconditions),
            "A favorable metric cannot excuse invalid or missing evidence.",
        ),
        _gate(
            "exact_v656_contract",
            "validity",
            True,
            contract.get("passed") is True,
            "==",
            contract.get("passed") is True,
            "Exact paired authorities prevent task identity and gate drift.",
        ),
        _gate(
            "private_contract_mutations_rejected",
            "validity",
            12,
            sum(row.get("rejected") is True for row in mutations),
            "==",
            len(mutations) == 12 and all(row.get("rejected") is True for row in mutations),
            "Planted count, identity, order, path, gate, and milestone defects must fail.",
        ),
        _gate(
            "v655_dispositions_authenticated",
            "validity",
            14,
            sum(row.get("authenticated") is True for row in dispositions),
            "==",
            len(dispositions) == 14
            and all(row.get("authenticated") is True for row in dispositions),
            "Thirteen producers and one absence prevent completed-task inflation.",
        ),
        _gate(
            "method_rows_bounded",
            "support",
            6,
            len(methods),
            "==",
            len(methods) == 6
            and all(row.get("external_claim_is_carnot_measurement") is False for row in methods),
            "Bounded method intake prevents paper results from becoming Carnot measurements.",
        ),
        _gate(
            "affected_validation",
            "validity",
            True,
            validation.get("affected_checks_passed") is True,
            "==",
            validation.get("affected_checks_passed") is True,
            "Scoped checks prevent unrelated suite state from laundering changed code.",
        ),
        _gate(
            "repository_guards",
            "validity",
            True,
            validation.get("repository_checks_passed") is True,
            "==",
            validation.get("repository_checks_passed") is True,
            "Unchanged schema, gate, exclusion, ARC, and priority guards retain policy.",
        ),
        _gate(
            "terminal_readers",
            "validity",
            True,
            validation.get("terminal_checks_passed") is True,
            "==",
            validation.get("terminal_checks_passed") is True,
            "Fresh readers prevent an in-process producer from acting as its own oracle.",
        ),
        _gate(
            "advisory_contract_ready",
            "support",
            True,
            bool(contract.get("passed")),
            "==",
            contract.get("passed") is True,
            "A valid scientific null must not block independent V656 measurements.",
        ),
        _gate(
            "scientific_benefit_not_claimed",
            "benefit",
            False,
            False,
            "==",
            True,
            "A favorable seed, fixture, or low-support result cannot replace held-out value.",
        ),
    ]


def failure_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name every failed check with its exact expected and observed values."""

    failed = [
        {
            "check": row.get("check"),
            "upstream": "current_exp7489_reduction",
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


def terminal_state(
    preconditions_passed: bool, validation_passed: bool, contract_ready: bool
) -> tuple[str, str, str]:
    """Classify external blocks before invalid work and valid advisory nulls."""

    if not preconditions_passed:
        return (
            "blocked_external_v656_contract_input",
            "blocked_external_v656_contract_input",
            "blocked",
        )
    if not validation_passed or not contract_ready:
        return (
            "complete_disqualified_v656_contract_or_validation",
            "complete_disqualified_v656_contract_or_validation",
            "disqualified",
        )
    return (
        "complete_advisory_v656_contract_and_methods",
        "complete_null_v656_contract_methods_ingested",
        "null",
    )


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain how each top-level field prevents one reporting failure."""

    required = {
        "schema": "Versioned schema, exact experiment_id, milestone and terminal status prevent reader drift.",
        "run_date": "Use 20260921; retain measured UTC and monotonic clock/process identity.",
        "preconditions_checked": "Record exact resource paths, observed values, ownership and input validity.",
        "MODEL_SPECS": (
            "Current LLM tasks name resolved models; this aggregation uses an empty list."
        ),
        "model_specs": "Lowercase model_specs mirrors the empty current-model declaration.",
        "model_invoked": (
            "Attempted live loads, forwards, and generations differ from historical events."
        ),
        "invocation_counts": (
            "Reconcile attempted, completed, failed, cancelled, and in-flight model work."
        ),
        "inference_substrate": (
            "Aggregation uses exactly aggregation_from_upstream_artifacts to prevent ambiguity."
        ),
        "inference_substrate_class": (
            "The aggregation class prevents synthetic duration floors and model-work confusion."
        ),
        "execution_venue": "Host execution stays distinct from archived board evidence.",
        "duration_s": "Measure actual work and separate inference, optimization, and validation.",
        "phase_spans": "Flushed progress and real checkpoints expose unfinished operations and stalls.",
        "random_seed": "Freeze role, optimizer, audit, order, and interval seeds; explain null seeds.",
        "reproducibility_checksum": (
            "Bind code, model identity, data roles, evidence bytes, and validation scope."
        ),
        "source_artifact_hashes": (
            "Preserve upstream bytes, verdicts, and flags without laundering history."
        ),
        "rows": "Per-unit rows with failures and censoring permit independent headline reduction.",
        "sample_size_budget": (
            "Separate planned, attempted, complete, failed, excluded, censored, and unstarted units."
        ),
        "acceptance_gate_results": (
            "Every validity, support, and benefit check states expected, observed, op, and purpose."
        ),
        "gate_check_summary": (
            "Every blocked verdict names the exact failed field, source, expected, and observed value."
        ),
        "honest_verdict": "Use a complete terminal finding without rewriting upstream blocked records.",
        "verdict_class": (
            "The closed enum keeps external absence blocked and reserves partial for owned retries."
        ),
        "verifier_is_oracle": "Oracle-defined fixtures cannot support a positive scientific verdict.",
        "flagged_adversarial": "Actual reader flags and failures stay visible and cannot open a gate.",
        "validation_receipts": (
            "Exact commands, exits, scopes, and log hashes make checks independently reviewable."
        ),
        "field_principles": "Every emitted top-level field states the failure it prevents.",
        "contract_ready_score": (
            "The bare score covers the exact contract and required checks, not scientific benefit."
        ),
        "v655_dispositions": "Fourteen rows include thirteen producers and the absent panel B.",
        "method_rows": "Source revisions and limitations map methods to V656 tasks.",
        "unresolved_obligations": (
            "The parser association and forbidden conductor edit remain explicit."
        ),
    }
    return {
        field: required.get(
            field,
            f"The {field} field preserves its exact scope and prevents silent evidence reinterpretation.",
        )
        for field in fields
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind the complete artifact while excluding only the checksum's own value."""

    payload = deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    return canonical_hash(payload)


def zero_test_phase_spans() -> list[JsonDict]:
    """Supply deterministic spans for pure reducer unit tests."""

    return [
        {
            "phase": phase,
            "start_s": 0.0,
            "end_s": 0.0,
            "duration_s": 0.0,
            "completed_units": 0,
            "heartbeat_count": 0,
            "checkpoint": "unit_test_fixture",
            "principle": "A named zero-work test span prevents invented runtime evidence.",
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
    """Build one schema-complete advisory artifact from reducible evidence rows."""

    contract_copy = deepcopy(dict(contract))
    mutations_copy = deepcopy(list(mutation_rows))
    dispositions_copy = deepcopy(list(dispositions))
    methods = method_rows()
    priority = priority_disposition(root)
    obligations = unresolved_obligations(root)
    preconditions = collect_preconditions(root, roadmap_path, dispositions_copy)
    receipts = deepcopy(list(validation.get("validation_receipts") or []))
    gates = _acceptance_gates(
        contract_copy,
        mutations_copy,
        dispositions_copy,
        methods,
        preconditions,
        validation,
    )
    preconditions_passed = all(row.get("passed") is True for row in preconditions)
    validation_passed = all(
        row.get("passed") is True for row in gates if row.get("gate_type") == "validity"
    )
    contract_ready = bool(
        preconditions_passed and validation_passed and contract_copy.get("passed")
    )
    status, honest, verdict = terminal_state(
        preconditions_passed, validation_passed, contract_ready
    )
    rows = [
        *deepcopy(contract_copy.get("contract_rows") or []),
        *mutations_copy,
        *dispositions_copy,
        *deepcopy(methods),
        deepcopy(priority),
        *deepcopy(obligations),
    ]
    duration_s = max(0.0, (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000)
    validation_duration = sum(
        float(row.get("duration_s", 0.0)) for row in receipts if isinstance(row, Mapping)
    )
    invalid_dispositions = sum(
        row.get("original_verdict_class") == "disqualified" for row in dispositions_copy
    )
    absent_dispositions = sum(row.get("producer_present") is False for row in dispositions_copy)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7489,
        "title": "Bind the V656 contract and disposition terminal V655 evidence",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": status,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "current_owner_pid": os.getpid(),
        "duration_s": duration_s,
        "clock_identity": {
            "utc_clock": "datetime.now(UTC)",
            "monotonic_clock": "time.monotonic_ns",
            "process_identity": "current_owner_pid",
        },
        "device_identity": {
            "machine": platform.machine(),
            "node": platform.node(),
            "processor": platform.processor(),
            "cuda_used": False,
        },
        "phase_spans": deepcopy(list(phase_spans)),
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "inference_substrate_details": {
            "current_llm_calls": 0,
            "historical_model_evidence_counted_as_current": False,
            "numeric_fitting_performed": False,
        },
        "execution_venue": "host",
        "small_ebm_training": {
            "receipt_class": "small_ebm_training",
            "performed": False,
            "current_llm_calls": 0,
            "parameter_count": 0,
        },
        "duration_components_s": {
            "model_load": 0.0,
            "model_forward": 0.0,
            "model_generation": 0.0,
            "optimization": 0.0,
            "validation": validation_duration,
            "total": duration_s,
        },
        "random_seed": deepcopy(RANDOM_SEED),
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "contract_comparison": contract_copy,
        "task_contract_rows": deepcopy(contract_copy.get("contract_rows") or []),
        "contract_mutation_rows": mutations_copy,
        "v655_dispositions": dispositions_copy,
        "method_rows": deepcopy(methods),
        "priority_disposition": deepcopy(priority),
        "unresolved_obligations": deepcopy(obligations),
        "historical_inference_sidecars": [
            {
                "task_id": row.get("task_id"),
                "path": row.get("observed_path"),
                "sha256": row.get("source_sha256") or row.get("capstone_sha256"),
                "scope": "historical_model_receipts",
                "original_verdict_class": row.get("original_verdict_class"),
                "flagged_adversarial": row.get("flagged_adversarial"),
                "counted_as_current_invocation": False,
            }
            for row in dispositions_copy
        ],
        "source_artifact_hashes": _source_hashes(root, roadmap_path, dispositions_copy),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows) - absent_dispositions,
            "complete": len(rows) - absent_dispositions,
            "completed": len(rows) - absent_dispositions,
            "failed": invalid_dispositions,
            "excluded": invalid_dispositions + absent_dispositions,
            "censored": absent_dispositions,
            "unstarted": absent_dispositions,
            "independent_unit": "contract_mutation_disposition_method_or_obligation_row",
            "counting_rule": "Failure, exclusion, and censoring are diagnostic overlaps.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": failure_summary(gates),
        "contract_ready_score": int(contract_ready),
        "method_ingestion_complete_score": int(
            len(methods) == 6
            and all(
                row.get("source_revision")
                and row.get("reusable_component")
                and row.get("limitation")
                for row in methods
            )
        ),
        "honest_verdict": honest,
        "verdict_class": verdict,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "validation_scope": {
            "affected_manifest": {
                "experiment_id": VALIDATION_MANIFEST.experiment_id,
                "test_paths": list(VALIDATION_MANIFEST.test_paths),
                "changed_modules": list(VALIDATION_MANIFEST.changed_modules),
                "static_paths": list(VALIDATION_MANIFEST.static_paths),
                "documentation_paths": [NOTE_PATH.as_posix(), STUDY_PATH.as_posix()],
            },
            "full_python_suite_run": False,
            "numbered_runtime_e2e_applicable": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": True,
            "numbered_runtime_e2e_applicable": False,
        },
        "advisory_only": True,
        "later_science_tasks_gate_on_exp7489": False,
        "promotion_score": 0,
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
    """Recompute advisory readiness and terminal class from raw artifact evidence."""

    contract = artifact.get("contract_comparison")
    mutations = artifact.get("contract_mutation_rows")
    dispositions = artifact.get("v655_dispositions")
    methods = artifact.get("method_rows")
    preconditions = artifact.get("preconditions_checked")
    receipts = artifact.get("validation_receipts")
    contract_ok = isinstance(contract, Mapping) and contract.get("passed") is True
    mutations_ok = (
        isinstance(mutations, list)
        and len(mutations) == 12
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
        and len(methods) == 6
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
    repository_ok = receipts_pass(receipts, REPOSITORY_CHECK_NAMES)
    terminal_ok = receipts_pass(receipts, TERMINAL_CHECK_NAMES) if require_terminal else True
    validation_ok = affected_ok and repository_ok and terminal_ok
    contract_ready = (
        contract_ok
        and mutations_ok
        and dispositions_ok
        and methods_ok
        and preconditions_ok
        and validation_ok
    )
    status, honest, verdict = terminal_state(preconditions_ok, validation_ok, contract_ready)
    return {
        "contract_ready_score": int(contract_ready),
        "method_ingestion_complete_score": int(methods_ok),
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
    }


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash all declared current and historical sources from this worktree."""

    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return False
    try:
        roadmap_path, _roadmap, _candidates = resolve_v656_roadmap(root)
        expected = _source_hashes(root, roadmap_path, collect_v655_dispositions(root))
    except (OSError, ValueError, json.JSONDecodeError):  # pragma: no cover - cold corruption.
        return False
    return hashes == expected


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, rows, hashes, gates, principles, and reduction."""

    if not isinstance(value, Mapping):  # pragma: no cover - CLI corruption defense.
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
        roadmap_path, roadmap, _candidates = resolve_v656_roadmap(root)
        markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
        contract = compare_contract_authorities(markdown, roadmap)
        expected_mutations = run_contract_mutation_controls(markdown, roadmap)
    except (OSError, ValueError, json.JSONDecodeError):  # pragma: no cover - cold corruption.
        roadmap_path, contract, expected_mutations = root / ACTIVE_ROADMAP_PATH, {}, []
    if artifact.get("task_contract_rows") != contract.get("contract_rows"):
        errors.append("task_contract_rows_mismatch")
    if artifact.get("contract_comparison") != contract:
        errors.append("contract_comparison_mismatch")
    if artifact.get("contract_mutation_rows") != expected_mutations:
        errors.append("contract_mutations_invalid")
    expected_dispositions = collect_v655_dispositions(root)
    if artifact.get("v655_dispositions") != expected_dispositions:
        errors.append("v655_dispositions_mismatch")
    if artifact.get("method_rows") != method_rows():
        errors.append("method_rows_mismatch")
    if artifact.get("priority_disposition") != priority_disposition(root):
        errors.append("priority_disposition_mismatch")
    if artifact.get("unresolved_obligations") != unresolved_obligations(root):
        errors.append("unresolved_obligations_mismatch")
    receipts = artifact.get("validation_receipts")
    if not receipts_recorded(receipts, validation_scope.REQUIRED_CHECK_NAMES):
        errors.append("affected_validation_invalid")
    if not receipts_recorded(receipts, REPOSITORY_CHECK_NAMES):
        errors.append("repository_validation_invalid")
    if require_terminal and not receipts_recorded(receipts, TERMINAL_CHECK_NAMES):
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
    if (
        not isinstance(principles, Mapping)
        or set(principles) != set(artifact)
        or any(not value for value in principles.values())
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
    del roadmap_path
    return list(dict.fromkeys(errors))


def _phase_span(  # pragma: no cover - measured run boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    units: int,
    checkpoint: str,
) -> JsonDict:
    """Close one real phase and name the completed checkpoint."""

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


def _terminal_commands(  # pragma: no cover - exercised by capability E2E.
    root: Path, candidate: Path
) -> list[PlannedCommand]:
    """Build entrypoint replay, independent reduction, and strict reader checks."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7489_v656_contract_methods import validate_artifact;"
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


def run_experiment(  # pragma: no cover - exercised by the capability E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Run exact scoped checks and publish only the validated terminal JSON."""

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
    roadmap_path, roadmap, candidates = resolve_v656_roadmap(root)
    dispositions = collect_v655_dispositions(root)
    preconditions = collect_preconditions(root, roadmap_path, dispositions)
    if not all(row["passed"] for row in preconditions):
        failed = next(row for row in preconditions if not row["passed"])
        raise RuntimeError(
            "blocked_missing_authenticated_input:"
            f"{failed['upstream']}:{failed['field']}:{failed['observed']}"
        )
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
            "contract", phase_started, started, units=contract_units, checkpoint="contract_rows"
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

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7489-validation-", dir="/tmp"))
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
            "documentation_paths": [NOTE_PATH.as_posix(), STUDY_PATH.as_posix()],
        },
    )
    phase_started = time.monotonic()
    progress(
        started,
        "validation",
        "before_subprocesses",
        planned_units=len(affected_plan) + len(REPOSITORY_CHECK_NAMES),
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
        [
            PlannedCommand(command, "required_validation", True)
            for command in build_repository_check_plan(root, roadmap_path)
        ],
        log_dir=raw_dir / "validation/repository",
        heartbeat_s=60.0,
    )
    repository_passed = receipts_pass(repository, REPOSITORY_CHECK_NAMES)
    spans.append(
        _phase_span(
            "validation",
            phase_started,
            started,
            units=len(affected) + len(repository),
            checkpoint="affected_and_repository_checks",
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
    candidate["roadmap_resolution_candidates"] = candidates
    candidate["field_principles"] = _field_principles(tuple(candidate))
    candidate["reproducibility_checksum"] = reproducibility_checksum(candidate)
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
            checkpoint="cold_replay_and_terminal_readers",
        )
    )
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        root,
        roadmap_path,
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
    final["roadmap_resolution_candidates"] = candidates
    final["field_principles"] = _field_principles(tuple(final))
    final["reproducibility_checksum"] = reproducibility_checksum(final)
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic", path=RESULT_PATH.as_posix())
    atomic_json(candidate_path, final)
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publish", "after_atomic", path=RESULT_PATH.as_posix())
    return final


def date_argument(value: str) -> str:
    """Accept only the execution date frozen by the V656 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the measured execution and cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date_argument)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(  # pragma: no cover - exercised by capability E2E.
    argv: Sequence[str] | None = None,
) -> int:
    """Run V656 contract ingestion or validate one measured candidate."""

    print("[exp7489] phase=startup event=flushed", flush=True)
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


if __name__ == "__main__":  # pragma: no cover - module execution boundary.
    raise SystemExit(main())

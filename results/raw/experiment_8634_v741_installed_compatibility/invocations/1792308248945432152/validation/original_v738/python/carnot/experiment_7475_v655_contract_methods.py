"""Audit the V655 task contract against immutable V654 evidence.

The task joins two planning authorities, historical result bytes, and six
bounded source methods. It is advisory. It does not make later experiments
depend on this report and it does not turn old model work into a current call.

Spec refs: REQ-REPORT-7475 and SCENARIO-REPORT-7475-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
import platform
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
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json


JsonDict = dict[str, Any]
SourceFetcher = Callable[[Mapping[str, str]], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260921"
MILESTONE = "2026.09.655"
EXPERIMENT_ID = "exp7475-contract-methods"
SCHEMA = "carnot.exp7475.v655.contract_methods.v1"
RANDOM_SEED = {
    "contract_mutations": 7_475_655_01,
    "source_order": 7_475_655_02,
    "audit": 7_475_655_03,
    "bootstrap": None,
}

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7475_v655_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7475_v655_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7475_v655_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7475_v655_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7475_v655_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v655-method-ingestion.md")
STUDY_PATH = Path("research-studying.md")
CAPSTONE_PATH = Path("results/experiment_7474_v654_capstone.json")

EXPECTED_TASK_IDS = (
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

SELECTED_SOURCES: tuple[JsonDict, ...] = (
    {
        "source_id": "semif",
        "version": "ca3ba65f142967030ecb453346e94d6f476a69df",
        "url": "https://github.com/TheoLeeCJ/SemIf/tree/ca3ba65f142967030ecb453346e94d6f476a69df",
        "review_status": "rechecked_2026-09-21",
        "kind": "primary_source_release",
    },
    {
        "source_id": "kan_cl",
        "version": "arXiv:2605.12306v1 (2026-05-12)",
        "url": "https://arxiv.org/html/2605.12306v1",
        "review_status": "method_and_ablations_read",
        "kind": "primary_paper",
    },
    {
        "source_id": "support_overlap",
        "version": "arXiv:2511.12828 (2025-11; AAAI 2026 record)",
        "url": "https://arxiv.org/abs/2511.12828",
        "review_status": "rechecked_2026-09-21",
        "kind": "primary_paper",
    },
    {
        "source_id": "limited_feedback",
        "version": "arXiv:2609.05820 (2026-09-05)",
        "url": "https://arxiv.org/abs/2609.05820",
        "review_status": "rechecked_2026-09-21",
        "kind": "primary_paper",
    },
    {
        "source_id": "efficiency",
        "version": "arXiv:2609.14839v1 (2026-09-13)",
        "url": "https://arxiv.org/html/2609.14839v1",
        "review_status": "reviewed_2026-09-21",
        "kind": "primary_paper",
    },
    {
        "source_id": "hardware_locality",
        "version": "arXiv:2602.02056v4 (2026-06-19)",
        "url": "https://arxiv.org/abs/2602.02056v4",
        "review_status": "rechecked_2026-09-21",
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
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/arc_levelup_guarantee_lint.py"),
    Path("scripts/overdue_priority_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7461_v654_contract_methods.py"),
    Path("research-references.md"),
    STUDY_PATH,
    DESIGN_PATH,
    SPEC_PATH,
    CAPSTONE_PATH,
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
WRITTEN_PATHS = (MODULE_PATH, WRAPPER_PATH, TEST_PATH, NOTE_PATH, STUDY_PATH, RESULT_PATH)


def utc_now() -> str:  # pragma: no cover - real clock boundary.
    """Return one real UTC boundary for the measured receipt."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush each boundary so a bounded task never appears inactive."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7475] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so a later source change fails cold replay."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_yaml(path: Path) -> JsonDict:
    """Load one YAML mapping and reject a malformed document."""

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


def resolve_v655_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer a matching staged roadmap, then accept the matching active one."""

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
                "observed": observed,
                "expected": MILESTONE,
                "matches_milestone": matches,
            }
        )
        if selected is None and matches:
            selected = (path, value)
    if selected is None:
        raise ValueError("V655 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def _parse_v655_markdown(text: str) -> JsonDict:
    """Normalize the V655 table labels before using the independent parser."""

    normalized = re.sub(
        r"(?m)^\| Order \| Task ID \| Title \| Phase \| Deliverable \| Substrate class \| Structured gates \|$",
        "| Order | Task ID | Exact title | Phase | Deliverable | Substrate class | Structured gate |",
        text,
        count=1,
    )
    return parse_markdown_contract(normalized)


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only fields declared independently by both authorities."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Prove that each gate reads a field from an earlier producer."""

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
        declarations: list[JsonDict] = []
        for gate in consumer.get("gated_on") or []:
            upstream = gate.get("upstream")
            producer_entry = positions.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            declared = _field_declared(str(producer.get("prompt", "")), gate.get("artifact_field"))
            precedes = producer_index is not None and producer_index < consumer_index
            declarations.append(
                {
                    "upstream": upstream,
                    "artifact_field": gate.get("artifact_field"),
                    "producer_precedes_consumer": precedes,
                    "declared_verbatim": declared,
                    "passed": precedes and declared,
                }
            )
        result[str(consumer.get("id"))] = declarations
    return result


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare all fields in the two V655 contract authorities."""

    try:
        markdown = _parse_v655_markdown(markdown_text)
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
    """Read the selected V655 authorities without repairing either one."""

    _path, roadmap, _candidates = resolve_v655_roadmap(root)
    return compare_contract_authorities((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)


def _table_rows(markdown: str) -> tuple[list[str], list[int]]:
    """Locate rows in the exact task table and nowhere else."""

    lines = markdown.splitlines()
    indices = [
        index for index, line in enumerate(lines) if re.match(r"^\|\s*\d+\s*\|\s*exp\d+", line)
    ]
    return lines, indices


def _mutate_markdown(markdown: str, mutation: str) -> str:
    """Change one Markdown field while its YAML peer stays intact."""

    if mutation == "milestone":
        return re.sub(
            r"(?m)^\*\*Milestone:\*\*\s*`?2026\.09\.655`?\s*$",
            "**Milestone:** 2026.09.654",
            markdown,
            count=1,
        )
    lines, indices = _table_rows(markdown)
    if mutation == "count":
        del lines[indices[-1]]
    elif mutation == "order":
        lines[indices[0]], lines[indices[1]] = lines[indices[1]], lines[indices[0]]
    else:
        cells = [cell.strip() for cell in lines[indices[0]].strip().strip("|").split("|")]
        positions = {"id": 1, "path": 4, "field": 3}
        replacements = {"id": "exp9999-changed", "path": "results/changed.json", "field": "9"}
        cells[positions[mutation]] = replacements[mutation]
        lines[indices[0]] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines) + "\n"


def _mutate_yaml(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Change one YAML field while its Markdown peer stays intact."""

    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "milestone":
        changed["milestone"] = "2026.09.654"
    elif mutation == "count":
        changed["tasks"] = tasks[:-1]
    elif mutation == "order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif mutation == "id":
        tasks[0]["id"] = "exp9999-changed"
    elif mutation == "path":
        tasks[0]["deliverable"] = "results/changed.json"
    elif mutation == "field":
        tasks[0]["phase"] = 9
    return changed


def run_contract_mutation_controls(
    markdown_text: str, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Reject six distinct drifts in each private authority copy."""

    mutations = ("count", "id", "order", "path", "field", "milestone")
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
                else _parse_v655_markdown(markdown_text)
            )
            rows.append(
                {
                    "unit_id": f"{authority}:{mutation}",
                    "arm": "contract_mutation",
                    "authority": authority,
                    "mutation": mutation,
                    "rejected": comparison["passed"] is False,
                    "other_authority_readable": other["milestone"] == MILESTONE
                    and len(other["tasks"]) == 14,
                    "errors": comparison["errors"],
                }
            )
    return rows


def independent_branch_schedule(contract: Mapping[str, Any]) -> list[JsonDict]:
    """Keep all thirteen later tasks independent of this advisory audit."""

    return [
        {
            "unit_id": task_id,
            "arm": "independent_branch_schedule",
            "task_id": task_id,
            "contract_audit_passed": contract.get("passed") is True,
            "scheduled_independently": True,
            "contract_failure_effect": "none",
        }
        for task_id in EXPECTED_TASK_IDS[1:]
    ]


def collect_v654_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate every V654 outcome from the capstone and source bytes."""

    capstone_path = root / CAPSTONE_PATH
    capstone = load_json(capstone_path)
    source_rows = capstone.get("task_dispositions")
    if not isinstance(source_rows, list) or len(source_rows) != 14:
        raise ValueError("V654 capstone must contain fourteen task_dispositions")
    if [row.get("task_id") for row in source_rows if isinstance(row, Mapping)] != [
        task.replace("exp7475", "exp7461") for task in ()
    ] and [row.get("task_id") for row in source_rows if isinstance(row, Mapping)] != [
        "exp7461-contract-methods",
        "exp7462-option-protocol",
        "exp7463-semif-e0-logprob-parity",
        "exp7464-semif-e6-decision-cost-profile",
        "exp7465-source-option-capture",
        "exp7466-typed-energy-calibration",
        "exp7467-factual-span-canary",
        "exp7468-residual-learner",
        "exp7469-continuous-residual-learning",
        "exp7470-independent-audit",
        "exp7471-arc-seam-observation",
        "exp7472-prefix-service",
        "exp7473-board-continuity",
        "exp7474-capstone",
    ]:
        raise ValueError("V654 capstone task order changed")
    capstone_hash = sha256_file(capstone_path)
    extraction_retired = any(
        isinstance(row, Mapping)
        and row.get("branch") == "compact_extraction"
        and row.get("decision") == "retire"
        for row in capstone.get("continuation_rows") or []
    )
    rows: list[JsonDict] = []
    for source in source_rows:
        assert isinstance(source, Mapping)  # The ordered-ID check above proves this shape.
        found = source.get("found_path")
        source_path = root / str(found) if found else None
        producer_present = bool(source_path and source_path.is_file())
        payload = load_json(source_path) if producer_present and source_path is not None else {}
        source_hash = (
            sha256_file(source_path) if producer_present and source_path is not None else None
        )
        hash_matches = source_hash == source.get("source_sha256") if producer_present else True
        honest_matches = (
            payload.get("honest_verdict") == source.get("honest_verdict")
            if producer_present
            else True
        )
        status_matches = (
            payload.get("status") == source.get("raw_status") if producer_present else True
        )
        if source.get("evidence_state") == "pre_gate":
            class_matches = source.get("verdict_class") == "blocked"
            flag_matches = source.get("flagged_adversarial") is False
        elif producer_present:
            class_matches = payload.get("verdict_class") == source.get("verdict_class")
            flag_matches = bool(payload.get("flagged_adversarial", False)) == bool(
                source.get("flagged_adversarial", False)
            )
        else:
            class_matches = source.get("verdict_class") in {"blocked", "disqualified"}
            flag_matches = True
        authenticated = bool(
            capstone_hash
            and hash_matches
            and honest_matches
            and status_matches
            and class_matches
            and flag_matches
        )
        failed_receipts = [
            {
                "name": receipt.get("name"),
                "exit_code": receipt.get("exit_code"),
                "log_sha256": receipt.get("log_sha256"),
            }
            for receipt in source.get("validation_receipts") or []
            if isinstance(receipt, Mapping)
            and (receipt.get("passed") is False or receipt.get("exit_code") not in {0, None})
        ]
        state = str(source.get("evidence_state"))
        source_kind = {
            "pre_gate": "conductor_pre_gate",
            "missing": "absent_producer",
            "current_work": "capstone_current_work",
            "invalid": "invalid_producer",
            "terminal": "producer",
        }.get(state, "unknown")
        task_id = str(source.get("task_id"))
        rows.append(
            {
                "unit_id": task_id,
                "arm": "v654_disposition_authentication",
                "task_id": task_id,
                "expected_path": source.get("expected_path"),
                "observed_path": found,
                "source_kind": source_kind,
                "producer_present": producer_present,
                "source_sha256": source_hash,
                "capstone_sha256": capstone_hash,
                "evidence_state": state,
                "original_status": source.get("raw_status"),
                "original_honest_verdict": source.get("honest_verdict"),
                "original_verdict_class": source.get("verdict_class"),
                "flagged_adversarial": bool(source.get("flagged_adversarial", False)),
                "valid": source.get("valid"),
                "gate_values": deepcopy(source.get("gate_values") or []),
                "failed_validation_receipts": failed_receipts,
                "validation_failures": list(source.get("validation_authentication_failures") or []),
                "extraction_retired": task_id == "exp7467-factual-span-canary"
                and extraction_retired,
                "authenticated": authenticated,
                "evidence_scope": "historical_model_receipts",
                "counted_as_current_invocation": False,
            }
        )
    return rows


def _direct_fetch(source: Mapping[str, str]) -> JsonDict:  # pragma: no cover - network boundary.
    """Read a bounded primary-page prefix and preserve access failure."""

    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "carnot-v655-contract-audit/1.0"}
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


def _check_one_source(source: Mapping[str, str], fetcher: SourceFetcher) -> JsonDict:
    """Normalize one source result for deterministic method reduction."""

    outcome = fetcher(source)
    return {
        "unit_id": source["source_id"],
        "arm": "primary_source_access",
        **deepcopy(dict(source)),
        "access_state": outcome.get("access_state", "failed"),
        "http_status": outcome.get("http_status"),
        "detail": str(outcome.get("detail", "no detail"))[:500],
    }


def check_selected_sources(fetcher: SourceFetcher = _direct_fetch) -> list[JsonDict]:
    """Check each of the six frozen primary sources once and serially."""

    return [_check_one_source(source, fetcher) for source in SELECTED_SOURCES]


def method_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Map six reviewed methods to exact V655 tasks and failure modes."""

    by_id = {str(row.get("source_id")): row for row in source_rows}

    def method(
        source_id: str,
        name: str,
        tasks: list[str],
        component: str,
        failure: str,
        **extra: Any,
    ) -> JsonDict:
        source = by_id.get(source_id, {})
        return {
            "unit_id": name,
            "arm": "method_ingestion",
            "method": name,
            "source_id": source_id,
            "primary_url": source.get("url"),
            "source_revision": source.get("version"),
            "review_status": source.get("review_status"),
            "access_state": source.get("access_state"),
            "usable_component": component,
            "failure_mode": failure,
            "task_mapping": tasks,
            "external_claim_is_local_measurement": False,
            **extra,
        }

    return [
        method(
            "semif",
            "semif_native_readout",
            ["exp7477", "exp7479", "exp7480"],
            "Read final-position logits for a declared option set without text generation.",
            "Token-boundary, option-order, or runtime drift can make option scores incomparable.",
        ),
        method(
            "kan_cl",
            "kan_cl_importance_anchor",
            ["exp7482", "exp7483"],
            "Anchor important residual-head spline knots during delayed updates.",
            "A strong anchor can block adaptation, and head results cannot transfer full-system gains.",
            study_scope="head_component_only",
            full_cnn_backbone_reproduction=False,
        ),
        method(
            "support_overlap",
            "support_overlap_limit",
            ["exp7482", "exp7483"],
            "Measure active-basis overlap and untouched-support retention separately.",
            "Local spline support does not guarantee whole-stream nonforgetting.",
        ),
        method(
            "limited_feedback",
            "budgeted_feedback",
            ["exp7483"],
            "Record label availability, request probability, and a fixed feedback budget.",
            "A small residual learner inherits no routing-regret guarantee from the source paper.",
        ),
        method(
            "efficiency",
            "efficiency_claim_controls",
            ["exp7483", "exp7487"],
            "Use identity, no-update, unchanged-state, and complete-service cost controls.",
            "Operation counts or changed kernels alone can create unsupported speed claims.",
        ),
        method(
            "hardware_locality",
            "hardware_locality",
            ["exp7487"],
            "Separate local update arithmetic from state movement and durable writes.",
            "CPU emulation and Amdahl bounds are not measured board acceleration.",
        ),
    ]


def build_repository_check_plan(
    root: Path, roadmap_path: Path
) -> list[validation_scope.CommandSpec]:
    """Build the five unchanged repository guards for selected V655 bytes."""

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
            "overdue_priority", (python, "-u", "-c", overdue_code, selected), "selected_roadmap"
        ),
    ]


def priority_mappings() -> list[JsonDict]:
    """Map each open scientific priority to its exact V655 experiments."""

    return [
        {"priority": "live_e6_follow_up", "task_ids": ["exp7478", "exp7485", "exp7486"]},
        {"priority": "calibration", "task_ids": ["exp7481"]},
        {"priority": "fr_11", "task_ids": ["exp7482", "exp7483"]},
    ]


def unresolved_obligations(root: Path) -> list[JsonDict]:
    """Keep inaccessible and prohibited work visible as deferred work."""

    return [
        {
            "obligation_id": "scored_blackwell_runtime",
            "state": "deferred_inaccessible",
            "evidence_path": "results/experiment_7463_v654_semif_e0_logprob_parity.json",
            "required_change": "exact mounted scored Blackwell runtime bytes become accessible",
            "current_task_action": "none",
        },
        {
            "obligation_id": "conductor_change",
            "state": "user_forbidden",
            "prohibited_path": "scripts/research_conductor.py",
            "path_exists": (root / "scripts/research_conductor.py").is_file(),
            "current_task_action": "none",
        },
    ]


def collect_preconditions(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Name exact local bytes and historical values before reduction."""

    rows: list[JsonDict] = []
    for relative in SOURCE_INPUT_PATHS:
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
    roadmap = load_yaml(roadmap_path)
    spec = (root / SPEC_PATH).read_text(encoding="utf-8")
    by_id = {row.get("task_id"): row for row in dispositions}
    historical_checks = (
        ("exp7462-option-protocol", "original_verdict_class", "disqualified"),
        ("exp7462-option-protocol", "flagged_adversarial", True),
        ("exp7467-factual-span-canary", "original_verdict_class", "null"),
        ("exp7467-factual-span-canary", "extraction_retired", True),
        ("exp7468-residual-learner", "original_verdict_class", "circular_positive"),
        ("exp7474-capstone", "original_verdict_class", "disqualified"),
    )
    rows.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": SPEC_PATH.as_posix(),
                "path": SPEC_PATH.as_posix(),
                "field": "REQ-*",
                "expected": "REQ-REPORT-7475",
                "observed": "REQ-REPORT-7475" if "REQ-REPORT-7475" in spec else None,
                "passed": "REQ-REPORT-7475" in spec,
            },
            {
                "check": "roadmap_identity",
                "upstream": roadmap_path.name,
                "path": roadmap_path.name,
                "field": "milestone",
                "expected": MILESTONE,
                "observed": roadmap.get("milestone"),
                "passed": roadmap.get("milestone") == MILESTONE,
            },
            {
                "check": "v654_disposition_count",
                "upstream": CAPSTONE_PATH.as_posix(),
                "path": CAPSTONE_PATH.as_posix(),
                "field": "task_dispositions",
                "expected": 14,
                "observed": len(dispositions),
                "passed": len(dispositions) == 14,
            },
            {
                "check": "v654_producer_count",
                "upstream": CAPSTONE_PATH.as_posix(),
                "path": CAPSTONE_PATH.as_posix(),
                "field": "producer_present",
                "expected": 10,
                "observed": sum(row.get("producer_present") is True for row in dispositions),
                "passed": sum(row.get("producer_present") is True for row in dispositions) == 10,
            },
            {
                "check": "host_device_identity",
                "upstream": "current_host",
                "path": "/proc/cpuinfo",
                "field": "machine",
                "expected": "nonempty_host_identity",
                "observed": platform.machine() or None,
                "passed": bool(platform.machine()),
            },
        ]
    )
    for task_id, field, expected in historical_checks:
        observed = by_id.get(task_id, {}).get(field)
        rows.append(
            {
                "check": f"historical:{task_id}:{field}",
                "upstream": task_id,
                "path": by_id.get(task_id, {}).get("observed_path") or CAPSTONE_PATH.as_posix(),
                "field": field,
                "expected": expected,
                "observed": observed,
                "passed": observed == expected,
            }
        )
    return rows


def _source_hashes(
    root: Path, roadmap_path: Path, dispositions: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Bind protocol, implementation, and historical evidence bytes."""

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
            }
    return hashes


def receipts_pass(receipts: object, names: Sequence[str]) -> bool:
    """Require one successful, bounded receipt for each exact command."""

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


def _gate(
    check: str,
    category: str,
    operator: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Attach one failure-prevention principle to every gate."""

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
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    preconditions: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Separate required validity, readiness, and scientific benefit."""

    return [
        _gate(
            "preconditions_authenticated",
            "required_validity",
            "all",
            True,
            all(row.get("passed") is True for row in preconditions),
            all(row.get("passed") is True for row in preconditions),
            "A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "exact_v655_contract",
            "required_validity",
            "==",
            True,
            contract.get("passed") is True,
            contract.get("passed") is True,
            "A partial contract can send a dependent task to the wrong producer or field.",
        ),
        _gate(
            "contract_mutation_controls",
            "required_validity",
            "==",
            12,
            sum(row.get("rejected") is True for row in mutations),
            len(mutations) == 12 and all(row.get("rejected") is True for row in mutations),
            "A reader that accepts one-sided drift cannot authenticate the executable plan.",
        ),
        _gate(
            "v654_dispositions_authenticated",
            "required_validity",
            "==",
            14,
            sum(row.get("authenticated") is True for row in dispositions),
            len(dispositions) == 14
            and all(row.get("authenticated") is True for row in dispositions),
            "Conductor completion cannot rehabilitate invalid or absent producer evidence.",
        ),
        _gate(
            "bounded_method_map",
            "readiness",
            "==",
            6,
            len(methods),
            len(methods) == 6
            and all(
                row.get("source_revision")
                and row.get("usable_component")
                and row.get("failure_mode")
                and row.get("task_mapping")
                for row in methods
            ),
            "A source name without a component and limit cannot make a measurement executable.",
        ),
        _gate(
            "affected_validation",
            "required_validity",
            "==",
            True,
            validation.get("affected_checks_passed") is True,
            validation.get("affected_checks_passed") is True,
            "A positive scientific metric cannot excuse invalid evidence.",
        ),
        _gate(
            "repository_guards",
            "required_validity",
            "==",
            True,
            validation.get("repository_checks_passed") is True,
            validation.get("repository_checks_passed") is True,
            "Changed guard meanings can hide roadmap, exclusion, ARC, or priority defects.",
        ),
        _gate(
            "terminal_readers",
            "required_validity",
            "==",
            True,
            validation.get("terminal_checks_passed") is True,
            validation.get("terminal_checks_passed") is True,
            "Independent replay prevents a producer from certifying its own inconsistent output.",
        ),
        _gate(
            "contract_ready_separate_from_benefit",
            "readiness",
            "==",
            True,
            contract.get("passed") is True,
            contract.get("passed") is True,
            "A valid null must not suppress an independent measurement.",
        ),
        _gate(
            "scientific_benefit_claimed",
            "scientific_benefit",
            "==",
            False,
            False,
            True,
            "A small sample, favorable seed, or analytic fixture cannot substitute for held-out value.",
        ),
    ]


def failure_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name each failed check and its exact expected and observed values."""

    failed = [row for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [
            {
                "check": row.get("check"),
                "upstream": "current_exp7475_reduction",
                "path": row.get("check"),
                "field": row.get("check"),
                "expected": deepcopy(row.get("expected")),
                "observed": deepcopy(row.get("observed")),
                "operator": row.get("operator"),
            }
            for row in failed
        ],
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain why every field exists without wrapping its value."""

    required = {
        "schema": "Versioned schema with exact roadmap experiment_id, milestone and terminal status prevents silent reader drift.",
        "run_date": "Use 20260921; retain measured UTC and monotonic times with clock/process identity.",
        "preconditions_checked": "Name resources, exact paths, ownership and observed prerequisite values before dependent work.",
        "MODEL_SPECS": "Use unsloth/Qwen3.8-27B-GGUF for current model tasks, [] for numeric/reducer work; also emit lowercase model_specs.",
        "model_invoked": "Any attempted current model call differs from archived or scripted events.",
        "invocation_counts": "Balance attempted, complete, failed, cancelled and in-flight loads, forwards and generations.",
        "inference_substrate": "Name actual native readout, bounded generation, numeric learning or artifact aggregation.",
        "inference_substrate_class": "Use the declared computation class; blocked_no_run applies only when nothing executed.",
        "execution_venue": "Use host and record actual CPU/CUDA identities; historical board evidence is separate.",
        "duration_s": "Measure current work without padding; separate model, numeric and validation work.",
        "phase_spans": "Timestamped flushed progress and checkpoints expose long silent or unfinished operations.",
        "random_seed": "Freeze ordering, fitting, audit and bootstrap seeds; deterministic reducers explain any null seed.",
        "reproducibility_checksum": "Bind code, protocol, data roles, model identity, raw shards and validation scope.",
        "source_artifact_hashes": "Preserve exact upstream bytes and their original flags and classes.",
        "rows": "Keep one row per independent contract, source, method, disposition, or scheduling unit.",
        "sample_size_budget": "Separate planned, attempted, complete, failed, censored, excluded and unstarted independent units.",
        "acceptance_gate_results": "Each check names category, expected, observed, operator, result, and failure-prevention principle.",
        "gate_check_summary": "Every blocked verdict names failed check, upstream, exact field or path, expected and observed value.",
        "honest_verdict": "Use a complete terminal finding and preserve a blocked conductor verdict in its historical row.",
        "verdict_class": "Use the closed terminal enum; partial is not an unchanged external absence.",
        "verifier_is_oracle": "Declare whether the verifier defines evaluation truth; an oracle cannot support a positive claim here.",
        "flagged_adversarial": "Keep real reader flags and never clear a historical flag to open a gate.",
        "validation_receipts": "Exact commands, exits, log hashes, and scope establish current validation evidence.",
        "field_principles": "Explain why each field and gate exists so the artifact stands alone.",
        "contract_ready_score": "Bare zero or one records an exact fourteen-task contract and does not certify benefit.",
        "method_rows": "Each source maps one usable method and limitation to exact tasks.",
        "v654_dispositions": "Fourteen immutable historical dispositions prevent completion from becoming a benefit claim.",
        "unresolved_obligations": "External runtime and forbidden conductor changes stay visible.",
    }
    return {
        field: required.get(
            field, f"The {field} field preserves independently checkable audit evidence."
        )
        for field in fields
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the checksum slot itself."""

    payload = {
        key: deepcopy(value) for key, value in artifact.items() if key != "reproducibility_checksum"
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def zero_test_phase_spans() -> list[JsonDict]:
    """Give pure reducer tests deterministic zero-duration phase records."""

    return [
        {
            "phase": phase,
            "start_s": 0.0,
            "end_s": 0.0,
            "duration_s": 0.0,
            "completed_units": 0,
            "heartbeat_count": 0,
            "checkpoint": "test_fixture",
        }
        for phase in (
            "preconditions",
            "model_load",
            "generation",
            "contract",
            "source_access",
            "validation",
            "terminal_validation",
        )
    ]


def _terminal_state(
    preconditions_passed: bool, validation_passed: bool, contract_ready: bool
) -> tuple[str, str, str]:
    """Classify external absence, invalid work, and a valid advisory null."""

    if not preconditions_passed:
        return (
            "complete_blocked_missing_authenticated_input",
            "complete_blocked_missing_authenticated_input",
            "blocked",
        )
    if not validation_passed or not contract_ready:
        return (
            "complete_disqualified_v655_contract_or_validation",
            "complete_disqualified_v655_contract_or_validation",
            "disqualified",
        )
    return (
        "complete_advisory_v655_contract_and_methods",
        "complete_null_v655_contract_methods_ingested",
        "null",
    )


def build_artifact(
    root: Path,
    roadmap_path: Path,
    contract: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one schema-complete artifact from independently reducible rows."""

    dispositions_copy = deepcopy(list(dispositions))
    sources_copy = deepcopy(list(source_rows))
    mutations_copy = deepcopy(list(mutation_rows))
    contract_copy = deepcopy(dict(contract))
    methods = method_rows(sources_copy)
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
    validity_passed = all(row["passed"] for row in gates if row["category"] == "required_validity")
    contract_ready = validity_passed and contract_copy.get("passed") is True
    status, honest, verdict = _terminal_state(
        all(row["passed"] for row in preconditions), validity_passed, contract_ready
    )
    branch_rows = independent_branch_schedule(contract_copy)
    priority_rows = priority_mappings()
    rows = [
        *deepcopy(contract_copy.get("contract_rows") or []),
        *mutations_copy,
        *dispositions_copy,
        *sources_copy,
        *deepcopy(methods),
        *branch_rows,
        *deepcopy(priority_rows),
    ]
    validation_duration = sum(
        float(row.get("duration_s", 0.0)) for row in receipts if isinstance(row, Mapping)
    )
    duration_s = max(0.0, (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7475,
        "title": "Bind fourteen tasks and ingest methods against terminal V654 evidence",
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "status": status,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "duration_s": duration_s,
        "clock_identity": {
            "utc_clock": "datetime.now(UTC)",
            "monotonic_clock": "time.monotonic_ns",
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
            "archived_events_are_current_inference": False,
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
            "numeric_work": 0.0,
            "validation": validation_duration,
            "total": duration_s,
        },
        "random_seed": deepcopy(RANDOM_SEED),
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "contract_comparison": contract_copy,
        "task_contract_rows": deepcopy(contract_copy.get("contract_rows") or []),
        "contract_mutation_rows": mutations_copy,
        "branch_schedule_rows": branch_rows,
        "v654_dispositions": dispositions_copy,
        "source_access_rows": sources_copy,
        "method_rows": deepcopy(methods),
        "priority_mappings": priority_rows,
        "unresolved_obligations": unresolved_obligations(root),
        "historical_inference_sidecars": [
            {
                "task_id": row.get("task_id"),
                "path": row.get("observed_path"),
                "sha256": row.get("source_sha256") or row.get("capstone_sha256"),
                "scope": "historical_model_receipts",
                "counted_as_current_invocation": False,
            }
            for row in dispositions_copy
        ],
        "source_artifact_hashes": _source_hashes(root, roadmap_path, dispositions_copy),
        "rows": rows,
        "sample_size_budget": {
            "planned": len(rows),
            "attempted": len(rows),
            "completed": len(rows),
            "failed": 0,
            "censored": sum(row.get("producer_present") is False for row in dispositions_copy),
            "excluded": 0,
            "unstarted": 0,
            "independent_unit": "contract_source_method_disposition_or_schedule_row",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": failure_summary(gates),
        "contract_ready_score": int(contract_ready),
        "method_ingestion_complete_score": int(
            len(methods) == 6 and all(row.get("source_revision") for row in methods)
        ),
        "honest_verdict": honest,
        "verdict_class": verdict,
        "verifier_is_oracle": False,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "repository_health": {
            "required_current_checks_passed": validity_passed,
            "historical_exp7462_required_validation_failed": True,
            "historical_failure_reclassified": False,
        },
        "capability_e2e": {
            "declared_entrypoint": WRAPPER_PATH.as_posix(),
            "fresh_process_cold_replay": True,
            "numbered_runtime_e2e_applicable": False,
        },
        "promotion_score": 0,
        "roadmap_activated": False,
        "production_defaults_changed": False,
        "generator_weights_changed": False,
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
    """Recompute readiness and terminal class from raw artifact evidence."""

    contract = artifact.get("contract_comparison")
    mutations = artifact.get("contract_mutation_rows")
    dispositions = artifact.get("v654_dispositions")
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
        contract_ok and mutations_ok and dispositions_ok and preconditions_ok and validation_ok
    )
    status, honest, verdict = _terminal_state(preconditions_ok, validation_ok, contract_ready)
    return {
        "contract_ready_score": int(contract_ready),
        "method_ingestion_complete_score": int(
            isinstance(artifact.get("method_rows"), list) and len(artifact["method_rows"]) == 6
        ),
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
    }


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash every declared current and historical source."""

    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return False
    try:
        roadmap_path, _roadmap, _candidates = resolve_v655_roadmap(root)
        expected = _source_hashes(root, roadmap_path, collect_v654_dispositions(root))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return hashes == expected


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, raw reductions, methods, hashes, and principles."""

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
        roadmap_path, roadmap, _candidates = resolve_v655_roadmap(root)
        markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
        contract = compare_contract_authorities(markdown, roadmap)
        expected_mutations = run_contract_mutation_controls(markdown, roadmap)
    except (OSError, ValueError, json.JSONDecodeError):
        roadmap_path, contract, expected_mutations = root / ACTIVE_ROADMAP_PATH, {}, []
    if artifact.get("task_contract_rows") != contract.get("contract_rows"):
        errors.append("task_contract_rows_mismatch")
    if artifact.get("contract_comparison") != contract:
        errors.append("contract_comparison_mismatch")
    if artifact.get("contract_mutation_rows") != expected_mutations:
        errors.append("contract_mutations_invalid")
    expected_dispositions = collect_v654_dispositions(root)
    if artifact.get("v654_dispositions") != expected_dispositions:
        errors.append("v654_dispositions_mismatch")
    sources = artifact.get("source_access_rows")
    source_ids = [row["source_id"] for row in SELECTED_SOURCES]
    if (
        not isinstance(sources, list)
        or [row.get("source_id") for row in sources if isinstance(row, Mapping)] != source_ids
    ):
        errors.append("source_access_rows_invalid")
        sources = []
    if artifact.get("method_rows") != method_rows(sources):
        errors.append("method_rows_mismatch")
    if artifact.get("branch_schedule_rows") != independent_branch_schedule(contract):
        errors.append("branch_schedule_mismatch")
    if artifact.get("priority_mappings") != priority_mappings():
        errors.append("priority_mappings_mismatch")
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
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_invalid")
    gates = artifact.get("acceptance_gate_results")
    if (
        not isinstance(gates, list)
        or not gates
        or any(not isinstance(row, Mapping) or not row.get("principle") for row in gates)
    ):
        errors.append("gate_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    del roadmap_path
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Build the Exp7358 file-scoped validation plan for this task."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets and command drift before a child starts."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _phase_span(  # pragma: no cover - real timing boundary.
    phase: str,
    phase_started: float,
    run_started: float,
    *,
    units: int,
    checkpoint: str,
) -> JsonDict:
    """Close one measured phase and name its durable checkpoint."""

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
    """Build fresh replay, reduction, adversarial, and row checks."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7475_v655_contract_methods import validate_artifact;"
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


def write_method_records(root: Path, source_rows: Sequence[Mapping[str, Any]]) -> None:
    """Write the bounded method note and one idempotent studying marker."""

    methods = method_rows(source_rows)
    access_lines = [
        f"- `{row['source_id']}`: `{row['access_state']}` ({row.get('http_status')}); "
        f"revision `{row['version']}`; {row['url']}"
        for row in source_rows
    ]
    method_lines = [
        f"| {row['method']} | {row['source_revision']} | {row['usable_component']} | "
        f"{row['failure_mode']} | {', '.join(row['task_mapping'])} |"
        for row in methods
    ]
    note = "\n".join(
        [
            "# V655 method ingestion",
            "",
            "Date: 2026-09-21. Scope: advisory contract accounting and bounded source ingestion.",
            "External findings are method inputs. They are not Carnot measurements.",
            "",
            "## Primary access receipts",
            "",
            *access_lines,
            "",
            "## Source-to-method map",
            "",
            "| Method | Revision | Usable component | Failure mode | Experiments |",
            "|---|---|---|---|---|",
            *method_lines,
            "",
            "## Claim boundary",
            "",
            "The KAN-CL work is a head-component study. It is not a reproduction of the",
            "paper's full CNN and backbone system. Hardware locality is an accounting",
            "method until real board execution exists. Failed source access remains a",
            "recorded access outcome. It does not remove the reviewed local method record.",
            "",
        ]
    )
    note_path = root / NOTE_PATH
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note, encoding="utf-8")
    marker = "<!-- EXP7475-V655-METHOD-INGESTION -->"
    study_path = root / STUDY_PATH
    studying = study_path.read_text(encoding="utf-8")
    if marker not in studying:
        addition = "\n".join(
            [
                "",
                marker,
                "## 2026-09-21 Exp7475 — V655 methods — INGESTED",
                "",
                "SemIf native readout, KAN-CL head anchoring, support-overlap limits,",
                "budgeted feedback, efficiency controls, and hardware locality map to V655.",
                "KAN-CL is a head-component study, not the full CNN and backbone system.",
                "Exact revisions, failure modes, and task mappings are in",
                "`docs/research-notes/v655-method-ingestion.md`.",
                "",
            ]
        )
        study_path.write_text(studying.rstrip() + "\n" + addition, encoding="utf-8")


def run_experiment(  # pragma: no cover - exercised by the capability E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Measure inputs, run exact checks, and atomically publish terminal JSON."""

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
    roadmap_path, roadmap, candidates = resolve_v655_roadmap(root)
    dispositions = collect_v654_dispositions(root)
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
            "contract", phase_started, started, units=contract_units, checkpoint="contract_evidence"
        )
    )
    progress(started, "contract", "after", completed_units=contract_units)

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
    write_method_records(root, source_rows)
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

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7475-validation-", dir="/tmp"))
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
        source_rows,
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
        source_rows,
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
    """Accept only the execution date frozen by the V655 contract."""

    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse normal execution and cold-validation modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date_argument)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(  # pragma: no cover - exercised by the capability E2E.
    argv: Sequence[str] | None = None,
) -> int:
    """Run V655 contract ingestion or validate one measured candidate."""

    print("[exp7475] phase=startup event=flushed", flush=True)
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

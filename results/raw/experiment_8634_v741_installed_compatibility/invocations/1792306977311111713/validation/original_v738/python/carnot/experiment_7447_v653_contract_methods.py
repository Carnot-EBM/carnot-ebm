"""Audit the V653 task contract and ingest six bounded method sources.

This task compares two independently written contract authorities. It also
preserves V652 failure causes without turning historical model work into a
current invocation. The result is advisory and cannot gate independent work.

Spec refs: REQ-REPORT-7447 and SCENARIO-REPORT-7447-*.
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
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json


JsonDict = dict[str, Any]
SourceFetcher = Callable[[Mapping[str, str]], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260920"
MILESTONE = "2026.09.653"
EXPERIMENT_ID = "exp7447-contract-methods"
SCHEMA = "carnot.exp7447.v653.contract_methods.v1"
RANDOM_SEED = {"contract_mutations": 7_447_653_01}

ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
RESULT_PATH = Path("results/experiment_7447_v653_contract_methods.json")
RAW_DIR = Path("results/raw/experiment_7447_v653_contract_methods")
MODULE_PATH = Path("python/carnot/experiment_7447_v653_contract_methods.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7447_v653_contract_methods.py")
TEST_PATH = Path("tests/python/test_experiment_7447_v653_contract_methods.py")
NOTE_PATH = Path("docs/research-notes/v653-method-ingestion.md")
STUDY_PATH = Path("research-studying.md")
CAPSTONE_PATH = Path("results/experiment_7446_v652_capstone.json")
COMPLETE_PATH = Path("research-complete.yaml")

EXPECTED_TASK_IDS = (
    "exp7447-contract-methods",
    "exp7448-capture-lifecycle",
    "exp7449-source-protocol",
    "exp7450-prediction-ledger",
    "exp7451-span-capture",
    "exp7452-source-embeddings",
    "exp7453-energy-calibration",
    "exp7454-continuous-learning",
    "exp7455-decision-audit",
    "exp7456-extraction-audit",
    "exp7457-arc-exposure",
    "exp7458-durable-updates",
    "exp7459-board-continuity",
    "exp7460-capstone",
)

SELECTED_SOURCES: tuple[JsonDict, ...] = (
    {
        "source_id": "hidden_probe",
        "version": "arXiv:2606.02628 (2026-05-30)",
        "url": "https://arxiv.org/abs/2606.02628",
        "review_status": "new_finding",
        "kind": "primary_paper",
    },
    {
        "source_id": "cross_block",
        "version": "arXiv:2609.14934v2 (2026-09-16)",
        "url": "https://arxiv.org/abs/2609.14934v2",
        "review_status": "new_finding",
        "kind": "primary_paper",
    },
    {
        "source_id": "faithbench",
        "version": "NAACL 2025; release cf89797d82812c23b5d5e5c121f1d9b8983bbbce",
        "url": "https://github.com/vectara/FaithBench/tree/cf89797d82812c23b5d5e5c121f1d9b8983bbbce",
        "review_status": "new_finding",
        "kind": "primary_release",
    },
    {
        "source_id": "expert_aggregation",
        "version": "arXiv:2607.20239v1 (2026-07-22)",
        "url": "https://arxiv.org/html/2607.20239v1",
        "review_status": "rechecked",
        "kind": "primary_paper",
    },
    {
        "source_id": "crane",
        "version": "arXiv:2502.09061 (2025-02)",
        "url": "https://arxiv.org/abs/2502.09061",
        "review_status": "rechecked",
        "kind": "primary_paper",
    },
    {
        "source_id": "on_chip_locality",
        "version": "arXiv:2602.02056 (2026-02)",
        "url": "https://arxiv.org/abs/2602.02056",
        "review_status": "rechecked",
        "kind": "primary_paper",
    },
)

V652_EVIDENCE_PATHS = {
    "static_null": Path("results/experiment_7439_v652_certified_decisions.json"),
    "missing_expert_predictions": Path("results/experiment_7441_v652_decision_audit.json"),
    "parse_status_exception": Path("results/experiment_7442_v652_span_capture.json"),
    "lost_lease_continuity": Path("results/experiment_7443_v652_span_audit.json"),
    "arc_62_action_exposure": Path("results/experiment_7444_v652_arc_supervisor_evidence.json"),
    "persistence_bound": Path("results/experiment_7445_v652_hardware_envelope.json"),
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
    Path("scripts/overdue_priority_lint.py"),
    Path("scripts/adversarial_verify.py"),
    Path("scripts/verdict_row_consistency_lint.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7434_v652_contract_methods.py"),
    Path("research-references.md"),
    STUDY_PATH,
    COMPLETE_PATH,
    DESIGN_PATH,
    SPEC_PATH,
    CAPSTONE_PATH,
    *V652_EVIDENCE_PATHS.values(),
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REPOSITORY_CHECK_NAMES = (
    "roadmap_schema",
    "exclusion_manifest",
    "roadmap_gate_audit",
    "overdue_priority",
)
TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)
WRITTEN_PATHS = (MODULE_PATH, WRAPPER_PATH, TEST_PATH, NOTE_PATH, STUDY_PATH, RESULT_PATH)


def utc_now() -> str:  # pragma: no cover - authentic runtime boundary.
    """Return one real UTC boundary for the measured execution receipt."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush each boundary so a long child process never looks abandoned."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7447] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact bytes so historical evidence cannot drift silently."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_yaml(path: Path) -> JsonDict:
    """Load one YAML mapping and reject malformed or sequence-shaped input."""

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


def resolve_v653_roadmap(root: Path) -> tuple[Path, JsonDict, list[JsonDict]]:
    """Prefer a staged V653 roadmap, then accept only an active V653 roadmap."""

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
        raise ValueError("V653 roadmap authority is unavailable")
    return selected[0], selected[1], candidates


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only fields that both authorities declare independently."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Check that each structured gate reads an earlier declared producer field."""

    raw_tasks = roadmap.get("tasks")
    tasks = raw_tasks if isinstance(raw_tasks, list) else []
    positions = {
        task.get("id"): (index, task)
        for index, task in enumerate(tasks)
        if isinstance(task, Mapping)
    }
    result: dict[str, list[JsonDict]] = {}
    for consumer_index, consumer in enumerate(tasks):
        if not isinstance(consumer, Mapping):  # pragma: no cover - parser rejects this shape.
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
    """Compare complete Markdown and YAML contracts under the exact V653 roster."""

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
    """Read the matching V653 authorities without repairing either source."""

    _path, roadmap, _candidates = resolve_v653_roadmap(root)
    return compare_contract_authorities((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)


def _table_rows(markdown: str) -> tuple[list[str], list[int]]:
    """Locate task rows inside the exact contract table only."""

    lines = markdown.splitlines()
    indices = [
        index for index, line in enumerate(lines) if re.match(r"^\|\s*\d+\s*\|\s*exp\d+", line)
    ]
    return lines, indices


def _mutate_markdown(markdown: str, mutation: str) -> str:
    """Change one Markdown authority field while leaving YAML unchanged."""

    if mutation == "milestone":
        return markdown.replace("**Milestone:** `2026.09.653`", "**Milestone:** `2026.09.652`", 1)
    lines, indices = _table_rows(markdown)
    if mutation == "count":
        del lines[indices[-1]]
    elif mutation == "order":
        lines[indices[0]], lines[indices[1]] = lines[indices[1]], lines[indices[0]]
    else:
        cells = [cell.strip() for cell in lines[indices[0]].strip().strip("|").split("|")]
        positions = {"id": 1, "path": 4, "field": 3}
        replacements = {
            "id": "exp9999-changed",
            "path": "results/changed.json",
            "field": "9",
        }
        cells[positions[mutation]] = replacements[mutation]
        lines[indices[0]] = "| " + " | ".join(cells) + " |"
    return "\n".join(lines) + "\n"


def _mutate_yaml(roadmap: Mapping[str, Any], mutation: str) -> JsonDict:
    """Change one YAML authority field while leaving Markdown unchanged."""

    changed = deepcopy(dict(roadmap))
    tasks = changed["tasks"]
    if mutation == "milestone":
        changed["milestone"] = "2026.09.652"
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
    """Reject count, ID, order, path, field, and milestone drift independently."""

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
                else parse_markdown_contract(markdown_text)
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
    """Keep the thirteen later tasks independent of this advisory contract audit."""

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


def collect_v652_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate all thirteen V652 dispositions without relabeling them."""

    capstone = load_json(root / CAPSTONE_PATH)
    source_rows = capstone.get("task_dispositions")
    if not isinstance(source_rows, list):
        raise ValueError("V652 capstone task_dispositions list required")
    rows: list[JsonDict] = []
    for source in source_rows:
        if not isinstance(source, Mapping):  # pragma: no cover - capstone schema rejects this.
            continue
        observed_path = Path(str(source.get("observed_path")))
        path = root / observed_path
        payload = load_json(path) if path.is_file() else {}
        original_class = source.get("verdict_class")
        original_flag = bool(source.get("flagged_adversarial", False))
        # The capstone cannot read its final top-level flag while building its
        # own current-work row. Authenticate that row's stable class and verdict;
        # retain its original row flag instead of replacing it with the later flag.
        flag_matches = (
            True
            if source.get("source_kind") == "current_work"
            else bool(payload.get("flagged_adversarial", False)) == original_flag
        )
        authenticated = bool(
            path.is_file()
            and payload.get("honest_verdict") == source.get("honest_verdict")
            and payload.get("verdict_class") == original_class
            and flag_matches
        )
        rows.append(
            {
                "unit_id": str(source.get("task_id")),
                "arm": "v652_disposition_authentication",
                "task_id": source.get("task_id"),
                "declared_path": source.get("declared_path"),
                "observed_path": observed_path.as_posix(),
                "source_kind": source.get("source_kind"),
                "sha256": sha256_file(path) if path.is_file() else None,
                "original_status": payload.get("status"),
                "original_honest_verdict": source.get("honest_verdict"),
                "original_verdict_class": original_class,
                "flagged_adversarial": original_flag,
                "raw_rows_available": source.get("raw_rows_available"),
                "authenticated": authenticated,
                "evidence_scope": "historical_model_receipts",
                "counted_as_current_invocation": False,
            }
        )
    return rows


def completion_archive_state(root: Path) -> JsonDict:
    """Separate the dated V653 planning observation from the later archive update."""

    design = (root / DESIGN_PATH).read_text(encoding="utf-8")
    planning_lag = bool(
        re.search(r"During planning\s+the\s+completed archive ended at V651", design)
    )
    complete = load_yaml(root / COMPLETE_PATH)
    milestones = complete.get("milestones")
    milestone_rows = milestones if isinstance(milestones, list) else []
    current = (
        str(milestone_rows[-1].get("id"))
        if milestone_rows and isinstance(milestone_rows[-1], Mapping)
        else None
    )
    return {
        "path": COMPLETE_PATH.as_posix(),
        "planning_authority_path": DESIGN_PATH.as_posix(),
        "planning_latest_milestone": "2026.09.651" if planning_lag else None,
        "planning_lag_recorded": planning_lag,
        "current_latest_milestone": current,
        "history_rewritten": False,
        "principle": "Keep the dated planning observation separate from the later archive state.",
    }


def build_v652_evidence_rows(root: Path) -> list[JsonDict]:
    """Read six distinct V652 causes from their original terminal artifacts."""

    artifacts = {key: load_json(root / path) for key, path in V652_EVIDENCE_PATHS.items()}
    lease_summary = artifacts["lost_lease_continuity"].get("gate_check_summary") or {}
    lease_observed = deepcopy(lease_summary.get("observed") or {})
    evidence = (
        (
            "static_null",
            "exp7439-certified-decisions",
            artifacts["static_null"].get("honest_verdict"),
            "honest_verdict",
        ),
        (
            "missing_expert_predictions",
            "exp7441-decision-audit",
            deepcopy(
                (artifacts["missing_expert_predictions"].get("online_audit") or {}).get("errors")
            ),
            "online_audit.errors",
        ),
        (
            "parse_status_exception",
            "exp7442-span-capture",
            artifacts["parse_status_exception"].get("producer_runtime_error"),
            "producer_runtime_error",
        ),
        (
            "lost_lease_continuity",
            "exp7443-span-audit",
            lease_observed,
            "gate_check_summary.observed",
        ),
        (
            "arc_62_action_exposure",
            "exp7444-arc-supervisor-evidence",
            [row.get("actions") for row in artifacts["arc_62_action_exposure"].get("rows", [])],
            "rows[*].actions",
        ),
        (
            "persistence_bound",
            "exp7445-hardware-envelope",
            (artifacts["persistence_bound"].get("rows") or [{}])[0].get(
                "observed_unaccelerated_fraction"
            ),
            "rows[0].observed_unaccelerated_fraction",
        ),
    )
    rows: list[JsonDict] = []
    for evidence_id, source_task, observed, field in evidence:
        path = V652_EVIDENCE_PATHS[evidence_id]
        rows.append(
            {
                "unit_id": evidence_id,
                "arm": "v652_cause_authentication",
                "evidence_id": evidence_id,
                "source_task": source_task,
                "path": path.as_posix(),
                "field": field,
                "observed": observed,
                "sha256": sha256_file(root / path),
                "authenticated": observed is not None,
                "history_rewritten": False,
            }
        )
    return rows


def _direct_fetch(source: Mapping[str, str]) -> JsonDict:  # pragma: no cover - network boundary.
    """Read a small primary-page prefix and preserve access failure as data."""

    request = urllib.request.Request(
        source["url"], headers={"User-Agent": "carnot-v653-contract-audit/1.0"}
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


def check_selected_sources(fetcher: SourceFetcher = _direct_fetch) -> list[JsonDict]:
    """Check each of the six frozen primary sources exactly once."""

    return [_check_one_source(source, fetcher) for source in SELECTED_SOURCES]


def method_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Map each selected method to its primary source, task hooks, and limit."""

    by_id = {str(row.get("source_id")): row for row in source_rows}

    def method(
        source_id: str,
        name: str,
        hooks: list[str],
        boundary: str,
        controls: list[str],
        **extra: Any,
    ) -> JsonDict:
        source = by_id.get(source_id, {})
        return {
            "unit_id": name,
            "arm": "method_ingestion",
            "method": name,
            "source_id": source_id,
            "primary_url": source.get("url"),
            "source_version": source.get("version"),
            "review_status": source.get("review_status"),
            "access_state": source.get("access_state"),
            "task_hooks": hooks,
            "controls": controls,
            "evidence_boundary": boundary,
            "external_claim_is_local_measurement": False,
            **extra,
        }

    return [
        method(
            "hidden_probe",
            "hidden_representation_probing",
            ["exp7452-source-embeddings", "exp7453-energy-calibration"],
            "A final-layer GGUF vector is not a replication of intermediate-layer NF4 probes.",
            ["matched_linear_control", "authenticated_layer_and_pooling"],
        ),
        method(
            "cross_block",
            "cross_block_conditioning",
            ["exp7449-source-protocol", "exp7453-energy-calibration"],
            "The source-plus-answer ablation borrows a design, not the paper's transfer claim.",
            ["answer_only", "source_plus_answer", "shuffled_source"],
        ),
        method(
            "faithbench",
            "faithbench_challenge_corpus",
            ["exp7449-source-protocol", "exp7453-energy-calibration", "exp7455-decision-audit"],
            "Disagreement-selected examples do not estimate deployment prevalence or certify safety.",
            ["external_only", "group_isolation", "label_blind_selection"],
            challenge_revision="cf89797d82812c23b5d5e5c121f1d9b8983bbbce",
            predictor_exclusions=["detector_predictions", "annotations"],
        ),
        method(
            "expert_aggregation",
            "delayed_expert_losses",
            ["exp7450-prediction-ledger", "exp7454-continuous-learning", "exp7455-decision-audit"],
            "Carnot inherits no delayed-feedback theorem without a separate derivation.",
            ["prediction_before_feedback", "independent_loss_replay"],
        ),
        method(
            "crane",
            "compact_output_semantics",
            ["exp7448-capture-lifecycle", "exp7451-span-capture", "exp7456-extraction-audit"],
            "Complete syntax and literal span recovery are not semantic correctness.",
            ["complete_json", "literal_reconstruction", "semantic_coverage"],
        ),
        method(
            "on_chip_locality",
            "on_chip_locality",
            ["exp7458-durable-updates", "exp7459-board-continuity"],
            "Host durability measurements are not FPGA results or vendor performance evidence.",
            ["whole_service_cost", "durable_acknowledgement", "crash_recovery"],
        ),
    ]


def build_repository_check_plan(
    root: Path, roadmap_path: Path
) -> list[validation_scope.CommandSpec]:
    """Run four shipped repository guards against the selected V653 authority."""

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
            "exclusion_manifest",
            (python, "-u", "scripts/exclusion_manifest_lint.py", selected),
            "selected_roadmap",
        ),
        validation_scope.CommandSpec(
            "roadmap_gate_audit",
            (python, "-u", "scripts/audit_roadmap_gates.py", selected),
            "selected_roadmap",
        ),
        validation_scope.CommandSpec(
            "overdue_priority", (python, "-u", "-c", overdue_code, selected), "selected_roadmap"
        ),
    ]


def unresolved_obligations(root: Path) -> list[JsonDict]:
    """Keep deferred work visible without modifying the prohibited conductor."""

    return [
        {
            "obligation_id": "conductor_size_gate",
            "state": "current_task_forbidden",
            "prohibited_path": "scripts/research_conductor.py",
            "current_task_action": "none",
            "path_exists": (root / "scripts/research_conductor.py").is_file(),
        },
        {
            "obligation_id": "roadmap_activation",
            "state": "not_authorized",
            "prohibited_path": ACTIVE_ROADMAP_PATH.as_posix(),
            "current_task_action": "read_only",
            "path_exists": (root / ACTIVE_ROADMAP_PATH).is_file(),
        },
    ]


def collect_preconditions(
    root: Path, roadmap_path: Path, priors: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Name every exact local input and observed identity before reduction."""

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
    rows.extend(
        [
            {
                "check": "driving_requirement",
                "upstream": SPEC_PATH.as_posix(),
                "path": SPEC_PATH.as_posix(),
                "field": "REQ-*",
                "expected": "REQ-REPORT-7447",
                "observed": "REQ-REPORT-7447" if "REQ-REPORT-7447" in spec else None,
                "passed": "REQ-REPORT-7447" in spec,
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
                "check": "v652_disposition_authentication",
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


def _source_hashes(
    root: Path, roadmap_path: Path, priors: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    """Bind current protocol bytes and typed historical receipt bytes."""

    hashes: dict[str, JsonDict] = {}
    current_paths = (
        *SOURCE_INPUT_PATHS,
        roadmap_path.relative_to(root),
        MODULE_PATH,
        WRAPPER_PATH,
        TEST_PATH,
    )
    for relative in current_paths:
        path = root / relative
        if path.is_file():
            hashes[relative.as_posix()] = {
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "evidence_type": "current_protocol_or_input_bytes",
            }
    for row in priors:
        path = str(row.get("observed_path") or "")
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


def receipts_pass(receipts: object, names: Sequence[str]) -> bool:
    """Require one successful untimed receipt for every exact command name."""

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
    """Keep gate values plain while recording their category and purpose."""

    return {
        "check": check,
        "category": category,
        "operator": operator,
        "expected": deepcopy(expected),
        "observed": deepcopy(observed),
        "passed": passed,
        "principle": principle,
    }


def failure_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name exact operands for every failed prerequisite or acceptance gate."""

    failures = [
        {
            "upstream": row.get("upstream", row.get("check")),
            "path": row.get("path", "authenticated_inputs"),
            "check": row.get("check"),
            "field": row.get("field", "passed"),
            "operator": row.get("operator", "=="),
            "expected": deepcopy(row.get("expected")),
            "observed": deepcopy(row.get("observed")),
            "category": row.get("category", "precondition"),
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
    """Explain each top-level field without wrapping executable scalar values."""

    principles = {
        "schema": "Use a versioned top-level schema and exact experiment_id, milestone and terminal status.",
        "run_date": "Use 20260920; retain actual UTC start/end and monotonic duration with boot/segment identity.",
        "preconditions_checked": "Name the actual source paths, resources, identity and observed gate values before dependent work.",
        "MODEL_SPECS": "Name unsloth/Qwen3.8-27B-GGUF for current LLM work; use [] because this task has none.",
        "model_invoked": "Attempted current model work is distinct from archived or scripted model-shaped events.",
        "invocation_counts": "Balance loads, forwards and generations across every terminal state.",
        "inference_substrate": "Use a truthful string; keep device details and historical evidence in typed sidecars.",
        "inference_substrate_class": "Declare actual aggregation work without simulated execution or padded time.",
        "execution_venue": "Use host and keep external-device evidence separate from current compute.",
        "duration_s": "Measure real current work and separate computation from validation.",
        "phase_spans": "Bind phase timings, progress boundaries, completed units and checkpoints.",
        "random_seed": "Freeze contract-mutation randomness; no fitting or resampling occurs.",
        "reproducibility_checksum": "Bind code, protocol, immutable inputs, rows and exact validation scope.",
        "source_artifact_hashes": "Preserve exact upstream bytes, original classes and flags.",
        "rows": "Retain each task, mutation, evidence, source and method unit, including failures.",
        "sample_size_budget": "Separate planned, attempted, completed, failed, censored and unstarted units.",
        "acceptance_gate_results": "Name check, category, operator, expected, observed, passed and principle.",
        "gate_check_summary": "Name exact upstream, path, check, field, expected and observed for each failure.",
        "verifier_is_oracle": "False because this task audits records and does not score model answers.",
        "honest_verdict": "Use complete_ for finished work and a specific blocked_* for external absence.",
        "verdict_class": "Use the closed enum; partial is only unfinished owned retryable work.",
        "flagged_adversarial": "Preserve critical findings and exclude flagged science from readiness.",
        "validation_receipts": "Record scoped argv, environment, exit, duration and log hash for every reader.",
        "field_principles": "Explain field intent here while gate fields remain bare scalars.",
        "promotion_score": "Always zero because this milestone authorizes no rollout or publication.",
        "contract_ready_score": "One only when the exact fourteen-task contract and required checks pass; it is advisory.",
        "method_rows": "Tie each selected method to its primary URL, evidence boundary and task.",
        "unresolved_obligations": "Keep deferred and user-forbidden work visible.",
    }
    return {
        field: principles.get(field, "Keep this field plain, reviewable, and checksum-bound.")
        for field in fields
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash every artifact field except the checksum slot itself."""

    payload = deepcopy(dict(artifact))
    payload.pop("reproducibility_checksum", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide deterministic spans for pure reducer tests."""

    return [
        {
            "phase": phase,
            "start_s": index * 0.1,
            "end_s": (index + 1) * 0.1,
            "duration_s": 0.1,
            "completed_units": 0,
            "heartbeat_count": 0,
            "checkpoint": "test_fixture",
        }
        for index, phase in enumerate(
            ("preconditions", "model_load", "generation", "contract", "sources", "validation")
        )
    ]


def _terminal_state(
    preconditions_passed: bool, validation_passed: bool, contract_ready: bool
) -> tuple[str, str, str]:
    """Classify terminal state without treating advisory science as readiness."""

    if not preconditions_passed:
        return (
            "blocked_missing_authenticated_v653_input",
            "blocked_missing_authenticated_v653_input",
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
            "complete_disqualified_v653_contract_authority",
            "complete_disqualified_v653_contract_authority",
            "disqualified",
        )
    return (
        "complete_advisory_v653_contract_and_methods",
        "complete_null_v653_contract_methods_ingested",
        "null",
    )


def _acceptance_gates(
    contract: Mapping[str, Any],
    mutations: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    evidence_rows: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    methods: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> list[JsonDict]:
    """Reduce validity gates without converting source access into benefit."""

    return [
        _gate(
            "exact_v653_contract",
            "contract_validity",
            "==",
            True,
            contract.get("passed"),
            contract.get("passed") is True,
            "Both authorities must agree on all fourteen tasks and structured gates.",
        ),
        _gate(
            "contract_mutation_controls",
            "contract_validity",
            "==",
            12,
            sum(
                row.get("rejected") is True and row.get("other_authority_readable") is True
                for row in mutations
            ),
            len(mutations) == 12
            and all(
                row.get("rejected") is True and row.get("other_authority_readable") is True
                for row in mutations
            ),
            "Each requested drift must fail in one authority without corrupting the other.",
        ),
        _gate(
            "v652_dispositions",
            "historical_evidence_validity",
            "==",
            13,
            sum(row.get("authenticated") is True for row in dispositions),
            len(dispositions) == 13
            and all(row.get("authenticated") is True for row in dispositions),
            "Original V652 classes and flags must match exact terminal bytes.",
        ),
        _gate(
            "v652_separate_causes",
            "historical_evidence_validity",
            "==",
            6,
            sum(row.get("authenticated") is True for row in evidence_rows),
            len(evidence_rows) == 6
            and all(row.get("authenticated") is True for row in evidence_rows),
            "Null, missing evidence, runtime fault, lease loss, exposure, and bound stay separate.",
        ),
        _gate(
            "bounded_method_ingestion",
            "method_accounting",
            "==",
            {"sources": 6, "methods": 6},
            {"sources": len(sources), "methods": len(methods)},
            len(sources) == 6 and len(methods) == 6,
            "An inaccessible page remains a recorded access outcome, not missing science.",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            "==",
            True,
            validation.get("affected_checks_passed"),
            validation.get("affected_checks_passed") is True,
            "Only the frozen affected manifest controls implementation validity.",
        ),
        _gate(
            "repository_guards",
            "required_validation",
            "==",
            True,
            validation.get("repository_checks_passed"),
            validation.get("repository_checks_passed") is True,
            "Shipped schema, exclusion, gate, and overdue checks keep their meanings.",
        ),
        _gate(
            "terminal_readers",
            "required_validation",
            "==",
            True,
            validation.get("terminal_checks_passed"),
            validation.get("terminal_checks_passed") is True,
            "Cold replay, independent reduction, adversarial review, and row lint must pass.",
        ),
    ]


def build_artifact(
    root: Path,
    roadmap_path: Path,
    contract: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    dispositions: Sequence[Mapping[str, Any]],
    v652_evidence_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    started_monotonic_ns: int,
    ended_monotonic_ns: int,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one schema-complete advisory artifact from independently reducible rows."""

    methods = method_rows(source_rows)
    branches = independent_branch_schedule(contract)
    preconditions = collect_preconditions(root, roadmap_path, dispositions)
    gates = _acceptance_gates(
        contract,
        mutation_rows,
        dispositions,
        v652_evidence_rows,
        source_rows,
        methods,
        validation,
    )
    preconditions_passed = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    validation_passed = all(
        row.get("passed") is True for row in gates if row.get("category") == "required_validation"
    )
    contract_ready = int(contract.get("passed") is True and validation_passed)
    status, honest, verdict = _terminal_state(
        preconditions_passed, validation_passed, bool(contract_ready)
    )
    rows = [
        *[
            deepcopy(dict(row)) | {"row_kind": "task_contract"}
            for row in contract.get("contract_rows", [])
        ],
        *[deepcopy(dict(row)) | {"row_kind": "contract_mutation"} for row in mutation_rows],
        *[deepcopy(dict(row)) | {"row_kind": "v652_disposition"} for row in dispositions],
        *[deepcopy(dict(row)) | {"row_kind": "v652_evidence"} for row in v652_evidence_rows],
        *[deepcopy(dict(row)) | {"row_kind": "source_access"} for row in source_rows],
        *[deepcopy(dict(row)) | {"row_kind": "method_ingestion"} for row in methods],
        *[deepcopy(dict(row)) | {"row_kind": "branch_schedule"} for row in branches],
    ]
    duration_s = (ended_monotonic_ns - started_monotonic_ns) / 1_000_000_000
    receipts = [deepcopy(dict(row)) for row in validation.get("validation_receipts", [])]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "experiment": 7447,
        "milestone": MILESTONE,
        "phase": 1,
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "started_monotonic_ns": started_monotonic_ns,
        "ended_monotonic_ns": ended_monotonic_ns,
        "clock_identity": {"clock": "time.monotonic_ns", "segment": "single_host_process"},
        "status": status,
        "title": "Bind fourteen tasks and ingest source-conditioning methods",
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
            "model_load": 0.0,
            "forward": 0.0,
            "generation": 0.0,
            "numeric_computation": 0.0,
            "validation": sum(
                float(row.get("duration_s", 0.0))
                for row in phase_spans
                if row.get("phase") in {"validation", "terminal_validation"}
            ),
        },
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": _source_hashes(root, roadmap_path, dispositions),
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
                "v652_dispositions": len(dispositions),
                "v652_causes": len(v652_evidence_rows),
                "source_endpoints": len(source_rows),
                "method_mappings": len(methods),
                "independent_branches": len(branches),
            },
            "stop_rule": "Stop after fourteen tasks, twelve mutations, thirteen V652 dispositions, six causes, six pages, six methods, and thirteen branch rows.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": failure_summary([*preconditions, *gates]),
        "verifier_is_oracle": False,
        "honest_verdict": honest,
        "verdict_class": verdict,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "field_principles": {},
        "promotion_score": 0,
        "contract_ready_score": contract_ready,
        "method_ingestion_complete_score": int(len(source_rows) == 6 and len(methods) == 6),
        "selected_roadmap_path": roadmap_path.relative_to(root).as_posix(),
        "task_contract_rows": deepcopy(list(contract.get("contract_rows", []))),
        "contract_comparison": deepcopy(dict(contract)),
        "contract_mutation_rows": [deepcopy(dict(row)) for row in mutation_rows],
        "branch_schedule_rows": branches,
        "v652_dispositions": [deepcopy(dict(row)) for row in dispositions],
        "v652_evidence_rows": [deepcopy(dict(row)) for row in v652_evidence_rows],
        "completion_archive_state": completion_archive_state(root),
        "source_access_rows": [deepcopy(dict(row)) for row in source_rows],
        "method_rows": methods,
        "small_ebm_training": {
            "performed": False,
            "attempted": 0,
            "completed": 0,
            "principle": "This audit trains no current energy head.",
        },
        "historical_inference_sidecars": [
            {
                "task_id": row.get("task_id"),
                "path": row.get("observed_path"),
                "sha256": row.get("sha256"),
                "scope": "historical_model_receipts",
                "original_verdict_class": row.get("original_verdict_class"),
                "flagged_adversarial": row.get("flagged_adversarial"),
                "counted_as_current_invocation": False,
            }
            for row in dispositions
        ],
        "unresolved_obligations": unresolved_obligations(root),
        "numbered_e2e_applicable": [],
        "capability_e2e": ["declared entrypoint", "fresh-process cold artifact replay"],
        "roadmap_activated": False,
        "production_defaults_changed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def independent_reduce(artifact: Mapping[str, Any], *, require_terminal: bool = True) -> JsonDict:
    """Recompute contract readiness and terminal class from raw artifact rows."""

    contract_rows = artifact.get("task_contract_rows")
    contract_valid = bool(
        isinstance(contract_rows, list)
        and len(contract_rows) == 14
        and [row.get("unit_id") for row in contract_rows if isinstance(row, Mapping)]
        == list(EXPECTED_TASK_IDS)
        and all(row.get("passed") is True for row in contract_rows if isinstance(row, Mapping))
    )
    mutations = artifact.get("contract_mutation_rows")
    mutations_valid = bool(
        isinstance(mutations, list)
        and len(mutations) == 12
        and all(
            row.get("rejected") is True and row.get("other_authority_readable") is True
            for row in mutations
            if isinstance(row, Mapping)
        )
    )
    dispositions = artifact.get("v652_dispositions")
    dispositions_valid = bool(
        isinstance(dispositions, list)
        and len(dispositions) == 13
        and all(
            row.get("authenticated") is True for row in dispositions if isinstance(row, Mapping)
        )
    )
    evidence = artifact.get("v652_evidence_rows")
    evidence_valid = bool(
        isinstance(evidence, list)
        and len(evidence) == 6
        and all(row.get("authenticated") is True for row in evidence if isinstance(row, Mapping))
    )
    sources = artifact.get("source_access_rows")
    methods = artifact.get("method_rows")
    methods_valid = (
        isinstance(sources, list)
        and len(sources) == 6
        and isinstance(methods, list)
        and len(methods) == 6
    )
    receipts = artifact.get("validation_receipts")
    affected_valid = receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES)
    repository_valid = receipts_pass(receipts, REPOSITORY_CHECK_NAMES)
    terminal_valid = True if not require_terminal else receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    required_valid = bool(
        mutations_valid
        and dispositions_valid
        and evidence_valid
        and methods_valid
        and affected_valid
        and repository_valid
        and terminal_valid
    )
    preconditions = artifact.get("preconditions_checked")
    preconditions_valid = bool(
        isinstance(preconditions, list)
        and preconditions
        and all(row.get("passed") is True for row in preconditions if isinstance(row, Mapping))
    )
    ready = int(contract_valid and required_valid)
    status, honest, verdict = _terminal_state(preconditions_valid, required_valid, bool(ready))
    return {
        "contract_ready_score": ready,
        "method_ingestion_complete_score": int(methods_valid),
        "required_checks_passed": required_valid,
        "status": status,
        "honest_verdict": honest,
        "verdict_class": verdict,
    }


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash every declared source and typed historical receipt."""

    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return False
    try:
        roadmap_path, _roadmap, _candidates = resolve_v653_roadmap(root)
        expected = _source_hashes(root, roadmap_path, collect_v652_dispositions(root))
    except (OSError, ValueError, json.JSONDecodeError):  # pragma: no cover - defensive cold I/O.
        return False
    return hashes == expected


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, authorities, evidence, scores, principles, and hashes."""

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

    try:
        roadmap_path, roadmap, _candidates = resolve_v653_roadmap(root)
        markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
        contract = compare_contract_authorities(markdown, roadmap)
        expected_mutations = run_contract_mutation_controls(markdown, roadmap)
    except (OSError, ValueError, json.JSONDecodeError):  # pragma: no cover - defensive cold I/O.
        roadmap_path, contract, expected_mutations = root / ACTIVE_ROADMAP_PATH, {}, []
    if artifact.get("task_contract_rows") != contract.get("contract_rows"):
        errors.append("task_contract_rows_mismatch")
    if artifact.get("contract_comparison") != contract:
        errors.append("contract_comparison_mismatch")
    if artifact.get("contract_mutation_rows") != expected_mutations:
        errors.append("contract_mutations_invalid")

    expected_dispositions = collect_v652_dispositions(root)
    if artifact.get("v652_dispositions") != expected_dispositions:
        errors.append("v652_dispositions_mismatch")
    expected_evidence = build_v652_evidence_rows(root)
    if artifact.get("v652_evidence_rows") != expected_evidence:
        errors.append("v652_evidence_mismatch")
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
    if artifact.get("completion_archive_state") != completion_archive_state(root):
        errors.append("archive_state_mismatch")
    if artifact.get("branch_schedule_rows") != independent_branch_schedule(contract):
        errors.append("branch_schedule_mismatch")

    receipts = artifact.get("validation_receipts")
    if not receipts_pass(receipts, validation_scope.REQUIRED_CHECK_NAMES):
        errors.append("affected_validation_invalid")
    if not receipts_pass(receipts, REPOSITORY_CHECK_NAMES):
        errors.append("repository_validation_invalid")
    if require_terminal and not receipts_pass(receipts, TERMINAL_CHECK_NAMES):
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
    del roadmap_path
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 file-scoped command plan for this experiment."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets and command drift before a child starts."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _phase_span(  # pragma: no cover - authentic timing boundary.
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
    """Build cold replay, independent reduction, and unchanged terminal readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7447_v653_contract_methods import validate_artifact;"
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
    """Write one durable V653 note and one idempotent studying-ledger marker."""

    methods = method_rows(source_rows)
    access_lines = [
        f"- `{row['source_id']}`: `{row['access_state']}` ({row.get('http_status')}); "
        f"review `{row['review_status']}`; {row['url']}"
        for row in source_rows
    ]
    method_lines = [
        f"- **{row['method']}** -> {', '.join(row['task_hooks'])}. "
        f"Boundary: {row['evidence_boundary']}"
        for row in methods
    ]
    note = "\n".join(
        [
            "# V653 method ingestion",
            "",
            "Date: 2026-09-20. Scope: advisory contract accounting and bounded source ingestion.",
            "External findings are method inputs. They are not Carnot measurements.",
            "",
            "## Primary access receipts",
            "",
            *access_lines,
            "",
            "## Method-to-task mapping",
            "",
            *method_lines,
            "",
            "## Preserved limits",
            "",
            "FaithBench stays an external challenge corpus. Detector predictions and annotations",
            "are excluded from predictor features. No roadmap is activated. The unresolved",
            "conductor size obligation remains visible because this task cannot modify",
            "`scripts/research_conductor.py`.",
            "",
        ]
    )
    note_path = root / NOTE_PATH
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text(note, encoding="utf-8")

    marker = "<!-- EXP7447-V653-METHOD-INGESTION -->"
    study_path = root / STUDY_PATH
    studying = study_path.read_text(encoding="utf-8")
    if marker not in studying:
        addition = "\n".join(
            [
                "",
                marker,
                "## 2026-09-20 Exp7447 — V653 methods — INGESTED",
                "",
                "Hidden-representation probing, cross-block conditioning, delayed expert losses,",
                "compact output semantics, and on-chip locality were mapped to the V653 roster.",
                "FaithBench remains an external challenge corpus. Exact access outcomes and",
                "evidence limits are in `docs/research-notes/v653-method-ingestion.md`.",
                "",
            ]
        )
        study_path.write_text(studying.rstrip() + "\n" + addition, encoding="utf-8")


def run_experiment(  # pragma: no cover - exercised by the declared capability E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Measure inputs, run scoped checks, and atomically publish the terminal JSON."""

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
    roadmap_path, roadmap, candidates = resolve_v653_roadmap(root)
    dispositions = collect_v652_dispositions(root)
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
    evidence_rows = build_v652_evidence_rows(root)
    contract_units = len(contract.get("contract_rows", [])) + len(mutations) + len(evidence_rows)
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

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7447-validation-", dir="/tmp"))
    affected_plan = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, affected_plan)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{plan_errors}")
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

    validation = {
        "validation_receipts": [*affected, *repository],
        "affected_checks_passed": affected_reduction["passed"],
        "repository_checks_passed": repository_passed,
        "terminal_checks_passed": True,
    }
    candidate = build_artifact(
        root,
        roadmap_path,
        contract,
        mutations,
        dispositions,
        evidence_rows,
        source_rows,
        validation,
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
        evidence_rows,
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
    """Accept only the execution date frozen by the V653 contract."""

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
    """Run V653 contract ingestion or validate one measured candidate."""

    print("[exp7447] phase=startup event=flushed", flush=True)
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

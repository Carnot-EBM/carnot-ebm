"""Build the V645 source and exact fourteen-task contract receipt.

This module adds V645 policy to the shipped independent contract parsers. It
also uses the real conductor reader, so a fixture cannot bypass the gate logic
that later tasks use.

Spec refs: REQ-REPORT-7343 and SCENARIO-REPORT-7343-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Any

import yaml

from carnot.experiment_7179_v633_contract_receipt import (
    _atomic_write_bytes,
    _atomic_write_json,
)
from carnot.experiment_7192_v634_source_contract import _fetch_url, reproducibility_checksum
from carnot.experiment_7329_v644_contract import (
    _field_declared,
    _passing_artifacts,
    _public_task,
    _write_gate_artifacts,
    parse_markdown_contract,
    parse_yaml_contract,
    run_gate_controls as run_six_gate_controls,
    sha256,
    sha256_bytes,
    upstream_precondition,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    reduce_required_checks,
    run_commands,
    run_scoped_validation,
)
from scripts.conductor_gates import evaluate_gates


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]
ValidationRunner = Callable[..., JsonDict]
TerminalRunner = Callable[[Path, Path, Path], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.645"
RUN_DATE = "20260916"
RETRIEVAL_DATE = "2026-09-16"
FIRST_TASK_ID = "exp7343-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7344-executor-fixture",
    "exp7345-arc-resume-check",
    "exp7346-learning-adapter",
    "exp7347-plan-canary",
    "exp7348-plan-capture",
    "exp7349-prospective-learning",
    "exp7350-learning-audit",
    "exp7351-acquisition-prototype",
    "exp7352-acquisition-cost",
    "exp7353-acquisition-audit",
    "exp7354-arc-transfer",
    "exp7355-board-state",
    "exp7356-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
RANDOM_SEED = {
    "development": 7_343_202_609_16,
    "evaluation": 7_343_202_609_17,
    "resampling": 7_343_202_609_18,
}

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
STUDY_PATH = Path("research-studying.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PLAN_PATH = Path("ops/e2e-test-plan.md")
HISTORY_PATH = Path("results/experiment_7329_v644_contract.json")

MODULE_PATH = Path("python/carnot/experiment_7343_v645_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7343_v645_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7343_v645_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7343_v645_contract.json")
RAW_DIR = Path("results/raw/experiment_7343")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7343_v645_contract.json")

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
GATE_AUDIT_PATH = Path("scripts/audit_roadmap_gates.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
PROMPT_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ARC_LINT_PATH = Path("scripts/arc_levelup_guarantee_lint.py")
OVERDUE_LINT_PATH = Path("scripts/overdue_priority_lint.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    REFERENCE_PATH,
    STUDY_PATH,
    EXCLUSION_PATH,
    E2E_PLAN_PATH,
    COMPLETE_PATH,
    SPEC_PATH,
    DESIGN_PATH,
    HISTORY_PATH,
    ROADMAP_SCHEMA_PATH,
    PRIOR_LINT_PATH,
    GATE_AUDIT_PATH,
    EXCLUSION_LINT_PATH,
    PROMPT_PATH_LINT_PATH,
    ARC_LINT_PATH,
    OVERDUE_LINT_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

ZERO_INVOCATION_COUNTS = {
    "model_loads_attempted": 0,
    "model_loads_completed": 0,
    "model_loads_failed": 0,
    "model_loads_cancelled": 0,
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_cancelled": 0,
    "generation_calls_in_flight": 0,
}

ROADMAP_CHECK_NAMES = (
    "roadmap_schema",
    "prior_failure_scope",
    "gate_audit",
    "exclusion_manifest",
    "prompt_paths",
    "arc_generalization",
    "overdue_priority",
)
TERMINAL_CHECK_NAMES = ("independent_reducer", "adversarial", "row_consistency")
ALL_VALIDATION_NAMES = (*ROADMAP_CHECK_NAMES, *REQUIRED_CHECK_NAMES, *TERMINAL_CHECK_NAMES)
GATE_CASES = (
    "passing",
    "zero_score",
    "missing_file",
    "missing_field",
    "missing_verdict_class",
    "disqualified_score_one",
    "quarantined_score_one",
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the record and retain ordinary top-level experiment_id and milestone.",
    "status": "Write a terminal result only after actual work and affected checks.",
    "run_date": "Use 20260916; record real UTC timestamps as well.",
    "preconditions_checked": "Record each actual input/resource check before dependent work.",
    "MODEL_SPECS": "List actual intended model identities; LLM tasks include unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True for any attempted current model load or generation, including failures.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled and in-flight operations.",
    "inference_substrate": "Declare actual computation; historical model receipts are not current inference.",
    "inference_substrate_class": "Use the closed duration class matching the actual run.",
    "execution_venue": "Use host; this milestone makes no new board-execution claim.",
    "duration_s": "Measure monotonic time; never wait merely to pass a duration floor.",
    "phase_spans": "Measure disjoint load, generation, evaluation, test and write spans.",
    "random_seed": "Freeze development, evaluation and resampling seeds before outcomes.",
    "reproducibility_checksum": "Bind code, settings, inputs, evaluator identity and raw evidence.",
    "source_artifact_hashes": "Authenticate exact producers and current same-milestone paths.",
    "rows": "Keep every comparative unit, arm, metric, cost, failure and censoring disposition.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and stopping rules.",
    "acceptance_gate_results": "Each gate records expected, observed, passed and its principle.",
    "gate_check_summary": "Every blocked_* names upstream, failed check, exact artifact field, expected and observed value.",
    "verifier_is_oracle": "True when the executor defines correctness; separate code does not remove circularity.",
    "honest_verdict": "Completed work starts complete_ or complete:; external absence starts blocked_ with its failed check.",
    "verdict_class": "Closed enum: positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "flagged_adversarial": "Set false only after current verification; a critical finding sets true and prevents promotion.",
    "validation_receipts": "Retain exact command, scope, exit code, elapsed time and log hash, including failures.",
    "repository_health": "Preserve dated unrelated failures separately from affected required validation.",
    "field_principles": "Explain fields separately; do not wrap numeric gates or ordinary dictionaries.",
    "contract_complete_score": "One means exact file parity and passing contract controls, not scientific value.",
    "source_method_map": "Bind each paper assumption to a named task and primary URL.",
    "contract_rows": "Preserve all fourteen independently parsed task rows.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

SOURCE_IDS = ("2609.12267", "2604.13283", "2607.20792", "2607.29549")
ACCESS_SPECS: tuple[JsonDict, ...] = (
    {
        "source_id": "2609.12267",
        "url": "https://arxiv.org/abs/2609.12267",
        "publication_or_version_date": "2026-09-10",
    },
    {
        "source_id": "2604.13283",
        "url": "https://arxiv.org/abs/2604.13283",
        "publication_or_version_date": "2026-04-14",
    },
    {
        "source_id": "2607.20792",
        "url": "https://arxiv.org/abs/2607.20792",
        "publication_or_version_date": "2026-07-22",
    },
    {
        "source_id": "2607.29549",
        "url": "https://arxiv.org/abs/2607.29549",
        "publication_or_version_date": "2026-07-31",
    },
)
SOURCE_METHODS: tuple[JsonDict, ...] = (
    {
        "source_id": "2609.12267",
        "method": "Compare bounded candidate elimination by full acquisition time and exact query count.",
        "task_mapping": ["exp7351-acquisition-prototype"],
        "limitations": "The neural oracle can mislabel queries. Target scopes are supplied during training, so it is not exact authority.",
    },
    {
        "source_id": "2604.13283",
        "method": "Retain conservative structural additions and charge every exact executor query.",
        "task_mapping": ["exp7349-prospective-learning", "exp7351-acquisition-prototype"],
        "limitations": "The restricted language does not cover arbitrary hidden rules. A shared executor remains a circular authority.",
    },
    {
        "source_id": "2607.20792",
        "method": "Keep prediction read-only and commit verified updates between distinct future requests.",
        "task_mapping": ["exp7349-prospective-learning"],
        "limitations": "A bounded procedural recall study does not establish general continual learning.",
    },
    {
        "source_id": "2607.29549",
        "method": "Record request, execution, delivery, resumed reasoning, and later action separately.",
        "task_mapping": ["exp7354-arc-transfer"],
        "limitations": "Mathematical tool-flow results do not establish ARC transfer or later-action value.",
    },
)


def progress(phase: int, event: str, detail: str) -> None:
    """Flush each boundary so a long validation run remains observable."""

    print(f"[exp7343] phase={phase} event={event} {detail}", flush=True)


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Use staged V645 bytes when present, then fall back to active V645 bytes."""

    selected: tuple[Path, JsonDict, bytes] | None = None
    candidates: list[JsonDict] = []
    for relative in (NEXT_ROADMAP_PATH, ACTIVE_ROADMAP_PATH):
        path = root / relative
        try:
            content = path.read_bytes()
            value = yaml.safe_load(content.decode("utf-8"))
            if not isinstance(value, dict) or not value:
                raise ValueError(f"YAML mapping required at {path}")
            observed: object = value.get("milestone")
            matches = observed == MILESTONE
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            content, value, observed, matches = b"", None, f"{type(exc).__name__}: {exc}", False
        candidates.append(
            {
                "check": f"yaml_candidate:{relative}",
                "path": str(relative),
                "upstream": str(relative),
                "artifact_field": "milestone",
                "expected_value": MILESTONE,
                "observed_value": observed,
                "available": matches,
                "blocking": False,
            }
        )
        if selected is None and matches and value is not None:
            selected = (relative, value, content)
    if selected is None:
        return None, None, None, candidates
    return *selected, candidates


def validate_gate_declarations(roadmap: Mapping[str, Any]) -> JsonDict:
    """Require earlier producers to promise each field that consumers read."""

    tasks = roadmap.get("tasks", [])
    by_id = {
        task.get("id"): (index, task)
        for index, task in enumerate(tasks)
        if isinstance(task, Mapping)
    }
    rows: list[JsonDict] = []
    for consumer_index, consumer in enumerate(tasks):
        if not isinstance(consumer, Mapping):
            continue
        for gate in consumer.get("gated_on") or []:
            upstream = gate.get("upstream")
            producer_entry = by_id.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            declared = _field_declared(str(producer.get("prompt", "")), gate.get("artifact_field"))
            passed = bool(
                producer_index is not None
                and producer_index < consumer_index
                and declared
                and upstream != FIRST_TASK_ID
            )
            rows.append(
                {
                    "consumer": consumer.get("id"),
                    "upstream": upstream,
                    "artifact_field": gate.get("artifact_field"),
                    "producer_precedes_consumer": producer_index is not None
                    and producer_index < consumer_index,
                    "declared_verbatim": declared,
                    "receipt_is_not_upstream": upstream != FIRST_TASK_ID,
                    "passed": passed,
                }
            )
    return {"rows": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare V645 rows without using the older fixed-milestone evaluator."""

    markdown = parse_markdown_contract(markdown_text)
    parsed_yaml = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    declarations = (
        validate_gate_declarations(yaml_document) if isinstance(yaml_document, Mapping) else {}
    )
    declaration_rows = declarations.get("rows", [])
    width = max(EXPECTED_TASK_COUNT, len(markdown_tasks), len(yaml_tasks))
    rows: list[JsonDict] = []
    for index in range(width):
        expected = markdown_tasks[index] if index < len(markdown_tasks) else None
        observed = yaml_tasks[index] if index < len(yaml_tasks) else None
        task_id = (observed or expected or {}).get("id")
        task_declarations = [row for row in declaration_rows if row.get("consumer") == task_id]
        checks = {
            "order": bool(
                expected
                and observed
                and expected.get("order") == observed.get("order") == index + 1
            ),
            "id": bool(expected and observed and expected.get("id") == observed.get("id")),
            "title": bool(expected and observed and expected.get("title") == observed.get("title")),
            "phase": bool(expected and observed and expected.get("phase") == observed.get("phase")),
            "substrate": bool(
                expected and observed and expected.get("substrate") == observed.get("substrate")
            ),
            "deliverable": bool(
                expected and observed and expected.get("deliverable") == observed.get("deliverable")
            ),
            "gates": bool(expected and observed and expected.get("gates") == observed.get("gates")),
            "gate_fields_declared": all(row.get("passed") is True for row in task_declarations),
        }
        failures = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "unit_id": task_id,
                "arm": "markdown_vs_yaml",
                "order": index + 1,
                "markdown": _public_task(expected),
                "yaml": _public_task(observed),
                "checks": checks,
                "metrics": {
                    "matched_field_count": sum(checks.values()),
                    "field_count": len(checks),
                },
                "costs": {"model_loads": 0, "generation_calls": 0, "network_requests": 0},
                "failures": failures,
                "abstentions": [],
                "censored": False,
                "passed": not failures,
            }
        )
    markdown_order = [task.get("id") for task in markdown_tasks]
    yaml_order = [task.get("id") for task in yaml_tasks]
    passed = bool(
        markdown.get("milestone") == parsed_yaml.get("milestone") == MILESTONE
        and markdown_order == yaml_order == list(EXPECTED_ID_ORDER)
        and len(rows) == EXPECTED_TASK_COUNT
        and declarations.get("passed") is True
        and all(row["passed"] for row in rows)
    )
    return {
        "markdown_milestone": markdown.get("milestone"),
        "yaml_milestone": parsed_yaml.get("milestone"),
        "markdown_task_rows": [_public_task(task) for task in markdown_tasks],
        "yaml_task_rows": [_public_task(task) for task in yaml_tasks],
        "contract_rows": rows,
        "markdown_id_order": markdown_order,
        "yaml_id_order": yaml_order,
        "gate_declaration_result": declarations,
        "passed": passed,
    }


def run_contract_mutations(
    markdown_text: str, roadmap: Mapping[str, Any], temp_root: Path
) -> list[JsonDict]:
    """Prove the independent comparison rejects each required change."""

    def missing_task(document: JsonDict) -> None:
        document["tasks"] = document["tasks"][:-1]

    def reordered_id(document: JsonDict) -> None:
        document["tasks"][0], document["tasks"][1] = document["tasks"][1], document["tasks"][0]

    def stale_milestone(document: JsonDict) -> None:
        document["milestone"] = "2026.09.644"

    def changed_path(document: JsonDict) -> None:
        document["tasks"][0]["deliverable"] = "results/mutated.json"

    def changed_title(document: JsonDict) -> None:
        document["tasks"][0]["title"] += " changed"

    def changed_phase(document: JsonDict) -> None:
        document["tasks"][0]["phase"] = 4

    def misspelled_gate_field(document: JsonDict) -> None:
        gated = next(task for task in document["tasks"] if task.get("gated_on"))
        gated["gated_on"][0]["artifact_feld"] = gated["gated_on"][0].pop("artifact_field")

    def malformed_operator(document: JsonDict) -> None:
        gated = next(task for task in document["tasks"] if task.get("gated_on"))
        gated["gated_on"][0]["op"] = "contains"

    mutations = (
        ("missing_task", missing_task),
        ("reordered_id", reordered_id),
        ("stale_milestone", stale_milestone),
        ("changed_path", changed_path),
        ("changed_title", changed_title),
        ("changed_phase", changed_phase),
        ("misspelled_gate_field", misspelled_gate_field),
        ("malformed_operator", malformed_operator),
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    for name, mutate in mutations:
        changed = deepcopy(dict(roadmap))
        mutate(changed)
        path = temp_root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
        try:
            observed = evaluate_contract(markdown_text, yaml.safe_load(path.read_text()))["passed"]
            error = None
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            observed = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "mutation": name,
                "fixture_path": str(path),
                "expected": "contract_rejected",
                "observed": observed,
                "error": error,
                "rejected": observed is False,
            }
        )
    return rows


def run_gate_controls(roadmap: Mapping[str, Any], results_dir: Path) -> list[JsonDict]:
    """Add a quarantined score-one case to the six shipped gate controls."""

    base_rows = run_six_gate_controls(roadmap, results_dir)
    edges = [
        (str(task.get("id")), list(task.get("gated_on") or []), local_index, gate)
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
        for local_index, gate in enumerate(task.get("gated_on") or [])
        if isinstance(gate, Mapping)
    ]
    rows: list[JsonDict] = []
    for edge_index, (consumer, consumer_gates, local_index, gate) in enumerate(edges):
        edge_base = [row for row in base_rows if row.get("edge_index") == edge_index]
        if gate.get("artifact_field") == "flagged_adversarial":
            zero_row = next(row for row in edge_base if row.get("case") == "zero_score")
            directory = results_dir / f"edge_{edge_index}_zero_score"
            artifacts = _passing_artifacts(consumer_gates)
            score_gate = next(
                candidate
                for candidate in consumer_gates
                if str(candidate.get("artifact_field", "")).endswith("_score")
            )
            artifacts[str(score_gate["upstream"])][str(score_gate["artifact_field"])] = 0
            _write_gate_artifacts(directory, artifacts)
            single = evaluate_gates({"id": consumer, "gated_on": [dict(gate)]}, directory)
            combined = evaluate_gates({"id": consumer, "gated_on": consumer_gates}, directory)
            intake = [
                upstream_precondition(
                    artifacts.get(str(candidate["upstream"])),
                    str(candidate["artifact_field"]),
                    candidate.get("value"),
                    str(candidate.get("op")),
                )
                for candidate in consumer_gates
            ]
            zero_row.update(
                observed_value=single.gates_evaluated[0].actual,
                conductor_passed=single.passed,
                conductor_reason=single.gates_evaluated[0].reason,
                combined_conductor_passed=combined.passed,
                combined_conductor_summary=combined.summary,
                experiment_precondition_passed=all(row["passed"] for row in intake),
                experiment_precondition_reasons=[row["reason"] for row in intake],
            )
        rows.extend(edge_base)
        progress(
            3,
            "gate_control_start",
            f"edge={edge_index + 1}/{len(edges)} case=quarantined_score_one",
        )
        directory = results_dir / f"edge_{edge_index}_quarantined_score_one"
        artifacts = _passing_artifacts(consumer_gates)
        upstream = str(gate["upstream"])
        target = artifacts[upstream]
        target["flagged_adversarial"] = True
        for candidate in consumer_gates:
            field = str(candidate["artifact_field"])
            if field.endswith("_score"):
                artifacts[str(candidate["upstream"])][field] = 1
        _write_gate_artifacts(directory, artifacts)
        single = evaluate_gates({"id": consumer, "gated_on": [dict(gate)]}, directory)
        combined = evaluate_gates({"id": consumer, "gated_on": consumer_gates}, directory)
        intake = [
            upstream_precondition(
                artifacts.get(str(candidate["upstream"])),
                str(candidate["artifact_field"]),
                candidate.get("value"),
                str(candidate.get("op")),
            )
            for candidate in consumer_gates
        ]
        observed = single.gates_evaluated[0]
        rows.append(
            {
                "edge_index": edge_index,
                "consumer_gate_index": local_index,
                "consumer": consumer,
                "upstream": upstream,
                "case": "quarantined_score_one",
                "artifact_field": gate.get("artifact_field"),
                "operator": gate.get("op"),
                "expected_value": gate.get("value"),
                "observed_value": observed.actual,
                "conductor_passed": single.passed,
                "conductor_reason": observed.reason,
                "combined_conductor_passed": combined.passed,
                "combined_conductor_summary": combined.summary,
                "experiment_precondition_passed": all(row["passed"] for row in intake),
                "experiment_precondition_reasons": [row["reason"] for row in intake],
                "validation_input": True,
                "research_result": False,
            }
        )
        progress(
            3, "gate_control_end", f"edge={edge_index + 1}/{len(edges)} case=quarantined_score_one"
        )
    return rows


def gate_controls_complete(rows: object, gate_count: int) -> bool:
    """Require one pass and six fail-closed cases for every gate edge."""

    if not isinstance(rows, list) or len(rows) != gate_count * len(GATE_CASES):
        return False
    for edge_index in range(gate_count):
        edge = [row for row in rows if row.get("edge_index") == edge_index]
        if [row.get("case") for row in edge] != list(GATE_CASES):
            return False
        expected = [True, False, False, False, False, False, False]
        if [row.get("combined_conductor_passed") for row in edge] != expected:
            return False
        if [row.get("experiment_precondition_passed") for row in edge] != expected:
            return False
    return True


def collect_source_method_map(
    fetcher: Fetcher = _fetch_url,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Refresh four primary pages while retaining every access failure."""

    access_rows: list[JsonDict] = []
    for index, spec in enumerate(ACCESS_SPECS, 1):
        progress(4, "source_request_start", f"completed={index - 1}/4 source={spec['source_id']}")
        try:
            receipt = dict(fetcher(str(spec["url"])))
        except Exception as exc:  # noqa: BLE001 - access failures are measured outcomes.
            receipt = {
                "ok": False,
                "status_code": None,
                "body": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        body = str(receipt.get("body", ""))
        status_code = receipt.get("status_code")
        if receipt.get("ok") is True:
            outcome = "http_success"
        elif status_code == 429:
            outcome = "http_429"
        elif "challenge" in f"{body} {receipt.get('error', '')}".lower():
            outcome = "browser_challenge"
        else:
            outcome = "access_failed"
        access_rows.append(
            {
                **spec,
                "retrieval_date": RETRIEVAL_DATE,
                "timeout_s": 20,
                "status_code": status_code,
                "error": receipt.get("error"),
                "response_sha256": sha256_bytes(body.encode("utf-8")),
                "access_outcome": outcome,
            }
        )
        progress(4, "source_request_end", f"completed={index}/4 outcome={outcome}")
    access_by_id = {row["source_id"]: row for row in access_rows}
    method_map: list[JsonDict] = []
    for method in SOURCE_METHODS:
        access = access_by_id[method["source_id"]]
        method_map.append(
            {
                **deepcopy(method),
                "primary_url": access["url"],
                "publication_or_version_date": access["publication_or_version_date"],
                "retrieval_date": access["retrieval_date"],
                "access_outcome": access["access_outcome"],
                "access_status_code": access["status_code"],
                "access_error": access["error"],
                "local_science_claimed": False,
            }
        )
    return method_map, access_rows


def source_map_complete(rows: object) -> bool:
    """Require all four method decisions without requiring network success."""

    outcomes = {"http_success", "http_429", "browser_challenge", "access_failed"}
    return bool(
        isinstance(rows, list)
        and len(rows) == len(SOURCE_METHODS)
        and {row.get("source_id") for row in rows} == set(SOURCE_IDS)
        and all(
            row.get("primary_url")
            and row.get("method")
            and row.get("task_mapping")
            and row.get("limitations")
            and row.get("access_outcome") in outcomes
            for row in rows
        )
    )


def roadmap_command_specs(root: Path, authority: Path) -> list[CommandSpec]:
    """Build all seven structural checks against the selected YAML."""

    python = str(root / ".venv/bin/python")
    roadmap = str(root / authority)
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    overdue_code = (
        "import pathlib,sys;import scripts.overdue_priority_lint as m;"
        f"m.ROADMAP_NEXT=pathlib.Path({roadmap!r});"
        "sys.argv=['overdue_priority_lint.py','--strict'];raise SystemExit(m.main())"
    )
    return [
        CommandSpec("roadmap_schema", (python, "-u", "-c", schema_code), "selected_yaml"),
        CommandSpec(
            "prior_failure_scope",
            (python, "-u", str(root / PRIOR_LINT_PATH), roadmap, str(root / COMPLETE_PATH)),
            "selected_yaml",
        ),
        CommandSpec(
            "gate_audit",
            (
                python,
                "-u",
                str(root / GATE_AUDIT_PATH),
                roadmap,
                "--complete",
                str(root / COMPLETE_PATH),
            ),
            "selected_yaml",
        ),
        CommandSpec(
            "exclusion_manifest",
            (python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap),
            "selected_yaml",
        ),
        CommandSpec(
            "prompt_paths",
            (python, "-u", str(root / PROMPT_PATH_LINT_PATH), "prompt-paths", roadmap),
            "selected_yaml",
        ),
        CommandSpec(
            "arc_generalization",
            (python, "-u", str(root / ARC_LINT_PATH), roadmap),
            "selected_yaml",
        ),
        CommandSpec("overdue_priority", (python, "-u", "-c", overdue_code), "selected_yaml"),
    ]


def _failed_names(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[str]:
    """Name each missing, duplicate, failed, or timed-out check."""

    failures: list[str] = []
    for name in names:
        rows = [row for row in receipts if row.get("name") == name]
        if (
            len(rows) != 1
            or rows[0].get("passed") is not True
            or rows[0].get("exit_code") != 0
            or rows[0].get("timed_out") is True
        ):
            failures.append(name)
    return failures


def run_affected_validation(
    root: Path,
    authority: Path,
    raw_dir: Path,
    *,
    test_paths: Sequence[str] | None = None,
    changed_modules: Sequence[str] | None = None,
    static_paths: Sequence[str] | None = None,
    historical_failures: Sequence[Mapping[str, Any]] = (),
    roadmap_runner: Callable[..., list[JsonDict]] = run_commands,
    scoped_runner: Callable[..., JsonDict] = run_scoped_validation,
) -> JsonDict:
    """Run current structural checks and the fixed Exp7303 scoped sequence."""

    basetemp = Path("/tmp/exp7343-v645-scoped")
    basetemp.mkdir(parents=True, exist_ok=True)
    roadmap_receipts = roadmap_runner(
        root, roadmap_command_specs(root, authority), log_dir=raw_dir / "validation/roadmap"
    )
    scoped = scoped_runner(
        root,
        test_paths=list(test_paths or [str(TEST_PATH)]),
        changed_modules=list(changed_modules or [str(MODULE_PATH)]),
        static_paths=list(static_paths or [str(WRAPPER_PATH)]),
        basetemp=basetemp,
        coverage_file=Path("/tmp/.coverage-exp7343-v645"),
        log_dir=raw_dir / "validation/scoped",
        historical_failures=historical_failures,
    )
    scoped_receipts = list(scoped.get("validation_receipts", []))
    reduction = reduce_required_checks(scoped_receipts)
    failures = _failed_names(roadmap_receipts, ROADMAP_CHECK_NAMES)
    failures.extend(reduction["failed_required_commands"])
    failures.extend(reduction["missing_required_commands"])
    failures.extend(reduction["duplicate_required_commands"])
    return {
        "validation_receipts": [*roadmap_receipts, *scoped_receipts],
        "required_checks_passed": not failures and reduction["required_checks_passed"],
        "failed_required_commands": failures,
        "repository_health": scoped.get("repository_health", {}),
    }


def run_terminal_validation(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Run cold reduction and both terminal artifact linters."""

    python = str(root / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib;"
        "from carnot.experiment_7343_v645_contract import independent_reduce;"
        f"errors=independent_reduce(json.loads(pathlib.Path({str(candidate)!r}).read_text()));"
        "print(errors,flush=True);raise SystemExit(bool(errors))"
    )
    commands = [
        CommandSpec(
            "independent_reducer", (python, "-u", "-c", reducer_code), "measured_candidate"
        ),
        CommandSpec(
            "adversarial",
            (python, "-u", str(root / ADVERSARIAL_PATH), str(candidate)),
            "measured_candidate",
        ),
        CommandSpec(
            "row_consistency",
            (python, "-u", str(root / ROW_LINT_PATH), "--strict", str(candidate)),
            "measured_candidate",
        ),
    ]
    return run_commands(root, commands, log_dir=raw_dir / "validation/terminal")


def _historical_failures(root: Path) -> list[JsonDict]:
    """Keep V644 validation failures visible but separate from current checks."""

    try:
        data = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        {
            "observed_at": "2026-09-15",
            "producer_path": str(HISTORY_PATH),
            "producer_sha256": sha256(root / HISTORY_PATH),
            "name": receipt.get("name"),
            "command": receipt.get("command"),
            "exit_code": receipt.get("exit_code"),
            "duration_s": receipt.get("duration_s"),
            "log_sha256": receipt.get("log_sha256"),
            "collection_errors": [],
            "resolved": False,
        }
        for receipt in data.get("validation_receipts", [])
        if isinstance(receipt, Mapping) and receipt.get("passed") is not True
    ]


def _path_label(path: Path, root: Path) -> str:
    """Use stable repository paths and retain absolute private fixture paths."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _hash_paths(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    """Hash each existing input, evaluator, frozen copy, and sidecar."""

    hashes: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        absolute = path if path.is_absolute() else root / path
        if absolute.is_file():
            hashes[_path_label(absolute, root)] = sha256(absolute)
    return hashes


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Record exact input identity and writable destinations before parsing."""

    authority, roadmap, yaml_bytes, candidates = select_yaml_authority(root)
    rows = list(candidates)
    rows.append(
        {
            "check": "yaml_authority",
            "path": str(authority) if authority else None,
            "upstream": "research-roadmap-next.yaml|research-roadmap.yaml",
            "artifact_field": "milestone",
            "expected_value": MILESTONE,
            "observed_value": str(authority) if authority else None,
            "available": authority is not None,
            "blocking": True,
        }
    )
    for relative in INPUT_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative}",
                "path": str(relative),
                "upstream": str(relative),
                "artifact_field": "bytes",
                "expected_value": "readable_nonempty_bytes",
                "observed_value": path.stat().st_size if available else "missing_or_empty",
                "available": available,
                "blocking": True,
            }
        )
    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    rows.append(
        {
            "check": "driving_requirement",
            "path": str(SPEC_PATH),
            "upstream": str(SPEC_PATH),
            "artifact_field": "REQ-*",
            "expected_value": "REQ-REPORT-7343",
            "observed_value": "REQ-REPORT-7343" if "REQ-REPORT-7343" in spec_text else "missing",
            "available": "REQ-REPORT-7343" in spec_text,
            "blocking": True,
        }
    )
    for name, directory in (
        ("result_directory", output_path.parent),
        ("raw_directory", raw_dir),
        ("checkpoint_directory", checkpoint_path.parent),
    ):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            available, observed = True, "directory_writable"
        except OSError as exc:
            available, observed = False, f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "check": name,
                "path": str(directory),
                "upstream": str(directory),
                "artifact_field": "writable",
                "expected_value": True,
                "observed_value": observed,
                "available": available,
                "blocking": True,
            }
        )
    return rows, authority, roadmap, yaml_bytes


def _failure_summary(
    upstream: object, check: object, field: object, expected: object, observed: object
) -> JsonDict:
    """Keep the five values needed to diagnose a terminal failure."""

    return {
        "upstream": upstream,
        "failed_check": check,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def apply_terminal_state(artifact: JsonDict) -> None:
    """Derive score and terminal class only from stored checks."""

    failed_precondition = next(
        (
            row
            for row in artifact.get("preconditions_checked", [])
            if row.get("blocking") and row.get("available") is not True
        ),
        None,
    )
    failed_gate = next(
        (
            row
            for row in artifact.get("acceptance_gate_results", [])
            if row.get("passed") is not True
        ),
        None,
    )
    artifact["contract_complete_score"] = int(
        failed_precondition is None
        and failed_gate is None
        and bool(artifact.get("acceptance_gate_results"))
    )
    artifact["science_readiness_score"] = 0
    artifact["science_value_score"] = 0
    artifact["science_promotion_score"] = 0
    artifact["inference_substrate"] = "aggregation_from_upstream_artifacts"
    artifact["inference_substrate_class"] = "aggregation"
    if failed_precondition is not None:
        artifact.update(
            status="blocked",
            verdict_class="blocked",
            honest_verdict="blocked_v645_contract_required_input_missing",
            gate_check_summary=_failure_summary(
                failed_precondition.get("upstream"),
                failed_precondition.get("check"),
                failed_precondition.get("artifact_field"),
                failed_precondition.get("expected_value"),
                failed_precondition.get("observed_value"),
            ),
        )
    elif failed_gate is not None:
        artifact.update(
            status="complete",
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_v645_contract_mismatched_or_invalid",
            gate_check_summary=_failure_summary(
                FIRST_TASK_ID,
                failed_gate.get("criterion"),
                failed_gate.get("criterion"),
                failed_gate.get("expected"),
                failed_gate.get("observed"),
            ),
        )
    else:
        artifact.update(
            status="complete",
            verdict_class="circular_positive",
            honest_verdict="complete_circular_positive_v645_exact_advisory_contract",
            gate_check_summary={
                "upstream": FIRST_TASK_ID,
                "failed_check": None,
                "artifact_field": "contract_complete_score",
                "expected_value": 1,
                "observed_value": 1,
                "passed": True,
            },
        )


def _acceptance_gates(artifact: Mapping[str, Any], *, require_terminal: bool) -> list[JsonDict]:
    """Reduce parity, controls, source mapping, model state, and validation."""

    contract_rows = artifact.get("contract_rows", [])
    gate_rows = artifact.get("gate_control_rows", [])
    gate_count = len(gate_rows) // len(GATE_CASES) if isinstance(gate_rows, list) else 0
    raw_rows = artifact.get("raw_authority_rows", [])
    receipts = artifact.get("validation_receipts", [])
    criteria: list[tuple[str, object, object, bool, str]] = [
        (
            "authority_selected",
            MILESTONE,
            artifact.get("yaml_milestone"),
            artifact.get("yaml_milestone") == MILESTONE,
            "Only selected V645 bytes can supply executable rows.",
        ),
        (
            "contract_exact",
            EXPECTED_TASK_COUNT,
            len(contract_rows) if isinstance(contract_rows, list) else None,
            artifact.get("contract_passed") is True,
            "Two files must describe the same executable work.",
        ),
        (
            "mutations_rejected",
            8,
            sum(row.get("rejected") is True for row in artifact.get("contract_mutation_rows", [])),
            len(artifact.get("contract_mutation_rows", [])) == 8
            and all(
                row.get("rejected") is True for row in artifact.get("contract_mutation_rows", [])
            ),
            "Every named contract change must fail closed.",
        ),
        (
            "raw_authorities_hash_bound",
            True,
            [row.get("hash_matches") for row in raw_rows] if isinstance(raw_rows, list) else None,
            isinstance(raw_rows, list)
            and len(raw_rows) == 2
            and all(row.get("hash_matches") is True for row in raw_rows),
            "Frozen copies bind the exact sources read.",
        ),
        (
            "gate_fields_declared",
            True,
            artifact.get("gate_declaration_result", {}).get("passed"),
            artifact.get("gate_declaration_result", {}).get("passed") is True,
            "Each consumed field must occur in its producer contract.",
        ),
        (
            "gate_controls_complete",
            gate_count * len(GATE_CASES),
            len(gate_rows) if isinstance(gate_rows, list) else None,
            gate_count > 0 and gate_controls_complete(gate_rows, gate_count),
            "Unusable evidence must never authorize downstream work.",
        ),
        (
            "source_method_map_complete",
            len(SOURCE_METHODS),
            len(artifact.get("source_method_map", [])),
            source_map_complete(artifact.get("source_method_map")),
            "Access can fail while each bounded method decision remains visible.",
        ),
        (
            "study_mapping_ingested",
            True,
            artifact.get("study_mapping_ingested"),
            artifact.get("study_mapping_ingested") is True,
            "The durable research ledger must record the V645 mapping.",
        ),
        (
            "model_contract",
            {"MODEL_SPECS": [], "model_invoked": False, "counts": ZERO_INVOCATION_COUNTS},
            {
                "MODEL_SPECS": artifact.get("MODEL_SPECS"),
                "model_invoked": artifact.get("model_invoked"),
                "counts": artifact.get("invocation_counts"),
            },
            artifact.get("MODEL_SPECS") == []
            and artifact.get("model_invoked") is False
            and artifact.get("invocation_counts") == ZERO_INVOCATION_COUNTS,
            "This aggregation receipt cannot claim a current model call.",
        ),
        (
            "affected_validation",
            True,
            artifact.get("required_checks_passed"),
            artifact.get("required_checks_passed") is True,
            "A current structural or scoped failure disqualifies completion.",
        ),
    ]
    if require_terminal:
        criteria.append(
            (
                "terminal_validation",
                list(TERMINAL_CHECK_NAMES),
                [row.get("name") for row in receipts if row.get("name") in TERMINAL_CHECK_NAMES],
                not _failed_names(receipts, TERMINAL_CHECK_NAMES),
                "Cold reduction and both terminal linters must pass.",
            )
        )
    return [
        {
            "criterion": name,
            "expected": expected,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }
        for name, expected, observed, passed, principle in criteria
    ]


def _phase_span(
    name: str, phase_start: float, run_start: float, units: int, checkpoint: str
) -> JsonDict:
    """Measure one real disjoint phase without a duration floor."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_elapsed_s": phase_start - run_start,
        "end_elapsed_s": ended - run_start,
        "duration_s": ended - phase_start,
        "completed_units": units,
        "checkpoint_boundary": checkpoint,
        "pending_operations": [],
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain each top-level field and preserve the required principles."""

    principles = {
        field: "This field keeps one part of the V645 receipt auditable." for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def _base_artifact(run_date: str, started_at: str, checkpoint_path: Path) -> JsonDict:
    """Create a running checkpoint that cannot resemble terminal success."""

    return {
        "schema": "carnot.exp7343.v645_contract.v1",
        "experiment_id": FIRST_TASK_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": dict(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "contract_rows": {"planned": 14, "attempted": 0, "completed": 0, "censored": 0},
            "gate_controls": {"planned": 0, "attempted": 0, "completed": 0, "censored": 0},
            "source_requests": {"planned": 4, "attempted": 0, "completed": 0, "censored": 0},
            "stopping_rule": "Stop after fourteen rows, eight mutations, seven controls per gate, four source requests, and one bounded validation sequence.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _failure_summary(
            FIRST_TASK_ID, "work_in_progress", "status", "terminal", "running"
        ),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_running_v645_contract",
        "verdict_class": "partial",
        "flagged_adversarial": None,
        "validation_receipts": [],
        "repository_health": {},
        "required_checks_passed": False,
        "contract_complete_score": 0,
        "science_readiness_score": 0,
        "science_value_score": 0,
        "science_promotion_score": 0,
        "contract_rows": [],
        "source_method_map": [],
        "checkpoint_path": str(checkpoint_path),
        "field_principles": {},
    }


def _finalize(artifact: JsonDict, started: float) -> None:
    """Seal real timestamps, field explanations, and the evidence checksum."""

    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute stored evidence without trusting the completion score."""

    errors: list[str] = []
    rows = artifact.get("contract_rows")
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_TASK_COUNT
        or any(
            not isinstance(row.get("checks"), Mapping)
            or row.get("passed") != all(row["checks"].values())
            or row.get("censored") is not False
            for row in rows
        )
    ):
        errors.append("contract_rows_invalid")
    gate_rows = artifact.get("gate_control_rows")
    gate_count = len(gate_rows) // len(GATE_CASES) if isinstance(gate_rows, list) else 0
    if gate_count == 0 or not gate_controls_complete(gate_rows, gate_count):
        errors.append("gate_controls_invalid")
    if not source_map_complete(artifact.get("source_method_map")):
        errors.append("source_method_map_invalid")
    raw_rows = artifact.get("raw_authority_rows")
    if (
        not isinstance(raw_rows, list)
        or len(raw_rows) != 2
        or any(row.get("source_sha256") != row.get("raw_sha256") for row in raw_rows)
    ):
        errors.append("raw_authorities_invalid")
    mutations = artifact.get("contract_mutation_rows")
    if (
        not isinstance(mutations, list)
        or len(mutations) != 8
        or any(row.get("rejected") is not True for row in mutations)
    ):
        errors.append("contract_mutations_invalid")
    spans = artifact.get("phase_spans")
    if not isinstance(spans, list) or any(
        row.get("start_elapsed_s", 0) > row.get("end_elapsed_s", -1)
        or (index and row.get("start_elapsed_s", 0) < spans[index - 1].get("end_elapsed_s", 0))
        for index, row in enumerate(spans)
    ):
        errors.append("phase_spans_invalid")
    if artifact.get("rows") != rows:
        errors.append("rows_invalid")
    return errors


def validate_artifact(artifact: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Recompute identity, hashes, evidence, terminal state, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema") != "carnot.exp7343.v645_contract.v1"
        or artifact.get("experiment_id") != FIRST_TASK_ID
        or artifact.get("milestone") != MILESTONE
    ):
        errors.append("identity_invalid")
    if artifact.get("run_date") != RUN_DATE or artifact.get("status") not in {
        "complete",
        "blocked",
    }:
        errors.append("lifecycle_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if (
        artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts"
        or artifact.get("inference_substrate_class") != "aggregation"
        or artifact.get("execution_venue") != "host"
    ):
        errors.append("substrate_invalid")
    principles = artifact.get("field_principles")
    if (
        not isinstance(principles, Mapping)
        or any(principles.get(field) != value for field, value in REQUIRED_FIELD_PRINCIPLES.items())
        or any(field not in principles for field in artifact)
    ):
        errors.append("field_principles_invalid")
    if artifact.get("verdict_class") != "blocked":
        errors.extend(independent_reduce(artifact))
        names = [row.get("name") for row in artifact.get("validation_receipts", [])]
        if any(names.count(name) != 1 for name in ALL_VALIDATION_NAMES):
            errors.append("validation_receipts_invalid")
        adversarial = next(
            (
                row
                for row in artifact.get("validation_receipts", [])
                if row.get("name") == "adversarial"
            ),
            {},
        )
        if artifact.get("flagged_adversarial") is not (adversarial.get("passed") is not True):
            errors.append("flagged_adversarial_invalid")
    for label, expected_hash in artifact.get("source_artifact_hashes", {}).items():
        path = Path(str(label))
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file() or sha256(absolute) != expected_hash:
            errors.append("source_hash_mismatch")
            break
    expected = deepcopy(dict(artifact))
    apply_terminal_state(expected)
    for field in (
        "status",
        "contract_complete_score",
        "verdict_class",
        "honest_verdict",
        "gate_check_summary",
        "science_readiness_score",
        "science_value_score",
        "science_promotion_score",
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"{field}_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    fetcher: Fetcher = _fetch_url,
    validation_runner: ValidationRunner = run_affected_validation,
    terminal_runner: TerminalRunner = run_terminal_validation,
) -> JsonDict:
    """Measure, validate, and atomically publish the V645 contract receipt."""

    started = time.monotonic()
    progress(0, "start", "write running checkpoint before preconditions")
    artifact = _base_artifact(run_date, datetime.now(UTC).isoformat(), checkpoint_path)
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(checkpoint_path, artifact)

    phase_start = time.monotonic()
    progress(1, "start", "check exact inputs and resolve authority")
    preconditions, authority, roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["phase_spans"].append(
        _phase_span("preconditions", phase_start, started, len(preconditions), "inputs_checked")
    )
    if any(row.get("blocking") and row.get("available") is not True for row in preconditions):
        apply_terminal_state(artifact)
        _finalize(artifact, started)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_path, artifact)
        progress(7, "terminal_write", f"blocked output={output_path}")
        return artifact
    assert authority is not None and roadmap is not None and yaml_bytes is not None

    phase_start = time.monotonic()
    progress(2, "start", "hash-copy and compare independent sources")
    raw_dir.mkdir(parents=True, exist_ok=True)
    design_path = root / str(roadmap.get("milestone_doc", DESIGN_PATH))
    design_bytes = design_path.read_bytes()
    design_copy = raw_dir / "selected-contract.md"
    yaml_copy = raw_dir / "selected-roadmap.yaml"
    _atomic_write_bytes(design_copy, design_bytes)
    _atomic_write_bytes(yaml_copy, yaml_bytes)
    artifact["raw_authority_rows"] = [
        {
            "source_type": "markdown",
            "source_path": _path_label(design_path, root),
            "raw_path": _path_label(design_copy, root),
            "source_sha256": sha256_bytes(design_bytes),
            "raw_sha256": sha256(design_copy),
            "hash_matches": sha256_bytes(design_bytes) == sha256(design_copy),
        },
        {
            "source_type": "yaml",
            "source_path": str(authority),
            "raw_path": _path_label(yaml_copy, root),
            "source_sha256": sha256_bytes(yaml_bytes),
            "raw_sha256": sha256(yaml_copy),
            "hash_matches": sha256_bytes(yaml_bytes) == sha256(yaml_copy),
        },
    ]
    contract = evaluate_contract(design_bytes.decode("utf-8"), roadmap)
    artifact.update(contract)
    artifact["contract_passed"] = contract["passed"]
    with tempfile.TemporaryDirectory(prefix="exp7343-mutations-", dir="/tmp") as directory:
        artifact["contract_mutation_rows"] = run_contract_mutations(
            design_bytes.decode("utf-8"), roadmap, Path(directory)
        )
    artifact["rows"] = artifact["contract_rows"]
    artifact["sample_size_budget"]["contract_rows"].update(attempted=14, completed=14)
    artifact["phase_spans"].append(
        _phase_span("contract", phase_start, started, 14, "sources_frozen")
    )

    phase_start = time.monotonic()
    progress(3, "start", "exercise real gate reader")
    gate_count = sum(len(task.get("gated_on") or []) for task in roadmap["tasks"])
    artifact["gate_control_rows"] = run_gate_controls(roadmap, raw_dir / "gate-controls")
    gate_units = gate_count * len(GATE_CASES)
    artifact["sample_size_budget"]["gate_controls"].update(
        planned=gate_units, attempted=gate_units, completed=gate_units
    )
    gate_sidecar = raw_dir / "scripted-model-shaped-gate-fixtures.json"
    _atomic_write_json(
        gate_sidecar,
        {
            "schema": "carnot.exp7343.scripted_fixtures.v1",
            "current_model_invoked": False,
            "MODEL_SPECS": ["injected/noncurrent-fixture"],
            "accepted_as_science": False,
        },
    )
    artifact["phase_spans"].append(
        _phase_span("gate_controls", phase_start, started, gate_units, "gate_controls_written")
    )

    phase_start = time.monotonic()
    progress(4, "start", "refresh four primary sources")
    source_map, access_rows = collect_source_method_map(fetcher)
    artifact["source_method_map"] = source_map
    artifact["source_access_failure_count"] = sum(
        row["access_outcome"] != "http_success" for row in access_rows
    )
    access_sidecar = raw_dir / "source-access-receipts.json"
    _atomic_write_json(
        access_sidecar, {"schema": "carnot.exp7343.source_access.v1", "receipts": access_rows}
    )
    history = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    history_sidecar = raw_dir / "historical-model-receipts.json"
    _atomic_write_json(
        history_sidecar,
        {
            "schema": "carnot.exp7343.historical_models.v1",
            "current_invocation": {
                "MODEL_SPECS": [],
                "model_invoked": False,
                "invocation_counts": ZERO_INVOCATION_COUNTS,
            },
            "historical_artifact": {
                "path": str(HISTORY_PATH),
                "sha256": sha256(root / HISTORY_PATH),
                "MODEL_SPECS": history.get("MODEL_SPECS"),
                "model_invoked": history.get("model_invoked"),
                "accepted_as_current_inference": False,
            },
        },
    )
    artifact["study_mapping_ingested"] = "EXP7343-V645-SOURCE-INGESTION-20260916" in (
        root / STUDY_PATH
    ).read_text(encoding="utf-8")
    artifact["sample_size_budget"]["source_requests"].update(
        attempted=4,
        completed=sum(row["access_outcome"] == "http_success" for row in access_rows),
        censored=artifact["source_access_failure_count"],
    )
    artifact["sidecar_receipts"] = {
        "source_access": {
            "path": _path_label(access_sidecar, root),
            "sha256": sha256(access_sidecar),
        },
        "scripted_gate_fixtures": {
            "path": _path_label(gate_sidecar, root),
            "sha256": sha256(gate_sidecar),
        },
        "historical_models": {
            "path": _path_label(history_sidecar, root),
            "sha256": sha256(history_sidecar),
        },
    }
    artifact["phase_spans"].append(
        _phase_span("sources", phase_start, started, 4, "source_sidecars_written")
    )

    phase_start = time.monotonic()
    progress(5, "start", "run structural and scoped validation")
    validation = validation_runner(
        root=root,
        authority=authority,
        raw_dir=raw_dir,
        test_paths=[str(TEST_PATH)],
        changed_modules=[str(MODULE_PATH)],
        static_paths=[str(WRAPPER_PATH)],
        historical_failures=_historical_failures(root),
    )
    artifact["validation_receipts"] = list(validation.get("validation_receipts", []))
    artifact["required_checks_passed"] = validation.get("required_checks_passed") is True
    artifact["repository_health"] = validation.get("repository_health", {})
    artifact["source_artifact_hashes"] = _hash_paths(
        root,
        [
            *INPUT_PATHS,
            authority,
            design_path,
            design_copy,
            yaml_copy,
            access_sidecar,
            gate_sidecar,
            history_sidecar,
        ],
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(artifact, require_terminal=False)
    apply_terminal_state(artifact)
    artifact["phase_spans"].append(
        _phase_span(
            "affected_validation",
            phase_start,
            started,
            len(artifact["validation_receipts"]),
            "candidate_next",
        )
    )
    _finalize(artifact, started)
    candidate = raw_dir / "measured-terminal-candidate.json"
    _atomic_write_json(candidate, artifact)
    progress(5, "candidate_written", f"path={candidate}")

    phase_start = time.monotonic()
    progress(6, "start", "run independent and strict terminal checks")
    terminal_receipts = terminal_runner(root, candidate, raw_dir)
    artifact["validation_receipts"].extend(terminal_receipts)
    adversarial = next((row for row in terminal_receipts if row.get("name") == "adversarial"), {})
    artifact["flagged_adversarial"] = adversarial.get("passed") is not True
    artifact["required_checks_passed"] = not _failed_names(
        artifact["validation_receipts"], ALL_VALIDATION_NAMES
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(artifact, require_terminal=True)
    apply_terminal_state(artifact)
    artifact["phase_spans"].append(
        _phase_span(
            "terminal_validation",
            phase_start,
            started,
            len(terminal_receipts),
            "terminal_output_ready",
        )
    )
    _finalize(artifact, started)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, artifact)
    progress(7, "terminal_write", f"output={output_path} verdict={artifact['honest_verdict']}")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the fixed V645 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by E2E.
    """Run the advisory receipt and require a valid terminal artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, type=date_argument)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else REPO_ROOT / args.raw_dir
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    artifact = build_artifact(
        REPO_ROOT, args.date, output_path=output, raw_dir=raw_dir, checkpoint_path=checkpoint
    )
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7343] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7343] complete verdict={artifact['honest_verdict']} score={artifact['contract_complete_score']}",
        flush=True,
    )
    return 0

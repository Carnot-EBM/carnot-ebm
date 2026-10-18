"""Build the V642 literature and exact execution-contract receipt.

The module configures the shipped V641 lifecycle with the new roster and
evidence. This keeps parsing, gate fixtures, checkpoints, and streaming
validation shared. The V642 code only adds changed metadata and policy.

Spec refs: REQ-REPORT-7302 and SCENARIO-REPORT-7302-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import re
import sys
import tempfile
from typing import Any, Iterator

import yaml

from carnot import experiment_7288_v641_source_contract as shipped
from carnot.experiment_7179_v633_contract_receipt import (
    _atomic_write_bytes,
    _atomic_write_json,
)
from carnot.experiment_7192_v634_source_contract import (
    _fetch_url,
    reproducibility_checksum,
)


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.642"
RUN_DATE = "20260914"
RETRIEVAL_DATE = "2026-09-14"
FIRST_TASK_ID = "exp7302-source-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7303-validation-scope",
    "exp7304-arc-receipt",
    "exp7305-arc-selfparse",
    "exp7306-batch-fixture",
    "exp7307-batch-canary",
    "exp7308-batch-measurement",
    "exp7309-batch-audit",
    "exp7310-factor-prototype",
    "exp7311-factor-learning",
    "exp7312-factor-audit",
    "exp7313-cost-envelope",
    "exp7314-board-continuity",
    "exp7315-capstone",
)
EXPECTED_TASK_COUNT = 14
EXPECTED_GATE_EDGE_COUNT = 8
GATE_CASES_PER_EDGE = 5
RANDOM_SEED = 7_302_202_609_14

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PLAN_PATH = Path("ops/e2e-test-plan.md")
HISTORY_PATH = Path("results/experiment_7288_v641_source_contract.json")

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
GATE_EVALUATOR_PATH = Path("scripts/conductor_gates.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
PROMPT_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")

MODULE_PATH = Path("python/carnot/experiment_7302_v642_source_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7302_v642_source_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7302_v642_source_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7302_v642_source_contract.json")
RAW_DIR = Path("results/raw/experiment_7302")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7302_v642_source_contract.json")

INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    E2E_PLAN_PATH,
    REFERENCE_PATH,
    COMPLETE_PATH,
    ACTIVE_ROADMAP_PATH,
    DESIGN_PATH,
    ROADMAP_SCHEMA_PATH,
    GATE_EVALUATOR_PATH,
    GATE_LINT_PATH,
    PRIOR_LINT_PATH,
    EXCLUSION_LINT_PATH,
    PROMPT_PATH_LINT_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    SPEC_COVERAGE_PATH,
    SPEC_PATH,
)
CODE_PATHS = (MODULE_PATH, WRAPPER_PATH, TEST_PATH)

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
    "usable_answers": 0,
}
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
POSITIVE_VERDICT = "complete_circular_positive_v642_exact_advisory_contract"
DISQUALIFIED_VERDICT = "complete_disqualified_v642_contract_mismatched_or_invalid"
BLOCKED_VERDICT = "blocked_v642_source_contract_external_prerequisite_missing"

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the record while keeping ordinary top-level experiment_id and milestone.",
    "status": "Write the terminal result only after the work and checks; checkpoints remain separate.",
    "run_date": "Use 20260914 with actual UTC start/end and monotonic phase timing.",
    "preconditions_checked": "Hash real inputs and record actual availability and failed checks.",
    "MODEL_SPECS": "Actual current executable model identities; historical identities remain in sidecars.",
    "model_invoked": "True for any attempted model load or generation, even when output is unusable.",
    "invocation_counts": "Separate attempted, completed, failed, cancelled, and in-flight loads and generations.",
    "inference_substrate": "Describe actual computation with a recognized literal, not intended work.",
    "inference_substrate_class": "Full generation has a 60s floor; bounded generation 10s; load-only 2s. Never pad time.",
    "execution_venue": "Record actual host/device work; a CPU replay is not GPU or FPGA execution.",
    "duration_s": "Measure total elapsed and disjoint phase spans including failed work.",
    "random_seed": "Seal development and independent evaluation seeds before seeing outcomes.",
    "reproducibility_checksum": "Bind code, inputs, config, model when used, and raw evidence.",
    "source_artifact_hashes": "Authenticate producer identity, terminal class, and quarantine state.",
    "rows": "Record every comparative unit/arm with metrics, costs, errors, abstentions, and censoring.",
    "sample_size_budget": "Keep planned, attempted, complete, and censored counts with the frozen stopping rule.",
    "acceptance_gate_results": "Every check has expected, observed, passed, and a one-line principle explaining its purpose.",
    "gate_check_summary": "Every blocked_* names upstream/check, exact field, observed value, and expected value.",
    "verifier_is_oracle": "Shared evaluator authority permits circular_positive only, not positive scientific value.",
    "honest_verdict": "Completed findings start complete_ or complete:; external failure starts blocked_. State the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial; unchanged external failure is blocked.",
    "validation_receipts": "Record exact commands, scope, exit codes, elapsed time, and log hashes; retain failures.",
    "contract_complete_score": "One only for the literal fourteen-row contract and all metadata checks; advisory.",
    "contract_rows": "Exact ordered values from both authorities allow independent comparison.",
    "source_dispositions": "URLs, version dates, access outcomes, adaptations, and falsifiers distinguish evidence from hypotheses.",
    "gate_control_rows": "All five input conditions per gate expose broken producer/consumer contracts.",
    "field_principles": "Keep explanations separate while consumers read ordinary top-level values.",
    "repository_health": "Keep repository-wide observations separate from required affected checks.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

ACCESS_SPECS: tuple[JsonDict, ...] = (
    {
        "access_id": "batch_prompting",
        "url": "https://arxiv.org/abs/2511.04108v4",
        "title": "Batch Prompting",
        "publication_or_version_date": "2025-11-06 / v4 2026-02-20",
    },
    {
        "access_id": "batch_interference",
        "url": "https://arxiv.org/abs/2503.15551v2",
        "title": "BATCHSAFEBENCH",
        "publication_or_version_date": "2025-03-18 / v2 2025-06-20",
    },
    {
        "access_id": "constraint_substructure_recovery",
        "url": "https://arxiv.org/abs/2509.24489",
        "title": "Interactive Constraint Refinement",
        "publication_or_version_date": "2025-09-29",
    },
    {
        "access_id": "selected_ebt_citation",
        "url": "https://arxiv.org/abs/2507.02092",
        "title": "Energy-Based Transformers",
        "publication_or_version_date": "2025-07-02",
    },
)

SOURCE_DISPOSITIONS: tuple[JsonDict, ...] = (
    {
        "method_family": "batch_prompting",
        "disposition": "adapt",
        "adapted_mechanism": "extract several claims jointly within one source version and give direct generation the same batch opportunity",
        "falsifier": "reject value when full cost, semantic parity, coverage, or matched direct-batch comparison fails",
        "deferred_boundary": "call reduction alone is not source-verification value and does not repeat the V641 cache mechanism",
        "target_tasks": [
            "exp7306-batch-fixture",
            "exp7307-batch-canary",
            "exp7308-batch-measurement",
            "exp7309-batch-audit",
        ],
    },
    {
        "method_family": "batch_interference",
        "disposition": "implement",
        "adapted_mechanism": "mix valid and invalid claims, permute identifiers, and inject an instruction inside source data",
        "falsifier": "any untouched claim changes because another batch member or source instruction contaminates it",
        "deferred_boundary": "delimiters are tested controls and do not establish isolation by construction",
        "target_tasks": ["exp7306-batch-fixture", "exp7309-batch-audit"],
    },
    {
        "method_family": "constraint_substructure_recovery",
        "disposition": "adapt",
        "adapted_mechanism": "revise only the factor contradicted by released feedback while retaining unaffected factors",
        "falsifier": "future error fails to improve with protected coverage or sealed past predictions change",
        "deferred_boundary": "the local factor update is not a reproduction of query-driven refinement or proof of continuous learning",
        "target_tasks": [
            "exp7310-factor-prototype",
            "exp7311-factor-learning",
            "exp7312-factor-audit",
        ],
    },
    {
        "method_family": "selected_ebt_citation",
        "disposition": "defer",
        "adapted_mechanism": "retain EBT as the selected energy-reasoning citation while exact execution checks extracted constraints",
        "falsifier": "adopt only after a compatible local model shows independent calibration or reasoning value",
        "deferred_boundary": "energy compatibility does not certify faithful extraction and does not reopen retired energy-generation work",
        "target_tasks": ["exp7315-capstone"],
    },
)

AFFECTED_VALIDATION_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "exclusion_manifest",
    "prompt_paths",
    "gate_fields",
    "focused_pytest",
    "focused_coverage",
    "ruff_check",
    "ruff_format",
    "changed_module_mypy",
    "scoped_spec_coverage",
    "exact_plan_e2e",
    "independent_reducer",
    "adversarial",
    "row_consistency",
)
REPOSITORY_HEALTH_NAMES = ("full_python_suite",)
VALIDATION_COMMAND_NAMES = (*AFFECTED_VALIDATION_NAMES, *REPOSITORY_HEALTH_NAMES)
PRETERMINAL_VALIDATION_COUNT = 11

_BUILD_ROOT: ContextVar[Path] = ContextVar("exp7302_build_root", default=REPO_ROOT)
_BASE_SELECT_AUTHORITY = shipped.select_yaml_authority
_BASE_EVALUATE_CONTRACT = shipped.evaluate_contract
_BASE_RUN_GATE_REPLAYS = shipped.run_gate_replays
_BASE_GATE_REPLAYS_COMPLETE = shipped.gate_replays_complete
_BASE_COLLECT_SOURCE_ROWS = shipped.collect_source_rows
_BASE_SOURCE_ROWS_COMPLETE = shipped.source_rows_complete
_BASE_BASE_ARTIFACT = shipped._base_artifact
_BASE_PRECONDITIONS = shipped._preconditions
_BASE_BUILD_ARTIFACT = shipped.build_artifact


def progress(phase: int, state: str, detail: str) -> None:
    """Flush each boundary so external supervision sees truthful progress."""

    print(f"[exp7302] phase {phase} {state}: {detail}", flush=True)


sha256 = shipped.sha256
sha256_bytes = shipped.sha256_bytes


def _yaml_phase(task: Mapping[str, Any]) -> int | None:
    """Read an explicit phase, with the prompt as a compatibility fallback."""

    explicit = task.get("phase")
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        return explicit
    match = re.search(
        rf"Milestone\s+{re.escape(MILESTONE)},\s*phase\s+(\d+)", str(task.get("prompt", ""))
    )
    return int(match.group(1)) if match else None


@contextmanager
def _configured_shipped(*, runtime: bool = False) -> Iterator[None]:
    """Apply V642 configuration to shared code, then restore prior values."""

    replacements: dict[str, Any] = {
        "MILESTONE": MILESTONE,
        "RUN_DATE": RUN_DATE,
        "RETRIEVAL_DATE": RETRIEVAL_DATE,
        "FIRST_TASK_ID": FIRST_TASK_ID,
        "EXPECTED_ID_ORDER": EXPECTED_ID_ORDER,
        "EXPECTED_TASK_COUNT": EXPECTED_TASK_COUNT,
        "EXPECTED_GATE_EDGE_COUNT": EXPECTED_GATE_EDGE_COUNT,
        "GATE_CASES_PER_EDGE": GATE_CASES_PER_EDGE,
        "RANDOM_SEED": RANDOM_SEED,
        "DESIGN_PATH": DESIGN_PATH,
        "ACTIVE_ROADMAP_PATH": ACTIVE_ROADMAP_PATH,
        "NEXT_ROADMAP_PATH": NEXT_ROADMAP_PATH,
        "ACCESS_SPECS": ACCESS_SPECS,
        "SOURCE_DISPOSITIONS": SOURCE_DISPOSITIONS,
        "ZERO_INVOCATION_COUNTS": ZERO_INVOCATION_COUNTS,
        "progress": progress,
        "_yaml_phase": _yaml_phase,
    }
    if runtime:
        replacements.update(
            {
                "COMPLETE_PATH": COMPLETE_PATH,
                "SPEC_PATH": SPEC_PATH,
                "REFERENCE_PATH": REFERENCE_PATH,
                "EXCLUSION_PATH": EXCLUSION_PATH,
                "E2E_PLAN_PATH": E2E_PLAN_PATH,
                "HISTORY_PATH": HISTORY_PATH,
                "MODULE_PATH": MODULE_PATH,
                "WRAPPER_PATH": WRAPPER_PATH,
                "TEST_PATH": TEST_PATH,
                "DEFAULT_OUTPUT_PATH": DEFAULT_OUTPUT_PATH,
                "RAW_DIR": RAW_DIR,
                "RAW_CANDIDATE_PATH": RAW_CANDIDATE_PATH,
                "CHECKPOINT_PATH": CHECKPOINT_PATH,
                "INPUT_PATHS": INPUT_PATHS,
                "CODE_PATHS": CODE_PATHS,
                "INFERENCE_SUBSTRATE": INFERENCE_SUBSTRATE,
                "INFERENCE_SUBSTRATE_CLASS": INFERENCE_SUBSTRATE_CLASS,
                "POSITIVE_VERDICT": POSITIVE_VERDICT,
                "DISQUALIFIED_VERDICT": DISQUALIFIED_VERDICT,
                "BLOCKED_VERDICT": BLOCKED_VERDICT,
                "REQUIRED_FIELD_PRINCIPLES": REQUIRED_FIELD_PRINCIPLES,
                "REQUIRED_ARTIFACT_FIELDS": REQUIRED_ARTIFACT_FIELDS,
                "VALIDATION_COMMAND_NAMES": VALIDATION_COMMAND_NAMES,
                "PRETERMINAL_VALIDATION_COUNT": PRETERMINAL_VALIDATION_COUNT,
                "select_yaml_authority": select_yaml_authority,
                "evaluate_contract": evaluate_contract,
                "run_gate_replays": run_gate_replays,
                "gate_replays_complete": gate_replays_complete,
                "collect_source_rows": collect_source_rows,
                "source_rows_complete": source_rows_complete,
                "build_sidecars": build_sidecars,
                "_field_principles": _field_principles,
                "_base_artifact": _base_artifact,
                "_preconditions": _preconditions,
                "_freeze_authorities": _freeze_authorities,
                "_sidecars_complete": _sidecars_complete,
                "_validation_complete": _validation_complete,
                "_acceptance_gate_results": _acceptance_gate_results,
                "_failure_summary": _failure_summary,
                "_apply_terminal_state": _apply_terminal_state,
                "independent_reduce": independent_reduce,
                "validate_artifact": validate_artifact,
                "validation_commands": validation_commands,
                "run_required_validation": run_required_validation,
            }
        )
    previous = {name: getattr(shipped, name) for name in replacements}
    for name, value in replacements.items():
        setattr(shipped, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(shipped, name, value)


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Select active V642 bytes first, then staged V642 bytes."""

    with _configured_shipped():
        return _BASE_SELECT_AUTHORITY(root)


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare five task dimensions through independent shipped parsers."""

    with _configured_shipped():
        contract = _BASE_EVALUATE_CONTRACT(markdown_text, yaml_document)
        markdown_phases = shipped._markdown_phases(markdown_text)
    tasks = yaml_document.get("tasks", []) if isinstance(yaml_document, Mapping) else []
    yaml_phases = {
        str(task.get("id")): _yaml_phase(task)
        for task in tasks
        if isinstance(task, Mapping) and task.get("id")
    }
    for row in contract["contract_rows"]:
        markdown = row.get("markdown")
        yaml_row = row.get("yaml")
        if isinstance(markdown, dict):
            markdown["phase"] = markdown_phases.get(str(markdown.get("id")))
        if isinstance(yaml_row, dict):
            yaml_row["phase"] = yaml_phases.get(str(yaml_row.get("id")))
        row["checks"]["phase"] = bool(
            isinstance(markdown, Mapping)
            and isinstance(yaml_row, Mapping)
            and markdown.get("phase") == yaml_row.get("phase")
            and markdown.get("phase") in {1, 2, 3, 4}
        )
        row["passed"] = all(row["checks"].values())
    contract["passed"] = bool(
        contract.get("markdown_milestone") == contract.get("yaml_milestone") == MILESTONE
        and len(contract["contract_rows"]) == EXPECTED_TASK_COUNT
        and [row["unit_id"] for row in contract["contract_rows"]] == list(EXPECTED_ID_ORDER)
        and all(row["passed"] for row in contract["contract_rows"])
        and all(row.get("passed") is True for row in contract["gate_producer_rows"])
        and all(row.get("passed") is True for row in contract["receipt_dependency_rows"])
    )
    return contract


def run_gate_replays(roadmap: Mapping[str, Any], results_dir: Path) -> list[JsonDict]:
    """Run five real evaluator controls for every V642 gate edge."""

    with _configured_shipped():
        return _BASE_RUN_GATE_REPLAYS(roadmap, results_dir)


def gate_replays_complete(rows: object) -> bool:
    """Require all forty distinct producer and consumer control outcomes."""

    with _configured_shipped():
        return _BASE_GATE_REPLAYS_COMPLETE(rows)


def upstream_precondition(data: object, field: str, expected: Any) -> JsonDict:
    """Reject absent, quarantined, or disqualified producer evidence first."""

    if not isinstance(data, Mapping):
        return {
            "field": field,
            "expected": expected,
            "actual": None,
            "quarantined": False,
            "disqualified": False,
            "field_passed": False,
            "passed": False,
            "reason": "missing_upstream_rejected_before_field_consumption",
        }
    result = shipped.gate_fixtures.upstream_precondition(data, field, expected)
    verdict = " ".join(
        str(data.get(name, "")) for name in ("status", "verdict_class", "honest_verdict")
    ).lower()
    disqualified = "disqualified" in verdict
    result["disqualified"] = disqualified
    if disqualified:
        result["passed"] = False
        result["reason"] = "disqualified_upstream_rejected_before_field_consumption"
    return result


def collect_source_rows(
    fetcher: Fetcher = _fetch_url,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check four primary URLs in sequence and retain failed access."""

    with _configured_shipped():
        return _BASE_COLLECT_SOURCE_ROWS(fetcher)


def source_rows_complete(rows: object) -> bool:
    """Require all four V642 decisions without requiring network success."""

    with _configured_shipped():
        return _BASE_SOURCE_ROWS_COMPLETE(rows)


def build_sidecars(
    root: Path,
    raw_dir: Path,
    gate_rows: Sequence[Mapping[str, Any]],
    access_rows: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Store historical, fixture, and response payloads outside the result."""

    history = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    if not isinstance(history, Mapping):
        raise ValueError(f"historical artifact is not a mapping: {HISTORY_PATH}")
    payloads = {
        "history": (
            "historical-model-receipts.json",
            {
                "schema": "carnot.exp7302.historical_model_receipts.v1",
                "current_invocation": {
                    "MODEL_SPECS": [],
                    "model_invoked": False,
                    "invocation_counts": ZERO_INVOCATION_COUNTS,
                },
                "historical_artifacts": [
                    {
                        "path": str(HISTORY_PATH),
                        "sha256": sha256(root / HISTORY_PATH),
                        "experiment_id": history.get("experiment_id"),
                        "status": history.get("status"),
                        "verdict_class": history.get("verdict_class"),
                        "honest_verdict": history.get("honest_verdict"),
                        "MODEL_SPECS": history.get("MODEL_SPECS"),
                        "model_invoked": history.get("model_invoked"),
                        "quarantined": shipped.base._is_quarantined(dict(history)),
                        "accepted_as_dependency": False,
                    }
                ],
            },
        ),
        "negative_fixtures": (
            "negative-gate-fixtures.json",
            {
                "schema": "carnot.exp7302.gate_controls.v1",
                "rows": [dict(row) for row in gate_rows if row.get("case") != "passing"],
            },
        ),
        "source_access": (
            "source-access-receipts.json",
            {
                "schema": "carnot.exp7302.source_access.v1",
                "request_count": len(access_rows),
                "receipts": [dict(row) for row in access_rows],
            },
        ),
    }
    receipts: dict[str, JsonDict] = {}
    for name, (filename, payload) in payloads.items():
        path = raw_dir / filename
        _atomic_write_json(path, payload)
        receipts[name] = {"path": str(path.relative_to(root)), "sha256": sha256(path)}
    return receipts


def _historical_repository_failures(root: Path) -> list[JsonDict]:
    """Copy exact prior full-suite failures from authenticated history."""

    try:
        history = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(history, Mapping):
        return []
    return [
        {
            "producer_path": str(HISTORY_PATH),
            "producer_sha256": sha256(root / HISTORY_PATH),
            "name": row.get("name"),
            "command": row.get("command"),
            "exit_code": row.get("exit_code"),
            "duration_s": row.get("duration_s"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in history.get("validation_receipts", [])
        if isinstance(row, Mapping)
        and row.get("name") in REPOSITORY_HEALTH_NAMES
        and row.get("passed") is not True
    ]


def _repository_health(
    receipts: object,
    historical_failures: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Separate current whole-suite observations from affected checks."""

    current = (
        [
            dict(row)
            for row in receipts
            if isinstance(row, Mapping) and row.get("name") in REPOSITORY_HEALTH_NAMES
        ]
        if isinstance(receipts, list)
        else []
    )
    return {
        "scope": "repository_wide_observation_not_contract_acceptance",
        "passed": bool(current) and all(row.get("passed") is True for row in current),
        "current_checks": current,
        "historical_failures": [dict(row) for row in historical_failures],
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Attach exact prompt principles while values remain top-level."""

    values = {field: "This field keeps one V642 contract fact auditable." for field in fields}
    values.update(REQUIRED_FIELD_PRINCIPLES)
    return values


def _base_artifact(run_date: str, checkpoint_path: Path, started_at: str) -> JsonDict:
    """Create a running checkpoint that cannot resemble success."""

    artifact = _BASE_BASE_ARTIFACT(run_date, checkpoint_path, started_at)
    historical = _historical_repository_failures(_BUILD_ROOT.get())
    artifact.update(
        schema="carnot.exp7302.v642_source_contract.v1",
        experiment_id=FIRST_TASK_ID,
        milestone=MILESTONE,
        random_seed=RANDOM_SEED,
        invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        honest_verdict="partial_running_v642_source_contract",
        contract_complete_score=0,
        repository_health=_repository_health([], historical),
    )
    artifact["expected_id_order"] = list(EXPECTED_ID_ORDER)
    artifact["sample_size_budget"]["gate_controls"].update(planned=40, independent_units=40)
    artifact["sample_size_budget"]["source_dispositions"].update(planned=4, independent_units=4)
    artifact["sample_size_budget"]["stopping_rule"] = (
        "Stop after fourteen contract rows, five controls for each of eight gates, "
        "four source dispositions, four sequential requests, and one validation sequence."
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Reuse ownership checks and bind them to the V642 requirement."""

    rows, authority, roadmap, content = _BASE_PRECONDITIONS(
        root, output_path, raw_dir, checkpoint_path
    )
    for row in rows:
        if row.get("check") == "v640_terminal_capstone":
            row["check"] = "v641_terminal_contract_evidence"
            row["expected_value"] = "terminal historical evidence; not a dependency gate"
        if row.get("check") == "driving_requirement":
            try:
                found = "REQ-REPORT-7302" in (root / SPEC_PATH).read_text(encoding="utf-8")
            except OSError:
                found = False
            row.update(
                expected_value="REQ-REPORT-7302",
                observed_value="REQ-REPORT-7302" if found else "missing",
                available=found,
            )
    return rows, authority, roadmap, content


def _freeze_authorities(
    root: Path, authority: Path, yaml_bytes: bytes, raw_dir: Path
) -> list[JsonDict]:
    """Freeze original Markdown and selected YAML bytes without normalization."""

    sources = (
        ("markdown", DESIGN_PATH, (root / DESIGN_PATH).read_bytes(), raw_dir / "v642-design.md"),
        ("yaml", authority, yaml_bytes, raw_dir / "selected-roadmap.yaml"),
    )
    rows: list[JsonDict] = []
    for source_type, source, content, target in sources:
        _atomic_write_bytes(target, content)
        rows.append(
            {
                "source_type": source_type,
                "source_path": str(source),
                "raw_path": str(target.relative_to(root)),
                "source_sha256": sha256_bytes(content),
                "raw_sha256": sha256(target),
                "hash_matches": sha256_bytes(content) == sha256(target),
            }
        )
    return rows


def _sidecars_complete(artifact: Mapping[str, Any]) -> bool:
    """Require all three sidecars to match authenticated source hashes."""

    receipts = artifact.get("sidecar_receipts")
    hashes = artifact.get("source_artifact_hashes")
    return (
        isinstance(receipts, Mapping)
        and set(receipts) == {"history", "negative_fixtures", "source_access"}
        and isinstance(hashes, Mapping)
        and all(
            isinstance(receipt, Mapping)
            and hashes.get(receipt.get("path")) == receipt.get("sha256")
            for receipt in receipts.values()
        )
    )


def _validation_complete(artifact: Mapping[str, Any]) -> bool:
    """Require affected checks while keeping whole-suite health advisory."""

    rows = artifact.get("validation_receipts")
    if not isinstance(rows, list) or [row.get("name") for row in rows] != list(
        VALIDATION_COMMAND_NAMES
    ):
        return False
    by_name = {row.get("name"): row for row in rows}
    return all(by_name[name].get("passed") is True for name in AFFECTED_VALIDATION_NAMES)


def _acceptance_gate_results(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute contract acceptance without repository-wide observations."""

    contract = artifact.get("contract_rows")
    sources = artifact.get("source_dispositions")
    access = artifact.get("source_access_rows")
    raw = artifact.get("raw_authority_rows")
    hashes = artifact.get("source_artifact_hashes")
    receipts = artifact.get("validation_receipts")
    criteria = (
        (
            "contract_exact",
            list(EXPECTED_ID_ORDER),
            [row.get("unit_id") for row in contract] if isinstance(contract, list) else None,
            isinstance(contract, list)
            and len(contract) == EXPECTED_TASK_COUNT
            and [row.get("unit_id") for row in contract] == list(EXPECTED_ID_ORDER)
            and all(row.get("passed") is True for row in contract)
            and artifact.get("markdown_milestone") == artifact.get("yaml_milestone") == MILESTONE,
            "Both byte authorities must agree on all five task dimensions.",
        ),
        (
            "gate_controls",
            40,
            len(artifact.get("gate_control_rows", [])),
            gate_replays_complete(artifact.get("gate_control_rows")),
            "Each real gate keeps five evaluator and intake observations.",
        ),
        (
            "source_dispositions",
            4,
            len(sources) if isinstance(sources, list) else None,
            source_rows_complete(sources),
            "Access failure stays evidence without blocking a complete source row.",
        ),
        (
            "request_budget",
            "exactly four sequential primary requests",
            len(access) if isinstance(access, list) else None,
            isinstance(access, list)
            and len(access) == 4
            and all(
                row.get("kind") == "primary_record" and row.get("timeout_s") == 20 for row in access
            ),
            "The literature recheck stays within its fixed request budget.",
        ),
        (
            "raw_authorities",
            2,
            len(raw) if isinstance(raw, list) else None,
            isinstance(raw, list)
            and len(raw) == 2
            and all(row.get("hash_matches") is True for row in raw),
            "Original Markdown and YAML bytes remain independently available.",
        ),
        (
            "sidecars",
            True,
            _sidecars_complete(artifact),
            _sidecars_complete(artifact),
            "Historical and fixture payloads remain outside current counters.",
        ),
        (
            "source_hashes",
            "all present",
            "all present"
            if isinstance(hashes, Mapping) and hashes and all(hashes.values())
            else "missing",
            isinstance(hashes, Mapping) and bool(hashes) and all(hashes.values()),
            "Every consumed input and sidecar has an exact digest.",
        ),
        (
            "affected_validation",
            list(AFFECTED_VALIDATION_NAMES),
            [
                row.get("name")
                for row in receipts
                if isinstance(row, Mapping) and row.get("name") in AFFECTED_VALIDATION_NAMES
            ]
            if isinstance(receipts, list)
            else None,
            _validation_complete(artifact),
            "Required checks pass independently of repository-wide observations.",
        ),
    )
    return [
        {
            "criterion": name,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
            "principle": principle,
        }
        for name, expected, observed, passed, principle in criteria
    ]


def _failure_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Name the exact first external, contract, or affected-check failure."""

    failed = shipped.base._failed_precondition(artifact)
    if failed is not None:
        return shipped.base._summary(
            str(failed.get("check")),
            failed.get("expected_value"),
            failed.get("observed_value"),
            upstream=failed.get("upstream"),
            field=failed.get("field"),
        )
    if artifact.get("markdown_milestone") != MILESTONE:
        return shipped.base._summary(
            "markdown_milestone",
            MILESTONE,
            artifact.get("markdown_milestone"),
            upstream=str(DESIGN_PATH),
            field="milestone",
        )
    contract = artifact.get("contract_rows")
    if isinstance(contract, list):
        row = next((value for value in contract if value.get("passed") is not True), None)
        if row is not None:
            return shipped.base._summary(
                "contract_parity",
                row.get("yaml"),
                row.get("markdown"),
                upstream=row.get("unit_id"),
                field="id|order|title|deliverable|phase|gates",
            )
    acceptance = _acceptance_gate_results(artifact)
    row = next((value for value in acceptance if value["passed"] is not True), None)
    return shipped.base._summary(
        f"acceptance:{row['criterion']}" if row else "stored_contract",
        row.get("expected") if row else "complete",
        row.get("observed") if row else "incomplete",
        field=row.get("criterion") if row else "acceptance_gate_results",
    )


def _apply_terminal_state(artifact: JsonDict) -> None:
    """Derive terminal lifecycle solely from stored evidence."""

    previous_health = artifact.get("repository_health")
    historical = (
        previous_health.get("historical_failures", [])
        if isinstance(previous_health, Mapping)
        else []
    )
    artifact["repository_health"] = _repository_health(
        artifact.get("validation_receipts"), historical
    )
    artifact["acceptance_gate_results"] = _acceptance_gate_results(artifact)
    artifact["contract_complete_score"] = int(
        len(artifact["acceptance_gate_results"]) == 8
        and all(row["passed"] for row in artifact["acceptance_gate_results"])
    )
    failed = shipped.base._failed_precondition(artifact)
    if failed is not None:
        artifact.update(
            status="blocked",
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
            verdict_class="blocked",
            honest_verdict=BLOCKED_VERDICT,
            gate_check_summary=_failure_summary(artifact),
        )
    elif artifact["contract_complete_score"] == 1:
        artifact.update(
            status="complete",
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
            verdict_class="circular_positive",
            honest_verdict=POSITIVE_VERDICT,
            gate_check_summary=shipped.base._summary(
                None,
                "all V642 advisory contract criteria pass",
                "all V642 advisory contract criteria pass",
                passed=True,
            ),
        )
    else:
        artifact.update(
            status="complete",
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
            verdict_class="disqualified",
            honest_verdict=DISQUALIFIED_VERDICT,
            gate_check_summary=_failure_summary(artifact),
        )


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Check row consistency without requiring a positive contract result."""

    errors: list[str] = []
    contract = artifact.get("contract_rows")
    if not isinstance(contract, list) or len(contract) != EXPECTED_TASK_COUNT:
        errors.append("contract_row_count")
    else:
        for row in contract:
            checks = row.get("checks")
            markdown = row.get("markdown")
            yaml_row = row.get("yaml")
            if (
                not isinstance(checks, Mapping)
                or row.get("passed") != all(checks.values())
                or not isinstance(markdown, Mapping)
                or not isinstance(yaml_row, Mapping)
                or markdown.get("phase") not in {1, 2, 3, 4}
                or yaml_row.get("phase") not in {1, 2, 3, 4}
                or row.get("censored") is not False
            ):
                errors.append("contract_row_reduction")
                break
    if not gate_replays_complete(artifact.get("gate_control_rows")):
        errors.append("gate_control_reduction")
    if not source_rows_complete(artifact.get("source_dispositions")):
        errors.append("source_disposition_reduction")
    raw = artifact.get("raw_authority_rows")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(row.get("source_sha256") != row.get("raw_sha256") for row in raw)
    ):
        errors.append("raw_authority_reduction")
    return errors


def validate_artifact(artifact: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Recompute required fields, evidence, terminal state, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if artifact.get("schema") != "carnot.exp7302.v642_source_contract.v1":
        errors.append("schema_invalid")
    if artifact.get("experiment_id") != FIRST_TASK_ID or artifact.get("milestone") != MILESTONE:
        errors.append("identity_invalid")
    if (
        artifact.get("status") not in {"complete", "blocked"}
        or artifact.get("run_date") != RUN_DATE
    ):
        errors.append("lifecycle_invalid")
    for field in ("started_at_utc", "completed_at_utc"):
        try:
            timestamp = datetime.fromisoformat(str(artifact.get(field)))
        except ValueError:
            errors.append(f"{field}_invalid")
        else:
            if timestamp.tzinfo is None:
                errors.append(f"{field}_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != text for field, text in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS
    ):
        errors.append("model_contract_invalid")
    if artifact.get("execution_venue") != "host" or artifact.get("execution_host") != (
        platform.node() or "unknown"
    ):
        errors.append("execution_invalid")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        errors.append("duration_invalid")
    if artifact.get("random_seed") != RANDOM_SEED or artifact.get("verifier_is_oracle") is not True:
        errors.append("authority_invalid")
    if artifact.get("rows") != artifact.get("contract_rows"):
        errors.append("rows_invalid")
    budget = artifact.get("sample_size_budget")
    budget_groups = ("contract_rows", "gate_controls", "source_dispositions", "network_requests")
    if (
        not isinstance(budget, Mapping)
        or any(not isinstance(budget.get(group), Mapping) for group in budget_groups)
        or any(
            field not in budget[group]
            for group in budget_groups
            for field in ("planned", "attempted", "completed", "censored", "independent_units")
        )
    ):
        errors.append("sample_size_budget_invalid")
    hashes = artifact.get("source_artifact_hashes")
    if artifact.get("verdict_class") != "blocked":
        if (
            not isinstance(hashes, Mapping)
            or not hashes
            or any(not value for value in hashes.values())
        ):
            errors.append("source_hashes_invalid")
        else:
            for relative, expected_hash in hashes.items():
                try:
                    actual_hash = sha256(root / str(relative))
                except OSError:
                    actual_hash = None
                if actual_hash != expected_hash:
                    errors.append("source_hash_mismatch")
                    break
        errors.extend(independent_reduce(artifact))
    receipts = artifact.get("validation_receipts")
    expected_names = (
        [] if artifact.get("verdict_class") == "blocked" else list(VALIDATION_COMMAND_NAMES)
    )
    if not isinstance(receipts, list) or [row.get("name") for row in receipts] != expected_names:
        errors.append("validation_receipts_invalid")
    elif any(
        not row.get("command")
        or not isinstance(row.get("exit_code"), int)
        or not str(row.get("log_sha256", "")).startswith("sha256:")
        for row in receipts
    ):
        errors.append("validation_receipt_shape_invalid")
    expected_history = _historical_repository_failures(root)
    health = artifact.get("repository_health")
    if not isinstance(health, Mapping) or health != _repository_health(receipts, expected_history):
        errors.append("repository_health_invalid")
    expected = deepcopy(dict(artifact))
    _apply_terminal_state(expected)
    for field in (
        "status",
        "acceptance_gate_results",
        "contract_complete_score",
        "inference_substrate",
        "inference_substrate_class",
        "verdict_class",
        "honest_verdict",
        "gate_check_summary",
        "repository_health",
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"{field}_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def e2e_contract_receipt(root: Path, temp_dir: Path) -> JsonDict:
    """Copy both real authorities through parser, gate reader, and reducer."""

    authority, roadmap, yaml_bytes, _candidates = select_yaml_authority(root)
    if authority is None or roadmap is None or yaml_bytes is None:
        return {"passed": False, "reason": "v642_yaml_authority_missing"}
    temp_dir.mkdir(parents=True, exist_ok=True)
    markdown_bytes = (root / DESIGN_PATH).read_bytes()
    markdown_copy = temp_dir / "plan.md"
    yaml_copy = temp_dir / "plan.yaml"
    _atomic_write_bytes(markdown_copy, markdown_bytes)
    _atomic_write_bytes(yaml_copy, yaml_bytes)
    contract = evaluate_contract(markdown_bytes.decode(), yaml.safe_load(yaml_bytes))
    gates = run_gate_replays(roadmap, temp_dir / "gate-inputs")
    payload = {
        "markdown_sha256": sha256(markdown_copy),
        "yaml_sha256": sha256(yaml_copy),
        "yaml_authority": str(authority),
        "contract_rows": len(contract["contract_rows"]),
        "gate_control_rows": len(gates),
        "contract_passed": contract["passed"],
        "gates_passed": gate_replays_complete(gates),
    }
    payload["passed"] = payload["contract_passed"] and payload["gates_passed"]
    payload["diagnostic_sha256"] = sha256_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


def validation_commands(
    root: Path, candidate_path: Path, authority: Path
) -> tuple[tuple[str, list[str]], ...]:
    """Return affected checks first and one separate repository observation."""

    python = str(root / ".venv/bin/python")
    roadmap = str(root / authority)
    candidate = str(candidate_path)
    coverage_file = "/tmp/.coverage-exp7302-v642"
    pytest_base = "/tmp/exp7302-v642-focused"
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    e2e_code = (
        "import pathlib,sys,tempfile;"
        "from carnot.experiment_7302_v642_source_contract import e2e_contract_receipt;"
        f"receipt=e2e_contract_receipt(pathlib.Path({str(root)!r}),pathlib.Path(tempfile.mkdtemp(prefix='exp7302-e2e-',dir='/tmp')));"
        "print(receipt);sys.exit(not receipt['passed'])"
    )
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7302_v642_source_contract import independent_reduce;"
        f"errors=independent_reduce(json.loads(pathlib.Path({candidate!r}).read_text()));"
        "print(errors);sys.exit(bool(errors))"
    )
    return (
        ("roadmap_schema", [python, "-u", "-c", schema_code]),
        ("prior_failure", [python, "-u", str(root / PRIOR_LINT_PATH), roadmap]),
        ("exclusion_manifest", [python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap]),
        (
            "prompt_paths",
            [python, "-u", str(root / PROMPT_PATH_LINT_PATH), "prompt-paths", roadmap],
        ),
        ("gate_fields", [python, "-u", str(root / GATE_LINT_PATH), roadmap]),
        (
            "focused_pytest",
            [
                str(root / ".venv/bin/coverage"),
                "run",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--no-cov",
                f"--basetemp={pytest_base}",
                str(root / TEST_PATH),
                "-q",
            ],
        ),
        (
            "focused_coverage",
            [
                str(root / ".venv/bin/coverage"),
                "report",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        (
            "ruff_check",
            [
                str(root / ".venv/bin/ruff"),
                "check",
                str(root / MODULE_PATH),
                str(root / WRAPPER_PATH),
                str(root / TEST_PATH),
            ],
        ),
        (
            "ruff_format",
            [
                str(root / ".venv/bin/ruff"),
                "format",
                "--check",
                str(root / MODULE_PATH),
                str(root / WRAPPER_PATH),
                str(root / TEST_PATH),
            ],
        ),
        ("changed_module_mypy", [str(root / ".venv/bin/mypy"), str(root / MODULE_PATH)]),
        (
            "scoped_spec_coverage",
            [python, "-u", str(root / SPEC_COVERAGE_PATH), str(root / TEST_PATH)],
        ),
        ("exact_plan_e2e", [python, "-u", "-c", e2e_code]),
        ("independent_reducer", [python, "-u", "-c", reducer_code]),
        ("adversarial", [python, "-u", str(root / ADVERSARIAL_PATH), candidate]),
        ("row_consistency", [python, "-u", str(root / ROW_LINT_PATH), "--strict", candidate]),
        (
            "full_python_suite",
            [
                str(root / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-o",
                "addopts=",
                "-n",
                "0",
                "--no-cov",
                "--basetemp=/tmp/exp7302-v642-full",
            ],
        ),
    )


def run_required_validation(
    root: Path, raw_dir: Path, artifact: JsonDict, started: float
) -> list[JsonDict]:
    """Validate the candidate, then observe repository health last."""

    authority = Path(str(artifact["yaml_authority_path"]))
    candidate_path = raw_dir / RAW_CANDIDATE_PATH.name
    commands = validation_commands(root, candidate_path, authority)
    log_dir = raw_dir / "validation"
    preterminal = shipped.base._execute_commands(
        root, log_dir, commands[:PRETERMINAL_VALIDATION_COUNT]
    )
    artifact["validation_receipts"] = list(preterminal)
    artifact["baseline_failures"] = [row for row in preterminal if row["passed"] is not True]
    _apply_terminal_state(artifact)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["measured_candidate_path"] = str(candidate_path.relative_to(root))
    shipped.base._set_time_and_checksum(artifact, started)
    _atomic_write_json(candidate_path, artifact)
    progress(6, "candidate written", f"terminal candidate={candidate_path.relative_to(root)}")
    affected_post = shipped.base._execute_commands(
        root, log_dir, commands[PRETERMINAL_VALIDATION_COUNT:-1]
    )
    repository = shipped.base._execute_commands(root, log_dir, commands[-1:])
    rows = [*preterminal, *affected_post, *repository]
    artifact["repository_health"] = _repository_health(
        rows, artifact["repository_health"]["historical_failures"]
    )
    return rows


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    fetcher: Fetcher = _fetch_url,
) -> JsonDict:
    """Measure and atomically publish an exact, disqualified, or blocked receipt."""

    token = _BUILD_ROOT.set(root)
    try:
        with _configured_shipped(runtime=True):
            return _BASE_BUILD_ARTIFACT(
                root,
                run_date,
                output_path=output_path,
                raw_dir=raw_dir,
                checkpoint_path=checkpoint_path,
                fetcher=fetcher,
            )
    finally:
        _BUILD_ROOT.reset(token)


def date_argument(value: str) -> str:
    """Accept only the fixed V642 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI integration.
    """Run the advisory receipt and accept every valid terminal class."""

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
        REPO_ROOT,
        args.date,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
    )
    errors = validate_artifact(artifact)
    if errors:
        print(f"[exp7302] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7302] complete verdict={artifact['honest_verdict']} score={artifact['contract_complete_score']}",
        flush=True,
    )
    return 0

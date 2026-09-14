"""Build the V641 source-ingestion and exact execution-contract receipt.

The module adapts shipped parsers, gate fixtures, lifecycle code, and validators.
It adds only the V641 phase comparison, source dispositions, and terminal field
names. External papers remain motivation rather than local scientific evidence.

Spec refs: REQ-REPORT-7288 and SCENARIO-REPORT-7288-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import platform
import re
import shlex
import sys
import tempfile
from typing import Any, Iterator

import yaml

from carnot import experiment_7219_v636_source_contract as gate_fixtures
from carnot import experiment_7274_v640_source_contract as base
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
MILESTONE = "2026.09.641"
RUN_DATE = "20260914"
RETRIEVAL_DATE = "2026-09-14"
FIRST_TASK_ID = "exp7288-source-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7289-arc-boundary",
    "exp7290-arc-selfparse",
    "exp7291-reuse-fixture",
    "exp7292-reuse-canary",
    "exp7293-reuse-measurement",
    "exp7294-reuse-audit",
    "exp7295-mixture-prototype",
    "exp7296-mixture-learning",
    "exp7297-mixture-audit",
    "exp7298-snapshot-journal",
    "exp7299-snapshot-cost",
    "exp7300-board-continuity",
    "exp7301-capstone",
)
EXPECTED_TASK_COUNT = 14
EXPECTED_GATE_EDGE_COUNT = 8
GATE_CASES_PER_EDGE = 5
RANDOM_SEED = 7_288_202_609_14

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PLAN_PATH = Path("ops/e2e-test-plan.md")
HISTORY_PATH = Path("results/experiment_7287_v640_capstone.json")

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
GATE_EVALUATOR_PATH = Path("scripts/conductor_gates.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
PROMPT_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")

MODULE_PATH = Path("python/carnot/experiment_7288_v641_source_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7288_v641_source_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7288_v641_source_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7288_v641_source_contract.json")
RAW_DIR = Path("results/raw/experiment_7288")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7288_v641_source_contract.json")

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
    "model_loads_in_flight": 0,
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "generation_calls_failed": 0,
    "generation_calls_in_flight": 0,
    "usable_answers": 0,
}
INFERENCE_SUBSTRATE = "cpu_exact_solver_or_simulator"
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
POSITIVE_VERDICT = "complete_circular_positive_v641_exact_advisory_contract"
DISQUALIFIED_VERDICT = "complete_disqualified_v641_contract_mismatched_or_invalid"
BLOCKED_VERDICT = "blocked_v641_source_contract_external_prerequisite_missing"

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the artifact; retain ordinary top-level experiment_id and milestone.",
    "status": "Use a terminal complete or blocked record; unfinished own work belongs in separate checkpoints.",
    "run_date": "Use 20260914, real UTC start/end and monotonic timing.",
    "field_principles": "Store explanations here while consumer values remain ordinary top-level values.",
    "preconditions_checked": "Hash actual inputs, authority boundaries, resource ownership and failed checks.",
    "MODEL_SPECS": "Actual executable local model identities; keep historical models in hashed sidecars.",
    "model_invoked": "True for any actual attempted model load or generation, including failed and unusable work.",
    "invocation_counts": "Separate attempted/completed/failed loads and generation; retain in-flight events on timeout.",
    "inference_substrate": "Use the recognized literal for actual computation; never infer from intended task.",
    "inference_substrate_class": "Full generation60s, bounded10s, load-only2s, or actual no-LLM class; never pad elapsed time.",
    "execution_venue": "Host is host; identify actual GPU/native/device execution separately.",
    "duration_s": "Measured monotonic elapsed and disjoint phase spans, including failures and initialization.",
    "random_seed": "Freeze development and independent evaluation seeds before observing outcomes.",
    "reproducibility_checksum": "Bind code, config, inputs, model identity if any and immutable raw evidence.",
    "source_artifact_hashes": "Keep exact producer identities, terminal classes, retirement and quarantine state.",
    "rows": "Every comparative unit/arm/seed with metric, cost, error, abstention and censoring; no aggregate-only claim.",
    "sample_size_budget": "Planned, attempted, complete and censored units plus the frozen stopping rule.",
    "acceptance_gate_results": "Each completeness/value check names expected, observed, passed and principle.",
    "gate_check_summary": "Every blocked_* verdict names upstream/check, exact field, observed and expected value.",
    "verifier_is_oracle": "Expose shared verifier/evaluator authority; same-authority mechanics are not learned correctness.",
    "honest_verdict": "Complete findings start complete_ or complete:; external absence starts blocked_; state the actual finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed efficacy gates forbid positive. Only own unfinished work is partial; unchanged external failure is terminal blocked.",
    "validation_receipts": "Command, exit code, elapsed time and log hash; preserve actual failures.",
    "contract_complete_score": "One only for an independently matched, valid fourteen-row contract; advisory only.",
    "contract_rows": "Literal ordered IDs, titles, phases, deliverables and structured gates from both files.",
    "source_dispositions": "Primary URL, version/date, observed access, adapted mechanism, falsifier and deferred boundary.",
    "gate_control_rows": "All five independent control outcomes per actual gate, including exact diagnostic values.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

ACCESS_SPECS: tuple[JsonDict, ...] = (
    {
        "access_id": "source_freshness",
        "url": "https://arxiv.org/abs/2605.27494",
        "title": "Grounded Cache Routing",
        "publication_or_version_date": "2026-05-26 / v1",
    },
    {
        "access_id": "fixed_budget_hypothesis_mixtures",
        "url": "https://arxiv.org/abs/2609.10873",
        "title": "When Validation Stops Learning",
        "publication_or_version_date": "2026-09-09 / v1",
    },
    {
        "access_id": "sqlite_persistent_journals",
        "url": "https://www.sqlite.org/atomiccommit.html",
        "title": "Atomic Commit In SQLite",
        "publication_or_version_date": "checked 2026-09-14",
    },
    {
        "access_id": "deferred_kan_tsu",
        "url": "https://arxiv.org/abs/2503.21076",
        "title": "Kolmogorov-Arnold Classifier",
        "publication_or_version_date": "2025-03-27 / v1",
    },
)

SOURCE_DISPOSITIONS: tuple[JsonDict, ...] = (
    {
        "method_family": "source_freshness",
        "disposition": "adapt",
        "adapted_mechanism": "hash source content and version, invalidate dependent compilations, and retain a warm-prefix direct comparator",
        "falsifier": "any stale serve, cached/fresh semantic mismatch, or full-cost failure against the warm direct arm",
        "deferred_boundary": "answer caching and external serving-stack speedups are not Carnot source-reuse evidence",
        "target_tasks": [
            "exp7291-reuse-fixture",
            "exp7292-reuse-canary",
            "exp7293-reuse-measurement",
            "exp7294-reuse-audit",
        ],
    },
    {
        "method_family": "fixed_budget_hypothesis_mixtures",
        "disposition": "adapt",
        "adapted_mechanism": "fixed-share voting over a bounded archive with a shared delayed-label schedule and future pre-label scoring",
        "falsifier": "future error, recurrence, false-accept, coverage, chronology, or byte-budget gate failure",
        "deferred_boundary": "the local mixture is not the paper's theorem and does not repeat V640 eight-label admission",
        "target_tasks": [
            "exp7295-mixture-prototype",
            "exp7296-mixture-learning",
            "exp7297-mixture-audit",
        ],
    },
    {
        "method_family": "sqlite_persistent_journals",
        "disposition": "implement",
        "adapted_mechanism": "store one complete native snapshot in a single-writer PERSIST/FULL SQLite transaction",
        "falsifier": "effective pragma mismatch, missing acknowledged state, recovery mismatch, or full-cost latency/throughput gate failure",
        "deferred_boundary": "process-kill recovery does not prove device firmware behavior or physical power-loss safety",
        "target_tasks": ["exp7298-snapshot-journal", "exp7299-snapshot-cost"],
    },
    {
        "method_family": "deferred_kan_tsu",
        "disposition": "defer",
        "adapted_mechanism": "retain KAC as a future comparator and count FPGA companion and host costs for any later TSU work",
        "falsifier": "reopen only with target-task predictor value plus authenticated local hardware or a compatible checkpoint",
        "deferred_boundary": "no KAN training, TSU execution, Kona benchmark, procurement, or vendor claim becomes local evidence",
        "target_tasks": ["exp7300-board-continuity", "exp7301-capstone"],
    },
)

VALIDATION_COMMAND_NAMES = (
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
    "full_python_suite",
    "exact_plan_e2e",
    "independent_reducer",
    "adversarial",
    "row_consistency",
)
PRETERMINAL_VALIDATION_COUNT = 13

_BASE_EVALUATE_CONTRACT = base.evaluate_contract
_BASE_SELECT_AUTHORITY = base.select_yaml_authority
_BASE_BASE_ARTIFACT = base._base_artifact
_BASE_PRECONDITIONS = base._preconditions
_BASE_BUILD_ARTIFACT = base.build_artifact
_BASE_RUN_REQUIRED_VALIDATION = base.run_required_validation


def progress(phase: int, state: str, detail: str) -> None:
    """Flush every boundary so a long validator remains externally visible."""

    print(f"[exp7288] phase {phase} {state}: {detail}", flush=True)


def sha256(path: Path) -> str:
    """Use the repository's shared prefixed file digest."""

    return base.sha256(path)


def sha256_bytes(content: bytes) -> str:
    """Use the same digest representation for immutable byte evidence."""

    return base.sha256_bytes(content)


@contextmanager
def _configured_parser() -> Iterator[None]:
    """Parameterize the shipped parser with V641 identities, then restore it."""

    replacements = {
        "MILESTONE": MILESTONE,
        "RUN_DATE": RUN_DATE,
        "RETRIEVAL_DATE": RETRIEVAL_DATE,
        "FIRST_TASK_ID": FIRST_TASK_ID,
        "EXPECTED_ID_ORDER": EXPECTED_ID_ORDER,
        "EXPECTED_TASK_COUNT": EXPECTED_TASK_COUNT,
        "EXPECTED_GATE_EDGE_COUNT": EXPECTED_GATE_EDGE_COUNT,
        "RANDOM_SEED": RANDOM_SEED,
        "DESIGN_PATH": DESIGN_PATH,
        "ACTIVE_ROADMAP_PATH": ACTIVE_ROADMAP_PATH,
        "NEXT_ROADMAP_PATH": NEXT_ROADMAP_PATH,
        "progress": progress,
    }
    previous = {name: getattr(base, name) for name in replacements}
    for name, value in replacements.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Select active V641 bytes first, then staged V641 bytes before activation."""

    with _configured_parser():
        return _BASE_SELECT_AUTHORITY(root)


def _markdown_phases(markdown_text: str) -> dict[str, int]:
    """Read phase cells from the literal table instead of trusting YAML."""

    lines = markdown_text.splitlines()
    for index, line in enumerate(lines):
        headers = [cell.strip().lower() for cell in line.strip().strip("|").split("|")]
        if "task id" not in headers or "phase" not in headers or "deliverable" not in headers:
            continue
        id_column = headers.index("task id")
        phase_column = headers.index("phase")
        phases: dict[str, int] = {}
        for row in lines[index + 2 :]:
            if not row.lstrip().startswith("|"):
                break
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            if max(id_column, phase_column) >= len(cells):
                continue
            try:
                phases[cells[id_column]] = int(cells[phase_column])
            except ValueError:
                continue
        return phases
    return {}


def _yaml_phase(task: Mapping[str, Any]) -> int | None:
    """Read an explicit phase or the prompt's declared execution phase."""

    explicit = task.get("phase")
    if isinstance(explicit, int) and not isinstance(explicit, bool):
        return explicit
    match = re.search(r"Milestone\s+2026\.09\.641,\s*phase\s+(\d+)", str(task.get("prompt", "")))
    return int(match.group(1)) if match else None


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare shipped fields and independently extracted phase values."""

    with _configured_parser():
        contract = _BASE_EVALUATE_CONTRACT(markdown_text, yaml_document)
    phases = _markdown_phases(markdown_text)
    tasks = yaml_document.get("tasks", []) if isinstance(yaml_document, Mapping) else []
    yaml_phases = {
        str(task.get("id")): _yaml_phase(task)
        for task in tasks
        if isinstance(task, Mapping) and task.get("id")
    }
    for row in contract["markdown_task_rows"]:
        row["phase"] = phases.get(str(row.get("id")))
    for row in contract["yaml_task_rows"]:
        row["phase"] = yaml_phases.get(str(row.get("id")))
    for row in contract["contract_rows"]:
        task_id = str(row["unit_id"])
        markdown_row = row.get("markdown")
        yaml_row = row.get("yaml")
        if isinstance(markdown_row, dict):
            markdown_row["phase"] = phases.get(task_id)
        if isinstance(yaml_row, dict):
            yaml_row["phase"] = yaml_phases.get(task_id)
        row["checks"]["phase"] = (
            isinstance(markdown_row, Mapping)
            and isinstance(yaml_row, Mapping)
            and markdown_row.get("phase") == yaml_row.get("phase")
            and markdown_row.get("phase") in {1, 2, 3, 4}
        )
        row["passed"] = all(row["checks"].values())
        row["cost"] = None
        row["censored"] = False
    contract["passed"] = (
        contract.get("markdown_milestone") == contract.get("yaml_milestone") == MILESTONE
        and len(contract["contract_rows"]) == EXPECTED_TASK_COUNT
        and [row["unit_id"] for row in contract["contract_rows"]] == list(EXPECTED_ID_ORDER)
        and all(row["passed"] for row in contract["contract_rows"])
    )
    return contract


def run_gate_replays(roadmap: Mapping[str, Any], results_dir: Path) -> list[JsonDict]:
    """Reuse five isolated real evaluator and quarantine controls per gate."""

    previous = gate_fixtures.progress
    gate_fixtures.progress = progress
    try:
        rows = gate_fixtures.run_gate_validation_fixtures(roadmap, results_dir)
    finally:
        gate_fixtures.progress = previous
    names = {
        "passing": "passing",
        "failed": "false",
        "missing_field": "missing_field",
        "missing_file": "missing_file",
        "quarantined_passing": "quarantined",
    }
    for row in rows:
        row["case"] = names[str(row["case"])]
        row["artifact_field"] = row["field"]
        row["expected_value"] = row["expected"]
        row["observed_value"] = row["actual"]
    return rows


def gate_replays_complete(rows: object) -> bool:
    """Require all five distinct observations and both authority outcomes."""

    if not isinstance(rows, list) or len(rows) != EXPECTED_GATE_EDGE_COUNT * 5:
        return False
    expected_cases = ["passing", "false", "missing_field", "missing_file", "quarantined"]
    for edge_index in range(EXPECTED_GATE_EDGE_COUNT):
        edge = [row for row in rows if row.get("edge_index") == edge_index]
        if [row.get("case") for row in edge] != expected_cases:
            return False
        if [row.get("conductor_passed") for row in edge] != [True, False, False, False, True]:
            return False
        if [row.get("experiment_precondition_passed") for row in edge] != [
            True,
            False,
            False,
            False,
            False,
        ]:
            return False
        if edge[1].get("observed_value") != 0:
            return False
    return True


def observed_version(body: str) -> str | None:
    """Return only an explicit source version marker."""

    return base.observed_version(body)


def collect_source_rows(
    fetcher: Fetcher = _fetch_url,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Make four bounded requests and keep access separate from disposition."""

    access_rows: list[JsonDict] = []
    for index, spec in enumerate(ACCESS_SPECS, 1):
        progress(4, "request start", f"completed={index - 1}/4 {spec['access_id']}")
        try:
            receipt = dict(fetcher(str(spec["url"])))
        except Exception as exc:  # noqa: BLE001 - failed access is a source observation.
            receipt = {
                "ok": False,
                "status_code": None,
                "body": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        body = str(receipt.get("body", ""))
        accessed = receipt.get("ok") is True
        row = {
            "access_id": spec["access_id"],
            "kind": "primary_record",
            "primary_url": spec["url"],
            "retrieval_date": RETRIEVAL_DATE,
            "publication_or_version_date": spec["publication_or_version_date"],
            "timeout_s": 20,
            "status_code": receipt.get("status_code"),
            "error": receipt.get("error"),
            "response_body": body,
            "response_sha256": sha256_bytes(body.encode()),
            "observed_version": observed_version(body),
            "title_verified": str(spec["title"]).lower() in body.lower() if accessed else None,
            "observed_access": "http_success" if accessed else "access_failed",
            "access_status": "http_success" if accessed else "access_failed",
        }
        access_rows.append(row)
        progress(4, "request end", f"completed={index}/4 {row['observed_access']}")
    access_by_id = {str(row["access_id"]): row for row in access_rows}
    source_rows: list[JsonDict] = []
    for disposition in SOURCE_DISPOSITIONS:
        row = deepcopy(disposition)
        access = access_by_id[str(row["method_family"])]
        row.update(
            primary_url=access["primary_url"],
            retrieval_date=access["retrieval_date"],
            publication_or_version_date=access["publication_or_version_date"],
            observed_version=access["observed_version"],
            observed_access=access["observed_access"],
            access_error=access["error"],
            access_status_code=access["status_code"],
            local_evidence_claimed=False,
        )
        source_rows.append(row)
    return source_rows, access_rows


def source_rows_complete(rows: object) -> bool:
    """Require all four source families even when a bounded fetch failed."""

    expected = {row["method_family"] for row in SOURCE_DISPOSITIONS}
    return (
        isinstance(rows, list)
        and len(rows) == 4
        and {row.get("method_family") for row in rows} == expected
        and {row.get("disposition") for row in rows} == {"adapt", "implement", "defer"}
        and all(
            row.get("primary_url")
            and row.get("publication_or_version_date")
            and row.get("observed_access") in {"http_success", "access_failed"}
            and row.get("adapted_mechanism")
            and row.get("falsifier")
            and row.get("deferred_boundary")
            and row.get("target_tasks")
            and row.get("local_evidence_claimed") is False
            for row in rows
        )
    )


def build_sidecars(
    root: Path,
    raw_dir: Path,
    gate_rows: Sequence[Mapping[str, Any]],
    access_rows: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Store historical models, control fixtures, and response bytes separately."""

    history_data = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    if not isinstance(history_data, Mapping):
        raise ValueError(f"historical artifact is not a mapping: {HISTORY_PATH}")
    payloads = {
        "history": (
            "historical-model-receipts.json",
            {
                "schema": "carnot.exp7288.historical_model_receipts.v1",
                "current_invocation": {
                    "MODEL_SPECS": [],
                    "model_invoked": False,
                    "invocation_counts": ZERO_INVOCATION_COUNTS,
                },
                "historical_artifacts": [
                    {
                        "path": str(HISTORY_PATH),
                        "sha256": sha256(root / HISTORY_PATH),
                        "status": history_data.get("status"),
                        "verdict_class": history_data.get("verdict_class"),
                        "honest_verdict": history_data.get("honest_verdict"),
                        "MODEL_SPECS": history_data.get("MODEL_SPECS"),
                        "model_invoked": history_data.get("model_invoked"),
                        "quarantined": base._is_quarantined(dict(history_data)),
                        "retired": False,
                        "accepted_for_current_values": False,
                    }
                ],
            },
        ),
        "negative_fixtures": (
            "negative-gate-fixtures.json",
            {
                "schema": "carnot.exp7288.gate_controls.v1",
                "rows": [dict(row) for row in gate_rows if row.get("case") != "passing"],
            },
        ),
        "source_access": (
            "source-access-receipts.json",
            {
                "schema": "carnot.exp7288.source_access.v1",
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


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Attach prompt principles while ordinary values stay at top level."""

    values = {field: "This field keeps one V641 contract fact auditable." for field in fields}
    values.update(REQUIRED_FIELD_PRINCIPLES)
    return values


def _base_artifact(run_date: str, checkpoint_path: Path, started_at: str) -> JsonDict:
    """Adapt the shipped running checkpoint without making it success-shaped."""

    artifact = _BASE_BASE_ARTIFACT(run_date, checkpoint_path, started_at)
    artifact.update(
        schema="carnot.exp7288.v641_source_contract.v1",
        experiment_id=FIRST_TASK_ID,
        milestone=MILESTONE,
        random_seed=RANDOM_SEED,
        invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        honest_verdict="partial_running_v641_source_contract",
        contract_complete_score=0,
    )
    artifact.pop("source_contract_complete_score", None)
    artifact["expected_id_order"] = list(EXPECTED_ID_ORDER)
    artifact["sample_size_budget"]["gate_controls"].update(planned=40, independent_units=40)
    artifact["sample_size_budget"]["source_dispositions"].update(planned=4, independent_units=4)
    artifact["sample_size_budget"]["stopping_rule"] = (
        "Stop after fourteen contract rows, five real controls for each of eight gates, "
        "four source dispositions, four bounded requests, and the declared validation sequence."
    )
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Reuse shipped ownership checks but authenticate the V641 requirement."""

    rows, authority, roadmap, content = _BASE_PRECONDITIONS(
        root, output_path, raw_dir, checkpoint_path
    )
    for row in rows:
        if row.get("check") == "v639_terminal_capstone":
            row["check"] = "v640_terminal_capstone"
        if row.get("check") == "driving_requirement":
            try:
                found = "REQ-REPORT-7288" in (root / SPEC_PATH).read_text(encoding="utf-8")
            except OSError:
                found = False
            row.update(
                expected_value="REQ-REPORT-7288",
                observed_value="REQ-REPORT-7288" if found else "missing",
                available=found,
            )
    return rows, authority, roadmap, content


def _freeze_authorities(
    root: Path, authority: Path, yaml_bytes: bytes, raw_dir: Path
) -> list[JsonDict]:
    """Freeze original Markdown and YAML bytes without normalization."""

    sources = (
        ("markdown", DESIGN_PATH, (root / DESIGN_PATH).read_bytes(), raw_dir / "v641-design.md"),
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
    """Require every named sidecar to match the authenticated hash map."""

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
    """Require every declared command and preserve any nonzero result."""

    rows = artifact.get("validation_receipts")
    return (
        isinstance(rows, list)
        and [row.get("name") for row in rows] == list(VALIDATION_COMMAND_NAMES)
        and all(row.get("passed") is True for row in rows)
    )


def _acceptance_gate_results(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute structural acceptance without trusting a stored score."""

    contract = artifact.get("contract_rows")
    sources = artifact.get("source_dispositions")
    access = artifact.get("source_access_rows")
    raw = artifact.get("raw_authority_rows")
    hashes = artifact.get("source_artifact_hashes")
    criteria = (
        (
            "contract_exact",
            list(EXPECTED_ID_ORDER),
            [row.get("unit_id") for row in contract] if isinstance(contract, list) else None,
            isinstance(contract, list)
            and len(contract) == 14
            and [row.get("unit_id") for row in contract] == list(EXPECTED_ID_ORDER)
            and all(row.get("passed") is True for row in contract)
            and artifact.get("markdown_milestone") == artifact.get("yaml_milestone") == MILESTONE,
            "Both byte authorities must independently agree on all five task dimensions.",
        ),
        (
            "gate_controls",
            40,
            len(artifact.get("gate_control_rows", []))
            if isinstance(artifact.get("gate_control_rows"), list)
            else None,
            gate_replays_complete(artifact.get("gate_control_rows")),
            "Each real gate keeps five evaluator and quarantine observations.",
        ),
        (
            "source_dispositions",
            4,
            len(sources) if isinstance(sources, list) else None,
            source_rows_complete(sources),
            "Source access can fail without becoming unrelated science evidence.",
        ),
        (
            "request_budget",
            "at most four primary URLs",
            len(access) if isinstance(access, list) else None,
            isinstance(access, list)
            and len(access) <= 4
            and len(access) == 4
            and all(row.get("kind") == "primary_record" for row in access),
            "The dated refresh stays bounded to four primary requests.",
        ),
        (
            "raw_authorities",
            2,
            len(raw) if isinstance(raw, list) else None,
            isinstance(raw, list)
            and len(raw) == 2
            and all(row.get("hash_matches") is True for row in raw),
            "Original Markdown and YAML bytes remain independently recheckable.",
        ),
        (
            "sidecars",
            True,
            _sidecars_complete(artifact),
            _sidecars_complete(artifact),
            "Historical models and injected fixtures do not enter current counters.",
        ),
        (
            "source_hashes",
            "all present",
            "all present"
            if isinstance(hashes, Mapping) and hashes and all(hashes.values())
            else "missing",
            isinstance(hashes, Mapping) and bool(hashes) and all(hashes.values()),
            "Every consumed local input and raw sidecar has an exact digest.",
        ),
        (
            "validation",
            list(VALIDATION_COMMAND_NAMES),
            [row.get("name") for row in artifact.get("validation_receipts", [])]
            if isinstance(artifact.get("validation_receipts"), list)
            else None,
            _validation_complete(artifact),
            "Every scoped check keeps its real command, exit, elapsed time, and log hash.",
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
    """Name the exact external, contract, or acceptance failure."""

    failed = base._failed_precondition(artifact)
    if failed is not None:
        return base._summary(
            str(failed.get("check")),
            failed.get("expected_value"),
            failed.get("observed_value"),
            upstream=failed.get("upstream"),
            field=failed.get("field"),
        )
    if artifact.get("markdown_milestone") != MILESTONE:
        return base._summary(
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
            return base._summary(
                "contract_parity",
                row.get("yaml"),
                row.get("markdown"),
                upstream=row.get("unit_id"),
                field="id|order|title|deliverable|phase|gates",
            )
    acceptance = _acceptance_gate_results(artifact)
    row = next((value for value in acceptance if value["passed"] is not True), None)
    return base._summary(
        f"acceptance:{row['criterion']}" if row else "stored_contract",
        row.get("expected") if row else "complete",
        row.get("observed") if row else "incomplete",
        field=row.get("criterion") if row else "acceptance_gate_results",
    )


def _apply_terminal_state(artifact: JsonDict) -> None:
    """Derive terminal lifecycle solely from stored evidence."""

    artifact["acceptance_gate_results"] = _acceptance_gate_results(artifact)
    artifact["contract_complete_score"] = int(
        len(artifact["acceptance_gate_results"]) == 8
        and all(row["passed"] for row in artifact["acceptance_gate_results"])
    )
    failed = base._failed_precondition(artifact)
    if failed is not None:
        artifact.update(
            status="blocked",
            inference_substrate="blocked_no_run",
            inference_substrate_class="blocked_no_run",
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
            gate_check_summary=base._summary(
                None,
                "all V641 advisory contract criteria pass",
                "all V641 advisory contract criteria pass",
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
    """Reduce raw rows without trusting aggregate scores or verdict text."""

    errors: list[str] = []
    contract = artifact.get("contract_rows")
    if not isinstance(contract, list) or len(contract) != 14:
        errors.append("contract_row_count")
    elif any(
        row.get("passed") != all(row.get("checks", {}).values())
        or row.get("markdown", {}).get("phase") not in {1, 2, 3, 4}
        or row.get("censored") is not False
        for row in contract
    ):
        errors.append("contract_row_reduction")
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
    """Recompute required fields, rows, hashes, terminal state, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if artifact.get("schema") != "carnot.exp7288.v641_source_contract.v1":
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
    expected = dict(artifact)
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
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"{field}_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def e2e_contract_receipt(root: Path, temp_dir: Path) -> JsonDict:
    """Run exact plan copies through parse, real gates, and a diagnostic hash."""

    temp_dir.mkdir(parents=True, exist_ok=True)
    markdown_bytes = (root / DESIGN_PATH).read_bytes()
    yaml_bytes = (root / ACTIVE_ROADMAP_PATH).read_bytes()
    markdown_copy = temp_dir / "plan.md"
    yaml_copy = temp_dir / "plan.yaml"
    _atomic_write_bytes(markdown_copy, markdown_bytes)
    _atomic_write_bytes(yaml_copy, yaml_bytes)
    roadmap = yaml.safe_load(yaml_bytes)
    contract = evaluate_contract(markdown_bytes.decode(), roadmap)
    gates = run_gate_replays(roadmap, temp_dir / "gate-inputs")
    payload = {
        "markdown_sha256": sha256(markdown_copy),
        "yaml_sha256": sha256(yaml_copy),
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
    """Return the exact scoped checks and one required full Python-suite run."""

    python = str(root / ".venv/bin/python")
    roadmap = str(root / authority)
    candidate = str(candidate_path)
    coverage_file = "/tmp/.coverage-exp7288-v641"
    pytest_base = "/tmp/exp7288-v641-focused"
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    e2e_code = (
        "import pathlib,sys,tempfile;"
        "from carnot.experiment_7288_v641_source_contract import e2e_contract_receipt;"
        f"receipt=e2e_contract_receipt(pathlib.Path({str(root)!r}),pathlib.Path(tempfile.mkdtemp(prefix='exp7288-e2e-',dir='/tmp')));"
        "print(receipt);sys.exit(not receipt['passed'])"
    )
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7288_v641_source_contract import independent_reduce;"
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
                "--basetemp=/tmp/exp7288-v641-full",
            ],
        ),
        ("exact_plan_e2e", [python, "-u", "-c", e2e_code]),
        ("independent_reducer", [python, "-u", "-c", reducer_code]),
        ("adversarial", [python, "-u", str(root / ADVERSARIAL_PATH), candidate]),
        ("row_consistency", [python, "-u", str(root / ROW_LINT_PATH), candidate]),
    )


run_required_validation = _BASE_RUN_REQUIRED_VALIDATION


@contextmanager
def _configured_runtime() -> Iterator[None]:
    """Route the shipped lifecycle through the small V641-specific adapters."""

    replacements = {
        "MILESTONE": MILESTONE,
        "RUN_DATE": RUN_DATE,
        "RETRIEVAL_DATE": RETRIEVAL_DATE,
        "FIRST_TASK_ID": FIRST_TASK_ID,
        "EXPECTED_ID_ORDER": EXPECTED_ID_ORDER,
        "EXPECTED_TASK_COUNT": EXPECTED_TASK_COUNT,
        "EXPECTED_GATE_EDGE_COUNT": EXPECTED_GATE_EDGE_COUNT,
        "RANDOM_SEED": RANDOM_SEED,
        "DESIGN_PATH": DESIGN_PATH,
        "ACTIVE_ROADMAP_PATH": ACTIVE_ROADMAP_PATH,
        "NEXT_ROADMAP_PATH": NEXT_ROADMAP_PATH,
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
        "ZERO_INVOCATION_COUNTS": ZERO_INVOCATION_COUNTS,
        "INFERENCE_SUBSTRATE": INFERENCE_SUBSTRATE,
        "INFERENCE_SUBSTRATE_CLASS": INFERENCE_SUBSTRATE_CLASS,
        "POSITIVE_VERDICT": POSITIVE_VERDICT,
        "DISQUALIFIED_VERDICT": DISQUALIFIED_VERDICT,
        "BLOCKED_VERDICT": BLOCKED_VERDICT,
        "REQUIRED_FIELD_PRINCIPLES": REQUIRED_FIELD_PRINCIPLES,
        "REQUIRED_ARTIFACT_FIELDS": REQUIRED_ARTIFACT_FIELDS,
        "SOURCE_DISPOSITIONS": SOURCE_DISPOSITIONS,
        "VALIDATION_COMMAND_NAMES": VALIDATION_COMMAND_NAMES,
        "PRETERMINAL_VALIDATION_COUNT": PRETERMINAL_VALIDATION_COUNT,
        "progress": progress,
        "select_yaml_authority": select_yaml_authority,
        "evaluate_contract": evaluate_contract,
        "run_gate_replays": run_gate_replays,
        "gate_replays_complete": gate_replays_complete,
        "collect_source_rows": collect_source_rows,
        "build_sidecars": build_sidecars,
        "_field_principles": _field_principles,
        "_base_artifact": _base_artifact,
        "_preconditions": _preconditions,
        "_freeze_authorities": _freeze_authorities,
        "_source_rows_complete": source_rows_complete,
        "_sidecars_complete": _sidecars_complete,
        "_validation_complete": _validation_complete,
        "_acceptance_gate_results": _acceptance_gate_results,
        "_score_from_artifact": lambda artifact: int(
            all(row["passed"] for row in _acceptance_gate_results(artifact))
        ),
        "_failure_summary": _failure_summary,
        "_apply_terminal_state": _apply_terminal_state,
        "independent_reduce": independent_reduce,
        "validate_artifact": validate_artifact,
        "validation_commands": validation_commands,
        "run_required_validation": run_required_validation,
    }
    previous = {name: getattr(base, name) for name in replacements}
    for name, value in replacements.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


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

    with _configured_runtime():
        return _BASE_BUILD_ARTIFACT(
            root,
            run_date,
            output_path=output_path,
            raw_dir=raw_dir,
            checkpoint_path=checkpoint_path,
            fetcher=fetcher,
        )


def date_argument(value: str) -> str:
    """Accept only the fixed V641 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI integration.
    """Run the advisory contract and accept every internally valid terminal class."""

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
        print(f"[exp7288] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7288] complete verdict={artifact['honest_verdict']} score={artifact['contract_complete_score']}",
        flush=True,
    )
    return 0

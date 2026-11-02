"""Build the V636 source and exact task-contract receipt.

The V635 receipt already ships reliable Markdown, YAML, hash, and atomic-write
helpers. This module reuses those helpers and keeps only the V636 policy here.
It invokes no model and makes no scientific-value claim.

Spec refs: REQ-REPORT-7219 and SCENARIO-REPORT-7219-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
import html
import json
import platform
from pathlib import Path
import re
import shlex
import sys
import tempfile
import time
from typing import Any, Iterator

import yaml

from carnot import experiment_7205_v635_source_contract as base
from carnot.experiment_7179_v633_contract_receipt import (
    _atomic_write_json,
    _run_streaming_command,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - depends on the caller's path.
    sys.path.insert(0, str(SCRIPTS_DIR))

from adversarial_verify import (  # noqa: E402
    SUBSTRATE_CLASS_FLOORS,
    check_substrate_class,
    duration_floor_for_artifact,
)
from conductor_gates import _is_quarantined, _upstream_is_final, evaluate_gates  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]

MILESTONE = "2026.09.636"
RUN_DATE = "20260911"
RETRIEVAL_DATE = "2026-09-11"
PLANNING_CUTOFF = RETRIEVAL_DATE
FIRST_TASK_ID = "exp7219-source-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7220-xml-canary",
    "exp7221-arc-session",
    "exp7222-span-fixture",
    "exp7223-span-canary",
    "exp7224-span-capture",
    "exp7225-semantics-audit",
    "exp7226-belief-compiler",
    "exp7227-belief-learning",
    "exp7228-belief-cold-audit",
    "exp7229-rare-event-audit",
    "exp7230-native-belief",
    "exp7231-board-continuity",
    "exp7232-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
EXPECTED_GATE_EDGE_COUNT = 7
GATE_CASES_PER_EDGE = 5
RANDOM_SEED = 7_219_202_609_11

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
COMPLETE_PATH = Path("research-complete.yaml")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
V635_RECEIPT_PATH = Path("results/experiment_7205_v635_source_contract.json")
V635_CAPSTONE_PATH = Path("results/experiment_7218_v635_capstone.json")
V635_CAPSTONE_MODULE_PATH = Path("python/carnot/experiment_7218_v635_capstone.py")
V635_CAPSTONE_WRAPPER_PATH = Path("scripts/experiments/experiment_7218_v635_capstone.py")
MODULE_PATH = Path("python/carnot/experiment_7219_v636_source_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7219_v636_source_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7219_v636_source_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7219_v636_source_contract.json")
RAW_DIR = Path("results/raw/experiment_7219")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7219_v636_source_contract.json")
RAW_MARKDOWN_NAME = "research-roadmap-vNEXT.md"
RAW_YAML_NAME = "selected-roadmap.yaml"

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
INVENTED_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ARC_LINT_PATH = Path("scripts/arc_levelup_guarantee_lint.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")
TOOL_PATHS = (
    ROADMAP_SCHEMA_PATH,
    Path("scripts/conductor_gates.py"),
    PRIOR_LINT_PATH,
    EXCLUSION_LINT_PATH,
    INVENTED_PATH_LINT_PATH,
    ARC_LINT_PATH,
    GATE_LINT_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    SPEC_COVERAGE_PATH,
)
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    REFERENCE_PATH,
    COMPLETE_PATH,
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    DESIGN_PATH,
    SPEC_PATH,
    V635_RECEIPT_PATH,
    V635_CAPSTONE_PATH,
    Path("results/experiment_7208_v635_span_fixture.json"),
    V635_CAPSTONE_MODULE_PATH,
    V635_CAPSTONE_WRAPPER_PATH,
    Path("python/carnot/experiment_7205_v635_source_contract.py"),
    Path("python/carnot/experiment_7179_v633_contract_receipt.py"),
    *TOOL_PATHS,
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
POSITIVE_VERDICT = "complete_positive_v636_source_contract_exact_agreement"
DISQUALIFIED_VERDICT = "complete_disqualified_v636_source_contract_mismatch"
BLOCKED_VERDICT = "blocked_v636_source_contract_prerequisite_missing"

REQUIRED_FIELD_PRINCIPLES = {
    "field_principles": "Annotate actual values in this map; do not wrap arbitrary dictionaries as principle/value records.",
    "status": "Write a terminal artifact only when done or externally blocked; running checkpoints use a different path.",
    "run_date": "Use 20260911 and record actual UTC timestamps, never copy an upstream run date.",
    "preconditions_checked": "Actual code, resource, identity and gate observations before expensive work.",
    "inference_substrate": "Use the recognized literal for the work actually executed; custom free text caused the Exp7208 quarantine.",
    "inference_substrate_class": "Match actual generation, load-only, CPU or aggregation work and its duration floor.",
    "execution_venue": "Exactly host, kv260, gatemate or polarfire; the top-level orchestration here is host.",
    "execution_host": "Actual hostname separate from venue.",
    "duration_s": "Measured monotonic work time; no padding or reclassification to evade a floor.",
    "source_artifact_hashes": "Bind code, source documents, manifests and raw evidence to claims.",
    "rows": "Per unit/arm/seed metric, error and abstention for every comparison; retain full denominators.",
    "sample_size_budget": "Planned, attempted, completed, censored and independent units; no silent removal.",
    "random_seed": "Freeze random choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash exact source, inputs, settings and raw unit rows.",
    "gate_check_summary": "Every blocked_* verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when correctness authority is reused as the verifier; independent code alone is not distinct authority.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. partial means unfinished own work only.",
    "honest_verdict": "Use complete_ or complete: for completed findings; blocked_* for external absence. A failed acceptance gate forbids positive.",
    "MODEL_SPECS": "Only models actually invoked; [] for CPU/aggregation, mandated Qwen3.8 for every model task.",
    "model_invoked": "True only for actual model execution; upstream model outputs are cached evidence.",
    "source_contract_complete_score": "One means structural/source receipt complete, not scientific value.",
    "contract_rows": "All fourteen independently parsed task records.",
    "activation_validation_rows": "Real validators and isolated gate cases, including negative controls.",
    "source_method_rows": "Primary URL, date, access outcome and bounded use.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

SOURCES: tuple[JsonDict, ...] = (
    {
        "source_id": "doi:10.3233/FAIA250893",
        "url": "https://journals.sagepub.com/doi/10.3233/FAIA250893",
        "title": "Learning Compact Representations of Constraint Networks",
        "planning_date": "2026-08-25",
        "method_boundary": "Carnot's bitset vote is not the paper's template-learning algorithm.",
        "target_task": "exp7226-belief-compiler",
    },
    {
        "source_id": "arxiv:2607.20792",
        "url": "https://arxiv.org/abs/2607.20792",
        "title": "Memoir: Should a Model Write to Its Memory While It Thinks?",
        "planning_date": "2026-07-22",
        "method_boundary": "One procedural-memory result motivates a delayed-write control, not a universal theorem.",
        "target_task": "exp7228-belief-cold-audit",
    },
    {
        "source_id": "vllm:tool-calling",
        "url": "https://docs.vllm.ai/en/latest/features/tool_calling/",
        "title": "Tool Calling - vLLM",
        "planning_date": RETRIEVAL_DATE,
        "method_boundary": "Documented Qwen3-Coder parser support does not establish Qwen3.8 compatibility.",
        "target_task": "exp7220-xml-canary",
    },
    {
        "source_id": "vllm:gguf",
        "url": "https://docs.vllm.ai/en/latest/features/quantization/gguf/",
        "title": "GGUF - vLLM",
        "planning_date": RETRIEVAL_DATE,
        "method_boundary": "Experimental GGUF support is not evidence that the local mandated model loads.",
        "target_task": "exp7220-xml-canary",
    },
    {
        "source_id": "arxiv:2512.12850",
        "url": "https://arxiv.org/abs/2512.12850",
        "title": "KANELÉ: Kolmogorov-Arnold Networks for Efficient LUT-based Evaluation",
        "planning_date": "2025-12-14",
        "method_boundary": "A bitset constraint controller is not a KAN and inherits no FPGA speedup.",
        "target_task": "exp7230-native-belief",
    },
)

VALIDATION_COMMAND_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "exclusion_manifest",
    "invented_paths",
    "arc_floor",
    "gate_declarations",
    "focused_tests",
    "focused_coverage",
    "scoped_spec_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "artifact",
    "adversarial",
    "row_consistency",
)
PRETERMINAL_VALIDATION_NAMES = VALIDATION_COMMAND_NAMES[:12]


def progress(phase: int, state: str, detail: str) -> None:
    """Flush observed phase state so a bounded operation never appears silent."""

    print(f"[exp7219] phase {phase} {state}: {detail}", flush=True)


@contextmanager
def _configured_base() -> Iterator[None]:
    """Temporarily give the shipped parser helpers the V636 constants.

    The context restores every value. This prevents V636 tests from changing
    V635 behavior when both modules share one Python process.
    """

    replacements = {
        "MILESTONE": MILESTONE,
        "RUN_DATE": RUN_DATE,
        "PLANNING_CUTOFF": PLANNING_CUTOFF,
        "FIRST_TASK_ID": FIRST_TASK_ID,
        "EXPECTED_ID_ORDER": EXPECTED_ID_ORDER,
        "EXPECTED_TASK_COUNT": EXPECTED_TASK_COUNT,
        "RANDOM_SEED": RANDOM_SEED,
        "V634_RECEIPT_PATH": V635_RECEIPT_PATH,
        "V634_CAPSTONE_PATH": V635_CAPSTONE_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "DEFAULT_OUTPUT_PATH": DEFAULT_OUTPUT_PATH,
        "RAW_DIR": RAW_DIR,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "SOURCE_PATHS": SOURCE_PATHS,
        "POSITIVE_VERDICT": POSITIVE_VERDICT,
        "DISQUALIFIED_VERDICT": DISQUALIFIED_VERDICT,
        "BLOCKED_VERDICT": BLOCKED_VERDICT,
        "REQUIRED_FIELD_PRINCIPLES": REQUIRED_FIELD_PRINCIPLES,
        "REQUIRED_ARTIFACT_FIELDS": REQUIRED_ARTIFACT_FIELDS,
        "_progress": progress,
    }
    previous = {name: getattr(base, name) for name in replacements}
    for name, value in replacements.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Reuse the independent shipped parsers with the exact V636 roster."""

    normalized_markdown = markdown_text.replace(
        "| Order | Task ID | Title | Deliverable | Structured gates |",
        "| Order | Task ID | Exact title | Deliverable | Structured gate |",
        1,
    )
    with _configured_base():
        return base.evaluate_contract(normalized_markdown, yaml_document)


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Prefer active V636 bytes and use staged V636 bytes only as fallback."""

    with _configured_base():
        return base._select_yaml_authority(root)


def unwrap_principle_value(value: Any) -> Any:
    """Unwrap only the shared mapping that contains both required keys."""

    return base._unwrap_principle_value(value)


def upstream_precondition(data: Mapping[str, Any], field: str, expected: Any) -> JsonDict:
    """Apply quarantine authentication before accepting a structured value."""

    return base.upstream_precondition(data, field, expected)


def authenticate_upstream(data: object, field: str, expected: Any) -> JsonDict:
    """Authenticate one historical receipt without making it an authority."""

    mapping = data if isinstance(data, dict) else {}
    quarantined = _is_quarantined(mapping)
    terminal = _upstream_is_final(mapping)
    actual = unwrap_principle_value(mapping.get(field))
    known_failed = actual != expected or "disqualified" in str(
        unwrap_principle_value(mapping.get("honest_verdict", ""))
    )
    accepted = isinstance(data, dict) and terminal and not quarantined and not known_failed
    return {
        "checked_field": field,
        "expected_value": expected,
        "observed_value": actual,
        "mapping": isinstance(data, dict),
        "terminal": terminal,
        "quarantined": quarantined,
        "known_failed_value": known_failed,
        "accepted_for_evidence": accepted,
        "consumed": False,
        "authentication_checked": True,
    }


def upstream_intake_rows(root: Path) -> list[JsonDict]:
    """Record V635 history while rejecting failed or unauthenticated input."""

    rows: list[JsonDict] = []
    for relative, field in (
        (V635_RECEIPT_PATH, "source_contract_complete_score"),
        (V635_CAPSTONE_PATH, "capstone_complete_score"),
    ):
        progress(1, "check start", f"historical intake {relative}")
        path = root / relative
        try:
            data: object = json.loads(path.read_text(encoding="utf-8"))
            error = None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            data = None
            error = f"{type(exc).__name__}: {exc}"
        row = authenticate_upstream(data, field, 1)
        row.update(
            {
                "source_path": str(relative),
                "source_sha256": base._sha256(path) if error is None else None,
                "error": error,
                "reason": error
                or (
                    "authenticated_history_not_contract_authority"
                    if row["accepted_for_evidence"]
                    else "failed_or_unauthenticated_history_rejected"
                ),
            }
        )
        rows.append(row)
        progress(
            1, "check end", f"historical intake {relative} accepted={row['accepted_for_evidence']}"
        )
    return rows


def run_gate_validation_fixtures(roadmap: Mapping[str, Any], results_dir: Path) -> list[JsonDict]:
    """Exercise five real evaluator/authentication cases for every V636 edge."""

    edges = [
        (str(task.get("id")), dict(gate))
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
        for gate in task.get("gated_on", [])
        if isinstance(gate, Mapping)
    ]
    rows: list[JsonDict] = []
    case_values: tuple[tuple[str, object], ...] = (
        ("passing", 1),
        ("failed", 0),
        ("missing_field", ...),
        ("missing_file", None),
        ("quarantined_passing", 1),
    )
    for edge_index, (consumer, gate) in enumerate(edges):
        for case, value in case_values:
            progress(4, "check start", f"gate edge {edge_index + 1}/{len(edges)} case={case}")
            case_dir = results_dir / f"edge_{edge_index}_{case}"
            case_dir.mkdir(parents=True, exist_ok=True)
            field = str(gate["artifact_field"])
            data: JsonDict = {"status": "complete"}
            if value is not ...:
                data[field] = value
            if case == "quarantined_passing":
                data["flagged_adversarial"] = True
            path: Path | None = None
            if case != "missing_file":
                number = re.match(r"exp(\d+)", str(gate["upstream"]))
                assert number is not None
                path = case_dir / f"experiment_{number.group(1)}_validation_input.json"
                _atomic_write_json(path, data)
            check = evaluate_gates({"id": consumer, "gated_on": [gate]}, case_dir)
            intake = upstream_precondition(data, field, gate.get("value"))
            observed = check.gates_evaluated[0]
            rows.append(
                {
                    "edge_index": edge_index,
                    "consumer": consumer,
                    "upstream": gate["upstream"],
                    "case": case,
                    "validation_input": True,
                    "research_result": False,
                    "fixture_path": str(path) if path else None,
                    "field": field,
                    "operator": gate["op"],
                    "expected": gate.get("value"),
                    "actual": observed.actual,
                    "conductor_passed": check.passed,
                    "conductor_reason": observed.reason,
                    "experiment_precondition_passed": intake["passed"],
                    "experiment_precondition_reason": intake["reason"],
                    "field_gate_authenticates_artifact": False,
                }
            )
            progress(
                4, "check end", f"gate edge {edge_index + 1} case={case} passed={check.passed}"
            )
    return rows


def activation_rows(roadmap: JsonDict) -> list[JsonDict]:
    """Run the real schema and all isolated conductor cases in private files."""

    progress(4, "check start", "real Roadmap parser")
    try:
        Roadmap.model_validate(roadmap)
        error = None
    except Exception as exc:  # noqa: BLE001 - the exact parser error belongs in evidence.
        error = f"{type(exc).__name__}: {exc}"
    schema = {
        "case": "roadmap_schema_parse",
        "validation_input": True,
        "research_result": False,
        "passed": error is None,
        "error": error,
    }
    progress(4, "check end", f"real Roadmap parser passed={error is None}")
    with tempfile.TemporaryDirectory(prefix="exp7219-gates-", dir="/tmp") as directory:
        return [schema, *run_gate_validation_fixtures(roadmap, Path(directory))]


def activation_complete(rows: object) -> bool:
    """Require the schema plus the exact five-case pattern on all seven edges."""

    if not isinstance(rows, list) or len(rows) != 1 + EXPECTED_GATE_EDGE_COUNT * 5:
        return False
    schema = rows[0]
    if not isinstance(schema, Mapping) or schema.get("passed") is not True:
        return False
    expected = [True, False, False, False, True]
    for edge_index in range(EXPECTED_GATE_EDGE_COUNT):
        edge_rows = [
            row
            for row in rows[1:]
            if isinstance(row, Mapping) and row.get("edge_index") == edge_index
        ]
        if [row.get("conductor_passed") for row in edge_rows] != expected:
            return False
        if edge_rows[-1].get("experiment_precondition_passed") is not False:
            return False
    return all(isinstance(row, Mapping) and row.get("validation_input") is True for row in rows)


def _page_title(body: str) -> str | None:
    """Extract one visible HTML title without treating cached prose as access."""

    match = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    if not match:
        return None
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", match.group(1))).split())


def collect_source_method_rows(fetcher: Fetcher = base._fetch_url) -> list[JsonDict]:
    """Check the five bounded V636 primary pages and retain access failures."""

    rows: list[JsonDict] = []
    for source in SOURCES:
        url = str(source["url"])
        progress(3, "source request start", url)
        try:
            receipt = dict(fetcher(url))
        except Exception as exc:  # noqa: BLE001 - source access must fail visibly.
            receipt = {"ok": False, "status_code": None, "body": "", "error": str(exc)}
        accessed = receipt.get("ok") is True
        body = str(receipt.get("body") or "")
        observed_title = _page_title(body)
        expected_title = str(source["title"])
        title_verified = (
            expected_title.casefold() in observed_title.casefold()
            if accessed and observed_title
            else None
        )
        _version, observed_date = base._observed_version_and_date(
            body, str(source["planning_date"])
        )
        delta = bool(accessed and observed_date and observed_date > PLANNING_CUTOFF)
        rows.append(
            {
                **source,
                "retrieval_date": RETRIEVAL_DATE,
                "observed_title": observed_title,
                "title_verified": title_verified,
                "version_or_date": observed_date,
                "access_outcome": (
                    f"http_{receipt.get('status_code')}"
                    if accessed
                    else "unavailable_cached_primary_evidence"
                ),
                "access_error": receipt.get("error"),
                "content_sha256": base._sha256_bytes(body.encode()) if body else None,
                "post_planning_delta": delta,
                "verified_new_finding": "dated_primary_page_change" if delta else None,
                "ingested_change": delta,
            }
        )
        progress(3, "source request end", f"{url} ok={accessed} title_verified={title_verified}")
    return rows


def duration_classifier_rows(root: Path) -> list[JsonDict]:
    """Exercise the unchanged duration classifier on positive and negative pairs."""

    aggregation: JsonDict = {
        "run_date": RUN_DATE,
        "status": "complete",
        "honest_verdict": POSITIVE_VERDICT,
        "verdict_class": "positive",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "aggregation",
        "duration_s": 1.0,
        "MODEL_SPECS": [],
        "model_invoked": False,
    }
    flags: list[Any] = []
    check_substrate_class(aggregation, flags)
    floor = duration_floor_for_artifact(aggregation) or {}
    negative_path = root / "results/experiment_7208_v635_span_fixture.json"
    negative = json.loads(negative_path.read_text(encoding="utf-8"))
    negative_floor = duration_floor_for_artifact(negative) or {}
    return [
        {
            "case": "v636_aggregation_pair",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": "aggregation",
            "class_floor_s": SUBSTRATE_CLASS_FLOORS["aggregation"],
            "name_floor_s": floor.get("min_duration_s"),
            "name_floor_reason": floor.get("reason"),
            "class_flags": [flag.kind for flag in flags],
            "passed": not flags
            and floor.get("min_duration_s") == SUBSTRATE_CLASS_FLOORS["aggregation"],
        },
        {
            "case": "exp7208_negative_counterexample",
            "source_path": "results/experiment_7208_v635_span_fixture.json",
            "source_sha256": base._sha256(negative_path),
            "inference_substrate": negative.get("inference_substrate"),
            "inference_substrate_class": negative.get("inference_substrate_class"),
            "duration_s": negative.get("duration_s"),
            "name_floor_s": negative_floor.get("min_duration_s"),
            "name_floor_reason": negative_floor.get("reason"),
            "flagged_adversarial": negative.get("flagged_adversarial"),
            "preserved_without_reclassification": True,
            "passed": negative.get("flagged_adversarial") is True,
        },
    ]


def archive_lag_row(root: Path) -> JsonDict:
    """Record the planning-time lag and the archive state observed at execution."""

    document = yaml.safe_load((root / COMPLETE_PATH).read_text(encoding="utf-8"))
    milestones = document.get("milestones", []) if isinstance(document, Mapping) else []
    latest = (
        milestones[-1].get("id") if milestones and isinstance(milestones[-1], Mapping) else None
    )
    return {
        "planning_refresh_archive_latest_milestone": "2026.09.634",
        "archive_latest_milestone": str(latest) if latest is not None else None,
        "active_evidence_milestone": "2026.09.635",
        "archive_lag_observed_at_execution": str(latest) != "2026.09.635",
        "research_complete_rewritten": False,
    }


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Reuse shipped path checks and replace stale V635 labels with V636 facts."""

    with _configured_base():
        rows, authority, roadmap, yaml_bytes = base._preconditions(
            root, output_path, raw_dir, checkpoint_path
        )
    for row in rows:
        if row.get("check") == "driving_requirement":
            try:
                available = "REQ-REPORT-7219" in (root / SPEC_PATH).read_text(encoding="utf-8")
            except OSError:
                available = False
            row.update(
                expected_value="REQ-REPORT-7219",
                observed_value="REQ-REPORT-7219" if available else "missing",
                available=available,
            )
        elif row.get("check") == "planning_source_table":
            try:
                text = (root / REFERENCE_PATH).read_text(encoding="utf-8")
                available = all(
                    token in text
                    for token in (
                        "V636-PLANNER-REFRESH-20260911-START",
                        "10.3233/FAIA250893",
                        "2607.20792",
                        "features/tool_calling",
                        "features/quantization/gguf",
                        "2512.12850",
                    )
                )
            except OSError:
                available = False
            row.update(
                expected_value="dated V636 table with all five selected URLs",
                observed_value="available" if available else "missing_or_incomplete",
                available=available,
                field="V636-PLANNER-REFRESH-20260911",
            )
    return rows, authority, roadmap, yaml_bytes


def _summary(
    failed_check: str | None,
    expected: Any,
    observed: Any,
    *,
    upstream: Any = None,
    field: Any = None,
    passed: bool = False,
) -> JsonDict:
    """Keep one complete diagnostic shape for success, block, and mismatch."""

    return {
        "failed_check": failed_check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain every value and preserve the operator's required wording."""

    principles = {
        field: f"This field preserves auditable V636 receipt evidence." for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def _base_artifact(run_date: str, checkpoint_path: Path, started_at: str) -> JsonDict:
    """Create the nonterminal state that can only live in checkpoints."""

    gate_units = EXPECTED_GATE_EDGE_COUNT * GATE_CASES_PER_EDGE
    artifact: JsonDict = {
        "schema": "carnot.exp7219.v636_source_contract.v1",
        "experiment_id": FIRST_TASK_ID,
        "status": "running",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.000001,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "contract_rows": {
                "planned": EXPECTED_TASK_COUNT,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": EXPECTED_TASK_COUNT,
            },
            "source_urls": {
                "planned": len(SOURCES),
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": len(SOURCES),
            },
            "gate_validation_inputs": {
                "planned": gate_units,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": gate_units,
            },
            "independent_units": EXPECTED_TASK_COUNT,
            "exclusions": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _summary(
            "preconditions_not_checked", "all required resources", "not_checked"
        ),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_v636_source_contract",
        "source_contract_complete_score": 0,
        "contract_rows": [],
        "source_method_rows": [],
        "activation_validation_rows": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "markdown_task_rows": [],
        "yaml_task_rows": [],
        "gate_producer_rows": [],
        "prior_failure_rows": [],
        "receipt_dependency_rows": [],
        "raw_source_rows": [],
        "upstream_intake_rows": [],
        "duration_classifier_rows": [],
        "archive_lag_row": {},
        "validation_command_rows": [],
        "validation_required": False,
        "contract_authorities": [],
        "yaml_authority_path": None,
        "markdown_milestone": None,
        "yaml_milestone": None,
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "markdown_id_order": [],
        "observed_id_order": [],
        "source_mapping_complete": False,
        "structural_checks_complete": False,
        "post_planning_source_change_count": 0,
        "research_files_updated": False,
        "checkpoint_path": str(checkpoint_path),
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _failed_precondition(artifact: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the first unchanged external absence that stops the run."""

    rows = artifact.get("preconditions_checked")
    if not isinstance(rows, list):
        return None
    return next(
        (
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("blocking_external") is True
            and row.get("available") is not True
        ),
        None,
    )


def _score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Return one only after every advisory receipt component is complete."""

    contract = artifact.get("contract_rows")
    sources = artifact.get("source_method_rows")
    producers = artifact.get("gate_producer_rows")
    priors = artifact.get("prior_failure_rows")
    dependencies = artifact.get("receipt_dependency_rows")
    intake = artifact.get("upstream_intake_rows")
    duration = artifact.get("duration_classifier_rows")
    if not all(
        isinstance(rows, list)
        for rows in (contract, sources, producers, priors, dependencies, intake, duration)
    ):
        return 0
    if (
        artifact.get("markdown_milestone") != MILESTONE
        or artifact.get("yaml_milestone") != MILESTONE
        or len(contract) != EXPECTED_TASK_COUNT
        or [row.get("unit_id") for row in contract] != list(EXPECTED_ID_ORDER)
        or not all(row.get("passed") is True for row in contract)
        or not all(row.get("passed") is True for row in producers + priors + dependencies)
        or len(sources) != len(SOURCES)
        or any(
            not row.get("url")
            or not row.get("title")
            or not row.get("retrieval_date")
            or not row.get("access_outcome")
            or not row.get("method_boundary")
            for row in sources
        )
        or not activation_complete(artifact.get("activation_validation_rows"))
        or len(intake) != 2
        or not all(row.get("authentication_checked") is True for row in intake)
        or any(row.get("consumed") is not False for row in intake)
        or len(duration) != 2
        or not all(row.get("passed") is True for row in duration)
    ):
        return 0
    command_rows = artifact.get("validation_command_rows")
    if artifact.get("validation_required") is True:
        if not isinstance(command_rows, list) or any(
            row.get("passed") is not True for row in command_rows
        ):
            return 0
        completed_names = [row.get("name") for row in command_rows]
        if any(name not in completed_names for name in PRETERMINAL_VALIDATION_NAMES):
            return 0
    return 1


def _failure_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Name the first failed source, contract, gate, duration, or command."""

    if artifact.get("markdown_milestone") != MILESTONE:
        return _summary(
            "markdown_milestone",
            MILESTONE,
            artifact.get("markdown_milestone"),
            upstream=str(DESIGN_PATH),
            field="milestone",
        )
    row_groups = (
        ("contract_rows", "contract_parity"),
        ("gate_producer_rows", "gate_producer_contract"),
        ("prior_failure_rows", "prior_failure_contract"),
        ("duration_classifier_rows", "duration_classifier"),
    )
    for field, check in row_groups:
        rows = artifact.get(field)
        if isinstance(rows, list):
            failed = next((row for row in rows if row.get("passed") is not True), None)
            if failed:
                return _summary(check, True, failed, upstream=failed.get("upstream"), field=field)
    if not activation_complete(artifact.get("activation_validation_rows")):
        return _summary(
            "activation_validation",
            "schema plus 35 exact gate/authentication outcomes",
            artifact.get("activation_validation_rows"),
            field="activation_validation_rows",
        )
    commands = artifact.get("validation_command_rows")
    if isinstance(commands, list):
        failed_command = next((row for row in commands if row.get("passed") is not True), None)
        if failed_command:
            return _summary(
                f"validation:{failed_command.get('name')}",
                0,
                failed_command.get("exit_code"),
                field="validation_command_rows",
            )
    return _summary(
        "stored_source_contract_evidence",
        "complete V636 mapping and structural checks",
        "incomplete evidence",
    )


def _apply_terminal_state(artifact: JsonDict) -> None:
    """Derive the terminal class from evidence instead of planned intent."""

    artifact["status"] = "complete"
    failed = _failed_precondition(artifact)
    score = _score_from_artifact(artifact)
    artifact["source_contract_complete_score"] = score
    if failed:
        artifact.update(
            inference_substrate="blocked_no_run",
            inference_substrate_class="blocked_no_run",
            verdict_class="blocked",
            honest_verdict=BLOCKED_VERDICT,
            gate_check_summary=_summary(
                str(failed.get("check")),
                failed.get("expected_value"),
                failed.get("observed_value"),
                upstream=failed.get("upstream"),
                field=failed.get("field"),
            ),
        )
    elif score == 1:
        artifact.update(
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class="aggregation",
            verdict_class="positive",
            honest_verdict=POSITIVE_VERDICT,
            gate_check_summary=_summary(
                None,
                "all_14_v636_contract_rows_and_structural_checks_pass",
                "all_14_v636_contract_rows_and_structural_checks_pass",
                passed=True,
            ),
        )
    else:
        artifact.update(
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class="aggregation",
            verdict_class="disqualified",
            honest_verdict=DISQUALIFIED_VERDICT,
            gate_check_summary=_failure_summary(artifact),
        )


def _set_time_and_checksum(artifact: JsonDict, started: float) -> None:
    """Store actual elapsed time and bind the complete current evidence."""

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.000001)
    artifact["reproducibility_checksum"] = base.reproducibility_checksum(artifact)


def _checkpoint(path: Path, artifact: JsonDict, started: float) -> None:
    """Persist mutable work only at the designated checkpoint path."""

    _set_time_and_checksum(artifact, started)
    _atomic_write_json(path, artifact)


def validate_artifact(artifact: object) -> list[str]:
    """Recompute the score, lifecycle, denominators, and evidence checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != value for field, value in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles_invalid")
    if artifact.get("status") != "complete":
        errors.append("status_invalid")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    for timestamp in ("started_at_utc", "completed_at_utc"):
        try:
            parsed = datetime.fromisoformat(str(artifact.get(timestamp)))
        except ValueError:
            errors.append(f"{timestamp}_invalid")
        else:
            if parsed.tzinfo is None:
                errors.append(f"{timestamp}_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_contract_invalid")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("execution_host") != (platform.node() or "unknown"):
        errors.append("execution_host_invalid")
    duration = artifact.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
        errors.append("duration_invalid")
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    if artifact.get("rows") != artifact.get("contract_rows"):
        errors.append("rows_invalid")
    budget = artifact.get("sample_size_budget")
    if not isinstance(budget, Mapping) or any(
        not isinstance(budget.get(group), Mapping)
        or any(
            field not in budget[group]
            for field in ("planned", "attempted", "completed", "censored", "independent_units")
        )
        for group in ("contract_rows", "source_urls", "gate_validation_inputs")
    ):
        errors.append("sample_size_budget_invalid")
    hashes = artifact.get("source_artifact_hashes")
    if (
        not isinstance(hashes, Mapping)
        or not hashes
        or any(value is None for value in hashes.values())
    ):
        errors.append("source_hashes_invalid")
    raw_rows = artifact.get("raw_source_rows")
    if artifact.get("verdict_class") != "blocked" and (
        not isinstance(raw_rows, list)
        or len(raw_rows) != 2
        or any(
            row.get("hash_matches") is not True or row.get("source_sha256") != row.get("raw_sha256")
            for row in raw_rows
        )
    ):
        errors.append("raw_source_rows_invalid")
    commands = artifact.get("validation_command_rows")
    if not isinstance(commands, list) or [row.get("name") for row in commands] != list(
        VALIDATION_COMMAND_NAMES[: len(commands)]
    ):
        errors.append("validation_command_rows_invalid")
    expected_score = _score_from_artifact(artifact)
    if artifact.get("source_contract_complete_score") != expected_score:
        errors.append("source_contract_complete_score_invalid")
    expected = dict(artifact)
    _apply_terminal_state(expected)
    for field in (
        "inference_substrate",
        "inference_substrate_class",
        "verdict_class",
        "honest_verdict",
        "gate_check_summary",
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"{field}_invalid")
    if artifact.get("reproducibility_checksum") != base.reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def validation_commands(
    root: Path, artifact_path: Path, authority: Path
) -> tuple[tuple[str, list[str]], ...]:
    """Return shipped planning checks and changed-scope quality checks."""

    python = str(Path(sys.executable))
    roadmap = str(root / authority)
    artifact = str(artifact_path)
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    artifact_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7219_v636_source_contract import validate_artifact;"
        f"errors=validate_artifact(json.loads(pathlib.Path({artifact!r}).read_text()));"
        "print(errors);sys.exit(bool(errors))"
    )
    coverage_file = "/tmp/.coverage-exp7219-v636-validation"
    pytest_base = "/tmp/exp7219-v636-focused-pytest"
    return (
        ("roadmap_schema", [python, "-u", "-c", schema_code]),
        ("prior_failure", [python, "-u", str(root / PRIOR_LINT_PATH), roadmap]),
        ("exclusion_manifest", [python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap]),
        (
            "invented_paths",
            [python, "-u", str(root / INVENTED_PATH_LINT_PATH), "prompt-paths", roadmap],
        ),
        ("arc_floor", [python, "-u", str(root / ARC_LINT_PATH), roadmap]),
        ("gate_declarations", [python, "-u", str(root / GATE_LINT_PATH), roadmap]),
        (
            "focused_tests",
            [
                str(root / ".venv/bin/coverage"),
                "run",
                f"--data-file={coverage_file}",
                "--include=*/experiment_7219_v636_source_contract.py",
                "-m",
                "pytest",
                "--no-cov",
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
                "--include=*/experiment_7219_v636_source_contract.py",
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        (
            "scoped_spec_coverage",
            [python, "-u", str(root / SPEC_COVERAGE_PATH), str(root / TEST_PATH)],
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
        ("mypy", [str(root / ".venv/bin/mypy"), str(root / MODULE_PATH)]),
        ("artifact", [python, "-u", "-c", artifact_code]),
        ("adversarial", [python, "-u", str(root / ADVERSARIAL_PATH), artifact]),
        ("row_consistency", [python, "-u", str(root / ROW_LINT_PATH), artifact]),
    )


def _run_validation_commands(
    root: Path, checkpoint_path: Path, artifact: JsonDict, started: float
) -> list[JsonDict]:
    """Stream bounded subprocesses and checkpoint every observed completion."""

    authority = Path(str(artifact["yaml_authority_path"]))
    commands = validation_commands(root, checkpoint_path, authority)
    rows: list[JsonDict] = []
    artifact["validation_required"] = True
    for index, (name, command) in enumerate(commands, 1):
        if name == "artifact":
            _apply_terminal_state(artifact)
            artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
        _checkpoint(checkpoint_path, artifact, started)
        progress(5, "subprocess start", f"{index}/{len(commands)} {name}")
        receipt = _run_streaming_command(
            command,
            cwd=root,
            timeout_s=600,
            heartbeat_s=60,
            operation=f"exp7219:{name}",
        )
        row = {
            "name": name,
            "command": shlex.join(command),
            **receipt,
            "passed": receipt["exit_code"] == 0,
        }
        rows.append(row)
        artifact["validation_command_rows"] = list(rows)
        _checkpoint(checkpoint_path, artifact, started)
        progress(5, "subprocess end", f"{index}/{len(commands)} {name} exit={row['exit_code']}")
    return rows


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    fetcher: Fetcher = base._fetch_url,
    run_commands: bool = True,
) -> JsonDict:
    """Build one positive, disqualified, or externally blocked V636 receipt."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "print before checking any prerequisite")
    artifact = _base_artifact(run_date, checkpoint_path, started_at)
    _checkpoint(checkpoint_path, artifact, started)

    progress(1, "start", "check required bytes, imports, outputs, identity, and gates")
    preconditions, authority, roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["yaml_authority_path"] = str(authority) if authority else None
    with _configured_base():
        artifact["source_artifact_hashes"] = base._source_hashes(root, authority)
    artifact["upstream_intake_rows"] = upstream_intake_rows(root)
    failed = _failed_precondition(artifact)
    if failed:
        progress(1, "end", f"blocked on {failed.get('check')}")
    else:
        assert authority is not None and roadmap is not None and yaml_bytes is not None
        artifact["contract_authorities"] = [str(DESIGN_PATH), str(authority)]
        with _configured_base():
            artifact["raw_source_rows"] = base._raw_source_rows(
                root, authority, yaml_bytes, raw_dir
            )
        artifact["archive_lag_row"] = archive_lag_row(root)
        progress(1, "end", "authority bytes frozen; V635 history authenticated separately")

        progress(2, "start", "independently parse fourteen Markdown and YAML rows")
        contract = evaluate_contract((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)
        for field in (
            "markdown_milestone",
            "yaml_milestone",
            "markdown_task_rows",
            "yaml_task_rows",
            "contract_rows",
            "gate_producer_rows",
            "prior_failure_rows",
            "receipt_dependency_rows",
            "expected_id_order",
            "markdown_id_order",
            "observed_id_order",
        ):
            artifact[field] = contract[field]
        artifact["rows"] = artifact["contract_rows"]
        artifact["observed_task_count"] = len(artifact["yaml_task_rows"])
        artifact["source_mapping_complete"] = bool(contract["passed"])
        contract_budget = artifact["sample_size_budget"]["contract_rows"]
        contract_budget["attempted"] = len(artifact["contract_rows"])
        contract_budget["completed"] = len(artifact["contract_rows"])
        _checkpoint(checkpoint_path, artifact, started)
        progress(2, "end", f"source mapping complete={artifact['source_mapping_complete']}")

        progress(3, "start", "check five bounded primary source URLs")
        artifact["source_method_rows"] = collect_source_method_rows(fetcher)
        source_budget = artifact["sample_size_budget"]["source_urls"]
        source_budget["attempted"] = len(artifact["source_method_rows"])
        source_budget["completed"] = len(artifact["source_method_rows"])
        source_budget["censored"] = sum(
            row["access_outcome"] == "unavailable_cached_primary_evidence"
            for row in artifact["source_method_rows"]
        )
        artifact["post_planning_source_change_count"] = sum(
            row["post_planning_delta"] for row in artifact["source_method_rows"]
        )
        _checkpoint(checkpoint_path, artifact, started)
        progress(3, "end", f"source changes={artifact['post_planning_source_change_count']}")

        progress(4, "start", "run real schema, gate, quarantine, and duration checks")
        artifact["activation_validation_rows"] = activation_rows(roadmap)
        artifact["duration_classifier_rows"] = duration_classifier_rows(root)
        gate_budget = artifact["sample_size_budget"]["gate_validation_inputs"]
        gate_budget["attempted"] = len(artifact["activation_validation_rows"]) - 1
        gate_budget["completed"] = len(artifact["activation_validation_rows"]) - 1
        artifact["structural_checks_complete"] = activation_complete(
            artifact["activation_validation_rows"]
        ) and all(row["passed"] for row in artifact["gate_producer_rows"])
        _checkpoint(checkpoint_path, artifact, started)
        progress(4, "end", f"structural checks complete={artifact['structural_checks_complete']}")

        if run_commands:
            progress(5, "start", "run scoped validation subprocesses")
            _run_validation_commands(root, checkpoint_path, artifact, started)
            progress(5, "end", "validation subprocess receipts stored")

    progress(6, "validation start", "derive and recompute terminal evidence")
    _apply_terminal_state(artifact)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    _set_time_and_checksum(artifact, started)
    errors = validate_artifact(artifact)
    progress(6, "validation end", f"errors={errors}")
    if errors:  # pragma: no cover - command-line execution must fail closed.
        raise ValueError(f"invalid Exp7219 artifact: {errors}")
    progress(7, "write start", "atomically write checkpoint and terminal deliverable")
    _set_time_and_checksum(artifact, started)
    _atomic_write_json(checkpoint_path, artifact)
    _atomic_write_json(output_path, artifact)
    progress(7, "write end", f"{artifact['verdict_class']} artifact complete")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the operator-specified V636 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover - wrapper is integration-only.
    """Run the receipt and accept any internally valid terminal class."""

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
        print(f"[exp7219] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7219] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['source_contract_complete_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - use the required thin wrapper.
    raise SystemExit(main())

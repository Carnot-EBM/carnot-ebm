"""Build the V637 source and execution-contract receipt.

The module reuses the shipped V636 parsers and process runner. It adds only
the V637 roster, isolated classifier evidence, and the new receipt lifecycle.
It invokes no model and makes no scientific-value claim.

Spec refs: REQ-REPORT-7233 and SCENARIO-REPORT-7233-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
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

from carnot import experiment_7219_v636_source_contract as prior
from carnot.experiment_7179_v633_contract_receipt import (
    _atomic_write_json,
    _run_streaming_command,
    _sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - caller controls import paths.
    sys.path.insert(0, str(SCRIPTS_DIR))

from adversarial_verify import verify_artifact  # noqa: E402
from conductor_gates import _is_quarantined, evaluate_gates  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]

MILESTONE = "2026.09.637"
RUN_DATE = "20260912"
RETRIEVAL_DATE = "2026-09-12"
FIRST_TASK_ID = "exp7233-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7234-arc-scored-dryrun",
    "exp7235-arc-path-audit",
    "exp7236-mention-fixture",
    "exp7237-mention-canary",
    "exp7238-mention-capture",
    "exp7239-semantic-audit",
    "exp7240-recurrence-fixture",
    "exp7241-recurrence-learning",
    "exp7242-recurrence-audit",
    "exp7243-native-memory",
    "exp7244-board-disposition",
    "exp7245-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
EXPECTED_GATE_EDGE_COUNT = 5
GATE_CASES_PER_EDGE = 4
RANDOM_SEED = 7_233_202_609_12

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-v637.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
MODULE_PATH = Path("python/carnot/experiment_7233_v637_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7233_v637_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7233_v637_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7233_v637_contract.json")
RAW_DIR = Path("results/raw/experiment_7233")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7233_v637_contract.json")
RAW_MARKDOWN_NAME = "research-roadmap-v637.md"
RAW_YAML_NAME = "selected-roadmap.yaml"
CLASSIFIER_SIDECAR_NAME = "classifier-receipts.json"

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
GATE_EVALUATOR_PATH = Path("scripts/conductor_gates.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
PROMPT_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")

HISTORICAL_SOURCES: tuple[tuple[Path, str, Any], ...] = (
    (
        Path("results/experiment_7219_v636_source_contract.json"),
        "source_contract_complete_score",
        1,
    ),
    (Path("results/experiment_7223_v636_span_canary.json"), "flagged_adversarial", True),
    (Path("results/experiment_7228_v636_belief_cold_audit.json"), "flagged_adversarial", True),
    (Path("results/experiment_7230_v636_native_belief.json"), "flagged_adversarial", True),
    (Path("results/experiment_7232_v636_capstone.json"), "flagged_adversarial", True),
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
    ROADMAP_SCHEMA_PATH,
    GATE_EVALUATOR_PATH,
    GATE_LINT_PATH,
    PRIOR_LINT_PATH,
    EXCLUSION_LINT_PATH,
    PROMPT_PATH_LINT_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    SPEC_COVERAGE_PATH,
    Path("python/carnot/experiment_7219_v636_source_contract.py"),
    *(path for path, _field, _expected in HISTORICAL_SOURCES),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
POSITIVE_VERDICT = "complete_positive_v637_source_contract_exact_agreement"
DISQUALIFIED_VERDICT = "complete_disqualified_v637_source_contract_incomplete_or_mismatched"
BLOCKED_VERDICT = "blocked_v637_source_contract_prerequisite_missing"

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the artifact and bind experiment_id and milestone to this task.",
    "status": "Terminal complete or blocked only; unfinished work uses a separate checkpoint path.",
    "run_date": "Use 20260912; retain actual UTC start and end timestamps.",
    "field_principles": "Keep ordinary values at top level; put their explanations in this map.",
    "preconditions_checked": "Observed paths, resources, model identity and upstream checks before expensive work.",
    "inference_substrate": "Use the recognized literal for the operation actually executed.",
    "inference_substrate_class": "Actual class determines the duration floor; never pad duration or relabel to pass.",
    "execution_venue": "Top-level orchestration is host; board rows name kv260, gatemate or polarfire.",
    "execution_host": "Actual hostname, distinct from execution_venue.",
    "duration_s": "Measured monotonic elapsed work; record phase spans separately.",
    "MODEL_SPECS": "Models actually invoked; [] for tasks with no LLM.",
    "model_invoked": "Current task execution only; historical sources and injected fixtures are separate.",
    "source_artifact_hashes": "Hash source code, public inputs, private evaluator inputs and raw output files.",
    "rows": "Every comparison retains one row per independent unit and arm, with errors and abstentions.",
    "sample_size_budget": "Predeclared independent units, attempted/completed/censored units, and stopping rule.",
    "random_seed": "Freeze seeds and schedules before observing evaluation labels.",
    "reproducibility_checksum": "Hash the exact settings, inputs and raw rows supporting the result.",
    "gate_check_summary": "Every blocked_* verdict names check, upstream, artifact_field, expected and observed value.",
    "verifier_is_oracle": "True when the verification authority also defines correctness; separate code is insufficient independence.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. External incompleteness is blocked.",
    "honest_verdict": "Completed findings start complete_ or complete:. External absence starts blocked_. Failed acceptance forbids positive.",
    "acceptance_gate_results": "Preserve each frozen criterion, actual value and pass/fail independently of task completion.",
    "source_contract_complete_score": "One only after all thirteen contract records and validation receipts are complete.",
    "contract_rows": "Independently parsed task order, title, deliverable and exact structured gates.",
    "gate_replay_rows": "Real evaluate_gates outcomes for positive and negative temporary cases.",
    "source_method_rows": "Primary URL, retrieved date, version and adoption limit.",
    "classifier_receipt_path": "Hashed sidecar preserves injected negative fixtures outside current invocation fields.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

SOURCES: tuple[JsonDict, ...] = (
    {
        "source_id": "arxiv:2506.14790",
        "url": "https://arxiv.org/abs/2506.14790",
        "title": "Continuous Evolution Pool",
        "planning_version": "v3",
        "planning_version_date": "2026-08-18",
        "target_task": "exp7240-recurrence-fixture",
        "adoption_limit": "A bounded specialist pool motivates archived hypotheses; moment retrieval is not sufficient for label-conditional drift.",
    },
    {
        "source_id": "arxiv:2507.02310",
        "url": "https://arxiv.org/abs/2507.02310",
        "title": "Holistic Continual Learning under Concept Drift with Adaptive Memory Realignment",
        "planning_version": "v1",
        "planning_version_date": "2025-07-03",
        "target_task": "exp7241-recurrence-learning",
        "adoption_limit": "Visual-class memory results motivate reset and stale-memory controls; they do not establish Carnot learning value.",
    },
    {
        "source_id": "arxiv:2609.05025",
        "url": "https://arxiv.org/abs/2609.05025",
        "title": "Leveraging Low-Level Symbolic Competences for Unsupervised Grounding in Hallucination Detection",
        "planning_version": "v1",
        "planning_version_date": "2026-09-04",
        "target_task": "exp7236-mention-fixture",
        "adoption_limit": "Public mention identifiers reduce offset burden; correct execution does not establish source semantics.",
    },
)

VALIDATION_COMMAND_NAMES = (
    "roadmap_schema",
    "exclusion_manifest",
    "prior_failure",
    "prompt_paths",
    "gate_declarations",
    "focused_tests",
    "focused_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "artifact",
    "adversarial",
    "row_consistency",
)


def progress(phase: int, state: str, detail: str) -> None:
    """Flush one observed phase event so long work never appears silent."""

    print(f"[exp7233] phase {phase} {state}: {detail}", flush=True)


def sha256(path: Path) -> str:
    """Expose the shared byte hash for sidecar and test verification."""

    return _sha256(path)


@contextmanager
def _configured_prior() -> Iterator[None]:
    """Give reused V636 helpers the V637 constants, then restore them."""

    replacements = {
        "MILESTONE": MILESTONE,
        "RUN_DATE": RUN_DATE,
        "PLANNING_CUTOFF": RETRIEVAL_DATE,
        "FIRST_TASK_ID": FIRST_TASK_ID,
        "EXPECTED_ID_ORDER": EXPECTED_ID_ORDER,
        "EXPECTED_TASK_COUNT": EXPECTED_TASK_COUNT,
        "EXPECTED_GATE_EDGE_COUNT": EXPECTED_GATE_EDGE_COUNT,
        "RANDOM_SEED": RANDOM_SEED,
        "DESIGN_PATH": DESIGN_PATH,
        "SPEC_PATH": SPEC_PATH,
        "REFERENCE_PATH": REFERENCE_PATH,
        "COMPLETE_PATH": COMPLETE_PATH,
        "EXCLUSION_PATH": EXCLUSION_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "DEFAULT_OUTPUT_PATH": DEFAULT_OUTPUT_PATH,
        "RAW_DIR": RAW_DIR,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "RAW_MARKDOWN_NAME": RAW_MARKDOWN_NAME,
        "RAW_YAML_NAME": RAW_YAML_NAME,
        "SOURCE_PATHS": SOURCE_PATHS,
        "REQUIRED_FIELD_PRINCIPLES": REQUIRED_FIELD_PRINCIPLES,
        "REQUIRED_ARTIFACT_FIELDS": REQUIRED_ARTIFACT_FIELDS,
        "POSITIVE_VERDICT": POSITIVE_VERDICT,
        "DISQUALIFIED_VERDICT": DISQUALIFIED_VERDICT,
        "BLOCKED_VERDICT": BLOCKED_VERDICT,
        "progress": progress,
    }
    previous = {name: getattr(prior, name) for name in replacements}
    prior_design = prior.base.DESIGN_PATH
    for name, value in replacements.items():
        setattr(prior, name, value)
    prior.base.DESIGN_PATH = DESIGN_PATH
    try:
        yield
    finally:
        prior.base.DESIGN_PATH = prior_design
        for name, value in previous.items():
            setattr(prior, name, value)


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare independently parsed V637 Markdown and YAML records."""

    with _configured_prior():
        return prior.evaluate_contract(markdown_text, yaml_document)


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Select active, then staged YAML, only by the V637 milestone value."""

    with _configured_prior():
        return prior.select_yaml_authority(root)


def unwrap_principle_value(value: Any) -> Any:
    """Unwrap only mappings that contain both principle and value."""

    return prior.unwrap_principle_value(value)


def authenticate_value(data: Mapping[str, Any], field: str, expected: Any) -> JsonDict:
    """Reject a quarantine before a passing field can authorize evidence."""

    actual = unwrap_principle_value(data.get(field))
    quarantined = _is_quarantined(dict(data))
    return {
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": actual,
        "field_passed": actual == expected,
        "quarantined": quarantined,
        "accepted": actual == expected and not quarantined,
    }


def _failed_gate_value(expected: Any) -> Any:
    """Choose a deterministic unequal value for the current equality gates."""

    return 0 if expected != 0 else 1


def run_gate_replays(roadmap: Mapping[str, Any], results_dir: Path) -> list[JsonDict]:
    """Exercise four real evaluator cases for every declared V637 gate."""

    edges = [
        (str(task.get("id")), dict(gate))
        for task in roadmap.get("tasks", [])
        if isinstance(task, Mapping)
        for gate in task.get("gated_on", [])
        if isinstance(gate, Mapping)
    ]
    cases = ("passing", "failed", "absent_field", "absent_file")
    rows: list[JsonDict] = []
    for edge_index, (consumer, gate) in enumerate(edges):
        for case in cases:
            progress(4, "gate start", f"edge={edge_index + 1}/{len(edges)} case={case}")
            case_dir = results_dir / f"edge_{edge_index}_{case}"
            case_dir.mkdir(parents=True, exist_ok=True)
            field = str(gate["artifact_field"])
            data: JsonDict = {"status": "complete"}
            if case == "passing":
                data[field] = gate.get("value")
            elif case == "failed":
                data[field] = _failed_gate_value(gate.get("value"))
            path: Path | None = None
            if case != "absent_file":
                match = re.match(r"exp(\d+)", str(gate["upstream"]))
                if match is None:
                    raise ValueError(f"invalid upstream task id: {gate['upstream']}")
                path = case_dir / f"experiment_{match.group(1)}_gate_fixture.json"
                _atomic_write_json(path, data)
            check = evaluate_gates({"id": consumer, "gated_on": [gate]}, case_dir)
            observed = check.gates_evaluated[0]
            rows.append(
                {
                    "edge_index": edge_index,
                    "consumer": consumer,
                    "upstream": gate["upstream"],
                    "case": case,
                    "artifact_field": field,
                    "operator": gate["op"],
                    "expected_value": gate.get("value"),
                    "observed_value": observed.actual,
                    "fixture_path": str(path) if path else None,
                    "passed": check.passed,
                    "reason": observed.reason,
                    "validation_input": True,
                    "research_result": False,
                }
            )
            progress(4, "gate end", f"edge={edge_index + 1} case={case} passed={check.passed}")
    return rows


def gate_replays_complete(rows: object) -> bool:
    """Require the exact true, false, false, false pattern on five gates."""

    if not isinstance(rows, list) or len(rows) != EXPECTED_GATE_EDGE_COUNT * 4:
        return False
    for edge_index in range(EXPECTED_GATE_EDGE_COUNT):
        edge = [
            row for row in rows if isinstance(row, Mapping) and row.get("edge_index") == edge_index
        ]
        if [row.get("case") for row in edge] != [
            "passing",
            "failed",
            "absent_field",
            "absent_file",
        ]:
            return False
        if [row.get("passed") for row in edge] != [True, False, False, False]:
            return False
    return True


def _fixture_receipt(
    substrate: str, substrate_class: str, duration_s: float, model_invoked: bool
) -> JsonDict:
    """Build a small synthetic receipt that cannot be mistaken for this run."""

    return {
        "schema": "carnot.exp7233.classifier_fixture.v1",
        "status": "complete",
        "run_date": RUN_DATE,
        "honest_verdict": "complete_positive_classifier_fixture",
        "verdict_class": "positive",
        "inference_substrate": substrate,
        "inference_substrate_class": substrate_class,
        "execution_venue": "host",
        "duration_s": duration_s,
        "MODEL_SPECS": (
            [{"hf_id": "unsloth/Qwen3.8-27B-GGUF", "quantization": "Q4_K_M"}]
            if model_invoked
            else []
        ),
        "model_invoked": model_invoked,
        "preconditions_checked": [{"resource": "synthetic_fixture", "available": True}],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "sha256:synthetic-classifier-fixture",
        "verifier_is_oracle": False,
    }


def _critical_flags(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return only critical classifier findings from one stored report."""

    flags = report.get("flags")
    if not isinstance(flags, list):
        return []
    return [
        flag
        for flag in flags
        if isinstance(flag, Mapping) and str(flag.get("severity", "")).lower() == "critical"
    ]


def build_classifier_sidecar(root: Path, raw_dir: Path) -> JsonDict:
    """Run the unchanged classifier and keep all synthetic evidence in a sidecar."""

    valid = (
        ("aggregation", _fixture_receipt(INFERENCE_SUBSTRATE, "aggregation", 0.1, False)),
        (
            "bounded_generation",
            _fixture_receipt(
                "live_llm_inference_local_gguf_sota",
                "model_bounded_generation",
                10.5,
                True,
            ),
        ),
        (
            "full_generation",
            _fixture_receipt("live_llm_inference", "model_full_generation", 60.5, True),
        ),
        (
            "load_only",
            _fixture_receipt(
                "live_llm_embedding_extraction", "model_load_no_generation", 2.5, True
            ),
        ),
    )
    aggregation_live = _fixture_receipt(INFERENCE_SUBSTRATE, "aggregation", 0.1, True)
    bounded_negative = _fixture_receipt(
        "live_llm_inference_local_gguf_sota", "model_bounded_generation", 10.5, False
    )
    nested_diagnostic = _fixture_receipt(INFERENCE_SUBSTRATE, "aggregation", 0.1, False)
    nested_diagnostic["preconditions_checked"].append(
        {
            "check": "historical invocation shape",
            "expected_value": {"model_invoked": True},
            "observed_value": {"model_invoked": True},
        }
    )
    fixtures = (
        *valid,
        ("aggregation_with_live_claim", aggregation_live),
        ("bounded_with_negative_claim", bounded_negative),
        ("nested_expected_value_scope", nested_diagnostic),
    )
    fixture_rows: list[JsonDict] = []
    progress(5, "classifier start", f"isolated fixtures={len(fixtures)}")
    with tempfile.TemporaryDirectory(prefix="exp7233-classifier-", dir="/tmp") as directory:
        fixture_dir = Path(directory)
        for index, (case, payload) in enumerate(fixtures, 1):
            progress(5, "classifier unit start", f"{index}/{len(fixtures)} {case}")
            path = fixture_dir / f"classifier_{case}.json"
            _atomic_write_json(path, payload)
            report = verify_artifact(path, declared=False)
            critical = _critical_flags(report)
            fixture_rows.append(
                {
                    "case": case,
                    "current_task_fixture": False,
                    "fixture": payload,
                    "verifier_report": {**report, "artifact": f"isolated:{case}"},
                    "critical_flag_count": len(critical),
                    "flag_kinds": [str(flag.get("kind")) for flag in critical],
                }
            )
            progress(5, "classifier unit end", f"{index}/{len(fixtures)} flags={len(critical)}")
    historical_rows: list[JsonDict] = []
    for relative, field, expected in HISTORICAL_SOURCES:
        path = root / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"historical source must be a mapping: {relative}")
        authentication = authenticate_value(data, field, expected)
        report = verify_artifact(path)
        historical_rows.append(
            {
                "source_path": str(relative),
                "source_sha256": _sha256(path),
                "checked_field": field,
                "expected_value": expected,
                "observed_value": unwrap_principle_value(data.get(field)),
                "quarantined": _is_quarantined(data),
                "accepted_for_evidence": authentication["accepted"],
                "verifier_report": report,
            }
        )
    sidecar = {
        "schema": "carnot.exp7233.classifier_receipts.v1",
        "run_date": RUN_DATE,
        "isolated_fixture_reports": fixture_rows,
        "historical_source_rows": historical_rows,
        "unresolved_classifier_problem": {
            "status": "unresolved",
            "case": "nested_expected_value_scope",
            "finding": "Typed model_invoked values inside generic expected_value and observed_value mappings are attributed to the current task.",
            "impact": "Aggregation and CPU audit artifacts can receive false invocation contradictions unless historical and synthetic receipts stay in a recognized sidecar scope.",
            "action": "Record the limitation here; do not change the shared classifier in this task.",
        },
    }
    sidecar_path = raw_dir / CLASSIFIER_SIDECAR_NAME
    _atomic_write_json(sidecar_path, sidecar)
    progress(5, "classifier end", f"sidecar={sidecar_path}")
    return {"path": str(sidecar_path.relative_to(root)), "sha256": _sha256(sidecar_path)}


def _observed_version(body: str) -> str | None:
    """Return the greatest visible arXiv version marker without guessing."""

    versions = [int(value) for value in re.findall(r"\[v(\d+)\]", body, re.I)]
    return f"v{max(versions)}" if versions else None


def collect_source_method_rows(fetcher: Fetcher = prior.base._fetch_url) -> list[JsonDict]:
    """Recheck three V637 primary records and retain no-delta or access failure."""

    rows: list[JsonDict] = []
    for source in SOURCES:
        url = str(source["url"])
        progress(6, "source request start", url)
        try:
            receipt = dict(fetcher(url))
        except Exception as exc:  # noqa: BLE001 - source access failures are evidence.
            receipt = {"ok": False, "status_code": None, "body": "", "error": str(exc)}
        accessed = receipt.get("ok") is True
        body = str(receipt.get("body") or "")
        title = prior._page_title(body)
        observed_version = _observed_version(body) if accessed else None
        changed = bool(accessed and observed_version != source["planning_version"])
        if not accessed:
            outcome = "access_failed"
        elif changed:
            outcome = "version_changed_review_required"
        else:
            outcome = "no_delta"
        rows.append(
            {
                **source,
                "retrieval_date": RETRIEVAL_DATE,
                "observed_title": title,
                "title_verified": (
                    str(source["title"]).casefold() in title.casefold() if title else None
                ),
                "observed_version": observed_version,
                "access_outcome": (
                    f"http_{receipt.get('status_code')}"
                    if accessed
                    else "unavailable_cached_primary_evidence"
                ),
                "access_error": receipt.get("error"),
                "execution_time_version_changed": changed,
                "recheck_outcome": outcome,
                "carnot_result_claimed": False,
            }
        )
        progress(6, "source request end", f"{url} outcome={outcome}")
    return rows


def archive_lag_row(root: Path) -> JsonDict:
    """Record both the planning-time lag and the current archive boundary."""

    document = yaml.safe_load((root / COMPLETE_PATH).read_text(encoding="utf-8"))
    milestones = document.get("milestones", []) if isinstance(document, Mapping) else []
    latest = (
        milestones[-1].get("id") if milestones and isinstance(milestones[-1], Mapping) else None
    )
    terminal = all(
        (root / relative).is_file() for relative, _field, _expected in HISTORICAL_SOURCES
    )
    return {
        "planning_refresh_archive_latest_milestone": "2026.09.635",
        "archive_latest_milestone_at_execution": str(latest) if latest is not None else None,
        "v636_terminal_artifacts_observed": terminal,
        "planning_archive_lag_observed": True,
        "execution_archive_lag_observed": str(latest) != "2026.09.636",
        "research_complete_rewritten": False,
    }


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Reuse path checks, then add V637 requirements and historical hashes."""

    with _configured_prior():
        rows, authority, roadmap, yaml_bytes = prior._preconditions(
            root, output_path, raw_dir, checkpoint_path
        )
    for row in rows:
        if row.get("check") == "driving_requirement":
            try:
                available = "REQ-REPORT-7233" in (root / SPEC_PATH).read_text(encoding="utf-8")
            except OSError:
                available = False
            row.update(
                expected_value="REQ-REPORT-7233",
                observed_value="REQ-REPORT-7233" if available else "missing",
                available=available,
            )
        elif row.get("check") == "planning_source_table":
            try:
                text = (root / REFERENCE_PATH).read_text(encoding="utf-8")
                available = all(
                    token in text
                    for token in (
                        "V637-PLANNER-REFRESH-20260912-START",
                        "2506.14790",
                        "2507.02310",
                        "2609.05025",
                    )
                )
            except OSError:
                available = False
            row.update(
                expected_value="dated V637 refresh with all three selected primary records",
                observed_value="available" if available else "missing_or_incomplete",
                available=available,
                field="V637-PLANNER-REFRESH-20260912",
            )
    rows.append(
        {
            "check": "required_imports",
            "path": "python_runtime",
            "upstream": "yaml|adversarial_verify|conductor_gates|roadmap_schema",
            "field": "imports",
            "expected_value": "all imports loaded",
            "observed_value": "all imports loaded",
            "available": True,
            "blocking_external": True,
        }
    )
    for relative, field, expected in HISTORICAL_SOURCES:
        path = root / relative
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("top level is not a mapping")
            authentication = authenticate_value(data, field, expected)
            is_regression = expected is True and field == "flagged_adversarial"
            available = authentication["field_passed"] and (
                authentication["quarantined"] if is_regression else authentication["accepted"]
            )
            observed: Any = {
                "sha256": _sha256(path),
                "artifact_field": field,
                "value_matches": authentication["field_passed"],
                "quarantined": authentication["quarantined"],
                "accepted_for_evidence": authentication["accepted"],
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            available = False
            observed = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "check": f"historical_authentication:{relative}",
                "path": str(relative),
                "upstream": str(relative),
                "field": field,
                "expected_value": expected,
                "observed_value": observed,
                "available": available,
                "blocking_external": True,
            }
        )
    return rows, authority, roadmap, yaml_bytes


def _source_hashes(root: Path, authority: Path | None) -> dict[str, str | None]:
    """Hash every declared input through the reused V636 hash helper."""

    with _configured_prior(), prior._configured_base():
        return prior.base._source_hashes(root, authority)


def _raw_source_rows(
    root: Path, authority: Path, yaml_bytes: bytes, raw_dir: Path
) -> list[JsonDict]:
    """Freeze exact V637 Markdown and YAML bytes without normalization."""

    sources = (
        ("markdown", DESIGN_PATH, (root / DESIGN_PATH).read_bytes(), raw_dir / RAW_MARKDOWN_NAME),
        ("yaml", authority, yaml_bytes, raw_dir / RAW_YAML_NAME),
    )
    rows: list[JsonDict] = []
    for source_type, relative, content, destination in sources:
        prior.base._atomic_write_bytes(destination, content)
        source_hash = prior.base._sha256_bytes(content)
        raw_hash = sha256(destination)
        rows.append(
            {
                "source_type": source_type,
                "source_path": str(relative),
                "raw_path": str(destination.relative_to(root)),
                "source_sha256": source_hash,
                "raw_sha256": raw_hash,
                "hash_matches": source_hash == raw_hash,
            }
        )
    return rows


def _summary(
    failed_check: str | None,
    expected: Any,
    observed: Any,
    *,
    upstream: Any = None,
    artifact_field: Any = None,
    passed: bool = False,
) -> JsonDict:
    """Use the required diagnostic shape for every terminal state."""

    return {
        "failed_check": failed_check,
        "upstream": upstream,
        "artifact_field": artifact_field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Preserve every task principle and explain additional receipt fields."""

    values = {field: f"This field keeps one V637 contract fact auditable." for field in fields}
    values.update(REQUIRED_FIELD_PRINCIPLES)
    return values


def base_artifact(run_date: str, checkpoint_path: Path, started_at: str) -> JsonDict:
    """Create a schema-complete running state only for the checkpoint path."""

    gate_units = EXPECTED_GATE_EDGE_COUNT * GATE_CASES_PER_EDGE
    artifact: JsonDict = {
        "schema": "carnot.exp7233.v637_contract.v1",
        "experiment_id": FIRST_TASK_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "phase_spans": [],
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.000001,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "model_invocation_count": 0,
        "model_generation_count": 0,
        "model_load_count": 0,
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
            "gate_replays": {
                "planned": gate_units,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": gate_units,
            },
            "source_records": {
                "planned": len(SOURCES),
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": len(SOURCES),
            },
            "stopping_rule": "Stop after thirteen contract rows, four replays per gate, seven classifier fixtures, five historical sources, and three source rechecks.",
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _summary(
            "preconditions_not_checked", "all required inputs", "not_checked"
        ),
        "verifier_is_oracle": False,
        "verdict_class": "partial",
        "honest_verdict": "partial_running_v637_source_contract",
        "acceptance_gate_results": [],
        "source_contract_complete_score": 0,
        "contract_rows": [],
        "gate_replay_rows": [],
        "source_method_rows": [],
        "classifier_receipt_path": None,
        "classifier_receipt_sha256": None,
        "classifier_receipt_summary": {},
        "raw_source_rows": [],
        "archive_lag_row": {},
        "validation_command_rows": [],
        "validation_required": False,
        "contract_authorities": [],
        "yaml_authority_path": None,
        "markdown_milestone": None,
        "yaml_milestone": None,
        "markdown_task_rows": [],
        "yaml_task_rows": [],
        "gate_producer_rows": [],
        "prior_failure_rows": [],
        "receipt_dependency_rows": [],
        "expected_task_count": EXPECTED_TASK_COUNT,
        "observed_task_count": 0,
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "markdown_id_order": [],
        "observed_id_order": [],
        "checkpoint_path": str(checkpoint_path),
        "activation_files_modified": False,
        "scientific_value_claimed": False,
        "publication_or_submission_performed": False,
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _failed_precondition(artifact: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the first unavailable blocking prerequisite."""

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


def _classifier_complete(artifact: Mapping[str, Any]) -> bool:
    """Check only the sidecar summary, never its invocation-shaped contents."""

    summary = artifact.get("classifier_receipt_summary")
    return isinstance(summary, Mapping) and all(
        (
            summary.get("fixture_count") == 7,
            summary.get("valid_fixture_critical_flags") == 0,
            summary.get("contradictory_fixtures_detected") == 3,
            summary.get("historical_source_count") == 5,
            summary.get("quarantined_sources_rejected") == 4,
            summary.get("unresolved_problem_recorded") is True,
        )
    )


def _validation_complete(artifact: Mapping[str, Any]) -> bool:
    """Require every scoped command receipt and a zero exit status."""

    rows = artifact.get("validation_command_rows")
    if artifact.get("validation_required") is not True or not isinstance(rows, list):
        return False
    return [row.get("name") for row in rows] == list(VALIDATION_COMMAND_NAMES) and all(
        row.get("passed") is True for row in rows
    )


def _acceptance_gate_results(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute each frozen structural criterion independently."""

    contract = artifact.get("contract_rows")
    sources = artifact.get("source_method_rows")
    dependencies = artifact.get("receipt_dependency_rows")
    raw = artifact.get("raw_source_rows")
    hashes = artifact.get("source_artifact_hashes")
    rows = (
        (
            "contract_exact",
            EXPECTED_TASK_COUNT,
            len(contract) if isinstance(contract, list) else None,
            isinstance(contract, list)
            and [row.get("unit_id") for row in contract] == list(EXPECTED_ID_ORDER)
            and all(row.get("passed") is True for row in contract),
        ),
        (
            "gate_replays",
            EXPECTED_GATE_EDGE_COUNT * 4,
            len(artifact.get("gate_replay_rows", []))
            if isinstance(artifact.get("gate_replay_rows"), list)
            else None,
            gate_replays_complete(artifact.get("gate_replay_rows")),
        ),
        (
            "classifier_sidecar",
            True,
            _classifier_complete(artifact),
            _classifier_complete(artifact),
        ),
        (
            "source_rechecks",
            len(SOURCES),
            len(sources) if isinstance(sources, list) else None,
            isinstance(sources, list)
            and len(sources) == len(SOURCES)
            and all(
                row.get("retrieval_date") == RETRIEVAL_DATE
                and row.get("adoption_limit")
                and row.get("recheck_outcome")
                for row in sources
            ),
        ),
        (
            "receipt_is_advisory",
            [],
            dependencies[0].get("observed_consumers")
            if isinstance(dependencies, list) and dependencies
            else None,
            isinstance(dependencies, list)
            and len(dependencies) == 1
            and dependencies[0].get("passed") is True,
        ),
        (
            "raw_contract_hashes",
            2,
            len(raw) if isinstance(raw, list) else None,
            isinstance(raw, list)
            and len(raw) == 2
            and all(row.get("hash_matches") is True for row in raw),
        ),
        (
            "source_hashes",
            "all present",
            "all present"
            if isinstance(hashes, Mapping) and hashes and all(hashes.values())
            else "missing",
            isinstance(hashes, Mapping) and bool(hashes) and all(hashes.values()),
        ),
        (
            "validation_receipts",
            list(VALIDATION_COMMAND_NAMES),
            [row.get("name") for row in artifact.get("validation_command_rows", [])]
            if isinstance(artifact.get("validation_command_rows"), list)
            else None,
            _validation_complete(artifact),
        ),
    )
    return [
        {
            "criterion": criterion,
            "expected_value": expected,
            "observed_value": observed,
            "passed": bool(passed),
        }
        for criterion, expected, observed, passed in rows
    ]


def _score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Return one only when every independently stored acceptance gate passes."""

    results = _acceptance_gate_results(artifact)
    return int(len(results) == 8 and all(row["passed"] for row in results))


def _failure_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Name the first failed precondition, contract row, or acceptance gate."""

    failed = _failed_precondition(artifact)
    if failed is not None:
        return _summary(
            str(failed.get("check")),
            failed.get("expected_value"),
            failed.get("observed_value"),
            upstream=failed.get("upstream"),
            artifact_field=failed.get("field"),
        )
    contract = artifact.get("contract_rows")
    if isinstance(contract, list):
        row = next((item for item in contract if item.get("passed") is not True), None)
        if row is not None:
            return _summary(
                "contract_parity",
                row.get("markdown"),
                row.get("yaml"),
                upstream=row.get("unit_id"),
                artifact_field="id|title|deliverable|gates",
            )
    acceptance = _acceptance_gate_results(artifact)
    row = next((item for item in acceptance if item["passed"] is not True), None)
    return _summary(
        f"acceptance:{row['criterion']}" if row else "stored_contract_evidence",
        row.get("expected_value") if row else "complete structural evidence",
        row.get("observed_value") if row else "incomplete",
        artifact_field=row.get("criterion") if row else "acceptance_gate_results",
    )


def _apply_terminal_state(artifact: JsonDict) -> None:
    """Derive terminal class from observed work, never from planned intent."""

    artifact["status"] = "complete"
    artifact["acceptance_gate_results"] = _acceptance_gate_results(artifact)
    score = _score_from_artifact(artifact)
    artifact["source_contract_complete_score"] = score
    failed = _failed_precondition(artifact)
    if failed is not None:
        artifact.update(
            inference_substrate="blocked_no_run",
            inference_substrate_class="blocked_no_run",
            verdict_class="blocked",
            honest_verdict=BLOCKED_VERDICT,
            gate_check_summary=_failure_summary(artifact),
        )
    elif score == 1:
        artifact.update(
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class="aggregation",
            verdict_class="positive",
            honest_verdict=POSITIVE_VERDICT,
            gate_check_summary=_summary(
                None,
                "all thirteen contract rows and validation receipts complete",
                "all thirteen contract rows and validation receipts complete",
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
    """Store measured monotonic work time and bind the current evidence."""

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.000001)
    artifact["reproducibility_checksum"] = prior.base.reproducibility_checksum(artifact)


def _checkpoint(path: Path, artifact: JsonDict, started: float) -> None:
    """Persist mutable state only at the checkpoint path."""

    _set_time_and_checksum(artifact, started)
    _atomic_write_json(path, artifact)


def _record_phase(artifact: JsonDict, phase: int, started: float, detail: str) -> None:
    """Retain a monotonic span for one completed phase."""

    artifact["phase_spans"].append(
        {"phase": phase, "detail": detail, "duration_s": round(time.monotonic() - started, 6)}
    )


def validate_artifact(artifact: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Recompute required fields, lifecycle, score, sidecar hash, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != principle for field, principle in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles_invalid")
    if artifact.get("schema") != "carnot.exp7233.v637_contract.v1":
        errors.append("schema_invalid")
    if artifact.get("status") != "complete":
        errors.append("status_invalid")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    for field in ("started_at_utc", "completed_at_utc"):
        try:
            parsed = datetime.fromisoformat(str(artifact.get(field)))
        except ValueError:
            errors.append(f"{field}_invalid")
        else:
            if parsed.tzinfo is None:
                errors.append(f"{field}_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or any(
            artifact.get(field) != 0
            for field in ("model_invocation_count", "model_generation_count", "model_load_count")
        )
    ):
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
    if (
        not isinstance(budget, Mapping)
        or not budget.get("stopping_rule")
        or any(
            not isinstance(budget.get(group), Mapping)
            or any(
                field not in budget[group]
                for field in ("planned", "attempted", "completed", "censored", "independent_units")
            )
            for group in ("contract_rows", "gate_replays", "source_records")
        )
    ):
        errors.append("sample_size_budget_invalid")
    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        errors.append("source_hashes_invalid")
    raw = artifact.get("raw_source_rows")
    if artifact.get("verdict_class") != "blocked" and (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(row.get("hash_matches") is not True for row in raw)
    ):
        errors.append("raw_source_rows_invalid")
    commands = artifact.get("validation_command_rows")
    if not isinstance(commands, list) or [row.get("name") for row in commands] != list(
        VALIDATION_COMMAND_NAMES[: len(commands)]
    ):
        errors.append("validation_command_rows_invalid")
    classifier_path = artifact.get("classifier_receipt_path")
    if artifact.get("verdict_class") != "blocked":
        path = root / str(classifier_path)
        try:
            actual_classifier_hash = _sha256(path)
        except OSError:
            actual_classifier_hash = None
        if (
            not classifier_path
            or artifact.get("classifier_receipt_sha256") != actual_classifier_hash
            or not isinstance(hashes, Mapping)
            or hashes.get(str(classifier_path)) != actual_classifier_hash
        ):
            errors.append("classifier_receipt_invalid")
    expected = dict(artifact)
    _apply_terminal_state(expected)
    for field in (
        "acceptance_gate_results",
        "source_contract_complete_score",
        "inference_substrate",
        "inference_substrate_class",
        "verdict_class",
        "honest_verdict",
        "gate_check_summary",
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"{field}_invalid")
    if artifact.get("reproducibility_checksum") != prior.base.reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def validation_commands(
    root: Path, artifact_path: Path, authority: Path
) -> tuple[tuple[str, list[str]], ...]:
    """Return unchanged roadmap checks and explicit changed-scope checks."""

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
        "from carnot.experiment_7233_v637_contract import validate_artifact;"
        f"root=pathlib.Path({str(root)!r});"
        f"errors=validate_artifact(json.loads(pathlib.Path({artifact!r}).read_text()),root=root);"
        "print(errors);sys.exit(bool(errors))"
    )
    coverage_file = "/tmp/.coverage-exp7233-v637"
    pytest_base = "/tmp/exp7233-v637-focused-pytest"
    return (
        ("roadmap_schema", [python, "-u", "-c", schema_code]),
        ("exclusion_manifest", [python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap]),
        ("prior_failure", [python, "-u", str(root / PRIOR_LINT_PATH), roadmap]),
        (
            "prompt_paths",
            [python, "-u", str(root / PROMPT_PATH_LINT_PATH), "prompt-paths", roadmap],
        ),
        ("gate_declarations", [python, "-u", str(root / GATE_LINT_PATH), roadmap]),
        (
            "focused_tests",
            [
                str(root / ".venv/bin/coverage"),
                "run",
                f"--data-file={coverage_file}",
                "--include=*/experiment_7233_v637_contract.py",
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
                "--include=*/experiment_7233_v637_contract.py",
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
        ("mypy", [str(root / ".venv/bin/mypy"), str(root / MODULE_PATH)]),
        (
            "scoped_spec_coverage",
            [python, "-u", str(root / SPEC_COVERAGE_PATH), str(root / TEST_PATH)],
        ),
        ("artifact", [python, "-u", "-c", artifact_code]),
        ("adversarial", [python, "-u", str(root / ADVERSARIAL_PATH), artifact]),
        ("row_consistency", [python, "-u", str(root / ROW_LINT_PATH), artifact]),
    )


def _run_validation_commands(
    root: Path, checkpoint_path: Path, artifact: JsonDict, started: float
) -> list[JsonDict]:
    """Stream every child, emit heartbeats, and checkpoint exact receipts."""

    authority = Path(str(artifact["yaml_authority_path"]))
    commands = validation_commands(root, checkpoint_path, authority)
    rows: list[JsonDict] = []
    artifact["validation_required"] = True
    for index, (name, command) in enumerate(commands, 1):
        if name in {"artifact", "adversarial", "row_consistency"}:
            _apply_terminal_state(artifact)
            artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
        _checkpoint(checkpoint_path, artifact, started)
        progress(7, "subprocess start", f"{index}/{len(commands)} {name}")
        receipt = _run_streaming_command(
            command,
            cwd=root,
            timeout_s=600,
            heartbeat_s=60,
            operation=f"exp7233:{name}",
        )
        row = {
            "name": name,
            "command": shlex.join(command),
            **receipt,
            "passed": receipt["exit_code"] == 0,
            "failure_scope": None if receipt["exit_code"] == 0 else "changed_scope_or_baseline",
        }
        rows.append(row)
        artifact["validation_command_rows"] = list(rows)
        _checkpoint(checkpoint_path, artifact, started)
        progress(7, "subprocess end", f"{index}/{len(commands)} {name} exit={row['exit_code']}")
    return rows


def _skip_phases(artifact: JsonDict, checkpoint_path: Path, started: float) -> None:
    """Emit truthful boundaries for phases skipped after a blocking precondition."""

    for phase in range(1, 8):
        phase_started = time.monotonic()
        progress(phase, "start", "skipped after blocking precondition")
        _record_phase(artifact, phase, phase_started, "skipped after blocking precondition")
        _checkpoint(checkpoint_path, artifact, started)
        progress(phase, "end", "skipped after blocking precondition")


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    fetcher: Fetcher = prior.base._fetch_url,
    run_commands: bool = True,
) -> JsonDict:
    """Build one positive, disqualified, or externally blocked V637 receipt."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "preconditions before all substantive work")
    phase_started = time.monotonic()
    artifact = base_artifact(run_date, checkpoint_path, started_at)
    _checkpoint(checkpoint_path, artifact, started)
    preconditions, authority, roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["yaml_authority_path"] = str(authority) if authority else None
    artifact["source_artifact_hashes"] = _source_hashes(root, authority)
    artifact["archive_lag_row"] = archive_lag_row(root)
    _record_phase(artifact, 0, phase_started, "preconditions")
    _checkpoint(checkpoint_path, artifact, started)
    failed = _failed_precondition(artifact)
    progress(0, "end", f"blocked={failed is not None}")

    if failed is not None:
        _skip_phases(artifact, checkpoint_path, started)
    else:
        assert authority is not None and roadmap is not None and yaml_bytes is not None

        phase_started = time.monotonic()
        progress(1, "start", "verify progress and checkpoint policy")
        _record_phase(artifact, 1, phase_started, "progress and checkpoint policy")
        _checkpoint(checkpoint_path, artifact, started)
        progress(1, "end", "progress and checkpoint policy verified")

        phase_started = time.monotonic()
        progress(2, "start", "bind the no-model aggregation declaration")
        artifact["MODEL_SPECS"] = []
        artifact["model_invoked"] = False
        artifact["model_invocation_count"] = 0
        artifact["model_generation_count"] = 0
        artifact["model_load_count"] = 0
        _record_phase(artifact, 2, phase_started, "no-model aggregation declaration")
        _checkpoint(checkpoint_path, artifact, started)
        progress(2, "end", "all current invocation counters are zero")

        phase_started = time.monotonic()
        progress(3, "start", "freeze and independently parse V637 authorities")
        artifact["contract_authorities"] = [str(DESIGN_PATH), str(authority)]
        artifact["raw_source_rows"] = _raw_source_rows(root, authority, yaml_bytes, raw_dir)
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
        budget = artifact["sample_size_budget"]["contract_rows"]
        budget["attempted"] = len(artifact["contract_rows"])
        budget["completed"] = len(artifact["contract_rows"])
        for row in artifact["raw_source_rows"]:
            artifact["source_artifact_hashes"][row["raw_path"]] = row["raw_sha256"]
        _record_phase(artifact, 3, phase_started, "contract freeze and independent parse")
        _checkpoint(checkpoint_path, artifact, started)
        progress(3, "end", f"contract_match={contract['passed']}")

        phase_started = time.monotonic()
        progress(4, "start", "run schema parser and all real gate replays")
        Roadmap.model_validate(roadmap)
        with tempfile.TemporaryDirectory(prefix="exp7233-gates-", dir="/tmp") as directory:
            artifact["gate_replay_rows"] = run_gate_replays(roadmap, Path(directory))
        budget = artifact["sample_size_budget"]["gate_replays"]
        budget["attempted"] = len(artifact["gate_replay_rows"])
        budget["completed"] = len(artifact["gate_replay_rows"])
        _record_phase(artifact, 4, phase_started, "schema and gate replay")
        _checkpoint(checkpoint_path, artifact, started)
        progress(
            4, "end", f"replays_complete={gate_replays_complete(artifact['gate_replay_rows'])}"
        )

        phase_started = time.monotonic()
        progress(5, "start", "run isolated classifier fixtures and historical regressions")
        classifier = build_classifier_sidecar(root, raw_dir)
        artifact["classifier_receipt_path"] = classifier["path"]
        artifact["classifier_receipt_sha256"] = classifier["sha256"]
        artifact["source_artifact_hashes"][classifier["path"]] = classifier["sha256"]
        sidecar = json.loads((root / classifier["path"]).read_text(encoding="utf-8"))
        reports = sidecar["isolated_fixture_reports"]
        historical = sidecar["historical_source_rows"]
        artifact["classifier_receipt_summary"] = {
            "fixture_count": len(reports),
            "valid_fixture_critical_flags": sum(row["critical_flag_count"] for row in reports[:4]),
            "contradictory_fixtures_detected": sum(
                row["critical_flag_count"] > 0 for row in reports[4:]
            ),
            "historical_source_count": len(historical),
            "quarantined_sources_rejected": sum(
                row["quarantined"] and not row["accepted_for_evidence"] for row in historical
            ),
            "unresolved_problem_recorded": sidecar["unresolved_classifier_problem"]["status"]
            == "unresolved",
        }
        _record_phase(artifact, 5, phase_started, "classifier sidecar")
        _checkpoint(checkpoint_path, artifact, started)
        progress(5, "end", f"sidecar={classifier['path']}")

        phase_started = time.monotonic()
        progress(6, "start", "recheck three primary source versions")
        artifact["source_method_rows"] = collect_source_method_rows(fetcher)
        budget = artifact["sample_size_budget"]["source_records"]
        budget["attempted"] = len(artifact["source_method_rows"])
        budget["completed"] = len(artifact["source_method_rows"])
        budget["censored"] = sum(
            row["recheck_outcome"] == "access_failed" for row in artifact["source_method_rows"]
        )
        _record_phase(artifact, 6, phase_started, "primary source recheck")
        _checkpoint(checkpoint_path, artifact, started)
        progress(6, "end", f"checked={len(artifact['source_method_rows'])}")

        phase_started = time.monotonic()
        progress(7, "start", "run scoped validation subprocesses")
        if run_commands:
            _run_validation_commands(root, checkpoint_path, artifact, started)
            detail = "validation subprocess receipts stored"
        else:
            detail = "validation subprocesses disabled by isolated caller"
        _record_phase(artifact, 7, phase_started, detail)
        _checkpoint(checkpoint_path, artifact, started)
        progress(7, "end", detail)

    phase_started = time.monotonic()
    progress(8, "start", "derive, validate, and atomically write terminal evidence")
    _apply_terminal_state(artifact)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    _record_phase(artifact, 8, phase_started, "terminal validation before atomic write")
    _set_time_and_checksum(artifact, started)
    progress(8, "validation start", "validate complete in-memory artifact")
    errors = validate_artifact(artifact, root=root)
    progress(8, "validation end", f"errors={errors}")
    if errors:  # pragma: no cover - command-line execution fails closed.
        raise ValueError(f"invalid Exp7233 artifact: {errors}")
    progress(8, "write start", "atomic checkpoint and terminal deliverable")
    _set_time_and_checksum(artifact, started)
    _atomic_write_json(checkpoint_path, artifact)
    _atomic_write_json(output_path, artifact)
    progress(8, "write end", f"verdict={artifact['verdict_class']}")
    progress(8, "end", "terminal write complete")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the operator-specified V637 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI integration only.
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
        print(f"[exp7233] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7233] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['source_contract_complete_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - use the required thin wrapper.
    raise SystemExit(main())

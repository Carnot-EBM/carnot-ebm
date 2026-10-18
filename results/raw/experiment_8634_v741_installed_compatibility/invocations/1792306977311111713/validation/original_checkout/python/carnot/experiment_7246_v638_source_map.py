"""Build the V638 source-ingestion and sealed execution map.

The task is an advisory aggregation. It invokes no model and makes no science
claim. Shipped parsers and gate evaluators perform the contract checks.

Spec refs: REQ-REPORT-7246 and SCENARIO-REPORT-7246-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
import json
import os
import platform
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterator

import yaml

from carnot import experiment_7233_v637_contract as prior
from carnot.experiment_7179_v633_contract_receipt import (
    _atomic_write_bytes,
    _atomic_write_json,
    _sha256,
)
from carnot.experiment_7192_v634_source_contract import (
    _check_writable_directory,
    _fetch_url,
    reproducibility_checksum,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:  # pragma: no cover - normal CLI setup.
    sys.path.insert(0, str(SCRIPTS_DIR))

from conductor_gates import _is_quarantined  # noqa: E402
from roadmap_schema import Roadmap  # noqa: E402


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]

MILESTONE = "2026.09.638"
RUN_DATE = "20260912"
RETRIEVAL_DATE = "2026-09-12"
FIRST_TASK_ID = "exp7246-source-map"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7247-duration-class",
    "exp7248-arc-witness",
    "exp7249-arc-live",
    "exp7250-mention-canary",
    "exp7251-mention-heldout",
    "exp7252-semantic-audit",
    "exp7253-coverage-memory",
    "exp7254-coverage-learning",
    "exp7255-coverage-audit",
    "exp7256-native-controller",
    "exp7257-native-cost",
    "exp7258-board-state",
    "exp7259-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
EXPECTED_GATE_EDGE_COUNT = 5
GATE_CASES_PER_EDGE = 4
RANDOM_SEED = 7_246_202_609_12

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
STUDYING_PATH = Path("research-studying.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
MODULE_PATH = Path("python/carnot/experiment_7246_v638_source_map.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7246_v638_source_map.py")
TEST_PATH = Path("tests/python/test_experiment_7246_v638_source_map.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7246_v638_source_map.json")
RAW_DIR = Path("results/raw/experiment_7246")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7246_v638_source_map.json")

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
GATE_EVALUATOR_PATH = Path("scripts/conductor_gates.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
PROMPT_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ARC_LINT_PATH = Path("scripts/arc_levelup_guarantee_lint.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")

SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    EXCLUSION_PATH,
    Path("ops/e2e-test-plan.md"),
    REFERENCE_PATH,
    STUDYING_PATH,
    COMPLETE_PATH,
    ACTIVE_ROADMAP_PATH,
    DESIGN_PATH,
    ROADMAP_SCHEMA_PATH,
    GATE_EVALUATOR_PATH,
    GATE_LINT_PATH,
    PRIOR_LINT_PATH,
    EXCLUSION_LINT_PATH,
    PROMPT_PATH_LINT_PATH,
    ARC_LINT_PATH,
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    SPEC_COVERAGE_PATH,
    SPEC_PATH,
    Path("python/carnot/experiment_7233_v637_contract.py"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

HISTORY_PATHS = tuple(
    next((REPO_ROOT / "results").glob(f"experiment_{number}_*.json")).relative_to(REPO_ROOT)
    for number in range(7233, 7246)
)

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
POSITIVE_VERDICT = "complete_positive_v638_source_ingestion_and_execution_map"
DISQUALIFIED_VERDICT = "complete_disqualified_v638_plan_contract_incomplete_or_mismatched"
BLOCKED_VERDICT = "blocked_v638_source_map_external_prerequisite_missing"

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version this artifact; also emit experiment_id and milestone as ordinary top-level values.",
    "status": "A terminal artifact records complete or blocked; unfinished work belongs in a separate checkpoint.",
    "run_date": "Use 20260912 and retain actual UTC start and end timestamps.",
    "field_principles": "Store explanations here while leaving ordinary values at top level for consumers.",
    "preconditions_checked": "Record exact observed inputs, resource ownership, hashes and failed checks before expensive work.",
    "MODEL_SPECS": "Name only models this invocation may execute; source-model history lives in hashed sidecars.",
    "model_invoked": "Derive from actual current calls, not usable-answer count or a nested control arm.",
    "inference_substrate": "Describe the compute actually performed with a recognized literal.",
    "inference_substrate_class": "Use the closed compute class and its duration floor; never sleep or relabel to pass.",
    "execution_venue": "Use host for orchestration; actual board receipts separately name kv260, gatemate or polarfire.",
    "execution_host": "Record the real hostname separately from the closed venue vocabulary.",
    "duration_s": "Measure monotonic elapsed time for this invocation, with disjoint phase spans and no invented time.",
    "random_seed": "Freeze seeds before seeing outcomes so replay cannot select favorable runs.",
    "reproducibility_checksum": "Bind source code, input manifests, configuration and raw rows to the result.",
    "source_artifact_hashes": "Authenticate input bytes and preserve quarantine; readiness alone is insufficient.",
    "rows": "Retain every unit, arm, seed, metric, error, abstention and censoring state; aggregates must be recomputable.",
    "sample_size_budget": "State planned, attempted, completed and censored independent units and the fixed stopping rule.",
    "acceptance_gate_results": "For each frozen criterion record expected, observed and passed, plus its principle.",
    "gate_check_summary": "For every blocked_* verdict name the upstream, exact field or check, observed and expected values.",
    "verifier_is_oracle": "Expose exact-oracle use; oracle conformance cannot become learned verification evidence.",
    "honest_verdict": "Use complete_* for terminal measured findings and blocked_* for absent external prerequisites.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; a failed scientific acceptance gate forbids positive. External incompleteness is blocked, not retryable partial.",
    "validation_receipts": "Record actual command, exit code and log hash; no skipped, weakened, deleted or reverted tests.",
    "source_ingestion_complete_score": "One means the bounded ingestion and execution map completed; it does not certify science.",
    "source_method_rows": "Record primary URLs, dates, access outcome, prior mention and specific adopted or deferred method.",
    "contract_rows": "Exactly fourteen entries bind independent Markdown and YAML bytes.",
    "gate_replay_rows": "Every edge must distinguish a genuine failed metric from missing producer fields.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

METHODS: tuple[JsonDict, ...] = (
    {
        "method_id": "contcore",
        "primary_url": "https://arxiv.org/abs/2608.15277",
        "source_date": "2026-08-15",
        "prior_mention": "research-references.md:V638-PLANNER-REFRESH-20260912",
        "adopted_or_deferred": "adopt coverage-based admission and eviction over released witness signatures",
        "target_tasks": ["exp7253-coverage-memory", "exp7254-coverage-learning"],
        "implementation_cost": "medium: bounded signature index, eviction policy, and replay controls",
        "falsifying_control": "matched reset and stale-memory controls with future labels withheld",
        "retired_overlap": "does not reopen the V637 lossy committed-predicate implementation",
        "recheck_id": "contcore",
    },
    {
        "method_id": "semloc",
        "primary_url": "https://arxiv.org/abs/2603.29109",
        "source_date": "2026-03-31",
        "prior_mention": "research-references.md and research-studying.md",
        "adopted_or_deferred": "adopt typed transition anchors and counterfactual mismatch witnesses",
        "target_tasks": ["exp7248-arc-witness", "exp7249-arc-live"],
        "implementation_cost": "medium: typed witness schema over the existing ARC feedback path",
        "falsifying_control": "withhold the witness while preserving actions, budgets, and observed transitions",
        "retired_overlap": "does not expose game source or repeat the disallowed engine receipt",
        "recheck_id": "semloc",
    },
    {
        "method_id": "grounding",
        "primary_url": "https://arxiv.org/abs/2605.30054",
        "source_date": "2026-05-28",
        "prior_mention": "research-references.md:V638-PLANNER-REFRESH-20260912",
        "adopted_or_deferred": "adopt explicit unknown, source removal, and direction-change controls",
        "target_tasks": [
            "exp7250-mention-canary",
            "exp7251-mention-heldout",
            "exp7252-semantic-audit",
        ],
        "implementation_cost": "low: reuse the shipped mention compiler and independent labels",
        "falsifying_control": "paired source removal, direction reversal, and identifier renaming",
        "retired_overlap": "keeps external-text scorers and self-certified extraction retired",
        "recheck_id": None,
    },
    {
        "method_id": "retrieval_control",
        "primary_url": "https://arxiv.org/abs/2606.26476",
        "source_date": "2026-06-25",
        "prior_mention": "research-references.md:V638-PLANNER-REFRESH-20260912",
        "adopted_or_deferred": "adopt aligned, constant, random, shuffled, and oracle memory controls",
        "target_tasks": ["exp7254-coverage-learning", "exp7255-coverage-audit"],
        "implementation_cost": "low: controller-only arms over one frozen event stream",
        "falsifying_control": "shuffled and feedback-withheld memory at matched lookup cost",
        "retired_overlap": "does not revive the stored-value-gate failure as a positive result",
        "recheck_id": None,
    },
    {
        "method_id": "sparsekan",
        "primary_url": "https://arxiv.org/abs/2608.00859",
        "source_date": "2026-08-01",
        "prior_mention": "research-references.md:V638-PLANNER-REFRESH-20260912",
        "adopted_or_deferred": "adopt physical compaction and complete transfer-cost accounting; defer a KAN model",
        "target_tasks": ["exp7256-native-controller", "exp7257-native-cost"],
        "implementation_cost": "medium: persistent native ownership and durable event timing",
        "falsifying_control": "masked-only storage and host-round-trip baselines",
        "retired_overlap": "does not transfer SparseKAN FPGA gains to the Rust controller",
        "recheck_id": "sparsekan",
    },
    {
        "method_id": "ebt",
        "primary_url": "https://arxiv.org/abs/2507.02092",
        "source_date": "2025-07-02",
        "prior_mention": "research-references.md foundation anchor",
        "adopted_or_deferred": "retain as architecture context; defer learned energy claims",
        "target_tasks": ["exp7259-capstone"],
        "implementation_cost": "none in V638: citation and boundary review only",
        "falsifying_control": "oracle-distinct semantic labels and matched direct baselines",
        "retired_overlap": "external-text energy scoring remains retired",
        "recheck_id": "ebt_citations",
    },
    {
        "method_id": "arm_ebm",
        "primary_url": "https://arxiv.org/abs/2512.15605",
        "source_date": "2026-05-25",
        "prior_mention": "research-references.md foundation anchor",
        "adopted_or_deferred": "retain the function-space relation as context; defer learned verifier claims",
        "target_tasks": ["exp7259-capstone"],
        "implementation_cost": "none in V638: citation and boundary review only",
        "falsifying_control": "same-generator direct decision and shuffled-source controls",
        "retired_overlap": "ARM equivalence cannot reopen self-certified verification",
        "recheck_id": "arm_ebm_citations",
    },
)

ACCESS_SPECS: tuple[JsonDict, ...] = (
    {
        "access_id": "contcore",
        "kind": "primary_record",
        "url": "https://arxiv.org/abs/2608.15277",
        "expected_marker": "Memory-Bounded Continuation of Greedy Sampling",
        "expected_version": "v1",
    },
    {
        "access_id": "semloc",
        "kind": "primary_record",
        "url": "https://arxiv.org/abs/2603.29109",
        "expected_marker": "SemLoc: Structured Grounding",
        "expected_version": "v1",
    },
    {
        "access_id": "sparsekan",
        "kind": "primary_record",
        "url": "https://arxiv.org/abs/2608.00859",
        "expected_marker": "SparseKAN: Compressing Kolmogorov",
        "expected_version": "v1",
    },
    {
        "access_id": "ebt_citations",
        "kind": "citation_endpoint",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?fields=title,year,externalIds,url&limit=100",
        "expected_count": 35,
    },
    {
        "access_id": "arm_ebm_citations",
        "kind": "citation_endpoint",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?fields=title,year,externalIds,url&limit=100",
        "expected_count": 8,
    },
)

VALIDATION_COMMAND_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "exclusion_manifest",
    "prompt_paths",
    "gate_declarations",
    "arc_floor",
    "focused_tests",
    "focused_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "full_python_suite",
    "independent_reducer",
    "adversarial",
    "row_consistency",
)


def progress(phase: int, state: str, detail: str) -> None:
    """Flush one factual boundary so a long operation never appears silent."""

    print(f"[exp7246] phase {phase} {state}: {detail}", flush=True)


def sha256(path: Path) -> str:
    """Return the repository's standard prefixed SHA-256 value."""

    return _sha256(path)


def sha256_bytes(content: bytes) -> str:
    """Hash bytes through the same helper used for file receipts."""

    import hashlib

    return "sha256:" + hashlib.sha256(content).hexdigest()


@contextmanager
def _configured_prior() -> Iterator[None]:
    """Parameterize the shipped V637 parser without copying its validator."""

    replacements = {
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
        "COMPLETE_PATH": COMPLETE_PATH,
        "SPEC_PATH": SPEC_PATH,
        "REFERENCE_PATH": REFERENCE_PATH,
        "EXCLUSION_PATH": EXCLUSION_PATH,
        "MODULE_PATH": MODULE_PATH,
        "WRAPPER_PATH": WRAPPER_PATH,
        "TEST_PATH": TEST_PATH,
        "DEFAULT_OUTPUT_PATH": DEFAULT_OUTPUT_PATH,
        "RAW_DIR": RAW_DIR,
        "CHECKPOINT_PATH": CHECKPOINT_PATH,
        "SOURCE_PATHS": SOURCE_PATHS,
        "progress": progress,
    }
    previous = {name: getattr(prior, name) for name in replacements}
    for name, value in replacements.items():
        setattr(prior, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(prior, name, value)


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Select active then staged YAML, but only for the exact V638 milestone."""

    with _configured_prior():
        return prior.select_yaml_authority(root)


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Use shipped independent parsers with the frozen V638 roster."""

    with _configured_prior():
        return prior.evaluate_contract(markdown_text, yaml_document)


def run_gate_replays(roadmap: Mapping[str, Any], results_dir: Path) -> list[JsonDict]:
    """Run four real evaluator cases for each structured V638 edge."""

    with _configured_prior():
        rows = prior.run_gate_replays(roadmap, results_dir)
    for row in rows:
        if row["case"] == "failed":
            row["case"] = "zero"
        if row["case"] == "zero":
            payload: JsonDict | None = {"status": "complete", row["artifact_field"]: 0}
        elif row["case"] == "absent_field":
            payload = {"status": "complete"}
        elif row["case"] == "absent_file":
            payload = None
        else:
            payload = {"status": "complete", row["artifact_field"]: row["expected_value"]}
        row["fixture_payload"] = payload
        row["fixture_sha256"] = sha256_bytes(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        )
    return rows


def gate_replays_complete(rows: object) -> bool:
    """Require pass, genuine zero, missing field, and missing file per edge."""

    if not isinstance(rows, list) or len(rows) != EXPECTED_GATE_EDGE_COUNT * 4:
        return False
    expected_cases = ["passing", "zero", "absent_field", "absent_file"]
    for edge_index in range(EXPECTED_GATE_EDGE_COUNT):
        edge = [row for row in rows if row.get("edge_index") == edge_index]
        if [row.get("case") for row in edge] != expected_cases:
            return False
        if [row.get("passed") for row in edge] != [True, False, False, False]:
            return False
        if edge[1].get("observed_value") != 0:
            return False
    return True


def _observed_version(body: str) -> str | None:
    """Return the largest visible arXiv version marker without guessing."""

    versions = [int(value) for value in re.findall(r"\[v(\d+)\]", body, re.I)]
    return f"v{max(versions)}" if versions else None


def collect_source_evidence(
    fetcher: Fetcher = _fetch_url,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Make exactly three primary and two citation requests in sequence."""

    access_rows: list[JsonDict] = []
    for index, spec in enumerate(ACCESS_SPECS, 1):
        progress(4, "request start", f"{index}/{len(ACCESS_SPECS)} {spec['access_id']}")
        try:
            receipt = dict(fetcher(str(spec["url"])))
        except Exception as exc:  # noqa: BLE001 - access failure is retained evidence.
            receipt = {
                "ok": False,
                "status_code": None,
                "body": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        body = str(receipt.get("body", ""))
        accessed = receipt.get("ok") is True
        row: JsonDict = {
            "access_id": spec["access_id"],
            "kind": spec["kind"],
            "url": spec["url"],
            "retrieval_date": RETRIEVAL_DATE,
            "status_code": receipt.get("status_code"),
            "error": receipt.get("error"),
            "response_body": body,
            "response_sha256": sha256_bytes(body.encode()),
            "access_status": "http_success"
            if accessed
            else "access_failed_cached_refresh_retained",
        }
        if spec["kind"] == "primary_record":
            observed = _observed_version(body)
            marker_ok = str(spec["expected_marker"]).lower() in body.lower() if accessed else None
            row.update(
                expected_version=spec["expected_version"],
                observed_version=observed,
                title_verified=marker_ok,
                source_delta=(
                    "no_delta"
                    if accessed and observed == spec["expected_version"] and marker_ok
                    else "version_or_identity_changed"
                    if accessed
                    else "access_failed"
                ),
            )
        else:
            try:
                parsed = json.loads(body) if accessed else {}
                data = parsed.get("data") if isinstance(parsed, Mapping) else None
                count = len(data) if isinstance(data, list) else None
                next_value = parsed.get("next") if isinstance(parsed, Mapping) else None
            except json.JSONDecodeError:
                count, next_value = None, None
            row.update(
                expected_count=spec["expected_count"],
                observed_count=count,
                next_page=next_value,
                source_delta=(
                    "no_delta"
                    if accessed and count == spec["expected_count"] and next_value is None
                    else "citation_result_changed_or_malformed"
                    if accessed
                    else "access_failed"
                ),
            )
        access_rows.append(row)
        progress(4, "request end", f"{index}/{len(ACCESS_SPECS)} {row['access_status']}")

    by_id = {row["access_id"]: row for row in access_rows}
    methods: list[JsonDict] = []
    for source in METHODS:
        row = dict(source)
        recheck_id = row.pop("recheck_id")
        if recheck_id is None:
            row.update(
                access_status="ingested_from_authenticated_v638_refresh",
                source_delta="not_rechecked_in_bounded_execution",
            )
        else:
            access = by_id[recheck_id]
            row.update(
                access_status=access["access_status"],
                source_delta=access["source_delta"],
                access_receipt_id=recheck_id,
            )
        methods.append(row)
    return methods, access_rows


def build_evidence_sidecars(
    root: Path,
    raw_dir: Path,
    gate_rows: Sequence[Mapping[str, Any]],
    access_rows: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Keep historical model facts and fixtures outside current provenance."""

    historical: list[JsonDict] = []
    for relative in HISTORY_PATHS:
        path = root / relative
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"historical artifact is not a mapping: {relative}")
        historical.append(
            {
                "path": str(relative),
                "sha256": sha256(path),
                "status": data.get("status"),
                "honest_verdict": data.get("honest_verdict"),
                "verdict_class": data.get("verdict_class"),
                "MODEL_SPECS": data.get("MODEL_SPECS"),
                "model_invoked": data.get("model_invoked"),
                "model_invocation_count": data.get("model_invocation_count"),
                "model_load_count": data.get("model_load_count"),
                "model_generation_count": data.get("model_generation_count"),
                "quarantined": _is_quarantined(data),
                "accepted_for_current_values": False,
            }
        )
    history = {
        "schema": "carnot.exp7246.historical_model_receipts.v1",
        "run_date": RUN_DATE,
        "current_invocation": {
            "MODEL_SPECS": [],
            "model_invoked": False,
            "model_invocation_count": 0,
            "model_load_count": 0,
            "model_generation_count": 0,
        },
        "historical_artifacts": historical,
    }
    negative = {
        "schema": "carnot.exp7246.negative_gate_fixtures.v1",
        "run_date": RUN_DATE,
        "fixture_rows": [dict(row) for row in gate_rows if row.get("case") != "passing"],
    }
    source_access = {
        "schema": "carnot.exp7246.source_access_receipts.v1",
        "run_date": RUN_DATE,
        "request_count": len(access_rows),
        "receipts": [dict(row) for row in access_rows],
    }
    payloads = {
        "history": ("historical-model-receipts.json", history),
        "negative_fixtures": ("negative-gate-fixtures.json", negative),
        "source_access": ("source-access-receipts.json", source_access),
    }
    metadata: dict[str, JsonDict] = {}
    for name, (filename, payload) in payloads.items():
        path = raw_dir / filename
        _atomic_write_json(path, payload)
        metadata[name] = {"path": str(path.relative_to(root)), "sha256": sha256(path)}
    return metadata


def archive_lag_row(root: Path) -> JsonDict:
    """Record the archive state that exists now, not the prompt's old premise."""

    document = yaml.safe_load((root / COMPLETE_PATH).read_text(encoding="utf-8"))
    milestones = document.get("milestones", []) if isinstance(document, Mapping) else []
    latest = (
        milestones[-1].get("id") if milestones and isinstance(milestones[-1], Mapping) else None
    )
    return {
        "prompt_expected_latest_milestone": "2026.09.636",
        "observed_latest_milestone": str(latest) if latest is not None else None,
        "archive_advanced_since_prompt": str(latest) == "2026.09.637",
        "research_complete_rewritten": False,
    }


def _summary(
    failed_check: str | None,
    expected: Any,
    observed: Any,
    *,
    upstream: Any = None,
    field: Any = None,
    passed: bool = False,
) -> JsonDict:
    """Use one stable diagnostic shape for terminal outcomes."""

    return {
        "failed_check": failed_check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _hash_sources(root: Path, authority: Path | None) -> dict[str, str | None]:
    """Hash every declared input and selected authority byte-for-byte."""

    paths = (*SOURCE_PATHS, *HISTORY_PATHS, *((authority,) if authority else ()))
    hashes: dict[str, str | None] = {}
    for relative in dict.fromkeys(paths):
        try:
            hashes[str(relative)] = sha256(root / relative)
        except OSError:
            hashes[str(relative)] = None
    return hashes


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Observe exact bytes, ownership, imports, gates, and writable paths."""

    authority, roadmap, yaml_bytes, candidates = select_yaml_authority(root)
    rows = [dict(row) for row in candidates]
    rows.append(
        {
            "check": "yaml_authority",
            "upstream": "research-roadmap.yaml|research-roadmap-next.yaml",
            "field": "milestone",
            "expected_value": MILESTONE,
            "observed_value": str(authority) if authority else None,
            "available": authority is not None,
            "blocking_external": True,
        }
    )
    for relative in SOURCE_PATHS:
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative}",
                "upstream": str(relative),
                "field": "bytes",
                "expected_value": "readable_nonempty_bytes",
                "observed_value": path.stat().st_size if available else "missing_or_empty",
                "available": available,
                "blocking_external": True,
            }
        )
    try:
        spec_ok = "REQ-REPORT-7246" in (root / SPEC_PATH).read_text(encoding="utf-8")
    except OSError:
        spec_ok = False
    rows.append(
        {
            "check": "driving_requirement",
            "upstream": str(SPEC_PATH),
            "field": "REQ-*",
            "expected_value": "REQ-REPORT-7246",
            "observed_value": "REQ-REPORT-7246" if spec_ok else "missing",
            "available": spec_ok,
            "blocking_external": True,
        }
    )
    design_matches = bool(roadmap and roadmap.get("milestone_doc") == str(DESIGN_PATH))
    rows.append(
        {
            "check": "design_authority",
            "upstream": str(authority) if authority else None,
            "field": "milestone_doc",
            "expected_value": str(DESIGN_PATH),
            "observed_value": roadmap.get("milestone_doc") if roadmap else None,
            "available": design_matches and (root / DESIGN_PATH).is_file(),
            "blocking_external": True,
        }
    )
    for relative in HISTORY_PATHS:
        path = root / relative
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            available = isinstance(data, Mapping) and data.get("status") in {"complete", "blocked"}
            observed: Any = {
                "sha256": sha256(path),
                "status": data.get("status") if isinstance(data, Mapping) else None,
                "quarantined": _is_quarantined(dict(data)) if isinstance(data, Mapping) else None,
            }
        except (OSError, json.JSONDecodeError) as exc:
            available, observed = False, f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "check": f"v637_terminal:{relative}",
                "upstream": str(relative),
                "field": "status",
                "expected_value": "complete|blocked",
                "observed_value": observed,
                "available": available,
                "blocking_external": True,
            }
        )
    rows.append(
        {
            "check": "required_imports",
            "upstream": "roadmap_schema|conductor_gates|prior_contract_helpers",
            "field": "imports",
            "expected_value": "loaded",
            "observed_value": "loaded",
            "available": True,
            "blocking_external": True,
        }
    )
    if roadmap is not None:
        own = next(
            (task for task in roadmap.get("tasks", []) if task.get("id") == FIRST_TASK_ID), None
        )
        gates = own.get("gated_on", []) if isinstance(own, Mapping) else None
        rows.append(
            {
                "check": "task_upstream_gates",
                "upstream": FIRST_TASK_ID,
                "field": "gated_on",
                "expected_value": [],
                "observed_value": gates,
                "available": gates == [] or gates is None,
                "blocking_external": True,
            }
        )
    for label, directory in (
        ("result_directory", output_path.parent),
        ("raw_directory", raw_dir),
        ("checkpoint_directory", checkpoint_path.parent),
    ):
        available, observed = _check_writable_directory(directory)
        rows.append(
            {
                "check": label,
                "upstream": str(directory),
                "field": "writable",
                "expected_value": "temporary_write_succeeded",
                "observed_value": observed,
                "available": available,
                "blocking_external": True,
                "owner_uid": directory.stat().st_uid if directory.exists() else None,
            }
        )
    hashes = _hash_sources(root, authority)
    rows.append(
        {
            "check": "source_hashes",
            "upstream": str(root),
            "field": "sha256",
            "expected_value": "all required hashes present",
            "observed_value": [key for key, value in hashes.items() if value is None],
            "available": all(hashes.values()),
            "blocking_external": True,
        }
    )
    return rows, authority, roadmap, yaml_bytes


def _freeze_authorities(
    root: Path, authority: Path, yaml_bytes: bytes, raw_dir: Path
) -> list[JsonDict]:
    """Freeze exact design and YAML bytes without normalization."""

    sources = (
        ("markdown", DESIGN_PATH, (root / DESIGN_PATH).read_bytes(), raw_dir / "v638-design.md"),
        ("yaml", authority, yaml_bytes, raw_dir / "selected-roadmap.yaml"),
    )
    rows: list[JsonDict] = []
    for kind, source, content, target in sources:
        _atomic_write_bytes(target, content)
        rows.append(
            {
                "source_type": kind,
                "source_path": str(source),
                "raw_path": str(target.relative_to(root)),
                "source_sha256": sha256_bytes(content),
                "raw_sha256": sha256(target),
                "hash_matches": sha256_bytes(content) == sha256(target),
            }
        )
    return rows


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Preserve requested principles and explain additional evidence fields."""

    values = {field: f"This field keeps one V638 source-map fact auditable." for field in fields}
    values.update(REQUIRED_FIELD_PRINCIPLES)
    return values


def _base_artifact(run_date: str, checkpoint_path: Path, started_at: str) -> JsonDict:
    """Create a nonterminal checkpoint with all required consumer fields."""

    artifact: JsonDict = {
        "schema": "carnot.exp7246.v638_source_map.v1",
        "experiment_id": FIRST_TASK_ID,
        "milestone": MILESTONE,
        "status": "running",
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "preconditions_checked": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "model_invocation_count": 0,
        "model_load_count": 0,
        "model_generation_count": 0,
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node() or "unknown",
        "duration_s": 0.000001,
        "phase_spans": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "contract_rows": {
                "planned": 14,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": 14,
            },
            "gate_replays": {
                "planned": 20,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": 20,
            },
            "source_methods": {
                "planned": 7,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": 7,
            },
            "network_requests": {
                "planned": 5,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": 5,
            },
            "stopping_rule": "Stop after fourteen contract rows, four cases per gate edge, seven method mappings, and exactly five bounded requests.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _summary(
            "preconditions_not_checked", "all required inputs", "not_checked"
        ),
        "verifier_is_oracle": False,
        "honest_verdict": "partial_running_v638_source_map",
        "verdict_class": "partial",
        "validation_receipts": [],
        "source_ingestion_complete_score": 0,
        "source_method_rows": [],
        "source_access_rows": [],
        "contract_rows": [],
        "gate_replay_rows": [],
        "raw_source_rows": [],
        "sidecar_receipts": {},
        "archive_lag_row": {},
        "yaml_authority_path": None,
        "markdown_milestone": None,
        "yaml_milestone": None,
        "markdown_task_rows": [],
        "yaml_task_rows": [],
        "gate_producer_rows": [],
        "receipt_dependency_rows": [],
        "expected_id_order": list(EXPECTED_ID_ORDER),
        "markdown_id_order": [],
        "observed_id_order": [],
        "checkpoint_path": str(checkpoint_path),
        "activation_files_modified": False,
        "scientific_value_claimed": False,
        "publication_or_submission_performed": False,
        "baseline_failures": [],
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _failed_precondition(artifact: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the first unavailable external prerequisite."""

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


def _validation_complete(artifact: Mapping[str, Any]) -> bool:
    """Require all exact command names and zero recorded exits."""

    rows = artifact.get("validation_receipts")
    return (
        isinstance(rows, list)
        and [row.get("name") for row in rows] == list(VALIDATION_COMMAND_NAMES)
        and all(row.get("passed") is True for row in rows)
    )


def _sidecars_complete(artifact: Mapping[str, Any]) -> bool:
    """Require all three named sidecars and matching source hashes."""

    metadata = artifact.get("sidecar_receipts")
    hashes = artifact.get("source_artifact_hashes")
    return (
        isinstance(metadata, Mapping)
        and isinstance(hashes, Mapping)
        and set(metadata)
        == {
            "history",
            "negative_fixtures",
            "source_access",
        }
        and all(
            isinstance(receipt, Mapping)
            and hashes.get(receipt.get("path")) == receipt.get("sha256")
            for receipt in metadata.values()
        )
    )


def _acceptance_gate_results(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute every frozen structural criterion from stored rows."""

    contract = artifact.get("contract_rows")
    gates = artifact.get("gate_replay_rows")
    methods = artifact.get("source_method_rows")
    access = artifact.get("source_access_rows")
    raw = artifact.get("raw_source_rows")
    hashes = artifact.get("source_artifact_hashes")
    criteria = (
        (
            "contract_exact",
            "14 matching exp7246-through-exp7259 rows",
            [row.get("unit_id") for row in contract] if isinstance(contract, list) else None,
            isinstance(contract, list)
            and len(contract) == 14
            and [row.get("unit_id") for row in contract] == list(EXPECTED_ID_ORDER)
            and all(row.get("passed") is True for row in contract),
            "Independent Markdown and YAML values must agree; readable disagreement is disqualifying.",
        ),
        (
            "gate_replays",
            20,
            len(gates) if isinstance(gates, list) else None,
            gate_replays_complete(gates),
            "Each real edge must distinguish zero from missing fields and files.",
        ),
        (
            "source_methods",
            7,
            len(methods) if isinstance(methods, list) else None,
            isinstance(methods, list)
            and len(methods) == 7
            and all(
                row.get("implementation_cost") and row.get("falsifying_control") for row in methods
            ),
            "Bounded ingestion must map every requested method to cost and a falsifying control.",
        ),
        (
            "request_budget",
            "3 primary plus 2 citation requests",
            [row.get("kind") for row in access] if isinstance(access, list) else None,
            isinstance(access, list)
            and [row.get("kind") for row in access].count("primary_record") == 3
            and [row.get("kind") for row in access].count("citation_endpoint") == 2,
            "Access failures stay visible and cannot cause extra requests.",
        ),
        (
            "raw_authority_hashes",
            2,
            len(raw) if isinstance(raw, list) else None,
            isinstance(raw, list)
            and len(raw) == 2
            and all(row.get("hash_matches") is True for row in raw),
            "The exact Markdown and YAML bytes must survive the freeze unchanged.",
        ),
        (
            "evidence_sidecars",
            True,
            _sidecars_complete(artifact),
            _sidecars_complete(artifact),
            "Historical models and negative fixtures must stay outside current invocation fields.",
        ),
        (
            "source_hashes",
            "all present",
            "all present"
            if isinstance(hashes, Mapping) and hashes and all(hashes.values())
            else "missing",
            isinstance(hashes, Mapping) and bool(hashes) and all(hashes.values()),
            "Every consumed local input and sidecar must have an exact hash.",
        ),
        (
            "archive_observed",
            "2026.09.637 observed without rewrite",
            artifact.get("archive_lag_row"),
            isinstance(artifact.get("archive_lag_row"), Mapping)
            and artifact["archive_lag_row"].get("observed_latest_milestone") == "2026.09.637"
            and artifact["archive_lag_row"].get("research_complete_rewritten") is False,
            "The receipt records the actual archive state and does not modify it.",
        ),
        (
            "validation_receipts",
            list(VALIDATION_COMMAND_NAMES),
            [row.get("name") for row in artifact.get("validation_receipts", [])]
            if isinstance(artifact.get("validation_receipts"), list)
            else None,
            _validation_complete(artifact),
            "All scoped checks and the single required full Python suite must retain exact outcomes.",
        ),
    )
    return [
        {
            "criterion": name,
            "expected_value": expected,
            "observed_value": observed,
            "passed": bool(passed),
            "principle": principle,
        }
        for name, expected, observed, passed, principle in criteria
    ]


def _score_from_artifact(artifact: Mapping[str, Any]) -> int:
    """Return one only when all advisory ingestion criteria pass."""

    rows = _acceptance_gate_results(artifact)
    return int(len(rows) == 9 and all(row["passed"] for row in rows))


def _failure_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Name the first precondition, Markdown, row, or validation failure."""

    failed = _failed_precondition(artifact)
    if failed is not None:
        return _summary(
            str(failed.get("check")),
            failed.get("expected_value"),
            failed.get("observed_value"),
            upstream=failed.get("upstream"),
            field=failed.get("field"),
        )
    if artifact.get("markdown_milestone") != MILESTONE:
        return _summary(
            "markdown_milestone",
            MILESTONE,
            artifact.get("markdown_milestone"),
            upstream=str(DESIGN_PATH),
            field="milestone",
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
                field="id|title|deliverable|gates",
            )
    acceptance = _acceptance_gate_results(artifact)
    row = next((item for item in acceptance if item["passed"] is not True), None)
    return _summary(
        f"acceptance:{row['criterion']}" if row else "stored_source_map",
        row.get("expected_value") if row else "complete",
        row.get("observed_value") if row else "incomplete",
        field=row.get("criterion") if row else "acceptance_gate_results",
    )


def _apply_terminal_state(artifact: JsonDict) -> None:
    """Derive the terminal class from evidence, not intended scope."""

    artifact["acceptance_gate_results"] = _acceptance_gate_results(artifact)
    artifact["source_ingestion_complete_score"] = _score_from_artifact(artifact)
    failed = _failed_precondition(artifact)
    if failed is not None:
        artifact.update(
            status="blocked",
            inference_substrate="blocked_no_run",
            inference_substrate_class="blocked_no_run",
            verdict_class="blocked",
            honest_verdict=BLOCKED_VERDICT,
            gate_check_summary=_failure_summary(artifact),
        )
    elif artifact["source_ingestion_complete_score"] == 1:
        artifact.update(
            status="complete",
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class="aggregation",
            verdict_class="positive",
            honest_verdict=POSITIVE_VERDICT,
            gate_check_summary=_summary(
                None,
                "all V638 source-map criteria pass",
                "all V638 source-map criteria pass",
                passed=True,
            ),
        )
    else:
        artifact.update(
            status="complete",
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class="aggregation",
            verdict_class="disqualified",
            honest_verdict=DISQUALIFIED_VERDICT,
            gate_check_summary=_failure_summary(artifact),
        )


def _set_time_and_checksum(artifact: JsonDict, started: float) -> None:
    """Measure real elapsed work and bind the current complete evidence."""

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.000001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def _checkpoint(path: Path, artifact: JsonDict, started: float) -> None:
    """Write mutable state only to the declared checkpoint path."""

    _set_time_and_checksum(artifact, started)
    _atomic_write_json(path, artifact)


def _record_phase(artifact: JsonDict, phase: int, started: float, detail: str) -> None:
    """Append one measured, disjoint phase span."""

    end = time.monotonic()
    origin = artifact.setdefault("monotonic_origin", started)
    artifact["phase_spans"].append(
        {
            "phase": phase,
            "detail": detail,
            "start_elapsed_s": round(started - origin, 6),
            "end_elapsed_s": round(end - origin, 6),
            "duration_s": round(end - started, 6),
        }
    )


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute raw-row shapes without trusting aggregate acceptance fields."""

    errors: list[str] = []
    rows = artifact.get("contract_rows")
    if not isinstance(rows, list) or len(rows) != 14:
        errors.append("contract_row_count")
    elif any(row.get("passed") != all(row.get("checks", {}).values()) for row in rows):
        errors.append("contract_row_reduction")
    if not gate_replays_complete(artifact.get("gate_replay_rows")):
        errors.append("gate_replay_reduction")
    methods = artifact.get("source_method_rows")
    if not isinstance(methods, list) or len(methods) != 7:
        errors.append("source_method_count")
    raw = artifact.get("raw_source_rows")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(row.get("source_sha256") != row.get("raw_sha256") for row in raw)
    ):
        errors.append("raw_hash_reduction")
    return errors


def validate_artifact(artifact: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Recompute schema, lifecycle, hashes, rows, score, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if artifact.get("schema") != "carnot.exp7246.v638_source_map.v1":
        errors.append("schema_invalid")
    if artifact.get("experiment_id") != FIRST_TASK_ID or artifact.get("milestone") != MILESTONE:
        errors.append("identity_invalid")
    if artifact.get("status") not in {"complete", "blocked"}:
        errors.append("status_invalid")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_invalid")
    for field in ("started_at_utc", "completed_at_utc"):
        try:
            value = datetime.fromisoformat(str(artifact.get(field)))
        except ValueError:
            errors.append(f"{field}_invalid")
        else:
            if value.tzinfo is None:
                errors.append(f"{field}_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or any(
        principles.get(field) != text for field, text in REQUIRED_FIELD_PRINCIPLES.items()
    ):
        errors.append("field_principles_invalid")
    if (
        artifact.get("MODEL_SPECS") != []
        or artifact.get("model_invoked") is not False
        or any(
            artifact.get(field) != 0
            for field in ("model_invocation_count", "model_load_count", "model_generation_count")
        )
    ):
        errors.append("model_contract_invalid")
    if artifact.get("execution_venue") != "host" or artifact.get("execution_host") != (
        platform.node() or "unknown"
    ):
        errors.append("execution_invalid")
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
    groups = ("contract_rows", "gate_replays", "source_methods", "network_requests")
    if (
        not isinstance(budget, Mapping)
        or not budget.get("stopping_rule")
        or any(
            not isinstance(budget.get(group), Mapping)
            or any(
                field not in budget[group]
                for field in ("planned", "attempted", "completed", "censored", "independent_units")
            )
            for group in groups
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
            for relative, expected in hashes.items():
                try:
                    observed = sha256(root / str(relative))
                except OSError:
                    observed = None
                if observed != expected:
                    errors.append("source_hash_mismatch")
                    break
        errors.extend(independent_reduce(artifact))
    receipts = artifact.get("validation_receipts")
    if not isinstance(receipts, list) or [row.get("name") for row in receipts] not in (
        [],
        list(VALIDATION_COMMAND_NAMES),
    ):
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
        "source_ingestion_complete_score",
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


def validation_commands(
    root: Path, artifact_path: Path, authority: Path
) -> tuple[tuple[str, list[str]], ...]:
    """Return shipped plan checks and explicit changed-scope verification."""

    python = str(root / ".venv/bin/python")
    roadmap = str(root / authority)
    artifact = str(artifact_path)
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7246_v638_source_map import independent_reduce;"
        f"errors=independent_reduce(json.loads(pathlib.Path({artifact!r}).read_text()));"
        "print(errors);sys.exit(bool(errors))"
    )
    coverage_file = "/tmp/.coverage-exp7246-v638"
    pytest_base = "/tmp/exp7246-v638-focused-pytest"
    return (
        ("roadmap_schema", [python, "-u", "-c", schema_code]),
        ("prior_failure", [python, "-u", str(root / PRIOR_LINT_PATH), roadmap]),
        ("exclusion_manifest", [python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap]),
        (
            "prompt_paths",
            [python, "-u", str(root / PROMPT_PATH_LINT_PATH), "prompt-paths", roadmap],
        ),
        ("gate_declarations", [python, "-u", str(root / GATE_LINT_PATH), roadmap]),
        ("arc_floor", [python, "-u", str(root / ARC_LINT_PATH), roadmap, "--min", "1"]),
        (
            "focused_tests",
            [
                str(root / ".venv/bin/coverage"),
                "run",
                f"--data-file={coverage_file}",
                f"--include=*/{MODULE_PATH.name}",
                "-m",
                "pytest",
                "-o",
                "addopts=",
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
        ("mypy", [str(root / ".venv/bin/mypy"), str(root / MODULE_PATH)]),
        (
            "scoped_spec_coverage",
            [python, "-u", str(root / SPEC_COVERAGE_PATH), str(root / TEST_PATH)],
        ),
        ("full_python_suite", [str(root / ".venv/bin/pytest"), "tests/python", "-q"]),
        ("independent_reducer", [python, "-u", "-c", reducer_code]),
        ("adversarial", [python, "-u", str(root / ADVERSARIAL_PATH), artifact]),
        ("row_consistency", [python, "-u", str(root / ROW_LINT_PATH), artifact]),
    )


def _run_streaming_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_s: float,
    heartbeat_s: float,
    operation: str,
) -> JsonDict:
    """Stream a process group and bound pipes inherited by child workers."""

    started = time.monotonic()
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
    )
    assert process.stdout is not None
    pipe_fd = process.stdout.fileno()
    selector = selectors.DefaultSelector()
    selector.register(pipe_fd, selectors.EVENT_READ)
    chunks: list[bytes] = []
    pipe_open = True
    next_heartbeat = started + heartbeat_s
    timed_out = False
    while process.poll() is None:
        now = time.monotonic()
        if now - started >= timeout_s:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:  # pragma: no cover - child exited during poll.
                pass
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:  # pragma: no cover - hostile native child.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=0.5)
            break
        wait_s = min(0.25, max(next_heartbeat - now, 0.001), timeout_s - (now - started))
        events = selector.select(timeout=max(wait_s, 0.001))
        if process.poll() is not None:
            break
        for _key, _mask in events:
            chunk = os.read(pipe_fd, 65_536)
            if chunk:
                chunks.append(chunk)
                print(chunk.decode(errors="replace"), end="", flush=True)
            else:
                selector.unregister(pipe_fd)
                pipe_open = False
        now = time.monotonic()
        if now >= next_heartbeat:
            print(
                f"[exp7246] heartbeat elapsed_s={now - started:.1f} completed_units=0 "
                f"operation={operation}",
                flush=True,
            )
            next_heartbeat = now + heartbeat_s

    if timed_out:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:  # pragma: no cover - process group already reaped.
            pass
    elif process.returncode is None:  # pragma: no cover - defensive poll race.
        process.wait()
    if pipe_open:
        while selector.select(timeout=0):
            chunk = os.read(pipe_fd, 65_536)
            if not chunk:
                break
            chunks.append(chunk)
            print(chunk.decode(errors="replace"), end="", flush=True)
    selector.close()
    process.stdout.close()
    stdout = b"".join(chunks).decode(errors="replace")[-12_000:]
    return {
        "exit_code": 124 if timed_out else int(process.returncode or 0),
        "duration_s": max(round(time.monotonic() - started, 6), 0.000001),
        "output": stdout,
        "stdout": stdout,
        "stderr": "",
        "timed_out": timed_out,
    }


def run_validation_commands(
    root: Path,
    checkpoint_path: Path,
    artifact: JsonDict,
    started: float,
    *,
    commands: Sequence[tuple[str, list[str]]] | None = None,
) -> list[JsonDict]:
    """Stream commands, retain exact exits, and hash each complete log."""

    selected = commands or validation_commands(
        root, checkpoint_path, Path(str(artifact["yaml_authority_path"]))
    )
    rows: list[JsonDict] = []
    for index, (name, command) in enumerate(selected, 1):
        if "schema" in artifact:
            _apply_terminal_state(artifact)
            artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
            artifact["validation_receipts"] = list(rows)
            _checkpoint(checkpoint_path, artifact, started)
        progress(6, "subprocess start", f"{index}/{len(selected)} {name}")
        receipt = _run_streaming_command(
            command,
            cwd=root,
            timeout_s=1200,
            heartbeat_s=60,
            operation=f"exp7246:{name}",
        )
        stdout = str(receipt.get("stdout", ""))
        stderr = str(receipt.get("stderr", ""))
        row = {
            "name": name,
            "command": shlex.join(command),
            "exit_code": int(receipt["exit_code"]),
            "duration_s": receipt["duration_s"],
            "timed_out": receipt["timed_out"],
            "log_sha256": sha256_bytes((stdout + "\0" + stderr).encode()),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "passed": receipt["exit_code"] == 0,
            "baseline_failure": name == "full_python_suite" and receipt["exit_code"] != 0,
        }
        rows.append(row)
        if "schema" in artifact:
            artifact["validation_receipts"] = list(rows)
            artifact["baseline_failures"] = [
                item for item in rows if item["baseline_failure"] is True
            ]
            _checkpoint(checkpoint_path, artifact, started)
        progress(6, "subprocess end", f"{index}/{len(selected)} {name} exit={row['exit_code']}")
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
    """Build one positive, disqualified, or externally blocked receipt."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress(0, "start", "authenticate inputs, imports, ownership, and outputs")
    phase_started = time.monotonic()
    artifact = _base_artifact(run_date, checkpoint_path, started_at)
    _checkpoint(checkpoint_path, artifact, started)
    preconditions, authority, roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["yaml_authority_path"] = str(authority) if authority else None
    artifact["source_artifact_hashes"] = _hash_sources(root, authority)
    artifact["archive_lag_row"] = archive_lag_row(root)
    _record_phase(artifact, 0, phase_started, "preconditions")
    _checkpoint(checkpoint_path, artifact, started)
    failed = _failed_precondition(artifact)
    progress(0, "end", f"completed={len(preconditions)} blocked={failed is not None}")

    if failed is None:
        assert authority is not None and roadmap is not None and yaml_bytes is not None

        phase_started = time.monotonic()
        progress(1, "start", "bind zero current model calls")
        artifact.update(
            MODEL_SPECS=[],
            model_invoked=False,
            model_invocation_count=0,
            model_load_count=0,
            model_generation_count=0,
        )
        _record_phase(artifact, 1, phase_started, "no-model declaration")
        _checkpoint(checkpoint_path, artifact, started)
        progress(1, "end", "current model/load/generation counters=0")

        phase_started = time.monotonic()
        progress(2, "start", "freeze and independently parse Markdown and YAML")
        artifact["raw_source_rows"] = _freeze_authorities(root, authority, yaml_bytes, raw_dir)
        contract = evaluate_contract((root / DESIGN_PATH).read_text(encoding="utf-8"), roadmap)
        for field in (
            "markdown_milestone",
            "yaml_milestone",
            "markdown_task_rows",
            "yaml_task_rows",
            "contract_rows",
            "gate_producer_rows",
            "receipt_dependency_rows",
            "expected_id_order",
            "markdown_id_order",
            "observed_id_order",
        ):
            artifact[field] = contract[field]
        artifact["rows"] = artifact["contract_rows"]
        budget = artifact["sample_size_budget"]["contract_rows"]
        budget.update(
            attempted=len(artifact["contract_rows"]), completed=len(artifact["contract_rows"])
        )
        for row in artifact["raw_source_rows"]:
            artifact["source_artifact_hashes"][row["raw_path"]] = row["raw_sha256"]
        _record_phase(artifact, 2, phase_started, "authority freeze and contract parse")
        _checkpoint(checkpoint_path, artifact, started)
        progress(2, "end", f"completed=14 contract_match={contract['passed']}")

        phase_started = time.monotonic()
        progress(3, "start", "run schema and four cases for each gate edge")
        Roadmap.model_validate(roadmap)
        with tempfile.TemporaryDirectory(prefix="exp7246-gates-", dir="/tmp") as directory:
            artifact["gate_replay_rows"] = run_gate_replays(roadmap, Path(directory))
        budget = artifact["sample_size_budget"]["gate_replays"]
        budget.update(
            attempted=len(artifact["gate_replay_rows"]), completed=len(artifact["gate_replay_rows"])
        )
        _record_phase(artifact, 3, phase_started, "schema and real gate replay")
        _checkpoint(checkpoint_path, artifact, started)
        progress(3, "end", f"completed={len(artifact['gate_replay_rows'])}")

        phase_started = time.monotonic()
        progress(4, "start", "ingest seven methods with five bounded requests")
        methods, access = collect_source_evidence(fetcher)
        artifact["source_method_rows"] = methods
        artifact["source_access_rows"] = [
            {key: value for key, value in row.items() if key != "response_body"} for row in access
        ]
        method_budget = artifact["sample_size_budget"]["source_methods"]
        method_budget.update(attempted=len(methods), completed=len(methods))
        request_budget = artifact["sample_size_budget"]["network_requests"]
        request_budget.update(
            attempted=len(access),
            completed=len(access),
            censored=sum(row["access_status"].startswith("access_failed") for row in access),
        )
        _record_phase(artifact, 4, phase_started, "bounded source ingestion")
        _checkpoint(checkpoint_path, artifact, started)
        progress(4, "end", f"methods={len(methods)} requests={len(access)}")

        phase_started = time.monotonic()
        progress(5, "start", "write hashed history, fixture, and source sidecars")
        sidecars = build_evidence_sidecars(root, raw_dir, artifact["gate_replay_rows"], access)
        artifact["sidecar_receipts"] = sidecars
        for receipt in sidecars.values():
            artifact["source_artifact_hashes"][receipt["path"]] = receipt["sha256"]
        _record_phase(artifact, 5, phase_started, "isolated evidence sidecars")
        _checkpoint(checkpoint_path, artifact, started)
        progress(5, "end", f"completed={len(sidecars)}")

        phase_started = time.monotonic()
        progress(6, "start", "run scoped and required validation subprocesses")
        artifact["validation_receipts"] = run_validation_commands(
            root, checkpoint_path, artifact, started
        )
        artifact["baseline_failures"] = [
            row for row in artifact["validation_receipts"] if row["baseline_failure"] is True
        ]
        _record_phase(artifact, 6, phase_started, "validation subprocesses")
        _checkpoint(checkpoint_path, artifact, started)
        progress(6, "end", f"completed={len(artifact['validation_receipts'])}")
    else:
        for phase in range(1, 7):
            phase_started = time.monotonic()
            progress(phase, "start", "skipped after external prerequisite block")
            _record_phase(
                artifact, phase, phase_started, "skipped after external prerequisite block"
            )
            _checkpoint(checkpoint_path, artifact, started)
            progress(phase, "end", "completed=0 skipped=1")

    phase_started = time.monotonic()
    progress(7, "start", "validate and atomically write terminal artifact")
    _apply_terminal_state(artifact)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    _record_phase(artifact, 7, phase_started, "terminal derivation")
    artifact.pop("monotonic_origin", None)
    _set_time_and_checksum(artifact, started)
    progress(7, "validation start", "independent in-memory validation")
    errors = validate_artifact(artifact, root=root)
    progress(7, "validation end", f"errors={errors}")
    if errors:  # pragma: no cover - CLI must fail closed on internal defects.
        raise ValueError(f"invalid Exp7246 artifact: {errors}")
    progress(7, "write start", "atomic checkpoint and terminal output")
    _set_time_and_checksum(artifact, started)
    _atomic_write_json(checkpoint_path, artifact)
    _atomic_write_json(output_path, artifact)
    progress(7, "write end", f"verdict={artifact['verdict_class']}")
    progress(7, "end", "terminal write complete")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the fixed V638 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI integration.
    """Run the source map and accept any internally valid terminal class."""

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
        print(f"[exp7246] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7246] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['source_ingestion_complete_score']}",
        flush=True,
    )
    return 0

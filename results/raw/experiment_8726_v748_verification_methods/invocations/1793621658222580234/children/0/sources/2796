"""Build the V639 source ingestion and exact execution contract.

This task records plan structure and literature dispositions. It does not run a
model and does not convert an external paper result into Carnot evidence. The
module reuses the shipped independent roadmap parser and real gate evaluator.

Spec refs: REQ-REPORT-7260 and SCENARIO-REPORT-7260-*.
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

from carnot import experiment_7246_v638_source_map as shipped
from carnot.experiment_7179_v633_contract_receipt import (
    _atomic_write_bytes,
    _atomic_write_json,
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

MILESTONE = "2026.09.639"
RUN_DATE = "20260913"
RETRIEVAL_DATE = "2026-09-13"
FIRST_TASK_ID = "exp7260-source-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7261-compute-contract",
    "exp7262-arc-witness-receipt",
    "exp7263-arc-live",
    "exp7264-mention-canary",
    "exp7265-mention-heldout",
    "exp7266-semantic-audit",
    "exp7267-recognition-prototype",
    "exp7268-recognition-learning",
    "exp7269-recognition-audit",
    "exp7270-durable-profile",
    "exp7271-delta-log",
    "exp7272-board-state",
    "exp7273-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
EXPECTED_GATE_EDGE_COUNT = 7
RANDOM_SEED = 7_260_202_609_13

DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
ACTIVE_ROADMAP_PATH = Path("research-roadmap.yaml")
NEXT_ROADMAP_PATH = Path("research-roadmap-next.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
REFERENCE_PATH = Path("research-references.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PLAN_PATH = Path("ops/e2e-test-plan.md")
HISTORY_PATH = Path("results/experiment_7259_v638_capstone.json")

ROADMAP_SCHEMA_PATH = Path("scripts/roadmap_schema.py")
GATE_EVALUATOR_PATH = Path("scripts/conductor_gates.py")
GATE_LINT_PATH = Path("scripts/audit_roadmap_gates.py")
PRIOR_LINT_PATH = Path("scripts/validate_prior_failures.py")
EXCLUSION_LINT_PATH = Path("scripts/exclusion_manifest_lint.py")
PROMPT_PATH_LINT_PATH = Path("scripts/harness_consumer_checks.py")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")
SPEC_COVERAGE_PATH = Path("scripts/check_spec_coverage.py")

MODULE_PATH = Path("python/carnot/experiment_7260_v639_source_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7260_v639_source_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7260_v639_source_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7260_v639_source_contract.json")
RAW_DIR = Path("results/raw/experiment_7260")
RAW_CANDIDATE_PATH = RAW_DIR / "measured-terminal-candidate.json"
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7260_v639_source_contract.json")

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
    "generation_calls_attempted": 0,
    "generation_calls_completed": 0,
    "usable_answers": 0,
}

INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
INFERENCE_SUBSTRATE_CLASS = "aggregation"
POSITIVE_VERDICT = "complete_circular_positive_v639_source_contract_exact_no_science_claim"
DISQUALIFIED_VERDICT = "complete_disqualified_v639_source_contract_mismatched_or_invalid"
BLOCKED_VERDICT = "blocked_v639_source_contract_external_prerequisite_missing"

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version the result; retain ordinary top-level experiment_id and milestone.",
    "status": "Use complete or blocked only for terminal work; unfinished work stays in a separate checkpoint.",
    "run_date": "Use 20260913, with actual UTC start/end timestamps, so dated evidence is auditable.",
    "field_principles": "Store explanations here; consumers read ordinary top-level values, not nested wrappers.",
    "preconditions_checked": "Retain observed input hashes, resource ownership and failures before expensive work.",
    "MODEL_SPECS": "Declare models executable in this invocation; keep historical model metadata in hashed sidecars.",
    "model_invoked": "Derive from actual calls; a parse failure does not erase a model invocation.",
    "invocation_counts": "Separate attempted/completed loads and generation calls from usable answers.",
    "inference_substrate": "Use an existing recognized literal that describes actual computation.",
    "inference_substrate_class": "Declare actual compute: full generation 60s, bounded generation 10s, load-only 2s; never pad time.",
    "execution_venue": "Use host for host orchestration; identify real boards separately in board rows.",
    "duration_s": "Measure monotonic invocation time and disjoint phase spans; do not invent elapsed time.",
    "random_seed": "Freeze independent-unit seeds before inspecting outcomes.",
    "reproducibility_checksum": "Bind code, input manifests, configuration and raw evidence to the result.",
    "source_artifact_hashes": "Authenticate exact inputs and preserve quarantine and retirement state.",
    "rows": "Retain each independent unit, arm, seed, metric, error, abstention and censoring state for recomputation.",
    "sample_size_budget": "Record planned, attempted, completed and censored units and the fixed stopping rule.",
    "acceptance_gate_results": "Each criterion retains expected, observed, passed and principle; completion is separate from value.",
    "gate_check_summary": "For blocked_* name the upstream, exact field/check, observed value and expected value.",
    "verifier_is_oracle": "Expose shared evaluator/verifier authority; exact conformance is not learned correctness.",
    "honest_verdict": "Use complete_* for terminal measurements, blocked_* for external absence, and explain the finding.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle=true forbids positive; failed scientific gates forbid positive. Only incomplete own work is partial; unchanged external blocks are blocked.",
    "validation_receipts": "Record actual command, exit code and log hash; preserve failures and never suppress checks.",
    "source_contract_complete_score": "One means all 14 contract rows and bounded source-disposition rows are recorded and agree.",
    "contract_rows": "One literal row per YAML task detects stale design, missing paths and changed gates.",
    "source_rows": "Version, access status and method-to-task mappings separate discovery from adoption.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

ACCESS_SPECS: tuple[JsonDict, ...] = (
    {
        "access_id": "warm_start_recognition",
        "url": "https://arxiv.org/abs/2606.21253",
        "expected_title": "Gradient-Free Warm-Start Library Recovery",
        "refresh_version": "v2",
    },
    {
        "access_id": "interactive_acquisition",
        "url": "https://arxiv.org/abs/2509.24489",
        "expected_title": "Query-Driven Interactive Refinement",
        "refresh_version": "v1",
    },
    {
        "access_id": "span_grounding",
        "url": "https://arxiv.org/abs/2607.00895",
        "expected_title": "Beyond Document Grounding",
        "refresh_version": "v1",
    },
    {
        "access_id": "cost_accounting",
        "url": "https://arxiv.org/abs/2602.15985",
        "expected_title": "FPGA Ising Decomposition",
        "refresh_version": "v1",
    },
)

SOURCE_DISPOSITIONS: tuple[JsonDict, ...] = (
    {
        "row_id": "warm_start_recognition_adapt",
        "method_family": "warm_start_recognition",
        "disposition": "adapt",
        "method": "actively distinguish stored constraint hypotheses before reactivation",
        "target_tasks": [
            "exp7267-recognition-prototype",
            "exp7268-recognition-learning",
            "exp7269-recognition-audit",
        ],
        "input_needs": "released event prefixes, stored hypotheses, hidden change boundaries, and delayed labels",
        "falsifying_controls": [
            "reset",
            "random query",
            "full memory",
            "shuffled association",
            "overlapping regimes",
        ],
        "retirement_overlap": "does not revive the V638 always-rebased archive or claim autonomous segmentation from supplied boundaries",
        "access_id": "warm_start_recognition",
    },
    {
        "row_id": "interactive_acquisition_implement",
        "method_family": "interactive_acquisition",
        "disposition": "implement",
        "method": "select released-feedback queries that distinguish surviving hypotheses and charge every query",
        "target_tasks": ["exp7267-recognition-prototype", "exp7268-recognition-learning"],
        "input_needs": "query candidates, explicit released feedback, hypothesis votes, and per-query cost",
        "falsifying_controls": ["random query", "fixed query order", "feedback withheld"],
        "retirement_overlap": "does not make unsupported constraints true by weight changes or reopen lossy predicate commits",
        "access_id": "interactive_acquisition",
    },
    {
        "row_id": "span_grounding_adapt",
        "method_family": "span_grounding",
        "disposition": "adapt",
        "method": "retain exact source and claim offsets with localized held-out error categories",
        "target_tasks": [
            "exp7264-mention-canary",
            "exp7265-mention-heldout",
            "exp7266-semantic-audit",
        ],
        "input_needs": "authenticated source bytes, claim bytes, exact offsets, and independent held-out labels",
        "falsifying_controls": ["source removal", "direction reversal", "identifier renaming"],
        "retirement_overlap": "keeps external-text scorers and self-certified extraction retired",
        "access_id": "span_grounding",
    },
    {
        "row_id": "cost_accounting_implement",
        "method_family": "cost_accounting",
        "disposition": "implement",
        "method": "decompose durable event cost across lookup, update, validation, transfer, and commit tails",
        "target_tasks": ["exp7270-durable-profile", "exp7271-delta-log"],
        "input_needs": "matched durable event traces, representation bytes, operation counts, and storage semantics",
        "falsifying_controls": ["host round trip", "no durability", "matched full rewrite"],
        "retirement_overlap": "does not transfer custom Ising hardware speed claims to host storage work",
        "access_id": "cost_accounting",
    },
    {
        "row_id": "cost_accounting_architecture_defer",
        "method_family": "cost_accounting",
        "disposition": "defer",
        "method": "defer a new KAN or custom Ising architecture until extraction yields usable evidence",
        "target_tasks": ["exp7270-durable-profile"],
        "input_needs": "usable extraction training evidence and an available target device",
        "falsifying_controls": ["complete host cost profile", "matched representation baseline"],
        "retirement_overlap": "preserves the unavailable-custom-chip and failed small-graph parallel-update boundaries",
        "access_id": "cost_accounting",
    },
)

VALIDATION_COMMAND_NAMES = (
    "roadmap_schema",
    "prior_failure",
    "exclusion_manifest",
    "prompt_paths",
    "gate_declarations",
    "focused_tests",
    "focused_coverage",
    "ruff_check",
    "ruff_format",
    "mypy",
    "scoped_spec_coverage",
    "e2e_gate_controls",
    "independent_reducer",
    "adversarial",
    "row_consistency",
)
PRETERMINAL_VALIDATION_COUNT = 12


def progress(phase: int, state: str, detail: str) -> None:
    """Flush one factual boundary so long work stays externally observable."""

    print(f"[exp7260] phase {phase} {state}: {detail}", flush=True)


def sha256(path: Path) -> str:
    """Return the shared prefixed SHA-256 representation."""

    return shipped.sha256(path)


def sha256_bytes(content: bytes) -> str:
    """Hash sidecar bytes with the same representation as file hashes."""

    return shipped.sha256_bytes(content)


@contextmanager
def _configured_shipped_contract() -> Iterator[None]:
    """Parameterize shipped parsers without copying their contract logic."""

    replacements = {
        "MILESTONE": MILESTONE,
        "RUN_DATE": RUN_DATE,
        "RETRIEVAL_DATE": RETRIEVAL_DATE,
        "FIRST_TASK_ID": FIRST_TASK_ID,
        "EXPECTED_ID_ORDER": EXPECTED_ID_ORDER,
        "EXPECTED_TASK_COUNT": EXPECTED_TASK_COUNT,
        "EXPECTED_GATE_EDGE_COUNT": EXPECTED_GATE_EDGE_COUNT,
        "GATE_CASES_PER_EDGE": 4,
        "RANDOM_SEED": RANDOM_SEED,
        "DESIGN_PATH": DESIGN_PATH,
        "ACTIVE_ROADMAP_PATH": ACTIVE_ROADMAP_PATH,
        "NEXT_ROADMAP_PATH": NEXT_ROADMAP_PATH,
        "progress": progress,
    }
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
    """Select the active V639 YAML, or staged V639 before activation."""

    with _configured_shipped_contract():
        return shipped.select_yaml_authority(root)


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Independently parse both sources through the shipped contract parser."""

    with _configured_shipped_contract():
        contract = shipped.evaluate_contract(markdown_text, yaml_document)
    for row in contract["contract_rows"]:
        row["censored"] = False
    return contract


def run_gate_replays(roadmap: Mapping[str, Any], results_dir: Path) -> list[JsonDict]:
    """Run true, false, missing-field, and missing-file real gate controls."""

    with _configured_shipped_contract():
        rows = shipped.run_gate_replays(roadmap, results_dir)
    case_names = {
        "passing": "true",
        "zero": "false",
        "absent_field": "missing_field",
        "absent_file": "missing_file",
    }
    for row in rows:
        row["case"] = case_names[str(row["case"])]
        row["observation"] = {
            "passed": row["passed"],
            "observed_value": row.get("observed_value"),
            "reason": row.get("reason"),
            "fixture_payload": row.get("fixture_payload"),
        }
        row["research_result"] = False
    return rows


def gate_replays_complete(rows: object) -> bool:
    """Require four ordered and distinct observations for every V639 edge."""

    if not isinstance(rows, list) or len(rows) != EXPECTED_GATE_EDGE_COUNT * 4:
        return False
    expected_cases = ["true", "false", "missing_field", "missing_file"]
    for edge_index in range(EXPECTED_GATE_EDGE_COUNT):
        edge = [row for row in rows if row.get("edge_index") == edge_index]
        if [row.get("case") for row in edge] != expected_cases:
            return False
        if [row.get("passed") for row in edge] != [True, False, False, False]:
            return False
        if edge[1].get("observed_value") != 0:
            return False
        observations = [json.dumps(row.get("observation"), sort_keys=True) for row in edge]
        if len(set(observations)) != 4:
            return False
    return True


def observed_version(body: str) -> str | None:
    """Return the largest explicit arXiv version marker without guessing."""

    versions = [int(value) for value in re.findall(r"\[v(\d+)\]", body, re.I)]
    return f"v{max(versions)}" if versions else None


def collect_source_rows(
    fetcher: Fetcher = _fetch_url,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Make four bounded primary requests and retain non-local dispositions."""

    access_rows: list[JsonDict] = []
    for index, spec in enumerate(ACCESS_SPECS, 1):
        progress(4, "request start", f"completed={index - 1}/4 {spec['access_id']}")
        try:
            receipt = dict(fetcher(str(spec["url"])))
        except Exception as exc:  # noqa: BLE001 - the access failure is evidence.
            receipt = {
                "ok": False,
                "status_code": None,
                "body": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
        body = str(receipt.get("body", ""))
        accessed = receipt.get("ok") is True
        version = observed_version(body)
        title_ok = str(spec["expected_title"]).lower() in body.lower() if accessed else None
        row = {
            "access_id": spec["access_id"],
            "kind": "primary_record",
            "url": spec["url"],
            "retrieval_date": RETRIEVAL_DATE,
            "timeout_s": 20,
            "status_code": receipt.get("status_code"),
            "error": receipt.get("error"),
            "response_body": body,
            "response_sha256": sha256_bytes(body.encode()),
            "refresh_version": spec["refresh_version"],
            "observed_version": version,
            "title_verified": title_ok,
            "access_status": "http_success" if accessed else "access_failed_refresh_retained",
            "source_delta": (
                "no_version_delta"
                if accessed and version == spec["refresh_version"] and title_ok
                else "version_or_identity_delta"
                if accessed
                else "access_failed"
            ),
        }
        access_rows.append(row)
        progress(4, "request end", f"completed={index}/4 {row['access_status']}")

    access_by_id = {row["access_id"]: row for row in access_rows}
    source_rows: list[JsonDict] = []
    for disposition in SOURCE_DISPOSITIONS:
        row = dict(disposition)
        access = access_by_id[str(row.pop("access_id"))]
        row.update(
            primary_url=access["url"],
            refresh_date=RETRIEVAL_DATE,
            refresh_version=access["refresh_version"],
            observed_version=access["observed_version"],
            access_status=access["access_status"],
            source_delta=access["source_delta"],
            access_receipt_id=access["access_id"],
            local_evidence_claimed=False,
        )
        source_rows.append(row)
    return source_rows, access_rows


def build_sidecars(
    root: Path,
    raw_dir: Path,
    gate_rows: Sequence[Mapping[str, Any]],
    access_rows: Sequence[Mapping[str, Any]],
) -> dict[str, JsonDict]:
    """Keep historical models, controls, and response bytes in hashed files."""

    history_data = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    if not isinstance(history_data, Mapping):
        raise ValueError(f"historical artifact is not a mapping: {HISTORY_PATH}")
    history = {
        "schema": "carnot.exp7260.historical_model_receipts.v1",
        "run_date": RUN_DATE,
        "current_invocation": {
            "MODEL_SPECS": [],
            "model_invoked": False,
            "invocation_counts": dict(ZERO_INVOCATION_COUNTS),
        },
        "historical_artifacts": [
            {
                "path": str(HISTORY_PATH),
                "sha256": sha256(root / HISTORY_PATH),
                "status": history_data.get("status"),
                "honest_verdict": history_data.get("honest_verdict"),
                "verdict_class": history_data.get("verdict_class"),
                "MODEL_SPECS": history_data.get("MODEL_SPECS"),
                "model_invoked": history_data.get("model_invoked"),
                "invocation_counts": history_data.get("current_invocation_counters"),
                "quarantined": _is_quarantined(dict(history_data)),
                "accepted_for_current_values": False,
            }
        ],
    }
    negative = {
        "schema": "carnot.exp7260.negative_gate_fixtures.v1",
        "run_date": RUN_DATE,
        "fixture_rows": [dict(row) for row in gate_rows if row.get("case") != "true"],
    }
    source_access = {
        "schema": "carnot.exp7260.source_access_receipts.v1",
        "run_date": RUN_DATE,
        "request_count": len(access_rows),
        "receipts": [dict(row) for row in access_rows],
    }
    payloads = {
        "history": ("historical-model-receipts.json", history),
        "negative_fixtures": ("negative-gate-fixtures.json", negative),
        "source_access": ("source-access-receipts.json", source_access),
    }
    receipts: dict[str, JsonDict] = {}
    for name, (filename, payload) in payloads.items():
        path = raw_dir / filename
        _atomic_write_json(path, payload)
        receipts[name] = {"path": str(path.relative_to(root)), "sha256": sha256(path)}
    return receipts


def _summary(
    failed_check: str | None,
    expected: Any,
    observed: Any,
    *,
    upstream: Any = None,
    field: Any = None,
    passed: bool = False,
) -> JsonDict:
    """Use one stable diagnostic shape for all terminal outcomes."""

    return {
        "failed_check": failed_check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _hash_sources(root: Path, authority: Path | None) -> dict[str, str | None]:
    """Authenticate exact local inputs, code, history, and selected authority."""

    paths = (*INPUT_PATHS, *CODE_PATHS, HISTORY_PATH, *((authority,) if authority else ()))
    hashes: dict[str, str | None] = {}
    for relative in dict.fromkeys(paths):
        try:
            hashes[str(relative)] = sha256(root / relative)
        except OSError:
            hashes[str(relative)] = None
    return hashes


def _preconditions(
    root: Path,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Observe all listed inputs, terminal history, ownership, and outputs."""

    authority, roadmap, yaml_bytes, candidates = select_yaml_authority(root)
    rows = [dict(row) for row in candidates]
    rows.append(
        {
            "check": "yaml_authority",
            "upstream": f"{ACTIVE_ROADMAP_PATH}|{NEXT_ROADMAP_PATH}",
            "field": "milestone",
            "expected_value": MILESTONE,
            "observed_value": str(authority) if authority else None,
            "available": authority is not None,
            "blocking_external": True,
        }
    )
    for relative in (*INPUT_PATHS, *CODE_PATHS):
        path = root / relative
        available = path.is_file() and path.stat().st_size > 0
        rows.append(
            {
                "check": f"source_bytes:{relative}",
                "upstream": str(relative),
                "field": "bytes|sha256",
                "expected_value": "readable_nonempty_authenticated_bytes",
                "observed_value": {
                    "bytes": path.stat().st_size if available else None,
                    "sha256": sha256(path) if available else None,
                },
                "available": available,
                "blocking_external": True,
            }
        )
    try:
        history = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
        quarantined = _is_quarantined(dict(history)) if isinstance(history, Mapping) else True
        history_ok = (
            isinstance(history, Mapping)
            and history.get("status") in {"complete", "blocked"}
            and not quarantined
        )
        history_observed: Any = {
            "status": history.get("status") if isinstance(history, Mapping) else None,
            "quarantined": quarantined,
            "sha256": sha256(root / HISTORY_PATH),
        }
    except (OSError, json.JSONDecodeError) as exc:
        history_ok = False
        history_observed = f"{type(exc).__name__}: {exc}"
    rows.append(
        {
            "check": "v638_terminal_capstone",
            "upstream": str(HISTORY_PATH),
            "field": "status|quarantine",
            "expected_value": "terminal and not quarantined",
            "observed_value": history_observed,
            "available": history_ok,
            "blocking_external": True,
        }
    )
    try:
        spec_ok = "REQ-REPORT-7260" in (root / SPEC_PATH).read_text(encoding="utf-8")
    except OSError:
        spec_ok = False
    rows.append(
        {
            "check": "driving_requirement",
            "upstream": str(SPEC_PATH),
            "field": "REQ-*",
            "expected_value": "REQ-REPORT-7260",
            "observed_value": "REQ-REPORT-7260" if spec_ok else "missing",
            "available": spec_ok,
            "blocking_external": True,
        }
    )
    design_path = roadmap.get("milestone_doc") if roadmap else None
    rows.append(
        {
            "check": "design_authority",
            "upstream": str(authority) if authority else None,
            "field": "milestone_doc",
            "expected_value": str(DESIGN_PATH),
            "observed_value": design_path,
            "available": design_path == str(DESIGN_PATH) and (root / DESIGN_PATH).is_file(),
            "blocking_external": True,
        }
    )
    if roadmap is not None:
        own = next(
            (task for task in roadmap.get("tasks", []) if task.get("id") == FIRST_TASK_ID), None
        )
        own_gates = own.get("gated_on", []) if isinstance(own, Mapping) else None
        rows.append(
            {
                "check": "task_upstream_gates",
                "upstream": FIRST_TASK_ID,
                "field": "gated_on",
                "expected_value": [],
                "observed_value": own_gates,
                "available": own_gates in (None, []),
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
    return rows, authority, roadmap, yaml_bytes


def _freeze_authorities(
    root: Path,
    authority: Path,
    yaml_bytes: bytes,
    raw_dir: Path,
) -> list[JsonDict]:
    """Freeze exact Markdown and YAML bytes without normalization."""

    sources = (
        ("markdown", DESIGN_PATH, (root / DESIGN_PATH).read_bytes(), raw_dir / "v639-design.md"),
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


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain extra fields and retain every prompt-specified principle."""

    values = {field: "This field keeps one V639 contract fact auditable." for field in fields}
    values.update(REQUIRED_FIELD_PRINCIPLES)
    return values


def _base_artifact(run_date: str, checkpoint_path: Path, started_at: str) -> JsonDict:
    """Create a running checkpoint that cannot be mistaken for success."""

    artifact: JsonDict = {
        "schema": "carnot.exp7260.v639_source_contract.v1",
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
            "gate_controls": {
                "planned": 28,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": 28,
            },
            "source_dispositions": {
                "planned": len(SOURCE_DISPOSITIONS),
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": len(SOURCE_DISPOSITIONS),
            },
            "network_requests": {
                "planned": 4,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "independent_units": 4,
            },
            "stopping_rule": "Stop after fourteen contract rows, four controls for each of seven gates, five bounded dispositions, and four primary requests.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _summary(
            "preconditions_not_checked", "all required inputs", "not_checked"
        ),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_running_v639_source_contract",
        "verdict_class": "partial",
        "validation_receipts": [],
        "source_contract_complete_score": 0,
        "contract_rows": [],
        "source_rows": [],
        "source_access_rows": [],
        "gate_replay_rows": [],
        "raw_authority_rows": [],
        "sidecar_receipts": {},
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
        "measured_candidate_path": None,
        "activation_files_modified": False,
        "research_roadmap_modified": False,
        "scientific_value_claimed": False,
        "publication_performed": False,
        "submission_performed": False,
        "upload_performed": False,
        "production_default_changed": False,
        "external_message_performed": False,
        "baseline_failures": [],
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    return artifact


def _failed_precondition(artifact: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Return the first unavailable external prerequisite, when present."""

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


def _source_rows_complete(rows: object) -> bool:
    """Require all planned dispositions and their falsification boundaries."""

    return (
        isinstance(rows, list)
        and len(rows) == len(SOURCE_DISPOSITIONS)
        and {row.get("method_family") for row in rows}
        == {
            "warm_start_recognition",
            "interactive_acquisition",
            "span_grounding",
            "cost_accounting",
        }
        and {row.get("disposition") for row in rows} == {"implement", "adapt", "defer"}
        and all(
            row.get("input_needs")
            and row.get("falsifying_controls")
            and row.get("retirement_overlap")
            and row.get("target_tasks")
            and row.get("access_status")
            and row.get("local_evidence_claimed") is False
            for row in rows
        )
    )


def _validation_complete(artifact: Mapping[str, Any]) -> bool:
    """Require every exact scoped command and a zero exit."""

    rows = artifact.get("validation_receipts")
    return (
        isinstance(rows, list)
        and [row.get("name") for row in rows] == list(VALIDATION_COMMAND_NAMES)
        and all(row.get("passed") is True for row in rows)
    )


def _sidecars_complete(artifact: Mapping[str, Any]) -> bool:
    """Require named sidecars and matching authenticated hashes."""

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


def _acceptance_gate_results(artifact: Mapping[str, Any]) -> list[JsonDict]:
    """Recompute the frozen structural criteria from stored evidence."""

    contract = artifact.get("contract_rows")
    sources = artifact.get("source_rows")
    access = artifact.get("source_access_rows")
    raw = artifact.get("raw_authority_rows")
    hashes = artifact.get("source_artifact_hashes")
    criteria = (
        (
            "contract_exact",
            "14 matching exp7260-through-exp7273 rows",
            [row.get("unit_id") for row in contract] if isinstance(contract, list) else None,
            isinstance(contract, list)
            and len(contract) == 14
            and [row.get("unit_id") for row in contract] == list(EXPECTED_ID_ORDER)
            and all(row.get("passed") is True for row in contract)
            and artifact.get("markdown_milestone") == artifact.get("yaml_milestone") == MILESTONE,
            "Both independently parsed authorities must agree exactly.",
        ),
        (
            "gate_controls",
            28,
            len(artifact.get("gate_replay_rows", []))
            if isinstance(artifact.get("gate_replay_rows"), list)
            else None,
            gate_replays_complete(artifact.get("gate_replay_rows")),
            "Every real gate edge must distinguish false from missing evidence.",
        ),
        (
            "source_dispositions",
            len(SOURCE_DISPOSITIONS),
            len(sources) if isinstance(sources, list) else None,
            _source_rows_complete(sources),
            "Each source adoption remains bounded, falsifiable, and non-local.",
        ),
        (
            "request_budget",
            "four primary records maximum",
            len(access) if isinstance(access, list) else None,
            isinstance(access, list)
            and len(access) == 4
            and all(row.get("kind") == "primary_record" for row in access),
            "The recheck cannot expand beyond four primary URLs.",
        ),
        (
            "raw_authorities",
            2,
            len(raw) if isinstance(raw, list) else None,
            isinstance(raw, list)
            and len(raw) == 2
            and all(row.get("hash_matches") is True for row in raw),
            "Exact Markdown and YAML bytes must survive the raw freeze.",
        ),
        (
            "sidecars",
            True,
            _sidecars_complete(artifact),
            _sidecars_complete(artifact),
            "Historical models, source bodies, and controls stay outside current counters.",
        ),
        (
            "source_hashes",
            "all present",
            "all present"
            if isinstance(hashes, Mapping) and hashes and all(hashes.values())
            else "missing",
            isinstance(hashes, Mapping) and bool(hashes) and all(hashes.values()),
            "Every consumed local file and evidence sidecar has an exact digest.",
        ),
        (
            "validation",
            list(VALIDATION_COMMAND_NAMES),
            [row.get("name") for row in artifact.get("validation_receipts", [])]
            if isinstance(artifact.get("validation_receipts"), list)
            else None,
            _validation_complete(artifact),
            "All scoped checks must retain exact commands and successful exits.",
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
    """Return one only when every structural acceptance row passes."""

    rows = _acceptance_gate_results(artifact)
    return int(len(rows) == 8 and all(row["passed"] for row in rows))


def _failure_summary(artifact: Mapping[str, Any]) -> JsonDict:
    """Name the first external, authority, row, or validation failure."""

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
        row = next((value for value in contract if value.get("passed") is not True), None)
        if row is not None:
            return _summary(
                "contract_parity",
                row.get("yaml"),
                row.get("markdown"),
                upstream=row.get("unit_id"),
                field="id|order|title|deliverable|gates",
            )
    acceptance = _acceptance_gate_results(artifact)
    row = next((value for value in acceptance if value["passed"] is not True), None)
    return _summary(
        f"acceptance:{row['criterion']}" if row else "stored_contract",
        row.get("expected_value") if row else "complete",
        row.get("observed_value") if row else "incomplete",
        field=row.get("criterion") if row else "acceptance_gate_results",
    )


def _apply_terminal_state(artifact: JsonDict) -> None:
    """Derive terminal lifecycle and verdict solely from stored evidence."""

    artifact["acceptance_gate_results"] = _acceptance_gate_results(artifact)
    artifact["source_contract_complete_score"] = _score_from_artifact(artifact)
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
    elif artifact["source_contract_complete_score"] == 1:
        artifact.update(
            status="complete",
            inference_substrate=INFERENCE_SUBSTRATE,
            inference_substrate_class=INFERENCE_SUBSTRATE_CLASS,
            verdict_class="circular_positive",
            honest_verdict=POSITIVE_VERDICT,
            gate_check_summary=_summary(
                None,
                "all V639 source-contract criteria pass",
                "all V639 source-contract criteria pass",
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


def _set_time_and_checksum(artifact: JsonDict, started: float) -> None:
    """Measure real elapsed work and bind all stable evidence fields."""

    artifact["duration_s"] = max(round(time.monotonic() - started, 6), 0.000001)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def _checkpoint(path: Path, artifact: JsonDict, started: float) -> None:
    """Write mutable work only to the declared checkpoint path."""

    _set_time_and_checksum(artifact, started)
    _atomic_write_json(path, artifact)


def _record_phase(artifact: JsonDict, phase: int, started: float, detail: str) -> None:
    """Append one monotonic phase span relative to the run origin."""

    ended = time.monotonic()
    origin = artifact.setdefault("monotonic_origin", started)
    artifact["phase_spans"].append(
        {
            "phase": phase,
            "detail": detail,
            "start_elapsed_s": round(started - origin, 6),
            "end_elapsed_s": round(ended - origin, 6),
            "duration_s": round(ended - started, 6),
        }
    )


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Reduce literal rows without trusting aggregate scores or verdicts."""

    errors: list[str] = []
    contract = artifact.get("contract_rows")
    if not isinstance(contract, list) or len(contract) != 14:
        errors.append("contract_row_count")
    elif any(
        row.get("passed") != all(row.get("checks", {}).values()) or row.get("censored") is not False
        for row in contract
    ):
        errors.append("contract_row_reduction")
    if not gate_replays_complete(artifact.get("gate_replay_rows")):
        errors.append("gate_replay_reduction")
    sources = artifact.get("source_rows")
    if not isinstance(sources, list) or len(sources) != len(SOURCE_DISPOSITIONS):
        errors.append("source_row_count")
    elif not _source_rows_complete(sources):
        errors.append("source_row_reduction")
    raw = artifact.get("raw_authority_rows")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(row.get("source_sha256") != row.get("raw_sha256") for row in raw)
    ):
        errors.append("raw_authority_reduction")
    return errors


def validate_artifact(artifact: object, *, root: Path = REPO_ROOT) -> list[str]:
    """Recompute fields, hashes, rows, terminal state, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if artifact.get("schema") != "carnot.exp7260.v639_source_contract.v1":
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
    if artifact.get("random_seed") != RANDOM_SEED:
        errors.append("random_seed_invalid")
    if artifact.get("verifier_is_oracle") is not True:
        errors.append("verifier_authority_invalid")
    if artifact.get("rows") != artifact.get("contract_rows"):
        errors.append("rows_invalid")
    budget = artifact.get("sample_size_budget")
    budget_groups = (
        "contract_rows",
        "gate_controls",
        "source_dispositions",
        "network_requests",
    )
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
    expected = dict(artifact)
    _apply_terminal_state(expected)
    for field in (
        "status",
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
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return errors


def validation_commands(
    root: Path,
    candidate_path: Path,
    authority: Path,
) -> tuple[tuple[str, list[str]], ...]:
    """Return only the shipped and explicitly scoped validation commands."""

    python = str(root / ".venv/bin/python")
    roadmap = str(root / authority)
    candidate = str(candidate_path)
    schema_code = (
        "import pathlib,sys,yaml;"
        f"sys.path.insert(0,{str(root / 'scripts')!r});"
        "from roadmap_schema import Roadmap;"
        f"Roadmap.model_validate(yaml.safe_load(pathlib.Path({roadmap!r}).read_text()))"
    )
    e2e_code = (
        "import pathlib,sys,tempfile,yaml;"
        "from carnot.experiment_7260_v639_source_contract import "
        "run_gate_replays,gate_replays_complete;"
        f"roadmap=yaml.safe_load(pathlib.Path({roadmap!r}).read_text());"
        "directory=tempfile.TemporaryDirectory(prefix='exp7260-e2e-',dir='/tmp');"
        "rows=run_gate_replays(roadmap,pathlib.Path(directory.name));"
        "print({'rows':len(rows),'complete':gate_replays_complete(rows)});"
        "sys.exit(not gate_replays_complete(rows))"
    )
    reducer_code = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7260_v639_source_contract import independent_reduce;"
        f"errors=independent_reduce(json.loads(pathlib.Path({candidate!r}).read_text()));"
        "print(errors);sys.exit(bool(errors))"
    )
    coverage_file = "/tmp/.coverage-exp7260-v639"
    pytest_base = "/tmp/exp7260-v639-focused-pytest"
    return (
        ("roadmap_schema", [python, "-u", "-c", schema_code]),
        ("prior_failure", [python, "-u", str(root / PRIOR_LINT_PATH), roadmap]),
        ("exclusion_manifest", [python, "-u", str(root / EXCLUSION_LINT_PATH), roadmap]),
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
        ("mypy", [str(root / ".venv/bin/mypy"), str(root / MODULE_PATH)]),
        (
            "scoped_spec_coverage",
            [python, "-u", str(root / SPEC_COVERAGE_PATH), str(root / TEST_PATH)],
        ),
        ("e2e_gate_controls", [python, "-u", "-c", e2e_code]),
        ("independent_reducer", [python, "-u", "-c", reducer_code]),
        ("adversarial", [python, "-u", str(root / ADVERSARIAL_PATH), candidate]),
        ("row_consistency", [python, "-u", str(root / ROW_LINT_PATH), candidate]),
    )


def _execute_commands(
    root: Path,
    log_dir: Path,
    commands: Sequence[tuple[str, list[str]]],
    *,
    phase: int = 6,
) -> list[JsonDict]:
    """Stream bounded subprocesses and preserve their exact outputs and hashes."""

    rows: list[JsonDict] = []
    for index, (name, command) in enumerate(commands, 1):
        progress(phase, "subprocess start", f"completed={index - 1}/{len(commands)} {name}")
        receipt = shipped._run_streaming_command(
            command,
            cwd=root,
            timeout_s=1200,
            heartbeat_s=60,
            operation=f"exp7260:{name}",
        )
        output = str(receipt.get("stdout", ""))
        log_path = log_dir / f"{name}.log"
        _atomic_write_bytes(log_path, output.encode())
        row = {
            "name": name,
            "command": shlex.join(command),
            "exit_code": int(receipt["exit_code"]),
            "duration_s": receipt["duration_s"],
            "timed_out": receipt["timed_out"],
            "log_path": str(log_path.relative_to(root)),
            "log_sha256": sha256(log_path),
            "stdout_tail": output[-4000:],
            "passed": receipt["exit_code"] == 0,
            "baseline_failure": False,
        }
        rows.append(row)
        progress(
            phase,
            "subprocess end",
            f"completed={index}/{len(commands)} {name} exit={row['exit_code']}",
        )
    return rows


def run_required_validation(
    root: Path,
    raw_dir: Path,
    artifact: JsonDict,
    started: float,
) -> list[JsonDict]:
    """Validate a measured terminal candidate before final publication."""

    authority = Path(str(artifact["yaml_authority_path"]))
    candidate_path = raw_dir / RAW_CANDIDATE_PATH.name
    commands = validation_commands(root, candidate_path, authority)
    log_dir = raw_dir / "validation"
    preterminal = _execute_commands(root, log_dir, commands[:PRETERMINAL_VALIDATION_COUNT])
    artifact["validation_receipts"] = list(preterminal)
    artifact["baseline_failures"] = [row for row in preterminal if row["passed"] is not True]
    _apply_terminal_state(artifact)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["measured_candidate_path"] = str(candidate_path.relative_to(root))
    _set_time_and_checksum(artifact, started)
    _atomic_write_json(candidate_path, artifact)
    progress(6, "candidate written", f"terminal candidate={candidate_path.relative_to(root)}")
    postterminal = _execute_commands(root, log_dir, commands[PRETERMINAL_VALIDATION_COUNT:])
    return [*preterminal, *postterminal]


def build_artifact(
    root: Path,
    run_date: str,
    *,
    output_path: Path,
    raw_dir: Path,
    checkpoint_path: Path,
    fetcher: Fetcher = _fetch_url,
) -> JsonDict:
    """Measure and publish one exact, disqualified, or externally blocked receipt."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    artifact = _base_artifact(run_date, checkpoint_path, started_at)
    progress(0, "start", "authenticate inputs, ownership, imports, and output paths")
    phase_started = time.monotonic()
    _checkpoint(checkpoint_path, artifact, started)
    preconditions, authority, roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["yaml_authority_path"] = str(authority) if authority else None
    artifact["source_artifact_hashes"] = _hash_sources(root, authority)
    _record_phase(artifact, 0, phase_started, "preconditions")
    _checkpoint(checkpoint_path, artifact, started)
    failed = _failed_precondition(artifact)
    progress(0, "end", f"completed={len(preconditions)} blocked={failed is not None}")

    if failed is None:
        assert authority is not None and roadmap is not None and yaml_bytes is not None

        phase_started = time.monotonic()
        progress(1, "start", "bind no-model declaration and zero invocation counts")
        artifact.update(
            MODEL_SPECS=[],
            model_invoked=False,
            invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        )
        _record_phase(artifact, 1, phase_started, "no-model declaration")
        _checkpoint(checkpoint_path, artifact, started)
        progress(1, "end", "completed=1 current model/load/generation counts=0")

        phase_started = time.monotonic()
        progress(2, "start", "freeze and independently parse Markdown and YAML")
        artifact["raw_authority_rows"] = _freeze_authorities(root, authority, yaml_bytes, raw_dir)
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
        for row in artifact["raw_authority_rows"]:
            artifact["source_artifact_hashes"][row["raw_path"]] = row["raw_sha256"]
        _record_phase(artifact, 2, phase_started, "authority freeze and contract comparison")
        _checkpoint(checkpoint_path, artifact, started)
        progress(2, "end", f"completed=14 contract_match={contract['passed']}")

        phase_started = time.monotonic()
        progress(3, "start", "run schema and four real controls for seven gate edges")
        Roadmap.model_validate(roadmap)
        with tempfile.TemporaryDirectory(prefix="exp7260-gates-", dir="/tmp") as directory:
            artifact["gate_replay_rows"] = run_gate_replays(roadmap, Path(directory))
        gate_budget = artifact["sample_size_budget"]["gate_controls"]
        gate_budget.update(
            attempted=len(artifact["gate_replay_rows"]),
            completed=len(artifact["gate_replay_rows"]),
        )
        _record_phase(artifact, 3, phase_started, "schema and real gate controls")
        _checkpoint(checkpoint_path, artifact, started)
        progress(3, "end", f"completed={len(artifact['gate_replay_rows'])}")

        phase_started = time.monotonic()
        progress(4, "start", "ingest five dispositions with four bounded primary requests")
        sources, access = collect_source_rows(fetcher)
        artifact["source_rows"] = sources
        artifact["source_access_rows"] = [
            {key: value for key, value in row.items() if key != "response_body"} for row in access
        ]
        source_budget = artifact["sample_size_budget"]["source_dispositions"]
        source_budget.update(attempted=len(sources), completed=len(sources))
        request_budget = artifact["sample_size_budget"]["network_requests"]
        request_budget.update(
            attempted=len(access),
            completed=len(access),
            censored=sum(row["access_status"].startswith("access_failed") for row in access),
        )
        _record_phase(artifact, 4, phase_started, "bounded source ingestion")
        _checkpoint(checkpoint_path, artifact, started)
        progress(4, "end", f"completed_dispositions={len(sources)} requests={len(access)}")

        phase_started = time.monotonic()
        progress(5, "start", "write hashed history, source, and control sidecars")
        sidecars = build_sidecars(root, raw_dir, artifact["gate_replay_rows"], access)
        artifact["sidecar_receipts"] = sidecars
        for receipt in sidecars.values():
            artifact["source_artifact_hashes"][receipt["path"]] = receipt["sha256"]
        _record_phase(artifact, 5, phase_started, "evidence sidecars")
        _checkpoint(checkpoint_path, artifact, started)
        progress(5, "end", f"completed={len(sidecars)}")

        phase_started = time.monotonic()
        progress(6, "start", "run scoped checks against a measured raw candidate")
        artifact["validation_receipts"] = run_required_validation(root, raw_dir, artifact, started)
        artifact["baseline_failures"] = [
            row for row in artifact["validation_receipts"] if row["passed"] is not True
        ]
        _record_phase(artifact, 6, phase_started, "scoped validation")
        _checkpoint(checkpoint_path, artifact, started)
        progress(6, "end", f"completed={len(artifact['validation_receipts'])}")
    else:
        for phase in range(1, 7):
            phase_started = time.monotonic()
            progress(phase, "start", "skip after external prerequisite block")
            _record_phase(artifact, phase, phase_started, "skipped after external block")
            _checkpoint(checkpoint_path, artifact, started)
            progress(phase, "end", "completed=0 blocked=1")

    phase_started = time.monotonic()
    progress(7, "start", "derive, validate, and atomically publish terminal evidence")
    _apply_terminal_state(artifact)
    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    _record_phase(artifact, 7, phase_started, "terminal derivation")
    artifact.pop("monotonic_origin", None)
    _set_time_and_checksum(artifact, started)
    progress(7, "validation start", "recompute stored terminal artifact")
    errors = validate_artifact(artifact, root=root)
    progress(7, "validation end", f"errors={errors}")
    if errors:  # pragma: no cover - the CLI fails closed on internal defects.
        raise ValueError(f"invalid Exp7260 artifact: {errors}")
    _set_time_and_checksum(artifact, started)
    _atomic_write_json(checkpoint_path, artifact)
    _atomic_write_json(output_path, artifact)
    progress(7, "end", f"published={output_path} verdict={artifact['verdict_class']}")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the fixed V639 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI integration.
    """Run the source contract and accept every internally valid terminal class."""

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
        print(f"[exp7260] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7260] complete verdict={artifact['honest_verdict']} "
        f"score={artifact['source_contract_complete_score']}",
        flush=True,
    )
    return 0

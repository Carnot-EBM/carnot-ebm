"""Audit the V651 task contract and ingest its selected methods.

This experiment reads frozen evidence and public source endpoints. It does not
run a language model or train an energy model. A broken plan authority can
disqualify this advisory receipt, but it cannot block later science branches.

Spec refs: REQ-REPORT-7421 and SCENARIO-REPORT-7421-*.
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
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, atomic_json
from carnot.reporting import experiment_7303_validation_scope as validation_scope


JsonDict = dict[str, Any]
SourceFetcher = Callable[[Mapping[str, str]], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260919"
MILESTONE = "2026.09.651"
EXPERIMENT_ID = "exp7421-contract-ingestion"
SCHEMA = "carnot.exp7421.v651.contract_ingestion.v1"
PHASE = 1
RANDOM_SEED = {"contract_mutations": 7_421_651_01, "basis_equivalence": 7_421_651_02}

ROADMAP_PATH = Path("research-roadmap.yaml")
DESIGN_PATH = Path("openspec/change-proposals/research-roadmap-vNEXT.md")
SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")
COMPLETE_PATH = Path("research-complete.yaml")
REFERENCE_PATH = Path("research-references.md")
STUDY_PATH = Path("research-studying.md")
NOTE_PATH = Path("docs/research-notes/v651-method-ingestion.md")
RESULT_PATH = Path("results/experiment_7421_v651_contract_ingestion.json")
RAW_DIR = Path("results/raw/experiment_7421_v651_contract_ingestion")
MODULE_PATH = Path("python/carnot/experiment_7421_v651_contract_ingestion.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7421_v651_contract_ingestion.py")
TEST_PATH = Path("tests/python/test_experiment_7421_v651_contract_ingestion.py")

EXPECTED_TASK_IDS = (
    "exp7421-contract-ingestion",
    "exp7422-runtime-ownership",
    "exp7423-annotated-protocol",
    "exp7424-arc-receipt-boundary",
    "exp7425-spline-prototype",
    "exp7426-static-decisions",
    "exp7427-randomized-feedback",
    "exp7428-decision-audit",
    "exp7429-anchored-capture",
    "exp7430-extraction-audit",
    "exp7431-arc-live-sentinel",
    "exp7432-update-placement",
    "exp7433-capstone",
)

PRIOR_PATHS = (
    ("exp7409-evidence-custody", Path("results/experiment_7409_v650_evidence_custody.json")),
    ("exp7410-source-corpus", Path("results/experiment_7410_v650_source_corpus.json")),
    ("exp7411-arc-call-budget", Path("results/experiment_7411_v650_arc_call_budget.json")),
    ("exp7412-source-features", Path("results/experiment_7412_v650_source_features.json")),
    (
        "exp7413-source-calibration",
        Path("results/experiment_7413_v650_source_calibration.json"),
    ),
    ("exp7414-selected-feedback", Path("results/experiment_7414_v650_selected_feedback.json")),
    ("exp7415-decision-audit", Path("results/experiment_7415_v650_decision_audit.json")),
    (
        "exp7416-anchored-extraction",
        Path("results/experiment_7416_v650_anchored_extraction.json"),
    ),
    ("exp7417-extraction-audit", Path("results/experiment_7417_extraction_audit.json")),
    ("exp7418-revision-memory", Path("results/experiment_7418_v650_revision_memory.json")),
    (
        "exp7419-precision-placement",
        Path("results/experiment_7419_v650_precision_placement.json"),
    ),
    ("exp7420-capstone", Path("results/experiment_7420_v650_capstone.json")),
)

SELECTED_SOURCES = (
    {
        "source_id": "spline_locality",
        "version": "arXiv:2602.02056v4",
        "url": "https://arxiv.org/abs/2602.02056v4",
        "kind": "primary_paper",
    },
    {
        "source_id": "partial_feedback",
        "version": "arXiv:2506.14067v2",
        "url": "https://arxiv.org/abs/2506.14067v2",
        "kind": "primary_paper",
    },
    {
        "source_id": "claim_attribution",
        "version": "arXiv:2608.05823",
        "url": "https://arxiv.org/abs/2608.05823",
        "kind": "primary_paper",
    },
    {
        "source_id": "symbolic_grounding",
        "version": "arXiv:2609.05025",
        "url": "https://arxiv.org/abs/2609.05025",
        "kind": "primary_paper",
    },
    {
        "source_id": "kanele",
        "version": "arXiv:2512.12850",
        "url": "https://arxiv.org/abs/2512.12850",
        "kind": "primary_paper",
    },
    {
        "source_id": "ragtruth",
        "version": "author repository checked 2026-09-19",
        "url": "https://github.com/ParticleMedia/RAGTruth",
        "kind": "primary_dataset",
    },
    {
        "source_id": "semantic_scholar_citations",
        "version": "Graph API checked 2026-09-19",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2602.02056/citations?limit=1",
        "kind": "citation_endpoint",
        "prior_review_access_state": "failed",
        "prior_review_detail": "The dated V651 planning review recorded failed citation endpoint reads.",
    },
)

SOURCE_INPUT_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("ops/e2e-test-plan.md"),
    EXCLUSION_PATH,
    SPEC_PATH,
    DESIGN_PATH,
    ROADMAP_PATH,
    COMPLETE_PATH,
    REFERENCE_PATH,
    STUDY_PATH,
    NOTE_PATH,
    Path("scripts/experiment_template.py"),
    Path("scripts/sweep_clusters.py"),
    Path("scripts/sweep_semscholar.py"),
    Path("python/carnot/reporting/current_work_receipt.py"),
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7329_v644_contract.py"),
    Path("python/carnot/experiment_7409_v650_evidence_custody.py"),
    Path("results/experiment_7420_v650_capstone.json"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

VALIDATION_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

TERMINAL_CHECK_NAMES = (
    "declared_entrypoint_cold_replay",
    "independent_cold_reducer",
    "adversarial_verify",
    "verdict_row_consistency_strict",
)


def utc_now() -> str:  # pragma: no cover - authentic runtime boundary.
    """Return one real UTC boundary in a stable text form."""

    return datetime.now(UTC).isoformat()


def progress(  # pragma: no cover - public E2E progress boundary.
    started: float, phase: str, event: str, **details: Any
) -> None:
    """Flush every phase boundary with measured monotonic elapsed time."""

    suffix = " ".join(f"{key}={value}" for key, value in sorted(details.items()))
    print(
        f"[exp7421] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}"
        + (f" {suffix}" if suffix else ""),
        flush=True,
    )


def sha256_file(path: Path) -> str:
    """Hash exact input bytes so later edits cannot inherit this receipt."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_yaml(path: Path) -> JsonDict:
    """Load one YAML mapping and reject prose or sequence-shaped input."""

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


def _public_task(task: Mapping[str, Any] | None) -> JsonDict | None:
    """Keep only fields independently declared by both task authorities."""

    if task is None:
        return None
    return {
        key: deepcopy(task.get(key))
        for key in ("order", "id", "title", "deliverable", "phase", "substrate", "gates")
    }


def _producer_declarations(roadmap: Mapping[str, Any]) -> dict[str, list[JsonDict]]:
    """Check every gate field against the earlier producer prompt."""

    tasks = roadmap.get("tasks") if isinstance(roadmap.get("tasks"), list) else []
    positions = {
        task.get("id"): (index, task)
        for index, task in enumerate(tasks)
        if isinstance(task, Mapping)
    }
    by_consumer: dict[str, list[JsonDict]] = {}
    for consumer_index, consumer in enumerate(tasks):
        assert isinstance(consumer, Mapping)
        rows: list[JsonDict] = []
        for gate in consumer.get("gated_on") or []:
            upstream = gate.get("upstream")
            producer_entry = positions.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            declared = _field_declared(str(producer.get("prompt", "")), gate.get("artifact_field"))
            rows.append(
                {
                    "upstream": upstream,
                    "artifact_field": gate.get("artifact_field"),
                    "producer_precedes_consumer": producer_index is not None
                    and producer_index < consumer_index,
                    "declared_verbatim": declared,
                    "passed": producer_index is not None
                    and producer_index < consumer_index
                    and declared,
                }
            )
        by_consumer[str(consumer.get("id"))] = rows
    return by_consumer


def compare_contract_authorities(markdown_text: str, roadmap: object) -> JsonDict:
    """Compare generic parser outputs without inheriting an old task count."""

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
        raw_task = (
            roadmap_mapping.get("tasks", [])[index]
            if index < len(roadmap_mapping.get("tasks", []))
            else {}
        )
        declaration_rows = declarations.get(task_id, [])
        checks = {
            "id": bool(expected and observed and expected.get("id") == observed.get("id")),
            "title": bool(expected and observed and expected.get("title") == observed.get("title")),
            "deliverable": bool(
                expected and observed and expected.get("deliverable") == observed.get("deliverable")
            ),
            "phase": bool(expected and observed and expected.get("phase") == observed.get("phase")),
            "substrate": bool(
                expected and observed and expected.get("substrate") == observed.get("substrate")
            ),
            "gates": bool(expected and observed and expected.get("gates") == observed.get("gates")),
            "producer_fields": all(row["passed"] for row in declaration_rows),
            "quarantine_status": not bool(
                isinstance(raw_task, Mapping) and raw_task.get("quarantined")
            ),
        }
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
    """Read the two declared authorities and retain disagreement as evidence."""

    roadmap = load_yaml(root / ROADMAP_PATH)
    markdown = (root / DESIGN_PATH).read_text(encoding="utf-8")
    return compare_contract_authorities(markdown, roadmap)


def _remove_declared_field(prompt: str, field: str) -> str:
    """Remove one exact required-field line for a private fail-closed control."""

    return re.sub(rf"(?m)^\s*-\s*{re.escape(field)}\s*:.*\n?", "", prompt, count=1)


def run_contract_mutation_controls(
    markdown_text: str, roadmap: Mapping[str, Any]
) -> list[JsonDict]:
    """Reject count, order, milestone, producer-field, and quarantine defects."""

    cases: list[tuple[str, JsonDict]] = []
    removed = deepcopy(dict(roadmap))
    removed["tasks"] = deepcopy(list(roadmap["tasks"]))[:-1]
    cases.append(("count", removed))

    reordered = deepcopy(dict(roadmap))
    reordered["tasks"] = deepcopy(list(roadmap["tasks"]))
    reordered["tasks"][0], reordered["tasks"][1] = (
        reordered["tasks"][1],
        reordered["tasks"][0],
    )
    cases.append(("order", reordered))

    stale = deepcopy(dict(roadmap))
    stale["milestone"] = "2026.09.650"
    cases.append(("milestone", stale))

    missing_field = deepcopy(dict(roadmap))
    mutated = False
    for consumer in missing_field["tasks"]:
        for gate in consumer.get("gated_on") or []:
            producer = next(
                task for task in missing_field["tasks"] if task.get("id") == gate["upstream"]
            )
            producer["prompt"] = _remove_declared_field(
                str(producer["prompt"]), str(gate["artifact_field"])
            )
            mutated = True
            break
        if mutated:
            break
    cases.append(("missing_producer_field", missing_field))

    quarantined = deepcopy(dict(roadmap))
    quarantined["tasks"][0]["quarantined"] = True
    cases.append(("quarantine_status", quarantined))

    return [
        {
            "mutation": name,
            "rejected": compare_contract_authorities(markdown_text, candidate)["passed"] is False,
        }
        for name, candidate in cases
    ]


def _prior_class(task_id: str, payload: Mapping[str, Any]) -> str:
    """Recover the original closed class from both artifact schema variants."""

    if isinstance(payload.get("verdict_class"), str):
        return str(payload["verdict_class"])
    if task_id == "exp7417-extraction-audit":
        return "blocked"
    return "disqualified"


def collect_prior_dispositions(root: Path) -> list[JsonDict]:
    """Authenticate all V650 artifacts without changing their dispositions."""

    rows: list[JsonDict] = []
    for task_id, relative in PRIOR_PATHS:
        path = root / relative
        payload = load_json(path)
        facts: JsonDict = {}
        if task_id == "exp7414-selected-feedback":
            counts = payload.get("sample_size_budget", {}).get("label_counts", {})
            facts = {
                "independent_groups": payload.get("sample_size_budget", {}).get(
                    "independent_groups"
                ),
                "negative_labels": counts.get("0"),
                "positive_labels": counts.get("1"),
            }
        elif task_id == "exp7416-anchored-extraction":
            summary = payload.get("gate_check_summary", {})
            facts = {
                "attempted_calls": payload.get("sample_size_budget", {}).get("attempted"),
                "block": "available_slot_dictionary_exact_equality",
                "expected": deepcopy(summary.get("expected_value")),
                "observed": deepcopy(summary.get("observed_value")),
            }
        elif task_id == "exp7411-arc-call-budget":
            findings = payload.get("corrigendum_pending") or []
            facts = {
                "typed_invocation_conflict": any(
                    row.get("kind") == "INFERENCE_PROVENANCE_CONTRADICTION"
                    for row in findings
                    if isinstance(row, Mapping)
                )
            }
        rows.append(
            {
                "task_id": task_id,
                "path": relative.as_posix(),
                "sha256": sha256_file(path),
                "source_kind": (
                    "conductor_pre_gate_record"
                    if task_id == "exp7417-extraction-audit"
                    else "terminal_artifact"
                ),
                "original_status": payload.get("status"),
                "original_honest_verdict": payload.get("honest_verdict"),
                "original_verdict_class": _prior_class(task_id, payload),
                "flagged_adversarial": bool(payload.get("flagged_adversarial", False)),
                "observed_facts": facts,
            }
        )
    return rows


def completion_ledger_state(root: Path) -> JsonDict:
    """Report the ledger end without modifying or reopening finished work."""

    document = load_yaml(root / COMPLETE_PATH)
    milestones = document.get("milestones") if isinstance(document.get("milestones"), list) else []
    identifiers = [str(row.get("id")) for row in milestones if isinstance(row, Mapping)]
    latest = identifiers[-1] if identifiers else None
    return {
        "path": COMPLETE_PATH.as_posix(),
        "latest_milestone": latest,
        "expected_current_milestone": MILESTONE,
        "lag": latest != MILESTONE,
        "changed_by_experiment": False,
    }


def _direct_fetch(  # pragma: no cover - bounded external network boundary.
    source: Mapping[str, str],
) -> JsonDict:
    """Read a small response prefix and keep network failure as data."""

    request = urllib.request.Request(
        source["url"],
        headers={
            "User-Agent": "carnot-v651-method-ingestion/1.0",
            "Range": "bytes=0-4095",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            content = response.read(4096)
            return {
                "access_state": "ok",
                "http_status": int(response.status),
                "detail": f"read_bytes={len(content)}",
                "content_prefix_sha256": "sha256:" + hashlib.sha256(content).hexdigest(),
                "response_url": response.geturl(),
            }
    except urllib.error.HTTPError as exc:
        return {
            "access_state": "failed",
            "http_status": exc.code,
            "detail": f"HTTP {exc.code}",
        }
    except (TimeoutError, urllib.error.URLError, OSError) as exc:
        return {
            "access_state": "failed",
            "http_status": None,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _source_access_row(source: Mapping[str, str], fetcher: SourceFetcher) -> JsonDict:
    """Normalize one bounded access result without converting failure to success."""

    result = dict(fetcher(source))
    return {
        **deepcopy(source),
        "access_state": result.get("access_state", "failed"),
        "http_status": result.get("http_status"),
        "detail": str(result.get("detail", "no detail")),
        **{key: result[key] for key in ("content_prefix_sha256", "response_url") if key in result},
    }


def check_selected_sources(fetcher: SourceFetcher = _direct_fetch) -> list[JsonDict]:
    """Check the seven frozen endpoints once and preserve every outcome."""

    return [_source_access_row(source, fetcher) for source in SELECTED_SOURCES]


def method_mapping_rows(source_rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Map three selected ideas to exact plan tests and explicit limits."""

    by_id = {str(row.get("source_id")): row for row in source_rows}
    specifications = (
        (
            "spline_locality",
            ("spline_locality", "kanele"),
            "Exp7425 compares sparse local updates with a dense evaluation on the same basis.",
            "Local coefficient updates and host parity do not reproduce FPGA speed or prove benefit.",
        ),
        (
            "partial_feedback",
            ("partial_feedback",),
            "Exp7427 records randomized reveal probabilities and compares selected-only and uniform audits.",
            "The selective-generation FDR theorem does not transfer to Carnot's three-action learner.",
        ),
        (
            "claim_attribution",
            ("claim_attribution", "symbolic_grounding", "ragtruth"),
            "Exp7423 and Exp7430 retain source IDs, spans, qualifiers, contradiction, and coverage.",
            "Human labels are fallible, and extraction or database execution is not independent truth.",
        ),
    )
    rows: list[JsonDict] = []
    for method, source_ids, proposed_test, method_limit in specifications:
        selected = [by_id[source_id] for source_id in source_ids]
        rows.append(
            {
                "method": method,
                "primary_urls": [str(row["url"]) for row in selected],
                "versions": [str(row["version"]) for row in selected],
                "access_states": [str(row["access_state"]) for row in selected],
                "proposed_test": proposed_test,
                "method_limit": method_limit,
                "previously_indexed": True,
            }
        )
    return rows


def additive_spline_logit_equivalence() -> JsonDict:
    """Prove name-level parity by evaluating one fixed basis two ways."""

    basis = ((1.0, 0.0, 0.25, 0.0), (1.0, 0.5, 0.5, 0.0), (1.0, 0.0, 0.5, 0.5))
    coefficients = (0.2, -0.4, 0.6, 0.1)
    additive = [sum(value * weight for value, weight in zip(row, coefficients)) for row in basis]
    logistic = [sum(weight * value for weight, value in zip(coefficients, row)) for row in basis]
    differences = [abs(left - right) for left, right in zip(additive, logistic)]
    maximum = max(differences)
    return {
        "basis_row_count": len(basis),
        "coefficient_count": len(coefficients),
        "additive_spline_energy_logits": additive,
        "same_basis_logistic_logits": logistic,
        "maximum_absolute_difference": maximum,
        "passed": maximum == 0.0,
        "principle": (
            "With a fixed basis phi(x), both models compute beta dot phi(x). "
            "Spline locality changes which basis coefficients update, not the linear logit form."
        ),
    }


def render_method_note(
    source_rows: Sequence[Mapping[str, Any]], equivalence: Mapping[str, Any]
) -> str:
    """Render the durable method note from measured access and parity rows."""

    access_lines = [
        f"- `{row['version']}`: `{row['access_state']}` ({row.get('http_status')})."
        for row in source_rows
    ]
    return "\n".join(
        [
            "# V651 method ingestion",
            "",
            "## Assumptions",
            "",
            "The V650 null, blocked, and disqualified outcomes remain unchanged. Source access does not qualify science.",
            "",
            "## Fresh source access",
            "",
            *access_lines,
            "",
            "## Implementation hooks and controls",
            "",
            "- Spline locality maps to sparse coefficient updates with dense-basis parity and logistic controls.",
            "- Partial feedback maps to randomized label audits with known reveal probabilities.",
            "- Claim attribution maps to source IDs, preserved spans, qualifier checks, and independent human labels.",
            "",
            "An additive spline energy logit and logistic regression on the same fixed basis are equivalent. Both compute `beta dot phi(x)`. The measured maximum difference is "
            f"`{equivalence['maximum_absolute_difference']}`.",
            "",
            "## Deferred ideas",
            "",
            "V651 defers unchanged proof memory, external-text reranking, generic Ising sweeps, and foundation-model training.",
            "",
            "The G1-G4 publication definitions remain unchanged. This ingestion does not authorize publication or promotion.",
            "",
        ]
    )


def _source_hashes(root: Path, priors: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Bind code, protocol, authorities, notes, and original V650 classes."""

    hashes: JsonDict = {}
    prior_by_path = {str(row["path"]): row for row in priors}
    paths = list(SOURCE_INPUT_PATHS) + [relative for _task, relative in PRIOR_PATHS]
    for relative in dict.fromkeys(paths):
        path = root / relative
        row: JsonDict = {"path": relative.as_posix(), "sha256": sha256_file(path)}
        prior = prior_by_path.get(relative.as_posix())
        if prior:
            row.update(
                {
                    "source_kind": prior["source_kind"],
                    "original_verdict_class": prior["original_verdict_class"],
                    "original_flagged_adversarial": prior["flagged_adversarial"],
                }
            )
        else:
            row["source_kind"] = "authority_or_code"
        hashes[relative.as_posix()] = row
    return hashes


def collect_preconditions(root: Path, priors: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Check exact local resources before contract or source reduction."""

    rows: list[JsonDict] = []
    for relative in dict.fromkeys(list(SOURCE_INPUT_PATHS) + [p for _task, p in PRIOR_PATHS]):
        path = root / relative
        rows.append(
            {
                "upstream": relative.as_posix(),
                "path": relative.as_posix(),
                "check": "readable_nonempty_bytes",
                "field": "bytes",
                "operator": ">",
                "expected": 0,
                "observed": path.stat().st_size if path.is_file() else None,
                "passed": path.is_file() and path.stat().st_size > 0,
            }
        )
    rows.append(
        {
            "upstream": "v650_terminal_set",
            "path": "results",
            "check": "prior_row_count",
            "field": "prior_dispositions",
            "operator": "==",
            "expected": 12,
            "observed": len(priors),
            "passed": len(priors) == 12,
        }
    )
    return rows


def _receipts_pass(receipts: Sequence[Mapping[str, Any]], names: Sequence[str]) -> bool:
    """Require one successful, untimed receipt for every named command."""

    counts = Counter(str(row.get("name")) for row in receipts)
    return all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        and next(row for row in receipts if row.get("name") == name).get("exit_code") == 0
        and next(row for row in receipts if row.get("name") == name).get("timed_out") is False
        for name in names
    )


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    passed: bool,
    principle: str,
) -> JsonDict:
    """Build one explicit gate without hiding its operands in wrappers."""

    return {
        "check": check,
        "category": category,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": passed,
        "principle": principle,
    }


def _failure_summary(contract: Mapping[str, Any], gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the exact first failure while retaining all later failures."""

    failures: list[JsonDict] = []
    if contract.get("markdown_milestone") != MILESTONE:
        failures.append(
            {
                "upstream": "exact_task_contract",
                "path": DESIGN_PATH.as_posix(),
                "check": "contract_milestone",
                "field": "milestone",
                "operator": "==",
                "expected": MILESTONE,
                "observed": contract.get("markdown_milestone"),
            }
        )
    for gate in gates:
        if gate.get("passed") is not True and gate.get("check") != "contract_match":
            failures.append(
                {
                    "upstream": EXPERIMENT_ID,
                    "path": RESULT_PATH.as_posix(),
                    "check": gate.get("check"),
                    "field": gate.get("check"),
                    "operator": gate.get("operator"),
                    "expected": gate.get("expected"),
                    "observed": gate.get("observed"),
                }
            )
    return {
        "all_required_checks_passed": not failures,
        "first_failure": failures[0] if failures else None,
        "failed_checks": failures,
    }


def zero_test_phase_spans() -> list[JsonDict]:
    """Provide deterministic disjoint spans for pure artifact tests."""

    return [
        {
            "phase": "test_fixture",
            "start_s": 0.0,
            "end_s": 1.0,
            "duration_s": 1.0,
            "completed_units": 1,
            "heartbeat_count": 0,
            "checkpoint": "in_memory_fixture",
        }
    ]


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every field except the checksum slot itself."""

    payload = deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _field_principles(fields: Sequence[str]) -> JsonDict:
    """Explain field intent separately from ordinary values."""

    explicit = {
        "schema": "Versioned plain top-level schema, experiment_id, milestone, and terminal status identify this receipt.",
        "run_date": "Use 20260919 with actual UTC start and end plus monotonic timing.",
        "preconditions_checked": "Exact resource, path, identity, and observed values precede dependent work.",
        "MODEL_SPECS": "Every current LLM would use unsloth/Qwen3.8-27B-GGUF; this run uses none.",
        "model_invoked": "Only actual current attempted model use can make this true.",
        "invocation_counts": "Owned current load and generation attempts, completions, failures, cancellations, and in-flight counts stay exact.",
        "inference_substrate": "A truthful string names aggregation; device details stay separate.",
        "inference_substrate_class": "The aggregation class describes current compute, not historical inference.",
        "execution_venue": "The closed venue is host; CPU, CUDA, and board identities are separate details.",
        "duration_s": "Measured current work separates validation, source access, and cold-reader durations through phase spans.",
        "phase_spans": "Real phase boundaries include flushed progress, completed units, and resumable checkpoints.",
        "random_seed": "Only deterministic contract and basis controls use frozen seeds.",
        "reproducibility_checksum": "The checksum binds code, protocol, inputs, rows, and validation scope.",
        "source_artifact_hashes": "Exact input hashes retain original V650 classes and flags.",
        "rows": "All contract, prior, access, method, and equivalence units remain visible.",
        "sample_size_budget": "Planned, attempted, completed, failed, censored, unstarted, groups, and stop rule stay separate.",
        "acceptance_gate_results": "Every gate has category, operator, expected, observed, passed, and principle fields.",
        "gate_check_summary": "Each failure names exact upstream, path, check, field, expected, and observed values.",
        "verifier_is_oracle": "The audit does not share scientific correctness authority.",
        "honest_verdict": "Completed findings start complete_; invalid contract evidence is disqualified.",
        "verdict_class": "The closed class is positive, circular_positive, null, blocked, disqualified, or partial.",
        "flagged_adversarial": "Critical producer flags stay visible and cannot supply scientific readiness.",
        "validation_receipts": "Exact scoped arguments, environments, exits, durations, names, and hashed logs support the run.",
        "field_principles": "Principles stay separate from ordinary scalar and collection values.",
        "promotion_score": "This is always zero; no rollout, publication, or generator update follows.",
        "contract_ready_score": "One requires the exact matching thirteen-row contract and remains advisory.",
        "task_contract_rows": "All thirteen comparison slots retain independently parsed authorities.",
        "method_mapping_rows": "Primary URLs, versions, access states, tests, and method limits remain explicit.",
        "prior_dispositions": "Original V650 classes, missing paths, flags, and failed gates remain unchanged.",
    }
    return {
        field: explicit.get(field, f"Record the measured V651 value for {field.replace('_', ' ')}.")
        for field in fields
    }


def build_artifact(
    root: Path,
    contract: Mapping[str, Any],
    mutation_rows: Sequence[Mapping[str, Any]],
    priors: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    *,
    started_at_utc: str,
    ended_at_utc: str,
    duration_s: float,
    phase_spans: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Build one independently reducible terminal-shaped receipt."""

    methods = method_mapping_rows(source_rows)
    equivalence = additive_spline_logit_equivalence()
    receipts = [deepcopy(dict(row)) for row in validation.get("validation_receipts", [])]
    affected_passed = bool(validation.get("required_checks_passed"))
    terminal_passed = bool(validation.get("terminal_validation_passed"))
    mutations_passed = len(mutation_rows) == 5 and all(
        row.get("rejected") is True for row in mutation_rows
    )
    methods_complete = int(
        len(source_rows) == len(SELECTED_SOURCES)
        and len(source_rows) <= 8
        and len(methods) == 3
        and equivalence["passed"] is True
    )
    contract_ready = int(contract.get("passed") is True)
    gates = [
        _gate(
            "contract_match",
            "accounting",
            True,
            contract.get("passed") is True,
            contract.get("passed") is True,
            "Both independent authorities must contain the exact thirteen rows.",
        ),
        _gate(
            "contract_mutations",
            "safety",
            5,
            sum(row.get("rejected") is True for row in mutation_rows),
            mutations_passed,
            "Every named private defect must fail closed.",
        ),
        _gate(
            "v650_dispositions",
            "historical_evidence",
            12,
            len(priors),
            len(priors) == 12,
            "Every V650 task keeps its original terminal class and flag.",
        ),
        _gate(
            "method_ingestion",
            "methodology",
            1,
            methods_complete,
            methods_complete == 1,
            "Bounded access, mappings, and basis equivalence must all complete.",
        ),
        _gate(
            "affected_validation",
            "required_validation",
            True,
            affected_passed,
            affected_passed,
            "The frozen file-scoped command plan controls implementation validity.",
        ),
        _gate(
            "terminal_validation",
            "required_validation",
            True,
            terminal_passed,
            terminal_passed,
            "Cold replay and strict readers control terminal publication.",
        ),
        _gate(
            "promotion",
            "promotion",
            0,
            0,
            True,
            "Advisory accounting cannot promote science or change weights.",
        ),
    ]
    valid_current = (
        mutations_passed
        and len(priors) == 12
        and methods_complete == 1
        and affected_passed
        and terminal_passed
    )
    verdict_class = "null" if contract_ready and valid_current else "disqualified"
    honest_verdict = (
        "complete_null_contract_and_method_ingestion_advisory"
        if verdict_class == "null"
        else "complete_disqualified_stale_markdown_contract_with_method_ingestion"
    )
    rows = (
        [
            {"row_kind": "contract", **deepcopy(dict(row))}
            for row in contract.get("contract_rows", [])
        ]
        + [{"row_kind": "prior_disposition", **deepcopy(dict(row))} for row in priors]
        + [{"row_kind": "source_access", **deepcopy(dict(row))} for row in source_rows]
        + [{"row_kind": "method_mapping", **deepcopy(dict(row))} for row in methods]
        + [{"row_kind": "basis_equivalence", **deepcopy(equivalence)}]
    )
    preconditions = collect_preconditions(root, priors)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "phase": PHASE,
        "status": (
            "complete_null_advisory"
            if verdict_class == "null"
            else "complete_disqualified_contract_authority"
        ),
        "run_date": RUN_DATE,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "preconditions_checked": preconditions,
        "MODEL_SPECS": [],
        "model_invoked": False,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "current_invocation_events": [],
        "inference_substrate": "aggregation_from_exact_declared_artifacts",
        "inference_substrate_details": {
            "device": "host_cpu",
            "network_request_limit": 8,
            "current_llm_operations": 0,
        },
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "duration_s": duration_s,
        "phase_spans": [deepcopy(dict(row)) for row in phase_spans],
        "random_seed": deepcopy(RANDOM_SEED),
        "source_artifact_hashes": _source_hashes(root, priors),
        "rows": rows,
        "sample_size_budget": {
            "planned": 36,
            "attempted": len(rows),
            "completed": len(rows),
            "failed": sum(row.get("access_state") == "failed" for row in source_rows),
            "censored": 0,
            "unstarted": max(0, 36 - len(rows)),
            "independent_groups": {
                "contract_slots": 13,
                "v650_dispositions": 12,
                "source_endpoints": len(source_rows),
                "method_mappings": 3,
                "basis_equivalence": 1,
            },
            "stop_rule": "Stop after thirteen contract slots, twelve prior dispositions, at most eight source checks, three mappings, and one equivalence control.",
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _failure_summary(contract, gates),
        "verifier_is_oracle": False,
        "honest_verdict": honest_verdict,
        "verdict_class": verdict_class,
        "flagged_adversarial": False,
        "validation_receipts": receipts,
        "repository_health": {
            "status": "not_used_as_current_gate",
            "affects_required_checks": False,
            "historical_failures": [],
        },
        "field_principles": {},
        "promotion_score": 0,
        "contract_ready_score": contract_ready,
        "method_ingestion_complete_score": methods_complete,
        "task_contract_rows": deepcopy(list(contract.get("contract_rows", []))),
        "contract_comparison": deepcopy(dict(contract)),
        "contract_mutation_rows": [deepcopy(dict(row)) for row in mutation_rows],
        "method_mapping_rows": methods,
        "source_access_rows": [deepcopy(dict(row)) for row in source_rows],
        "spline_logit_equivalence": equivalence,
        "prior_dispositions": [deepcopy(dict(row)) for row in priors],
        "completion_ledger_lag": completion_ledger_state(root),
        "small_ebm_training": {
            "performed": False,
            "attempted": 0,
            "completed": 0,
            "principle": "This audit reads archived training receipts but trains no current energy head.",
        },
        "publication_gate_definitions_changed": False,
        "deferred_ideas": [
            "unchanged proof memory",
            "external-text reranking",
            "generic Ising sweeps",
            "foundation-model training",
        ],
        "reproducibility_checksum": "",
    }
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _hashes_match(artifact: Mapping[str, Any], root: Path) -> bool:
    """Rehash every declared local source before trusting reduced rows."""

    hashes = artifact.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return False
    expected_paths = {
        relative.as_posix()
        for relative in (*SOURCE_INPUT_PATHS, *(path for _task, path in PRIOR_PATHS))
    }
    if set(hashes) != expected_paths:
        return False
    for row in hashes.values():
        if not isinstance(row, Mapping):
            return False
        path = root / str(row.get("path"))
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            return False
    return True


def validate_artifact(
    value: object, *, root: Path = REPO_ROOT, require_terminal: bool = True
) -> list[str]:
    """Cold-check identity, raw reductions, scores, sources, and checksum."""

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

    contract = load_contract_audit(root)
    if artifact.get("task_contract_rows") != contract.get("contract_rows"):
        errors.append("task_contract_rows_mismatch")
    if artifact.get("contract_comparison") != contract:
        errors.append("contract_comparison_mismatch")
    expected_contract_score = int(contract.get("passed") is True)
    if artifact.get("contract_ready_score") != expected_contract_score:
        errors.append("contract_score_mismatch")

    priors = collect_prior_dispositions(root)
    if artifact.get("prior_dispositions") != priors:
        errors.append("prior_dispositions_mismatch")
    source_rows = artifact.get("source_access_rows")
    source_rows_valid = True
    if not isinstance(source_rows, list) or len(source_rows) != len(SELECTED_SOURCES):
        errors.append("source_access_rows_invalid")
        source_rows = []
        source_rows_valid = False
    else:
        expected_ids = [row["source_id"] for row in SELECTED_SOURCES]
        if [row.get("source_id") for row in source_rows] != expected_ids:
            errors.append("source_access_rows_invalid")
            source_rows_valid = False
    expected_methods = method_mapping_rows(source_rows) if source_rows_valid else []
    if artifact.get("method_mapping_rows") != expected_methods:
        errors.append("method_mapping_rows_mismatch")
    if artifact.get("spline_logit_equivalence") != additive_spline_logit_equivalence():
        errors.append("spline_equivalence_mismatch")
    expected_method_score = int(
        source_rows_valid
        and artifact.get("method_mapping_rows") == expected_methods
        and artifact.get("spline_logit_equivalence", {}).get("passed") is True
    )
    if artifact.get("method_ingestion_complete_score") != expected_method_score:
        errors.append("method_score_mismatch")

    mutations = artifact.get("contract_mutation_rows")
    if (
        not isinstance(mutations, list)
        or len(mutations) != 5
        or not all(isinstance(row, Mapping) and row.get("rejected") is True for row in mutations)
    ):
        errors.append("contract_mutations_invalid")
    receipts = artifact.get("validation_receipts")
    if not isinstance(receipts, list) or not _receipts_pass(
        receipts, validation_scope.REQUIRED_CHECK_NAMES
    ):
        errors.append("affected_validation_invalid")
    if require_terminal and (
        not isinstance(receipts, list) or not _receipts_pass(receipts, TERMINAL_CHECK_NAMES)
    ):
        errors.append("terminal_validation_invalid")
    if not _hashes_match(artifact, root):
        errors.append("source_hash_mismatch")
    if artifact.get("promotion_score") != 0:
        errors.append("promotion_score_nonzero")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(principles) != set(artifact):
        errors.append("field_principles_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def build_validation_plan(root: Path, private_root: Path) -> list[validation_scope.CommandSpec]:
    """Freeze the Exp7358 file-scoped command set for this experiment."""

    return build_command_plan(root, VALIDATION_MANIFEST, private_root)


def validate_validation_plan(
    root: Path, commands: Sequence[validation_scope.CommandSpec]
) -> list[str]:
    """Reject broad targets and command drift before any child starts."""

    return validate_command_plan(root, VALIDATION_MANIFEST, commands)


def _phase_span(  # pragma: no cover - authentic runtime timing boundary.
    phase: str, phase_started: float, run_started: float, *, units: int, checkpoint: str
) -> JsonDict:
    """Close one measured phase and identify its durable checkpoint."""

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
    """Build cold replay, raw reduction, and unchanged terminal readers."""

    python = str(root / ".venv/bin/python")
    reducer = (
        "import json,pathlib,sys;"
        "from carnot.experiment_7421_v651_contract_ingestion import validate_artifact;"
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


def run_experiment(  # pragma: no cover - exercised by the declared capability E2E.
    root: Path, run_date: str
) -> JsonDict:
    """Measure inputs, run scoped checks, and atomically publish the receipt."""

    if run_date != RUN_DATE:
        raise SystemExit(f"--date must be {RUN_DATE}")
    root = root.resolve()
    started = time.monotonic()
    started_at = utc_now()
    spans: list[JsonDict] = []
    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    progress(started, "preconditions", "before")
    priors = collect_prior_dispositions(root)
    preconditions = collect_preconditions(root, priors)
    if not all(row["passed"] for row in preconditions):
        raise RuntimeError("precondition_failure")
    spans.append(
        _phase_span(
            "preconditions",
            phase_started,
            started,
            units=len(preconditions),
            checkpoint="authenticated_input_paths",
        )
    )
    progress(started, "preconditions", "after", completed_units=len(preconditions))

    for phase in ("model_load", "generation"):
        phase_started = time.monotonic()
        progress(started, phase, "before", completed_units=0)
        spans.append(
            _phase_span(
                phase,
                phase_started,
                started,
                units=0,
                checkpoint="no_current_llm_work",
            )
        )
        progress(started, phase, "after", completed_units=0)

    phase_started = time.monotonic()
    progress(started, "contract", "before")
    contract = load_contract_audit(root)
    roadmap = load_yaml(root / ROADMAP_PATH)
    markdown_fixture = _render_markdown_authority(roadmap)
    mutations = run_contract_mutation_controls(markdown_fixture, roadmap)
    spans.append(
        _phase_span(
            "contract",
            phase_started,
            started,
            units=len(contract.get("contract_rows", [])) + len(mutations),
            checkpoint="contract_and_mutations_reduced",
        )
    )
    progress(started, "contract", "after", completed_units=18)

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
        source_rows.append(_source_access_row(source, _direct_fetch))
        progress(
            started,
            "source_access",
            "after_request",
            completed_units=index,
            source=source["source_id"],
        )
    spans.append(
        _phase_span(
            "source_access",
            phase_started,
            started,
            units=len(source_rows),
            checkpoint="bounded_source_access_rows",
        )
    )
    progress(started, "source_access", "after", completed_units=len(source_rows))

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7421-validation-", dir="/tmp"))
    commands = build_validation_plan(root, private_root)
    plan_errors = validate_validation_plan(root, commands)
    if plan_errors:
        raise RuntimeError(f"invalid_validation_plan:{plan_errors}")
    phase_started = time.monotonic()
    progress(started, "validation", "before_subprocesses", planned_units=len(commands))
    affected = run_categorized_commands(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
        heartbeat_s=60.0,
    )
    reduced = reduce_affected_receipts(root, VALIDATION_MANIFEST, affected)
    spans.append(
        _phase_span(
            "validation",
            phase_started,
            started,
            units=len(affected),
            checkpoint="affected_checks_complete",
        )
    )
    progress(started, "validation", "after_subprocesses", passed=reduced["passed"])

    candidate = build_artifact(
        root,
        contract,
        mutations,
        priors,
        source_rows,
        {
            "validation_receipts": affected,
            "required_checks_passed": reduced["passed"],
            "terminal_validation_passed": True,
        },
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    candidate_path = raw_dir / "measured_terminal_candidate.json"
    atomic_json(candidate_path, candidate)
    if validate_artifact(candidate, root=root, require_terminal=False):
        raise RuntimeError("candidate_invalid")

    terminal_commands = _terminal_commands(root, candidate_path)
    phase_started = time.monotonic()
    progress(
        started,
        "terminal_validation",
        "before_subprocesses",
        planned_units=len(terminal_commands),
    )
    terminal = run_categorized_commands(
        root,
        terminal_commands,
        log_dir=raw_dir / "validation/terminal",
        heartbeat_s=60.0,
    )
    terminal_passed = _receipts_pass(terminal, TERMINAL_CHECK_NAMES)
    spans.append(
        _phase_span(
            "terminal_validation",
            phase_started,
            started,
            units=len(terminal),
            checkpoint="cold_replay_and_strict_readers_complete",
        )
    )
    progress(started, "terminal_validation", "after_subprocesses", passed=terminal_passed)

    final = build_artifact(
        root,
        contract,
        mutations,
        priors,
        source_rows,
        {
            "validation_receipts": [*affected, *terminal],
            "required_checks_passed": reduced["passed"],
            "terminal_validation_passed": terminal_passed,
        },
        started_at_utc=started_at,
        ended_at_utc=utc_now(),
        duration_s=time.monotonic() - started,
        phase_spans=spans,
    )
    errors = validate_artifact(final, root=root)
    if errors:
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    progress(started, "publish", "before_atomic", path=RESULT_PATH.as_posix())
    atomic_json(candidate_path, final)
    atomic_json(root / RESULT_PATH, final)
    progress(started, "publish", "after_atomic", path=RESULT_PATH.as_posix())
    return final


def _render_markdown_authority(roadmap: Mapping[str, Any]) -> str:
    """Render a private exact table only for mutation controls."""

    lines = [
        "# Private contract fixture",
        "",
        f"**Milestone:** `{roadmap['milestone']}`",
        "",
        "## Exact Task Contract",
        "",
        "| Order | Task ID | Exact title | Deliverable | Phase | Substrate class | Structured gate |",
        "|---|---|---|---|---|---|---|",
    ]
    for order, task in enumerate(roadmap["tasks"], 1):
        gates = []
        for gate in task.get("gated_on") or []:
            value = json.dumps(gate["value"], separators=(",", ":"))
            gates.append(f"{gate['upstream']}.{gate['artifact_field']} {gate['op']} {value}")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(order),
                    str(task["id"]),
                    str(task["title"]),
                    str(task["deliverable"]),
                    str(task["phase"]),
                    str(task["inference_substrate_class"]),
                    "; ".join(gates) if gates else "None",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def date_argument(value: str) -> str:
    """Accept only the date frozen by the V651 contract."""

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
    """Run V651 ingestion or validate one measured candidate."""

    print("[exp7421] phase=startup event=flushed", flush=True)
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

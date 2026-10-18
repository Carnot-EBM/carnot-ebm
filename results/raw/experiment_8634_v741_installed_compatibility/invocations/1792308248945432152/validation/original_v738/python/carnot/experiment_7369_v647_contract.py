"""Build the V647 advisory source and exact twelve-task contract receipt.

The module reuses the shipped contract parsers, conductor gate controls, and
scoped validation plan. It invokes no model and creates no science result.

Spec refs: REQ-REPORT-7369 and SCENARIO-REPORT-7369-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import re
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
    _public_task,
    parse_markdown_contract,
    parse_yaml_contract,
    sha256,
    sha256_bytes,
)
from carnot.experiment_7343_v645_contract import (
    GATE_CASES,
    _failed_names,
    gate_controls_complete,
    roadmap_command_specs,
    run_gate_controls,
)
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    reduce_affected_receipts,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    run_commands,
)
from scripts.roadmap_schema import Roadmap


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]
ValidationRunner = Callable[..., JsonDict]
TerminalRunner = Callable[[Path, Path, Path], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.647"
RUN_DATE = "20260917"
RETRIEVAL_DATE = "2026-09-17"
FIRST_TASK_ID = "exp7369-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7370-proof-memory",
    "exp7371-proof-boundary",
    "exp7372-qwen-canary",
    "exp7373-proposal-capture",
    "exp7374-prospective-memory",
    "exp7375-memory-audit",
    "exp7376-arc-outcomes",
    "exp7377-ising-law",
    "exp7378-ising-audit",
    "exp7379-hardware-envelope",
    "exp7380-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
EXPECTED_NUMBERS = tuple(range(7369, 7381))
RANDOM_SEED = {
    "experiment": 7_369_202_609_17,
    "mutation_controls": 7_369_202_609_18,
    "resampling": 7_369_202_609_19,
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
HISTORY_PATH = Path("results/experiment_7357_v646_contract.json")

MODULE_PATH = Path("python/carnot/experiment_7369_v647_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7369_v647_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7369_v647_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7369_v647_contract.json")
RAW_DIR = Path("results/raw/experiment_7369_v647_contract")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7369_v647_contract.json")
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
    Path("scripts/roadmap_schema.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    Path("scripts/harness_consumer_checks.py"),
    Path("scripts/arc_levelup_guarantee_lint.py"),
    Path("scripts/overdue_priority_lint.py"),
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7329_v644_contract.py"),
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
ALL_REQUIRED_CHECK_NAMES = (
    *ROADMAP_CHECK_NAMES,
    *REQUIRED_CHECK_NAMES,
    *TERMINAL_CHECK_NAMES,
)
CONTRACT_MUTATION_NAMES = (
    "deleted_task",
    "reordered_task",
    "changed_milestone",
    "changed_title",
    "changed_path",
    "changed_phase",
    "changed_substrate",
    "misspelled_gate",
    "removed_retire_if_same_verdict",
)

V647_MANIFEST = AffectedManifest(
    experiment_id=FIRST_TASK_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Versioned schema with ordinary top-level experiment_id and milestone.",
    "status": "Terminal only after actual work and required validation; no success-shaped bootstrap artifact.",
    "run_date": "Use 20260917 plus actual start/end UTC timestamps.",
    "preconditions_checked": "Exact paths, producer identity/hash/class and resource checks before dependent work.",
    "MODEL_SPECS": "Actual intended model identities; empty for host-only work, Qwen3.8 mandated for current model work.",
    "model_invoked": "True for any attempted current model load or generation, even a failed attempt.",
    "invocation_counts": "Attempted/completed/failed/cancelled/in-flight loads and generations; do not count historical input as current inference.",
    "inference_substrate": "Actual host or owned native CUDA computation with historical provenance in labeled sidecars.",
    "inference_substrate_class": "The declared closed class matching actual work; no duration padding.",
    "execution_venue": "Measured host CPU or owned CUDA device; no current board execution in V647.",
    "duration_s": "Measured monotonic duration, never invented time or sleep used to meet a floor.",
    "phase_spans": "Measured read/build/load/generate/evaluate/validate/write spans and checkpoint timestamps.",
    "random_seed": "Frozen experiment and resampling seeds; null only when not applicable.",
    "reproducibility_checksum": "Bind exact code, settings, source/formula/protocol and raw rows.",
    "source_artifact_hashes": "Exact producer paths and byte hashes; preserve historical verdicts and flags.",
    "rows": "Every unit/arm outcome, metric, cost, failure and censoring disposition behind comparative claims.",
    "sample_size_budget": "Predeclared planned/attempted/completed/censored units, stopping rules and remaining work.",
    "acceptance_gate_results": "Expected/observed/passed separated for required checks, safety, completion and scientific efficacy.",
    "gate_check_summary": "Every blocked_* names failed upstream/check, exact field, expected and observed value, including missing paths.",
    "verifier_is_oracle": "True if the evaluator defines truth; different code alone does not remove circularity.",
    "honest_verdict": "Completed work uses complete_ or complete: with precise scope; blocked_* names an unavailable prerequisite.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Only unfinished retryable OWN work is partial; unchanged external absence is blocked.",
    "flagged_adversarial": "True for a critical independent finding; excludes producer from readiness gates.",
    "validation_receipts": "Executed command argv/environment/scope/return code/duration/log hash, including failures.",
    "repository_health": "Dated unrelated failures distinct from required affected checks; no weakening or hiding tests.",
    "field_principles": "Explain each field without wrapping ordinary dictionaries or numeric gate values.",
    "promotion_score": "Zero for this milestone: no automatic production rollout or external publication.",
    "contract_complete_score": "One only for an exact twelve-row contract and all applicable plan checks. Advisory only.",
    "source_method_rows": "Dated primary URL, access result, adopted control or defer reason; no invented novelty.",
    "mutation_rows": "Each private contract mutation and the exact check that rejects it.",
    "science_value_score": "Zero: a correct plan is not scientific value.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

ACCESS_SPECS: tuple[JsonDict, ...] = (
    {
        "access_id": "parameterized_2sat",
        "url": "https://arxiv.org/abs/2602.12665",
        "publication_or_version_date": "2026-02-13",
    },
    {
        "access_id": "semantic_realizability",
        "url": "https://arxiv.org/html/2509.00360v1",
        "publication_or_version_date": "2025-08-30",
    },
    {
        "access_id": "no_free_checker",
        "url": "https://arxiv.org/abs/2609.09250",
        "publication_or_version_date": "2026-09-08",
    },
    {
        "access_id": "t_oracle_fastca",
        "url": "https://arxiv.org/html/2609.12267v1",
        "publication_or_version_date": "2026-09-10",
    },
    {
        "access_id": "kan_cl",
        "url": "https://arxiv.org/abs/2605.12306",
        "publication_or_version_date": "2026-05-12",
    },
    {
        "access_id": "ebt",
        "url": "https://arxiv.org/abs/2507.02092",
        "publication_or_version_date": "2025-07-02",
    },
    {
        "access_id": "arm_ebm",
        "url": "https://arxiv.org/abs/2512.15605v4",
        "publication_or_version_date": "2026-05-25",
    },
    {
        "access_id": "ising_hardware",
        "url": "https://arxiv.org/abs/2602.15985",
        "publication_or_version_date": "2026-02-17",
    },
    {
        "access_id": "semantic_scholar_ebt",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2507.02092/citations?fields=title,year,externalIds&limit=20",
        "publication_or_version_date": RETRIEVAL_DATE,
    },
    {
        "access_id": "semantic_scholar_arm_ebm",
        "url": "https://api.semanticscholar.org/graph/v1/paper/ARXIV:2512.15605/citations?fields=title,year,externalIds&limit=20",
        "publication_or_version_date": RETRIEVAL_DATE,
    },
)
ACCESS_IDS = tuple(row["access_id"] for row in ACCESS_SPECS)
DATED_DELTA_ACCESS_OUTCOMES: tuple[JsonDict, ...] = (
    {
        "access_id": "semantic_scholar_ebt",
        "observed_at": "2026-09-17 planning literature scan",
        "access_outcome": "http_429",
        "provenance": "research-references.md V647 source coverage and access receipts",
        "inferred_citation_count": None,
    },
    {
        "access_id": "semantic_scholar_arm_ebm",
        "observed_at": "2026-09-17 planning literature scan",
        "access_outcome": "http_429",
        "provenance": "research-references.md V647 source coverage and access receipts",
        "inferred_citation_count": None,
    },
)

SOURCE_METHODS: tuple[JsonDict, ...] = (
    {
        "method_family": "parameterized_2sat",
        "access_ids": ["parameterized_2sat"],
        "adopted_control_or_defer_reason": "Use controlled contradiction, backbone, bridge, renaming, and duplication families with version splits and a persistent exact solver.",
        "destination_experiment_ids": ["exp7370-proof-memory", "exp7374-prospective-memory"],
        "evidence_limit": "Formal 2-SAT families do not establish natural-language truth.",
    },
    {
        "method_family": "semantic_realizability",
        "access_ids": ["semantic_realizability"],
        "adopted_control_or_defer_reason": "Reject a partial assignment only when original clauses yield a finite contradiction path.",
        "destination_experiment_ids": ["exp7371-proof-boundary", "exp7373-proposal-capture"],
        "evidence_limit": "Completed-proposal filtering is not token-level constrained decoding.",
    },
    {
        "method_family": "verifier_authority",
        "access_ids": ["no_free_checker", "t_oracle_fastca"],
        "adopted_control_or_defer_reason": "Separate learner, checker, and evaluator information; admit implications only against exact current source clauses.",
        "destination_experiment_ids": ["exp7371-proof-boundary", "exp7375-memory-audit"],
        "evidence_limit": "A learned oracle cannot define truth or establish a verifier moat.",
    },
    {
        "method_family": "kan_deferral",
        "access_ids": ["kan_cl"],
        "adopted_control_or_defer_reason": "Defer a KAN head until a non-oracle prediction target and useful exact baseline exist.",
        "destination_experiment_ids": [],
        "evidence_limit": "Vision retention results do not certify logical consequences.",
    },
    {
        "method_family": "ebt_deferral",
        "access_ids": ["ebt", "arm_ebm"],
        "adopted_control_or_defer_reason": "Keep EBT and ARM-EBM as architecture context; add no foundation-model training branch.",
        "destination_experiment_ids": [],
        "evidence_limit": "Compatibility energy and function-space equivalence do not certify source semantics.",
    },
    {
        "method_family": "ising_hardware_cost",
        "access_ids": ["ising_hardware"],
        "adopted_control_or_defer_reason": "Preserve the source Boltzmann law and count construction, updates, transfers, and amortization.",
        "destination_experiment_ids": ["exp7377-ising-law", "exp7379-hardware-envelope"],
        "evidence_limit": "External hardware results are not local Carnot performance or availability evidence.",
    },
)


def progress(phase: str, event: str, started: float, detail: str = "") -> None:
    """Print each boundary with real monotonic elapsed time and flush it."""

    suffix = f" {detail}" if detail else ""
    print(
        f"[exp7369] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Prefer the staged roadmap only when its milestone is exactly V647."""

    selected: tuple[Path, JsonDict, bytes] | None = None
    candidates: list[JsonDict] = []
    for relative in (NEXT_ROADMAP_PATH, ACTIVE_ROADMAP_PATH):
        path = root / relative
        try:
            content = path.read_bytes()
            document = yaml.safe_load(content.decode("utf-8"))
            if not isinstance(document, dict) or not document:
                raise ValueError(f"YAML mapping required at {path}")
            observed: object = document.get("milestone")
            matches = observed == MILESTONE
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            content, document, observed, matches = b"", None, f"{type(exc).__name__}: {exc}", False
        candidates.append(
            {
                "check": f"yaml_candidate:{relative}",
                "path": relative.as_posix(),
                "upstream": relative.as_posix(),
                "artifact_field": "milestone",
                "expected_value": MILESTONE,
                "observed_value": observed,
                "available": matches,
                "blocking": False,
            }
        )
        if selected is None and matches and document is not None:
            selected = (relative, document, content)
    if selected is None:
        return None, None, None, candidates
    return *selected, candidates


def parse_active_markdown_contract(text: str) -> JsonDict:
    """Parse one active Exact Task Contract and reject duplicate sections."""

    active = re.split(r"(?m)^## Historical ", text, maxsplit=1)[0]
    count = len(re.findall(r"(?im)^## Exact Task Contract\s*$", active))
    if count != 1:
        raise ValueError(f"expected exactly one active Exact Task Contract, observed {count}")
    parsed = parse_markdown_contract(active)
    parsed["active_exact_contract_count"] = count
    return parsed


def validate_gate_declarations(roadmap: Mapping[str, Any]) -> JsonDict:
    """Require each consumed field from an earlier declared producer."""

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
            precedes = producer_index is not None and producer_index < consumer_index
            rows.append(
                {
                    "consumer": consumer.get("id"),
                    "upstream": upstream,
                    "artifact_field": gate.get("artifact_field"),
                    "producer_precedes_consumer": precedes,
                    "declared_verbatim": declared,
                    "receipt_is_not_upstream": upstream != FIRST_TASK_ID,
                    "passed": bool(precedes and declared and upstream != FIRST_TASK_ID),
                }
            )
    return {"rows": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def validate_retirement_declarations(roadmap: Mapping[str, Any]) -> JsonDict:
    """Require each V647 prior-failure row to keep its retirement trigger."""

    rows: list[JsonDict] = []
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            rows.append({"task_id": None, "prior_index": None, "passed": False})
            continue
        for index, prior in enumerate(task.get("prior_failures") or []):
            passed = isinstance(prior, Mapping) and prior.get("retire_if_same_verdict") is True
            rows.append(
                {
                    "task_id": task.get("id"),
                    "prior_index": index,
                    "retire_if_same_verdict": (
                        prior.get("retire_if_same_verdict") if isinstance(prior, Mapping) else None
                    ),
                    "passed": passed,
                }
            )
    return {"rows": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare all V647 row fields from independently parsed authorities."""

    markdown = parse_active_markdown_contract(markdown_text)
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
            "deliverable": bool(
                expected and observed and expected.get("deliverable") == observed.get("deliverable")
            ),
            "phase": bool(expected and observed and expected.get("phase") == observed.get("phase")),
            "substrate": bool(
                expected and observed and expected.get("substrate") == observed.get("substrate")
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
        and markdown.get("active_exact_contract_count") == 1
        and declarations.get("passed") is True
        and all(row["passed"] for row in rows)
    )
    return {
        "markdown_milestone": markdown.get("milestone"),
        "yaml_milestone": parsed_yaml.get("milestone"),
        "active_exact_contract_count": markdown.get("active_exact_contract_count"),
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
    """Apply every frozen private mutation and name the rejecting check."""

    def deleted_task(document: JsonDict) -> None:
        document["tasks"] = document["tasks"][:-1]

    def reordered_task(document: JsonDict) -> None:
        document["tasks"][0], document["tasks"][1] = document["tasks"][1], document["tasks"][0]

    def changed_milestone(document: JsonDict) -> None:
        document["milestone"] = "2026.09.646"

    def changed_title(document: JsonDict) -> None:
        document["tasks"][0]["title"] += " changed"

    def changed_path(document: JsonDict) -> None:
        document["tasks"][0]["deliverable"] = "results/mutated.json"

    def changed_phase(document: JsonDict) -> None:
        document["tasks"][0]["phase"] = 4

    def changed_substrate(document: JsonDict) -> None:
        prompt = document["tasks"][0]["prompt"]
        document["tasks"][0]["prompt"] = prompt.replace(
            "inference_substrate_class=aggregation",
            "inference_substrate_class=cpu_exact_solver_or_simulator",
            1,
        )

    def misspelled_gate(document: JsonDict) -> None:
        gated = next(task for task in document["tasks"] if task.get("gated_on"))
        gated["gated_on"][0]["artifact_feld"] = gated["gated_on"][0].pop("artifact_field")

    def removed_retire(document: JsonDict) -> None:
        prior = next(task for task in document["tasks"] if task.get("prior_failures"))
        prior["prior_failures"][0].pop("retire_if_same_verdict")

    mutations: tuple[tuple[str, Callable[[JsonDict], None], str], ...] = (
        ("deleted_task", deleted_task, "exact_contract"),
        ("reordered_task", reordered_task, "exact_contract"),
        ("changed_milestone", changed_milestone, "exact_contract"),
        ("changed_title", changed_title, "exact_contract"),
        ("changed_path", changed_path, "exact_contract"),
        ("changed_phase", changed_phase, "exact_contract"),
        ("changed_substrate", changed_substrate, "exact_contract"),
        ("misspelled_gate", misspelled_gate, "roadmap_schema"),
        ("removed_retire_if_same_verdict", removed_retire, "retirement_contract"),
    )
    baseline_passed = bool(
        evaluate_contract(markdown_text, roadmap)["passed"]
        and validate_retirement_declarations(roadmap)["passed"]
    )
    Roadmap.model_validate(roadmap)
    temp_root.mkdir(parents=True, exist_ok=True)
    rows: list[JsonDict] = []
    for name, mutate, check in mutations:
        changed = deepcopy(dict(roadmap))
        mutate(changed)
        path = temp_root / f"{name}.yaml"
        path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
        error: str | None = None
        try:
            if check == "roadmap_schema":
                Roadmap.model_validate(changed)
                observed = True  # pragma: no cover - the missing required gate field must reject.
            elif check == "retirement_contract":
                observed = validate_retirement_declarations(changed)["passed"]
            else:
                observed = evaluate_contract(markdown_text, changed)["passed"]
        except (TypeError, ValueError, yaml.YAMLError) as exc:
            observed = False
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "mutation": name,
                "fixture_path": str(path),
                "baseline_passed": baseline_passed,
                "rejecting_check": check,
                "expected": "mutation_rejected",
                "observed_check_passed": observed,
                "error": error,
                "rejected": observed is False,
            }
        )
    return rows


def _archived_ids(complete_path: Path) -> set[str]:
    """Read archived task IDs without treating their outcomes as current work."""

    try:
        document = yaml.safe_load(complete_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return set()
    return {
        str(task.get("id"))
        for milestone in document.get("milestones", [])
        if isinstance(milestone, Mapping)
        for task in milestone.get("tasks", [])
        if isinstance(task, Mapping) and task.get("id")
    }


def check_fresh_ids(root: Path, roadmap: Mapping[str, Any]) -> JsonDict:
    """Require the exact unused experiment-number interval 7369 through 7380."""

    archived = _archived_ids(root / COMPLETE_PATH)
    tasks = roadmap.get("tasks", [])
    ids = [task.get("id") if isinstance(task, Mapping) else None for task in tasks]
    counts = Counter(ids)
    rows: list[JsonDict] = []
    for expected_number, expected_id in zip(EXPECTED_NUMBERS, EXPECTED_ID_ORDER, strict=True):
        observed = next(
            (
                task_id
                for task_id in ids
                if isinstance(task_id, str) and task_id.startswith(f"exp{expected_number}-")
            ),
            None,
        )
        passed = (
            observed == expected_id and counts[expected_id] == 1 and expected_id not in archived
        )
        rows.append(
            {
                "experiment_number": expected_number,
                "expected_id": expected_id,
                "observed_id": observed,
                "duplicate_count": counts[expected_id],
                "archived_before_v647": expected_id in archived,
                "passed": passed,
            }
        )
    return {
        "rows": rows,
        "passed": len(ids) == EXPECTED_TASK_COUNT and all(r["passed"] for r in rows),
    }


def _access_outcome(receipt: Mapping[str, Any]) -> str:
    """Classify one real access attempt without inferring unseen content."""

    if receipt.get("ok") is True:
        return "http_success"
    if receipt.get("status_code") == 429:
        return "http_429"
    if "challenge" in f"{receipt.get('body', '')} {receipt.get('error', '')}".lower():
        return "browser_challenge"
    return "access_failed"


def collect_source_method_rows(
    fetcher: Fetcher = _fetch_url,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Check the bounded V647 links sequentially and keep every outcome."""

    access_rows: list[JsonDict] = []
    total = len(ACCESS_SPECS)
    started = time.monotonic()
    for index, spec in enumerate(ACCESS_SPECS, 1):
        progress(
            "evaluate",
            "source_request_start",
            started,
            f"completed={index - 1}/{total} source={spec['access_id']}",
        )
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
        outcome = _access_outcome(receipt)
        access_rows.append(
            {
                **spec,
                "retrieval_date": RETRIEVAL_DATE,
                "timeout_s": 20,
                "status_code": receipt.get("status_code"),
                "error": receipt.get("error"),
                "response_sha256": sha256_bytes(body.encode("utf-8")),
                "access_outcome": outcome,
            }
        )
        progress(
            "evaluate",
            "source_request_end",
            started,
            f"completed={index}/{total} outcome={outcome}",
        )
    by_id = {row["access_id"]: row for row in access_rows}
    source_rows: list[JsonDict] = []
    for method in SOURCE_METHODS:
        access_ids = list(method["access_ids"])
        source_rows.append(
            {
                **deepcopy(method),
                "retrieval_date": RETRIEVAL_DATE,
                "primary_urls": [by_id[access_id]["url"] for access_id in access_ids],
                "publication_or_version_dates": [
                    by_id[access_id]["publication_or_version_date"] for access_id in access_ids
                ],
                "access_outcomes": {
                    access_id: by_id[access_id]["access_outcome"] for access_id in access_ids
                },
                "access_status_codes": {
                    access_id: by_id[access_id]["status_code"] for access_id in access_ids
                },
                "runtime_dependency_added": False,
                "frozen_roster_changed": False,
                "local_science_claimed": False,
            }
        )
    return source_rows, access_rows


def source_rows_complete(source_rows: object, access_rows: object) -> bool:
    """Require all method and access rows while allowing bounded access failure."""

    outcomes = {"http_success", "http_429", "browser_challenge", "access_failed"}
    return bool(
        isinstance(source_rows, list)
        and isinstance(access_rows, list)
        and len(source_rows) == len(SOURCE_METHODS)
        and len(access_rows) == len(ACCESS_SPECS)
        and {row.get("method_family") for row in source_rows}
        == {row["method_family"] for row in SOURCE_METHODS}
        and [row.get("access_id") for row in access_rows] == list(ACCESS_IDS)
        and all(row.get("access_outcome") in outcomes for row in access_rows)
        and all(
            row.get("primary_urls")
            and row.get("adopted_control_or_defer_reason")
            and row.get("evidence_limit")
            and row.get("runtime_dependency_added") is False
            and row.get("frozen_roster_changed") is False
            and row.get("local_science_claimed") is False
            for row in source_rows
        )
    )


def run_affected_validation(
    root: Path,
    authority: Path,
    raw_dir: Path,
    *,
    test_paths: Sequence[str] | None = None,
    changed_modules: Sequence[str] | None = None,
    static_paths: Sequence[str] | None = None,
    historical_observations: Sequence[Mapping[str, Any]] = (),
    roadmap_runner: Callable[..., list[JsonDict]] = run_commands,
    affected_runner: Callable[..., list[JsonDict]] = run_categorized_commands,
) -> JsonDict:
    """Validate the Exp7358 plan before executing it through Exp7303."""

    manifest = AffectedManifest(
        experiment_id=FIRST_TASK_ID,
        test_paths=tuple(test_paths or V647_MANIFEST.test_paths),
        changed_modules=tuple(changed_modules or V647_MANIFEST.changed_modules),
        static_paths=tuple(static_paths or V647_MANIFEST.static_paths),
    )
    private = Path(tempfile.mkdtemp(prefix="exp7369-validation-", dir="/tmp"))
    commands = build_command_plan(root, manifest, private)
    plan_errors = validate_command_plan(root, manifest, commands)
    if plan_errors:
        return {
            "validation_receipts": [],
            "required_checks_passed": False,
            "plan_errors": plan_errors,
            "repository_health": {
                "status": "plan_rejected_before_subprocess",
                "affects_required_checks": True,
                "observations": [deepcopy(dict(row)) for row in historical_observations],
            },
        }
    roadmap_receipts = roadmap_runner(
        root,
        roadmap_command_specs(root, authority),
        log_dir=raw_dir / "validation/roadmap",
    )
    affected_receipts = affected_runner(
        root,
        [PlannedCommand(command, "required_validation", True) for command in commands],
        log_dir=raw_dir / "validation/affected",
    )
    roadmap_failures = _failed_names(roadmap_receipts, ROADMAP_CHECK_NAMES)
    reduced = reduce_affected_receipts(root, manifest, affected_receipts)
    return {
        "validation_receipts": [*roadmap_receipts, *affected_receipts],
        "required_checks_passed": not roadmap_failures and reduced["passed"],
        "plan_errors": [],
        "repository_health": {
            "status": "historical_observations_retained",
            "affects_required_checks": False,
            "observations": [deepcopy(dict(row)) for row in historical_observations],
        },
    }


def run_terminal_validation(root: Path, candidate: Path, raw_dir: Path) -> list[JsonDict]:
    """Run cold reduction, adversarial verification, and strict row checks."""

    python = str(root / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib;"
        "from carnot.experiment_7369_v647_contract import validate_artifact;"
        f"v=json.loads(pathlib.Path({str(candidate)!r}).read_text());"
        "e=validate_artifact(v,require_terminal=False);"
        "print(e,flush=True);raise SystemExit(bool(e))"
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


def _path_label(path: Path, root: Path) -> str:
    """Use repository-relative labels while preserving private paths."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _hash_paths(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    """Hash each readable source and sidecar exactly once."""

    hashes: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        absolute = path if path.is_absolute() else root / path
        if absolute.is_file():
            hashes[_path_label(absolute, root)] = sha256(absolute)
    return hashes


def _historical_observations(root: Path) -> list[JsonDict]:
    """Keep the V646 contract defect as labeled history, never readiness."""

    try:
        value = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        {
            "observed_at": "2026-09-17",
            "producer_path": HISTORY_PATH.as_posix(),
            "producer_sha256": sha256(root / HISTORY_PATH),
            "producer_experiment_id": value.get("experiment_id"),
            "producer_milestone": value.get("milestone"),
            "producer_verdict_class": value.get("verdict_class"),
            "producer_flagged_adversarial": value.get("flagged_adversarial"),
            "producer_inference_substrate_class": value.get("inference_substrate_class"),
            "classification": "historical_v646_contract_defect_not_current_readiness",
            "affects_v647_required_checks": False,
        }
    ]


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Check exact authorities, producer identity, requirements, and resources."""

    authority, roadmap, yaml_bytes, candidates = select_yaml_authority(root)
    rows = list(candidates)
    rows.append(
        {
            "check": "yaml_authority",
            "path": authority.as_posix() if authority else None,
            "upstream": "research-roadmap-next.yaml|research-roadmap.yaml",
            "artifact_field": "milestone",
            "expected_value": MILESTONE,
            "observed_value": authority.as_posix() if authority else None,
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
                "path": relative.as_posix(),
                "upstream": relative.as_posix(),
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
            "path": SPEC_PATH.as_posix(),
            "upstream": SPEC_PATH.as_posix(),
            "artifact_field": "REQ-*",
            "expected_value": "REQ-REPORT-7369",
            "observed_value": "REQ-REPORT-7369" if "REQ-REPORT-7369" in spec_text else "missing",
            "available": "REQ-REPORT-7369" in spec_text,
            "blocking": True,
        }
    )
    history = _historical_observations(root)
    expected_history = {
        "producer_experiment_id": "exp7357-contract",
        "producer_milestone": "2026.09.646",
        "producer_verdict_class": "disqualified",
        "producer_inference_substrate_class": "aggregation",
    }
    observed_history = history[0] if history else {}
    for field, expected in expected_history.items():
        rows.append(
            {
                "check": f"historical_producer:{field}",
                "path": HISTORY_PATH.as_posix(),
                "upstream": HISTORY_PATH.as_posix(),
                "artifact_field": field,
                "expected_value": expected,
                "observed_value": observed_history.get(field),
                "available": observed_history.get(field) == expected,
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
        except OSError as exc:  # pragma: no cover - depends on host filesystem failure.
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
    """Record the five exact values needed to diagnose one failed gate."""

    return {
        "upstream": upstream,
        "failed_check": check,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def _terminal_state(artifact: JsonDict) -> None:
    """Derive completion and terminal class from stored evidence only."""

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
    score = int(
        failed_precondition is None
        and failed_gate is None
        and bool(artifact.get("acceptance_gate_results"))
    )
    artifact.update(
        contract_complete_score=score,
        science_value_score=0,
        promotion_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="aggregation",
        execution_venue="host",
    )
    if failed_precondition is not None:
        artifact.update(
            status="blocked",
            verdict_class="blocked",
            honest_verdict="blocked_v647_contract_required_input_missing",
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
            honest_verdict="complete_disqualified_v647_contract_or_validation_defect",
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
            honest_verdict="complete_circular_positive_v647_exact_advisory_contract",
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
    """Reduce contract, sources, controls, and executed checks separately."""

    mutation_rows = artifact.get("mutation_rows", [])
    gate_rows = artifact.get("gate_control_rows", [])
    gate_count = len(gate_rows) // len(GATE_CASES) if isinstance(gate_rows, list) else 0
    receipts = artifact.get("validation_receipts", [])
    required_names = (*ROADMAP_CHECK_NAMES, *REQUIRED_CHECK_NAMES)
    counts = Counter(row.get("name") for row in receipts)
    affected_passed = all(
        counts[name] == 1
        and next(row for row in receipts if row.get("name") == name).get("passed") is True
        for name in required_names
    )
    criteria: list[tuple[str, object, object, bool, str]] = [
        (
            "authority_selected",
            MILESTONE,
            artifact.get("yaml_milestone"),
            artifact.get("yaml_milestone") == MILESTONE,
            "Only an exact V647 roadmap can supply executable rows.",
        ),
        (
            "contract_exact",
            {"task_count": 12, "milestone": MILESTONE},
            {
                "task_count": len(artifact.get("contract_rows", [])),
                "markdown_milestone": artifact.get("markdown_milestone"),
                "yaml_milestone": artifact.get("yaml_milestone"),
            },
            artifact.get("contract_passed") is True,
            "Independent parsers must agree on every ordered task field and gate.",
        ),
        (
            "mutations_rejected",
            list(CONTRACT_MUTATION_NAMES),
            [row.get("mutation") for row in mutation_rows],
            len(mutation_rows) == len(CONTRACT_MUTATION_NAMES)
            and all(row.get("baseline_passed") and row.get("rejected") for row in mutation_rows),
            "Each required mutation must fail its named check from an exact baseline.",
        ),
        (
            "fresh_ids",
            list(EXPECTED_NUMBERS),
            [row.get("experiment_number") for row in artifact.get("fresh_id_rows", [])],
            artifact.get("fresh_ids_passed") is True,
            "V647 IDs must be ordered, unique, and absent from archived milestones.",
        ),
        (
            "gate_controls_complete",
            gate_count * len(GATE_CASES),
            len(gate_rows) if isinstance(gate_rows, list) else None,
            gate_count > 0 and gate_controls_complete(gate_rows, gate_count),
            "Missing, blocked, partial, disqualified, or quarantined science must fail closed.",
        ),
        (
            "source_method_rows_complete",
            len(SOURCE_METHODS),
            len(artifact.get("source_method_rows", [])),
            source_rows_complete(
                artifact.get("source_method_rows"), artifact.get("source_access_rows")
            ),
            "Access can fail while every bounded method choice remains explicit.",
        ),
        (
            "study_mapping_ingested",
            True,
            artifact.get("study_mapping_ingested"),
            artifact.get("study_mapping_ingested") is True,
            "The dated V647 mapping must exist in the durable study ledger.",
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
            "Historical model receipts cannot become current inference.",
        ),
        (
            "affected_validation",
            list(required_names),
            [row.get("name") for row in receipts if row.get("name") in required_names],
            affected_passed and artifact.get("required_checks_passed") is True,
            "The validated Exp7358 plan must run each required scoped check once.",
        ),
    ]
    if require_terminal:
        terminal_counts = Counter(row.get("name") for row in receipts)
        terminal_passed = all(
            terminal_counts[name] == 1
            and next(row for row in receipts if row.get("name") == name).get("passed") is True
            for name in TERMINAL_CHECK_NAMES
        )
        criteria.append(
            (
                "terminal_validation",
                list(TERMINAL_CHECK_NAMES),
                [row.get("name") for row in receipts if row.get("name") in TERMINAL_CHECK_NAMES],
                terminal_passed,
                "Cold reduction and both strict artifact readers must pass.",
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


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain every top-level field and retain the required exact principles."""

    principles = {
        field: "This field keeps one part of the V647 receipt auditable." for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def _base_artifact(run_date: str, started_at: str, checkpoint_path: Path) -> JsonDict:
    """Create a running checkpoint that cannot resemble terminal success."""

    return {
        "schema": "carnot.exp7369.v647_contract.v1",
        "experiment_id": FIRST_TASK_ID,
        "milestone": MILESTONE,
        "phase": 1,
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
        "host_work": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "contract_rows": {"planned": 12, "attempted": 0, "completed": 0, "censored": 0},
            "mutations": {"planned": 9, "attempted": 0, "completed": 0, "censored": 0},
            "gate_controls": {"planned": 0, "attempted": 0, "completed": 0, "censored": 0},
            "source_requests": {"planned": 10, "attempted": 0, "completed": 0, "censored": 0},
            "stopping_rule": "Run twelve comparisons, nine frozen mutations, seven controls per gate, ten bounded sequential source requests, and one exact validation plan.",
            "remaining_work": "No extension from observed outcomes.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _failure_summary(
            FIRST_TASK_ID, "work_in_progress", "status", "terminal", "running"
        ),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_running_v647_contract",
        "verdict_class": "partial",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {},
        "field_principles": {},
        "promotion_score": 0,
        "contract_complete_score": 0,
        "source_method_rows": [],
        "mutation_rows": [],
        "science_value_score": 0,
        "contract_rows": [],
        "source_access_rows": [],
        "gate_control_rows": [],
        "fresh_id_rows": [],
        "raw_authority_rows": [],
        "historical_inference_sidecars": [],
        "study_mapping_ingested": False,
        "semantic_scholar_access_outcomes": [],
        "dated_delta_access_outcomes": [deepcopy(dict(row)) for row in DATED_DELTA_ACCESS_OUTCOMES],
        "required_checks_passed": False,
        "checkpoint_path": str(checkpoint_path),
        "production_defaults_changed": False,
        "active_roadmap_changed": False,
        "research_conductor_changed": False,
    }


def _phase_span(
    name: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:
    """Measure one phase with its completed units and checkpoint boundary."""

    ended = time.monotonic()
    return {
        "phase": name,
        "start_elapsed_s": phase_started - run_started,
        "end_elapsed_s": ended - run_started,
        "duration_s": ended - phase_started,
        "completed_units": units,
        "checkpoint_utc": datetime.now(UTC).isoformat(),
        "checkpoint_boundary": checkpoint,
        "pending_operations": [],
    }


def _checkpoint(artifact: JsonDict, checkpoint_path: Path) -> None:
    """Persist completed work while the checkpoint remains visibly nonterminal."""

    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(checkpoint_path, artifact)


def _finalize(artifact: JsonDict, started: float) -> None:
    """Seal real timestamps, explanations, and the reproducibility checksum."""

    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def independent_reduce(artifact: Mapping[str, Any], *, require_terminal: bool = True) -> list[str]:
    """Recompute raw receipt structure without trusting completion scores."""

    errors: list[str] = []
    rows = artifact.get("contract_rows")
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_TASK_COUNT
        or [row.get("unit_id") for row in rows] != list(EXPECTED_ID_ORDER)
        or any(
            not isinstance(row.get("checks"), Mapping)
            or row.get("passed") != all(row["checks"].values())
            or row.get("censored") is not False
            for row in rows
        )
    ):
        errors.append("contract_rows_invalid")
    mutations = artifact.get("mutation_rows")
    if (
        not isinstance(mutations, list)
        or [row.get("mutation") for row in mutations] != list(CONTRACT_MUTATION_NAMES)
        or any(
            row.get("baseline_passed") is not True or row.get("rejected") is not True
            for row in mutations
        )
    ):
        errors.append("mutation_rows_invalid")
    if not source_rows_complete(
        artifact.get("source_method_rows"), artifact.get("source_access_rows")
    ):
        errors.append("source_method_rows_invalid")
    access_rows = artifact.get("source_access_rows")
    if not isinstance(access_rows, list) or [row.get("access_id") for row in access_rows] != list(
        ACCESS_IDS
    ):
        errors.append("access_rows_invalid")
    gate_rows = artifact.get("gate_control_rows")
    gate_count = len(gate_rows) // len(GATE_CASES) if isinstance(gate_rows, list) else 0
    if gate_count == 0 or not gate_controls_complete(gate_rows, gate_count):
        errors.append("gate_controls_invalid")
    fresh = artifact.get("fresh_id_rows")
    if (
        not isinstance(fresh, list)
        or [row.get("experiment_number") for row in fresh] != list(EXPECTED_NUMBERS)
        or any(row.get("passed") is not True for row in fresh)
    ):
        errors.append("fresh_ids_invalid")
    raw = artifact.get("raw_authority_rows")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(row.get("source_sha256") != row.get("raw_sha256") for row in raw)
    ):
        errors.append("raw_authorities_invalid")
    if artifact.get("rows") != rows:
        errors.append("rows_invalid")
    spans = artifact.get("phase_spans")
    if not isinstance(spans, list) or any(
        row.get("start_elapsed_s", 0) > row.get("end_elapsed_s", -1)
        or (index and row.get("start_elapsed_s", 0) < spans[index - 1].get("end_elapsed_s", 0))
        for index, row in enumerate(spans)
    ):
        errors.append("phase_spans_invalid")
    required = (
        ALL_REQUIRED_CHECK_NAMES
        if require_terminal
        else (
            *ROADMAP_CHECK_NAMES,
            *REQUIRED_CHECK_NAMES,
        )
    )
    receipt_counts = Counter(row.get("name") for row in artifact.get("validation_receipts", []))
    if any(receipt_counts[name] != 1 for name in required):
        errors.append("validation_receipts_invalid")
    return errors


def validate_artifact(
    artifact: object,
    *,
    root: Path = REPO_ROOT,
    require_terminal: bool = True,
) -> list[str]:
    """Cold-check identity, hashes, raw rows, terminal state, and checksum."""

    if not isinstance(artifact, Mapping):
        return ["artifact_mapping_required"]
    missing = sorted(field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact)
    if missing:
        return [f"missing_required_field:{missing[0]}"]
    errors: list[str] = []
    if (
        artifact.get("schema") != "carnot.exp7369.v647_contract.v1"
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
        errors.extend(independent_reduce(artifact, require_terminal=require_terminal))
    for label, expected_hash in artifact.get("source_artifact_hashes", {}).items():
        path = Path(str(label))
        absolute = path if path.is_absolute() else root / path
        if not absolute.is_file() or sha256(absolute) != expected_hash:
            errors.append("source_hash_mismatch")
            break
    expected = deepcopy(dict(artifact))
    _terminal_state(expected)
    for field in (
        "status",
        "contract_complete_score",
        "verdict_class",
        "honest_verdict",
        "gate_check_summary",
        "science_value_score",
        "promotion_score",
    ):
        if artifact.get(field) != expected.get(field):
            errors.append(f"{field}_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_invalid")
    return list(dict.fromkeys(errors))


def build_artifact(  # noqa: C901 - the phases mirror the experiment protocol.
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
    """Measure, validate, and atomically publish the V647 contract receipt."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress("checkpoint", "start", started, "write nonterminal checkpoint")
    artifact = _base_artifact(run_date, started_at, checkpoint_path)
    _checkpoint(artifact, checkpoint_path)

    phase_started = time.monotonic()
    progress("read", "start", started, "check exact inputs and select V647 authority")
    preconditions, authority, selected_roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["phase_spans"].append(
        _phase_span("read", phase_started, started, len(preconditions), "inputs_checked")
    )
    if any(row.get("blocking") and row.get("available") is not True for row in preconditions):
        phase_started = time.monotonic()
        artifact["acceptance_gate_results"] = []
        _terminal_state(artifact)
        artifact["phase_spans"].append(
            _phase_span("write", phase_started, started, 1, "blocked_terminal_output")
        )
        _finalize(artifact, started)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_path, artifact)
        progress("write", "end", started, f"blocked output={output_path}")
        return artifact
    assert authority is not None and selected_roadmap is not None and yaml_bytes is not None
    _checkpoint(artifact, checkpoint_path)

    phase_started = time.monotonic()
    progress("build", "start", started, "freeze authorities and build controls")
    raw_dir.mkdir(parents=True, exist_ok=True)
    design_path = root / str(selected_roadmap.get("milestone_doc", DESIGN_PATH))
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
            "source_path": authority.as_posix(),
            "raw_path": _path_label(yaml_copy, root),
            "source_sha256": sha256_bytes(yaml_bytes),
            "raw_sha256": sha256(yaml_copy),
            "hash_matches": sha256_bytes(yaml_bytes) == sha256(yaml_copy),
        },
    ]
    roadmap = yaml.safe_load(yaml_copy.read_text(encoding="utf-8"))
    contract = evaluate_contract(design_copy.read_text(encoding="utf-8"), roadmap)
    artifact.update(contract)
    artifact["contract_passed"] = contract["passed"]
    with tempfile.TemporaryDirectory(prefix="exp7369-mutations-", dir="/tmp") as directory:
        artifact["mutation_rows"] = run_contract_mutations(
            design_copy.read_text(encoding="utf-8"), roadmap, Path(directory)
        )
    freshness = check_fresh_ids(root, roadmap)
    artifact["fresh_id_rows"] = freshness["rows"]
    artifact["fresh_ids_passed"] = freshness["passed"]
    artifact["rows"] = artifact["contract_rows"]
    artifact["sample_size_budget"]["contract_rows"].update(attempted=12, completed=12)
    artifact["sample_size_budget"]["mutations"].update(attempted=9, completed=9)

    progress("build", "gate_controls_start", started, "exercise real conductor gate reader")
    gate_count = sum(len(task.get("gated_on") or []) for task in roadmap["tasks"])
    artifact["gate_control_rows"] = run_gate_controls(roadmap, raw_dir / "gate-controls")
    gate_units = gate_count * len(GATE_CASES)
    artifact["sample_size_budget"]["gate_controls"].update(
        planned=gate_units, attempted=gate_units, completed=gate_units
    )
    progress("build", "gate_controls_end", started, f"completed={gate_units}/{gate_units}")
    artifact["phase_spans"].append(
        _phase_span(
            "build",
            phase_started,
            started,
            EXPECTED_TASK_COUNT + len(CONTRACT_MUTATION_NAMES) + gate_units,
            "contract_controls_complete",
        )
    )
    _checkpoint(artifact, checkpoint_path)

    phase_started = time.monotonic()
    progress("load", "before_model_load", started, "MODEL_SPECS=[] no load attempted")
    artifact["phase_spans"].append(_phase_span("load", phase_started, started, 0, "no_model_load"))
    progress("load", "after_model_load", started, "attempted=0 completed=0")

    phase_started = time.monotonic()
    progress("generate", "before_generation", started, "model_invoked=false")
    artifact["phase_spans"].append(
        _phase_span("generate", phase_started, started, 0, "no_model_generation")
    )
    progress("generate", "after_generation", started, "attempted=0 completed=0")

    phase_started = time.monotonic()
    progress("evaluate", "start", started, "refresh bounded source delta sequentially")
    source_rows, access_rows = collect_source_method_rows(fetcher)
    artifact["source_method_rows"] = source_rows
    artifact["source_access_rows"] = access_rows
    artifact["semantic_scholar_access_outcomes"] = [
        row["access_outcome"]
        for row in access_rows
        if str(row["access_id"]).startswith("semantic_scholar_")
    ]
    artifact["dated_delta_access_outcomes"] = [
        deepcopy(dict(row)) for row in DATED_DELTA_ACCESS_OUTCOMES
    ]
    artifact["study_mapping_ingested"] = "EXP7369-V647-SOURCE-INGESTION-20260917" in (
        root / STUDY_PATH
    ).read_text(encoding="utf-8")
    success_count = sum(row["access_outcome"] == "http_success" for row in access_rows)
    artifact["sample_size_budget"]["source_requests"].update(
        attempted=len(access_rows),
        completed=success_count,
        censored=len(access_rows) - success_count,
    )
    access_sidecar = raw_dir / "source-access-receipts.json"
    _atomic_write_json(
        access_sidecar,
        {"schema": "carnot.exp7369.source_access.v1", "receipts": access_rows},
    )
    history = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    history_sidecar = raw_dir / "historical-model-receipts.json"
    _atomic_write_json(
        history_sidecar,
        {
            "schema": "carnot.exp7369.historical_models.v1",
            "current_invocation": {
                "MODEL_SPECS": [],
                "model_invoked": False,
                "invocation_counts": ZERO_INVOCATION_COUNTS,
            },
            "historical_artifact": {
                "path": HISTORY_PATH.as_posix(),
                "sha256": sha256(root / HISTORY_PATH),
                "MODEL_SPECS": history.get("MODEL_SPECS"),
                "model_invoked": history.get("model_invoked"),
                "invocation_counts": history.get("invocation_counts"),
                "verdict_class": history.get("verdict_class"),
                "flagged_adversarial": history.get("flagged_adversarial"),
                "accepted_as_current_inference": False,
            },
        },
    )
    artifact["historical_inference_sidecars"] = [
        {"path": _path_label(history_sidecar, root), "sha256": sha256(history_sidecar)}
    ]
    artifact["source_access_sidecar"] = {
        "path": _path_label(access_sidecar, root),
        "sha256": sha256(access_sidecar),
    }
    artifact["phase_spans"].append(
        _phase_span("evaluate", phase_started, started, len(access_rows), "source_receipts_written")
    )
    _checkpoint(artifact, checkpoint_path)

    phase_started = time.monotonic()
    progress("validate", "before_affected_subprocesses", started, "validated Exp7358 plan")
    validation = validation_runner(
        root=root,
        authority=authority,
        raw_dir=raw_dir,
        test_paths=[TEST_PATH.as_posix()],
        changed_modules=[MODULE_PATH.as_posix()],
        static_paths=[WRAPPER_PATH.as_posix()],
        historical_observations=_historical_observations(root),
    )
    artifact["validation_receipts"] = list(validation.get("validation_receipts", []))
    artifact["required_checks_passed"] = validation.get("required_checks_passed") is True
    artifact["validation_plan_errors"] = list(validation.get("plan_errors", []))
    artifact["repository_health"] = deepcopy(validation.get("repository_health", {}))
    artifact["source_artifact_hashes"] = _hash_paths(
        root,
        [
            *INPUT_PATHS,
            authority,
            design_path,
            design_copy,
            yaml_copy,
            access_sidecar,
            history_sidecar,
        ],
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(artifact, require_terminal=False)
    _terminal_state(artifact)
    _finalize(artifact, started)
    candidate = raw_dir / "measured-terminal-candidate.json"
    _atomic_write_json(candidate, artifact)
    progress("validate", "after_affected_subprocesses", started, f"candidate={candidate}")

    progress("validate", "before_terminal_subprocesses", started, "cold replay and strict readers")
    terminal_receipts = terminal_runner(root, candidate, raw_dir)
    artifact["validation_receipts"].extend(terminal_receipts)
    adversarial = next((row for row in terminal_receipts if row.get("name") == "adversarial"), {})
    artifact["flagged_adversarial"] = bool(
        adversarial.get("exit_code") == 2 or "CRITICAL" in str(adversarial.get("output_tail") or "")
    )
    artifact["required_checks_passed"] = not _failed_names(
        artifact["validation_receipts"], ALL_REQUIRED_CHECK_NAMES
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(artifact, require_terminal=True)
    _terminal_state(artifact)
    artifact["phase_spans"].append(
        _phase_span(
            "validate",
            phase_started,
            started,
            len(artifact["validation_receipts"]),
            "terminal_checks_complete",
        )
    )
    _finalize(artifact, started)
    progress(
        "validate",
        "after_terminal_subprocesses",
        started,
        f"passed={artifact['required_checks_passed']}",
    )

    phase_started = time.monotonic()
    progress("write", "before_atomic_terminal", started, f"output={output_path}")
    artifact["phase_spans"].append(
        _phase_span("write", phase_started, started, 1, "terminal_output")
    )
    _finalize(artifact, started)
    errors = validate_artifact(artifact, root=root, require_terminal=True)
    if errors:  # pragma: no cover - tests validate the same object before E2E publication.
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(candidate, artifact)
    _atomic_write_json(output_path, artifact)
    progress("write", "after_atomic_terminal", started, f"verdict={artifact['honest_verdict']}")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the fixed V647 execution date."""

    datetime.strptime(value, "%Y%m%d")
    if value != RUN_DATE:
        raise argparse.ArgumentTypeError(f"run date must be {RUN_DATE}")
    return value


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - entrypoint E2E.
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
        REPO_ROOT,
        args.date,
        output_path=output,
        raw_dir=raw_dir,
        checkpoint_path=checkpoint,
    )
    errors = validate_artifact(artifact, root=REPO_ROOT, require_terminal=True)
    if errors:
        print(f"[exp7369] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7369] complete verdict={artifact['honest_verdict']} score={artifact['contract_complete_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

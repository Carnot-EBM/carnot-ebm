"""Build the V648 advisory source and exact fourteen-task contract receipt.

The task compares two planning authorities and preserves V647 evidence. It
invokes no language model and creates no science result.

Spec refs: REQ-REPORT-7381 and SCENARIO-REPORT-7381-*.
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
MILESTONE = "2026.09.648"
RUN_DATE = "20260918"
RETRIEVAL_DATE = "2026-09-18"
FIRST_TASK_ID = "exp7381-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7382-decision-protocol",
    "exp7383-canary-reducer",
    "exp7384-arc-invocation-boundary",
    "exp7385-decision-training",
    "exp7386-online-decisions",
    "exp7387-decision-audit",
    "exp7388-proposal-capture",
    "exp7389-proof-learning",
    "exp7390-proof-audit",
    "exp7391-arc-generalization",
    "exp7392-ising-reduction",
    "exp7393-hardware-placement",
    "exp7394-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
EXPECTED_NUMBERS = tuple(range(7381, 7395))
RANDOM_SEED = {
    "experiment": 7_381_202_609_18,
    "mutation_controls": 7_381_202_609_19,
    "resampling": 7_381_202_609_20,
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

MODULE_PATH = Path("python/carnot/experiment_7381_v648_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7381_v648_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7381_v648_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7381_v648_contract.json")
RAW_DIR = Path("results/raw/experiment_7381_v648_contract")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7381_v648_contract.json")
ADVERSARIAL_PATH = Path("scripts/adversarial_verify.py")
ROW_LINT_PATH = Path("scripts/verdict_row_consistency_lint.py")

V647_PRESERVED_DESIGN = Path(
    "openspec/change-proposals/research-roadmap-v647-preserved-20260918.md"
)
V647_RAW_DESIGN = Path("results/raw/experiment_7369_v647_contract/selected-contract.md")
V647_CAPSTONE_PATH = Path("results/experiment_7380_v647_capstone.json")
V647_PRODUCERS: tuple[tuple[int, Path, str, str], ...] = (
    (
        7372,
        Path("results/experiment_7372_v647_qwen_canary.json"),
        "exp7372-v647-qwen-canary",
        "model_bounded_generation",
    ),
    (
        7376,
        Path("results/experiment_7376_v647_arc_outcomes.json"),
        "exp7376-arc-outcomes",
        "model_load_no_generation",
    ),
    (
        7378,
        Path("results/experiment_7378_v647_ising_audit.json"),
        "exp7378-v647-ising-audit",
        "cpu_exact_solver_or_simulator",
    ),
)

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
    V647_PRESERVED_DESIGN,
    V647_RAW_DESIGN,
    V647_CAPSTONE_PATH,
    *(row[1] for row in V647_PRODUCERS),
    Path("scripts/experiment_template.py"),
    Path("scripts/roadmap_schema.py"),
    Path("scripts/audit_roadmap_gates.py"),
    Path("scripts/validate_prior_failures.py"),
    Path("scripts/exclusion_manifest_lint.py"),
    ADVERSARIAL_PATH,
    ROW_LINT_PATH,
    Path("python/carnot/reporting/experiment_7303_validation_scope.py"),
    Path("python/carnot/experiment_7358_v646_validation_contract.py"),
    Path("python/carnot/experiment_7369_v647_contract.py"),
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
    "missing_task",
    "reordered_task",
    "wrong_milestone",
    "wrong_title",
    "wrong_path",
    "wrong_phase",
    "wrong_substrate",
    "missing_gate_artifact_field",
    "missing_producer_field",
    "missing_retirement_flag",
)

V648_MANIFEST = AffectedManifest(
    experiment_id=FIRST_TASK_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Versioned schema with ordinary top-level experiment_id and milestone.",
    "status": "Terminal only after actual work and required validation; no success-shaped bootstrap artifact.",
    "run_date": "Use 20260918 plus actual start/end UTC timestamps.",
    "preconditions_checked": "Exact paths, producer identity/hash/class and resource checks before dependent work.",
    "MODEL_SPECS": "Include unsloth/Qwen3.8-27B-GGUF for current LLM work; [] for no current LLM work. Record small Gibbs training separately.",
    "model_invoked": "True for any attempted current LLM load or generation, even a failed attempt; distinguish small EBM training.",
    "invocation_counts": "Attempted/completed/failed/cancelled/in-flight current LLM loads and generations, with raw receipts; no historical calls counted as current.",
    "inference_substrate": "Actual CPU/JAX or owned native CUDA computation, device identity and resource lease; historical provenance stays labeled.",
    "inference_substrate_class": "Closed class matching the actual current computation and authentic duration; no padding.",
    "execution_venue": "Exactly host in V648. This is a closed string, never host_cpu, a device name or a dictionary; put device details in inference_substrate.",
    "duration_s": "Measured monotonic duration, never invented time or sleep to meet a floor.",
    "phase_spans": "Measured read/build/load/generate/evaluate/validate/write spans and checkpoint timestamps.",
    "random_seed": "Frozen experiment and resampling seeds; null only when not applicable.",
    "reproducibility_checksum": "Bind exact code, settings, protocol, data/source versions and raw rows.",
    "source_artifact_hashes": "Exact producer paths and byte hashes; preserve original verdicts and flags.",
    "rows": "Every unit/arm/seed outcome, metric, cost, failure and censoring disposition behind comparative claims.",
    "sample_size_budget": "Predeclared planned/attempted/completed/censored/unstarted units, stopping rules and remaining work.",
    "acceptance_gate_results": "Expected/observed/operator/passed separated for required validation, safety, completion and scientific efficacy; no equality substituted for >=.",
    "gate_check_summary": "Every blocked_* names the failed check/upstream, exact field/path, expected value and observed value, including missing or None.",
    "verifier_is_oracle": "True when formal evaluator defines truth; independently coded oracle checks do not remove circularity.",
    "honest_verdict": "Completed work uses complete_ or complete: with precise scope; blocked_* names an unavailable prerequisite.",
    "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial. Oracle-positive claims are circular_positive. Only retryable unfinished OWN work is partial; unchanged external absence is blocked.",
    "flagged_adversarial": "True for a critical independent finding; excludes producer from readiness gates.",
    "validation_receipts": "Executed argv/environment/scope/return code/duration/log hash, including failures and exact required-check list.",
    "repository_health": "Dated unrelated failures distinct from required affected checks; never hide affected failures.",
    "field_principles": "Explain every output field without wrapping ordinary dictionaries or numeric gate values.",
    "promotion_score": "Always zero: no automatic production rollout, generator weight change or external publication.",
    "contract_complete_score": "One only when both authorities and all relevant checks agree; advisory, never a science gate.",
    "science_value_score": "Always zero for contract work.",
    "source_method_rows": "Primary URL, checked date, adopted control or defer reason, and access status.",
    "mutation_rows": "Private changed contract and actual rejecting check.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

SOURCE_DELTA_URL = "https://arxiv.org/abs/2608.21748"
SOURCE_METHODS: tuple[JsonDict, ...] = (
    {
        "method_family": "ship",
        "primary_url": SOURCE_DELTA_URL,
        "adopted_control_or_defer_reason": "Freeze a finite typed-policy grid, calibrate the deployed policy, and issue no certificate for an empty selected group.",
        "destination_experiment_ids": ["exp7382-decision-protocol", "exp7385-decision-training"],
        "evidence_limit": "The source image domain is not the evaluated Carnot scope.",
    },
    {
        "method_family": "calarena",
        "primary_url": "https://arxiv.org/abs/2605.30188",
        "adopted_control_or_defer_reason": "Use proper scores with prevalence and logistic calibration controls.",
        "destination_experiment_ids": ["exp7382-decision-protocol", "exp7387-decision-audit"],
        "evidence_limit": "External calibration results do not establish local decision value.",
    },
    {
        "method_family": "calvert",
        "primary_url": "https://arxiv.org/abs/2606.21777",
        "adopted_control_or_defer_reason": "Keep confidence, evidence grounding, and typed action selection separate.",
        "destination_experiment_ids": ["exp7382-decision-protocol", "exp7391-arc-generalization"],
        "evidence_limit": "A hallucination detector does not define ARC correctness.",
    },
    {
        "method_family": "online_calibration",
        "primary_url": "https://arxiv.org/abs/2504.09096",
        "adopted_control_or_defer_reason": "Predict before feedback and compare delayed updates with frozen, no-feedback, recent-frequency, and online-logistic controls.",
        "destination_experiment_ids": ["exp7386-online-decisions", "exp7387-decision-audit"],
        "evidence_limit": "An IID coverage theorem does not transfer to constructed drift.",
    },
    {
        "method_family": "thermodynamic_learning",
        "primary_url": "https://arxiv.org/abs/2609.04732",
        "adopted_control_or_defer_reason": "Separate learned couplings from source truth, charge update cost, and defer new hardware training.",
        "destination_experiment_ids": ["exp7392-ising-reduction", "exp7393-hardware-placement"],
        "evidence_limit": "Vendor or paper hardware results are not local Carnot measurements.",
    },
)


def progress(phase: str, event: str, started: float, detail: str = "") -> None:
    """Print a flushed phase boundary with measured monotonic elapsed time."""

    suffix = f" {detail}" if detail else ""
    print(
        f"[exp7381] phase={phase} event={event} elapsed_s={time.monotonic() - started:.3f}{suffix}",
        flush=True,
    )


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Prefer staged YAML only when it is V648, then try active YAML."""

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
    """Parse one active table and reject a second active contract section."""

    active = re.split(r"(?m)^## Historical ", text, maxsplit=1)[0]
    count = len(re.findall(r"(?im)^## Exact Task Contract\s*$", active))
    if count != 1:
        raise ValueError(f"expected exactly one active Exact Task Contract, observed {count}")
    parsed = parse_markdown_contract(active)
    parsed["active_exact_contract_count"] = count
    return parsed


def _retired_ids(path: Path) -> set[str]:
    """Read numeric retired IDs so dependency checks reject dead chains."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):  # pragma: no cover - preconditions block it.
        return set()
    ids: set[str] = set()
    for section in ("retired", "retired_experiments", "retired_extras"):
        for row in document.get(section, []):
            if not isinstance(row, Mapping):
                continue
            values = [row.get("experiment_id"), *(row.get("experiment_ids") or [])]
            for value in values:
                text = str(value or "")
                match = re.search(r"(?:exp)?(\d+)", text)
                if match:
                    ids.add(f"exp{match.group(1)}")
    return ids


def validate_gate_declarations(roadmap: Mapping[str, Any]) -> JsonDict:
    """Require declared earlier producers and reject retired dependencies."""

    tasks = roadmap.get("tasks", [])
    by_id = {
        task.get("id"): (index, task)
        for index, task in enumerate(tasks)
        if isinstance(task, Mapping)
    }
    retired = _retired_ids(REPO_ROOT / EXCLUSION_PATH)
    rows: list[JsonDict] = []
    for consumer_index, consumer in enumerate(tasks):
        if not isinstance(consumer, Mapping):
            continue
        for gate in consumer.get("gated_on") or []:
            upstream = gate.get("upstream")
            producer_entry = by_id.get(upstream)
            producer_index = producer_entry[0] if producer_entry else None
            producer = producer_entry[1] if producer_entry else {}
            field = gate.get("artifact_field")
            declared = _field_declared(str(producer.get("prompt", "")), field)
            precedes = producer_index is not None and producer_index < consumer_index
            numeric = re.match(r"exp\d+", str(upstream or ""))
            retired_dependency = bool(numeric and numeric.group(0) in retired)
            rows.append(
                {
                    "consumer": consumer.get("id"),
                    "upstream": upstream,
                    "artifact_field": field,
                    "producer_precedes_consumer": precedes,
                    "declared_verbatim": declared,
                    "receipt_is_not_upstream": upstream != FIRST_TASK_ID,
                    "retired_dependency": retired_dependency,
                    "passed": bool(
                        precedes
                        and declared
                        and upstream != FIRST_TASK_ID
                        and not retired_dependency
                    ),
                }
            )
    return {"rows": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def validate_retirement_declarations(roadmap: Mapping[str, Any]) -> JsonDict:
    """Require all four prior-failure fields and an active retirement trigger."""

    required = ("experiment_id", "verdict", "addressed_by", "retire_if_same_verdict")
    rows: list[JsonDict] = []
    for task in roadmap.get("tasks", []):
        if not isinstance(task, Mapping):
            rows.append(
                {
                    "task_id": None,
                    "prior_index": None,
                    "missing_fields": list(required),
                    "passed": False,
                }
            )
            continue
        for index, prior in enumerate(task.get("prior_failures") or []):
            missing = [
                field
                for field in required
                if not isinstance(prior, Mapping)
                or field not in prior
                or prior.get(field) in (None, "")
            ]
            passed = not missing and prior.get("retire_if_same_verdict") is True
            rows.append(
                {
                    "task_id": task.get("id"),
                    "prior_index": index,
                    "missing_fields": missing,
                    "retire_if_same_verdict": (
                        prior.get("retire_if_same_verdict") if isinstance(prior, Mapping) else None
                    ),
                    "passed": passed,
                }
            )
    return {"rows": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare every V648 row field from independently parsed authorities."""

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
    """Apply each private V648 defect and record the rejecting check."""

    def missing_task(document: JsonDict) -> None:
        document["tasks"] = document["tasks"][:-1]

    def reordered_task(document: JsonDict) -> None:
        document["tasks"][0], document["tasks"][1] = document["tasks"][1], document["tasks"][0]

    def wrong_milestone(document: JsonDict) -> None:
        document["milestone"] = "2026.09.647"

    def wrong_title(document: JsonDict) -> None:
        document["tasks"][0]["title"] += " changed"

    def wrong_path(document: JsonDict) -> None:
        document["tasks"][0]["deliverable"] = "results/mutated.json"

    def wrong_phase(document: JsonDict) -> None:
        document["tasks"][0]["phase"] = 4

    def wrong_substrate(document: JsonDict) -> None:
        prompt = document["tasks"][0]["prompt"]
        document["tasks"][0]["prompt"] = prompt.replace(
            "inference_substrate_class=aggregation",
            "inference_substrate_class=cpu_exact_solver_or_simulator",
            1,
        )

    def missing_gate_artifact_field(document: JsonDict) -> None:
        consumer = next(task for task in document["tasks"] if task.get("gated_on"))
        consumer["gated_on"][0].pop("artifact_field")

    def missing_producer_field(document: JsonDict) -> None:
        consumer = next(task for task in document["tasks"] if task.get("gated_on"))
        gate = consumer["gated_on"][0]
        producer = next(task for task in document["tasks"] if task["id"] == gate["upstream"])
        producer["prompt"] = producer["prompt"].replace(
            f"- {gate['artifact_field']}:", f"- removed_{gate['artifact_field']}:", 1
        )

    def missing_retirement_flag(document: JsonDict) -> None:
        task = next(task for task in document["tasks"] if task.get("prior_failures"))
        task["prior_failures"][0].pop("retire_if_same_verdict")

    mutations: tuple[tuple[str, Callable[[JsonDict], None], str], ...] = (
        ("missing_task", missing_task, "exact_contract"),
        ("reordered_task", reordered_task, "exact_contract"),
        ("wrong_milestone", wrong_milestone, "exact_contract"),
        ("wrong_title", wrong_title, "exact_contract"),
        ("wrong_path", wrong_path, "exact_contract"),
        ("wrong_phase", wrong_phase, "exact_contract"),
        ("wrong_substrate", wrong_substrate, "exact_contract"),
        ("missing_gate_artifact_field", missing_gate_artifact_field, "roadmap_schema"),
        ("missing_producer_field", missing_producer_field, "gate_declaration"),
        ("missing_retirement_flag", missing_retirement_flag, "retirement_contract"),
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
                observed = True  # pragma: no cover - required gate fields reject.
            elif check == "gate_declaration":
                observed = validate_gate_declarations(changed)["passed"]
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
                "fixture_sha256": sha256(path),
                "baseline_passed": baseline_passed,
                "rejecting_check": check,
                "expected": "mutation_rejected",
                "observed_check_passed": observed,
                "error": error,
                "rejected": observed is False,
            }
        )
    return rows


def _archived_ids(path: Path) -> set[str]:
    """Read archived task IDs without promoting their outcomes."""

    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
    """Require the ordered unused experiment interval 7381 through 7394."""

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
                "archived_before_v648": expected_id in archived,
                "passed": passed,
            }
        )
    return {
        "rows": rows,
        "passed": len(ids) == EXPECTED_TASK_COUNT and all(r["passed"] for r in rows),
    }


def _access_outcome(receipt: Mapping[str, Any]) -> str:
    """Classify a bounded request without inferring unavailable content."""

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
    """Make one bounded source-delta request and preserve all method rows."""

    started = time.monotonic()
    progress("evaluate", "source_request_start", started, "completed=0/1 source=ship")
    try:
        receipt = dict(fetcher(SOURCE_DELTA_URL))
    except Exception as exc:  # noqa: BLE001 - an access failure is an evidence row.
        receipt = {
            "ok": False,
            "status_code": None,
            "body": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    body = str(receipt.get("body", ""))
    outcome = _access_outcome(receipt)
    access_rows = [
        {
            "access_id": "ship_delta",
            "url": SOURCE_DELTA_URL,
            "retrieval_date": RETRIEVAL_DATE,
            "timeout_s": 20,
            "status_code": receipt.get("status_code"),
            "error": receipt.get("error"),
            "response_sha256": sha256_bytes(body.encode("utf-8")),
            "access_outcome": outcome,
            "inferred_result": None,
        }
    ]
    progress("evaluate", "source_request_end", started, f"completed=1/1 outcome={outcome}")
    source_rows: list[JsonDict] = []
    for method in SOURCE_METHODS:
        source_rows.append(
            {
                **deepcopy(method),
                "checked_date": RETRIEVAL_DATE,
                "access_status": (
                    outcome
                    if method["method_family"] == "ship"
                    else "planning_review_authenticated"
                ),
                "source_delta_requested": method["method_family"] == "ship",
                "frozen_roster_changed": False,
                "runtime_dependency_added": False,
                "local_science_claimed": False,
                "full_citation_census_claimed": False,
            }
        )
    return source_rows, access_rows


def source_rows_complete(source_rows: object, access_rows: object) -> bool:
    """Require five method rows while allowing the one request to fail."""

    outcomes = {"http_success", "http_429", "browser_challenge", "access_failed"}
    return bool(
        isinstance(source_rows, list)
        and isinstance(access_rows, list)
        and len(source_rows) == len(SOURCE_METHODS)
        and len(access_rows) == 1
        and access_rows[0].get("access_outcome") in outcomes
        and {row.get("method_family") for row in source_rows}
        == {row["method_family"] for row in SOURCE_METHODS}
        and all(
            row.get("primary_url")
            and row.get("checked_date") == RETRIEVAL_DATE
            and row.get("access_status")
            and row.get("adopted_control_or_defer_reason")
            and row.get("evidence_limit")
            and row.get("frozen_roster_changed") is False
            and row.get("runtime_dependency_added") is False
            and row.get("local_science_claimed") is False
            and row.get("full_citation_census_claimed") is False
            for row in source_rows
        )
    )


def _load_json(path: Path) -> JsonDict:
    """Load one JSON mapping or return an empty mapping for unavailable bytes."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def collect_historical_boundary(root: Path) -> JsonDict:
    """Authenticate preserved V647 design and three disqualified producers."""

    preserved = root / V647_PRESERVED_DESIGN
    raw = root / V647_RAW_DESIGN
    preserved_hash = sha256(preserved) if preserved.is_file() else None
    raw_hash = sha256(raw) if raw.is_file() else None
    capstone = _load_json(root / V647_CAPSTONE_PATH)
    capstone_hashes = capstone.get("source_artifact_hashes", {})
    ledger = (
        yaml.safe_load((root / COMPLETE_PATH).read_text(encoding="utf-8"))
        if (root / COMPLETE_PATH).is_file()
        else {}
    )
    milestones = ledger.get("milestones", []) if isinstance(ledger, Mapping) else []
    current_terminal = milestones[-1].get("id") if milestones else None
    conductor_tasks = {
        task.get("id"): task.get("result")
        for milestone in milestones
        if isinstance(milestone, Mapping) and milestone.get("id") == "2026.09.647"
        for task in milestone.get("tasks", [])
        if isinstance(task, Mapping)
    }
    producer_rows: list[JsonDict] = []
    for number, relative, expected_identity, expected_class in V647_PRODUCERS:
        value = _load_json(root / relative)
        actual_hash = sha256(root / relative) if (root / relative).is_file() else None
        task_id = next((task for task in conductor_tasks if task.startswith(f"exp{number}-")), None)
        capstone_hash_row = capstone_hashes.get(task_id, {}) if task_id else {}
        expected_hash = (
            capstone_hash_row.get("sha256") if isinstance(capstone_hash_row, Mapping) else None
        )
        producer_rows.append(
            {
                "experiment_number": number,
                "path": relative.as_posix(),
                "experiment_id": value.get("experiment_id"),
                "expected_experiment_id": expected_identity,
                "milestone": value.get("milestone"),
                "sha256": actual_hash,
                "capstone_recorded_sha256": expected_hash,
                "hash_matches_capstone": bool(actual_hash and actual_hash == expected_hash),
                "verdict_class": value.get("verdict_class"),
                "flagged_adversarial": value.get("flagged_adversarial"),
                "inference_substrate_class": value.get("inference_substrate_class"),
                "expected_inference_substrate_class": expected_class,
                "honest_verdict": value.get("honest_verdict"),
                "MODEL_SPECS": value.get("MODEL_SPECS"),
                "model_invoked": value.get("model_invoked"),
                "invocation_counts": value.get("invocation_counts"),
                "conductor_result": conductor_tasks.get(task_id),
                "accepted_as_current_readiness": False,
                "passed": bool(
                    value.get("experiment_id") == expected_identity
                    and value.get("milestone") == "2026.09.647"
                    and actual_hash
                    and actual_hash == expected_hash
                    and value.get("verdict_class") == "disqualified"
                    and value.get("flagged_adversarial") is True
                    and value.get("inference_substrate_class") == expected_class
                    and conductor_tasks.get(task_id) == "FLAGGED"
                ),
            }
        )
    return {
        "preserved_design_path": V647_PRESERVED_DESIGN.as_posix(),
        "preserved_design_sha256": preserved_hash,
        "raw_design_path": V647_RAW_DESIGN.as_posix(),
        "raw_design_sha256": raw_hash,
        "design_hash_matches": bool(preserved_hash and preserved_hash == raw_hash),
        "planning_snapshot_ledger_terminal_milestone": "2026.09.646",
        "planning_snapshot_provenance": DESIGN_PATH.as_posix(),
        "current_ledger_terminal_milestone": current_terminal,
        "ledger_lag_reopened_completed_work": False,
        "producer_rows": producer_rows,
        "passed": bool(
            preserved_hash
            and preserved_hash == raw_hash
            and current_terminal == "2026.09.647"
            and len(producer_rows) == 3
            and all(row["passed"] for row in producer_rows)
        ),
    }


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
    """Validate the Exp7358 plan, then execute it through Exp7303."""

    manifest = AffectedManifest(
        experiment_id=FIRST_TASK_ID,
        test_paths=tuple(test_paths or V648_MANIFEST.test_paths),
        changed_modules=tuple(changed_modules or V648_MANIFEST.changed_modules),
        static_paths=tuple(static_paths or V648_MANIFEST.static_paths),
    )
    private = Path(tempfile.mkdtemp(prefix="exp7381-validation-", dir="/tmp"))
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
    """Run cold replay, adversarial verification, and strict row checks."""

    python = str(root / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib;"
        "from carnot.experiment_7381_v648_contract import validate_artifact;"
        f"v=json.loads(pathlib.Path({str(candidate)!r}).read_text());"
        f"e=validate_artifact(v,root=pathlib.Path({str(root)!r}),require_terminal=False);"
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
    """Use stable repository paths while keeping private paths readable."""

    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _hash_paths(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    """Hash each readable authority, evaluator, and sidecar once."""

    hashes: dict[str, str] = {}
    for path in dict.fromkeys(paths):
        absolute = path if path.is_absolute() else root / path
        if absolute.is_file():
            hashes[_path_label(absolute, root)] = sha256(absolute)
    return hashes


def _historical_observations(root: Path) -> list[JsonDict]:
    """Keep V647 disqualifications visible but outside current readiness."""

    history = collect_historical_boundary(root)
    return [
        {
            "observed_at": RETRIEVAL_DATE,
            "producer_path": row["path"],
            "producer_sha256": row["sha256"],
            "producer_experiment_id": row["experiment_id"],
            "producer_verdict_class": row["verdict_class"],
            "producer_flagged_adversarial": row["flagged_adversarial"],
            "producer_inference_substrate_class": row["inference_substrate_class"],
            "classification": "historical_v647_disqualified_not_current_readiness",
            "affects_v648_required_checks": False,
        }
        for row in history["producer_rows"]
    ]


def _preconditions(
    root: Path, output_path: Path, raw_dir: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], Path | None, JsonDict | None, bytes | None]:
    """Check authorities, producer identity, hashes, classes, and resources."""

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
            "expected_value": "REQ-REPORT-7381",
            "observed_value": "REQ-REPORT-7381" if "REQ-REPORT-7381" in spec_text else "missing",
            "available": "REQ-REPORT-7381" in spec_text,
            "blocking": True,
        }
    )
    history = collect_historical_boundary(root)
    rows.append(
        {
            "check": "v647_design_hash",
            "path": V647_PRESERVED_DESIGN.as_posix(),
            "upstream": V647_RAW_DESIGN.as_posix(),
            "artifact_field": "sha256",
            "expected_value": history.get("preserved_design_sha256"),
            "observed_value": history.get("raw_design_sha256"),
            "available": history.get("design_hash_matches") is True,
            "blocking": True,
        }
    )
    for producer in history["producer_rows"]:
        rows.append(
            {
                "check": f"historical_producer:{producer['experiment_number']}",
                "path": producer["path"],
                "upstream": producer["path"],
                "artifact_field": "identity/hash/class/flag",
                "expected_value": {
                    "experiment_id": producer["expected_experiment_id"],
                    "sha256": producer["capstone_recorded_sha256"],
                    "verdict_class": "disqualified",
                    "flagged_adversarial": True,
                    "inference_substrate_class": producer["expected_inference_substrate_class"],
                },
                "observed_value": {
                    "experiment_id": producer["experiment_id"],
                    "sha256": producer["sha256"],
                    "verdict_class": producer["verdict_class"],
                    "flagged_adversarial": producer["flagged_adversarial"],
                    "inference_substrate_class": producer["inference_substrate_class"],
                },
                "available": producer["passed"],
                "blocking": True,
            }
        )
    retired = _retired_ids(root / EXCLUSION_PATH)
    rows.append(
        {
            "check": "current_task_not_retired",
            "path": EXCLUSION_PATH.as_posix(),
            "upstream": EXCLUSION_PATH.as_posix(),
            "artifact_field": FIRST_TASK_ID,
            "expected_value": False,
            "observed_value": "exp7381" in retired,
            "available": "exp7381" not in retired,
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
        except OSError as exc:  # pragma: no cover - host filesystem failure.
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
    """Record the exact values needed to diagnose one terminal failure."""

    return {
        "upstream": upstream,
        "failed_check": check,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def _terminal_state(artifact: JsonDict) -> None:
    """Derive advisory completion without turning it into a science gate."""

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
            honest_verdict="blocked_v648_contract_required_input_missing",
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
            honest_verdict="complete_disqualified_v648_contract_or_validation_defect",
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
            honest_verdict="complete_circular_positive_v648_exact_advisory_contract",
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
    """Reduce completion, safety, validation, and null efficacy separately."""

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
    criteria: list[tuple[str, object, str, object, bool, str]] = [
        (
            "authority_selected",
            MILESTONE,
            "==",
            artifact.get("yaml_milestone"),
            artifact.get("yaml_milestone") == MILESTONE,
            "Only an exact V648 roadmap supplies executable rows.",
        ),
        (
            "contract_exact",
            {"task_count": EXPECTED_TASK_COUNT, "milestone": MILESTONE},
            "all_equal",
            {
                "task_count": len(artifact.get("contract_rows", [])),
                "markdown_milestone": artifact.get("markdown_milestone"),
                "yaml_milestone": artifact.get("yaml_milestone"),
            },
            artifact.get("contract_passed") is True,
            "Independent parsers must agree on every ordered field and gate.",
        ),
        (
            "mutations_rejected",
            list(CONTRACT_MUTATION_NAMES),
            "exact_set_and_all_rejected",
            [row.get("mutation") for row in mutation_rows],
            len(mutation_rows) == len(CONTRACT_MUTATION_NAMES)
            and all(row.get("baseline_passed") and row.get("rejected") for row in mutation_rows),
            "Each private defect must fail its owning check.",
        ),
        (
            "fresh_ids",
            list(EXPECTED_NUMBERS),
            "exact_order",
            [row.get("experiment_number") for row in artifact.get("fresh_id_rows", [])],
            artifact.get("fresh_ids_passed") is True,
            "V648 IDs must be ordered, unique, and absent from prior milestones.",
        ),
        (
            "prior_failures_complete",
            True,
            "is",
            artifact.get("retirement_declaration_result", {}).get("passed"),
            artifact.get("retirement_declaration_result", {}).get("passed") is True,
            "Every prior failure keeps four fields and its retirement trigger.",
        ),
        (
            "gate_controls_complete",
            gate_count * len(GATE_CASES),
            "==",
            len(gate_rows) if isinstance(gate_rows, list) else None,
            gate_count > 0 and gate_controls_complete(gate_rows, gate_count),
            "Missing, blocked, partial, disqualified, or quarantined science fails closed.",
        ),
        (
            "source_method_rows_complete",
            len(SOURCE_METHODS),
            "==",
            len(artifact.get("source_method_rows", [])),
            source_rows_complete(
                artifact.get("source_method_rows"), artifact.get("source_access_rows")
            ),
            "One access failure cannot create an inferred source result.",
        ),
        (
            "study_mapping_ingested",
            True,
            "is",
            artifact.get("study_mapping_ingested"),
            artifact.get("study_mapping_ingested") is True,
            "The dated source map must be durable before completion.",
        ),
        (
            "historical_boundary",
            True,
            "is",
            artifact.get("historical_boundary", {}).get("passed"),
            artifact.get("historical_boundary", {}).get("passed") is True,
            "Preserved V647 bytes and disqualified producer facts must match.",
        ),
        (
            "model_safety",
            {"MODEL_SPECS": [], "model_invoked": False, "counts": ZERO_INVOCATION_COUNTS},
            "all_equal",
            {
                "MODEL_SPECS": artifact.get("MODEL_SPECS"),
                "model_invoked": artifact.get("model_invoked"),
                "counts": artifact.get("invocation_counts"),
            },
            artifact.get("MODEL_SPECS") == []
            and artifact.get("model_invoked") is False
            and artifact.get("invocation_counts") == ZERO_INVOCATION_COUNTS,
            "Historical calls cannot become current LLM work.",
        ),
        (
            "required_validation",
            list(required_names),
            "exact_once_and_pass",
            [row.get("name") for row in receipts if row.get("name") in required_names],
            affected_passed and artifact.get("required_checks_passed") is True,
            "The Exp7358 plan must run each scoped check exactly once.",
        ),
        (
            "scientific_efficacy",
            None,
            "is",
            None,
            True,
            "This advisory contract has no scientific-benefit gate.",
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
                "exact_once_and_pass",
                [row.get("name") for row in receipts if row.get("name") in TERMINAL_CHECK_NAMES],
                terminal_passed,
                "Cold replay and both strict readers must pass.",
            )
        )
    return [
        {
            "criterion": name,
            "expected": expected,
            "operator": operator,
            "observed": observed,
            "passed": passed,
            "principle": principle,
        }
        for name, expected, operator, observed, passed, principle in criteria
    ]


def _field_principles(fields: Sequence[str]) -> dict[str, str]:
    """Explain every top-level field and retain the required principles."""

    principles = {
        field: "This field keeps one part of the V648 receipt auditable." for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def _base_artifact(run_date: str, started_at: str, checkpoint_path: Path) -> JsonDict:
    """Create a running checkpoint that cannot resemble terminal success."""

    return {
        "schema": "carnot.exp7381.v648_contract.v1",
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
        "llm_invocation_receipts": [],
        "small_ebm_training": {
            "performed": False,
            "training_steps": 0,
            "note": "No Gibbs fitting belongs to this advisory task.",
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "inference_substrate_details": {
            "device": platform.processor() or platform.machine() or "unknown_cpu",
            "python": platform.python_version(),
            "resource_lease": "host_cpu_no_exclusive_accelerator",
            "jax_work": False,
            "native_cuda_work": False,
        },
        "duration_s": 0.0,
        "phase_spans": [],
        "random_seed": deepcopy(RANDOM_SEED),
        "reproducibility_checksum": "",
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "contract_rows": {
                "planned": 14,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "unstarted": 14,
            },
            "mutations": {
                "planned": len(CONTRACT_MUTATION_NAMES),
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "unstarted": len(CONTRACT_MUTATION_NAMES),
            },
            "gate_controls": {
                "planned": 0,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "unstarted": 0,
            },
            "source_requests": {
                "planned": 1,
                "attempted": 0,
                "completed": 0,
                "censored": 0,
                "unstarted": 1,
            },
            "stopping_rule": "Run fourteen comparisons, ten private mutations, all gate controls, one bounded source request, and one exact validation plan.",
            "remaining_work": "No extension based on observed outcomes.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _failure_summary(
            FIRST_TASK_ID, "work_in_progress", "status", "terminal", "running"
        ),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_running_v648_contract",
        "verdict_class": "partial",
        "flagged_adversarial": False,
        "validation_receipts": [],
        "repository_health": {},
        "field_principles": {},
        "promotion_score": 0,
        "contract_complete_score": 0,
        "science_value_score": 0,
        "source_method_rows": [],
        "mutation_rows": [],
        "contract_rows": [],
        "source_access_rows": [],
        "gate_control_rows": [],
        "fresh_id_rows": [],
        "raw_authority_rows": [],
        "historical_boundary": {},
        "historical_inference_sidecars": [],
        "study_mapping_ingested": False,
        "required_checks_passed": False,
        "validation_plan_errors": [],
        "retirement_declaration_result": {},
        "checkpoint_path": str(checkpoint_path),
        "production_defaults_changed": False,
        "active_research_roadmap_changed": False,
        "research_conductor_changed": False,
        "external_publication_authorized": False,
    }


def _phase_span(
    name: str, phase_started: float, run_started: float, units: int, checkpoint: str
) -> JsonDict:
    """Measure one phase with completed units and a checkpoint boundary."""

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
    """Persist current work while the checkpoint remains visibly running."""

    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(checkpoint_path, artifact)


def _finalize(artifact: JsonDict, started: float) -> None:
    """Seal timestamps, principles, and checksum from real measured work."""

    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def independent_reduce(artifact: Mapping[str, Any], *, require_terminal: bool = True) -> list[str]:
    """Recompute raw receipt structure without trusting summary scores."""

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
    history = artifact.get("historical_boundary")
    if (
        not isinstance(history, Mapping)
        or history.get("passed") is not True
        or history.get("design_hash_matches") is not True
        or history.get("planning_snapshot_ledger_terminal_milestone") != "2026.09.646"
        or history.get("current_ledger_terminal_milestone") != "2026.09.647"
        or [row.get("experiment_number") for row in history.get("producer_rows", [])]
        != [7372, 7376, 7378]
        or any(
            row.get("accepted_as_current_readiness") is not False
            for row in history.get("producer_rows", [])
        )
    ):
        errors.append("history_invalid")
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
        else (*ROADMAP_CHECK_NAMES, *REQUIRED_CHECK_NAMES)
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
        artifact.get("schema") != "carnot.exp7381.v648_contract.v1"
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


def build_artifact(  # noqa: C901 - phases follow the declared experiment protocol.
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
    """Measure, validate, and atomically publish the V648 contract receipt."""

    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    progress("checkpoint", "start", started, "write nonterminal checkpoint")
    artifact = _base_artifact(run_date, started_at, checkpoint_path)
    _checkpoint(artifact, checkpoint_path)

    phase_started = time.monotonic()
    progress("read", "start", started, "check exact inputs and select V648 authority")
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
    artifact["retirement_declaration_result"] = validate_retirement_declarations(roadmap)
    with tempfile.TemporaryDirectory(prefix="exp7381-mutations-", dir="/tmp") as directory:
        artifact["mutation_rows"] = run_contract_mutations(
            design_copy.read_text(encoding="utf-8"), roadmap, Path(directory)
        )
    freshness = check_fresh_ids(root, roadmap)
    artifact["fresh_id_rows"] = freshness["rows"]
    artifact["fresh_ids_passed"] = freshness["passed"]
    artifact["rows"] = artifact["contract_rows"]
    artifact["sample_size_budget"]["contract_rows"].update(attempted=14, completed=14, unstarted=0)
    artifact["sample_size_budget"]["mutations"].update(
        attempted=len(CONTRACT_MUTATION_NAMES),
        completed=len(CONTRACT_MUTATION_NAMES),
        unstarted=0,
    )

    progress("build", "gate_controls_start", started, "exercise real conductor gate reader")
    gate_count = sum(len(task.get("gated_on") or []) for task in roadmap["tasks"])
    artifact["gate_control_rows"] = run_gate_controls(roadmap, raw_dir / "gate-controls")
    gate_units = gate_count * len(GATE_CASES)
    artifact["sample_size_budget"]["gate_controls"].update(
        planned=gate_units, attempted=gate_units, completed=gate_units, unstarted=0
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
    progress("evaluate", "start", started, "one bounded source delta and V647 reduction")
    source_rows, access_rows = collect_source_method_rows(fetcher)
    artifact["source_method_rows"] = source_rows
    artifact["source_access_rows"] = access_rows
    artifact["study_mapping_ingested"] = "EXP7381-V648-SOURCE-INGESTION-20260918" in (
        root / STUDY_PATH
    ).read_text(encoding="utf-8")
    artifact["sample_size_budget"]["source_requests"].update(
        attempted=1,
        completed=int(access_rows[0]["access_outcome"] == "http_success"),
        censored=int(access_rows[0]["access_outcome"] != "http_success"),
        unstarted=0,
    )
    artifact["historical_boundary"] = collect_historical_boundary(root)
    access_sidecar = raw_dir / "source-access-receipts.json"
    history_sidecar = raw_dir / "historical-v647-receipts.json"
    _atomic_write_json(
        access_sidecar,
        {"schema": "carnot.exp7381.source_access.v1", "receipts": access_rows},
    )
    _atomic_write_json(
        history_sidecar,
        {
            "schema": "carnot.exp7381.historical_v647.v1",
            "current_invocation": {
                "MODEL_SPECS": [],
                "model_invoked": False,
                "invocation_counts": ZERO_INVOCATION_COUNTS,
            },
            "historical_boundary": artifact["historical_boundary"],
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
        _phase_span("evaluate", phase_started, started, 4, "source_and_history_receipts_written")
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
    if errors:  # pragma: no cover - tests validate the same object before publication.
        raise RuntimeError(f"terminal_artifact_invalid:{errors}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(candidate, artifact)
    _atomic_write_json(output_path, artifact)
    progress("write", "after_atomic_terminal", started, f"verdict={artifact['honest_verdict']}")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the fixed V648 execution date."""

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
        print(f"[exp7381] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7381] complete verdict={artifact['honest_verdict']} score={artifact['contract_complete_score']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

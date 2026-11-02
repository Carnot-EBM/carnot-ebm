"""Build the V646 advisory source and twelve-task contract receipt.

The experiment composes existing parsers and gate readers. The new code only
defines V646 identities, evidence reduction, and its terminal artifact shape.

Spec refs: REQ-REPORT-7357 and SCENARIO-REPORT-7357-*.
"""

from __future__ import annotations

import argparse
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
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    REQUIRED_CHECK_NAMES,
    reduce_required_checks,
    run_commands,
    run_scoped_validation,
)


JsonDict = dict[str, Any]
Fetcher = Callable[[str], Mapping[str, Any]]
ValidationRunner = Callable[..., JsonDict]
TerminalRunner = Callable[[Path, Path, Path], list[JsonDict]]

REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.646"
RUN_DATE = "20260917"
RETRIEVAL_DATE = "2026-09-17"
FIRST_TASK_ID = "exp7357-contract"
EXPECTED_ID_ORDER = (
    FIRST_TASK_ID,
    "exp7358-validation-contract",
    "exp7359-capture-reducer",
    "exp7360-learning-fixture",
    "exp7361-fresh-plan-capture",
    "exp7362-prospective-learning",
    "exp7363-learning-audit",
    "exp7364-acquisition-adjudication",
    "exp7365-supervisor-support",
    "exp7366-supervisor-live",
    "exp7367-board-disposition",
    "exp7368-capstone",
)
EXPECTED_TASK_COUNT = len(EXPECTED_ID_ORDER)
RANDOM_SEED = {
    "development": 7_357_202_609_17,
    "evaluation": 7_357_202_609_18,
    "resampling": 7_357_202_609_19,
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
HISTORY_PATH = Path("results/experiment_7343_v645_contract.json")

MODULE_PATH = Path("python/carnot/experiment_7357_v646_contract.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7357_v646_contract.py")
TEST_PATH = Path("tests/python/test_experiment_7357_v646_contract.py")
DEFAULT_OUTPUT_PATH = Path("results/experiment_7357_v646_contract.json")
RAW_DIR = Path("results/raw/experiment_7357")
CHECKPOINT_PATH = Path("results/checkpoints/experiment_7357_v646_contract.json")

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
    Path("python/carnot/experiment_7329_v644_contract.py"),
    Path("python/carnot/experiment_7343_v645_contract.py"),
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
CONTRACT_MUTATION_NAMES = (
    "missing_task",
    "reordered_task",
    "stale_milestone",
    "changed_path",
    "changed_title",
    "changed_phase",
    "misspelled_gate_field",
    "bad_operator",
)

REQUIRED_FIELD_PRINCIPLES = {
    "schema": "Version this record and retain ordinary top-level experiment_id and milestone.",
    "status": "Terminal only after actual work and affected validation; never a success-shaped placeholder.",
    "run_date": "Use 20260917 and actual UTC timestamps.",
    "preconditions_checked": "Exact input, resource and required-field checks before dependent work.",
    "MODEL_SPECS": "Actual intended identities; every LLM task includes unsloth/Qwen3.8-27B-GGUF.",
    "model_invoked": "True for any attempted current model load or generation, even failure.",
    "invocation_counts": "Attempted/completed/failed/cancelled/in-flight calls; separate current from historical.",
    "inference_substrate": "Actual computation, with historical inference in explicitly labeled hash-bound sidecars.",
    "inference_substrate_class": "Actual closed duration class; no duration padding.",
    "execution_venue": "Host CPU or owned CUDA runtime as measured; no new board execution in V646.",
    "duration_s": "Measured monotonic elapsed, never synthetic elapsed or sleep to pass a floor.",
    "phase_spans": "Disjoint measured load, generation, evaluation, validation and write spans.",
    "random_seed": "Frozen development/evaluation/resampling seeds; null if truly inapplicable.",
    "reproducibility_checksum": "Bind exact code, settings, evaluator, inputs and raw evidence.",
    "source_artifact_hashes": "Exact producer paths and immutable byte hashes; preserve original classes/flags.",
    "rows": "Every comparative unit/arm/metric/cost/failure/censoring disposition, not only pooled means.",
    "sample_size_budget": "Frozen planned/attempted/completed/censored units and stopping rules.",
    "acceptance_gate_results": "Expected, observed and passed separately for required validation, safety and scientific value; never mark failed value as successful.",
    "gate_check_summary": "Every blocked_* names upstream/check, exact field, expected and observed value, including missing paths.",
    "verifier_is_oracle": "True when the evaluator defines truth; independent code alone cannot remove circularity.",
    "honest_verdict": "Precise free-text terminal outcome; distinguish accounting, null science and unavailable work.",
    "verdict_class": "Closed enum positive | circular_positive | null | blocked | disqualified | partial. Only unfinished retryable OWN work is partial; external unchanged absence is blocked.",
    "flagged_adversarial": "Current independent verification state; critical findings set true and prevent promotion.",
    "validation_receipts": "Exact command/scope/return code/elapsed/log hash for every required and diagnostic check, including failures.",
    "repository_health": "Dated unrelated failures kept separately from affected required validation.",
    "field_principles": "Explain each field without wrapping scalar gates or ordinary dictionaries.",
    "contract_complete_score": "One only after independent exact contract comparison and all planner controls pass; no scientific-value meaning.",
    "contract_rows": "Twelve parsed rows including gate lists from both files.",
    "source_method_map": "Primary URLs, retrieval outcomes, assumptions and destination experiment IDs.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

SOURCE_IDS = ("2609.14858", "2609.14857", "2604.13283", "2607.20792", "2603.27752")
ACCESS_SPECS: tuple[JsonDict, ...] = (
    {
        "source_id": "2609.14858",
        "source_name": "Dream-RSI",
        "url": "https://arxiv.org/abs/2609.14858",
        "publication_or_version_date": "2026-09-14",
    },
    {
        "source_id": "2609.14857",
        "source_name": "ModularRSI",
        "url": "https://arxiv.org/abs/2609.14857",
        "publication_or_version_date": "2026-09-14",
    },
    {
        "source_id": "2604.13283",
        "source_name": "Conservative Constraint Acquisition",
        "url": "https://arxiv.org/abs/2604.13283",
        "publication_or_version_date": "2026-04-14",
    },
    {
        "source_id": "2607.20792",
        "source_name": "Memoir",
        "url": "https://arxiv.org/abs/2607.20792",
        "publication_or_version_date": "2026-07-22",
    },
    {
        "source_id": "2603.27752",
        "source_name": "RT4CHART",
        "url": "https://arxiv.org/abs/2603.27752",
        "publication_or_version_date": "2026-03-26",
    },
)
SOURCE_METHODS: tuple[JsonDict, ...] = (
    {
        "source_id": "2609.14858",
        "assumption": "Recorded discovery-tree support can rank only replayed supervisor choices.",
        "adopted_method": "Measure support coverage and abstain on branches absent from the ledger.",
        "destination_experiment_ids": [
            "exp7365-supervisor-support",
            "exp7366-supervisor-live",
        ],
        "limitations": "A redirect followed by progress is observed support, not the causal outcome of an unobserved alternative.",
        "observed_support_replay": True,
        "causal_counterfactual_supported": False,
    },
    {
        "source_id": "2609.14857",
        "assumption": "Harness-module contrasts require task-disjoint evaluation and multiple trajectories.",
        "adopted_method": "Freeze one supervisor module and evaluate it on disjoint live games.",
        "destination_experiment_ids": [
            "exp7365-supervisor-support",
            "exp7366-supervisor-live",
        ],
        "limitations": "One failed trajectory cannot identify a general module defect or establish transfer.",
        "observed_support_replay": False,
        "causal_counterfactual_supported": False,
    },
    {
        "source_id": "2604.13283",
        "assumption": "Pairwise separation and global capacity use Boolean feasibility feedback.",
        "adopted_method": "Charge every acquisition query and preserve unsupported higher-order counterexamples.",
        "destination_experiment_ids": ["exp7362-prospective-learning"],
        "limitations": "The restricted acquisition language does not prove arbitrary hidden-rule learning.",
        "observed_support_replay": False,
        "causal_counterfactual_supported": False,
    },
    {
        "source_id": "2607.20792",
        "assumption": "Prediction reads a frozen memory snapshot and writes only between requests.",
        "adopted_method": "Test delayed verified updates on distinct future requests and version changes.",
        "destination_experiment_ids": ["exp7362-prospective-learning"],
        "limitations": "Memory writes and bounded recall results do not establish a Carnot learning benefit.",
        "observed_support_replay": False,
        "causal_counterfactual_supported": False,
    },
    {
        "source_id": "2603.27752",
        "assumption": "Claim-level evidence links are separate from executable validity.",
        "adopted_method": "Keep source-fidelity rows for quantities, entities, and ordering beside executor rows.",
        "destination_experiment_ids": ["exp7361-fresh-plan-capture"],
        "limitations": "Parser success or executable output can still misrepresent the source request.",
        "observed_support_replay": False,
        "causal_counterfactual_supported": False,
    },
)


def progress(phase: int, event: str, detail: str) -> None:
    """Flush each boundary so long checks remain visible to the conductor."""

    print(f"[exp7357] phase={phase} event={event} {detail}", flush=True)


def select_yaml_authority(
    root: Path,
) -> tuple[Path | None, JsonDict | None, bytes | None, list[JsonDict]]:
    """Prefer a matching staged V646 roadmap, then its matching active file."""

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


def parse_active_markdown_contract(text: str) -> JsonDict:
    """Parse the active document and reject duplicate active contract sections."""

    active = re.split(r"(?m)^## Historical ", text, maxsplit=1)[0]
    count = len(re.findall(r"(?im)^## Exact Task Contract\s*$", active))
    if count != 1:
        raise ValueError(f"expected exactly one active Exact Task Contract, observed {count}")
    parsed = parse_markdown_contract(active)
    parsed["active_exact_contract_count"] = count
    return parsed


def validate_gate_declarations(roadmap: Mapping[str, Any]) -> JsonDict:
    """Require every consumed field from an earlier non-receipt producer."""

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


def evaluate_contract(markdown_text: str, yaml_document: object) -> JsonDict:
    """Compare V646 sources while retaining stale and unexpected Markdown rows."""

    markdown = parse_active_markdown_contract(markdown_text)
    parsed_yaml = parse_yaml_contract(yaml_document)
    markdown_tasks = markdown["tasks"]
    yaml_tasks = parsed_yaml["tasks"]
    markdown_by_id = {task.get("id"): task for task in markdown_tasks}
    yaml_by_id = {task.get("id"): task for task in yaml_tasks}
    declarations = (
        validate_gate_declarations(yaml_document) if isinstance(yaml_document, Mapping) else {}
    )
    declaration_rows = declarations.get("rows", [])
    rows: list[JsonDict] = []
    for order, task_id in enumerate(EXPECTED_ID_ORDER, 1):
        expected = markdown_by_id.get(task_id)
        observed = yaml_by_id.get(task_id)
        task_declarations = [row for row in declaration_rows if row.get("consumer") == task_id]
        checks = {
            "order": bool(
                expected and observed and expected.get("order") == observed.get("order") == order
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
                "order": order,
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
    unexpected = [task for task in markdown_tasks if task.get("id") not in EXPECTED_ID_ORDER]
    missing = [task_id for task_id in EXPECTED_ID_ORDER if task_id not in markdown_by_id]
    passed = bool(
        markdown.get("milestone") == parsed_yaml.get("milestone") == MILESTONE
        and markdown_order == yaml_order == list(EXPECTED_ID_ORDER)
        and markdown.get("active_exact_contract_count") == 1
        and not unexpected
        and not missing
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
        "unexpected_markdown_rows": [_public_task(task) for task in unexpected],
        "missing_markdown_ids": missing,
        "gate_declaration_result": declarations,
        "passed": passed,
    }


def run_contract_mutations(
    markdown_text: str, roadmap: Mapping[str, Any], temp_root: Path
) -> list[JsonDict]:
    """Reject every required contract mutation from an exact baseline."""

    def missing_task(document: JsonDict) -> None:
        document["tasks"] = document["tasks"][:-1]

    def reordered_task(document: JsonDict) -> None:
        document["tasks"][0], document["tasks"][1] = document["tasks"][1], document["tasks"][0]

    def stale_milestone(document: JsonDict) -> None:
        document["milestone"] = "2026.09.645"

    def changed_path(document: JsonDict) -> None:
        document["tasks"][0]["deliverable"] = "results/mutated.json"

    def changed_title(document: JsonDict) -> None:
        document["tasks"][0]["title"] += " changed"

    def changed_phase(document: JsonDict) -> None:
        document["tasks"][0]["phase"] = 4

    def misspelled_gate_field(document: JsonDict) -> None:
        gated = next(task for task in document["tasks"] if task.get("gated_on"))
        gated["gated_on"][0]["artifact_feld"] = gated["gated_on"][0].pop("artifact_field")

    def bad_operator(document: JsonDict) -> None:
        gated = next(task for task in document["tasks"] if task.get("gated_on"))
        gated["gated_on"][0]["op"] = "contains"

    mutations = (
        ("missing_task", missing_task),
        ("reordered_task", reordered_task),
        ("stale_milestone", stale_milestone),
        ("changed_path", changed_path),
        ("changed_title", changed_title),
        ("changed_phase", changed_phase),
        ("misspelled_gate_field", misspelled_gate_field),
        ("bad_operator", bad_operator),
    )
    try:
        baseline_passed = evaluate_contract(markdown_text, roadmap)["passed"] is True
    except (TypeError, ValueError, yaml.YAMLError):
        baseline_passed = False
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
                "baseline_passed": baseline_passed,
                "expected": "contract_rejected",
                "observed": observed,
                "error": error,
                "rejected": observed is False,
            }
        )
    return rows


def collect_source_method_map(
    fetcher: Fetcher = _fetch_url,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Refresh five primary pages and preserve bounded access failures."""

    access_rows: list[JsonDict] = []
    total = len(ACCESS_SPECS)
    for index, spec in enumerate(ACCESS_SPECS, 1):
        progress(
            4, "source_request_start", f"completed={index - 1}/{total} source={spec['source_id']}"
        )
        try:
            receipt = dict(fetcher(str(spec["url"])))
        except Exception as exc:  # noqa: BLE001 - a fetch failure is a measured source outcome.
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
        progress(4, "source_request_end", f"completed={index}/{total} outcome={outcome}")
    access_by_id = {row["source_id"]: row for row in access_rows}
    method_map = []
    for method in SOURCE_METHODS:
        access = access_by_id[method["source_id"]]
        method_map.append(
            {
                **deepcopy(method),
                "primary_url": access["url"],
                "publication_or_version_date": access["publication_or_version_date"],
                "retrieval_date": access["retrieval_date"],
                "retrieval_outcome": access["access_outcome"],
                "retrieval_status_code": access["status_code"],
                "retrieval_error": access["error"],
                "new_runtime_dependency": False,
                "additional_research_branch": False,
                "local_science_claimed": False,
            }
        )
    return method_map, access_rows


def source_map_complete(rows: object) -> bool:
    """Require five method decisions without requiring successful access."""

    outcomes = {"http_success", "http_429", "browser_challenge", "access_failed"}
    return bool(
        isinstance(rows, list)
        and len(rows) == len(SOURCE_METHODS)
        and {row.get("source_id") for row in rows} == set(SOURCE_IDS)
        and all(
            row.get("primary_url")
            and row.get("assumption")
            and row.get("adopted_method")
            and row.get("destination_experiment_ids")
            and row.get("limitations")
            and row.get("retrieval_outcome") in outcomes
            and row.get("new_runtime_dependency") is False
            and row.get("additional_research_branch") is False
            for row in rows
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
    historical_failures: Sequence[Mapping[str, Any]] = (),
    roadmap_runner: Callable[..., list[JsonDict]] = run_commands,
    scoped_runner: Callable[..., JsonDict] = run_scoped_validation,
) -> JsonDict:
    """Run the seven planner checks and the fixed explicit-file sequence."""

    basetemp = Path("/tmp/exp7357-v646-scoped")
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
        coverage_file=Path("/tmp/.coverage-exp7357-v646"),
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
    """Reload the candidate through the reducer and both strict artifact tools."""

    python = str(root / ".venv/bin/python")
    reducer_code = (
        "import json,pathlib;"
        "from carnot.experiment_7357_v646_contract import independent_reduce;"
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
    """Keep failed V645 checks as dated repository health, not current gates."""

    try:
        data = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [
        {
            "observed_at": "2026-09-16",
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
    """Use stable repository paths while retaining private absolute paths."""

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
    """Check exact identities, requirement text, and writable destinations."""

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
            "expected_value": "REQ-REPORT-7357",
            "observed_value": "REQ-REPORT-7357" if "REQ-REPORT-7357" in spec_text else "missing",
            "available": "REQ-REPORT-7357" in spec_text,
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
    """Keep the exact five diagnostic values for one terminal failure."""

    return {
        "upstream": upstream,
        "failed_check": check,
        "artifact_field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": False,
    }


def apply_terminal_state(artifact: JsonDict) -> None:
    """Derive the accounting score and verdict only from stored checks."""

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
        science_readiness_score=0,
        science_value_score=0,
        science_promotion_score=0,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="aggregation",
        execution_venue="host",
    )
    if failed_precondition is not None:
        artifact.update(
            status="blocked",
            verdict_class="blocked",
            honest_verdict="blocked_v646_contract_required_input_missing",
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
            honest_verdict="complete_disqualified_v646_contract_mismatched_or_invalid",
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
            honest_verdict="complete_circular_positive_v646_exact_advisory_contract",
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
    """Reduce exact parity, controls, sources, model state, and validation."""

    contract_rows = artifact.get("contract_rows", [])
    gate_rows = artifact.get("gate_control_rows", [])
    gate_count = len(gate_rows) // len(GATE_CASES) if isinstance(gate_rows, list) else 0
    raw_rows = artifact.get("raw_authority_rows", [])
    mutation_rows = artifact.get("contract_mutation_rows", [])
    receipts = artifact.get("validation_receipts", [])
    criteria: list[tuple[str, object, object, bool, str]] = [
        (
            "authority_selected",
            MILESTONE,
            artifact.get("yaml_milestone"),
            artifact.get("yaml_milestone") == MILESTONE,
            "Only matching V646 YAML can supply executable tasks.",
        ),
        (
            "contract_exact",
            {"task_count": EXPECTED_TASK_COUNT, "milestone": MILESTONE},
            {
                "task_count": len(contract_rows) if isinstance(contract_rows, list) else None,
                "markdown_milestone": artifact.get("markdown_milestone"),
                "yaml_milestone": artifact.get("yaml_milestone"),
                "unexpected_markdown_rows": len(artifact.get("unexpected_markdown_rows", [])),
            },
            artifact.get("contract_passed") is True,
            "Both independent authorities must describe the same twelve tasks.",
        ),
        (
            "mutations_rejected",
            list(CONTRACT_MUTATION_NAMES),
            [row.get("mutation") for row in mutation_rows],
            len(mutation_rows) == len(CONTRACT_MUTATION_NAMES)
            and all(
                row.get("baseline_passed") is True and row.get("rejected") is True
                for row in mutation_rows
            ),
            "Mutation controls are meaningful only from an exact baseline.",
        ),
        (
            "raw_authorities_hash_bound",
            True,
            [row.get("hash_matches") for row in raw_rows] if isinstance(raw_rows, list) else None,
            isinstance(raw_rows, list)
            and len(raw_rows) == 2
            and all(row.get("hash_matches") is True for row in raw_rows),
            "Frozen byte copies must match both source authorities.",
        ),
        (
            "gate_fields_declared",
            True,
            artifact.get("gate_declaration_result", {}).get("passed"),
            artifact.get("gate_declaration_result", {}).get("passed") is True,
            "Every consumed field must be promised by an earlier producer.",
        ),
        (
            "gate_controls_complete",
            gate_count * len(GATE_CASES),
            len(gate_rows) if isinstance(gate_rows, list) else None,
            gate_count > 0 and gate_controls_complete(gate_rows, gate_count),
            "Missing or ineligible producer evidence must fail closed.",
        ),
        (
            "source_method_map_complete",
            len(SOURCE_METHODS),
            len(artifact.get("source_method_map", [])),
            source_map_complete(artifact.get("source_method_map")),
            "Source access can fail while each bounded method decision remains recorded.",
        ),
        (
            "study_mapping_ingested",
            True,
            artifact.get("study_mapping_ingested"),
            artifact.get("study_mapping_ingested") is True,
            "The durable study ledger must contain the dated V646 mapping.",
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
            "This aggregation receipt cannot claim current model inference.",
        ),
        (
            "affected_validation",
            True,
            artifact.get("required_checks_passed"),
            artifact.get("required_checks_passed") is True,
            "Every selected-roadmap and scoped affected check must pass.",
        ),
    ]
    if require_terminal:
        criteria.append(
            (
                "terminal_validation",
                list(TERMINAL_CHECK_NAMES),
                [row.get("name") for row in receipts if row.get("name") in TERMINAL_CHECK_NAMES],
                not _failed_names(receipts, TERMINAL_CHECK_NAMES),
                "Cold reduction and both terminal artifact checks must pass.",
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
    """Measure one real disjoint phase without padding its duration."""

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
    """Explain every top-level field and retain the required exact principles."""

    principles = {
        field: "This field keeps one part of the V646 receipt auditable." for field in fields
    }
    principles.update(REQUIRED_FIELD_PRINCIPLES)
    return principles


def _base_artifact(run_date: str, started_at: str, checkpoint_path: Path) -> JsonDict:
    """Create a running checkpoint that cannot resemble terminal success."""

    return {
        "schema": "carnot.exp7357.v646_contract.v1",
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
            "contract_rows": {"planned": 12, "attempted": 0, "completed": 0, "censored": 0},
            "gate_controls": {"planned": 0, "attempted": 0, "completed": 0, "censored": 0},
            "source_requests": {"planned": 5, "attempted": 0, "completed": 0, "censored": 0},
            "stopping_rule": "Stop after twelve comparisons, eight mutations, seven controls per gate, five bounded source requests, and one scoped validation sequence.",
        },
        "acceptance_gate_results": [],
        "gate_check_summary": _failure_summary(
            FIRST_TASK_ID, "work_in_progress", "status", "terminal", "running"
        ),
        "verifier_is_oracle": True,
        "honest_verdict": "partial_running_v646_contract",
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
    """Seal actual timestamps, field explanations, and evidence checksum."""

    artifact["completed_at_utc"] = datetime.now(UTC).isoformat()
    artifact["duration_s"] = time.monotonic() - started
    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)


def _checkpoint(artifact: JsonDict, checkpoint_path: Path) -> None:
    """Persist completed phase units while retaining a nonterminal status."""

    artifact["field_principles"] = _field_principles(tuple(artifact))
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(checkpoint_path, artifact)


def independent_reduce(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute evidence structure without trusting the completion score."""

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
        or [row.get("mutation") for row in mutations] != list(CONTRACT_MUTATION_NAMES)
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
        artifact.get("schema") != "carnot.exp7357.v646_contract.v1"
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
    """Measure, validate, and atomically publish the V646 contract receipt."""

    started = time.monotonic()
    progress(0, "start", "write running checkpoint before preconditions")
    artifact = _base_artifact(run_date, datetime.now(UTC).isoformat(), checkpoint_path)
    _checkpoint(artifact, checkpoint_path)

    phase_start = time.monotonic()
    progress(1, "start", "check exact inputs and resolve V646 authority")
    preconditions, authority, selected_roadmap, yaml_bytes = _preconditions(
        root, output_path, raw_dir, checkpoint_path
    )
    artifact["preconditions_checked"] = preconditions
    artifact["phase_spans"].append(
        _phase_span("load", phase_start, started, len(preconditions), "inputs_checked")
    )
    if any(row.get("blocking") and row.get("available") is not True for row in preconditions):
        apply_terminal_state(artifact)
        _finalize(artifact, started)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_path, artifact)
        progress(7, "terminal_write", f"blocked output={output_path}")
        return artifact
    assert authority is not None and selected_roadmap is not None and yaml_bytes is not None
    _checkpoint(artifact, checkpoint_path)

    phase_start = time.monotonic()
    progress(2, "start", "record zero current model work")
    artifact["phase_spans"].append(
        _phase_span("generation", phase_start, started, 0, "no_model_invoked")
    )
    progress(2, "end", "MODEL_SPECS=[] model_invoked=false")
    _checkpoint(artifact, checkpoint_path)

    phase_start = time.monotonic()
    progress(3, "start", "hash-copy authorities before contract evaluation")
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
            "source_path": str(authority),
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
    with tempfile.TemporaryDirectory(prefix="exp7357-mutations-", dir="/tmp") as directory:
        artifact["contract_mutation_rows"] = run_contract_mutations(
            design_copy.read_text(encoding="utf-8"), roadmap, Path(directory)
        )
    artifact["rows"] = artifact["contract_rows"]
    artifact["sample_size_budget"]["contract_rows"].update(attempted=12, completed=12)

    progress(3, "gate_controls_start", "exercise the real conductor reader")
    gate_count = sum(len(task.get("gated_on") or []) for task in roadmap["tasks"])
    artifact["gate_control_rows"] = run_gate_controls(roadmap, raw_dir / "gate-controls")
    gate_units = gate_count * len(GATE_CASES)
    artifact["sample_size_budget"]["gate_controls"].update(
        planned=gate_units, attempted=gate_units, completed=gate_units
    )
    progress(3, "gate_controls_end", f"completed={gate_units}/{gate_units}")

    progress(4, "start", "refresh five primary sources")
    source_map, access_rows = collect_source_method_map(fetcher)
    artifact["source_method_map"] = source_map
    artifact["source_access_failure_count"] = sum(
        row["access_outcome"] != "http_success" for row in access_rows
    )
    access_sidecar = raw_dir / "source-access-receipts.json"
    _atomic_write_json(
        access_sidecar, {"schema": "carnot.exp7357.source_access.v1", "receipts": access_rows}
    )
    history = json.loads((root / HISTORY_PATH).read_text(encoding="utf-8"))
    history_sidecar = raw_dir / "historical-model-receipts.json"
    _atomic_write_json(
        history_sidecar,
        {
            "schema": "carnot.exp7357.historical_models.v1",
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
                "invocation_counts": history.get("invocation_counts"),
                "verdict_class": history.get("verdict_class"),
                "flagged_adversarial": history.get("flagged_adversarial"),
                "accepted_as_current_inference": False,
            },
        },
    )
    artifact["study_mapping_ingested"] = "EXP7357-V646-SOURCE-INGESTION-20260917" in (
        root / STUDY_PATH
    ).read_text(encoding="utf-8")
    artifact["sample_size_budget"]["source_requests"].update(
        attempted=5,
        completed=sum(row["access_outcome"] == "http_success" for row in access_rows),
        censored=artifact["source_access_failure_count"],
    )
    artifact["sidecar_receipts"] = {
        "source_access": {
            "path": _path_label(access_sidecar, root),
            "sha256": sha256(access_sidecar),
        },
        "historical_models": {
            "path": _path_label(history_sidecar, root),
            "sha256": sha256(history_sidecar),
        },
    }
    artifact["phase_spans"].append(
        _phase_span(
            "evaluation",
            phase_start,
            started,
            EXPECTED_TASK_COUNT + gate_units + len(ACCESS_SPECS),
            "raw_evidence_written",
        )
    )
    _checkpoint(artifact, checkpoint_path)

    phase_start = time.monotonic()
    progress(5, "start", "run structural and scoped affected validation")
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
            history_sidecar,
        ],
    )
    artifact["acceptance_gate_results"] = _acceptance_gates(artifact, require_terminal=False)
    apply_terminal_state(artifact)
    _finalize(artifact, started)
    candidate = raw_dir / "measured-terminal-candidate.json"
    _atomic_write_json(candidate, artifact)
    progress(5, "candidate_written", f"path={candidate}")

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
            "validation",
            phase_start,
            started,
            len(artifact["validation_receipts"]),
            "terminal_checks_complete",
        )
    )
    _finalize(artifact, started)

    phase_start = time.monotonic()
    progress(7, "start", "atomically write terminal artifact")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact["phase_spans"].append(_phase_span("write", phase_start, started, 1, "terminal_output"))
    _finalize(artifact, started)
    _atomic_write_json(output_path, artifact)
    progress(7, "terminal_write", f"output={output_path} verdict={artifact['honest_verdict']}")
    return artifact


def date_argument(value: str) -> str:
    """Accept only the fixed V646 execution date."""

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
        print(f"[exp7357] invalid artifact: {errors}", file=sys.stderr, flush=True)
        return 1
    print(
        f"[exp7357] complete verdict={artifact['honest_verdict']} score={artifact['contract_complete_score']}",
        flush=True,
    )
    return 0

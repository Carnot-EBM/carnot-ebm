#!/usr/bin/env python3
"""Build the REQ-REPORT-7204 V634 evidence matrix.

The command reads frozen contracts and terminal producer artifacts. It checks
their provenance and rebuilds selected claims from retained rows. It does not
load a model or rerun a scientific workload. This keeps completed measurement,
method value, and product-level value as separate statements.

Spec refs: REQ-REPORT-7204 and SCENARIO-REPORT-7204-*.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MILESTONE = "2026.09.634"
RUN_DATE = "20260911"
RANDOM_SEED = 720420260911
SPEC_PATH = "openspec/capabilities/research-reporting/spec.md"
FROZEN_ROADMAP_PATH = "results/raw/experiment_7192/selected-roadmap.yaml"
ACTIVE_ROADMAP_PATH = "research-roadmap.yaml"
STAGED_ROADMAP_PATH = "research-roadmap-next.yaml"
DEFAULT_ARTIFACT_PATH = "results/experiment_7204_v634_capstone.json"
DEFAULT_CHECKPOINT_PATH = "results/checkpoints/experiment_7204_v634_capstone.json"
GATE_IDS = ("G1", "G2", "G3", "G4")
BRANCH_ACTIONS = {"continue", "retire", "needs_changed_prerequisite"}

EXPECTED_CONTRACT = (
    (
        "exp7192-source-contract",
        "V634 source ingestion and exact execution contract",
        "results/experiment_7192_v634_source_contract.json",
    ),
    (
        "exp7193-arc-direct-tool",
        "Live ARC direct-tool generalization measurement",
        "results/experiment_7193_v634_arc_direct_tool.json",
    ),
    (
        "exp7194-arc-gap-audit",
        "ARC tool-gap and banked-progress causal audit",
        "results/experiment_7194_v634_arc_gap_audit.json",
    ),
    (
        "exp7195-typed-grounding",
        "Typed source execution and abstention prototype",
        "results/experiment_7195_v634_typed_grounding.json",
    ),
    (
        "exp7196-qwen-atomic-capture",
        "Qwen3.8 independent source and claim capture",
        "results/experiment_7196_v634_qwen_atomic_capture.json",
    ),
    (
        "exp7197-grounding-value-audit",
        "Typed grounding value and independent semantic audit",
        "results/experiment_7197_v634_grounding_value_audit.json",
    ),
    (
        "exp7198-feedback-capacity-stream",
        "Sealed constraint stream with bounded pending feedback",
        "results/experiment_7198_v634_feedback_capacity_stream.json",
    ),
    (
        "exp7199-bounded-acquisition",
        "Continuous self-learning by bounded constraint acquisition",
        "results/experiment_7199_v634_bounded_acquisition.json",
    ),
    (
        "exp7200-acquisition-cold-audit",
        "Cold feedback-causality and memory rollback audit",
        "results/experiment_7200_v634_acquisition_cold_audit.json",
    ),
    (
        "exp7201-slice-pyo3",
        "Persistent PyO3 fixed-cardinality sampler prototype",
        "results/experiment_7201_v634_slice_pyo3.json",
    ),
    (
        "exp7202-slice-cost-quality",
        "Sampler boundary cost and sample-quality comparison",
        "results/experiment_7202_v634_slice_cost_quality.json",
    ),
    (
        "exp7203-hardware-correction",
        "Board continuity and quantized-correction cost envelope",
        "results/experiment_7203_v634_hardware_correction.json",
    ),
    (
        "exp7204-capstone",
        "V634 independent evidence matrix and branch decisions",
        DEFAULT_ARTIFACT_PATH,
    ),
)
EXPECTED_TASK_IDS = tuple(row[0] for row in EXPECTED_CONTRACT)

EXPECTED_GATES: dict[str, list[dict[str, Any]]] = {
    "exp7194-arc-gap-audit": [
        {
            "upstream": "exp7193-arc-direct-tool",
            "artifact_field": "arc_tool_measurement_complete_score",
            "op": "==",
            "value": 1,
        }
    ],
    "exp7196-qwen-atomic-capture": [
        {
            "upstream": "exp7195-typed-grounding",
            "artifact_field": "typed_executor_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    "exp7197-grounding-value-audit": [
        {
            "upstream": "exp7195-typed-grounding",
            "artifact_field": "typed_executor_ready_score",
            "op": "==",
            "value": 1,
        },
        {
            "upstream": "exp7196-qwen-atomic-capture",
            "artifact_field": "atomic_capture_complete_score",
            "op": "==",
            "value": 1,
        },
    ],
    "exp7199-bounded-acquisition": [
        {
            "upstream": "exp7198-feedback-capacity-stream",
            "artifact_field": "stream_capacity_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
    "exp7200-acquisition-cold-audit": [
        {
            "upstream": "exp7199-bounded-acquisition",
            "artifact_field": "acquisition_run_complete_score",
            "op": "==",
            "value": 1,
        }
    ],
    "exp7202-slice-cost-quality": [
        {
            "upstream": "exp7201-slice-pyo3",
            "artifact_field": "pyo3_slice_ready_score",
            "op": "==",
            "value": 1,
        }
    ],
}

REQUIRED_FIELD_PRINCIPLES = {
    "field_principles": "Echo each declared reason beside the actual evidence contract.",
    "status": "Terminal only after completion or a diagnosed external block.",
    "run_date": "Use 20260911, never a historical date.",
    "preconditions_checked": "Record each required resource and its actual observed state.",
    "inference_substrate": "Describe executed computation, not the planned workload.",
    "inference_substrate_class": "Apply the duration floor for the work actually performed.",
    "execution_venue": "Host or device identity limits the scope of the evidence.",
    "duration_s": "Measure monotonic elapsed work; never pad time to pass a floor.",
    "source_artifact_hashes": "Bind code, inputs and frozen contracts to the result.",
    "rows": "Retain unit ID, arm, seed, metric, error and abstention for every comparison.",
    "sample_size_budget": "Record planned and completed counts, independent units and exclusions.",
    "random_seed": "Freeze all stochastic choices before reading held-out outcomes.",
    "reproducibility_checksum": "Hash inputs, code, seeds and raw rows.",
    "gate_check_summary": "Every blocked verdict names the failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "True when verification uses the same correctness authority; separate implementations alone do not remove circularity.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only incomplete own work can be partial.",
    "honest_verdict": "Use complete_ or complete: for completed findings, including nulls; blocked_* for external blocks. Never promote infrastructure readiness as scientific benefit.",
    "capstone_complete_score": "A complete matrix can describe blocked science.",
    "evidence_matrix": "Every contracted task remains visible.",
    "recomputed_claim_rows": "Headlines must be reconstructable without rerunning models.",
    "branch_decisions": "Continue only with evidence or a genuinely changed prerequisite.",
    "scope_reduction_compliance": "Record adherence to active priorities and retired mechanism boundaries.",
    "publication_gate": "Use existing G1-G4 definitions; do not redefine readiness.",
    "e2e_receipts": "Promoted implementation claims need scoped integration evidence.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is cited, not repeated.",
}
REQUIRED_ARTIFACT_FIELDS = frozenset(REQUIRED_FIELD_PRINCIPLES)

SOURCE_PATHS = (
    "AGENTS.md",
    "CLAUDE.md",
    "CODEX.md",
    "research-program.md",
    "research-references.md",
    "ops/exclusion_manifest.yaml",
    "ops/e2e-test-plan.md",
    "results/experiment_7191_v633_capstone.json",
    "scripts/adversarial_verify.py",
    "scripts/verdict_row_consistency_lint.py",
    "scripts/publication_gate.py",
    "ops/conductor-log.md",
    SPEC_PATH,
    "scripts/experiments/experiment_7191_v633_capstone.py",
    "scripts/experiments/experiment_7204_v634_capstone.py",
    "tests/python/test_experiment_7204_v634_capstone.py",
)

CheckerLoader = Callable[
    [Path], tuple[Callable[[Path], Mapping[str, Any]], Callable[[Path], tuple[str, list[str]]]]
]
PublicationRunner = Callable[[Path], dict[str, Any]]


def _load_v633_base() -> Any:
    """Reuse the shipped capstone's byte hashing and checker loading helpers."""

    path = REPO_ROOT / "scripts/experiments/experiment_7191_v633_capstone.py"
    spec = importlib.util.spec_from_file_location("exp7204_v633_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shipped capstone helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_v633_base()
sha256_bytes = _BASE.sha256_bytes
write_atomic = _BASE.write_atomic
read_json_object = _BASE.read_json_object
default_checker_loader = _BASE.default_checker_loader

LARGE_ARTIFACT_THRESHOLD_BYTES = 512 * 1024
SUMMARY_LIST_FIELDS: dict[int, dict[str, tuple[str, ...]]] = {
    7196: {
        "rows": ("metric",),
        "completion_rows": ("terminal_state",),
    },
    7197: {
        "rows": ("arm", "abstention", "amortized_latency_s"),
        "cold_audit_rows": ("decision_disagreement",),
        "paired_interval_rows": ("comparison", "metric", "ci95_lower", "estimate"),
    },
    7198: {"mutation_audit_rows": ("detected",)},
    7199: {
        "comparison_rows": (
            "capacity",
            "delay_schedule",
            "arm",
            "control",
            "window",
            "metric",
            "ci95_high",
        )
    },
    7200: {
        "rows": ("passed",),
        "reconstruction_rows": ("passed",),
        "metric_recomputation_rows": ("passed",),
        "causal_control_rows": ("passed", "control"),
        "cold_reload_rows": ("passed",),
        "rollback_rows": ("passed",),
        "whole_learning_reset_rows": ("passed", "acquisition_dependence_established"),
        "mutation_rows": ("passed",),
        "source_grounding_rows": ("passed",),
    },
    7201: {
        "transition_rows": ("passed",),
        "distribution_rows": ("passed",),
        "e2e_receipts": ("passed",),
    },
    7202: {
        "quality_comparison_rows": ("passed",),
        "control_rows": ("control_detected",),
    },
    7203: {"exact_law_rows": ("passed",)},
}


def progress(phase: int, state: str, detail: str) -> None:
    """Flush every boundary so a slow artifact read is not mistaken for a stall."""

    print(f"[exp7204] phase {phase} {state}: {detail}", flush=True)


def sha256_path(path: Path) -> str:
    """Hash large producer bytes incrementally so aggregation has bounded memory."""

    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256")
    return "sha256:" + digest.hexdigest()


def task_number(task_id: str) -> int:
    """Extract the contract number only from a complete task identifier."""

    match = re.fullmatch(r"exp(\d+)-[a-z0-9-]+", task_id)
    if match is None:
        raise ValueError(f"invalid task id: {task_id}")
    return int(match.group(1))


def canonical_gate_block_path(task_id: str) -> str:
    """Name only the conductor's canonical gate-block fallback for this task."""

    slug = task_id.split("-", 1)[1].replace("-", "_")
    return f"results/experiment_{task_number(task_id)}_{slug}.json"


def read_json_summary(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Bound memory for very large ledgers while preserving rows used by claims.

    Exp7199 keeps hundreds of thousands of audit-detail rows. The capstone needs
    their exact counts, but its numeric gate uses the smaller comparison ledger.
    A bounded child frees the large parse arena when it exits.
    """

    if path.stat().st_size <= LARGE_ARTIFACT_THRESHOLD_BYTES:
        return read_json_object(path)
    match = re.search(r"experiment_(\d+)_", path.name)
    number = int(match.group(1)) if match else 0
    keep = SUMMARY_LIST_FIELDS.get(number, {})
    code = """
import json
import sys

path = sys.argv[1]
keep = json.loads(sys.argv[2])
with open(path, encoding="utf-8") as handle:
    data = json.load(handle)
counts = {}
for key, value in list(data.items()):
    if not isinstance(value, list):
        continue
    counts[key] = len(value)
    wanted = keep.get(key)
    if wanted is None:
        data[key] = []
    else:
        data[key] = [
            {name: row.get(name) for name in wanted}
            for row in value
            if isinstance(row, dict)
        ]
data["_isolated_row_counts"] = counts
json.dump(data, sys.stdout, separators=(",", ":"))
"""
    completed = subprocess.run(
        [sys.executable, "-u", "-c", code, str(path), json.dumps(keep)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        return None, f"isolated_json_read_failed:{completed.stderr.strip()}"
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return None, f"JSONDecodeError:{exc}"
    if not isinstance(value, dict):
        return None, "json_root_not_object"
    return value, None


def _contract_tasks(value: Any) -> tuple[str | None, list[dict[str, Any]]]:
    """Decode both repository and compact test-fixture roadmap shapes."""

    if isinstance(value, dict):
        milestone = value.get("milestone")
        tasks = value.get("tasks")
    elif isinstance(value, list):
        tasks = value
        milestone = tasks[0].get("milestone") if tasks and isinstance(tasks[0], dict) else None
    else:
        return None, []
    if not isinstance(tasks, list) or not all(isinstance(row, dict) for row in tasks):
        return str(milestone) if milestone is not None else None, []
    return str(milestone) if milestone is not None else None, tasks


def _contract_errors(milestone: str | None, tasks: Sequence[Mapping[str, Any]]) -> list[str]:
    """Compare identity, public fields, and exact structured gates independently."""

    errors: list[str] = []
    if milestone != MILESTONE or tuple(str(task.get("id")) for task in tasks) != EXPECTED_TASK_IDS:
        errors.append("contract_identity")
    expected = {task_id: (title, deliverable) for task_id, title, deliverable in EXPECTED_CONTRACT}
    for task in tasks:
        task_id = str(task.get("id"))
        if task_id not in expected:
            continue
        title, deliverable = expected[task_id]
        if task.get("title") != title or task.get("deliverable") != deliverable:
            if "contract_fields" not in errors:
                errors.append("contract_fields")
        if (task.get("gated_on") or []) != EXPECTED_GATES.get(task_id, []):
            if "contract_gates" not in errors:
                errors.append("contract_gates")
    return errors


def resolve_contract(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None, list[str]]:
    """Prefer the immutable Exp7192 contract, then exact mutable candidates."""

    candidates = (FROZEN_ROADMAP_PATH, ACTIVE_ROADMAP_PATH, STAGED_ROADMAP_PATH)
    source_rows: list[dict[str, Any]] = []
    selected: str | None = None
    selected_tasks: list[dict[str, Any]] = []
    selected_errors = ["contract_authority"]
    for relative in candidates:
        path = root / relative
        row: dict[str, Any] = {
            "path": relative,
            "present": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256_path(path) if path.is_file() else None,
            "milestone": None,
            "id_order": [],
            "selected": False,
            "read_error": None,
        }
        tasks: list[dict[str, Any]] = []
        if path.is_file():
            try:
                milestone, tasks = _contract_tasks(yaml.safe_load(path.read_text(encoding="utf-8")))
                row["milestone"] = milestone
                row["id_order"] = [task.get("id") for task in tasks]
                errors = _contract_errors(milestone, tasks)
                if selected is None and "contract_identity" not in errors:
                    selected = relative
                    selected_tasks = tasks
                    selected_errors = errors
                    row["selected"] = True
            except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
                row["read_error"] = f"{type(exc).__name__}:{exc}"
        source_rows.append(row)
    return selected_tasks, source_rows, selected, selected_errors


def _unwrap(value: Any) -> Any:
    """Read principle-wrapped producer fields without treating prose as evidence."""

    if isinstance(value, Mapping) and "value" in value and "principle" in value:
        return value["value"]
    return value


def _manifest_matches(value: Any, targets: frozenset[Any]) -> bool:
    """Find exact retired artifact or experiment identities in the manifest."""

    if isinstance(value, Mapping):
        return any(_manifest_matches(item, targets) for item in value.values())
    if isinstance(value, list):
        return any(_manifest_matches(item, targets) for item in value)
    return value in targets


def quarantine_receipt(
    payload: Mapping[str, Any], task_id: str, artifact_path: str, manifest: Any
) -> dict[str, Any]:
    """Reject explicit artifact flags and exact exclusion-manifest matches."""

    flag_names = (
        "artifact_quarantined",
        "upstream_quarantined",
        "quarantine_flag",
        "quarantined",
        "excluded_from_use",
        "flagged_adversarial",
    )
    declared = {
        name: _unwrap(payload[name])
        for name in flag_names
        if name in payload and _unwrap(payload[name]) is True
    }
    number = task_number(task_id)
    targets = frozenset((number, str(number), task_id, artifact_path, Path(artifact_path).name))
    manifest_match = _manifest_matches(manifest, targets)
    return {
        "declared_flags": declared,
        "exclusion_manifest_match": manifest_match,
        "quarantined": bool(declared) or manifest_match,
    }


def load_task_evidence(root: Path, task: Mapping[str, Any], manifest: Any) -> dict[str, Any]:
    """Read the declared deliverable first and use only the canonical fallback."""

    task_id = str(task.get("id"))
    declared = str(task.get("deliverable"))
    canonical = canonical_gate_block_path(task_id)
    declared_path = root / declared
    canonical_path = root / canonical
    if declared_path.is_file():
        selected, source = declared_path, "declared_deliverable"
    elif canonical_path.is_file():
        selected, source = canonical_path, "conductor_gate_block"
    else:
        selected, source = None, "missing"
    payload: dict[str, Any] | None = None
    error: str | None = None
    if selected is not None:
        payload, error = read_json_summary(selected)
        if error is not None:
            source = "unreadable"
    selected_display = str(selected.relative_to(root)) if selected is not None else None
    quarantine = quarantine_receipt(payload or {}, task_id, selected_display or declared, manifest)
    return {
        "task_id": task_id,
        "declared_deliverable_path": declared,
        "canonical_gate_block_path": canonical,
        "selected_evidence_path": selected_display,
        "evidence_source": source,
        "artifact_size_bytes": selected.stat().st_size if selected is not None else 0,
        "artifact_sha256": sha256_path(selected) if selected is not None else None,
        "read_error": error,
        **quarantine,
        "accepted_for_promoted_evidence": payload is not None and not quarantine["quarantined"],
        "payload": payload,
    }


def _claim(
    task_id: str,
    name: str,
    payload: Mapping[str, Any],
    recomputed: Any,
    fields: Sequence[str],
) -> dict[str, Any]:
    """Bind one declared numeric headline to its retained evidence fields."""

    declared = payload.get(name)
    matches = declared == recomputed
    return {
        "unit_id": f"{task_id}:{name}",
        "arm": "recomputed_upstream_claim",
        "seed": RANDOM_SEED,
        "metric": name,
        "error": None if matches else "declared_value_mismatch",
        "abstention": False,
        "claim_score": recomputed,
        "task_id": task_id,
        "claim": name,
        "declared_value": declared,
        "recomputed_value": recomputed,
        "evidence_fields": list(fields),
        "matches": matches,
    }


def _rows(payload: Mapping[str, Any], field: str = "rows") -> list[Mapping[str, Any]]:
    """Return only mapping rows so malformed lists cannot pass by length alone."""

    value = payload.get(field)
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


def _row_count(payload: Mapping[str, Any], field: str) -> int:
    """Use an isolated exact count when a large detail ledger was not retained."""

    isolated = payload.get("_isolated_row_counts")
    if isinstance(isolated, Mapping) and field in isolated:
        return int(isolated[field])
    return len(_rows(payload, field))


def _all_true(rows: Sequence[Mapping[str, Any]], field: str = "passed") -> bool:
    """Require a nonempty collection with one explicit true field per row."""

    return bool(rows) and all(row.get(field) is True for row in rows)


def _grounding_value(payload: Mapping[str, Any]) -> int:
    """Reapply the two preregistered value branches to raw arm and interval rows."""

    rows = _rows(payload)
    typed = [row for row in rows if row.get("arm") == "typed_execution_unknown_abstention"]
    direct = [row for row in rows if row.get("arm") == "direct_judgment"]
    intervals = _rows(payload, "paired_interval_rows")

    def interval(metric: str) -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in intervals
                if row.get("comparison") == "typed_vs_direct" and row.get("metric") == metric
            ),
            None,
        )

    accuracy = interval("accuracy")
    false_accept = interval("false_accept_rate")
    coverage_delta = interval("coverage")
    audit = _rows(payload, "cold_audit_rows")
    audit_passed = len(audit) == 128 and not any(
        row.get("decision_disagreement") is True for row in audit
    )
    if not typed or not direct or None in (accuracy, false_accept, coverage_delta):
        return 0
    typed_coverage = sum(not bool(row.get("abstention")) for row in typed) / len(typed)
    direct_coverage = sum(not bool(row.get("abstention")) for row in direct) / len(direct)
    typed_latency = sum(float(row.get("amortized_latency_s", 0.0)) for row in typed)
    direct_latency = sum(float(row.get("amortized_latency_s", 0.0)) for row in direct)
    accuracy_branch = bool(
        float(accuracy["ci95_lower"]) > 0
        and float(false_accept["estimate"]) <= 0
        and typed_coverage >= 0.60
        and audit_passed
    )
    efficiency_branch = bool(
        float(accuracy["ci95_lower"]) >= -0.02
        and typed_coverage == direct_coverage
        and float(coverage_delta["estimate"]) == 0
        and typed_latency > 0
        and direct_latency / typed_latency >= 2.0
        and audit_passed
    )
    return int(accuracy_branch or efficiency_branch)


def _acquisition_value(payload: Mapping[str, Any]) -> tuple[int, int]:
    """Apply the frozen capacity-four burst gate to paired stream-seed rows."""

    comparisons = [
        row
        for row in _rows(payload, "comparison_rows")
        if row.get("capacity") == 4
        and row.get("delay_schedule") == "burst"
        and row.get("arm") == "priority_admission"
    ]

    def high(control: str, window: str, metric: str) -> float:
        row = next(
            (
                item
                for item in comparisons
                if item.get("control") == control
                and item.get("window") == window
                and item.get("metric") == metric
            ),
            None,
        )
        return float("inf") if row is None else float(row["ci95_high"])

    safe = payload.get("memory_capacity_violation_count") == 0
    controls = ("warmup_frozen", "fifo_admission")
    value = safe and all(
        high(control, "prospective", "error_rate") < 0
        and high(control, "prospective", "false_accept_rate") <= 0
        and high(control, "recurrence", "error_rate") <= 0.02
        for control in controls
    )
    priority = safe and high("random_admission", "prospective", "error_rate") < 0
    return int(value), int(priority)


def _cold_audit_scores(payload: Mapping[str, Any]) -> tuple[int, int]:
    """Rebuild cold-audit completion and causal promotion from owned rows."""

    controls = {row.get("control") for row in _rows(payload, "causal_control_rows")}
    required_lists = (
        "rows",
        "reconstruction_rows",
        "metric_recomputation_rows",
        "causal_control_rows",
        "cold_reload_rows",
        "rollback_rows",
        "whole_learning_reset_rows",
        "mutation_rows",
        "source_grounding_rows",
    )
    complete = all(_all_true(_rows(payload, field)) for field in required_lists)
    complete = (
        complete
        and {
            "delayed_feedback",
            "shuffled_feedback",
            "template_only_deletion",
        }
        <= controls
    )
    isolation = payload.get("runtime_isolation_receipt", {})
    complete = complete and all(
        isolation.get(field) is True
        for field in (
            "fresh_process",
            "no_model_load",
            "network_disabled",
            "protected_inputs_unchanged",
        )
    )
    reset = _rows(payload, "whole_learning_reset_rows")
    promotion = bool(
        complete
        and payload.get("upstream_gate_receipt", {}).get("acquisition_value_score") == 1
        and all(row.get("acquisition_dependence_established") is True for row in reset)
        and payload.get("scheduler_control_summary", {}).get("scheduling_benefit_supported") is True
    )
    return int(complete), int(promotion)


def recompute_claims(task_id: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Rebuild every numeric headline that this capstone promotes or limits."""

    rows = _rows(payload)
    claims: list[dict[str, Any]] = []
    if task_id == "exp7192-source-contract":
        value = int(len(rows) == 13 and _all_true(rows))
        claims.append(_claim(task_id, "source_contract_complete_score", payload, value, ["rows"]))
    elif task_id == "exp7193-arc-direct-tool":
        complete = int(
            len(rows) == 1
            and payload.get("status") == "complete"
            and payload.get("gate_check_summary", {}).get("passed") is True
        )
        inductions = sum(int(row.get("metric_value", 0) or 0) for row in rows)
        budget = payload.get("sample_size_budget", {})
        volume = int(inductions >= int(budget.get("evidence_target_new_inductions", 10)))
        claims.extend(
            (
                _claim(
                    task_id,
                    "arc_tool_measurement_complete_score",
                    payload,
                    complete,
                    ["rows", "gate_check_summary"],
                ),
                _claim(
                    task_id,
                    "arc_tool_engagement_score",
                    payload,
                    int(inductions > 0),
                    ["rows.metric_value"],
                ),
                _claim(
                    task_id,
                    "arc_volume_sufficient_score",
                    payload,
                    volume,
                    ["rows.metric_value", "sample_size_budget"],
                ),
            )
        )
    elif task_id == "exp7194-arc-gap-audit":
        counts = Counter(row.get("metric") for row in rows)
        complete = int(
            payload.get("status") == "complete"
            and len(rows) == 7
            and counts
            == {"parsed_tool_calls": 3, "captured_gap_events": 3, "banked_level_transitions": 1}
        )
        claims.append(_claim(task_id, "arc_gap_audit_complete_score", payload, complete, ["rows"]))
    elif task_id == "exp7195-typed-grounding":
        mutations = _rows(payload, "semantic_mutation_rows")
        ready = int(
            len(rows) == 192
            and all(row.get("metric") == 1 for row in rows)
            and len(mutations) == 9
            and _all_true(mutations)
        )
        claims.append(
            _claim(
                task_id,
                "typed_executor_ready_score",
                payload,
                ready,
                ["rows", "semantic_mutation_rows"],
            )
        )
    elif task_id == "exp7196-qwen-atomic-capture":
        completions = _rows(payload, "completion_rows")
        complete = int(
            len(rows) == 192
            and all(row.get("metric") == 1 for row in rows)
            and len(completions) == 576
            and all(
                row.get("terminal_state") in {"response", "request_error"} for row in completions
            )
            and payload.get("gpu_receipts", {}).get("provenance_ok") is True
            and float(payload.get("duration_s", 0.0)) >= 60.0
        )
        claims.append(
            _claim(
                task_id,
                "atomic_capture_complete_score",
                payload,
                complete,
                ["rows", "completion_rows", "gpu_receipts", "duration_s"],
            )
        )
    elif task_id == "exp7197-grounding-value-audit":
        arms = Counter(row.get("arm") for row in rows)
        complete = int(
            _row_count(payload, "rows") == 640
            and len(arms) == 5
            and set(arms.values()) == {128}
            and len(_rows(payload, "cold_audit_rows")) == 128
            and len(_rows(payload, "paired_interval_rows")) == 12
        )
        value = _grounding_value(payload)
        claims.extend(
            (
                _claim(
                    task_id,
                    "grounding_audit_complete_score",
                    payload,
                    complete,
                    ["rows", "cold_audit_rows", "paired_interval_rows"],
                ),
                _claim(
                    task_id,
                    "grounding_value_score",
                    payload,
                    value,
                    ["rows", "cold_audit_rows", "paired_interval_rows"],
                ),
            )
        )
    elif task_id == "exp7198-feedback-capacity-stream":
        ready = int(
            _row_count(payload, "rows") == 360
            and _row_count(payload, "information_budget_rows") == 360
            and _row_count(payload, "pending_queue_rows") == 9216
            and _row_count(payload, "warmup_state_rows") == 360
            and _row_count(payload, "headroom_rows") == 121
            and all(row.get("detected") is True for row in _rows(payload, "mutation_audit_rows"))
        )
        claims.append(
            _claim(
                task_id,
                "stream_capacity_ready_score",
                payload,
                ready,
                [
                    "rows",
                    "information_budget_rows",
                    "pending_queue_rows",
                    "warmup_state_rows",
                    "headroom_rows",
                    "mutation_audit_rows",
                ],
            )
        )
    elif task_id == "exp7199-bounded-acquisition":
        value, priority = _acquisition_value(payload)
        complete = int(
            _row_count(payload, "rows") == 600
            and _row_count(payload, "comparison_rows") == 432
            and _row_count(payload, "decision_rows") == 614400
            and _row_count(payload, "validation_partition_rows") == 10240
            and _row_count(payload, "update_rows") > 0
            and payload.get("memory_capacity_violation_count") == 0
        )
        claims.extend(
            (
                _claim(
                    task_id,
                    "acquisition_run_complete_score",
                    payload,
                    complete,
                    [
                        "rows",
                        "comparison_rows",
                        "decision_rows",
                        "validation_partition_rows",
                        "update_rows",
                    ],
                ),
                _claim(
                    task_id,
                    "acquisition_value_score",
                    payload,
                    value,
                    ["comparison_rows", "memory_capacity_violation_count"],
                ),
                _claim(
                    task_id,
                    "priority_specific_benefit_score",
                    payload,
                    priority,
                    ["comparison_rows", "memory_capacity_violation_count"],
                ),
            )
        )
    elif task_id == "exp7200-acquisition-cold-audit":
        complete, promotion = _cold_audit_scores(payload)
        claims.extend(
            (
                _claim(
                    task_id,
                    "acquisition_audit_complete_score",
                    payload,
                    complete,
                    [
                        "rows",
                        "causal_control_rows",
                        "cold_reload_rows",
                        "rollback_rows",
                        "whole_learning_reset_rows",
                        "mutation_rows",
                    ],
                ),
                _claim(
                    task_id,
                    "memory_promotion_score",
                    payload,
                    promotion,
                    [
                        "upstream_gate_receipt",
                        "whole_learning_reset_rows",
                        "scheduler_control_summary",
                    ],
                ),
            )
        )
    elif task_id == "exp7201-slice-pyo3":
        binding = payload.get("compiled_binding_receipt", {})
        ready = int(
            _row_count(payload, "rows") == 436
            and _row_count(payload, "transition_rows") == 96
            and _all_true(_rows(payload, "transition_rows"))
            and _row_count(payload, "distribution_rows") == 20
            and _all_true(_rows(payload, "distribution_rows"))
            and _row_count(payload, "phase_cost_rows") == 320
            and _all_true(_rows(payload, "e2e_receipts"))
            and binding.get("compiled") is True
            and binding.get("python_fallback_used") is False
            and payload.get("buffer_reuse_receipt", {}).get("passed") is True
        )
        claims.append(
            _claim(
                task_id,
                "pyo3_slice_ready_score",
                payload,
                ready,
                [
                    "rows",
                    "transition_rows",
                    "distribution_rows",
                    "phase_cost_rows",
                    "compiled_binding_receipt",
                    "buffer_reuse_receipt",
                    "e2e_receipts",
                ],
            )
        )
    elif task_id == "exp7202-slice-cost-quality":
        quality = _rows(payload, "quality_comparison_rows")
        quality_sufficient = len(quality) == 6 and _all_true(quality)
        complete = int(
            _row_count(payload, "rows") == 1288
            and _row_count(payload, "distribution_rows") == 2
            and _row_count(payload, "cold_initialization_rows") == 18
            and _row_count(payload, "throughput_rows") == 1080
            and _row_count(payload, "quality_rows") == 180
            and len(quality) == 6
            and all(row.get("control_detected") is True for row in _rows(payload, "control_rows"))
        )
        primary = payload.get("primary_gate", {})
        parity = primary.get("parity_passed") is True
        zero_sector = primary.get("zero_sector_violations") is True
        lower = primary.get("latency_speedup_over_subprocess_ci95", {}).get("lower")
        value = int(
            quality_sufficient
            and parity
            and zero_sector
            and isinstance(lower, (int, float))
            and lower > 1.0
        )
        nfr = bool(value and float(lower) >= 10.0)
        claims.extend(
            (
                _claim(
                    task_id,
                    "slice_comparison_complete_score",
                    payload,
                    complete,
                    [
                        "rows",
                        "distribution_rows",
                        "throughput_rows",
                        "quality_rows",
                        "quality_comparison_rows",
                        "control_rows",
                    ],
                ),
                _claim(
                    task_id,
                    "boundary_value_score",
                    payload,
                    value,
                    ["throughput_rows", "quality_comparison_rows", "primary_gate"],
                ),
                _claim(
                    task_id,
                    "nfr_01_10x_met",
                    payload,
                    nfr,
                    ["throughput_rows", "quality_comparison_rows", "primary_gate"],
                ),
            )
        )
    elif task_id == "exp7203-hardware-correction":
        exact = _rows(payload, "exact_law_rows")
        complete = int(
            _row_count(payload, "rows") == 540
            and _row_count(payload, "board_rows") == 3
            and _row_count(payload, "exact_law_rows") == 33
            and _all_true(exact)
            and _row_count(payload, "correction_cost_rows") == 360
            and _row_count(payload, "break_even_rows") == 144
        )
        claims.append(
            _claim(
                task_id,
                "hardware_envelope_complete_score",
                payload,
                complete,
                ["rows", "board_rows", "exact_law_rows", "correction_cost_rows", "break_even_rows"],
            )
        )
    return claims


def _gate_replay(
    task: Mapping[str, Any], evidence: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Re-evaluate structured fields and the separate quarantine condition."""

    rows = []
    for gate in task.get("gated_on") or []:
        producer_id = str(gate.get("upstream"))
        producer = evidence.get(producer_id, {})
        payload = producer.get("payload")
        field = str(gate.get("artifact_field"))
        observed = payload.get(field) if isinstance(payload, Mapping) else None
        structured = gate.get("op") == "==" and observed == gate.get("value")
        quarantine_passed = producer.get("quarantined") is False
        rows.append(
            {
                "consumer": task.get("id"),
                "producer": producer_id,
                "producer_path": producer.get("selected_evidence_path"),
                "field": field,
                "operator": gate.get("op"),
                "expected_value": gate.get("value"),
                "observed_value": observed,
                "structured_field_passed": structured,
                "quarantine_passed": quarantine_passed,
                "passed": structured and quarantine_passed,
            }
        )
    return rows


def _source_hash_shape(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Count valid producer source hashes without trusting arbitrary strings."""

    hashes = payload.get("source_artifact_hashes")
    if not isinstance(hashes, Mapping):
        return {"type": type(hashes).__name__, "entry_count": 0, "valid_sha256_count": 0}
    valid = sum(
        isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None
        for value in hashes.values()
    )
    return {"type": "object", "entry_count": len(hashes), "valid_sha256_count": valid}


def _boundaries(task_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep pilot, circular, live, learning, and deployment evidence distinct."""

    return {
        "measurement_completed": payload.get("status") == "complete",
        "method_value_established": task_id in {"exp7201-slice-pyo3"},
        "execution_grounded_circular_gain": task_id
        in {
            "exp7195-typed-grounding",
            "exp7198-feedback-capacity-stream",
            "exp7203-hardware-correction",
        },
        "source_semantics_value_established": task_id == "exp7197-grounding-value-audit"
        and payload.get("grounding_value_score") == 1,
        "live_generalization_efficacy_established": task_id == "exp7193-arc-direct-tool"
        and payload.get("arc_volume_sufficient_score") == 1
        and payload.get("new_solve_claimed") is True,
        "useful_continual_learning": task_id
        in {
            "exp7199-bounded-acquisition",
            "exp7200-acquisition-cold-audit",
        }
        and payload.get("acquisition_value_score", payload.get("memory_promotion_score")) == 1,
        "measured_production_deployment": False,
        "hardware_execution_claimed": bool(payload.get("hardware_execution_claimed", False)),
        "current_task_model_invoked": False,
        "upstream_llm_evidence_scope": (
            "live_qwen_generation"
            if task_id in {"exp7193-arc-direct-tool", "exp7196-qwen-atomic-capture"}
            else "cached_or_no_llm"
        ),
        "limits": {
            "source_pilot_not_general_verification": task_id
            in {
                "exp7195-typed-grounding",
                "exp7196-qwen-atomic-capture",
                "exp7197-grounding-value-audit",
            },
            "memory_pilot_not_foundation_model_learning": task_id
            in {
                "exp7198-feedback-capacity-stream",
                "exp7199-bounded-acquisition",
                "exp7200-acquisition-cold-audit",
            },
            "cpu_or_host_receipt_not_device_deployment": task_id
            in {"exp7201-slice-pyo3", "exp7202-slice-cost-quality", "exp7203-hardware-correction"},
        },
    }


ACTION_MAP = {
    "exp7192-source-contract": "continue",
    "exp7193-arc-direct-tool": "needs_changed_prerequisite",
    "exp7194-arc-gap-audit": "retire",
    "exp7195-typed-grounding": "continue",
    "exp7196-qwen-atomic-capture": "retire",
    "exp7197-grounding-value-audit": "retire",
    "exp7198-feedback-capacity-stream": "continue",
    "exp7199-bounded-acquisition": "retire",
    "exp7200-acquisition-cold-audit": "retire",
    "exp7201-slice-pyo3": "continue",
    "exp7202-slice-cost-quality": "retire",
    "exp7203-hardware-correction": "needs_changed_prerequisite",
    "exp7204-capstone": "retire",
}

ACTION_REASONS = {
    "exp7192-source-contract": "Keep the immutable V634 contract as the authority for this completed milestone.",
    "exp7193-arc-direct-tool": "Require enough independent live sessions and a nondegenerate world model before any efficacy conclusion.",
    "exp7194-arc-gap-audit": "Retire missing-tool and shared banked-credit explanations; the engaged session requested no missing tool and banked progress was noncausal.",
    "exp7195-typed-grounding": "Keep the exact typed executor as a pilot fixture; its same-authority readiness is not verifier value.",
    "exp7196-qwen-atomic-capture": "Retire the current atomic prompt and syntax contract because source and claim parsing remained poor.",
    "exp7197-grounding-value-audit": "Retire this typed extraction policy because it did not pass either frozen value branch.",
    "exp7198-feedback-capacity-stream": "Keep the sealed stream as a bounded information-capacity fixture without a learning claim.",
    "exp7199-bounded-acquisition": "Retire the current acquisition policy because it failed the frozen primary-cell value gate.",
    "exp7200-acquisition-cold-audit": "Retire memory promotion for this policy because the producer value stayed null despite complete causal controls.",
    "exp7201-slice-pyo3": "Continue the persistent compiled boundary within its measured parity and bridge-overhead scope.",
    "exp7202-slice-cost-quality": "Retire the current 10x production boundary claim because quality was insufficient and NFR-01 was not met.",
    "exp7203-hardware-correction": "Require a new dated GateMate physical-state receipt and measured device timing before deployment conclusions.",
    "exp7204-capstone": "Retire this one-time milestone synthesis after the complete matrix is preserved.",
}


def _branch_decision(task: Mapping[str, Any], matrix_row: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve exact prior records and avoid text-normalized recurrence claims."""

    task_id = str(task.get("id"))
    prior = deepcopy(task.get("prior_failures") or [])
    verdict = matrix_row.get("honest_verdict")
    recurrence = any(item.get("verdict") == verdict for item in prior)
    action = ACTION_MAP[task_id]
    changed = None
    if task_id == "exp7193-arc-direct-tool":
        changed = "At least ten independent new live inductions with nondegenerate executable-world-model evidence."
    elif task_id == "exp7203-hardware-correction":
        changed = "A dated post-Exp6559 GateMate physical-state receipt and measured board timing."
    return {
        "task_id": task_id,
        "action": action,
        "reason": ACTION_REASONS[task_id],
        "evidence_path": matrix_row.get("selected_evidence_path"),
        "prior_failures": prior,
        "exact_same_verdict_recurrence": recurrence,
        "retirement_signal": ACTION_REASONS[task_id] if action == "retire" else None,
        "changed_prerequisite": changed,
    }


def run_publication_gate(root: Path) -> dict[str, Any]:
    """Run the shipped G1-G4 command with a hard timeout and parse its JSON."""

    command = [sys.executable, "-u", "scripts/publication_gate.py", "--json"]
    progress(6, "before", "publication_gate subprocess")
    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    progress(6, "after", f"publication_gate subprocess returncode={completed.returncode}")
    if completed.returncode != 0:
        raise RuntimeError(f"publication gate failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"publication gate returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or tuple(payload.get("gates", {})) != GATE_IDS:
        raise RuntimeError("publication gate did not return unchanged G1-G4 fields")
    return payload


def _publication_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add the operator boundary without changing the shipped gate definitions."""

    return {**deepcopy(dict(payload)), "operator_only": True, "definitions_unchanged": True}


def initialize_artifact(run_date: str) -> dict[str, Any]:
    """Create a schema-complete shell before reading any producer evidence."""

    try:
        parsed = datetime.strptime(run_date, "%Y%m%d")
    except (TypeError, ValueError) as exc:
        raise ValueError("run date must be 20260911") from exc
    if parsed.strftime("%Y%m%d") != run_date or run_date != RUN_DATE:
        raise ValueError("run date must be 20260911")
    artifact: dict[str, Any] = {
        "schema": "carnot.exp7204.v634_capstone.v1",
        "experiment_id": "exp7204-capstone",
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": run_date,
        "preconditions_checked": [],
        "inference_substrate": "blocked_no_run",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "execution_host": platform.node(),
        "duration_s": 0.0,
        "source_artifact_hashes": {},
        "rows": [],
        "sample_size_budget": {
            "planned_task_rows": 13,
            "completed_task_rows": 0,
            "independent_units": 0,
            "exclusions": [],
        },
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": None,
        "gate_check_summary": {
            "passed": False,
            "failed_check": "driving_capability_spec",
            "upstream": SPEC_PATH,
            "field": "REQ-REPORT-7204",
            "expected_value": "REQ-REPORT-7204 present",
            "observed_value": False,
        },
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external: capstone preconditions have not completed",
        "capstone_complete_score": 0,
        "evidence_matrix": [],
        "recomputed_claim_rows": [],
        "branch_decisions": [],
        "scope_reduction_compliance": {
            "all_contracted_tasks_preserved": False,
            "protected_files_modified": False,
        },
        "publication_gate": {},
        "e2e_receipts": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "model_invocation_count": 0,
        "contract_source_rows": [],
        "selected_contract_path": None,
        "selected_contract_rows": [],
        "same_milestone_gate_replay_rows": [],
        "known_failed_value_receipts": [],
        "prd_completion": {},
        "validation_command_rows": [],
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash stable evidence while allowing measured runtime to vary honestly."""

    stable = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)


def _check_row(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool,
) -> dict[str, Any]:
    """Give every precondition the same exact diagnostic fields."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": expected,
        "observed_value": observed,
        "passed": passed,
    }


def _preconditions(
    root: Path, output: Path, checkpoint: Path
) -> tuple[list[dict[str, Any]], dict[str, str | None], dict[str, Any] | None]:
    """Check local bytes, tools, and destinations before evidence aggregation."""

    checks: list[dict[str, Any]] = []
    hashes: dict[str, str | None] = {}
    progress(0, "check", SPEC_PATH)
    spec = root / SPEC_PATH
    text = spec.read_text(encoding="utf-8") if spec.is_file() else ""
    present = "### REQ-REPORT-7204:" in text
    row = _check_row(
        "driving_capability_spec",
        SPEC_PATH,
        "REQ-REPORT-7204",
        "REQ-REPORT-7204 present",
        present,
        present,
    )
    checks.append(row)
    if not present:
        return checks, hashes, row
    for relative in SOURCE_PATHS:
        progress(0, "check", relative)
        path = root / relative
        size = path.stat().st_size if path.is_file() else 0
        hashes[relative] = sha256_path(path) if size else None
        row = _check_row(
            "required_source_bytes",
            relative,
            "size_bytes",
            "nonempty_file",
            size,
            size > 0,
        )
        checks.append(row)
        if size == 0:
            return checks, hashes, row
    for destination, field in (
        (output.parent, "artifact_parent"),
        (checkpoint.parent, "checkpoint_parent"),
    ):
        progress(0, "check", str(destination))
        try:
            destination.mkdir(parents=True, exist_ok=True)
            writable = destination.is_dir()
        except OSError:
            writable = False
        row = _check_row(
            "output_directory",
            str(destination),
            field,
            "writable_directory",
            writable,
            writable,
        )
        checks.append(row)
        if not writable:
            return checks, hashes, row
    tools = {
        "python": Path(sys.executable).is_file(),
        "adversarial_verify.py": (root / "scripts/adversarial_verify.py").is_file(),
        "verdict_row_consistency_lint.py": (
            root / "scripts/verdict_row_consistency_lint.py"
        ).is_file(),
        "publication_gate.py": (root / "scripts/publication_gate.py").is_file(),
    }
    progress(0, "check", "required tools")
    row = _check_row(
        "required_tools",
        "host_and_repository",
        "python_and_validation_tools",
        {name: True for name in tools},
        tools,
        all(tools.values()),
    )
    checks.append(row)
    return checks, hashes, None if row["passed"] else row


def _blocked_artifact(
    artifact: dict[str, Any], failed: Mapping[str, Any], started: float
) -> dict[str, Any]:
    """Finish an external precondition block without claiming aggregation ran."""

    artifact["gate_check_summary"] = {
        "passed": False,
        "failed_check": failed.get("check"),
        "upstream": failed.get("upstream"),
        "field": failed.get("field"),
        "expected_value": failed.get("expected_value"),
        "observed_value": failed.get("observed_value"),
    }
    artifact["honest_verdict"] = (
        "blocked_external: required capstone precondition failed before aggregation"
    )
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _first_scientific_block(
    matrix: Sequence[Mapping[str, Any]], gate_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any] | None:
    """Return the first exact missing, quarantined, blocked, or failed-gate receipt."""

    for row in matrix[:-1]:
        if row.get("evidence_source") in {"missing", "unreadable"}:
            return {
                "passed": False,
                "failed_check": "upstream_terminal_evidence",
                "upstream": row.get("task_id"),
                "field": "declared_deliverable_path",
                "expected_value": "readable_terminal_artifact",
                "observed_value": row.get("selected_evidence_path"),
            }
        if row.get("quarantined") is True:
            return {
                "passed": False,
                "failed_check": "upstream_quarantine",
                "upstream": row.get("task_id"),
                "field": "declared_flags,exclusion_manifest_match",
                "expected_value": False,
                "observed_value": row.get("quarantine_receipt"),
            }
        if row.get("verdict_class") == "blocked":
            gate = row.get("producer_gate_check_summary")
            if isinstance(gate, Mapping):
                return {
                    "passed": False,
                    "failed_check": gate.get("failed_check") or "upstream_terminal_evidence",
                    "upstream": row.get("task_id"),
                    "field": gate.get("field"),
                    "expected_value": gate.get("expected_value"),
                    "observed_value": gate.get("observed_value"),
                }
    failed_gate = next((row for row in gate_rows if row.get("passed") is not True), None)
    if failed_gate is not None:
        return {
            "passed": False,
            "failed_check": "same_milestone_gate",
            "upstream": failed_gate.get("producer"),
            "field": failed_gate.get("field"),
            "expected_value": failed_gate.get("expected_value"),
            "observed_value": failed_gate.get("observed_value"),
        }
    return None


def _e2e_receipts(evidence: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Point each promoted mechanism statement to its producer integration receipt."""

    def payload(task_id: str) -> Mapping[str, Any]:
        value = evidence[task_id].get("payload")
        return value if isinstance(value, Mapping) else {}

    arc = payload("exp7193-arc-direct-tool")
    grounding = payload("exp7197-grounding-value-audit")
    memory = payload("exp7200-acquisition-cold-audit")
    binding = payload("exp7201-slice-pyo3")
    planning = payload("exp7192-source-contract")
    return [
        {
            "e2e_ids": ["E2E-009", "E2E-010"],
            "scope": "ARC transport and cross-call seams; no efficacy promotion",
            "source": EXPECTED_CONTRACT[1][2],
            "passed": arc.get("adapter_isolation_receipt", {}).get("passed") is True
            and arc.get("arc_tool_measurement_complete_score") == 1,
        },
        {
            "e2e_ids": ["source-grounding-public-to-independent-score"],
            "scope": "public input through separated extraction, typed execution, and independent label scoring",
            "source": EXPECTED_CONTRACT[5][2],
            "passed": grounding.get("grounding_audit_complete_score") == 1
            and len(_rows(grounding, "cold_audit_rows")) == 128,
        },
        {
            "e2e_ids": ["E2E-007"],
            "scope": "bounded updates, cold reload, rollback, and whole-learning reset",
            "source": EXPECTED_CONTRACT[8][2],
            "passed": memory.get("acquisition_audit_complete_score") == 1,
        },
        {
            "e2e_ids": ["E2E-003", "E2E-004"],
            "scope": "compiled PyO3 round trip and serialized restart state",
            "source": EXPECTED_CONTRACT[9][2],
            "passed": _all_true(_rows(binding, "e2e_receipts")),
        },
        {
            "e2e_ids": ["planning-file-parser-real-gate"],
            "scope": "frozen file through independent parser and real gate evaluator fixtures",
            "source": EXPECTED_CONTRACT[0][2],
            "passed": bool(_rows(planning, "activation_validation_rows"))
            and all(
                row.get("passed") is True for row in _rows(planning, "activation_validation_rows")
            ),
        },
    ]


def validate_artifact(artifact: Mapping[str, Any], root: Path | None = None) -> list[str]:
    """Recompute the capstone roster, derived decisions, gate shape, and checksum."""

    errors: list[str] = []
    missing = sorted(REQUIRED_ARTIFACT_FIELDS - artifact.keys())
    if missing:
        errors.append("missing_required_fields:" + ",".join(missing))
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_invoked") is not False:
        errors.append("model_invocation_contract")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("inference_substrate_class") == "blocked_no_run":
        if artifact.get("status") != "blocked" or artifact.get("capstone_complete_score") != 0:
            errors.append("blocked_preflight_state")
        return errors
    matrix = artifact.get("evidence_matrix")
    rows = artifact.get("rows")
    if not isinstance(matrix, list) or len(matrix) != 13:
        errors.append("evidence_matrix_roster")
        matrix = []
    claims = artifact.get("recomputed_claim_rows")
    if rows != claims:
        errors.append("rows_recomputed_claim_parity")
    if [row.get("task_id") for row in matrix] != list(EXPECTED_TASK_IDS):
        errors.append("evidence_matrix_order")
    decisions = artifact.get("branch_decisions")
    if not isinstance(decisions, list) or [row.get("task_id") for row in decisions] != list(
        EXPECTED_TASK_IDS
    ):
        errors.append("branch_decision_roster")
    elif any(row.get("action") not in BRANCH_ACTIONS for row in decisions):
        errors.append("branch_decision_action")
    if not isinstance(claims, list) or not claims:
        errors.append("recomputed_claim_rows")
    elif {row.get("task_id") for row in claims} != set(EXPECTED_TASK_IDS[:-1]):
        errors.append("recomputed_claim_task_roster")
    elif not all(row.get("matches") is True for row in claims):
        errors.append("recomputed_claim_mismatch")
    publication = artifact.get("publication_gate")
    if not isinstance(publication, Mapping) or tuple(publication.get("gates", {})) != GATE_IDS:
        errors.append("publication_gate_shape")
    elif publication.get("paper_ready") is not all(
        publication["gates"][gate].get("pass") is True for gate in GATE_IDS
    ):
        errors.append("publication_gate_derivation")
    if artifact.get("capstone_complete_score") != 1:
        errors.append("capstone_complete_score")
    prd = artifact.get("prd_completion", {})
    external_block = bool(artifact.get("gate_check_summary", {}).get("passed") is False)
    expected_class = (
        "blocked" if external_block else ("positive" if prd and all(prd.values()) else "null")
    )
    if artifact.get("verdict_class") != expected_class:
        errors.append("verdict_class")
    if artifact.get("inference_substrate") != "aggregation_from_upstream_artifacts":
        errors.append("inference_substrate")
    if artifact.get("inference_substrate_class") != "aggregation":
        errors.append("inference_substrate_class")
    if root is not None and matrix:
        for row in matrix[:-1]:
            relative = row.get("selected_evidence_path")
            expected_hash = row.get("artifact_sha256")
            if not isinstance(relative, str) or not (root / relative).is_file():
                errors.append("producer_path_replay")
                break
            if sha256_path(root / relative) != expected_hash:
                errors.append("producer_hash_replay")
                break
    return errors


def build_artifact(
    root: Path,
    run_date: str,
    output_path: Path,
    checkpoint_path: Path,
    checker_loader: CheckerLoader = default_checker_loader,
    publication_runner: PublicationRunner = run_publication_gate,
) -> dict[str, Any]:
    """Build, atomically write, reload, and validate the V634 capstone."""

    started = time.monotonic()
    progress(0, "start", "preconditions")
    artifact = initialize_artifact(run_date)
    write_atomic(checkpoint_path, artifact)
    checks, hashes, failed = _preconditions(root, output_path, checkpoint_path)
    artifact["preconditions_checked"] = checks
    artifact["source_artifact_hashes"] = hashes
    if failed is not None:
        blocked = _blocked_artifact(artifact, failed, started)
        write_atomic(checkpoint_path, blocked)
        write_atomic(output_path, blocked)
        progress(0, "end", "terminal blocked artifact written")
        return blocked
    progress(0, "end", "local sources, tools, and destinations ready")

    progress(1, "start", "no-LLM aggregation contract")
    progress(1, "end", "MODEL_SPECS empty and no model invoked")

    progress(2, "start", "resolve frozen V634 contract")
    tasks, contract_sources, selected, contract_errors = resolve_contract(root)
    artifact["contract_source_rows"] = contract_sources
    artifact["selected_contract_path"] = selected
    artifact["selected_contract_rows"] = [
        {
            "order": index,
            "id": task.get("id"),
            "title": task.get("title"),
            "deliverable": task.get("deliverable"),
            "milestone": task.get("milestone"),
            "gated_on": task.get("gated_on") or [],
            "prior_failures": task.get("prior_failures") or [],
        }
        for index, task in enumerate(tasks, 1)
    ]
    contract_row = _check_row(
        "v634_contract_identity",
        selected or "missing",
        "milestone,id_order,title,deliverable,gated_on",
        {"milestone": MILESTONE, "id_order": list(EXPECTED_TASK_IDS)},
        {
            "milestone": MILESTONE if not contract_errors else None,
            "id_order": [task.get("id") for task in tasks],
            "errors": contract_errors,
        },
        not contract_errors,
    )
    artifact["preconditions_checked"].append(contract_row)
    if contract_errors:
        blocked = _blocked_artifact(artifact, contract_row, started)
        write_atomic(checkpoint_path, blocked)
        write_atomic(output_path, blocked)
        progress(2, "end", "contract failed closed")
        return blocked
    if selected is not None:
        hashes[selected] = sha256_path(root / selected)
    progress(2, "end", f"selected {selected} with 13 tasks")

    progress(3, "start", "read declared deliverables and authenticate evidence")
    manifest = yaml.safe_load((root / "ops/exclusion_manifest.yaml").read_text(encoding="utf-8"))
    loaded: dict[str, dict[str, Any]] = {}
    for task in tasks[:-1]:
        task_id = str(task["id"])
        progress(3, "before", f"read {task_id} declared deliverable")
        loaded[task_id] = load_task_evidence(root, task, manifest)
        progress(3, "after", f"read {task_id} declared deliverable")
    verify, row_check = checker_loader(root)
    matrix: list[dict[str, Any]] = []
    claims: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for order, task in enumerate(tasks[:-1], 1):
        task_id = str(task["id"])
        evidence = loaded[task_id]
        payload = evidence.get("payload") or {}
        selected_path = evidence.get("selected_evidence_path")
        replay = _gate_replay(task, loaded)
        gate_rows.extend(replay)
        progress(3, "before", f"protected verifiers {task_id}")
        if selected_path is not None and evidence.get("read_error") is None:
            adversarial = dict(verify(root / str(selected_path)))
            row_status, row_findings = row_check(root / str(selected_path))
        else:
            adversarial = {"loaded": False, "flag_count": 0, "max_severity": -1, "flags": []}
            row_status, row_findings = "skipped", ["artifact unavailable"]
        progress(3, "after", f"protected verifiers {task_id}")
        task_claims = recompute_claims(task_id, payload) if payload else []
        if evidence.get("accepted_for_promoted_evidence"):
            claims.extend(task_claims)
        quarantine = {
            "declared_flags": evidence.get("declared_flags"),
            "exclusion_manifest_match": evidence.get("exclusion_manifest_match"),
        }
        matrix.append(
            {
                "unit_id": task_id,
                "arm": "upstream_artifact_evidence",
                "seed": RANDOM_SEED,
                "metric": "terminal_evidence_and_claim_replay",
                "metric_value": int(
                    evidence.get("accepted_for_promoted_evidence") is True
                    and bool(task_claims)
                    and all(row["matches"] for row in task_claims)
                ),
                "error": evidence.get("read_error"),
                "abstention": not bool(evidence.get("accepted_for_promoted_evidence")),
                "order": order,
                "task_id": task_id,
                "title": task.get("title"),
                "declared_deliverable_path": evidence.get("declared_deliverable_path"),
                "canonical_gate_block_path": evidence.get("canonical_gate_block_path"),
                "selected_evidence_path": selected_path,
                "evidence_source": evidence.get("evidence_source"),
                "artifact_sha256": evidence.get("artifact_sha256"),
                "artifact_size_bytes": evidence.get("artifact_size_bytes"),
                "status": payload.get("status", "missing"),
                "verdict_class": payload.get("verdict_class", "blocked"),
                "honest_verdict": payload.get("honest_verdict", "missing_evidence"),
                "inference_substrate": payload.get("inference_substrate"),
                "inference_substrate_class": payload.get("inference_substrate_class"),
                "execution_venue": payload.get("execution_venue"),
                "raw_row_count": _row_count(payload, "rows"),
                "source_hash_receipt": _source_hash_shape(payload),
                "quarantined": evidence.get("quarantined"),
                "quarantine_receipt": quarantine,
                "accepted_for_promoted_evidence": evidence.get("accepted_for_promoted_evidence"),
                "authentication_flags": {
                    "flag_count": adversarial.get("flag_count", 0),
                    "max_severity": adversarial.get("max_severity", -1),
                    "flags": adversarial.get("flags", []),
                    "gate_version": adversarial.get("gate_version"),
                    "row_consistency_status": row_status,
                    "row_consistency_findings": row_findings,
                },
                "acceptance_gates": task.get("gated_on") or [],
                "acceptance_gate_replay_rows": replay,
                "producer_gate_check_summary": payload.get("gate_check_summary"),
                "promoted_claims_match_rows": bool(task_claims)
                and all(row["matches"] for row in task_claims),
                **_boundaries(task_id, payload),
            }
        )
    progress(3, "end", f"authenticated 12 producers and recomputed {len(claims)} claims")

    self_row = {
        "unit_id": "exp7204-capstone",
        "arm": "self_aggregation",
        "seed": RANDOM_SEED,
        "metric": "complete_evidence_matrix",
        "metric_value": 1,
        "error": None,
        "abstention": False,
        "order": 13,
        "task_id": "exp7204-capstone",
        "title": tasks[-1].get("title"),
        "declared_deliverable_path": tasks[-1].get("deliverable"),
        "canonical_gate_block_path": canonical_gate_block_path("exp7204-capstone"),
        "selected_evidence_path": (
            str(output_path.relative_to(root))
            if output_path.is_relative_to(root)
            else str(output_path)
        ),
        "evidence_source": "self",
        "artifact_sha256": None,
        "artifact_size_bytes": None,
        "status": "complete",
        "verdict_class": "null",
        "honest_verdict": "complete_null: V634 matrix complete; no PRD outcome gate passed",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "execution_venue": "host",
        "raw_row_count": 13,
        "source_hash_receipt": {
            "type": "object",
            "entry_count": len(hashes),
            "valid_sha256_count": sum(value is not None for value in hashes.values()),
        },
        "quarantined": False,
        "quarantine_receipt": {"declared_flags": {}, "exclusion_manifest_match": False},
        "accepted_for_promoted_evidence": True,
        "authentication_flags": {
            "flag_count": 0,
            "max_severity": -1,
            "flags": [],
            "gate_version": None,
            "row_consistency_status": "self_validated",
            "row_consistency_findings": [],
        },
        "acceptance_gates": [],
        "acceptance_gate_replay_rows": [],
        "producer_gate_check_summary": None,
        "promoted_claims_match_rows": True,
        "measurement_completed": True,
        "method_value_established": False,
        "execution_grounded_circular_gain": False,
        "source_semantics_value_established": False,
        "live_generalization_efficacy_established": False,
        "useful_continual_learning": False,
        "measured_production_deployment": False,
        "hardware_execution_claimed": False,
        "current_task_model_invoked": False,
        "upstream_llm_evidence_scope": "cited_only",
        "limits": {
            "source_pilot_not_general_verification": True,
            "memory_pilot_not_foundation_model_learning": True,
            "cpu_or_host_receipt_not_device_deployment": True,
        },
    }
    matrix.append(self_row)
    artifact["same_milestone_gate_replay_rows"] = gate_rows
    artifact["recomputed_claim_rows"] = claims
    artifact["evidence_matrix"] = matrix
    artifact["rows"] = claims
    artifact["sample_size_budget"] = {
        "planned_task_rows": 13,
        "completed_task_rows": len(matrix),
        "independent_units": 13,
        "producer_raw_rows": sum(int(row["raw_row_count"]) for row in matrix[:-1]),
        "exclusions": [row["task_id"] for row in matrix if row.get("quarantined")],
    }
    for row in matrix[:-1]:
        if row.get("selected_evidence_path") and row.get("artifact_sha256"):
            hashes[str(row["selected_evidence_path"])] = str(row["artifact_sha256"])
    artifact["source_artifact_hashes"] = hashes
    write_atomic(checkpoint_path, artifact)

    progress(4, "start", "issue one bounded branch decision per task")
    artifact["branch_decisions"] = [
        _branch_decision(task, matrix[index]) for index, task in enumerate(tasks)
    ]
    artifact["known_failed_value_receipts"] = [
        {
            "task_id": row["task_id"],
            "field": row["claim"],
            "observed_value": row["recomputed_value"],
            "promoted_as_positive": False,
        }
        for row in claims
        if row["recomputed_value"] in (0, False)
        and any(
            marker in row["claim"] for marker in ("value", "volume", "benefit", "promotion", "nfr")
        )
    ]
    artifact["scope_reduction_compliance"] = {
        "all_contracted_tasks_preserved": True,
        "contracted_task_count": 13,
        "represented_task_count": len(matrix),
        "protected_files_modified": False,
        "active_priorities": [
            "source semantics",
            "useful continual learning",
            "real live generalization",
            "measured production deployment",
        ],
        "retired_boundaries_preserved": True,
        "limits": [
            "A source extraction pilot is not general verification.",
            "A bounded memory pilot is not foundation-model learning.",
            "Tool engagement is not live generalization efficacy.",
            "CPU correction cost is not measured device deployment.",
        ],
    }
    progress(4, "end", "all 13 tasks have one allowed action")

    progress(5, "start", "separate matrix completion from PRD completion")
    artifact["prd_completion"] = {
        "source_semantics": any(row["source_semantics_value_established"] for row in matrix),
        "useful_continual_learning": any(row["useful_continual_learning"] for row in matrix),
        "real_live_generalization": any(
            row["live_generalization_efficacy_established"] for row in matrix
        ),
        "measured_production_deployment": any(
            row["measured_production_deployment"] for row in matrix
        ),
    }
    scientific_block = _first_scientific_block(matrix, gate_rows)
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": "aggregation_from_upstream_artifacts",
            "inference_substrate_class": "aggregation",
            "capstone_complete_score": 1,
            "gate_check_summary": scientific_block
            or {
                "passed": True,
                "failed_check": None,
                "upstream": None,
                "field": None,
                "expected_value": "all_declared_v634_artifacts_terminal_and_unquarantined",
                "observed_value": "all_declared_v634_artifacts_terminal_and_unquarantined",
            },
            "verdict_class": (
                "blocked"
                if scientific_block
                else ("positive" if all(artifact["prd_completion"].values()) else "null")
            ),
            "honest_verdict": (
                "blocked_external: V634 matrix is complete but a scientific input has a diagnosed external block"
                if scientific_block
                else "complete_null: V634 evidence matrix is complete; source semantics, useful continual learning, live generalization efficacy, and measured production deployment remain incomplete"
            ),
        }
    )
    self_row["verdict_class"] = artifact["verdict_class"]
    self_row["honest_verdict"] = artifact["honest_verdict"]
    progress(5, "end", f"verdict_class={artifact['verdict_class']}")

    progress(6, "start", "capture unchanged publication gate")
    artifact["publication_gate"] = _publication_receipt(publication_runner(root))
    progress(6, "end", "captured G1-G4 without redefining publication readiness")

    progress(7, "start", "record scoped integration and validation receipts")
    artifact["e2e_receipts"] = _e2e_receipts(loaded)
    artifact["validation_command_rows"] = [
        {
            "check": "producer_e2e_receipts",
            "scope": "E2E-003, E2E-004, E2E-007, E2E-009, E2E-010, source grounding, and planning",
            "passed": all(row["passed"] for row in artifact["e2e_receipts"]),
        },
        {
            "check": "current_task_inference",
            "scope": "aggregation only",
            "passed": True,
            "MODEL_SPECS": [],
            "model_invoked": False,
        },
    ]
    artifact["duration_s"] = time.monotonic() - started
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact, root=root)
    if errors:
        raise RuntimeError("capstone validation failed: " + ",".join(errors))
    progress(7, "end", "derived roster, claims, decisions, gates, and checksum pass")

    progress(8, "start", "final atomic write")
    write_atomic(output_path, artifact)
    reloaded, read_error = read_json_object(output_path)
    final_errors = [read_error] if read_error else validate_artifact(reloaded or {}, root=root)
    if final_errors:
        raise RuntimeError(
            "final file-to-parser gate failed: " + ",".join(str(error) for error in final_errors)
        )
    progress(8, "end", f"wrote {output_path}")
    return artifact


def main(
    argv: Sequence[str] | None = None,
    checker_loader: CheckerLoader = default_checker_loader,
) -> int:
    """Build the terminal artifact or validate one existing result file."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.artifact_path or root / DEFAULT_ARTIFACT_PATH
    checkpoint = args.checkpoint_path or root / DEFAULT_CHECKPOINT_PATH
    if args.validate:
        progress(7, "start", f"validate {output}")
        payload, read_error = read_json_object(output)
        errors = [read_error] if read_error else validate_artifact(payload or {}, root=root)
        progress(7, "end", "passed" if not errors else ";".join(str(error) for error in errors))
        return 0 if not errors else 1
    build_artifact(
        root,
        args.date,
        output,
        checkpoint,
        checker_loader=checker_loader,
        publication_runner=run_publication_gate,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - tests call main with explicit arguments.
    raise SystemExit(main())

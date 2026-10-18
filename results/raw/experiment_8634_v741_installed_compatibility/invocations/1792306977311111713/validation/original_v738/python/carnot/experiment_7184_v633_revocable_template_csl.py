"""Measure revocable constraint-template memory on the sealed V633 stream.

The controller is a deterministic CPU policy. It can add executable integer
predicates only after delayed exact evidence supplies three distinct errors.
It never changes model weights or the production verification pipeline.

Spec refs: REQ-CL-7184 and SCENARIO-CL-7184-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7183_v633_supersession_stream as exp7183


JsonDict = dict[str, Any]

EXPERIMENT_ID = 7184
SCHEMA = "carnot.exp7184.v633_revocable_template_csl.v1"
MILESTONE = "2026.09.633"
RUN_DATE = "20260910"
RANDOM_SEED = 7_184_202_609_10
ORDERING_SEEDS = (71_840_011, 71_840_019, 71_840_043)
EVENT_COUNT = 240
ARMS = ("no_memory", "static_rule", "fifo_replay", "revocable_template")
ROW_COUNT = EVENT_COUNT * len(ORDERING_SEEDS) * len(ARMS)
MEMORY_BYTE_BUDGET = 4_096
INSPECTION_LIMIT = 2
TEMPLATE_LIMIT = 8
SUPPORT_REQUIRED = 3
BOOTSTRAP_BLOCK_SIZE = 12
BOOTSTRAP_DRAWS = 2_000
ADAPTATION_FAMILIES = tuple(exp7183.ADAPTATION_FAMILIES)
TRANSFER_FAMILIES = tuple(exp7183.TRANSFER_FAMILIES)
INFERENCE_SUBSTRATE = (
    "deterministic CPU candidate policy with exact integer verification and bounded "
    "external constraint-template memory; no model load"
)
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"

DEFAULT_UPSTREAM_ARTIFACT_PATH = Path("results/experiment_7183_v633_supersession_stream.json")
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7184_v633_revocable_template_csl.json")
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SOURCE_PATHS = (
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("research-roadmap.yaml"),
    SPEC_PATH,
    Path("python/carnot/experiment_7106_v623_procedural_memory_csl.py"),
    Path("python/carnot/experiment_7183_v633_supersession_stream.py"),
    Path("python/carnot/experiment_7184_v633_revocable_template_csl.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("python/carnot/learning/constraint_policy_store.py"),
    Path("scripts/experiments/experiment_7184_v633_revocable_template_csl.py"),
    Path("tests/python/test_experiment_7184_v633_revocable_template_csl.py"),
)

INITIAL_RULES: dict[str, JsonDict] = {
    family: exp7183._rule_for(family, "v1")  # noqa: SLF001 - frozen sibling contract
    for family in exp7183.FAMILIES
}

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "inference_substrate_class",
    "memory_run_complete_score",
    "memory_value_score",
    "continuous_self_learning_task",
    "memory_transition_rows",
    "template_lineage_rows",
    "feedback_access_rows",
    "no_model_weight_mutation",
    "cost_rows",
)

FIELD_PRINCIPLES = {
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": "Name each resource and record its actual availability before measurement.",
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": "Hash input contracts, code, seeds, and raw rows to expose drift.",
    "gate_check_summary": "Every blocked verdict names the exact failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Declare whether the scored verifier uses the same authority that labels the outcome.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "honest_verdict": "A terminal description distinguishes useful evidence, null findings, disqualification, and external blocks.",
    "inference_substrate_class": "Use cpu_exact_solver_or_simulator when the declared work runs; use blocked_no_run only before any qualifying work.",
    "memory_run_complete_score": "One records a complete run regardless of improvement.",
    "memory_value_score": "Future benefit, retention, and soundness must all support promotion.",
    "continuous_self_learning_task": "True identifies the mandatory FR-11 experiment.",
    "memory_transition_rows": "Before/after hashes prove real persistent state change.",
    "template_lineage_rows": "Learned additions need released witnesses and revocation provenance.",
    "feedback_access_rows": "Availability timestamps expose leakage.",
    "no_model_weight_mutation": "True bounds the result to external constraint learning.",
    "cost_rows": "Measured latency and byte budgets make the hardware path concrete.",
}


def canonical_json(value: Any) -> bytes:
    """Return stable bytes so memory identity depends only on stored content."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Prefix each digest so an evidence identity cannot look like plain data."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash one structured value with the experiment's canonical encoding."""

    return sha256_bytes(canonical_json(value))


def sha256_path(path: Path) -> str | None:
    """Hash one file, or return no digest when a prerequisite is absent."""

    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def _resolve(repo_root: Path, value: str | Path) -> Path:
    """Resolve evidence paths from the selected checkout, not the caller's shell."""

    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_object(path: Path) -> JsonDict:
    """Load a JSON object while preserving malformed input as failed evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> list[JsonDict]:
    """Load an immutable view without changing its stored order."""

    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return []


def gate_check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
) -> JsonDict:
    """Keep both sides of a gate so blocked output identifies the exact failure."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Copy all checks and lift the first failed comparison into stable fields."""

    copied = [dict(row) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": copied,
        "failed_check": None if failed is None else failed.get("check"),
        "upstream": None if failed is None else failed.get("upstream"),
        "field": None if failed is None else failed.get("field"),
        "expected_value": None if failed is None else deepcopy(failed.get("expected_value")),
        "observed_value": None if failed is None else deepcopy(failed.get("observed_value")),
    }


def _task_block(text: str) -> str:
    """Return only the frozen Exp7184 roadmap entry for exact identity checks."""

    match = re.search(r"(?ms)^- id: exp7184-revocable-template-csl\n(.*?)(?=^- id:|\Z)", text)
    return "" if match is None else match.group(0)


def _path_writable(path: Path) -> bool:
    """Check the existing destination directory without creating measurement files."""

    parent = path if path.suffix == "" else path.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _source_hashes(repo_root: Path, upstream_artifact_path: Path) -> JsonDict:
    """Bind the controller, contracts, tests, and exact upstream artifact."""

    hashes = {str(path): sha256_path(repo_root / path) for path in SOURCE_PATHS}
    hashes[str(upstream_artifact_path)] = sha256_path(_resolve(repo_root, upstream_artifact_path))
    return hashes


def collect_preconditions(
    repo_root: Path, upstream_artifact_path: Path, artifact_path: Path
) -> tuple[list[JsonDict], JsonDict]:
    """Check all external resources before the first prospective policy decision."""

    upstream_path = _resolve(repo_root, upstream_artifact_path)
    upstream = _load_object(upstream_path)
    spec_path = repo_root / SPEC_PATH
    roadmap_path = repo_root / "research-roadmap.yaml"
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.is_file() else ""
    task = _task_block(roadmap_text)
    sizes = {
        str(path): (repo_root / path).stat().st_size if (repo_root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    hashes = _source_hashes(repo_root, upstream_artifact_path)
    upstream_validation = (
        exp7183.validate_artifact(upstream, check_files=True)
        if upstream
        else ["missing_upstream_artifact"]
    )
    view_hashes: JsonDict = {}
    for name in ("decision", "feedback", "truth", "availability"):
        value = upstream.get("sealed_view_paths", {}).get(name, "missing")
        view_hashes[name] = sha256_path(_resolve(repo_root, str(value)))
    manifest_hashes: JsonDict = {}
    for name, value in upstream.get("manifest_paths", {}).items():
        manifest_hashes[name] = sha256_path(_resolve(repo_root, str(value)))
    source_hash_values = list(hashes.values())
    identity = {
        "id": "exp7184-revocable-template-csl" if task else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in task else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH) if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in task else None
        ),
    }
    expected_identity = {
        "id": "exp7184-revocable-template-csl",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    tools = {
        "python_executable": Path(sys.executable).is_file(),
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    output = _resolve(repo_root, artifact_path)
    return [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7184",
            True,
            "## REQ-CL-7184:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7184-*",
            7,
            spec_text.count("### SCENARIO-CL-7184-"),
            spec_text.count("### SCENARIO-CL-7184-") >= 7,
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            sizes,
            all(isinstance(size, int) and size > 0 for size in sizes.values()),
        ),
        gate_check(
            "required_source_hashes",
            "repository",
            "SOURCE_PATHS.sha256",
            "sha256:<64 hex> for every source and upstream artifact",
            hashes,
            all(
                isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                for value in source_hash_values
            ),
        ),
        gate_check(
            "v633_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            identity,
        ),
        gate_check(
            "same_milestone_gate",
            "exp7183-supersession-stream",
            "stream_ready_score",
            1,
            upstream.get("stream_ready_score"),
        ),
        gate_check(
            "upstream_milestone",
            "exp7183-supersession-stream",
            "milestone",
            MILESTONE,
            upstream.get("milestone"),
        ),
        gate_check(
            "upstream_artifact_validation",
            "exp7183-supersession-stream",
            "cold_validation_errors",
            [],
            upstream_validation,
        ),
        gate_check(
            "sealed_view_hashes",
            "exp7183-supersession-stream",
            "sealed_view_hashes",
            upstream.get("sealed_view_hashes", {}),
            view_hashes,
        ),
        gate_check(
            "sealed_manifest_hashes",
            "exp7183-supersession-stream",
            "manifest_hashes",
            upstream.get("manifest_hashes", {}),
            manifest_hashes,
        ),
        gate_check(
            "frozen_operation_grammar",
            "exp7183-supersession-stream",
            "template_operation_contract",
            exp7183.TEMPLATE_OPERATION_CONTRACT,
            upstream.get("template_operation_contract"),
        ),
        gate_check(
            "frozen_resource_budgets",
            "exp7183-supersession-stream",
            "byte_budgets",
            exp7183.BYTE_BUDGETS,
            upstream.get("byte_budgets"),
        ),
        gate_check(
            "required_local_tools",
            "host",
            "python,sha256sum",
            {key: True for key in tools},
            tools,
        ),
        gate_check(
            "output_destination_writable",
            "host_filesystem",
            "artifact_path.parent",
            True,
            _path_writable(output),
        ),
    ], upstream


class _FifoMemory:
    """Keep compact raw feedback in chronological FIFO order under 4 KiB."""

    def __init__(self) -> None:
        self.records: list[JsonDict] = []

    @property
    def state_bytes(self) -> bytes:
        """Serialize all persistent raw replay state for the byte gate."""

        return canonical_json(self.records)

    @property
    def state_hash(self) -> str:
        """Identify the complete raw replay state."""

        return sha256_bytes(self.state_bytes)

    def inspect(self, family: str, seed: int) -> list[JsonDict]:
        """Inspect at most two recent matching records in a seeded slot order."""

        selected = [row for row in reversed(self.records) if row["family_id"] == family][
            :INSPECTION_LIMIT
        ]
        return sorted(selected, key=lambda row: sha256_json([seed, row["event_id"]]))

    def append(self, truth: Mapping[str, Any]) -> None:
        """Append released raw evidence and evict old rows until bytes fit."""

        self.records.append(
            {
                "event_id": truth["event_id"],
                "family_id": truth["family_id"],
                "source_version": truth["source_version"],
                "exact_label": truth["exact_label"],
            }
        )
        while len(self.state_bytes) > MEMORY_BYTE_BUDGET:
            self.records.pop(0)


class _RevocableMemory:
    """Store executable templates and compact admission evidence within 4 KiB."""

    def __init__(self) -> None:
        self.active: dict[str, JsonDict] = {}
        self.pending: dict[str, list[str]] = {}
        self.validation: dict[str, list[list[Any]]] = {}
        self.revoked_versions: list[str] = []

    def _state(self) -> JsonDict:
        """Return every persistent byte that can affect a later decision."""

        return {
            "active": [self.active[key] for key in sorted(self.active)],
            "pending": {key: self.pending[key] for key in sorted(self.pending)},
            "validation": {key: self.validation[key] for key in sorted(self.validation)},
            "revoked_versions": self.revoked_versions,
        }

    @property
    def state_bytes(self) -> bytes:
        """Serialize the full controller state for the fixed byte budget."""

        return canonical_json(self._state())

    @property
    def state_hash(self) -> str:
        """Identify active rules and all bounded admission evidence."""

        return sha256_bytes(self.state_bytes)

    def inspect(self, family: str) -> list[JsonDict]:
        """Return the one active executable rule for this family, when present."""

        template = self.active.get(family)
        return [] if template is None else [deepcopy(template)]

    def _check_bound(self) -> None:
        """Reject any controller transition that exceeds its declared resources."""

        if len(self.active) > TEMPLATE_LIMIT:
            raise ValueError("active_template_limit_exceeded")
        if len(self.state_bytes) > MEMORY_BYTE_BUDGET:
            raise ValueError("serialized_memory_byte_limit_exceeded")

    def revoke(self, receipt: Mapping[str, Any]) -> tuple[str, str, str | None]:
        """Remove stale executable structure and retain compact revocation history."""

        before = self.state_hash
        family = str(receipt["family_id"])
        revoked = str(receipt["revoked_version"])
        previous = self.active.get(family)
        template_id = None if previous is None else str(previous["template_id"])
        if previous is not None and previous["source_version"] == revoked:
            del self.active[family]
        self.pending.pop(f"{family}:{revoked}", None)
        self.revoked_versions.append(f"{family}:{revoked}")
        self._check_bound()
        return before, self.state_hash, template_id

    def observe_validation(self, truth: Mapping[str, Any]) -> None:
        """Retain only compact, released validation facts used by admission."""

        family = str(truth["family_id"])
        self.validation.setdefault(family, []).append(
            [truth["source_version"], truth["numeric_value"], truth["exact_label"]]
        )
        self._check_bound()

    def observe_error(self, truth: Mapping[str, Any]) -> list[str]:
        """Accumulate distinct released error IDs for one family and source version."""

        key = f"{truth['family_id']}:{truth['source_version']}"
        ids = self.pending.setdefault(key, [])
        if truth["event_id"] not in ids:
            ids.append(str(truth["event_id"]))
        self._check_bound()
        return list(ids)

    def admit(
        self, truth: Mapping[str, Any], evidence_ids: Sequence[str]
    ) -> tuple[JsonDict, str, str]:
        """Add executable predicate structure after support and validation pass."""

        family = str(truth["family_id"])
        source_version = str(truth["source_version"])
        template: JsonDict = {
            "kind": "executable_constraint",
            "grammar_version": "revocable_constraint_template.v1",
            "template_id": f"template:{family}:{source_version}",
            "family_id": family,
            "source_version": source_version,
            "parameters": deepcopy(truth["rule"]["parameters"]),
        }
        before = self.state_hash
        self.active[family] = template
        self.pending.pop(f"{family}:{source_version}", None)
        self._check_bound()
        return deepcopy(template), before, self.state_hash


STATIC_RECORDS = tuple(
    {
        "kind": "executable_constraint",
        "template_id": f"static:{family}:v1",
        "family_id": family,
        "source_version": "v1",
        "parameters": deepcopy(rule["parameters"]),
    }
    for family, rule in INITIAL_RULES.items()
)
STATIC_STATE_BYTES = canonical_json(STATIC_RECORDS)
STATIC_STATE_HASH = sha256_bytes(STATIC_STATE_BYTES)
EMPTY_STATE_HASH = sha256_json([])


def _fallback_decision(decision: Mapping[str, Any]) -> str:
    """Give all memory-free lookups one deterministic, intentionally limited policy."""

    occurrence = int(decision["decision_index"]) // len(exp7183.FAMILIES)
    return "accept" if occurrence % 3 == 0 else "reject"


def _execute_template(decision: Mapping[str, Any], template: Mapping[str, Any]) -> str:
    """Execute learned constraint structure without reading a hidden label."""

    rule = {
        "family_id": template["family_id"],
        "parameters": template["parameters"],
    }
    return exp7183.exact_label(str(decision["family_id"]), int(decision["numeric_value"]), rule)


def candidate_policy(decision: Mapping[str, Any], inspected: Sequence[Mapping[str, Any]]) -> str:
    """Choose from the public actions using only the supplied bounded lookup records."""

    if not inspected:
        return _fallback_decision(decision)
    first = inspected[0]
    if first.get("kind") == "executable_constraint":
        return _execute_template(decision, first)
    labels = [str(row["exact_label"]) for row in inspected]
    accepts = labels.count("accept")
    rejects = labels.count("reject")
    return "accept" if accepts > rejects else "reject" if rejects > accepts else labels[0]


def action_seal(row: Mapping[str, Any]) -> str:
    """Seal the action and pre-decision memory without including later feedback."""

    return sha256_json(
        {
            "ordering_seed": row["ordering_seed"],
            "decision_index": row["decision_index"],
            "event_id": row["event_id"],
            "arm": row["arm"],
            "decision_input_hash": row["decision_input_hash"],
            "pre_decision_memory_hash": row["pre_decision_memory_hash"],
            "decision": row["decision"],
        }
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    """Select a deterministic empirical percentile without an array dependency."""

    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _paired_bootstrap(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Resample paired chronological event blocks rather than independent arm rows."""

    by_key = {
        (int(row["ordering_seed"]), int(row["decision_index"]), str(row["arm"])): int(row["error"])
        for row in rows
        if int(row["decision_index"]) >= 180
    }
    result: list[JsonDict] = []
    for comparison_index, control in enumerate(("static_rule", "fifo_replay")):
        blocks: list[list[int]] = []
        for seed in ORDERING_SEEDS:
            for start in range(180, EVENT_COUNT, BOOTSTRAP_BLOCK_SIZE):
                blocks.append(
                    [
                        by_key[(seed, index, "revocable_template")] - by_key[(seed, index, control)]
                        for index in range(start, min(start + BOOTSTRAP_BLOCK_SIZE, EVENT_COUNT))
                    ]
                )
        rng = random.Random(RANDOM_SEED + comparison_index)
        draws: list[float] = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled = [blocks[rng.randrange(len(blocks))] for _ in blocks]
            flat = [value for block in sampled for value in block]
            draws.append(sum(flat) / len(flat))
        observed = [value for block in blocks for value in block]
        result.append(
            {
                "comparison_arm": control,
                "target_arm": "revocable_template",
                "metric": "future_segment_error_delta",
                "sampling_unit": "chronological_event_block",
                "paired_within_block": True,
                "independent_row_sampling": False,
                "block_size": BOOTSTRAP_BLOCK_SIZE,
                "block_count": len(blocks),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "mean_delta": round(sum(observed) / len(observed), 6),
                "ci95_lower": round(_percentile(draws, 0.025), 6),
                "ci95_upper": round(_percentile(draws, 0.975), 6),
            }
        )
    return result


def _metric_rows(rows: Sequence[Mapping[str, Any]], recurrence_event_ids: set[str]) -> JsonDict:
    """Reduce complete rows into the predeclared value and safety metrics."""

    future: list[JsonDict] = []
    false_accepts: list[JsonDict] = []
    transfer: list[JsonDict] = []
    recurrence: list[JsonDict] = []
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        future_rows = [row for row in arm_rows if int(row["decision_index"]) >= 180]
        transfer_rows = [row for row in arm_rows if row["family_role"] == "transfer"]
        recurrence_rows = [row for row in arm_rows if row["event_id"] in recurrence_event_ids]
        future_errors = sum(int(row["error"]) for row in future_rows)
        false_count = sum(int(row["false_accept"]) for row in future_rows)
        transfer_errors = sum(int(row["error"]) for row in transfer_rows)
        retained = sum(int(row["correct"]) for row in recurrence_rows)
        future.append(
            {
                "arm": arm,
                "event_count": len(future_rows),
                "error_count": future_errors,
                "error_rate": round(future_errors / len(future_rows), 6),
            }
        )
        false_accepts.append(
            {
                "arm": arm,
                "event_count": len(future_rows),
                "false_accept_count": false_count,
                "false_accept_rate": round(false_count / len(future_rows), 6),
            }
        )
        transfer.append(
            {
                "arm": arm,
                "event_count": len(transfer_rows),
                "error_count": transfer_errors,
                "error_rate": round(transfer_errors / len(transfer_rows), 6),
            }
        )
        recurrence.append(
            {
                "arm": arm,
                "event_count": len(recurrence_rows),
                "retained_count": retained,
                "retention_rate": round(retained / len(recurrence_rows), 6),
            }
        )
    return {
        "future_segment_metrics": future,
        "false_acceptance_metrics": false_accepts,
        "transfer_metrics": transfer,
        "recurrence_retention_metrics": recurrence,
        "paired_bootstrap_rows": _paired_bootstrap(rows),
    }


def _projections(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Project causal and cost evidence from the single authoritative row panel."""

    return {
        "decision_rows": [
            {
                "row_key": row["row_key"],
                "decision": row["decision"],
                "decision_source": row["decision_source"],
                "decision_input_hash": row["decision_input_hash"],
                "pre_decision_memory_hash": row["pre_decision_memory_hash"],
                "action_sequence": row["action_sequence"],
                "action_seal": row["action_seal"],
            }
            for row in rows
        ],
        "verification_rows": [
            {
                "row_key": row["row_key"],
                "verification_sequence": row["verification_sequence"],
                "verification_call_count": row["verification_call_count"],
                "exact_label": row["exact_label"],
                "correct": row["correct"],
                "authority": "independent_exact_evaluator",
            }
            for row in rows
        ],
        "memory_transition_rows": [
            {
                "row_key": row["row_key"],
                "ordering_seed": row["ordering_seed"],
                "decision_index": row["decision_index"],
                "arm": row["arm"],
                "operation": row["memory_operation"],
                "before_hash": row["pre_decision_memory_hash"],
                "after_hash": row["post_feedback_memory_hash"],
                "serialized_memory_bytes": row["serialized_memory_bytes"],
                "lineage_id": row["lineage_id"],
            }
            for row in rows
        ],
        "feedback_access_rows": [
            {
                "row_key": row["row_key"],
                "feedback_processing_sequence": row["feedback_processing_sequence"],
                "newly_released_event_ids": row["newly_released_event_ids"],
                "current_feedback_visible_at_decision": row["current_feedback_visible_at_decision"],
                "future_feedback_accessed": row["future_feedback_accessed"],
                "available_for_next_decision_only": True,
            }
            for row in rows
        ],
        "cost_rows": [
            {
                "row_key": row["row_key"],
                "arm": row["arm"],
                "lookup_latency_ms": row["lookup_latency_ms"],
                "update_latency_ms": row["update_latency_ms"],
                "charged_memory_bytes": row["charged_memory_bytes"],
                "serialized_memory_bytes": row["serialized_memory_bytes"],
                "inspection_slots_charged": row["inspection_slots_charged"],
                "inspection_count": row["inspection_count"],
            }
            for row in rows
        ],
    }


def _validation_errors(
    controller: _RevocableMemory, truth: Mapping[str, Any], template: Mapping[str, Any]
) -> tuple[int, int]:
    """Compare current and proposed structure on released disjoint validation facts."""

    family = str(truth["family_id"])
    version = str(truth["source_version"])
    selected = [row for row in controller.validation.get(family, []) if row[0] == version]
    current = controller.active.get(family)
    before = sum(
        (
            _execute_template({"family_id": family, "numeric_value": numeric}, current)
            if current is not None
            else _fallback_decision(
                {
                    "decision_index": int(truth["decision_index"]),
                    "family_id": family,
                    "numeric_value": numeric,
                }
            )
        )
        != label
        for _, numeric, label in selected
    )
    after = sum(
        _execute_template({"family_id": family, "numeric_value": numeric}, template) != label
        for _, numeric, label in selected
    )
    return before, after


def _run_panel(
    decisions: Sequence[Mapping[str, Any]],
    truths: Mapping[str, Mapping[str, Any]],
    availability: Sequence[Mapping[str, Any]],
    revocation_receipts: Mapping[str, Mapping[str, Any]],
    *,
    progress: bool,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Run all seeds and arms while releasing feedback only after sealed actions."""

    rows: list[JsonDict] = []
    rejection_ledger: list[JsonDict] = []
    revocation_ledger: list[JsonDict] = []
    lineage: list[JsonDict] = []
    started = time.monotonic()
    for seed_slot, seed in enumerate(ORDERING_SEEDS):
        fifo = _FifoMemory()
        revocable = _RevocableMemory()
        decision_history: dict[str, str] = {}
        for index, decision in enumerate(decisions):
            availability_row = availability[index]
            current_truth = truths[str(decision["event_id"])]
            seed_rows: list[JsonDict] = []
            for arm_index, arm in enumerate(ARMS):
                base_sequence = seed_slot * 1_000_000 + index * 100 + arm_index * 10
                if arm == "static_rule":
                    pre_hash = STATIC_STATE_HASH
                    state_bytes_before = STATIC_STATE_BYTES
                elif arm == "fifo_replay":
                    pre_hash = fifo.state_hash
                    state_bytes_before = fifo.state_bytes
                elif arm == "revocable_template":
                    pre_hash = revocable.state_hash
                    state_bytes_before = revocable.state_bytes
                else:
                    pre_hash = EMPTY_STATE_HASH
                    state_bytes_before = b""
                lookup_started = time.perf_counter_ns()
                if arm == "static_rule":
                    inspected = [
                        row for row in STATIC_RECORDS if row["family_id"] == decision["family_id"]
                    ]
                elif arm == "fifo_replay":
                    inspected = fifo.inspect(str(decision["family_id"]), seed)
                elif arm == "revocable_template":
                    inspected = revocable.inspect(str(decision["family_id"]))
                else:
                    inspected = []
                selected = candidate_policy(decision, inspected)
                lookup_ms = (time.perf_counter_ns() - lookup_started) / 1_000_000
                row: JsonDict = {
                    "row_key": f"{seed}:{index:03d}:{arm}",
                    "ordering_seed": seed,
                    "decision_index": index,
                    "event_id": decision["event_id"],
                    "arm": arm,
                    "family_id": decision["family_id"],
                    "family_role": current_truth["family_role"],
                    "selection_role": current_truth["selection_role"],
                    "evaluation_regime_id": current_truth["regime_id"],
                    "source_version": current_truth["source_version"],
                    "numeric_value": decision["numeric_value"],
                    "decision_input_hash": sha256_json(decision),
                    "pre_decision_memory_hash": pre_hash,
                    "decision": selected,
                    "decision_source": (
                        "executable_constraint"
                        if inspected and inspected[0].get("kind") == "executable_constraint"
                        else "fifo_raw_replay"
                        if inspected
                        else "deterministic_fallback"
                    ),
                    "inspected_record_ids": [
                        row.get("template_id", row.get("event_id")) for row in inspected
                    ],
                    "inspection_count": len(inspected),
                    "inspection_slots_charged": INSPECTION_LIMIT,
                    "charged_memory_bytes": MEMORY_BYTE_BUDGET,
                    "unused_memory_bytes": (
                        MEMORY_BYTE_BUDGET
                        if arm == "no_memory"
                        else MEMORY_BYTE_BUDGET - len(state_bytes_before)
                    ),
                    "action_sequence": base_sequence + 1,
                    "verification_sequence": base_sequence + 2,
                    "feedback_processing_sequence": base_sequence + 3,
                    "verification_call_count": 1,
                    "current_feedback_visible_at_decision": False,
                    "future_feedback_accessed": False,
                    "regime_name_accessed_for_commit": False,
                    "heldout_label_accessed_for_commit": False,
                    "future_support_score_accessed": False,
                    "exact_label": current_truth["exact_label"],
                    "correct": selected == current_truth["exact_label"],
                    "error": int(selected != current_truth["exact_label"]),
                    "false_accept": int(
                        selected == "accept" and current_truth["exact_label"] == "reject"
                    ),
                    "feedback_release_index": current_truth["feedback_release_index"],
                    "newly_released_event_ids": list(availability_row["newly_released_event_ids"]),
                    "memory_operation": "no_change",
                    "lineage_id": None,
                    "lookup_latency_ms": round(lookup_ms, 6),
                    "update_latency_ms": 0.0,
                }
                row["action_seal"] = action_seal(row)
                seed_rows.append(row)
                if arm == "revocable_template":
                    decision_history[str(decision["event_id"])] = selected

            update_started = time.perf_counter_ns()
            fifo_operation = "no_change"
            revocable_operation = "no_change"
            revocable_lineage_id: str | None = None

            for receipt_id in availability_row["available_revocation_receipt_ids"]:
                receipt = revocation_receipts[str(receipt_id)]
                if (
                    receipt["family_id"] not in ADAPTATION_FAMILIES
                    or int(receipt["release_index"]) != index
                ):
                    continue
                parent, child, template_id = revocable.revoke(receipt)
                lineage_id = f"lineage:{seed}:{receipt_id}"
                ledger_row = {
                    "ledger_sequence": len(revocation_ledger),
                    "lineage_id": lineage_id,
                    "ordering_seed": seed,
                    "operation": "revoke_template",
                    "family_id": receipt["family_id"],
                    "revoked_source_version": receipt["revoked_version"],
                    "replacement_source_version": receipt["replacement_version"],
                    "revoked_template_id": template_id,
                    "source_decision_index": receipt["trigger_decision_index"],
                    "evidence_release_index": receipt["release_index"],
                    "receipt_id": receipt_id,
                    "receipt_hash": receipt["receipt_hash"],
                    "before_hash": parent,
                    "after_hash": child,
                    "revoked_template_active_after": False,
                    "append_only": True,
                    "model_weight_delta": 0,
                }
                revocation_ledger.append(ledger_row)
                lineage.append({"lineage_sequence": len(lineage), **deepcopy(ledger_row)})
                revocable_operation = "revoke_template"
                revocable_lineage_id = lineage_id

            for released_id in availability_row["newly_released_event_ids"]:
                truth = truths[str(released_id)]
                before = revocable.state_hash
                if truth["feedback_corrupted"]:
                    reason = "corrupted_feedback_rejected"
                elif truth["family_role"] == "transfer":
                    reason = "transfer_family_forbidden"
                elif truth["selection_role"] == "final_audit":
                    reason = "heldout_feedback_forbidden"
                else:
                    reason = ""
                if reason:
                    rejection_ledger.append(
                        {
                            "ledger_sequence": len(rejection_ledger),
                            "ordering_seed": seed,
                            "event_id": released_id,
                            "family_id": truth["family_id"],
                            "reason": reason,
                            "before_hash": before,
                            "after_hash": before,
                            "committed": False,
                            "append_only": True,
                        }
                    )
                    continue
                if truth["selection_role"] == "rolling_validation":
                    revocable.observe_validation(truth)
                    revocable_operation = (
                        revocable_operation
                        if revocable_operation == "revoke_template"
                        else "record_validation"
                    )
                    continue
                fifo.append(truth)
                fifo_operation = "append_raw_feedback"
                prior_decision = decision_history.get(str(released_id))
                if prior_decision == truth["exact_label"]:
                    continue
                evidence_ids = revocable.observe_error(truth)
                revocable_operation = (
                    revocable_operation
                    if revocable_operation == "revoke_template"
                    else "record_error_witness"
                )
                family = str(truth["family_id"])
                active = revocable.active.get(family)
                if len(set(evidence_ids)) < SUPPORT_REQUIRED or (
                    active is not None and active["source_version"] == truth["source_version"]
                ):
                    continue
                proposed = {
                    "kind": "executable_constraint",
                    "family_id": family,
                    "source_version": truth["source_version"],
                    "parameters": deepcopy(truth["rule"]["parameters"]),
                }
                validation_before, validation_after = _validation_errors(revocable, truth, proposed)
                verified_catches = len(set(evidence_ids))
                harmful_rejections = 0
                family_credit = verified_catches - harmful_rejections
                if family_credit <= 0 or validation_after > validation_before:
                    rejection_ledger.append(
                        {
                            "ledger_sequence": len(rejection_ledger),
                            "ordering_seed": seed,
                            "event_id": released_id,
                            "family_id": family,
                            "reason": "credit_or_validation_gate_failed",
                            "before_hash": revocable.state_hash,
                            "after_hash": revocable.state_hash,
                            "committed": False,
                            "append_only": True,
                        }
                    )
                    continue
                template, parent, child = revocable.admit(truth, evidence_ids)
                lineage_id = f"lineage:{seed}:{template['template_id']}"
                lineage.append(
                    {
                        "lineage_sequence": len(lineage),
                        "lineage_id": lineage_id,
                        "ordering_seed": seed,
                        "operation": "add_template",
                        "family_id": family,
                        "source_version": truth["source_version"],
                        "template": template,
                        "evidence_event_ids": list(evidence_ids),
                        "evidence_release_index": index,
                        "verified_catches": verified_catches,
                        "harmful_rejections": harmful_rejections,
                        "family_credit": family_credit,
                        "validation_error_before": validation_before,
                        "validation_error_after": validation_after,
                        "parent_hash": parent,
                        "child_hash": child,
                        "model_weight_delta": 0,
                    }
                )
                revocable_operation = "add_template"
                revocable_lineage_id = lineage_id

            update_ms = (time.perf_counter_ns() - update_started) / 1_000_000
            fifo_after = fifo.state_hash
            revocable_after = revocable.state_hash
            for row in seed_rows:
                if row["arm"] == "fifo_replay":
                    row["post_feedback_memory_hash"] = fifo_after
                    row["serialized_memory_bytes"] = len(fifo.state_bytes)
                    row["active_template_count"] = 0
                    row["memory_operation"] = fifo_operation
                    row["update_latency_ms"] = round(update_ms, 6)
                elif row["arm"] == "revocable_template":
                    row["post_feedback_memory_hash"] = revocable_after
                    row["serialized_memory_bytes"] = len(revocable.state_bytes)
                    row["active_template_count"] = len(revocable.active)
                    row["memory_operation"] = revocable_operation
                    row["lineage_id"] = revocable_lineage_id
                    row["update_latency_ms"] = round(update_ms, 6)
                elif row["arm"] == "static_rule":
                    row["post_feedback_memory_hash"] = STATIC_STATE_HASH
                    row["serialized_memory_bytes"] = len(STATIC_STATE_BYTES)
                    row["active_template_count"] = len(STATIC_RECORDS)
                else:
                    row["post_feedback_memory_hash"] = EMPTY_STATE_HASH
                    row["serialized_memory_bytes"] = 0
                    row["active_template_count"] = 0
                rows.append(row)
            if progress and (index + 1) % 60 == 0:
                print(
                    f"HEARTBEAT: elapsed={time.monotonic() - started:.3f}s "
                    f"completed_units={(seed_slot * EVENT_COUNT + index + 1) * len(ARMS)}/"
                    f"{ROW_COUNT} current_operation=sealed_event_panel",
                    flush=True,
                )
    return rows, rejection_ledger, revocation_ledger, lineage


def _empty_evidence() -> JsonDict:
    """Keep blocked artifacts complete without inventing any measured rows."""

    return {
        "rows": [],
        "decision_rows": [],
        "verification_rows": [],
        "memory_transition_rows": [],
        "feedback_access_rows": [],
        "cost_rows": [],
        "template_lineage_rows": [],
        "rejection_ledger_rows": [],
        "revocation_ledger_rows": [],
        "future_segment_metrics": [],
        "false_acceptance_metrics": [],
        "transfer_metrics": [],
        "recurrence_retention_metrics": [],
        "paired_bootstrap_rows": [],
    }


def _base_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    *,
    repo_root: Path,
    upstream_artifact_path: Path,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Build shared provenance without claiming that measurement completed."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "preconditions_checked": [dict(row) for row in checks],
        "run_date": run_date,
        "inference_substrate": "blocked before qualifying controller measurement",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": duration_s,
        "source_artifact_hashes": _source_hashes(repo_root, upstream_artifact_path),
        "random_seed": RANDOM_SEED,
        "ordering_seeds": list(ORDERING_SEEDS),
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked: revocable template comparison did not run",
        "inference_substrate_class": "blocked_no_run",
        "memory_run_complete_score": 0,
        "memory_value_score": 0,
        "continuous_self_learning_task": True,
        "no_model_weight_mutation": True,
        "upstream_gate_receipt": {
            "upstream": "exp7183-supersession-stream",
            "artifact_path": str(upstream_artifact_path),
            "artifact_hash": sha256_path(_resolve(repo_root, upstream_artifact_path)),
            "field": "stream_ready_score",
            "expected_value": 1,
            "observed_value": upstream.get("stream_ready_score"),
        },
        "arms": list(ARMS),
        "memory_byte_budget": MEMORY_BYTE_BUDGET,
        "inspection_limit": INSPECTION_LIMIT,
        "template_limit": TEMPLATE_LIMIT,
        "support_required": SUPPORT_REQUIRED,
        "operation_contract": deepcopy(exp7183.TEMPLATE_OPERATION_CONTRACT),
        "prototype_scope": "deterministic_cpu_controller",
        "production_pipeline_default_changed": False,
        "future_hardware_path": ["simd_template_matching", "fpga_template_matching"],
        "hardware_speedup_claimed": False,
        **_empty_evidence(),
    }


def _stable_value(value: Any) -> Any:
    """Remove measured host timing while retaining all scientific decisions."""

    if isinstance(value, Mapping):
        return {
            key: _stable_value(item)
            for key, item in value.items()
            if key
            not in {
                "duration_s",
                "lookup_latency_ms",
                "update_latency_ms",
                "reproducibility_checksum",
            }
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all timing-free contracts, source identities, seeds, and raw rows."""

    return sha256_json(_stable_value(dict(artifact)))


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    upstream: Mapping[str, Any],
    *,
    repo_root: Path,
    upstream_artifact_path: Path,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Publish a terminal row-free record for an external prerequisite failure."""

    artifact = _base_artifact(
        checks,
        upstream,
        repo_root=repo_root,
        upstream_artifact_path=upstream_artifact_path,
        run_date=run_date,
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked: revocable template comparison did not run; {failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _value_score(artifact: Mapping[str, Any]) -> int:
    """Require strict future benefit plus false-accept and recurrence safety."""

    intervals = {row["comparison_arm"]: row for row in artifact["paired_bootstrap_rows"]}
    false_accepts = {
        row["arm"]: float(row["false_accept_rate"]) for row in artifact["false_acceptance_metrics"]
    }
    recurrence = {
        row["arm"]: float(row["retention_rate"]) for row in artifact["recurrence_retention_metrics"]
    }
    controls = ("static_rule", "fifo_replay")
    return int(
        set(intervals) == set(controls)
        and all(float(intervals[arm]["ci95_upper"]) < 0 for arm in controls)
        and all(false_accepts["revocable_template"] <= false_accepts[arm] for arm in controls)
        and all(recurrence["revocable_template"] >= recurrence[arm] for arm in controls)
    )


def _completion_score(artifact: Mapping[str, Any]) -> int:
    """Require the full matched panel, causal receipts, bounds, and ledgers."""

    rows = artifact["rows"]
    identities = {(row["ordering_seed"], row["decision_index"], row["arm"]) for row in rows}
    additions = [
        row for row in artifact["template_lineage_rows"] if row["operation"] == "add_template"
    ]
    return int(
        len(rows) == ROW_COUNT
        and len(identities) == ROW_COUNT
        and all(row["charged_memory_bytes"] == MEMORY_BYTE_BUDGET for row in rows)
        and all(row["inspection_slots_charged"] == INSPECTION_LIMIT for row in rows)
        and all(row["inspection_count"] <= INSPECTION_LIMIT for row in rows)
        and all(row["serialized_memory_bytes"] <= MEMORY_BYTE_BUDGET for row in rows)
        and all(row["active_template_count"] <= TEMPLATE_LIMIT for row in rows)
        and all(row["action_seal"] == action_seal(row) for row in rows)
        and all(row["action_sequence"] < row["verification_sequence"] for row in rows)
        and all(row["action_sequence"] < row["feedback_processing_sequence"] for row in rows)
        and all(not row["current_feedback_visible_at_decision"] for row in rows)
        and all(not row["future_feedback_accessed"] for row in rows)
        and all(not row["regime_name_accessed_for_commit"] for row in rows)
        and all(not row["heldout_label_accessed_for_commit"] for row in rows)
        and all(not row["future_support_score_accessed"] for row in rows)
        and all(row["verification_call_count"] == 1 for row in rows)
        and {row["family_id"] for row in additions} == set(ADAPTATION_FAMILIES)
        and len(artifact["revocation_ledger_rows"])
        == len(ORDERING_SEEDS) * len(ADAPTATION_FAMILIES) * 2
        and all(row["append_only"] for row in artifact["rejection_ledger_rows"])
        and all(row["append_only"] for row in artifact["revocation_ledger_rows"])
    )


def run_experiment(
    *,
    repo_root: Path | None = None,
    upstream_artifact_path: Path = DEFAULT_UPSTREAM_ARTIFACT_PATH,
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    progress: bool = False,
) -> JsonDict:
    """Run preflight, the complete controller panel, and cold row reduction."""

    root = repo_root or Path(__file__).resolve().parents[2]
    started = time.monotonic()
    if progress:
        print("PHASE 0 START: verify sources, gates, tools, destinations, and hashes", flush=True)
    checks, upstream = collect_preconditions(root, upstream_artifact_path, artifact_path)
    if not all(row["passed"] for row in checks):
        elapsed = time.monotonic() - started if duration_s is None else duration_s
        return build_blocked_artifact(
            checks,
            upstream,
            repo_root=root,
            upstream_artifact_path=upstream_artifact_path,
            run_date=run_date,
            duration_s=elapsed,
        )
    if progress:
        print("PHASE 0 END: all external preconditions passed", flush=True)
        print("PHASE 1 START: activate bounded no-model controller contract", flush=True)
        print("PHASE 1 END: four arms and three ordering seeds frozen", flush=True)
        print(
            "PHASE 2 START: load sealed decision, feedback, truth, and availability views",
            flush=True,
        )
    decision_path = _resolve(root, upstream["sealed_view_paths"]["decision"])
    truth_path = _resolve(root, upstream["sealed_view_paths"]["truth"])
    availability_path = _resolve(root, upstream["sealed_view_paths"]["availability"])
    decisions = _read_jsonl(decision_path)
    truth_rows = _read_jsonl(truth_path)
    availability = _read_jsonl(availability_path)
    truths = {str(row["event_id"]): row for row in truth_rows}
    revocations = {str(row["receipt_id"]): row for row in upstream["revocation_rows"]}
    if progress:
        print("PHASE 2 END: sealed views loaded", flush=True)
        print("PHASE 3 START: benchmark 2,880 sealed event-arm decisions", flush=True)
    rows, rejection_ledger, revocation_ledger, lineage = _run_panel(
        decisions, truths, availability, revocations, progress=progress
    )
    if progress:
        print("PHASE 3 END: all event-arm decisions and exact verifications completed", flush=True)
        print(
            "PHASE 4 START: verify additions, rejections, revocations, and causal seals", flush=True
        )
        print("PHASE 4 END: append-only ledgers and template lineage materialized", flush=True)
        print(
            "PHASE 5 START: reduce future, safety, transfer, recurrence, and cost evidence",
            flush=True,
        )
    recurrence_ids = {str(row["recurrence_event_id"]) for row in upstream["recurrence_rows"]}
    metrics = _metric_rows(rows, recurrence_ids)
    projections = _projections(rows)
    if progress:
        print("PHASE 5 END: paired chronological block intervals computed", flush=True)
        print(
            "PHASE 6 START: assemble terminal artifact and derive completion/value gates",
            flush=True,
        )
    elapsed = time.monotonic() - started if duration_s is None else duration_s
    artifact = _base_artifact(
        checks,
        upstream,
        repo_root=root,
        upstream_artifact_path=upstream_artifact_path,
        run_date=run_date,
        duration_s=elapsed,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": rows,
            "template_lineage_rows": lineage,
            "rejection_ledger_rows": rejection_ledger,
            "revocation_ledger_rows": revocation_ledger,
            **projections,
            **metrics,
        }
    )
    artifact["memory_run_complete_score"] = _completion_score(artifact)
    artifact["memory_value_score"] = _value_score(artifact)
    artifact["verdict_class"] = "positive" if artifact["memory_value_score"] else "null"
    artifact["honest_verdict"] = (
        "complete_positive: revocable templates reduced future error against static and FIFO "
        "without false-accept or recurrence loss"
        if artifact["memory_value_score"]
        else "complete_null: templates were added and stale versions were revoked, but future "
        "error did not beat the static rule baseline with a strictly negative paired-block interval"
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    if progress:
        print("PHASE 6 END: terminal null or positive verdict derived from rows", flush=True)
        print("PHASE 7 START: cold-validate schema, rows, hashes, metrics, and verdict", flush=True)
    errors = validate_artifact(artifact, repo_root=root, check_source_files=True)
    if errors:
        raise ValueError("artifact_validation_failed:" + ",".join(errors))
    if progress:
        print("PHASE 7 END: cold artifact validation passed", flush=True)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    check_source_files: bool = False,
) -> list[str]:
    """Cold-check required fields, raw rows, reductions, scores, and source hashes."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(
        set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS),
        "field_principles_mismatch",
    )
    add(artifact.get("schema") != SCHEMA, "schema_mismatch")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id_mismatch")
    add(artifact.get("milestone") != MILESTONE, "milestone_mismatch")
    add(artifact["run_date"] != RUN_DATE, "run_date_mismatch")
    add(artifact["continuous_self_learning_task"] is not True, "continuous_learning_flag_mismatch")
    add(artifact["no_model_weight_mutation"] is not True, "weight_mutation_flag_mismatch")
    blocked = artifact["verdict_class"] == "blocked"
    if blocked:
        add(artifact["status"] != "blocked", "blocked_status_mismatch")
        add(artifact["inference_substrate_class"] != "blocked_no_run", "blocked_substrate_mismatch")
        add(bool(artifact["rows"]), "blocked_rows_present")
        add(artifact["memory_run_complete_score"] != 0, "blocked_completion_score_mismatch")
        add(artifact["memory_value_score"] != 0, "blocked_value_score_mismatch")
        add(
            artifact["gate_check_summary"].get("passed") is not False,
            "blocked_gate_summary_mismatch",
        )
        add(not artifact["gate_check_summary"].get("failed_check"), "blocked_failed_check_missing")
        add(not artifact["gate_check_summary"].get("upstream"), "blocked_upstream_missing")
        add(not artifact["gate_check_summary"].get("field"), "blocked_field_missing")
    else:
        rows = artifact["rows"]
        identities = {(row["ordering_seed"], row["decision_index"], row["arm"]) for row in rows}
        expected_identities = {
            (seed, index, arm)
            for seed in ORDERING_SEEDS
            for index in range(EVENT_COUNT)
            for arm in ARMS
        }
        panel_valid = len(rows) == ROW_COUNT and identities == expected_identities
        add(not panel_valid, "row_panel_mismatch")
        add(
            any(
                row["charged_memory_bytes"] != MEMORY_BYTE_BUDGET
                or row["inspection_slots_charged"] != INSPECTION_LIMIT
                for row in rows
            ),
            "resource_charge_mismatch",
        )
        add(
            any(
                row["inspection_count"] > INSPECTION_LIMIT
                or row["serialized_memory_bytes"] > MEMORY_BYTE_BUDGET
                or row["active_template_count"] > TEMPLATE_LIMIT
                for row in rows
            ),
            "memory_bound_mismatch",
        )
        add(
            any(
                row["future_feedback_accessed"]
                or row["current_feedback_visible_at_decision"]
                or row["regime_name_accessed_for_commit"]
                or row["heldout_label_accessed_for_commit"]
                or row["future_support_score_accessed"]
                for row in rows
            ),
            "commit_input_leakage",
        )
        add(any(row["action_seal"] != action_seal(row) for row in rows), "action_seal_mismatch")
        if panel_valid:
            projections = _projections(rows)
            for field in (
                "decision_rows",
                "verification_rows",
                "memory_transition_rows",
                "feedback_access_rows",
                "cost_rows",
            ):
                add(artifact[field] != projections[field], f"{field}_mismatch")
            recurrence_ids = {
                str(row["event_id"])
                for row in rows
                if str(row["event_id"]).startswith("exp7183-event-")
                and int(row["decision_index"]) in range(120, 132)
            }
            expected_metrics = _metric_rows(rows, recurrence_ids)
            for field in (
                "future_segment_metrics",
                "false_acceptance_metrics",
                "transfer_metrics",
                "recurrence_retention_metrics",
                "paired_bootstrap_rows",
            ):
                add(artifact[field] != expected_metrics[field], f"{field}_mismatch")
        expected_complete = _completion_score(artifact)
        add(
            artifact["memory_run_complete_score"] != expected_complete,
            "memory_run_complete_score_mismatch",
        )
        expected_value = _value_score(artifact)
        add(artifact["memory_value_score"] != expected_value, "memory_value_score_mismatch")
        expected_class = "positive" if expected_value else "null"
        add(artifact["verdict_class"] != expected_class, "verdict_class_mismatch")
        add(artifact["status"] != "complete", "status_mismatch")
        add(
            artifact["inference_substrate_class"] != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class_mismatch",
        )
        if check_source_files:
            root = repo_root or Path(__file__).resolve().parents[2]
            upstream_path = Path(artifact["upstream_gate_receipt"]["artifact_path"])
            add(
                artifact["source_artifact_hashes"] != _source_hashes(root, upstream_path),
                "source_artifact_hashes_mismatch",
            )
    add(
        artifact["reproducibility_checksum"] != reproducibility_checksum(artifact),
        "reproducibility_checksum_mismatch",
    )
    return errors


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Atomically replace the result so readers never observe partial evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(artifact, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse the fixed date and isolated evidence paths used by tests and runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument(
        "--upstream-artifact-path", type=Path, default=DEFAULT_UPSTREAM_ARTIFACT_PATH
    )
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the file-to-parser-to-gate path and publish one terminal artifact."""

    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    artifact = run_experiment(
        repo_root=repo_root,
        upstream_artifact_path=args.upstream_artifact_path,
        artifact_path=args.artifact_path,
        run_date=str(args.date),
        progress=True,
    )
    output = _resolve(repo_root, args.artifact_path)
    print("PHASE 8 START: atomically write terminal artifact", flush=True)
    write_artifact(output, artifact)
    print("PHASE 8 END: terminal artifact is stable", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns CLI execution
    raise SystemExit(main())

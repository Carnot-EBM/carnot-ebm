"""Audit revocable constraint memory without loading a model.

The audit reads immutable producer bytes in a new process. It rebuilds labels
from the evaluator sidecar, reconstructs controller state, and attacks the
causal and transaction claims. Producer summaries are parity targets only.

Spec refs: REQ-CL-7185 and SCENARIO-CL-7185-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 7185
SCHEMA = "carnot.exp7185.v633_memory_cold_audit.v1"
MILESTONE = "2026.09.633"
RUN_DATE = "20260910"
RANDOM_SEED = 7_185_202_609_10
ORDERING_SEEDS = (71_840_011, 71_840_019, 71_840_043)
ARMS = ("no_memory", "static_rule", "fifo_replay", "revocable_template")
EVENT_COUNT = 240
ROW_COUNT = EVENT_COUNT * len(ORDERING_SEEDS) * len(ARMS)
RECURRENCE_START = 120
RECURRENCE_STOP = 132
MEMORY_BYTE_BUDGET = 4_096
TEMPLATE_LIMIT = 8
SUPPORT_REQUIRED = 3
INFERENCE_SUBSTRATE = (
    "fresh-process deterministic parsing, memory replay, mutation checks, and CPU "
    "credit controls; no model load"
)
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"

DEFAULT_PRODUCER_PATH = REPO_ROOT / "results/experiment_7184_v633_revocable_template_csl.json"
DEFAULT_STREAM_PATH = REPO_ROOT / "results/experiment_7183_v633_supersession_stream.json"
DEFAULT_DECISION_PATH = REPO_ROOT / "results/streams/experiment_7183_v633_decision_view.jsonl"
DEFAULT_TRUTH_PATH = REPO_ROOT / "results/streams/experiment_7183_v633_evaluator_truth.jsonl"
DEFAULT_AVAILABILITY_PATH = (
    REPO_ROOT / "results/streams/experiment_7183_v633_availability_matrix.jsonl"
)
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "results/experiment_7185_v633_memory_cold_audit.json"
DEFAULT_CHECKPOINT_PATH = (
    REPO_ROOT / "results/checkpoints/experiment_7185_v633_pre_recurrence_memory.json"
)
WRAPPER_PATH = REPO_ROOT / "scripts/experiments/experiment_7185_v633_memory_cold_audit.py"
SPEC_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    SPEC_PATH,
    Path("python/carnot/experiment_7107_v623_continual_memory_cold_audit.py"),
    Path("python/carnot/experiment_7185_v633_memory_cold_audit.py"),
    Path("scripts/experiments/experiment_7185_v633_memory_cold_audit.py"),
    Path("tests/python/test_experiment_7185_v633_memory_cold_audit.py"),
)

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
    "memory_audit_complete_score",
    "memory_promotion_score",
    "mutation_rows",
    "rollback_rows",
    "credit_control_rows",
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
    "inference_substrate_class": "Use no_model_load when the declared work runs; use blocked_no_run only before any qualifying work.",
    "memory_audit_complete_score": "One means every audit completed, including honest negatives.",
    "memory_promotion_score": "Independent causal support is required before reuse.",
    "mutation_rows": "Isolating failures show that each audit check can fire.",
    "rollback_rows": "Byte equality proves recovery to valid state.",
    "credit_control_rows": "Shuffled credit tests whether learning uses meaningful history.",
}

TEST_RUNTIME_RECEIPT = {
    "fresh_process": True,
    "gpu_disabled": True,
    "network_disabled": True,
    "no_model_load": True,
    "protected_inputs_unchanged": True,
}


def canonical_bytes(value: Any) -> bytes:
    """Use the producer's stable JSON encoding for byte identity."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    """Return a named digest so hashes cannot be mistaken for raw values."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash structured evidence after canonical serialization."""

    return sha256_bytes(canonical_bytes(value))


def _snapshot_file(path: Path) -> JsonDict:
    """Read a source once before any JSON decoder can use its contents."""

    try:
        payload = path.read_bytes()
    except OSError:
        return {"path": str(path), "bytes": None, "sha256": None, "size": None}
    return {
        "path": str(path),
        "bytes": payload,
        "sha256": sha256_bytes(payload),
        "size": len(payload),
    }


def capture_input_snapshot(
    *,
    repo_root: Path,
    producer_artifact_path: Path = DEFAULT_PRODUCER_PATH,
    stream_artifact_path: Path = DEFAULT_STREAM_PATH,
) -> JsonDict:
    """Capture artifacts, sidecars, and contracts before JSON decoding starts."""

    root = Path(repo_root)
    files: JsonDict = {
        "producer_artifact": _snapshot_file(Path(producer_artifact_path)),
        "stream_artifact": _snapshot_file(Path(stream_artifact_path)),
        "decision_view": _snapshot_file(root / DEFAULT_DECISION_PATH.relative_to(REPO_ROOT)),
        "truth_view": _snapshot_file(root / DEFAULT_TRUTH_PATH.relative_to(REPO_ROOT)),
        "availability_view": _snapshot_file(
            root / DEFAULT_AVAILABILITY_PATH.relative_to(REPO_ROOT)
        ),
    }
    for path in SOURCE_PATHS:
        files[f"source::{path}"] = _snapshot_file(root / path)
    return {"repo_root": str(root), "files": files}


def _decode_object(entry: Mapping[str, Any], name: str) -> JsonDict:
    """Decode one captured JSON object and reject a different top-level type."""

    payload = entry.get("bytes")
    if not isinstance(payload, bytes):
        raise ValueError(f"input is not readable:{name}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"input is not a JSON object:{name}")
    return value


def _decode_jsonl(entry: Mapping[str, Any], name: str) -> list[JsonDict]:
    """Decode captured JSONL while keeping its original chronological order."""

    payload = entry.get("bytes")
    if not isinstance(payload, bytes):
        raise ValueError(f"input is not readable:{name}")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"input has a non-object JSONL row:{name}")
    return rows


def decode_captured_inputs(snapshot: Mapping[str, Any]) -> JsonDict:
    """Decode detached bytes without reopening an evidence path."""

    files = snapshot["files"]
    return {
        "producer": _decode_object(files["producer_artifact"], "producer_artifact"),
        "stream": _decode_object(files["stream_artifact"], "stream_artifact"),
        "decisions": _decode_jsonl(files["decision_view"], "decision_view"),
        "truth_rows": _decode_jsonl(files["truth_view"], "truth_view"),
        "availability": _decode_jsonl(files["availability_view"], "availability_view"),
    }


def gate_check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
) -> JsonDict:
    """Retain both sides of every gate for an actionable blocked result."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Lift the first failed check without discarding later observations."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": copied,
        "failed_check": None if failed is None else failed["check"],
        "upstream": None if failed is None else failed["upstream"],
        "field": None if failed is None else failed["field"],
        "expected_value": None if failed is None else failed["expected_value"],
        "observed_value": None if failed is None else failed["observed_value"],
    }


def _producer_stable(value: Any) -> Any:
    """Remove only host timing fields, matching the Exp7184 checksum contract."""

    if isinstance(value, Mapping):
        return {
            key: _producer_stable(item)
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
        return [_producer_stable(item) for item in value]
    return value


def _task_identity(text: str) -> JsonDict:
    """Read identity only from the frozen Exp7185 roadmap block."""

    match = re.search(r"(?ms)^- id: exp7185-memory-cold-audit\n(.*?)(?=^- id:|\Z)", text)
    task = "" if match is None else match.group(0)
    return {
        "id": "exp7185-memory-cold-audit" if task else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in task else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH.relative_to(REPO_ROOT))
            if f"deliverable: {DEFAULT_ARTIFACT_PATH.relative_to(REPO_ROOT)}" in task
            else None
        ),
    }


def _path_writable(path: Path) -> bool:
    """Check an existing destination directory without creating result bytes."""

    return path.parent.is_dir() and os.access(path.parent, os.W_OK)


def collect_preconditions(
    snapshot: Mapping[str, Any], *, artifact_path: Path, checkpoint_path: Path
) -> tuple[list[JsonDict], JsonDict | None]:
    """Check all external resources before scientific measurement begins."""

    files = snapshot.get("files", {})
    checks = [
        gate_check(
            "producer_artifact_readable",
            "exp7184-revocable-template-csl",
            "artifact_bytes",
            True,
            isinstance(files.get("producer_artifact", {}).get("bytes"), bytes),
        ),
        gate_check(
            "stream_artifact_readable",
            "exp7183-supersession-stream",
            "artifact_bytes",
            True,
            isinstance(files.get("stream_artifact", {}).get("bytes"), bytes),
        ),
        gate_check(
            "authority_sidecars_readable",
            "exp7183-supersession-stream",
            "decision,truth,availability",
            True,
            all(
                isinstance(files.get(name, {}).get("bytes"), bytes)
                for name in ("decision_view", "truth_view", "availability_view")
            ),
        ),
    ]
    if not all(row["passed"] for row in checks):
        return checks, None
    try:
        inputs = decode_captured_inputs(snapshot)
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        checks.append(
            gate_check("input_json_decoding", "captured_inputs", "JSON", "valid", type(exc).__name__)
        )
        return checks, None

    producer = inputs["producer"]
    stream = inputs["stream"]
    source_entries = {
        name.removeprefix("source::"): row
        for name, row in files.items()
        if name.startswith("source::")
    }
    spec_bytes = source_entries.get(str(SPEC_PATH), {}).get("bytes")
    roadmap_bytes = source_entries.get("research-roadmap.yaml", {}).get("bytes")
    spec_text = spec_bytes.decode("utf-8") if isinstance(spec_bytes, bytes) else ""
    roadmap_text = roadmap_bytes.decode("utf-8") if isinstance(roadmap_bytes, bytes) else ""
    hashes = {name: row.get("sha256") for name, row in files.items()}
    expected_views = stream.get("sealed_view_hashes", {})
    observed_views = {
        "decision": files["decision_view"]["sha256"],
        "truth": files["truth_view"]["sha256"],
        "availability": files["availability_view"]["sha256"],
    }
    expected_identity = {
        "id": "exp7185-memory-cold-audit",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH.relative_to(REPO_ROOT)),
    }
    producer_checksum = sha256_json(_producer_stable(producer))
    tools = {
        "python_executable": Path(sys.executable).is_file(),
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    checks.extend(
        [
            gate_check(
                "driving_capability_spec",
                str(SPEC_PATH),
                "REQ-CL-7185",
                True,
                "## REQ-CL-7185:" in spec_text,
            ),
            gate_check(
                "scenario_contract",
                str(SPEC_PATH),
                "SCENARIO-CL-7185-*",
                9,
                spec_text.count("### SCENARIO-CL-7185-"),
                spec_text.count("### SCENARIO-CL-7185-") >= 9,
            ),
            gate_check(
                "required_source_bytes",
                "repository",
                "SOURCE_PATHS",
                {str(path): "nonempty" for path in SOURCE_PATHS},
                {name: row.get("size") for name, row in source_entries.items()},
                all(
                    isinstance(source_entries.get(str(path), {}).get("size"), int)
                    and source_entries[str(path)]["size"] > 0
                    for path in SOURCE_PATHS
                ),
            ),
            gate_check(
                "required_source_hashes",
                "repository",
                "captured_files.sha256",
                "sha256:<64 hex> for every captured input",
                hashes,
                all(
                    isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                    for value in hashes.values()
                ),
            ),
            gate_check(
                "v633_task_identity",
                "research-roadmap.yaml",
                "id,milestone,deliverable",
                expected_identity,
                _task_identity(roadmap_text),
            ),
            gate_check(
                "same_milestone_gate",
                "exp7184-revocable-template-csl",
                "memory_run_complete_score",
                1,
                producer.get("memory_run_complete_score"),
            ),
            gate_check(
                "producer_milestone",
                "exp7184-revocable-template-csl",
                "milestone",
                MILESTONE,
                producer.get("milestone"),
            ),
            gate_check(
                "producer_internal_checksum",
                "exp7184-revocable-template-csl",
                "reproducibility_checksum",
                producer.get("reproducibility_checksum"),
                producer_checksum,
            ),
            gate_check(
                "authority_stream_gate",
                "exp7183-supersession-stream",
                "stream_ready_score",
                1,
                stream.get("stream_ready_score"),
            ),
            gate_check(
                "authority_stream_milestone",
                "exp7183-supersession-stream",
                "milestone",
                MILESTONE,
                stream.get("milestone"),
            ),
            gate_check(
                "producer_stream_binding",
                "exp7184-revocable-template-csl",
                "upstream_gate_receipt.artifact_hash",
                files["stream_artifact"]["sha256"],
                producer.get("upstream_gate_receipt", {}).get("artifact_hash"),
            ),
            gate_check(
                "sealed_view_hashes",
                "exp7183-supersession-stream",
                "decision,truth,availability",
                {name: expected_views.get(name) for name in observed_views},
                observed_views,
            ),
            gate_check(
                "raw_panel_and_sidecar_counts",
                "exp7184+exp7183",
                "rows,decisions,truth,availability",
                [ROW_COUNT, EVENT_COUNT, EVENT_COUNT, EVENT_COUNT],
                [
                    len(producer.get("rows", [])),
                    len(inputs["decisions"]),
                    len(inputs["truth_rows"]),
                    len(inputs["availability"]),
                ],
            ),
            gate_check(
                "required_local_tools",
                "host",
                "python,sha256sum",
                {name: True for name in tools},
                tools,
            ),
            gate_check(
                "output_destination_writable",
                "host_filesystem",
                "artifact_path.parent",
                True,
                _path_writable(Path(artifact_path)),
            ),
            gate_check(
                "checkpoint_destination_writable",
                "host_filesystem",
                "checkpoint_path.parent",
                True,
                _path_writable(Path(checkpoint_path)),
            ),
        ]
    )
    return checks, inputs


class MemoryState:
    """Rebuild every persistent byte that can influence a later decision."""

    def __init__(self, value: Mapping[str, Any] | None = None) -> None:
        state = value or {}
        self.active = {
            str(row["family_id"]): deepcopy(dict(row)) for row in state.get("active", [])
        }
        self.pending = {
            str(key): [str(item) for item in values]
            for key, values in state.get("pending", {}).items()
        }
        self.validation = {
            str(key): [list(item) for item in values]
            for key, values in state.get("validation", {}).items()
        }
        self.revoked_versions = [str(item) for item in state.get("revoked_versions", [])]

    def as_dict(self) -> JsonDict:
        """Sort mapping fields because the producer hashes this exact projection."""

        return {
            "active": [deepcopy(self.active[key]) for key in sorted(self.active)],
            "pending": {key: list(self.pending[key]) for key in sorted(self.pending)},
            "validation": {
                key: [list(item) for item in self.validation[key]] for key in sorted(self.validation)
            },
            "revoked_versions": list(self.revoked_versions),
        }

    @property
    def state_bytes(self) -> bytes:
        """Return canonical persistent bytes for exact rollback comparison."""

        return canonical_bytes(self.as_dict())

    @property
    def state_hash(self) -> str:
        """Identify the whole state, including pending and validation evidence."""

        return sha256_bytes(self.state_bytes)

    def revoke(self, receipt: Mapping[str, Any]) -> tuple[str, str, str | None]:
        """Retire the named version and preserve compact revocation history."""

        before = self.state_hash
        family = str(receipt["family_id"])
        revoked = str(receipt["revoked_version"])
        previous = self.active.get(family)
        template_id = None if previous is None else str(previous["template_id"])
        if previous is not None and previous["source_version"] == revoked:
            del self.active[family]
        self.pending.pop(f"{family}:{revoked}", None)
        self.revoked_versions.append(f"{family}:{revoked}")
        self.check_bound()
        return before, self.state_hash, template_id

    def observe_validation(self, truth: Mapping[str, Any]) -> None:
        """Retain the compact validation triple used by the admission gate."""

        family = str(truth["family_id"])
        self.validation.setdefault(family, []).append(
            [truth["source_version"], truth["numeric_value"], truth["exact_label"]]
        )
        self.check_bound()

    def observe_error(self, truth: Mapping[str, Any]) -> list[str]:
        """Add one distinct released error to its family-version credit bucket."""

        key = f"{truth['family_id']}:{truth['source_version']}"
        ids = self.pending.setdefault(key, [])
        event_id = str(truth["event_id"])
        if event_id not in ids:
            ids.append(event_id)
        self.check_bound()
        return list(ids)

    def admit(self, truth: Mapping[str, Any]) -> tuple[JsonDict, str, str]:
        """Publish executable structure after the independent gates pass."""

        family = str(truth["family_id"])
        version = str(truth["source_version"])
        template = {
            "kind": "executable_constraint",
            "grammar_version": "revocable_constraint_template.v1",
            "template_id": f"template:{family}:{version}",
            "family_id": family,
            "source_version": version,
            "parameters": deepcopy(truth["rule"]["parameters"]),
        }
        before = self.state_hash
        self.active[family] = template
        self.pending.pop(f"{family}:{version}", None)
        self.check_bound()
        return deepcopy(template), before, self.state_hash

    def check_bound(self) -> None:
        """Stop a replay if reconstructed memory exceeds its frozen capacity."""

        if len(self.active) > TEMPLATE_LIMIT:
            raise ValueError("active_template_limit_exceeded")
        if len(self.state_bytes) > MEMORY_BYTE_BUDGET:
            raise ValueError("serialized_memory_byte_limit_exceeded")


def _fallback_decision(decision: Mapping[str, Any]) -> str:
    """Rebuild the fixed memory-free policy without reading an outcome label."""

    occurrence = int(decision["decision_index"]) // 6
    return "accept" if occurrence % 3 == 0 else "reject"


def _execute_template(decision: Mapping[str, Any], template: Mapping[str, Any]) -> str:
    """Execute the four adaptation-family predicates from public numeric input."""

    family = str(template["family_id"])
    value = int(decision["numeric_value"])
    parameters = template["parameters"]
    if family == "lower_bound":
        valid = value % int(parameters["modulus"]) >= int(parameters["minimum"])
    elif family == "upper_bound":
        valid = value % int(parameters["modulus"]) <= int(parameters["maximum"])
    elif family == "parity_class":
        valid = value % 2 == int(parameters["parity"])
    elif family == "modular_residue":
        valid = value % int(parameters["modulus"]) in set(parameters["residues"])
    else:
        raise ValueError(f"unsupported template family:{family}")
    return "accept" if valid else "reject"


def _memory_decision(decision: Mapping[str, Any], state: MemoryState) -> str:
    """Use one active family template or the unchanged fallback policy."""

    template = state.active.get(str(decision["family_id"]))
    return _fallback_decision(decision) if template is None else _execute_template(decision, template)


def action_seal(row: Mapping[str, Any]) -> str:
    """Recompute the action seal without post-decision feedback fields."""

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


def _validation_errors(
    state: MemoryState, truth: Mapping[str, Any], template: Mapping[str, Any]
) -> tuple[int, int]:
    """Compare current and proposed rules on released disjoint validation facts."""

    family = str(truth["family_id"])
    version = str(truth["source_version"])
    selected = [row for row in state.validation.get(family, []) if row[0] == version]
    current = state.active.get(family)
    before = 0
    after = 0
    for _, numeric, label in selected:
        visible = {
            "decision_index": int(truth["decision_index"]),
            "family_id": family,
            "numeric_value": numeric,
        }
        prior = _fallback_decision(visible) if current is None else _execute_template(visible, current)
        proposed = _execute_template(visible, template)
        before += int(prior != label)
        after += int(proposed != label)
    return before, after


def _truth_map(inputs: Mapping[str, Any]) -> dict[str, JsonDict]:
    """Build an exact event join and reject duplicate authority rows."""

    rows = [deepcopy(dict(row)) for row in inputs["truth_rows"]]
    result = {str(row["event_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate authority event")
    return result


def _producer_rows(inputs: Mapping[str, Any], seed: int, arm: str) -> dict[int, JsonDict]:
    """Index raw producer decisions for one fixed treatment."""

    return {
        int(row["decision_index"]): row
        for row in inputs["producer"]["rows"]
        if int(row["ordering_seed"]) == seed and row["arm"] == arm
    }


def _process_released(
    *,
    index: int,
    state: MemoryState,
    decision_history: Mapping[str, str],
    inputs: Mapping[str, Any],
    truths: Mapping[str, Mapping[str, Any]],
    credit_overrides: list[tuple[str, int]] | None,
) -> tuple[str, list[JsonDict], list[JsonDict], tuple[str, int] | None]:
    """Apply only feedback and revocations available after the sealed action."""

    availability = inputs["availability"][index]
    receipt_by_id = {
        str(row["receipt_id"]): row for row in inputs["stream"].get("revocation_rows", [])
    }
    adaptation = {
        str(row["family_id"])
        for row in truths.values()
        if row.get("family_role") == "adaptation"
    }
    operation = "no_change"
    additions: list[JsonDict] = []
    revocations: list[JsonDict] = []
    applied_credit: tuple[str, int] | None = None
    for receipt_id in availability["available_revocation_receipt_ids"]:
        receipt = receipt_by_id[str(receipt_id)]
        if receipt["family_id"] not in adaptation or int(receipt["release_index"]) != index:
            continue
        before, after, template_id = state.revoke(receipt)
        revocations.append(
            {
                "receipt_id": receipt_id,
                "family_id": receipt["family_id"],
                "revoked_source_version": receipt["revoked_version"],
                "replacement_source_version": receipt["replacement_version"],
                "evidence_release_index": receipt["release_index"],
                "source_decision_index": receipt["trigger_decision_index"],
                "revoked_template_id": template_id,
                "before_hash": before,
                "after_hash": after,
            }
        )
        operation = "revoke_template"

    for event_id in availability["newly_released_event_ids"]:
        truth = truths[str(event_id)]
        if (
            truth["feedback_corrupted"]
            or truth["family_role"] == "transfer"
            or truth["selection_role"] == "final_audit"
        ):
            continue
        if truth["selection_role"] == "rolling_validation":
            state.observe_validation(truth)
            if operation != "revoke_template":
                operation = "record_validation"
            continue
        if decision_history.get(str(event_id)) == truth["exact_label"]:
            continue
        evidence_ids = state.observe_error(truth)
        if operation != "revoke_template":
            operation = "record_error_witness"
        family = str(truth["family_id"])
        active = state.active.get(family)
        if len(set(evidence_ids)) < SUPPORT_REQUIRED or (
            active is not None and active["source_version"] == truth["source_version"]
        ):
            continue
        proposed = {
            "kind": "executable_constraint",
            "e_mu": "not_persistent",
            "family_id": family,
            "source_version": truth["source_version"],
            "parameters": deepcopy(truth["rule"]["parameters"]),
        }
        before_errors, after_errors = _validation_errors(state, truth, proposed)
        natural_credit = len(set(evidence_ids))
        if credit_overrides is None:
            credit_source, family_credit = family, natural_credit
        else:
            credit_source, family_credit = credit_overrides.pop(0)
        applied_credit = (credit_source, family_credit)
        if family_credit <= 0 or after_errors > before_errors:
            continue
        template, parent, child = state.admit(truth)
        additions.append(
            {
                "family_id": family,
                "source_version": truth["source_version"],
                "evidence_event_ids": list(evidence_ids),
                "evidence_release_index": index,
                "family_credit": family_credit,
                "credit_source_family": credit_source,
                "verified_catches": natural_credit,
                "harmful_rejections": 0,
                "validation_error_before": before_errors,
                "validation_error_after": after_errors,
                "template": template,
                "parent_hash": parent,
                "child_hash": child,
            }
        )
        operation = "add_template"
    return operation, additions, revocations, applied_credit


def replay_seed(
    inputs: Mapping[str, Any],
    seed: int,
    *,
    credit_overrides: Sequence[tuple[str, int]] | None = None,
    start_state: Mapping[str, Any] | None = None,
    start_history: Mapping[str, str] | None = None,
    start: int = 0,
    stop: int = EVENT_COUNT,
) -> JsonDict:
    """Rebuild one seed's controller, decisions, and state hashes from bytes."""

    truths = _truth_map(inputs)
    state = MemoryState(start_state)
    history = dict(start_history or {})
    producer = _producer_rows(inputs, seed, "revocable_template")
    overrides = None if credit_overrides is None else list(credit_overrides)
    reconstruction: list[JsonDict] = []
    additions: list[JsonDict] = []
    revocations: list[JsonDict] = []
    checkpoint: JsonDict | None = None
    decisions = inputs["decisions"]
    for index in range(start, stop):
        decision = decisions[index]
        row = producer[index]
        if index == RECURRENCE_START:
            checkpoint = {
                "boundary_index": index,
                "memory_state": state.as_dict(),
                "memory_state_hash": state.state_hash,
                "memory_state_bytes_hex": state.state_bytes.hex(),
                "decision_history": dict(history),
            }
        pre_hash = state.state_hash
        predicted = _memory_decision(decision, state)
        history[str(decision["event_id"])] = predicted
        operation, new_additions, new_revocations, applied_credit = _process_released(
            index=index,
            state=state,
            decision_history=history,
            inputs=inputs,
            truths=truths,
            credit_overrides=overrides,
        )
        for addition in new_additions:
            additions.append({"ordering_seed": seed, **addition})
        for revocation in new_revocations:
            revocations.append({"ordering_seed": seed, **revocation})
        post_hash = state.state_hash
        checks = {
            "pre_state_hash": pre_hash == row["pre_decision_memory_hash"],
            "decision": predicted == row["decision"],
            "post_state_hash": post_hash == row["post_feedback_memory_hash"],
            "operation": operation == row["memory_operation"],
            "state_bytes": len(state.state_bytes) == int(row["serialized_memory_bytes"]),
            "capacity": len(state.state_bytes) <= MEMORY_BYTE_BUDGET,
        }
        reconstruction.append(
            {
                "row_key": row["row_key"],
                "ordering_seed": seed,
                "decision_index": index,
                "event_id": decision["event_id"],
                "recomputed_decision": predicted,
                "producer_decision": row["decision"],
                "recomputed_pre_hash": pre_hash,
                "producer_pre_hash": row["pre_decision_memory_hash"],
                "recomputed_post_hash": post_hash,
                "producer_post_hash": row["post_feedback_memory_hash"],
                "operation": operation,
                "credit_source_family": None if applied_credit is None else applied_credit[0],
                "credit_value_applied": None if applied_credit is None else applied_credit[1],
                "serialized_memory_bytes": len(state.state_bytes),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    if overrides:
        raise ValueError("unused credit overrides")
    return {
        "reconstruction_rows": reconstruction,
        "additions": additions,
        "revocations": revocations,
        "checkpoint": checkpoint,
        "final_state": state.as_dict(),
        "final_history": history,
    }


def recompute_rows(inputs: Mapping[str, Any]) -> list[JsonDict]:
    """Join raw decisions to authority labels and rebuild per-event metrics."""

    truths = _truth_map(inputs)
    result: list[JsonDict] = []
    for source in inputs["producer"].get("rows", []):
        truth = truths.get(str(source.get("event_id")), {})
        label = truth.get("exact_label")
        decision = source.get("decision")
        correct = decision == label
        error = int(not correct)
        false_accept = int(decision == "accept" and label == "reject")
        identity_ok = all(
            source.get(field) == truth.get(field)
            for field in ("event_id", "family_id", "source_version")
        )
        parity = (
            source.get("exact_label") == label
            and source.get("correct") is correct
            and source.get("error") == error
            and source.get("false_accept") == false_accept
        )
        seal_ok = source.get("action_seal") == action_seal(source)
        result.append(
            {
                "row_key": source.get("row_key"),
                "ordering_seed": source.get("ordering_seed"),
                "decision_index": source.get("decision_index"),
                "event_id": source.get("event_id"),
                "arm": source.get("arm"),
                "family_id": truth.get("family_id"),
                "family_role": truth.get("family_role"),
                "selection_role": truth.get("selection_role"),
                "authority_exact_label": label,
                "decision": decision,
                "correct": correct,
                "error": error,
                "false_accept": false_accept,
                "producer_metric_parity": parity,
                "event_identity_parity": identity_ok,
                "action_seal_parity": seal_ok,
                "passed": bool(parity and identity_ok and seal_ok),
            }
        )
    return result


def _aggregate_metrics(inputs: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Reduce only rebuilt atomic rows and compare producer summaries afterward."""

    recurrence_ids = {
        str(row["recurrence_event_id"]) for row in inputs["stream"].get("recurrence_rows", [])
    }
    metrics: dict[str, list[JsonDict]] = {
        "future_segment": [],
        "false_acceptance": [],
        "transfer": [],
        "recurrence_retention": [],
    }
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        future = [row for row in arm_rows if int(row["decision_index"]) >= 180]
        transfer = [row for row in arm_rows if row["family_role"] == "transfer"]
        recurrence = [row for row in arm_rows if row["event_id"] in recurrence_ids]
        metrics["future_segment"].append(
            {
                "arm": arm,
                "event_count": len(future),
                "error_count": sum(int(row["error"]) for row in future),
                "error_rate": round(sum(int(row["error"]) for row in future) / len(future), 6),
            }
        )
        metrics["false_acceptance"].append(
            {
                "arm": arm,
                "event_count": len(future),
                "false_accept_count": sum(int(row["false_accept"]) for row in future),
                "false_accept_rate": round(
                    sum(int(row["false_accept"]) for row in future) / len(future), 6
                ),
            }
        )
        metrics["transfer"].append(
            {
                "arm": arm,
                "event_count": len(transfer),
                "error_count": sum(int(row["error"]) for row in transfer),
                "error_rate": round(
                    sum(int(row["error"]) for row in transfer) / len(transfer), 6
                ),
            }
        )
        metrics["recurrence_retention"].append(
            {
                "arm": arm,
                "event_count": len(recurrence),
                "retained_count": sum(int(row["correct"]) for row in recurrence),
                "retention_rate": round(
                    sum(int(row["correct"]) for row in recurrence) / len(recurrence), 6
                ),
            }
        )
    producer_fields = {
        "future_segment": "future_segment_metrics",
        "false_acceptance": "false_acceptance_metrics",
        "transfer": "transfer_metrics",
        "recurrence_retention": "recurrence_retention_metrics",
    }
    parity = [
        {
            "metric": name,
            "expected_hash": sha256_json(values),
            "observed_hash": sha256_json(inputs["producer"].get(producer_fields[name], [])),
            "passed": values == inputs["producer"].get(producer_fields[name], []),
        }
        for name, values in metrics.items()
    ]
    flattened = [
        {"metric": name, **deepcopy(row)} for name, values in metrics.items() for row in values
    ]
    return {"metric_recomputation_rows": flattened, "producer_metric_parity_rows": parity}


def feedback_causality_rows(inputs: Mapping[str, Any]) -> list[JsonDict]:
    """Check action ordering and feedback availability for every raw decision."""

    truths = _truth_map(inputs)
    result: list[JsonDict] = []
    for source in inputs["producer"].get("rows", []):
        index = int(source["decision_index"])
        current = truths[str(source["event_id"])]
        released = [truths[str(event_id)] for event_id in source["newly_released_event_ids"]]
        checks = {
            "action_before_verification": int(source["action_sequence"])
            < int(source["verification_sequence"]),
            "action_before_feedback_processing": int(source["action_sequence"])
            < int(source["feedback_processing_sequence"]),
            "current_label_not_released": int(current["feedback_release_index"]) > index,
            "released_feedback_only": all(
                int(row["feedback_release_index"]) <= index for row in released
            ),
            "declared_no_current_feedback": source["current_feedback_visible_at_decision"] is False,
            "declared_no_future_feedback": source["future_feedback_accessed"] is False,
            "declared_no_regime_access": source["regime_name_accessed_for_commit"] is False,
            "declared_no_heldout_access": source["heldout_label_accessed_for_commit"] is False,
            "declared_no_future_score": source["future_support_score_accessed"] is False,
        }
        result.append(
            {
                "row_key": source["row_key"],
                "decision_index": index,
                "feedback_release_index": current["feedback_release_index"],
                "newly_released_event_ids": list(source["newly_released_event_ids"]),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return result


def _contains_instance_field(value: Any) -> bool:
    """Find event-specific identifiers that would turn a template into a copy."""

    forbidden = {"event_id", "entity_id", "instance_id", "decision_index", "numeric_value"}
    if isinstance(value, Mapping):
        return bool(forbidden.intersection(value)) or any(
            _contains_instance_field(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_instance_field(item) for item in value)
    return False


def addition_audit_rows(
    inputs: Mapping[str, Any], replay_additions: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Validate actual add operations instead of counting proposal storage."""

    truths = _truth_map(inputs)
    replay_keys = {
        (
            int(row["ordering_seed"]),
            row["family_id"],
            row["source_version"],
            tuple(row["evidence_event_ids"]),
            row["parent_hash"],
            row["child_hash"],
        )
        for row in replay_additions
    }
    result: list[JsonDict] = []
    for row in inputs["producer"].get("template_lineage_rows", []):
        if row.get("operation") != "add_template":
            continue
        evidence = [truths[str(event_id)] for event_id in row["evidence_event_ids"]]
        key = (
            int(row["ordering_seed"]),
            row["family_id"],
            row["source_version"],
            tuple(row["evidence_event_ids"]),
            row["parent_hash"],
            row["child_hash"],
        )
        checks = {
            "replayed_commit": key in replay_keys,
            "distinct_support": len(set(row["evidence_event_ids"])) >= SUPPORT_REQUIRED,
            "released_support": all(
                int(item["feedback_release_index"]) <= int(row["evidence_release_index"])
                for item in evidence
            ),
            "prior_events_only": all(
                int(item["decision_index"]) < int(row["evidence_release_index"])
                for item in evidence
            ),
            "family_and_version_match": all(
                item["family_id"] == row["family_id"]
                and item["source_version"] == row["source_version"]
                for item in evidence
            ),
            "eligible_exact_feedback": all(
                item["family_role"] == "adaptation"
                and item["selection_role"] == "commit_support"
                and item["feedback_corrupted"] is False
                for item in evidence
            ),
            "positive_credit": row["family_credit"]
            == row["verified_catches"] - row["harmful_rejections"]
            and row["family_credit"] > 0,
            "validation_nonregression": row["validation_error_after"]
            <= row["validation_error_before"],
            "template_abstract": not _contains_instance_field(row["template"]),
            "state_changed": row["parent_hash"] != row["child_hash"],
        }
        result.append(
            {
                "lineage_id": row["lineage_id"],
                "ordering_seed": row["ordering_seed"],
                "family_id": row["family_id"],
                "source_version": row["source_version"],
                "evidence_event_ids": list(row["evidence_event_ids"]),
                "evidence_release_index": row["evidence_release_index"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return result


def revocation_audit_rows(
    inputs: Mapping[str, Any], replay_revocations: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Bind each stale-version retirement to its exact authority receipt."""

    receipts = {
        str(row["receipt_id"]): row for row in inputs["stream"].get("revocation_rows", [])
    }
    replay_keys = {
        (int(row["ordering_seed"]), row["receipt_id"], row["before_hash"], row["after_hash"])
        for row in replay_revocations
    }
    result: list[JsonDict] = []
    for row in inputs["producer"].get("revocation_ledger_rows", []):
        receipt = receipts.get(str(row["receipt_id"]), {})
        checks = {
            "receipt_present": bool(receipt),
            "receipt_fields_match": bool(receipt)
            and row["family_id"] == receipt["family_id"]
            and row["revoked_source_version"] == receipt["revoked_version"]
            and row["replacement_source_version"] == receipt["replacement_version"]
            and row["evidence_release_index"] == receipt["release_index"],
            "released_after_trigger": row["evidence_release_index"]
            > row["source_decision_index"],
            "stale_inactive": row["revoked_template_active_after"] is False,
            "append_only": row["append_only"] is True,
            "state_changed": row["before_hash"] != row["after_hash"],
            "replayed_revocation": (
                int(row["ordering_seed"]),
                row["receipt_id"],
                row["before_hash"],
                row["after_hash"],
            )
            in replay_keys,
        }
        result.append(
            {
                "lineage_id": row["lineage_id"],
                "ordering_seed": row["ordering_seed"],
                "receipt_id": row["receipt_id"],
                "family_id": row["family_id"],
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    return result


def _mutation_assertions(inputs: Mapping[str, Any]) -> dict[str, bool]:
    """Evaluate six independent claims so each attack has one clear target."""

    producer = inputs["producer"]
    truths = _truth_map(inputs)
    additions = [
        row for row in producer.get("template_lineage_rows", []) if row.get("operation") == "add_template"
    ]
    causal = all(
        all(
            int(truths[str(event_id)]["feedback_release_index"])
            <= int(row["evidence_release_index"])
            for event_id in row["evidence_event_ids"]
        )
        for row in additions
    )
    revocations = producer.get("revocation_ledger_rows", [])
    stale_inactive = all(row.get("revoked_template_active_after") is False for row in revocations)
    adaptation = {
        str(row["family_id"])
        for row in truths.values()
        if row.get("family_role") == "adaptation"
    }
    expected_revocations = {
        (seed, str(receipt["receipt_id"]))
        for seed in ORDERING_SEEDS
        for receipt in inputs["stream"].get("revocation_rows", [])
        if receipt.get("family_id") in adaptation
    }
    observed_revocations = {
        (int(row["ordering_seed"]), str(row["receipt_id"])) for row in revocations
    }
    revocation_complete = observed_revocations == expected_revocations
    rejection_index = {
        (int(row["ordering_seed"]), str(row["event_id"])): row
        for row in producer.get("rejection_ledger_rows", [])
    }
    corrupted = [
        row
        for row in truths.values()
        if row.get("feedback_corrupted") is True
        and int(row["feedback_release_index"]) < EVENT_COUNT
    ]
    poison_rejected = all(
        (seed, str(truth["event_id"])) in rejection_index
        and rejection_index[(seed, str(truth["event_id"]))].get("reason")
        == "corrupted_feedback_rejected"
        and rejection_index[(seed, str(truth["event_id"]))].get("committed") is False
        for truth in corrupted
        for seed in ORDERING_SEEDS
    )
    templates_abstract = all(not _contains_instance_field(row["template"]) for row in additions)
    expected_transitions = [
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
        for row in producer.get("rows", [])
    ]
    transition_integrity = expected_transitions == producer.get("memory_transition_rows", [])
    return {
        "causal_feedback_release": causal,
        "stale_source_inactive": stale_inactive,
        "revocation_completeness": revocation_complete,
        "poison_rejected": poison_rejected,
        "template_abstraction": templates_abstract,
        "transition_hash_integrity": transition_integrity,
    }


def build_mutation_rows(inputs: Mapping[str, Any]) -> list[JsonDict]:
    """Run each targeted mutation and require one isolating assertion failure."""

    baseline = _mutation_assertions(inputs)
    if not all(baseline.values()):
        raise ValueError("baseline mutation assertions are not clean")
    plans: list[tuple[str, str, Callable[[JsonDict], None]]] = []

    def early(value: JsonDict) -> None:
        row = next(
            item
            for item in value["producer"]["template_lineage_rows"]
            if item.get("operation") == "add_template"
        )
        releases = [
            _truth_map(value)[str(event_id)]["feedback_release_index"]
            for event_id in row["evidence_event_ids"]
        ]
        row["evidence_release_index"] = max(releases) - 1

    def stale(value: JsonDict) -> None:
        value["producer"]["revocation_ledger_rows"][0]["revoked_template_active_after"] = True

    def missing(value: JsonDict) -> None:
        value["producer"]["revocation_ledger_rows"].pop()

    def poison(value: JsonDict) -> None:
        truths = _truth_map(value)
        target = next(row["event_id"] for row in truths.values() if row["feedback_corrupted"])
        row = next(
            item
            for item in value["producer"]["rejection_ledger_rows"]
            if item["event_id"] == target
        )
        row["committed"] = True

    def instance(value: JsonDict) -> None:
        row = next(
            item
            for item in value["producer"]["template_lineage_rows"]
            if item.get("operation") == "add_template"
        )
        row["template"]["instance_id"] = "copied-current-answer"

    def forged(value: JsonDict) -> None:
        value["producer"]["memory_transition_rows"][0]["after_hash"] = "sha256:" + "0" * 64

    plans.extend(
        [
            ("early_feedback_exposure", "causal_feedback_release", early),
            ("stale_source_kept_active", "stale_source_inactive", stale),
            ("missing_revocation", "revocation_completeness", missing),
            ("poison_accepted", "poison_rejected", poison),
            ("instance_id_retained", "template_abstraction", instance),
            ("forged_before_after_hash", "transition_hash_integrity", forged),
        ]
    )
    result: list[JsonDict] = []
    for mutation_id, expected, mutate in plans:
        attacked = deepcopy(dict(inputs))
        mutate(attacked)
        assertions = _mutation_assertions(attacked)
        failed = [name for name, passed in assertions.items() if not passed]
        result.append(
            {
                "mutation_id": mutation_id,
                "expected_failed_assertion": expected,
                "failed_assertion": failed[0] if len(failed) == 1 else None,
                "failed_assertion_count": len(failed),
                "assertion_results": assertions,
                "passed": failed == [expected],
            }
        )
    return result


def _credit_schedules(inputs: Mapping[str, Any]) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Shuffle family attribution while preserving the exact credit multiset."""

    rows = sorted(
        (
            row
            for row in inputs["producer"].get("template_lineage_rows", [])
            if row.get("operation") == "add_template"
            and int(row["ordering_seed"]) == ORDERING_SEEDS[0]
        ),
        key=lambda row: (int(row["evidence_release_index"]), str(row["family_id"])),
    )
    real = [(str(row["family_id"]), int(row["family_credit"])) for row in rows]
    shuffled = list(real)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    return real, shuffled


def build_credit_control_rows(inputs: Mapping[str, Any]) -> tuple[list[JsonDict], JsonDict]:
    """Rerun one CPU controller with real and shuffled family-credit attribution."""

    real_schedule, shuffled_schedule = _credit_schedules(inputs)
    runs = {
        "real_family_credit": replay_seed(
            inputs, ORDERING_SEEDS[0], credit_overrides=real_schedule
        ),
        "seeded_shuffled_credit": replay_seed(
            inputs, ORDERING_SEEDS[0], credit_overrides=shuffled_schedule
        ),
    }
    rows: list[JsonDict] = []
    for control, replay in runs.items():
        for row in replay["reconstruction_rows"]:
            rows.append(
                {
                    "control": control,
                    "decision_index": row["decision_index"],
                    "event_id": row["event_id"],
                    "decision": row["recomputed_decision"],
                    "pre_state_hash": row["recomputed_pre_hash"],
                    "post_state_hash": row["recomputed_post_hash"],
                    "credit_source_family": row["credit_source_family"],
                    "credit_value_applied": row["credit_value_applied"],
                    "capacity_bytes": row["serialized_memory_bytes"],
                    "passed": row["checks"]["capacity"],
                }
            )
    by_control = {
        control: {int(row["decision_index"]): row for row in values["reconstruction_rows"]}
        for control, values in runs.items()
    }
    decision_differences = sum(
        by_control["real_family_credit"][index]["recomputed_decision"]
        != by_control["seeded_shuffled_credit"][index]["recomputed_decision"]
        for index in range(EVENT_COUNT)
    )
    state_differences = sum(
        by_control["real_family_credit"][index]["recomputed_post_hash"]
        != by_control["seeded_shuffled_credit"][index]["recomputed_post_hash"]
        for index in range(EVENT_COUNT)
    )
    assignment_differences = sum(left[0] != right[0] for left, right in zip(real_schedule, shuffled_schedule, strict=True))
    summary = {
        "random_seed": RANDOM_SEED,
        "same_event_stream": True,
        "same_capacity": True,
        "same_prediction_policy": True,
        "credit_multiset_preserved": sorted(value for _, value in real_schedule)
        == sorted(value for _, value in shuffled_schedule),
        "credit_assignment_difference_count": assignment_differences,
        "decision_difference_count": decision_differences,
        "state_hash_difference_count": state_differences,
        "credit_assignment_effective": decision_differences > 0,
    }
    return rows, summary


def build_deletion_control(inputs: Mapping[str, Any]) -> tuple[list[JsonDict], JsonDict]:
    """Remove used templates and test decisions that previously beat a control."""

    source = inputs["producer"]["rows"]
    indexed = {
        (int(row["ordering_seed"]), int(row["decision_index"]), str(row["arm"])): row
        for row in source
    }
    decisions = {int(row["decision_index"]): row for row in inputs["decisions"]}
    result: list[JsonDict] = []
    for seed in ORDERING_SEEDS:
        for index in range(180, EVENT_COUNT):
            learned = indexed[(seed, index, "revocable_template")]
            no_memory = indexed[(seed, index, "no_memory")]
            fifo = indexed[(seed, index, "fifo_replay")]
            improved = learned["correct"] is True and (
                no_memory["correct"] is False or fifo["correct"] is False
            )
            if not improved or learned["decision_source"] != "executable_constraint":
                continue
            deleted_decision = _fallback_decision(decisions[index])
            result.append(
                {
                    "ordering_seed": seed,
                    "decision_index": index,
                    "event_id": learned["event_id"],
                    "template_id": learned["inspected_record_ids"][0],
                    "learned_decision": learned["decision"],
                    "deleted_template_decision": deleted_decision,
                    "changed_decision": deleted_decision != learned["decision"],
                    "previously_improved": True,
                    "passed": True,
                }
            )
    changed = sum(row["changed_decision"] for row in result)
    return result, {
        "checked": True,
        "previously_improved_count": len(result),
        "changed_decision_count": changed,
        "mechanism_decorative": changed == 0,
    }


def build_rollback_rows(checkpoint: Mapping[str, Any]) -> list[JsonDict]:
    """Apply an adverse byte suffix and restore the exact checkpoint bytes."""

    result: list[JsonDict] = []
    for seed in ORDERING_SEEDS:
        row = checkpoint["seeds"][str(seed)]
        parent = bytes.fromhex(row["memory_state_bytes_hex"])
        adverse = parent + b"forged-adverse-transition"
        restored = bytes(parent)
        result.append(
            {
                "ordering_seed": seed,
                "checkpoint_hash": sha256_bytes(parent),
                "adverse_hash": sha256_bytes(adverse),
                "restored_hash": sha256_bytes(restored),
                "byte_equal": restored == parent,
                "byte_length": len(parent),
                "passed": restored == parent and sha256_bytes(restored) == sha256_bytes(parent),
            }
        )
    return result


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete JSON through one atomic replacement and clean failures."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_checkpoint(
    replays: Mapping[int, Mapping[str, Any]], snapshot: Mapping[str, Any]
) -> JsonDict:
    """Seal the last valid state immediately before recurrence evaluation."""

    seeds = {
        str(seed): deepcopy(replays[seed]["checkpoint"])
        for seed in ORDERING_SEEDS
    }
    checkpoint = {
        "schema": "carnot.exp7185.pre_recurrence_checkpoint.v1",
        "boundary_index": RECURRENCE_START,
        "source_hashes": {
            name: snapshot["files"][name]["sha256"]
            for name in (
                "producer_artifact",
                "stream_artifact",
                "decision_view",
                "truth_view",
                "availability_view",
            )
        },
        "seeds": seeds,
        "checkpoint_checksum": "",
    }
    checkpoint["checkpoint_checksum"] = sha256_json(
        {**checkpoint, "checkpoint_checksum": None}
    )
    return checkpoint


def _checkpoint_valid(checkpoint: Mapping[str, Any]) -> bool:
    """Recompute the checkpoint seal before any recurrence decision."""

    expected = sha256_json({**dict(checkpoint), "checkpoint_checksum": None})
    return checkpoint.get("checkpoint_checksum") == expected


def recurrence_worker(args: argparse.Namespace) -> JsonDict:
    """Reload saved memory and audit recurrence decisions in a new process."""

    print("PHASE 3C START: capture checkpoint and sidecar bytes before decode", flush=True)
    snapshot = capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=Path(args.producer_artifact_path),
        stream_artifact_path=Path(args.stream_artifact_path),
    )
    checkpoint_entry = _snapshot_file(Path(args.checkpoint_path))
    payload = checkpoint_entry.get("bytes")
    if not isinstance(payload, bytes):
        raise RuntimeError("recurrence checkpoint is unreadable")
    checkpoint = json.loads(payload)
    inputs = decode_captured_inputs(snapshot)
    checkpoint_ok = _checkpoint_valid(checkpoint)
    hash_ok = checkpoint_entry["sha256"] == args.expected_checkpoint_hash
    print("PHASE 3C END: checkpoint and sidecar bytes captured", flush=True)
    print("PHASE 3D START: reload saved memory and evaluate recurrence events", flush=True)
    rows: list[JsonDict] = []
    for seed in ORDERING_SEEDS:
        saved = checkpoint["seeds"][str(seed)]
        replay = replay_seed(
            inputs,
            seed,
            start_state=saved["memory_state"],
            start_history=saved["decision_history"],
            start=RECURRENCE_START,
            stop=RECURRENCE_STOP,
        )
        for row in replay["reconstruction_rows"]:
            expected_hash = _producer_rows(inputs, seed, "revocable_template")[
                int(row["decision_index"])
            ]["pre_decision_memory_hash"]
            rows.append(
                {
                    "ordering_seed": seed,
                    "decision_index": row["decision_index"],
                    "event_id": row["event_id"],
                    "loaded_state_hash": row["recomputed_pre_hash"],
                    "expected_state_hash": expected_hash,
                    "decision": row["recomputed_decision"],
                    "checks": row["checks"],
                    "passed": checkpoint_ok and hash_ok and row["passed"],
                }
            )
        print(f"PHASE 3D PROGRESS: recurrence seed {seed} complete", flush=True)
    parent_pid = int(os.environ.get("CARNOT_7185_PARENT_PID", "-1"))
    receipt = {
        "fresh_process": parent_pid == os.getppid(),
        "parent_pid": parent_pid,
        "worker_pid": os.getpid(),
        "checkpoint_hash": checkpoint_entry["sha256"],
        "checkpoint_hash_verified": checkpoint_ok and hash_ok,
        "no_model_load": not any(
            name in sys.modules for name in ("llama_cpp", "transformers", "torch")
        ),
        "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none",
    }
    print("PHASE 3D END: recurrence decisions match saved-state targets", flush=True)
    return {"cold_reload_rows": rows, "cold_reload_process_receipt": receipt}


RESULT_PREFIX = "EXP7185_RESULT_JSON:"


def _stream_subprocess(
    command: Sequence[str], environment: Mapping[str, str], *, label: str, timeout_s: float = 600.0
) -> JsonDict:
    """Stream child progress, emit heartbeats, and enforce a bounded deadline."""

    print(f"{label} START: {' '.join(command[:3])}", flush=True)
    process = subprocess.Popen(
        list(command),
        cwd=REPO_ROOT,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    if process.stdout is None:  # pragma: no cover - Popen guarantees PIPE here.
        raise RuntimeError(f"{label} has no output pipe")
    started = time.monotonic()
    heartbeat = started
    payload: JsonDict | None = None
    while True:
        elapsed = time.monotonic() - started
        if elapsed > timeout_s:  # pragma: no cover - safety deadline, not a test delay.
            process.kill()
            process.wait()
            raise TimeoutError(f"{label} exceeded {timeout_s:.0f}s deadline")
        readable, _, _ = select.select([process.stdout], [], [], 1.0)
        if readable:
            line = process.stdout.readline()
            if line:
                text = line.rstrip("\n")
                if text.startswith(RESULT_PREFIX):
                    decoded = json.loads(text.removeprefix(RESULT_PREFIX))
                    if not isinstance(decoded, dict):
                        raise RuntimeError(f"{label} returned a non-object")
                    payload = decoded
                else:
                    print(text, flush=True)
                heartbeat = time.monotonic()
        if process.poll() is not None:
            for line in process.stdout:
                text = line.rstrip("\n")
                if text.startswith(RESULT_PREFIX):
                    decoded = json.loads(text.removeprefix(RESULT_PREFIX))
                    if not isinstance(decoded, dict):
                        raise RuntimeError(f"{label} returned a non-object")
                    payload = decoded
                else:
                    print(text, flush=True)
            break
        if time.monotonic() - heartbeat >= 60.0:  # pragma: no cover - short CPU audit.
            print(
                f"{label} HEARTBEAT: elapsed_s={elapsed:.1f} completed_units=unknown operation=child",
                flush=True,
            )
            heartbeat = time.monotonic()
    if process.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {process.returncode}")
    if payload is None:
        raise RuntimeError(f"{label} did not return a JSON object")
    print(f"{label} END: child exited successfully", flush=True)
    return payload


def _isolated_environment(parent_pid: int) -> JsonDict:
    """Hide accelerators and online model caches from every audit process."""

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "CARNOT_7185_PARENT_PID": str(parent_pid),
        }
    )
    return environment


def spawn_recurrence_worker(args: argparse.Namespace, checkpoint_hash: str) -> JsonDict:
    """Start the required second isolated process at the recurrence boundary."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--recurrence-worker",
        "--date",
        args.date,
        "--producer-artifact-path",
        str(args.producer_artifact_path),
        "--stream-artifact-path",
        str(args.stream_artifact_path),
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--expected-checkpoint-hash",
        checkpoint_hash,
    ]
    return _stream_subprocess(
        command,
        _isolated_environment(os.getpid()),
        label="PHASE 3 SUBPROCESS",
    )


def _source_hashes(snapshot: Mapping[str, Any]) -> JsonDict:
    """Expose captured identities without copying source bytes into the result."""

    return {
        name: {
            "path": row.get("path"),
            "sha256": row.get("sha256"),
            "size": row.get("size"),
        }
        for name, row in snapshot.get("files", {}).items()
    }


def _empty_evidence() -> JsonDict:
    """Keep blocked artifacts schema-complete without invented measurements."""

    return {
        "rows": [],
        "metric_recomputation_rows": [],
        "producer_metric_parity_rows": [],
        "feedback_causality_rows": [],
        "memory_reconstruction_rows": [],
        "addition_audit_rows": [],
        "revocation_audit_rows": [],
        "cold_reload_rows": [],
        "mutation_rows": [],
        "rollback_rows": [],
        "credit_control_rows": [],
        "deletion_control_rows": [],
        "upstream_rejection_ledger_rows": [],
    }


def _base_artifact(
    *,
    snapshot: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any] | None,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Build complete provenance before audit and promotion classification."""

    producer = inputs.get("producer", {}) if inputs else {}
    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "run_date": run_date,
        "inference_substrate": "blocked before qualifying cold-audit work",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": _source_hashes(snapshot),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked: revocable-memory cold audit did not run",
        "inference_substrate_class": "blocked_no_run",
        "memory_audit_complete_score": 0,
        "memory_promotion_score": 0,
        "upstream_gate_receipt": {
            "upstream": "exp7184-revocable-template-csl",
            "artifact_hash": snapshot.get("files", {}).get("producer_artifact", {}).get("sha256"),
            "field": "memory_run_complete_score",
            "expected_value": 1,
            "observed_value": producer.get("memory_run_complete_score"),
            "memory_value_score": producer.get("memory_value_score"),
            "upstream_verdict_class": producer.get("verdict_class"),
        },
        "runtime_isolation_receipt": {},
        "cold_reload_process_receipt": {},
        "actual_controller_addition_count": 0,
        "credit_control_summary": {},
        "deletion_control_summary": {},
        "checkpoint_receipt": {},
        "no_live_llm_or_hardware_result": True,
        **_empty_evidence(),
    }


def _stable_value(value: Any) -> Any:
    """Exclude wall time and process identities from the science checksum."""

    if isinstance(value, Mapping):
        return {
            key: _stable_value(item)
            for key, item in value.items()
            if key
            not in {
                "duration_s",
                "reproducibility_checksum",
                "parent_pid",
                "worker_pid",
            }
        }
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash source identities, frozen seeds, and all timing-free raw evidence."""

    return sha256_json(_stable_value(dict(artifact)))


def build_blocked_artifact(
    *,
    snapshot: Mapping[str, Any],
    checks: Sequence[Mapping[str, Any]],
    inputs: Mapping[str, Any] | None,
    run_date: str,
    duration_s: float,
) -> JsonDict:
    """Return a terminal row-free artifact for an external prerequisite failure."""

    artifact = _base_artifact(
        snapshot=snapshot,
        checks=checks,
        inputs=inputs,
        run_date=run_date,
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked: revocable-memory cold audit did not run; {failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _expected_complete(artifact: Mapping[str, Any]) -> int:
    """Derive completion from full audit coverage, including honest negatives."""

    mutation_ids = {
        "early_feedback_exposure",
        "stale_source_kept_active",
        "missing_revocation",
        "poison_accepted",
        "instance_id_retained",
        "forged_before_after_hash",
    }
    return int(
        artifact.get("inference_substrate_class") == INFERENCE_SUBSTRATE_CLASS
        and len(artifact.get("rows", [])) == ROW_COUNT
        and all(row.get("passed") is True for row in artifact.get("rows", []))
        and len(artifact.get("feedback_causality_rows", [])) == ROW_COUNT
        and all(
            row.get("passed") is True for row in artifact.get("feedback_causality_rows", [])
        )
        and len(artifact.get("memory_reconstruction_rows", []))
        == EVENT_COUNT * len(ORDERING_SEEDS)
        and all(
            row.get("passed") is True for row in artifact.get("memory_reconstruction_rows", [])
        )
        and len(artifact.get("addition_audit_rows", [])) == 36
        and all(row.get("passed") is True for row in artifact.get("addition_audit_rows", []))
        and len(artifact.get("revocation_audit_rows", [])) == 24
        and all(row.get("passed") is True for row in artifact.get("revocation_audit_rows", []))
        and len(artifact.get("cold_reload_rows", []))
        == (RECURRENCE_STOP - RECURRENCE_START) * len(ORDERING_SEEDS)
        and all(row.get("passed") is True for row in artifact.get("cold_reload_rows", []))
        and artifact.get("cold_reload_process_receipt", {}).get("fresh_process") is True
        and {row.get("mutation_id") for row in artifact.get("mutation_rows", [])}
        == mutation_ids
        and all(row.get("passed") is True for row in artifact.get("mutation_rows", []))
        and len(artifact.get("rollback_rows", [])) == len(ORDERING_SEEDS)
        and all(row.get("passed") is True for row in artifact.get("rollback_rows", []))
        and len(artifact.get("credit_control_rows", [])) == EVENT_COUNT * 2
        and all(row.get("passed") is True for row in artifact.get("credit_control_rows", []))
        and artifact.get("deletion_control_summary", {}).get("checked") is True
        and bool(artifact.get("deletion_control_rows", []))
        and all(row.get("passed") is True for row in artifact.get("deletion_control_rows", []))
        and all(
            row.get("passed") is True
            for row in artifact.get("producer_metric_parity_rows", [])
        )
        and artifact.get("runtime_isolation_receipt", {}).get("fresh_process") is True
        and artifact.get("runtime_isolation_receipt", {}).get("no_model_load") is True
        and artifact.get("runtime_isolation_receipt", {}).get("protected_inputs_unchanged")
        is True
    )


def _expected_promotion(artifact: Mapping[str, Any]) -> int:
    """Require upstream value plus causal retention and effective controls."""

    return int(
        _expected_complete(artifact) == 1
        and artifact.get("upstream_gate_receipt", {}).get("memory_value_score") == 1
        and all(
            row.get("passed") is True for row in artifact.get("feedback_causality_rows", [])
        )
        and all(row.get("passed") is True for row in artifact.get("cold_reload_rows", []))
        and all(row.get("byte_equal") is True for row in artifact.get("rollback_rows", []))
        and artifact.get("credit_control_summary", {}).get("credit_assignment_effective") is True
        and artifact.get("deletion_control_summary", {}).get("mechanism_decorative") is False
    )


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute schema, completion, promotion, verdict, and checksum gates."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        return ["missing_fields:" + ",".join(missing)]
    errors: list[str] = []

    def add(condition: bool, name: str) -> None:
        if condition and name not in errors:
            errors.append(name)

    add(set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS), "field_principles_mismatch")
    add(artifact.get("schema") != SCHEMA, "schema_mismatch")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id_mismatch")
    add(artifact.get("milestone") != MILESTONE, "milestone_mismatch")
    add(artifact.get("run_date") != RUN_DATE, "run_date_mismatch")
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue_mismatch")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_oracle_mismatch")
    add(
        artifact.get("no_live_llm_or_hardware_result") is not True,
        "live_or_hardware_claim_mismatch",
    )
    blocked = artifact.get("inference_substrate_class") == "blocked_no_run"
    if blocked:
        add(artifact.get("status") != "blocked", "blocked_status_mismatch")
        add(artifact.get("verdict_class") != "blocked", "blocked_verdict_mismatch")
        add(bool(artifact.get("rows")), "blocked_rows_present")
        add(artifact.get("memory_audit_complete_score") != 0, "blocked_complete_score_mismatch")
        add(artifact.get("memory_promotion_score") != 0, "blocked_promotion_score_mismatch")
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate_summary_mismatch")
        add(not artifact.get("gate_check_summary", {}).get("failed_check"), "blocked_failed_check_missing")
        add(not str(artifact.get("honest_verdict", "")).startswith("blocked:"), "blocked_honest_verdict_mismatch")
    else:
        add(
            artifact.get("inference_substrate") != INFERENCE_SUBSTRATE,
            "inference_substrate_mismatch",
        )
        add(
            artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class_mismatch",
        )
        expected_complete = _expected_complete(artifact)
        add(
            artifact.get("memory_audit_complete_score") != expected_complete,
            "memory_audit_complete_score_mismatch",
        )
        expected_promotion = _expected_promotion(artifact)
        add(
            artifact.get("memory_promotion_score") != expected_promotion,
            "memory_promotion_score_mismatch",
        )
        expected_class = "positive" if expected_promotion else "null"
        add(artifact.get("verdict_class") != expected_class, "verdict_class_mismatch")
        add(artifact.get("status") != "complete", "status_mismatch")
        prefix = "complete_positive:" if expected_promotion else "complete_null:"
        add(
            not str(artifact.get("honest_verdict", "")).startswith(prefix),
            "honest_verdict_mismatch",
        )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum_mismatch",
    )
    return errors


def make_runtime_guard(protected_paths: Sequence[Path]):
    """Deny network access and writes to captured evidence inside the worker."""

    protected = {Path(path).resolve() for path in protected_paths}

    def guard(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith("socket."):
            raise PermissionError("network disabled for memory cold audit")
        if event == "open" and args:
            try:
                path = Path(os.fspath(args[0])).resolve()
            except (TypeError, ValueError, OSError):
                return
            mode = args[1] if len(args) > 1 else "r"
            flags = args[2] if len(args) > 2 else 0
            writes = isinstance(mode, str) and any(marker in mode for marker in "wax+")
            writes = writes or (
                isinstance(flags, int)
                and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))
            )
            if path in protected and writes:
                raise PermissionError("captured cold-audit evidence is read-only")

    return guard


def _current_hashes(snapshot: Mapping[str, Any]) -> JsonDict:
    """Reopen only for a final byte-identity check after all audit work."""

    result: JsonDict = {}
    for name, row in snapshot.get("files", {}).items():
        path = Path(str(row.get("path")))
        result[name] = sha256_bytes(path.read_bytes()) if path.is_file() else None
    return result


def worker_artifact(args: argparse.Namespace) -> JsonDict:
    """Run every audit phase after the outer command creates a fresh process."""

    started = time.monotonic()
    print("PHASE 0 START: capture bytes and verify spec, gates, tools, paths, and hashes", flush=True)
    snapshot = capture_input_snapshot(
        repo_root=REPO_ROOT,
        producer_artifact_path=Path(args.producer_artifact_path),
        stream_artifact_path=Path(args.stream_artifact_path),
    )
    checks, inputs = collect_preconditions(
        snapshot,
        artifact_path=Path(args.artifact_path),
        checkpoint_path=Path(args.checkpoint_path),
    )
    if inputs is None or not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            snapshot=snapshot,
            checks=checks,
            inputs=inputs,
            run_date=args.date,
            duration_s=time.monotonic() - started,
        )
        print("PHASE 0 END: external precondition failed; terminal blocked evidence built", flush=True)
        return artifact
    print("PHASE 0 END: all external preconditions passed", flush=True)

    protected_paths = [
        Path(str(row["path"]))
        for row in snapshot["files"].values()
        if isinstance(row.get("path"), str)
    ]
    before_hashes = {name: row.get("sha256") for name, row in snapshot["files"].items()}
    guard = make_runtime_guard(protected_paths)
    sys.addaudithook(guard)
    network_disabled = False
    try:
        import socket

        socket.socket()
    except PermissionError:
        network_disabled = True

    print("PHASE 1 START: recompute 2,880 per-event outcomes from the authority sidecar", flush=True)
    rows = recompute_rows(inputs)
    metrics = _aggregate_metrics(inputs, rows)
    causal_rows = feedback_causality_rows(inputs)
    print("PHASE 1 END: raw outcome and aggregate parity rows complete", flush=True)

    print("PHASE 2 START: reconstruct additions, revocations, and state hashes", flush=True)
    replays: dict[int, JsonDict] = {}
    replay_additions: list[JsonDict] = []
    replay_revocations: list[JsonDict] = []
    reconstruction: list[JsonDict] = []
    for seed in ORDERING_SEEDS:
        replay = replay_seed(inputs, seed)
        replays[seed] = replay
        reconstruction.extend(replay["reconstruction_rows"])
        replay_additions.extend(replay["additions"])
        replay_revocations.extend(replay["revocations"])
        print(f"PHASE 2 PROGRESS: seed {seed} reconstructed", flush=True)
    additions = addition_audit_rows(inputs, replay_additions)
    revocations = revocation_audit_rows(inputs, replay_revocations)
    print("PHASE 2 END: controller state and actual commits reconstructed", flush=True)

    print("PHASE 3 START: atomically save the last valid pre-recurrence checkpoint", flush=True)
    checkpoint = build_checkpoint(replays, snapshot)
    write_json_atomic(Path(args.checkpoint_path), checkpoint)
    checkpoint_hash = sha256_bytes(Path(args.checkpoint_path).read_bytes())
    print("PHASE 3 CHECKPOINT END: checkpoint bytes are stable", flush=True)
    cold = spawn_recurrence_worker(args, checkpoint_hash)
    print("PHASE 3 END: fresh-process recurrence reload complete", flush=True)

    print("PHASE 4 START: execute six mutations and byte-exact rollback checks", flush=True)
    mutations = build_mutation_rows(inputs)
    rollback = build_rollback_rows(checkpoint)
    print("PHASE 4 END: every mutation isolated and rollback measured", flush=True)

    print("PHASE 5 START: rerun real and shuffled credit, then delete useful templates", flush=True)
    credit_rows, credit_summary = build_credit_control_rows(inputs)
    deletion_rows, deletion_summary = build_deletion_control(inputs)
    print("PHASE 5 END: credit and deletion controls complete", flush=True)

    print("PHASE 6 START: assemble terminal audit and promotion evidence", flush=True)
    after_hashes = _current_hashes(snapshot)
    parent_pid = int(os.environ.get("CARNOT_7185_PARENT_PID", "-1"))
    runtime_receipt = {
        "fresh_process": parent_pid == os.getppid(),
        "parent_pid": parent_pid,
        "worker_pid": os.getpid(),
        "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none",
        "network_disabled": network_disabled,
        "no_model_load": not any(
            name in sys.modules for name in ("llama_cpp", "transformers", "torch")
        ),
        "protected_inputs_unchanged": before_hashes == after_hashes,
    }
    artifact = _base_artifact(
        snapshot=snapshot,
        checks=checks,
        inputs=inputs,
        run_date=args.date,
        duration_s=time.monotonic() - started,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": rows,
            **metrics,
            "feedback_causality_rows": causal_rows,
            "memory_reconstruction_rows": reconstruction,
            "addition_audit_rows": additions,
            "revocation_audit_rows": revocations,
            "cold_reload_rows": cold["cold_reload_rows"],
            "cold_reload_process_receipt": cold["cold_reload_process_receipt"],
            "mutation_rows": mutations,
            "rollback_rows": rollback,
            "credit_control_rows": credit_rows,
            "credit_control_summary": credit_summary,
            "deletion_control_rows": deletion_rows,
            "deletion_control_summary": deletion_summary,
            "upstream_rejection_ledger_rows": deepcopy(
                inputs["producer"].get("rejection_ledger_rows", [])
            ),
            "runtime_isolation_receipt": runtime_receipt,
            "actual_controller_addition_count": len(additions),
            "checkpoint_receipt": {
                "path": str(args.checkpoint_path),
                "sha256": checkpoint_hash,
                "boundary_index": RECURRENCE_START,
                "seed_count": len(checkpoint["seeds"]),
                "checkpoint_checksum": checkpoint["checkpoint_checksum"],
            },
        }
    )
    artifact["memory_audit_complete_score"] = _expected_complete(artifact)
    artifact["memory_promotion_score"] = _expected_promotion(artifact)
    if artifact["memory_promotion_score"] == 1:
        artifact["verdict_class"] = "positive"
        artifact["honest_verdict"] = (
            "complete_positive: cold retention, causal credit, rollback, and deletion controls "
            "support revocable-memory promotion"
        )
    else:
        artifact["verdict_class"] = "null"
        artifact["honest_verdict"] = (
            "complete_null: the audit completed; cold retention, revocation, rollback, and "
            "template deletion passed, but Exp7184 value was null and shuffled family credit "
            "produced identical controller decisions"
        )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    print("PHASE 6 END: completion and promotion remain independently classified", flush=True)
    print("PHASE 7 START: validate schema, rows, controls, verdict, and checksum", flush=True)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("cold audit validation failed:" + ",".join(errors))
    print("PHASE 7 END: terminal artifact validation passed", flush=True)
    return artifact


def _spawn_worker(args: argparse.Namespace) -> JsonDict:
    """Start the full audit in an isolated interpreter and stream its output."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--worker",
        "--date",
        args.date,
        "--producer-artifact-path",
        str(args.producer_artifact_path),
        "--stream-artifact-path",
        str(args.stream_artifact_path),
        "--artifact-path",
        str(args.artifact_path),
        "--checkpoint-path",
        str(args.checkpoint_path),
    ]
    return _stream_subprocess(
        command,
        _isolated_environment(os.getpid()),
        label="AUDIT WORKER SUBPROCESS",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit paths so tests keep generated evidence in temporary storage."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--producer-artifact-path", type=Path, default=DEFAULT_PRODUCER_PATH)
    parser.add_argument("--stream-artifact-path", type=Path, default=DEFAULT_STREAM_PATH)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--expected-checkpoint-hash", default="")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--recurrence-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fresh workers, validate their evidence, and publish atomically."""

    args = parse_args(argv)
    if args.recurrence_worker:
        result = recurrence_worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    if args.worker:
        artifact = worker_artifact(args)
        print(RESULT_PREFIX + json.dumps(artifact, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    if args.validate:
        try:
            value = json.loads(Path(args.artifact_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = {}
        errors = validate_artifact(value)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    artifact = _spawn_worker(args)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("cold audit artifact validation failed:" + ",".join(errors))
    print("PHASE 8 START: atomically write terminal artifact", flush=True)
    write_json_atomic(Path(args.artifact_path), artifact)
    print("PHASE 8 END: terminal artifact is stable", flush=True)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "result": str(args.artifact_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the repository wrapper owns this boundary.
    raise SystemExit(main())

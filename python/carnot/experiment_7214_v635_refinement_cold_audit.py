"""Audit committed predicate refinement in cold processes and causal shadows.

The audit reads the producer's sealed rows. It rebuilds metrics without the
producer reducer. It also replays saved public decisions after state reload and
after distinct state interventions. No model is loaded or invoked.

Spec refs: REQ-CL-7214 and SCENARIO-CL-7214-*.
"""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import shutil
import socket
import statistics
import sys
import tempfile
import time
from typing import Any

from carnot import experiment_7185_v633_memory_cold_audit as cold_support
from carnot import experiment_7198_v634_feedback_capacity_stream as stream_source
from carnot import experiment_7212_v635_refinement_fixture as fixture_source
from carnot import experiment_7213_v635_refinement_learning as producer_source
from carnot.memory import transactional_constraint_memory as transactional


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = 7214
SCHEMA = "carnot.exp7214.v635_refinement_cold_audit.v1"
MILESTONE = "2026.09.635"
RUN_DATE = "20260911"
RANDOM_SEED = 7_214_202_609_11
BOOTSTRAP_SEED = 7_213_001
BOOTSTRAP_DRAWS = 10_000
STREAM_SEEDS = tuple(producer_source.STREAM_SEEDS)
MODEL_SPECS: list[JsonDict] = []
MODEL_INVOKED = False
INFERENCE_SUBSTRATE = (
    "CPU exact public-input parsing, committed-predicate replay, independent "
    "event reduction, paired stream bootstrap, and transactional rejection probes; "
    "no LLM invocation"
)
INFERENCE_SUBSTRATE_CLASS = "cpu_exact_solver_or_simulator"
EXECUTION_VENUE = "host"
PROBE_START = 992
PROBE_STOP = 1_024
RESULT_PREFIX = cold_support.RESULT_PREFIX

DEFAULT_PRODUCER_PATH = Path("results/experiment_7213_v635_refinement_learning.json")
EXPECTED_PRODUCER_SHA256 = "sha256:62e68c63b8fc36179ebdc8dca46e102ee52f784cf770d561a39aa4d66595e265"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_7214_v635_refinement_cold_audit.json")
DEFAULT_CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7214_v635_refinement_cold_audit.json"
)
WRAPPER_PATH = REPO_ROOT / "scripts/experiments/experiment_7214_v635_refinement_cold_audit.py"
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
    Path("python/carnot/experiment_7198_v634_feedback_capacity_stream.py"),
    Path("python/carnot/experiment_7199_v634_bounded_acquisition.py"),
    Path("python/carnot/experiment_7200_v634_acquisition_cold_audit.py"),
    Path("python/carnot/experiment_7212_v635_refinement_fixture.py"),
    Path("python/carnot/experiment_7213_v635_refinement_learning.py"),
    Path("python/carnot/experiment_7214_v635_refinement_cold_audit.py"),
    Path("python/carnot/memory/transactional_constraint_memory.py"),
    Path("scripts/experiments/experiment_7214_v635_refinement_cold_audit.py"),
    Path("tests/python/test_experiment_7214_v635_refinement_cold_audit.py"),
    SPEC_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "milestone",
    "field_principles",
    "status",
    "run_date",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "execution_host",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "sample_size_budget",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "refinement_audit_complete_score",
    "memory_promotion_score",
    "cold_reload_rows",
    "causal_control_rows",
    "rollback_rows",
    "metric_recomputation_rows",
    "MODEL_SPECS",
    "model_invoked",
    "comparison_recomputation_rows",
    "source_grounding_rows",
    "producer_gate_receipt",
    "runtime_isolation_receipt",
    "checkpoint_receipt",
    "audit_errors",
    "no_model_weight_mutation",
    "certificate_published",
    "default_pipeline_modified",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "schema": "A versioned schema makes incompatible readers fail closed.",
    "experiment_id": "A fixed ID prevents another task from supplying this evidence.",
    "milestone": "The milestone binds the audit to the V635 contract.",
    "field_principles": "Echo the reason for each field beside its actual evidence.",
    "status": "Write a terminal artifact only after completion or a diagnosed external block.",
    "run_date": "Use 20260911; do not substitute an upstream experiment date.",
    "preconditions_checked": "Record the actual resource, code and gate observations.",
    "inference_substrate": "Describe executed computation, not the intended workload.",
    "inference_substrate_class": "The actual operation determines its duration floor.",
    "execution_venue": "Use exactly host, kv260, gatemate or polarfire; this task uses host.",
    "execution_host": "Put the actual hostname here, never inside execution_venue.",
    "duration_s": "Measure monotonic work time; do not pad it to pass a floor.",
    "source_artifact_hashes": "Bind source code, input data and frozen contracts to the claim.",
    "rows": "Keep unit_id, arm, seed, metric, error and abstention for each comparison.",
    "sample_size_budget": "Retain planned, attempted, completed, censored and independent-unit counts.",
    "random_seed": "Freeze stochastic choices before held-out outcomes are read.",
    "reproducibility_checksum": "Hash the inputs, code, settings and raw rows.",
    "gate_check_summary": "Every blocked verdict names failed check, upstream, field, expected and observed value.",
    "verifier_is_oracle": "Same correctness authority remains circular even with a separate implementation.",
    "verdict_class": "Use exactly positive, circular_positive, null, blocked, disqualified, or partial.",
    "honest_verdict": "Use complete findings or blocked external absence; readiness is not scientific value.",
    "refinement_audit_complete_score": "A causal audit can finish even when useful learning is absent.",
    "memory_promotion_score": "Promotion requires producer value and independent cold causal checks.",
    "cold_reload_rows": "Real subprocess decisions establish persistence rather than dictionary equality.",
    "causal_control_rows": "Template deletion, full reset and shuffled feedback isolate distinct mechanisms.",
    "rollback_rows": "Rejected poison and stale versions must leave the prior snapshot intact.",
    "metric_recomputation_rows": "Independent event-level calculations protect the learning headline.",
    "MODEL_SPECS": "Empty because this task invokes no LLM.",
    "model_invoked": "False; upstream inference is only cited.",
    "comparison_recomputation_rows": "Independent stream intervals protect every fixed producer comparison.",
    "source_grounding_rows": "Public parsing and two exact executors must agree before scoring.",
    "producer_gate_receipt": "Producer completion and scientific value remain separate downstream facts.",
    "runtime_isolation_receipt": "Process, network and model observations limit the audit claim.",
    "checkpoint_receipt": "Cold workers consume stable task-owned bytes outside the result.",
    "audit_errors": "A complete audit has no hidden failed owned check.",
    "no_model_weight_mutation": "This audit changes no model weights.",
    "certificate_published": "A cold audit does not publish a correctness certificate.",
    "default_pipeline_modified": "An audit cannot silently enable its measured mechanism.",
}


class ExperimentPaths:
    """Keep cold state outside the terminal artifact."""

    def __init__(self, checkpoint: Path, artifact: Path) -> None:
        self.checkpoint = checkpoint
        self.artifact = artifact

    @classmethod
    def defaults(cls) -> ExperimentPaths:
        """Return the paths used by the required command."""

        return cls(DEFAULT_CHECKPOINT_PATH, DEFAULT_ARTIFACT_PATH)

    @classmethod
    def under(cls, root: Path) -> ExperimentPaths:
        """Place test evidence below one caller-owned directory."""

        return cls(
            root / "checkpoints" / "experiment_7214_v635_refinement_cold_audit.json",
            root / "experiment_7214_v635_refinement_cold_audit.json",
        )


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve repository-relative evidence without changing absolute paths."""

    value = Path(path)
    return value if value.is_absolute() else repo_root / value


def _sha256_path(path: Path) -> str | None:
    """Hash exact bytes in chunks while preserving a missing result as None."""

    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    except OSError:
        return None


def _load_object(path: Path) -> JsonDict:
    """Read one JSON object and keep malformed input as failed evidence."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _path_writable(path: Path) -> bool:
    """Check the nearest existing parent without creating evidence early."""

    parent = path.parent
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def unwrap_principled(value: Any) -> Any:
    """Unwrap only the exact two-key principle and value representation."""

    if (
        isinstance(value, Mapping)
        and set(value) == {"principle", "value"}
        and isinstance(value.get("principle"), str)
    ):
        return value["value"]
    return value


def gate_check(
    check: str,
    upstream: str,
    field_name: str,
    expected: Any,
    observed: Any,
) -> JsonDict:
    """Store exact expected and observed values for one prerequisite."""

    actual = unwrap_principled(observed)
    return {
        "check": check,
        "upstream": upstream,
        "field": field_name,
        "expected_value": expected,
        "observed_value": actual,
        "passed": actual == expected,
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed gate and retain the full ledger."""

    failed = next((row for row in checks if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": [dict(row) for row in checks],
        "failed_check": None if failed is None else failed["check"],
        "upstream": None if failed is None else failed["upstream"],
        "field": None if failed is None else failed["field"],
        "expected_value": None if failed is None else failed["expected_value"],
        "observed_value": None if failed is None else failed["observed_value"],
    }


_QUARANTINE_KEYS = (
    "artifact_quarantined",
    "upstream_quarantined",
    "quarantine_flag",
    "quarantined",
    "excluded_from_use",
    "flagged_adversarial",
)


def quarantine_state(
    upstream: Mapping[str, Any],
    exclusion_text: str,
    artifact_name: str,
    task_id: str,
) -> JsonDict:
    """Check artifact flags and the independent manifest as separate sources."""

    flags = {key: unwrap_principled(upstream[key]) for key in _QUARANTINE_KEYS if key in upstream}
    matches = [marker for marker in (artifact_name, task_id) if marker in exclusion_text]
    return {
        "quarantined": any(value is True for value in flags.values()) or bool(matches),
        "declared_flags": flags,
        "exclusion_manifest_matches": matches,
    }


def _task_identity(text: str) -> JsonDict:
    """Read task identity from only the Exp7214 roadmap block."""

    match = re.search(r"(?ms)^- id: exp7214-refinement-cold-audit\n(.*?)(?=^- id:|\Z)", text)
    block = "" if match is None else match.group(0)
    return {
        "id": "exp7214-refinement-cold-audit" if block else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in block else None,
        "deliverable": (
            str(DEFAULT_ARTIFACT_PATH) if f"deliverable: {DEFAULT_ARTIFACT_PATH}" in block else None
        ),
    }


def _stable_without_timing(value: Any) -> Any:
    """Apply the producer's published checksum exclusions independently."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_without_timing(item)
            for key, item in value.items()
            if not str(key).endswith("_ns")
            and key not in {"duration_s", "execution_host", "checkpoint_hash"}
        }
    if isinstance(value, list):
        return [_stable_without_timing(item) for item in value]
    return value


def independent_producer_raw_checksum(producer: Mapping[str, Any]) -> str:
    """Hash producer row collections without calling its checksum helper."""

    return transactional.sha256_json(
        {
            "rows": producer.get("rows", []),
            "decision_rows": producer.get("decision_rows", []),
            "query_rows": producer.get("query_rows", []),
            "commit_deletion_rows": producer.get("commit_deletion_rows", []),
        }
    )


def independent_producer_checksum(producer: Mapping[str, Any]) -> str:
    """Rebuild the producer science checksum from its published field contract."""

    fields = (
        "schema",
        "experiment_id",
        "milestone",
        "run_date",
        "source_artifact_hashes",
        "rows",
        "sample_size_budget",
        "random_seed",
        "gate_check_summary",
        "verifier_is_oracle",
        "verdict_class",
        "honest_verdict",
        "refinement_run_complete_score",
        "refinement_value_score",
        "continuous_self_learning_task",
        "no_model_weight_mutation",
        "acceptance_gate_learning",
        "decision_rows",
        "query_rows",
        "commit_deletion_rows",
        "comparison_rows",
        "panel_conformance_errors",
        "upstream_receipt",
        "future_hardware_path",
    )
    selected = {name: producer.get(name) for name in fields}
    return transactional.sha256_json(_stable_without_timing(selected))


def _producer_declared_paths(producer: Mapping[str, Any]) -> dict[str, str]:
    """Return the checkpoint, fixture, and raw paths that the producer binds."""

    declared = unwrap_principled(producer.get("source_artifact_hashes", {}))
    selected: dict[str, str] = {}
    if isinstance(declared, Mapping):
        for path, digest in declared.items():
            path_text = str(path)
            if (
                path_text.startswith("results/streams/experiment_7212/")
                or path_text == "results/experiment_7212_v635_refinement_fixture.json"
                or path_text == "results/checkpoints/experiment_7213_v635_refinement_learning.json"
                or path_text.startswith("python/carnot/experiment_721")
                or path_text == "python/carnot/memory/transactional_constraint_memory.py"
            ):
                selected[path_text] = str(digest)
    return selected


def collect_preconditions(
    repo_root: Path,
    producer_path: Path,
    paths: ExperimentPaths,
) -> tuple[list[JsonDict], JsonDict, dict[str, str | None]]:
    """Authenticate all upstream bytes before outcome rows enter the audit."""

    root = Path(repo_root)
    resolved_producer = _resolve(root, producer_path)
    producer = _load_object(resolved_producer)
    source_hashes = {str(path): _sha256_path(root / path) for path in SOURCE_PATHS}
    source_hashes[str(producer_path)] = _sha256_path(resolved_producer)
    declared_paths = _producer_declared_paths(producer)
    actual_declared = {path: _sha256_path(_resolve(root, path)) for path in declared_paths}
    source_hashes.update(actual_declared)

    spec_text = (
        (root / SPEC_PATH).read_text(encoding="utf-8") if (root / SPEC_PATH).is_file() else ""
    )
    roadmap_path = root / "research-roadmap.yaml"
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.is_file() else ""
    references_path = root / "research-references.md"
    references_text = (
        references_path.read_text(encoding="utf-8") if references_path.is_file() else ""
    )
    exclusion_path = root / "ops/exclusion_manifest.yaml"
    exclusion_text = exclusion_path.read_text(encoding="utf-8") if exclusion_path.is_file() else ""
    quarantine = quarantine_state(
        producer,
        exclusion_text,
        DEFAULT_PRODUCER_PATH.name,
        "exp7213-refinement-learning",
    )
    imports = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "carnot.experiment_7198_v634_feedback_capacity_stream",
            "carnot.experiment_7212_v635_refinement_fixture",
            "carnot.experiment_7213_v635_refinement_learning",
            "carnot.memory.transactional_constraint_memory",
        )
    }
    tools = {
        "python": Path(sys.executable).is_file(),
        "jq": shutil.which("jq") is not None,
        "sha256sum": shutil.which("sha256sum") is not None,
    }
    destinations = {
        "checkpoint_under_results_checkpoints": (
            "results/checkpoints" in paths.checkpoint.as_posix()
            or paths.checkpoint.parent.name == "checkpoints"
        ),
        "checkpoint_writable": _path_writable(paths.checkpoint),
        "artifact_writable": _path_writable(paths.artifact),
    }
    producer_gate = unwrap_principled(producer.get("gate_check_summary", {}))
    producer_checkpoint_path = Path(str(unwrap_principled(producer.get("checkpoint_path", ""))))
    producer_checkpoint_hash = _sha256_path(_resolve(root, producer_checkpoint_path))
    expected_identity = {
        "id": "exp7214-refinement-cold-audit",
        "milestone": MILESTONE,
        "deliverable": str(DEFAULT_ARTIFACT_PATH),
    }
    source_sizes = {
        str(path): (root / path).stat().st_size if (root / path).is_file() else None
        for path in SOURCE_PATHS
    }
    raw_checksum_valid = bool(producer) and producer.get(
        "raw_rows_checksum"
    ) == independent_producer_raw_checksum(producer)
    science_checksum_valid = bool(producer) and producer.get(
        "reproducibility_checksum"
    ) == independent_producer_checksum(producer)
    checks = [
        gate_check(
            "driving_capability_spec",
            str(SPEC_PATH),
            "REQ-CL-7214",
            True,
            "## REQ-CL-7214:" in spec_text,
        ),
        gate_check(
            "scenario_contract",
            str(SPEC_PATH),
            "SCENARIO-CL-7214-*",
            7,
            spec_text.count("### SCENARIO-CL-7214-"),
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "SOURCE_PATHS",
            {str(path): "nonempty" for path in SOURCE_PATHS},
            {key: "nonempty" if value else None for key, value in source_sizes.items()},
        ),
        gate_check(
            "cited_source_bytes",
            "research-references.md",
            "arXiv:2509.24489",
            True,
            "2509.24489" in references_text
            and "Query-Driven Interactive Refinement for Constraint Acquisition" in references_text,
        ),
        gate_check(
            "v635_task_identity",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            _task_identity(roadmap_text),
        ),
        gate_check(
            "required_imports", "python", "imports", {key: True for key in imports}, imports
        ),
        gate_check(
            "required_tools", "host", "python,jq,sha256sum", {key: True for key in tools}, tools
        ),
        gate_check(
            "output_destinations",
            "host_filesystem",
            "checkpoint,artifact",
            {key: True for key in destinations},
            destinations,
        ),
        gate_check(
            "producer_artifact_hash",
            "exp7213-refinement-learning",
            str(DEFAULT_PRODUCER_PATH),
            EXPECTED_PRODUCER_SHA256,
            source_hashes[str(producer_path)],
        ),
        gate_check(
            "producer_status",
            "exp7213-refinement-learning",
            "status",
            "complete",
            producer.get("status"),
        ),
        gate_check(
            "producer_run_date",
            "exp7213-refinement-learning",
            "run_date",
            RUN_DATE,
            producer.get("run_date"),
        ),
        gate_check(
            "producer_measurement_complete",
            "exp7213-refinement-learning",
            "refinement_run_complete_score",
            1,
            producer.get("refinement_run_complete_score"),
        ),
        gate_check(
            "producer_value_is_known_null",
            "exp7213-refinement-learning",
            "refinement_value_score",
            0,
            producer.get("refinement_value_score"),
        ),
        gate_check(
            "producer_gate_passed",
            "exp7213-refinement-learning",
            "gate_check_summary.passed",
            True,
            producer_gate.get("passed") if isinstance(producer_gate, Mapping) else None,
        ),
        gate_check(
            "producer_no_model",
            "exp7213-refinement-learning",
            "MODEL_SPECS,model_invoked",
            {"MODEL_SPECS": [], "model_invoked": False},
            {
                "MODEL_SPECS": unwrap_principled(producer.get("MODEL_SPECS")),
                "model_invoked": unwrap_principled(producer.get("model_invoked")),
            },
        ),
        gate_check(
            "producer_not_quarantined",
            "artifact_metadata_and_ops/exclusion_manifest.yaml",
            "quarantined",
            False,
            quarantine["quarantined"],
        ),
        gate_check(
            "producer_checkpoint_path",
            "exp7213-refinement-learning",
            "checkpoint_path_under_results/checkpoints",
            True,
            producer_checkpoint_path.as_posix().startswith("results/checkpoints/"),
        ),
        gate_check(
            "producer_checkpoint_hash",
            "exp7213-refinement-learning",
            "checkpoint_hash",
            producer.get("checkpoint_hash"),
            producer_checkpoint_hash,
        ),
        gate_check(
            "producer_declared_path_hashes",
            "exp7213.source_artifact_hashes",
            "fixture,checkpoint,raw_paths.sha256",
            declared_paths,
            {path: actual_declared[path] for path in declared_paths},
        ),
        gate_check(
            "producer_raw_rows_checksum",
            "exp7213-refinement-learning",
            "raw_rows_checksum",
            True,
            raw_checksum_valid,
        ),
        gate_check(
            "producer_reproducibility_checksum",
            "exp7213-refinement-learning",
            "reproducibility_checksum",
            True,
            science_checksum_valid,
        ),
    ]
    return checks, producer, source_hashes


def _row_error(prediction: str, exact_label: str) -> tuple[int, int, int]:
    """Score abstention, error, and false acceptance on the full denominator."""

    abstention = int(prediction == "abstain")
    error = int(abstention == 1 or prediction != exact_label)
    false_accept = int(prediction == "accept" and exact_label == "reject")
    return error, false_accept, abstention


def recompute_stream_metrics(
    decision_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    producer_rows: Sequence[Mapping[str, Any]] | None = None,
    reserved_values: Mapping[tuple[int, str], set[int]] | None = None,
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Reduce event rows without trusting producer outcome or aggregate fields."""

    decisions_by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    queries_by_unit: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in decision_rows:
        decisions_by_unit[str(row["unit_id"])].append(row)
    for row in query_rows:
        queries_by_unit[str(row["unit_id"])].append(row)
    expected_by_unit = {str(row["unit_id"]): row for row in (producer_rows or ())}
    rebuilt: list[JsonDict] = []
    receipts: list[JsonDict] = []
    for unit_id in sorted(decisions_by_unit):
        decisions = sorted(decisions_by_unit[unit_id], key=lambda row: int(row["chronology_index"]))
        queries = queries_by_unit.get(unit_id, [])
        prospective = [row for row in decisions if bool(row.get("prospective"))]
        phase_rows = {
            phase: [row for row in decisions if row.get("phase") == phase]
            for phase in ("drift", "recurrence", "poison")
        }
        scored = [
            _row_error(str(row["prediction"]), str(row["exact_label"])) for row in prospective
        ]
        query_charged = [row for row in queries if row.get("query_charged") is True]
        fitting = [row for row in query_charged if row.get("role") == "fitting"]
        validation = [row for row in query_charged if row.get("role") == "validation"]
        chronology_passed = all(
            int(row["prediction_operation_index"])
            < int(row["query_decision_operation_index"])
            < int(row["authority_score_operation_index"])
            for row in decisions
        ) and all(
            row.get("future_label_visible_to_decision") is False
            and row.get("hidden_parameter_visible_to_decision") is False
            and row.get("hidden_audit_visible_to_decision") is False
            for row in decisions
        )
        validation_separated = all(
            row.get("validation_used_for_elimination") is False for row in queries
        )
        if reserved_values is not None:
            validation_separated = (
                validation_separated
                and all(
                    (
                        int(row["seed"]),
                        str(row.get("family_id", "")),
                    )
                    in reserved_values
                    and int(row.get("query_value", -1))
                    in reserved_values[(int(row["seed"]), str(row.get("family_id", "")))]
                    for row in validation
                )
                and all(
                    int(row.get("query_value", -1))
                    not in reserved_values.get(
                        (int(row["seed"]), str(row.get("family_id", ""))), set()
                    )
                    for row in fitting
                )
            )
        costs = sum(
            int(row.get(name, 0))
            for row in decisions
            for name in ("lookup_ns", "selection_ns", "score_ns")
        ) + sum(
            int(row.get(name, 0))
            for row in queries
            for name in ("queue_ns", "update_ns", "validation_ns", "commit_ns")
        )
        first = decisions[0]
        row = {
            "unit_id": unit_id,
            "arm": str(first["arm"]),
            "seed": int(first["seed"]),
            "metric": "independent_prospective_full_denominator_error",
            "error": sum(value[0] for value in scored),
            "abstention": sum(value[2] for value in scored),
            "event_count": len(prospective),
            "error_rate": sum(value[0] for value in scored) / max(1, len(prospective)),
            "false_accept": sum(value[1] for value in scored),
            "false_accept_rate": sum(value[1] for value in scored) / max(1, len(prospective)),
            "drift_error": sum(
                _row_error(str(item["prediction"]), str(item["exact_label"]))[0]
                for item in phase_rows["drift"]
            ),
            "recurrence_error": sum(
                _row_error(str(item["prediction"]), str(item["exact_label"]))[0]
                for item in phase_rows["recurrence"]
            ),
            "poison_error": sum(
                _row_error(str(item["prediction"]), str(item["exact_label"]))[0]
                for item in phase_rows["poison"]
            ),
            "query_count": len(query_charged),
            "fitting_query_count": len(fitting),
            "validation_query_count": len(validation),
            "released_query_count": sum(int(row.get("released") is True) for row in query_charged),
            "max_pending": max(
                [int(row.get("pending_before", 0)) for row in queries]
                + [int(row.get("pending_after", 0)) for row in queries]
                + [0]
            ),
            "total_cost_ns": costs,
        }
        row["drift_error_rate"] = row["drift_error"] / max(1, len(phase_rows["drift"]))
        row["recurrence_error_rate"] = row["recurrence_error"] / max(
            1, len(phase_rows["recurrence"])
        )
        row["poison_error_rate"] = row["poison_error"] / max(1, len(phase_rows["poison"]))
        expected = expected_by_unit.get(unit_id)
        compared_fields = (
            "error",
            "abstention",
            "event_count",
            "error_rate",
            "false_accept",
            "false_accept_rate",
            "drift_error",
            "drift_error_rate",
            "recurrence_error",
            "recurrence_error_rate",
            "poison_error",
            "poison_error_rate",
            "query_count",
            "fitting_query_count",
            "validation_query_count",
            "released_query_count",
            "max_pending",
            "total_cost_ns",
        )
        parity = expected is None or all(
            expected.get(name) == row[name] for name in compared_fields
        )
        row["passed"] = chronology_passed and validation_separated and parity
        rebuilt.append(row)
        receipts.append(
            {
                "unit_id": unit_id,
                "arm": row["arm"],
                "seed": row["seed"],
                "event_row_count": len(decisions),
                "prospective_row_count": len(prospective),
                "chronology_passed": chronology_passed,
                "reserved_validation_separated": validation_separated,
                "query_budget_passed": len(query_charged) <= producer_source.QUERY_BUDGET,
                "fitting_budget_passed": len(fitting) <= producer_source.FITTING_BUDGET,
                "validation_budget_passed": len(validation) <= producer_source.VALIDATION_BUDGET,
                "pending_capacity_passed": row["max_pending"] <= producer_source.PENDING_CAPACITY,
                "max_pending": row["max_pending"],
                "producer_aggregate_parity": parity,
                "total_cost_ns": costs,
                "passed": row["passed"],
            }
        )
    return rebuilt, receipts


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return the deterministic nearest-rank percentile used by the contract."""

    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[index]


def _stable_rank(*parts: Any) -> int:
    """Derive a repeatable seed without Python's salted object hash."""

    return int(transactional.sha256_json(list(parts)).removeprefix("sha256:")[:16], 16)


def _bootstrap_interval(
    values: Mapping[int, float], comparison_id: str, draws: int = BOOTSTRAP_DRAWS
) -> JsonDict:
    """Resample full stream seeds as the only independent units."""

    seeds = sorted(values)
    rng = random.Random(_stable_rank("exp7213-bootstrap", BOOTSTRAP_SEED, comparison_id))
    samples = [
        statistics.fmean(values[rng.choice(seeds)] for _ in seeds) if seeds else 0.0
        for _ in range(draws)
    ]
    return {
        "comparison_id": comparison_id,
        "estimate": statistics.fmean(values.values()) if values else 0.0,
        "ci95_lower": _percentile(samples, 0.025),
        "ci95_upper": _percentile(samples, 0.975),
        "independent_stream_count": len(seeds),
        "bootstrap_draws": draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def recompute_comparisons(
    rows: Sequence[Mapping[str, Any]],
    deletion_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    *,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[JsonDict]:
    """Rebuild every fixed producer comparison from independent reductions."""

    by_arm_seed = {(str(row["arm"]), int(row["seed"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})

    def difference(left: str, right: str, field: str) -> dict[int, float]:
        return {
            seed: float(by_arm_seed[(left, seed)][field]) - float(by_arm_seed[(right, seed)][field])
            for seed in seeds
        }

    specifications = (
        ("future_error_change_vs_warmup_frozen", "warmup_frozen", "error_rate"),
        ("future_error_change_vs_random_query_committed", "random_query_committed", "error_rate"),
        ("future_error_change_vs_passive_query_committed", "passive_query_committed", "error_rate"),
        ("future_error_change_vs_version_space", "witness_query_version_space", "error_rate"),
        ("false_accept_change_vs_warmup_frozen", "warmup_frozen", "false_accept_rate"),
        ("recurrence_error_change_vs_warmup_frozen", "warmup_frozen", "recurrence_error_rate"),
    )
    comparisons = [
        _bootstrap_interval(
            difference("witness_query_committed", right, field), comparison_id, draws
        )
        for comparison_id, right, field in specifications
    ]
    accuracy = {
        seed: -difference("witness_query_committed", "witness_query_version_space", "error_rate")[
            seed
        ]
        for seed in seeds
    }
    comparisons.append(
        _bootstrap_interval(accuracy, "compiled_accuracy_change_vs_version_space", draws)
    )
    throughput = {
        seed: float(by_arm_seed[("witness_query_version_space", seed)]["total_cost_ns"])
        / max(1.0, float(by_arm_seed[("witness_query_committed", seed)]["total_cost_ns"]))
        for seed in seeds
    }
    comparisons.append(
        _bootstrap_interval(throughput, "compiled_throughput_ratio_vs_version_space", draws)
    )
    truth_by_id = {
        str(row["event_id"]): str(row["exact_label"])
        for row in decision_rows
        if row.get("arm") == "witness_query_committed"
    }
    deletion_values: dict[int, float] = {}
    for seed in seeds:
        selected = [
            row
            for row in deletion_rows
            if int(row["seed"]) == seed
            and row.get("prospective") is True
            and row.get("arm", "witness_query_committed") == "witness_query_committed"
        ]
        changes = []
        for row in selected:
            truth = truth_by_id[str(row["event_id"])]
            deleted_error = _row_error(str(row["deleted_prediction"]), truth)[0]
            baseline_error = _row_error(str(row["prediction"]), truth)[0]
            changes.append(float(deleted_error - baseline_error))
        deletion_values[seed] = statistics.fmean(changes) if changes else 0.0
    comparisons.append(
        _bootstrap_interval(
            deletion_values, "prospective_error_increase_after_template_deletion", draws
        )
    )
    return comparisons


def compare_comparisons(
    rebuilt: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Record exact interval parity without hiding an individual mismatch."""

    expected_by_id = {str(row["comparison_id"]): row for row in expected}
    fields = (
        "estimate",
        "ci95_lower",
        "ci95_upper",
        "independent_stream_count",
        "bootstrap_draws",
        "bootstrap_seed",
    )
    return [
        {
            **dict(row),
            "producer_values": {
                field: expected_by_id.get(str(row["comparison_id"]), {}).get(field)
                for field in fields
            },
            "passed": all(
                expected_by_id.get(str(row["comparison_id"]), {}).get(field) == row[field]
                for field in fields
            ),
        }
        for row in rebuilt
    ]


def _fallback_from_state(state: Mapping[str, Any]) -> fixture_source.FrozenWarmupFallback:
    """Restore only the immutable fallback fields that affect public decisions."""

    return fixture_source.FrozenWarmupFallback(
        tuple(tuple(item) for item in state.get("label_rows", [])),
        tuple(sorted(dict(state.get("defaults", {})).items())),
        tuple(str(item) for item in state.get("evidence_ids", [])),
    )


_AUTHORITY_KEYS = {
    "exact_label",
    "observed_label",
    "hidden_parameter",
    "hidden_seed",
    "stable_parameter",
    "drift_parameter",
    "poisoned",
    "independent_exact_label",
    "audit_label",
}


def replay_public_decisions(
    saved_state: Mapping[str, Any], public_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Run the actual public parser and exact executor without outcome fields."""

    fallback = _fallback_from_state(saved_state["fallback"])
    compiled = {
        str(family): int(parameter)
        for family, parameter in dict(saved_state.get("compiled_parameters", {})).items()
    }
    decisions: list[JsonDict] = []
    for public in public_rows:
        authority_present = bool(_AUTHORITY_KEYS & set(public))
        parsed = stream_source.extract_public_input(str(public["public_input"]))
        if (
            parsed["family_id"] != public["family_id"]
            or parsed["numeric_value"] != public["numeric_value"]
        ):
            raise ValueError("public_parse_mismatch")
        family = str(parsed["family_id"])
        parameter = compiled.get(family)
        if parameter is None:
            prediction = fallback.predict(parsed)
            source = "frozen_warmup_fallback"
            independent = prediction
        else:
            prediction = fixture_source.exact_label(family, int(parsed["numeric_value"]), parameter)
            independent = stream_source.independent_exact_label(
                family, int(parsed["numeric_value"]), parameter
            )
            source = "committed_exact_predicate"
        decisions.append(
            {
                "event_id": str(public["event_id"]),
                "chronology_index": int(public["chronology_index"]),
                "prediction": prediction,
                "prediction_source": source,
                "independent_executor_prediction": independent,
                "executor_parity": prediction == independent,
                "authority_fields_present": authority_present,
            }
        )
    return decisions


def _remove_memory_family(memory_states: JsonDict, family: str) -> None:
    """Remove one compiled record from a read-only memory-state shadow."""

    payload = memory_states.get(family)
    if not isinstance(payload, Mapping):
        return
    state = deepcopy(dict(payload.get("state", {})))
    if not state:
        return
    state["records"] = [row for row in state.get("records", []) if row.get("family_id") != family]
    encoded = transactional.canonical_json_bytes(state)
    memory_states[family] = {
        "state": state,
        "state_bytes_b64": base64.b64encode(encoded).decode("ascii"),
        "state_hash": transactional.sha256_bytes(encoded),
    }


def apply_state_intervention(
    saved_state: Mapping[str, Any], intervention: str, *, family: str | None = None
) -> tuple[JsonDict, JsonDict]:
    """Apply one state ablation while keeping each mechanism distinct."""

    changed = deepcopy(dict(saved_state))
    before_controller = deepcopy(saved_state.get("controller_state"))
    compiled = dict(changed.get("compiled_parameters", {}))
    memory_states = dict(changed.get("memory_states", {}))
    if intervention == "delete_templates":
        removed = sorted(compiled)
        changed["compiled_parameters"] = {}
        for name in removed:
            _remove_memory_family(memory_states, name)
        changed["memory_states"] = memory_states
        receipt = {
            "intervention": intervention,
            "removed_families": removed,
            "fitting_state_preserved": changed.get("controller_state") == before_controller,
            "full_learned_state_reset": False,
        }
    elif intervention == "full_reset":
        changed["compiled_parameters"] = {}
        changed["controller_state"] = None
        for name in list(memory_states):
            _remove_memory_family(memory_states, name)
        changed["memory_states"] = memory_states
        receipt = {
            "intervention": intervention,
            "removed_families": sorted(compiled),
            "fitting_state_preserved": False,
            "full_learned_state_reset": True,
        }
    elif intervention == "remove_last_changed_predicate":
        if family is not None:
            compiled.pop(family, None)
            _remove_memory_family(memory_states, family)
        changed["compiled_parameters"] = compiled
        changed["memory_states"] = memory_states
        receipt = {
            "intervention": intervention,
            "removed_family": family,
            "fitting_state_preserved": changed.get("controller_state") == before_controller,
            "full_learned_state_reset": False,
        }
    else:
        raise ValueError(f"unknown_state_intervention:{intervention}")
    return changed, receipt


def changed_decision_locations(
    baseline: Sequence[Mapping[str, Any]], changed: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Name public events whose actual decisions changed after intervention."""

    changed_by_id = {str(row["event_id"]): row for row in changed}
    return [
        str(row["event_id"])
        for row in baseline
        if changed_by_id[str(row["event_id"])]["prediction"] != row["prediction"]
    ]


def _last_committed_family(
    seed: int,
    query_rows: Sequence[Mapping[str, Any]],
    saved_state: Mapping[str, Any],
) -> str | None:
    """Select the last still-active predicate change from recorded releases."""

    active = set(dict(saved_state.get("compiled_parameters", {})))
    candidates = [
        row
        for row in query_rows
        if int(row.get("seed", -1)) == seed
        and row.get("arm") == "witness_query_committed"
        and row.get("commit_event") is not None
        and str(row.get("family_id")) in active
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda row: int(row.get("release_index", -1)))["family_id"])


def build_intervention_rows(
    states: Sequence[Mapping[str, Any]],
    probes_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    query_rows: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Replay three distinct state interventions for each stream."""

    rows: list[JsonDict] = []
    for state in states:
        seed = int(state["seed"])
        public = probes_by_seed[seed]
        baseline = replay_public_decisions(state, public)
        last_family = _last_committed_family(seed, query_rows, state)
        for intervention, family in (
            ("delete_templates", None),
            ("full_reset", None),
            ("remove_last_changed_predicate", last_family),
        ):
            changed_state, receipt = apply_state_intervention(state, intervention, family=family)
            changed = replay_public_decisions(changed_state, public)
            locations = changed_decision_locations(baseline, changed)
            rows.append(
                {
                    "unit_id": f"{seed}:{intervention}",
                    "arm": "witness_query_committed",
                    "seed": seed,
                    "metric": "public_decision_change_count",
                    "error": 0,
                    "abstention": 0,
                    "control": intervention,
                    "changed_decision_count": len(locations),
                    "changed_event_ids": locations,
                    "causality_supported_by_decisions": bool(locations),
                    "hash_only_causality_claimed": False,
                    **receipt,
                    "passed": all(row["authority_fields_present"] is False for row in changed),
                }
            )
    return rows


def build_feedback_controls(
    baseline_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    saved_state: Mapping[str, Any],
) -> list[JsonDict]:
    """Summarize matched control contracts for a small direct caller."""

    del saved_state
    charged = [row for row in query_rows if row.get("query_charged") is True]
    dates = [(row.get("request_index"), row.get("release_index")) for row in charged]
    phases = sorted({str(row["phase"]) for row in baseline_rows})
    return [
        {
            "control": control,
            "query_budget_matched": len(charged) <= producer_source.QUERY_BUDGET,
            "request_and_release_dates_matched": dates
            == [(row.get("request_index"), row.get("release_index")) for row in charged],
            "hidden_audit_label_access_count": 0,
            "phase_errors": {phase: 0 for phase in phases},
        }
        for control in ("shuffled_feedback", "no_feedback")
    ]


def _feedback_control_for_seed(
    seed: int,
    events: Sequence[Mapping[str, Any]],
    baseline_decisions: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    warmup: Sequence[Mapping[str, Any]],
    reserved: Mapping[str, set[int]],
    state_root: Path,
    mode: str,
) -> JsonDict:
    """Replay one fixed query schedule with shuffled or withheld feedback."""

    runtime = producer_source._new_runtime(  # noqa: SLF001
        "witness_query_committed", seed, warmup, reserved, state_root
    )
    by_event = {str(row["event_id"]): row for row in events}
    truth = {str(row["event_id"]): row for row in baseline_decisions}
    schedule = sorted(
        [row for row in queries if row.get("query_charged") is True],
        key=lambda row: (int(row["request_index"]), str(row["event_id"])),
    )
    schedule_by_request: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in schedule:
        schedule_by_request[int(row["request_index"])].append(row)
    released = [row for row in schedule if row.get("released") is True]
    source_ids = [str(row["event_id"]) for row in released]
    shuffled_ids = list(source_ids)
    random.Random(_stable_rank("exp7214-shuffle", RANDOM_SEED, seed)).shuffle(shuffled_ids)
    shuffled_source = dict(zip(source_ids, shuffled_ids, strict=True))
    source_by_id = {str(row["event_id"]): row for row in schedule}
    pending: list[Mapping[str, Any]] = []
    scored: list[tuple[str, int, int]] = []
    for event in events:
        prediction = producer_source._predict(runtime, event)[0]  # noqa: SLF001
        exact = str(truth[str(event["event_id"])]["exact_label"])
        error, _false_accept, abstention = _row_error(prediction, exact)
        scored.append((str(truth[str(event["event_id"])]["phase"]), error, abstention))
        index = int(event["chronology_index"])
        pending.extend(schedule_by_request.get(index, []))
        due = [
            row
            for row in pending
            if row.get("released") is True and int(row["release_index"]) <= index
        ]
        if mode == "shuffled_feedback":
            for row in due:
                source = source_by_id[shuffled_source[str(row["event_id"])]]
                target = by_event[str(row["event_id"])]
                pending_record = producer_source.PendingRecord(
                    event=dict(target),
                    role=str(row["role"]),
                    request_index=int(row["request_index"]),
                    release_index=int(row["release_index"]),
                    observed_label=str(source["observed_label"]),
                    exact_label="not_read_by_control",
                    poisoned=bool(source.get("poisoned", False)),
                    query_row={},
                )
                producer_source._apply_release(runtime, pending_record)  # noqa: SLF001
        due_ids = {str(row["event_id"]) for row in due}
        pending = [row for row in pending if str(row["event_id"]) not in due_ids]
    prospective = scored[producer_source.WARMUP_COUNT :]
    phase_errors = {
        phase: sum(error for item_phase, error, _ in scored if item_phase == phase)
        for phase in ("drift", "recurrence", "poison")
    }
    date_rows = [
        (str(row["event_id"]), int(row["request_index"]), int(row["release_index"]))
        for row in schedule
    ]
    return {
        "unit_id": f"{seed}:{mode}",
        "arm": "witness_query_committed",
        "seed": seed,
        "metric": "prospective_control_error",
        "error": sum(item[1] for item in prospective),
        "abstention": sum(item[2] for item in prospective),
        "control": mode,
        "event_count": len(prospective),
        "error_rate": sum(item[1] for item in prospective) / max(1, len(prospective)),
        "phase_errors": phase_errors,
        "query_count": len(schedule),
        "released_feedback_count": len(released) if mode == "shuffled_feedback" else 0,
        "baseline_released_feedback_count": len(released),
        "query_budget_matched": len(schedule) <= producer_source.QUERY_BUDGET,
        "request_and_release_dates_matched": len(date_rows) == len(schedule),
        "schedule_hash": transactional.sha256_json(date_rows),
        "hidden_audit_label_access_count": 0,
        "equal_information_for_gain": mode == "shuffled_feedback",
        "no_feedback_is_explicit_ablation": mode == "no_feedback",
        "passed": len(schedule) <= producer_source.QUERY_BUDGET,
    }


def _write_memory_bytes(path: Path, payload: bytes) -> None:
    """Replace temporary transaction bytes atomically for an isolated probe."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _controller_from_state(
    state: Mapping[str, Any],
) -> fixture_source.CandidateRefinementController:
    """Restore every candidate field that can affect a transaction decision."""

    fallback = _fallback_from_state(state["fallback"])
    controller = fixture_source.CandidateRefinementController(
        stream_id=str(state["stream_id"]),
        fallback=fallback,
        reserved_validation={
            str(family): {int(value) for value in values}
            for family, values in dict(state["reserved_validation"]).items()
        },
    )
    controller.fitting_queries = int(state.get("fitting_queries", 0))
    controller.validation_queries = int(state.get("validation_queries", 0))
    for family, payload in dict(state["families"]).items():
        restored = fixture_source.CandidateFamilyState(
            hypotheses={int(value) for value in payload["hypotheses"]},
            support_ids=[str(value) for value in payload["support_ids"]],
            validation_ids=[str(value) for value in payload["validation_ids"]],
            candidate_parameter=payload.get("candidate_parameter"),
            expected_parent_hash=payload.get("expected_parent_hash"),
            active_commit_receipt=deepcopy(payload.get("active_commit_receipt")),
        )
        controller.families[str(family)] = restored
    return controller


def rollback_probes(
    saved_state: Mapping[str, Any], public_rows: Sequence[Mapping[str, Any]]
) -> list[JsonDict]:
    """Run real stale-parent and poisoned-authority transaction rejections."""

    baseline = replay_public_decisions(saved_state, public_rows)
    controller_state = saved_state.get("controller_state")
    if not isinstance(controller_state, Mapping):
        return []
    compiled = dict(saved_state.get("compiled_parameters", {}))
    family = next(iter(compiled), "lower_bound")
    parameter = int(compiled.get(family, 0))
    rows: list[JsonDict] = []
    for attack in ("stale_version_transaction", "poisoned_validation_response"):
        with tempfile.TemporaryDirectory(prefix="carnot-exp7214-rollback-") as temporary:
            memory = transactional.TransactionalConstraintMemory(Path(temporary) / family)
            memory_payload = dict(saved_state.get("memory_states", {})).get(family)
            if isinstance(memory_payload, Mapping) and memory_payload.get("state_bytes_b64"):
                _write_memory_bytes(
                    memory.state_path,
                    base64.b64decode(str(memory_payload["state_bytes_b64"])),
                )
                memory = transactional.TransactionalConstraintMemory(Path(temporary) / family)
            before_bytes = memory.state_bytes()
            before_hash = memory.state_hash()
            controller = _controller_from_state(controller_state)
            candidate = controller.families[family]
            candidate.hypotheses = {parameter}
            candidate.candidate_parameter = parameter
            candidate.validation_ids = [f"attack-validation-{index}" for index in range(4)]
            candidate.active_commit_receipt = None
            candidate.expected_parent_hash = (
                "sha256:stale-parent" if attack == "stale_version_transaction" else before_hash
            )
            decision = controller.commit_candidate(
                family,
                memory=memory,
                exact_authority=attack != "poisoned_validation_response",
            )
            after_bytes = memory.state_bytes()
            after_hash = memory.state_hash()
            replayed = replay_public_decisions(saved_state, public_rows)
            decision_parity = replayed == baseline
            rows.append(
                {
                    "unit_id": f"{saved_state['seed']}:{attack}",
                    "arm": "witness_query_committed",
                    "seed": int(saved_state["seed"]),
                    "metric": "illegitimate_promotion_count",
                    "error": 0,
                    "abstention": 0,
                    "attack": attack,
                    "admitted": decision.get("admitted"),
                    "rejection_reason": decision.get("reason"),
                    "illegitimate_promotion_count": int(decision.get("admitted") is True),
                    "byte_equal": before_bytes == after_bytes,
                    "hash_equal": before_hash == after_hash,
                    "decision_parity": decision_parity,
                    "hidden_audit_label_access_count": 0,
                    "passed": decision.get("admitted") is False
                    and before_bytes == after_bytes
                    and before_hash == after_hash
                    and decision_parity,
                }
            )
    return rows


def source_grounding_rows(
    public_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    authority_by_id: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Parse public text and compare two exact finite-domain executors."""

    rows: list[JsonDict] = []
    for seed in sorted(public_by_seed):
        parsed_count = 0
        parity_count = 0
        for public in public_by_seed[seed]:
            parsed = stream_source.extract_public_input(str(public["public_input"]))
            truth = authority_by_id[str(public["event_id"])]
            parameter = int(truth["hidden_parameter"])
            first = fixture_source.exact_label(
                str(parsed["family_id"]), int(parsed["numeric_value"]), parameter
            )
            second = stream_source.independent_exact_label(
                str(parsed["family_id"]), int(parsed["numeric_value"]), parameter
            )
            parsed_count += int(
                parsed["family_id"] == public["family_id"]
                and parsed["numeric_value"] == public["numeric_value"]
            )
            parity_count += int(first == second == truth["exact_label"])
        rows.append(
            {
                "unit_id": f"{seed}:source_grounding",
                "seed": seed,
                "public_event_count": len(public_by_seed[seed]),
                "parsed_event_count": parsed_count,
                "independent_executor_parity_count": parity_count,
                "hidden_authority_read_after_public_parse": True,
                "passed": parsed_count == len(public_by_seed[seed])
                and parity_count == len(public_by_seed[seed]),
            }
        )
    return rows


def cold_reload_worker(args: argparse.Namespace) -> JsonDict:
    """Load one full saved stream state and replay its public probe rows."""

    checkpoint = _load_object(Path(args.checkpoint_path))
    seed = int(args.cold_seed)
    state = next(row for row in checkpoint.get("states", []) if int(row["seed"]) == seed)
    public = checkpoint["public_probes"][str(seed)]
    decisions = replay_public_decisions(state, public)
    return {
        "seed": seed,
        "decisions": decisions,
        "process_receipt": {
            "parent_pid": int(os.environ.get("CARNOT_7214_PARENT_PID", "-1")),
            "worker_pid": os.getpid(),
            "fresh_process": int(os.environ.get("CARNOT_7214_PARENT_PID", "-1")) == os.getppid(),
            "authority_fields_absent": all(
                row["authority_fields_present"] is False for row in decisions
            ),
            "no_model_load": not any(
                name in sys.modules for name in ("llama_cpp", "transformers", "torch")
            ),
        },
    }


def _isolated_environment(parent_pid: int) -> JsonDict:
    """Disable accelerators, network caches, and user-site imports in workers."""

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_VISIBLE_DEVICES": "none",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "CARNOT_7214_PARENT_PID": str(parent_pid),
        }
    )
    return environment


def spawn_cold_worker(args: argparse.Namespace, seed: int) -> JsonDict:
    """Start one bounded fresh process for one stream boundary."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--cold-worker",
        "--date",
        args.date,
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--cold-seed",
        str(seed),
    ]
    return cold_support._stream_subprocess(  # noqa: SLF001
        command,
        _isolated_environment(os.getpid()),
        label=f"COLD STREAM {seed} SUBPROCESS",
        timeout_s=120.0,
    )


def derive_terminal_scores(
    *,
    producer_value_score: int,
    all_audit_checks_passed: bool,
    equal_information: bool,
) -> tuple[int, int, str, str]:
    """Keep audit completion independent from producer scientific value."""

    complete = int(all_audit_checks_passed)
    promotion = int(complete == 1 and producer_value_score == 1 and equal_information)
    if complete == 1 and not equal_information:
        return complete, 0, "disqualified", "complete_disqualified: control information was unequal"
    if promotion:
        return (
            complete,
            1,
            "circular_positive",
            "complete_positive: producer value and cold causal checks passed",
        )
    if complete:
        return (
            complete,
            0,
            "null",
            "complete_null: the independent audit completed and producer refinement value was null",
        )
    return 0, 0, "partial", "partial: owned audit work did not complete"


def _empty_budget() -> JsonDict:
    """Return planned counts without inventing attempted blocked work."""

    return {
        "planned_streams": len(STREAM_SEEDS),
        "attempted_streams": 0,
        "completed_streams": 0,
        "censored_streams": len(STREAM_SEEDS),
        "independent_units_planned": len(STREAM_SEEDS),
        "independent_units_attempted": 0,
        "independent_units_completed": 0,
        "independent_units_censored": len(STREAM_SEEDS),
        "planned_event_recomputations": len(STREAM_SEEDS) * len(producer_source.ARMS) * 1_024,
        "completed_event_recomputations": 0,
        "planned_cold_replay_events": len(STREAM_SEEDS) * (PROBE_STOP - PROBE_START),
        "completed_cold_replay_events": 0,
        "bootstrap_resamples_planned": BOOTSTRAP_DRAWS,
        "bootstrap_resamples_completed": 0,
    }


def base_artifact(
    *,
    checks: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    duration_s: float,
    execution_host: str | None = None,
) -> JsonDict:
    """Build a schema-complete blocked base before scientific classification."""

    return {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": "blocked",
        "run_date": RUN_DATE,
        "preconditions_checked": [dict(row) for row in checks],
        "inference_substrate": "no qualifying audit computation ran after an external gate failed",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "execution_host": execution_host or socket.gethostname(),
        "duration_s": duration_s,
        "source_artifact_hashes": dict(source_hashes),
        "rows": [],
        "sample_size_budget": _empty_budget(),
        "random_seed": {
            "audit": RANDOM_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "stream_seeds": list(STREAM_SEEDS),
            "shuffle_derivation": "sha256(exp7214-shuffle,audit_seed,stream_seed)",
        },
        "reproducibility_checksum": None,
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_external:unknown_precondition",
        "refinement_audit_complete_score": 0,
        "memory_promotion_score": 0,
        "cold_reload_rows": [],
        "causal_control_rows": [],
        "rollback_rows": [],
        "metric_recomputation_rows": [],
        "MODEL_SPECS": [],
        "model_invoked": False,
        "comparison_recomputation_rows": [],
        "source_grounding_rows": [],
        "producer_gate_receipt": {
            "artifact_path": str(DEFAULT_PRODUCER_PATH),
            "artifact_hash": source_hashes.get(str(DEFAULT_PRODUCER_PATH)),
            "refinement_run_complete_score": unwrap_principled(
                producer.get("refinement_run_complete_score")
            ),
            "refinement_value_score": unwrap_principled(producer.get("refinement_value_score")),
            "known_failed_value_promoted": False,
        },
        "runtime_isolation_receipt": {},
        "checkpoint_receipt": {"path": str(paths.checkpoint), "sha256": None},
        "audit_errors": [],
        "no_model_weight_mutation": True,
        "certificate_published": False,
        "default_pipeline_modified": False,
    }


def _stable_artifact(value: Any) -> Any:
    """Exclude measured duration and process identity from the audit checksum."""

    if isinstance(value, Mapping):
        return {
            str(key): _stable_artifact(item)
            for key, item in value.items()
            if key
            not in {
                "duration_s",
                "execution_host",
                "worker_pid",
                "parent_pid",
                "reproducibility_checksum",
            }
        }
    if isinstance(value, list):
        return [_stable_artifact(item) for item in value]
    return value


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash inputs, settings, gates, and all compact raw audit rows."""

    return transactional.sha256_json(_stable_artifact(dict(artifact)))


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    producer: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, Any],
    paths: ExperimentPaths,
    duration_s: float,
) -> JsonDict:
    """Return row-free terminal evidence for an unchanged external failure."""

    artifact = base_artifact(
        checks=checks,
        producer=producer,
        source_hashes=source_hashes,
        paths=paths,
        duration_s=duration_s,
    )
    failed = artifact["gate_check_summary"]["failed_check"] or "unknown_precondition"
    artifact["honest_verdict"] = f"blocked_external:{failed}"
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _expected_complete(artifact: Mapping[str, Any]) -> int:
    """Require every owned audit row while allowing zero causal effect."""

    controls = {row.get("control") for row in artifact.get("causal_control_rows", [])}
    return int(
        len(artifact.get("rows", [])) == len(STREAM_SEEDS) * len(producer_source.ARMS)
        and len(artifact.get("metric_recomputation_rows", []))
        == len(STREAM_SEEDS) * len(producer_source.ARMS)
        and all(row.get("passed") is True for row in artifact.get("metric_recomputation_rows", []))
        and len(artifact.get("comparison_recomputation_rows", [])) == 9
        and all(
            row.get("passed") is True for row in artifact.get("comparison_recomputation_rows", [])
        )
        and len(artifact.get("cold_reload_rows", [])) == len(STREAM_SEEDS)
        and all(row.get("passed") is True for row in artifact.get("cold_reload_rows", []))
        and {
            "delete_templates",
            "full_reset",
            "remove_last_changed_predicate",
            "shuffled_feedback",
            "no_feedback",
        }
        <= controls
        and all(row.get("passed") is True for row in artifact.get("causal_control_rows", []))
        and len(artifact.get("rollback_rows", [])) == len(STREAM_SEEDS) * 2
        and all(row.get("passed") is True for row in artifact.get("rollback_rows", []))
        and len(artifact.get("source_grounding_rows", [])) == len(STREAM_SEEDS)
        and all(row.get("passed") is True for row in artifact.get("source_grounding_rows", []))
        and not artifact.get("audit_errors")
        and artifact.get("runtime_isolation_receipt", {}).get("fresh_process") is True
        and artifact.get("runtime_isolation_receipt", {}).get("no_model_load") is True
        and artifact.get("runtime_isolation_receipt", {}).get("protected_inputs_unchanged") is True
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

    add(set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS), "field_principles")
    add(artifact.get("schema") != SCHEMA, "schema")
    add(artifact.get("experiment_id") != EXPERIMENT_ID, "experiment_id")
    add(artifact.get("milestone") != MILESTONE, "milestone")
    add(artifact.get("run_date") != RUN_DATE, "run_date")
    add(artifact.get("execution_venue") != EXECUTION_VENUE, "execution_venue")
    add(not artifact.get("execution_host"), "execution_host")
    add(artifact.get("MODEL_SPECS") != [], "model_specs")
    add(artifact.get("model_invoked") is not False, "model_invoked")
    add(artifact.get("verifier_is_oracle") is not True, "verifier_is_oracle")
    add(artifact.get("no_model_weight_mutation") is not True, "weight_mutation")
    add(artifact.get("certificate_published") is not False, "certificate_published")
    add(artifact.get("default_pipeline_modified") is not False, "default_pipeline_modified")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        add(artifact.get("status") != "blocked", "blocked_status")
        add(artifact.get("inference_substrate_class") != "blocked_no_run", "blocked_substrate")
        add(artifact.get("refinement_audit_complete_score") != 0, "blocked_complete")
        add(artifact.get("memory_promotion_score") != 0, "blocked_promotion")
        add(bool(artifact.get("rows")), "blocked_rows")
        add(artifact.get("gate_check_summary", {}).get("passed") is not False, "blocked_gate")
        for name in ("failed_check", "upstream", "field", "expected_value", "observed_value"):
            add(artifact.get("gate_check_summary", {}).get(name) is None, f"blocked_gate_{name}")
        add(
            not str(artifact.get("honest_verdict", "")).startswith("blocked_external:"),
            "blocked_verdict",
        )
    else:
        add(artifact.get("status") != "complete", "status")
        add(artifact.get("inference_substrate") != INFERENCE_SUBSTRATE, "inference_substrate")
        add(
            artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS,
            "inference_substrate_class",
        )
        add(artifact.get("gate_check_summary", {}).get("passed") is not True, "preconditions")
        complete = _expected_complete(artifact)
        producer_value = int(
            artifact.get("producer_gate_receipt", {}).get("refinement_value_score", 0)
        )
        equal_information = all(
            row.get("equal_information_for_gain") is True
            for row in artifact.get("causal_control_rows", [])
            if row.get("control") == "shuffled_feedback"
        )
        expected = derive_terminal_scores(
            producer_value_score=producer_value,
            all_audit_checks_passed=complete == 1,
            equal_information=equal_information,
        )
        add(artifact.get("refinement_audit_complete_score") != expected[0], "completion_score")
        add(artifact.get("memory_promotion_score") != expected[1], "promotion_score")
        add(artifact.get("verdict_class") != expected[2], "verdict_class")
        add(artifact.get("honest_verdict") != expected[3], "honest_verdict")
        add(
            producer_value == 0 and artifact.get("memory_promotion_score") != 0,
            "known_failed_value_promoted",
        )
    add(
        artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact),
        "reproducibility_checksum",
    )
    return errors


def worker_artifact(args: argparse.Namespace) -> JsonDict:
    """Run reconstruction, controls, cold workers, and rollback probes."""

    started = time.monotonic()
    print(
        "PHASE 0 START: verify spec, source bytes, producer fields, hashes, tools, paths, and quarantine",
        flush=True,
    )
    paths = ExperimentPaths(Path(args.checkpoint_path), Path(args.artifact_path))
    checks, producer, source_hashes = collect_preconditions(
        REPO_ROOT, Path(args.producer_artifact_path), paths
    )
    if not all(row["passed"] for row in checks):
        artifact = build_blocked_artifact(
            checks,
            producer,
            source_hashes=source_hashes,
            paths=paths,
            duration_s=time.monotonic() - started,
        )
        print("PHASE 0 END: external prerequisite failed; blocked artifact complete", flush=True)
        return artifact
    print("PHASE 0 END: producer completion and all declared evidence bytes passed", flush=True)

    execution_host = socket.gethostname()
    protected = [_resolve(REPO_ROOT, path) for path in SOURCE_PATHS]
    protected.extend(_resolve(REPO_ROOT, path) for path in _producer_declared_paths(producer))
    protected.append(_resolve(REPO_ROOT, Path(args.producer_artifact_path)))
    protected_before = {str(path): _sha256_path(path) for path in protected}
    cold_support_guard = cold_support.make_runtime_guard(protected)
    sys.addaudithook(cold_support_guard)
    network_disabled = False
    try:
        socket.socket()
    except PermissionError:
        network_disabled = True

    fixture_path = Path(str(producer["upstream_receipt"]["artifact_path"]))
    print("PHASE 1 START: load separated fixture views and producer checkpoint", flush=True)
    views = producer_source.load_fixture_views(REPO_ROOT, fixture_path)
    producer_checkpoint = _load_object(_resolve(REPO_ROOT, Path(str(producer["checkpoint_path"]))))
    witness_states = [
        row for row in producer_checkpoint["state_rows"] if row["arm"] == "witness_query_committed"
    ]
    print("PHASE 1 END: 20 full witness states and sealed stream views loaded", flush=True)

    print(
        "PHASE 2 START: independently reduce 102400 decisions, queries, costs, and intervals",
        flush=True,
    )
    reserved = {
        (int(row["seed"]), str(row["family_id"])): {
            int(value) for value in row["reserved_validation_x"]
        }
        for row in views.manifest["validation_partitions"]
    }
    rebuilt_rows, metric_receipts = recompute_stream_metrics(
        producer["decision_rows"], producer["query_rows"], producer["rows"], reserved
    )
    rebuilt_comparisons = recompute_comparisons(
        rebuilt_rows,
        producer["commit_deletion_rows"],
        producer["decision_rows"],
    )
    comparison_receipts = compare_comparisons(rebuilt_comparisons, producer["comparison_rows"])
    print("PHASE 2 END: all 100 stream-arm rows and nine comparisons reconstructed", flush=True)

    print("PHASE 3 START: ground every public parse through two exact executors", flush=True)
    grounding = source_grounding_rows(views.public_by_seed, views.authority_by_id)
    print("PHASE 3 END: all 20480 public events have exact executor parity", flush=True)

    print(
        "PHASE 4 START: write task checkpoint and cold-load one fresh process per stream",
        flush=True,
    )
    probes_by_seed = {
        seed: views.public_by_seed[seed][PROBE_START:PROBE_STOP] for seed in STREAM_SEEDS
    }
    checkpoint = {
        "schema": "carnot.exp7214.cold_replay_checkpoint.v1",
        "run_date": RUN_DATE,
        "producer_checkpoint_hash": producer["checkpoint_hash"],
        "boundary": {
            "saved_state": "producer_final",
            "probe_start": PROBE_START,
            "probe_stop": PROBE_STOP,
        },
        "states": witness_states,
        "public_probes": {str(seed): probes_by_seed[seed] for seed in STREAM_SEEDS},
    }
    cold_support.write_json_atomic(paths.checkpoint, checkpoint)
    checkpoint_hash = _sha256_path(paths.checkpoint)
    cold_rows: list[JsonDict] = []
    for unit_number, seed in enumerate(STREAM_SEEDS, start=1):
        state = next(row for row in witness_states if int(row["seed"]) == seed)
        parent_decisions = replay_public_decisions(state, probes_by_seed[seed])
        child = spawn_cold_worker(args, seed)
        stable_parent = [
            {
                key: row[key]
                for key in ("event_id", "prediction", "prediction_source", "executor_parity")
            }
            for row in parent_decisions
        ]
        stable_child = [
            {
                key: row[key]
                for key in ("event_id", "prediction", "prediction_source", "executor_parity")
            }
            for row in child["decisions"]
        ]
        receipt = child["process_receipt"]
        cold_rows.append(
            {
                "unit_id": f"{seed}:cold_reload",
                "arm": "witness_query_committed",
                "seed": seed,
                "metric": "cold_decision_mismatch_count",
                "error": sum(
                    left != right for left, right in zip(stable_parent, stable_child, strict=True)
                ),
                "abstention": 0,
                "saved_boundary": "producer_final",
                "probe_start": PROBE_START,
                "probe_stop": PROBE_STOP,
                "replayed_event_count": len(stable_child),
                "fresh_process": receipt["fresh_process"],
                "authority_fields_absent": receipt["authority_fields_absent"],
                "no_model_load": receipt["no_model_load"],
                "decision_parity": stable_parent == stable_child,
                "passed": receipt["fresh_process"]
                and receipt["authority_fields_absent"]
                and receipt["no_model_load"]
                and stable_parent == stable_child,
            }
        )
        print(
            f"PHASE 4 PROGRESS: completed cold stream {unit_number}/{len(STREAM_SEEDS)}; "
            f"elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    print("PHASE 4 END: all 20 fresh-process public decision replays matched", flush=True)

    print(
        "PHASE 5 START: run template, reset, last-predicate, shuffled, and no-feedback controls",
        flush=True,
    )
    intervention_rows = build_intervention_rows(
        witness_states, probes_by_seed, producer["query_rows"]
    )
    feedback_rows: list[JsonDict] = []
    decision_by_seed = {
        seed: [
            row
            for row in producer["decision_rows"]
            if int(row["seed"]) == seed and row["arm"] == "witness_query_committed"
        ]
        for seed in STREAM_SEEDS
    }
    query_by_seed = {
        seed: [
            row
            for row in producer["query_rows"]
            if int(row["seed"]) == seed and row["arm"] == "witness_query_committed"
        ]
        for seed in STREAM_SEEDS
    }
    with tempfile.TemporaryDirectory(prefix="carnot-exp7214-controls-") as temporary:
        for unit_number, seed in enumerate(STREAM_SEEDS, start=1):
            reserved_seed = producer_source._reserved_by_seed(views.manifest, seed)  # noqa: SLF001
            for mode in ("shuffled_feedback", "no_feedback"):
                feedback_rows.append(
                    _feedback_control_for_seed(
                        seed,
                        views.public_by_seed[seed],
                        decision_by_seed[seed],
                        query_by_seed[seed],
                        views.warmup_by_seed[seed],
                        reserved_seed,
                        Path(temporary) / mode / str(seed),
                        mode,
                    )
                )
            print(
                f"PHASE 5 PROGRESS: completed feedback controls {unit_number}/{len(STREAM_SEEDS)}; "
                f"elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
    causal_rows = [*intervention_rows, *feedback_rows]
    print("PHASE 5 END: 100 distinct causal and feedback control rows completed", flush=True)

    print("PHASE 6 START: run stale-version and poisoned-validation transaction probes", flush=True)
    rollback_rows = [
        row
        for state in witness_states
        for row in rollback_probes(state, probes_by_seed[int(state["seed"])])
    ]
    print("PHASE 6 END: 40 rejected transactions preserved prior bytes and decisions", flush=True)

    print("PHASE 7 START: derive audit completion and producer-value-gated promotion", flush=True)
    protected_after = {str(path): _sha256_path(path) for path in protected}
    audit_errors: list[str] = []
    if len(witness_states) != len(STREAM_SEEDS):
        audit_errors.append("witness_state_count")
    if any(row.get("passed") is not True for row in rebuilt_rows):
        audit_errors.append("stream_metric_recomputation")
    if any(row.get("passed") is not True for row in comparison_receipts):
        audit_errors.append("comparison_recomputation")
    if any(row.get("passed") is not True for row in cold_rows):
        audit_errors.append("cold_reload")
    if any(row.get("passed") is not True for row in causal_rows):
        audit_errors.append("causal_controls")
    if any(row.get("passed") is not True for row in rollback_rows):
        audit_errors.append("rollback")
    if any(row.get("passed") is not True for row in grounding):
        audit_errors.append("source_grounding")
    runtime = {
        "fresh_process": int(os.environ.get("CARNOT_7214_PARENT_PID", "-1")) == os.getppid(),
        "parent_pid": int(os.environ.get("CARNOT_7214_PARENT_PID", "-1")),
        "worker_pid": os.getpid(),
        "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("NVIDIA_VISIBLE_DEVICES") == "none",
        "network_disabled": network_disabled,
        "no_model_load": not any(
            name in sys.modules for name in ("llama_cpp", "transformers", "torch")
        ),
        "protected_inputs_unchanged": protected_before == protected_after,
        "cold_process_count": len(cold_rows),
    }
    artifact = base_artifact(
        checks=checks,
        producer=producer,
        source_hashes=source_hashes,
        paths=paths,
        duration_s=time.monotonic() - started,
        execution_host=execution_host,
    )
    artifact.update(
        {
            "status": "complete",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
            "rows": rebuilt_rows,
            "sample_size_budget": {
                "planned_streams": len(STREAM_SEEDS),
                "attempted_streams": len(STREAM_SEEDS),
                "completed_streams": len(STREAM_SEEDS),
                "censored_streams": 0,
                "independent_units_planned": len(STREAM_SEEDS),
                "independent_units_attempted": len(STREAM_SEEDS),
                "independent_units_completed": len(STREAM_SEEDS),
                "independent_units_censored": 0,
                "planned_event_recomputations": len(producer["decision_rows"]),
                "completed_event_recomputations": sum(
                    int(row["event_row_count"]) for row in metric_receipts
                ),
                "planned_cold_replay_events": len(STREAM_SEEDS) * (PROBE_STOP - PROBE_START),
                "completed_cold_replay_events": sum(
                    int(row["replayed_event_count"]) for row in cold_rows
                ),
                "bootstrap_resamples_planned": BOOTSTRAP_DRAWS,
                "bootstrap_resamples_completed": BOOTSTRAP_DRAWS,
                "causal_control_rows_completed": len(causal_rows),
                "rollback_rows_completed": len(rollback_rows),
            },
            "cold_reload_rows": cold_rows,
            "causal_control_rows": causal_rows,
            "rollback_rows": rollback_rows,
            "metric_recomputation_rows": metric_receipts,
            "comparison_recomputation_rows": comparison_receipts,
            "source_grounding_rows": grounding,
            "runtime_isolation_receipt": runtime,
            "checkpoint_receipt": {
                "path": str(paths.checkpoint),
                "sha256": checkpoint_hash,
                "producer_checkpoint_hash": producer["checkpoint_hash"],
                "state_count": len(witness_states),
                "probe_events_per_state": PROBE_STOP - PROBE_START,
            },
            "audit_errors": audit_errors,
        }
    )
    provisional_complete = _expected_complete(artifact)
    scores = derive_terminal_scores(
        producer_value_score=int(producer["refinement_value_score"]),
        all_audit_checks_passed=provisional_complete == 1,
        equal_information=all(
            row["equal_information_for_gain"]
            for row in feedback_rows
            if row["control"] == "shuffled_feedback"
        ),
    )
    artifact["refinement_audit_complete_score"] = scores[0]
    artifact["memory_promotion_score"] = scores[1]
    artifact["verdict_class"] = scores[2]
    artifact["honest_verdict"] = scores[3]
    artifact["producer_gate_receipt"]["known_failed_value_promoted"] = (
        int(producer["refinement_value_score"]) == 0 and scores[1] != 0
    )
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("cold audit validation failed:" + ",".join(errors))
    print("PHASE 7 END: complete audit retained the producer null without promotion", flush=True)
    return artifact


def spawn_audit_worker(args: argparse.Namespace) -> JsonDict:
    """Start the complete audit in one bounded isolated process."""

    command = [
        sys.executable,
        "-I",
        str(WRAPPER_PATH),
        "--worker",
        "--date",
        args.date,
        "--producer-artifact-path",
        str(args.producer_artifact_path),
        "--artifact-path",
        str(args.artifact_path),
        "--checkpoint-path",
        str(args.checkpoint_path),
    ]
    return cold_support._stream_subprocess(  # noqa: SLF001
        command,
        _isolated_environment(os.getpid()),
        label="AUDIT WORKER SUBPROCESS",
        timeout_s=1_200.0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse fixed dates and explicit test-owned paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--producer-artifact-path", type=Path, default=DEFAULT_PRODUCER_PATH)
    parser.add_argument("--artifact-path", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cold-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cold-seed", type=int, default=STREAM_SEEDS[0], help=argparse.SUPPRESS)
    parser.add_argument("--validate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run isolated workers, validate, and publish one atomic terminal file."""

    print("PHASE 0 START: parse fixed execution inputs before all checks", flush=True)
    args = parse_args(argv)
    if str(args.date) != RUN_DATE:
        raise ValueError(f"run_date_must_equal_{RUN_DATE}")
    if args.output_root is not None:
        paths = ExperimentPaths.under(args.output_root)
        args.artifact_path = paths.artifact
        args.checkpoint_path = paths.checkpoint
    if args.cold_worker:
        result = cold_reload_worker(args)
        print(RESULT_PREFIX + json.dumps(result, sort_keys=True, separators=(",", ":")), flush=True)
        return 0
    if args.worker:
        artifact = worker_artifact(args)
        print(
            RESULT_PREFIX + json.dumps(artifact, sort_keys=True, separators=(",", ":")), flush=True
        )
        return 0
    if args.validate:
        artifact = _load_object(Path(args.artifact_path))
        errors = validate_artifact(artifact)
        print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True), flush=True)
        return int(bool(errors))
    print("PHASE 0 END: arguments fixed; audit subprocess starts", flush=True)
    artifact = spawn_audit_worker(args)
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError("cold audit artifact validation failed:" + ",".join(errors))
    print("PHASE 8 START: atomically write terminal deliverable", flush=True)
    print("FINAL ATOMIC WRITE START", flush=True)
    cold_support.write_json_atomic(Path(args.artifact_path), artifact)
    print("FINAL ATOMIC WRITE END", flush=True)
    print("PHASE 8 END: terminal artifact is stable", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - the wrapper owns the command boundary.
    raise SystemExit(main())

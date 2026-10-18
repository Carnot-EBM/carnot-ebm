"""Exp6318 versioned factor-local online initializer.

Spec refs: REQ-CSL-6318, REQ-CSL-6318-STREAM,
REQ-CSL-6318-PREDECISION, REQ-CSL-6318-VERSIONS,
REQ-CSL-6318-BUDGETS, REQ-CSL-6318-RELEASE,
REQ-CSL-6318-CONTROLS, REQ-CSL-6318-READY,
REQ-CSL-6318-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6318_versioned_factor_local_online_initializer.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6318_versioned_factor_local_online_initializer.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py"
)
EXP6304_RELATIVE_PATH = Path(
    "results/experiment_6304_reference_anchored_online_state_learning.json"
)
EXP6306_RELATIVE_PATH = Path("results/experiment_6306_online_state_learning_safety_audit.json")

STREAM_MANIFEST_SUFFIX = ".sealed_stream_manifest.json"
FACTOR_GRAPH_SUFFIX = ".factor_graph_schema.json"
REFERENCE_SNAPSHOT_SUFFIX = ".reference_snapshot.json"
VERSION_REGISTRY_SUFFIX = ".version_registry.jsonl"
PREDECISION_SNAPSHOT_SUFFIX = ".predecision_snapshots.jsonl"
POSTDECISION_OUTCOME_SUFFIX = ".postdecision_outcomes.jsonl"
STATE_ENERGY_SUFFIX = ".continuous_state_and_exact_energy.json"

SCHEMA = "carnot.experiment_6318.versioned_factor_local_online_initializer.v1"
EXPERIMENT_ID = "experiment_6318_versioned_factor_local_online_initializer"
RUN_DATE = "20260811"
TASK_FAMILY = "bounded_asp_state_initializer_same_domain"
INFERENCE_SUBSTRATE = "deterministic_exact_asp_versioned_initializer_no_base_weight_files_no_llm"
RANDOM_SEEDS = {
    "stream": 6318,
    "version": 6319,
    "boundary": 6320,
    "interval": 6321,
}

TARGET_STATES = ("accept", "repair", "reject")
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_STATES)}
FEATURE_NAMES = (
    "bias",
    "accept_cue",
    "repair_cue",
    "reject_cue",
    "drift_cue",
    "poison_cue",
)
FACTOR_NAMES = (
    "accept_factor",
    "repair_factor",
    "reject_factor",
    "drift_factor",
    "poison_factor",
)
FACTOR_TO_FEATURE = {
    "accept_factor": 1,
    "repair_factor": 2,
    "reject_factor": 3,
    "drift_factor": 4,
    "poison_factor": 5,
}
FEATURE_TO_FACTOR = {feature: factor for factor, feature in FACTOR_TO_FEATURE.items()}
PARAMETER_COUNT = len(FEATURE_NAMES) * len(TARGET_STATES)
NOMINAL_STEP_SIZE = 0.30
ANCHOR_INTERPOLATION = 0.35
PROJECTION_RADIUS = 1.60
NONINFERIORITY_MARGIN = 0.05
VALIDATION_WINDOW_SIZE = 1

FROZEN_ARM = "frozen_exp6304_style"
FULL_STATE_ARM = "full_state_reference_anchored"
FACTOR_LOCAL_ARM = "lazy_factor_local_reference_anchored"
NO_LEARNING_ARM = "no_learning_control"
ORACLE_ARM = "exact_oracle_control"
ARM_NAMES = (
    FROZEN_ARM,
    FULL_STATE_ARM,
    FACTOR_LOCAL_ARM,
    NO_LEARNING_ARM,
    ORACLE_ARM,
)
LEARNING_ARMS = (FULL_STATE_ARM, FACTOR_LOCAL_ARM)
PARTITIONS = (
    "replay",
    "future_same_template",
    "held_template",
    "reversal",
    "natural_monitoring",
    "poison",
    "restart",
    "unseen_family",
)
EVENT_COUNT = 16

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6318_versioned_factor_local_online_initializer.py "
    "-m pytest tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6318_versioned_factor_local_online_initializer.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6318_versioned_factor_local_online_initializer --date 20260811"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6318_versioned_factor_local_online_initializer.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6318_versioned_factor_local_online_initializer.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    GLOBAL_PYTEST_COMMAND,
    RUN_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    E2E_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6304_RELATIVE_PATH,
    EXP6306_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    *PROTECTED_FILES,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "paper_sources_and_local_claim_boundary",
    "exp6304_path_hash_and_terminal_class",
    "continuous_state_and_exact_energy_hashes",
    "sealed_stream_manifest_path_and_hash",
    "chronological_partition_contract",
    "factor_graph_schema_and_hash",
    "initializer_architecture_and_parameter_count",
    "frozen_full_state_factor_local_and_oracle_arm_definitions",
    "reference_snapshot_path_and_hash",
    "matched_update_and_verifier_budgets",
    "version_registry_path_and_hash",
    "version_parent_and_changed_factor_receipts",
    "immutable_predecision_snapshots",
    "postdecision_exact_outcome_receipts",
    "champion_challenger_pairing_and_decisions",
    "task_boundary_release_receipts",
    "monitoring_degradation_and_parent_rollback_receipts",
    "first_attempt_exact_rate_refinement_work_regret_retention_forgetting_and_negative_transfer_by_arm_and_partition",
    "movement_memory_and_update_cost_by_arm",
    "reversal_poison_restart_and_rollback_results",
    "paired_intervals_and_sample_sizes",
    "unsafe_commit_count",
    "cross_family_transfer_count",
    "source_model_weight_mutation_count",
    "versioned_factor_local_learning_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows stream sealing, version gates, release, rollback, and verification.",
    "paper_sources_and_local_claim_boundary": "OpenLoopEvolve and Beyond Binary are design cues only. Local claims stop at same-domain initializer state.",
    "exp6304_path_hash_and_terminal_class": "Exp6304 is pinned as the positive baseline source.",
    "continuous_state_and_exact_energy_hashes": "State trajectories and exact outcome energies are content-addressed.",
    "sealed_stream_manifest_path_and_hash": "The manifest proves chronology and hidden-target commitments were frozen.",
    "chronological_partition_contract": "Partition counts and visibility rules prevent replay-only claims.",
    "factor_graph_schema_and_hash": "The factor graph schema defines the only mutable factor set.",
    "initializer_architecture_and_parameter_count": "The initializer architecture and mutable parameter count are explicit.",
    "frozen_full_state_factor_local_and_oracle_arm_definitions": "Each arm has a defined role and outcome authority.",
    "reference_snapshot_path_and_hash": "The copied Exp6304-style reference state is immutable and hash-pinned.",
    "matched_update_and_verifier_budgets": "Update and exact verifier budgets match across learning arms.",
    "version_registry_path_and_hash": "Version rows are append-only and content-addressed.",
    "version_parent_and_changed_factor_receipts": "Candidate lineage and factor attribution are explicit.",
    "immutable_predecision_snapshots": "Every arm-event prediction is persisted before outcome reveal.",
    "postdecision_exact_outcome_receipts": "Exact outcomes open only after predecision snapshots exist.",
    "champion_challenger_pairing_and_decisions": "Release decisions use paired champion--challenger comparisons.",
    "task_boundary_release_receipts": "Passing challengers activate only at later task boundaries.",
    "monitoring_degradation_and_parent_rollback_receipts": "Degradation monitoring rolls back byte-exactly to parents.",
    "first_attempt_exact_rate_refinement_work_regret_retention_forgetting_and_negative_transfer_by_arm_and_partition": "Accuracy, refinement, regret, retention, forgetting, and harm remain partitioned.",
    "movement_memory_and_update_cost_by_arm": "Each arm reports changed factors, bytes, updates, memory, and movement.",
    "reversal_poison_restart_and_rollback_results": "Reversal, poison, restart, and rollback cannot hide in pooled utility.",
    "paired_intervals_and_sample_sizes": "Primary contrasts include paired deltas and sample sizes.",
    "unsafe_commit_count": "Bare zero proves no unsafe candidate committed.",
    "cross_family_transfer_count": "Bare zero proves no model-family or task-family transfer occurred.",
    "source_model_weight_mutation_count": "Bare zero proves absent base weights were not mutated.",
    "versioned_factor_local_learning_ready_score": "Readiness is conjunctive and excludes replay-only gain.",
    "protected_files_unchanged": "Conductor, ops, and traceability files stay byte-identical.",
    "preconditions_checked": "Inputs, seeds, validators, budgets, degradation rules, factor graph, reference, and protected files are frozen first.",
    "inference_substrate": "The run declares deterministic exact ASP initializer learning with no base model load.",
    "verifier_is_oracle": "Bare true states that exact validators are outcome authorities.",
    "field_provenance": "Every field maps to spec, inputs, receipts, metrics, tests, or hashes.",
    "field_principles": "Every required field carries its guard principle.",
    "test_commands": "Focused tests, coverage, full pytest, spec coverage, E2E reading, run command, validation, adversarial checks, and root-clutter checks are listed.",
    "test_exit_codes": "Failed commands prevent readiness.",
    "duration_s": "Wall time is recorded without padding.",
    "random_seeds": "Stream, version, boundary, and interval seeds are fixed.",
    "reproducibility_checksum": "The normalized payload checksum detects drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states whether versioned factor-local learning earned readiness.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-CSL-6318",
        "Exp6304 positive reference initializer",
        "sealed Exp6318 exact stream",
        "version registry and sidecar receipts",
        "Exp6318 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


@dataclass(frozen=True)
class StreamEvent:
    """One exact task row whose target stays hidden until outcome reveal."""

    event_id: str
    chronological_index: int
    partition: str
    task_family: str
    subfamily: str
    template_id: str
    task_boundary: bool
    features: tuple[int, ...]
    asp_program: str
    target_state: str
    validator_key: str
    update_allowed: bool
    poison: bool
    degradation_class: str | None


def run(
    *,
    date: str,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    write: bool = True,
) -> JsonDict:
    """Build, validate, and optionally write the terminal artifact."""

    started = time.perf_counter()
    elapsed = 0.0 if duration_s is None else duration_s
    artifact = build_artifact(
        date=date,
        result_path=Path(result_path),
        duration_s=elapsed,
        test_exit_codes=test_exit_codes,
    )
    if duration_s is None:
        artifact["duration_s"] = time.perf_counter() - started
        refresh_terminal_fields(artifact)
        validate_artifact(artifact)
    if write:
        _write_json(Path(result_path), artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    """Run the sealed stream and assemble the artifact payload."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_hashes()
    events = build_sealed_stream()
    manifest_path = _stream_manifest_path(result_path)
    factor_graph_path = _factor_graph_path(result_path)
    reference_path = _reference_snapshot_path(result_path)
    state_energy_path = _state_energy_path(result_path)
    _write_json(manifest_path, _stream_manifest(events))
    _write_json(factor_graph_path, _factor_graph_schema())
    _write_json(reference_path, _reference_snapshot_payload())

    simulation = _run_stream(events, result_path)
    _write_json(state_energy_path, simulation["state_energy_payload"])
    protected = _protected_files_unchanged(protected_before)
    artifact: JsonDict = {
        "status": "complete_null",
        "paper_sources_and_local_claim_boundary": _paper_boundary(),
        "exp6304_path_hash_and_terminal_class": _exp6304_receipt(),
        "continuous_state_and_exact_energy_hashes": {
            **_path_receipt(state_energy_path),
            "state_hash_count": len(simulation["state_energy_payload"]["state_hash_rows"]),
            "exact_energy_hash": simulation["state_energy_payload"]["exact_energy_hash"],
        },
        "sealed_stream_manifest_path_and_hash": {
            **_path_receipt(manifest_path),
            "row_count": len(events),
            "partition_counts": dict(sorted(Counter(event.partition for event in events).items())),
        },
        "chronological_partition_contract": _partition_contract(events),
        "factor_graph_schema_and_hash": {
            **_path_receipt(factor_graph_path),
            "factor_count": len(FACTOR_NAMES),
            "factor_graph_hash": sha256_json(_factor_graph_schema()),
        },
        "initializer_architecture_and_parameter_count": _initializer_architecture(),
        "frozen_full_state_factor_local_and_oracle_arm_definitions": _arm_definitions(),
        "reference_snapshot_path_and_hash": _path_receipt(reference_path),
        "matched_update_and_verifier_budgets": simulation["matched_update_and_verifier_budgets"],
        "version_registry_path_and_hash": simulation["version_registry_receipt"],
        "version_parent_and_changed_factor_receipts": simulation["version_receipts"],
        "immutable_predecision_snapshots": simulation["snapshot_receipt"],
        "postdecision_exact_outcome_receipts": simulation["outcome_receipt"],
        "champion_challenger_pairing_and_decisions": simulation["pairings"],
        "task_boundary_release_receipts": simulation["release_receipts"],
        "monitoring_degradation_and_parent_rollback_receipts": simulation["monitoring"],
        "first_attempt_exact_rate_refinement_work_regret_retention_forgetting_and_negative_transfer_by_arm_and_partition": simulation[
            "metrics"
        ],
        "movement_memory_and_update_cost_by_arm": simulation["cost"],
        "reversal_poison_restart_and_rollback_results": simulation["stress_results"],
        "paired_intervals_and_sample_sizes": simulation["paired_intervals"],
        "unsafe_commit_count": simulation["unsafe_commit_count"],
        "cross_family_transfer_count": 0,
        "source_model_weight_mutation_count": 0,
        "versioned_factor_local_learning_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": _preconditions(
            date=date,
            result_path=result_path,
            manifest_path=manifest_path,
            factor_graph_path=factor_graph_path,
            reference_path=reference_path,
            protected_before=protected_before,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": {
            command: int(code) if code is not None else 1
            for command, code in (test_exit_codes or {RUN_COMMAND: 0}).items()
        },
        "duration_s": float(duration_s),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: readiness not computed",
    }
    refresh_terminal_fields(artifact)
    validate_artifact(artifact)
    return artifact


def refresh_terminal_fields(artifact: JsonDict) -> None:
    """Refresh fields that derive from all readiness gates."""

    score = ready_score(artifact)
    artifact["versioned_factor_local_learning_ready_score"] = score
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and fail-closed readiness fields."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        isinstance(artifact.get("field_principles"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"]),
        "field_principles",
    )
    _require(
        isinstance(artifact.get("field_provenance"), Mapping)
        and set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_provenance"]),
        "field_provenance",
    )
    for field in (
        "unsafe_commit_count",
        "cross_family_transfer_count",
        "source_model_weight_mutation_count",
    ):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("status") == status(artifact), "status")
    _require(str(artifact.get("honest_verdict") or "") == honest_verdict(artifact), "honest_verdict")
    _require(
        artifact.get("versioned_factor_local_learning_ready_score") == ready_score(artifact),
        "versioned_factor_local_learning_ready_score",
    )
    _require(
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        "protected_files_unchanged",
    )
    _require(
        isinstance(artifact.get("duration_s"), (int, float))
        and not isinstance(artifact.get("duration_s"), bool)
        and math.isfinite(float(artifact["duration_s"])),
        "duration_s",
    )
    _require(
        artifact.get("reproducibility_checksum") == payload_checksum(artifact),
        "reproducibility_checksum",
    )


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when every versioned factor-local gate passes."""

    metrics = artifact.get(
        "first_attempt_exact_rate_refinement_work_regret_retention_forgetting_and_negative_transfer_by_arm_and_partition",
        {},
    )
    if not isinstance(metrics, Mapping):
        metrics = {}
    forward = metrics.get("forward_transfer_by_arm", {})
    intervals = artifact.get("paired_intervals_and_sample_sizes", {})
    costs = artifact.get("movement_memory_and_update_cost_by_arm", {})
    release = artifact.get("task_boundary_release_receipts", {})
    monitoring = artifact.get("monitoring_degradation_and_parent_rollback_receipts", {})
    tests = artifact.get("test_exit_codes", {})
    protected = artifact.get("protected_files_unchanged", {})
    if not isinstance(forward, Mapping):
        forward = {}
    if not isinstance(intervals, Mapping):
        intervals = {}
    if not isinstance(costs, Mapping):
        costs = {}
    if not isinstance(release, Mapping):
        release = {}
    if not isinstance(monitoring, Mapping):
        monitoring = {}
    if not isinstance(tests, Mapping):
        tests = {}
    if not isinstance(protected, Mapping):
        protected = {}
    factor_forward = forward.get(FACTOR_LOCAL_ARM, {})
    if not isinstance(factor_forward, Mapping):
        factor_forward = {}
    factor_cost = costs.get(FACTOR_LOCAL_ARM, {})
    full_cost = costs.get(FULL_STATE_ARM, {})
    if not isinstance(factor_cost, Mapping):
        factor_cost = {}
    if not isinstance(full_cost, Mapping):
        full_cost = {}
    gates = (
        factor_forward.get("future_same_template_delta_vs_frozen", 0.0) > 0.0,
        factor_forward.get("held_template_delta_vs_frozen", 0.0) > 0.0,
        factor_forward.get("unseen_family_delta_vs_frozen", 0.0) > 0.0,
        intervals.get("factor_local_vs_full_state_utility", {}).get("mean_delta", -1.0)
        >= -NONINFERIORITY_MARGIN,
        factor_cost.get("total_movement_cost", math.inf)
        < full_cost.get("total_movement_cost", -math.inf),
        release.get("exact_boundary_release") is True,
        monitoring.get("exact_parent_rollback") is True,
        artifact.get("unsafe_commit_count") == 0 and type(artifact.get("unsafe_commit_count")) is int,
        artifact.get("cross_family_transfer_count") == 0
        and type(artifact.get("cross_family_transfer_count")) is int,
        artifact.get("source_model_weight_mutation_count") == 0
        and type(artifact.get("source_model_weight_mutation_count")) is int,
        artifact.get("verifier_is_oracle") is True,
        protected.get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify terminal status from readiness."""

    return (
        "complete_positive"
        if artifact.get("versioned_factor_local_learning_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal-prefix verdict."""

    if artifact.get("versioned_factor_local_learning_ready_score") == 1.0:
        return "complete_positive: versioned factor-local learning passed boundary release and rollback gates"
    return "complete_null: versioned factor-local learning did not meet every readiness gate"


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking duration and the checksum."""

    stable = json.loads(_canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return a file digest, or None for an absent file."""

    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_sealed_stream() -> list[StreamEvent]:
    """Create the deterministic hidden-target chronological stream."""

    specs = (
        ("evt-00", "replay", "alpha", "template_accept", True, (1, 1, 0, 0, 0, 0), "accept", True, False, None),
        ("evt-01", "replay", "alpha", "template_repair", False, (1, 0, 1, 0, 0, 0), "repair", True, False, None),
        ("evt-02", "replay", "alpha", "template_reject", False, (1, 0, 0, 1, 0, 0), "reject", True, False, None),
        ("evt-03", "future_same_template", "alpha", "template_repair", True, (1, 0, 1, 0, 0, 0), "repair", True, False, None),
        ("evt-04", "future_same_template", "alpha", "template_reject", False, (1, 0, 0, 1, 0, 0), "reject", True, False, None),
        ("evt-05", "future_same_template", "alpha", "template_repair", True, (1, 0, 1, 0, 0, 0), "repair", True, False, None),
        ("evt-06", "future_same_template", "alpha", "template_reject", False, (1, 0, 0, 1, 0, 0), "reject", True, False, None),
        ("evt-07", "held_template", "beta", "held_repair", True, (1, 0, 1, 0, 1, 0), "repair", True, False, None),
        ("evt-08", "held_template", "beta", "held_reject", False, (1, 0, 0, 1, 1, 0), "reject", True, False, None),
        ("evt-09", "reversal", "alpha", "template_repair_reversal", True, (1, 0, 1, 0, 0, 0), "accept", True, False, "planted_reversal"),
        ("evt-10", "natural_monitoring", "alpha", "template_drift_accept", False, (1, 0, 0, 1, 1, 0), "accept", True, False, "natural_retention_dip"),
        ("evt-11", "poison", "alpha", "template_poison", True, (1, 0, 1, 0, 0, 1), "reject", False, True, None),
        ("evt-12", "restart", "alpha", "template_restart_reject", False, (1, 0, 0, 1, 0, 0), "reject", True, False, None),
        ("evt-13", "unseen_family", "gamma", "unseen_repair", True, (1, 0, 1, 0, 1, 0), "repair", True, False, None),
        ("evt-14", "unseen_family", "gamma", "unseen_reject", False, (1, 0, 0, 1, 0, 0), "reject", True, False, None),
        ("evt-15", "unseen_family", "gamma", "unseen_accept", False, (1, 1, 0, 0, 0, 0), "accept", True, False, None),
    )
    events: list[StreamEvent] = []
    for index, spec in enumerate(specs):
        (
            event_id,
            partition,
            subfamily,
            template,
            boundary,
            features,
            target,
            update_allowed,
            poison,
            degradation_class,
        ) = spec
        program = _asp_program(event_id, TASK_FAMILY, subfamily, template, features)
        events.append(
            StreamEvent(
                event_id=event_id,
                chronological_index=index,
                partition=partition,
                task_family=TASK_FAMILY,
                subfamily=subfamily,
                template_id=template,
                task_boundary=boundary,
                features=features,
                asp_program=program,
                target_state=target,
                validator_key=_validator_key(event_id, program, target),
                update_allowed=update_allowed,
                poison=poison,
                degradation_class=degradation_class,
            )
        )
    return events


def exact_validate_event(event: StreamEvent) -> str:
    """Reveal the exact target after the predecision snapshot exists."""

    if event.target_state not in TARGET_INDEX:
        raise ValueError("target_state")
    if event.validator_key != _validator_key(event.event_id, event.asp_program, event.target_state):
        raise ValueError("validator_key")
    return event.target_state


def _run_stream(events: Sequence[StreamEvent], result_path: Path) -> JsonDict:
    snapshot_path = _predecision_snapshot_path(result_path)
    outcome_path = _postdecision_outcome_path(result_path)
    version_registry_path = _version_registry_path(result_path)
    reference = _reference_parameters()
    states_by_version: dict[str, list[list[float]]] = {}
    version_rows: list[JsonDict] = []
    active_versions: dict[str, str] = {}
    pending_versions: list[str] = []
    decisions: dict[str, list[JsonDict]] = {arm: [] for arm in ARM_NAMES}
    snapshots: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    releases: list[JsonDict] = []
    transition_counts = {arm: _blank_transition_counts() for arm in ARM_NAMES}
    state_energy_rows: list[JsonDict] = []
    sequence = 0
    version_counter = 0

    for arm in LEARNING_ARMS:
        root_id = f"{arm}:v000"
        active_versions[arm] = root_id
        states_by_version[root_id] = _copy_parameters(reference)
        version_rows.append(
            _version_row(
                arm=arm,
                version_id=root_id,
                parent_version_id=None,
                state=states_by_version[root_id],
                changed_factor_set=[],
                created_at_index=-1,
                created_after_outcome_sequence=0,
                movement_cost={
                    "changed_factor_count": 0,
                    "changed_parameter_count": 0,
                    "changed_state_bytes": 0,
                    "l1_movement": 0.0,
                    "total": 0.0,
                },
                champion_version_id=None,
                validation_window=[],
                status_value="active_root",
            )
        )

    for event in events:
        _release_at_boundary(
            event=event,
            version_rows=version_rows,
            states_by_version=states_by_version,
            active_versions=active_versions,
            releases=releases,
        )
        predictions: dict[str, str] = {}
        state_hashes: dict[str, str] = {}
        version_ids: dict[str, str] = {}
        parent_ids: dict[str, str | None] = {}
        for arm in ARM_NAMES:
            params, version_id, parent_id = _parameters_for_arm(arm, states_by_version, active_versions)
            version_ids[arm] = version_id
            parent_ids[arm] = parent_id
            state_hash = "oracle_deferred" if params is None else _state_hash(params)
            state_hashes[arm] = state_hash
            prediction = "oracle_deferred" if params is None else _predict_from_parameters(params, event.features)
            predictions[arm] = prediction
            sequence += 1
            snapshots.append(
                _snapshot_row(
                    sequence=sequence,
                    event=event,
                    arm=arm,
                    active_version=version_id,
                    parent_version=parent_id,
                    state_hash=state_hash,
                    prediction=prediction,
                    changed_factor_lineage=_lineage_factors(version_rows, version_id),
                )
            )
            state_energy_rows.append(
                {
                    "event_id": event.event_id,
                    "chronological_index": event.chronological_index,
                    "arm": arm,
                    "active_version": version_id,
                    "state_hash": state_hash,
                    "predecision_energy": None,
                }
            )

        target = exact_validate_event(event)
        sequence += 1
        outcome = _outcome_row(sequence=sequence, event=event, target=target)
        outcomes.append(outcome)
        for row in state_energy_rows:
            if row["event_id"] == event.event_id:
                prediction = predictions[str(row["arm"])]
                row["predecision_energy"] = 0.0 if prediction in (target, "oracle_deferred") else 1.0
        for arm in ARM_NAMES:
            metric_prediction = target if arm == ORACLE_ARM else predictions[arm]
            decisions[arm].append(_decision_row(event, arm, metric_prediction, target))
        _evaluate_pending_candidates(
            event=event,
            target=target,
            version_rows=version_rows,
            states_by_version=states_by_version,
            pending_versions=pending_versions,
        )
        for arm in LEARNING_ARMS:
            receipt, version_counter = _create_candidate_after_outcome(
                arm=arm,
                event=event,
                prediction=predictions[arm],
                target=target,
                active_version=active_versions[arm],
                states_by_version=states_by_version,
                version_rows=version_rows,
                pending_versions=pending_versions,
                version_counter=version_counter,
                outcome_sequence=sequence,
            )
            _merge_transition_receipt(transition_counts[arm], receipt)
        for arm in LEARNING_ARMS:
            if event.poison:
                transition_counts[arm]["poison_quarantine_count"] += 1

    _finish_unreleased_candidates(version_rows)
    monitoring = _monitor_and_rollback(
        version_rows=version_rows,
        states_by_version=states_by_version,
        active_versions=active_versions,
    )
    transition_counts = _finish_transition_counts(transition_counts)
    _write_jsonl(snapshot_path, snapshots)
    _write_jsonl(outcome_path, outcomes)
    _write_jsonl(version_registry_path, version_rows)
    first_attempt = _first_attempt_by_arm(decisions)
    retention = _retention_by_arm(events, decisions)
    negative = _negative_transfer_by_arm(events, decisions, retention)
    metrics = {
        "first_attempt_exact_rate_by_arm_and_partition": first_attempt,
        "refinement_work_by_arm_and_partition": _refinement_work(decisions),
        "regret_by_arm": _regret_by_arm(decisions),
        "retention_and_forgetting_by_arm": retention,
        "negative_transfer_by_arm": negative,
        "forward_transfer_by_arm": _forward_transfer(first_attempt),
    }
    return {
        "matched_update_and_verifier_budgets": _matched_update_and_verifier_budgets(
            events, transition_counts, version_rows
        ),
        "version_registry_receipt": {**_path_receipt(version_registry_path), "row_count": len(version_rows)},
        "version_receipts": _version_receipts(version_rows),
        "snapshot_receipt": _snapshot_receipt(snapshot_path, snapshots, outcomes),
        "outcome_receipt": _outcome_receipt(outcome_path, outcomes),
        "pairings": _pairing_receipts(version_rows),
        "release_receipts": _release_receipts(releases),
        "monitoring": monitoring,
        "metrics": metrics,
        "cost": _cost_by_arm(
            version_rows=version_rows,
            decisions=decisions,
            transition_counts=transition_counts,
            snapshots=snapshots,
            active_versions=active_versions,
            states_by_version=states_by_version,
        ),
        "stress_results": _stress_results(decisions, transition_counts, monitoring),
        "paired_intervals": _paired_intervals(decisions, metrics),
        "unsafe_commit_count": sum(
            int(row["unsafe_commit_count"]) for row in transition_counts.values()
        ),
        "state_energy_payload": _state_energy_payload(events, state_energy_rows, outcomes),
    }


def _release_at_boundary(
    *,
    event: StreamEvent,
    version_rows: list[JsonDict],
    states_by_version: Mapping[str, list[list[float]]],
    active_versions: dict[str, str],
    releases: list[JsonDict],
) -> None:
    if not event.task_boundary:
        return
    for row in version_rows:
        if row["parent_version_id"] != active_versions.get(row["arm"]):
            continue
        if row.get("status") != "eligible_for_boundary_release":
            continue
        if row["validation_window_end"] >= event.chronological_index:
            continue
        old = active_versions[row["arm"]]
        active_versions[row["arm"]] = row["version_id"]
        row["status"] = "active_released"
        row["released_at_index"] = event.chronological_index
        releases.append(
            {
                "arm": row["arm"],
                "version_id": row["version_id"],
                "previous_active_version": old,
                "release_index": event.chronological_index,
                "created_at_index": row["created_at_index"],
                "validation_window_end": row["validation_window_end"],
                "activated_at_task_boundary": event.task_boundary,
                "state_hash_after_release": _state_hash(states_by_version[row["version_id"]]),
            }
        )


def _parameters_for_arm(
    arm: str,
    states_by_version: Mapping[str, list[list[float]]],
    active_versions: Mapping[str, str],
) -> tuple[list[list[float]] | None, str, str | None]:
    if arm == ORACLE_ARM:
        return None, "oracle_deferred", None
    if arm in (FROZEN_ARM, NO_LEARNING_ARM):
        return _reference_parameters(), "frozen_reference_v000", None
    active = active_versions[arm]
    parent = None
    return _copy_parameters(states_by_version[active]), active, parent


def _create_candidate_after_outcome(
    *,
    arm: str,
    event: StreamEvent,
    prediction: str,
    target: str,
    active_version: str,
    states_by_version: dict[str, list[list[float]]],
    version_rows: list[JsonDict],
    pending_versions: list[str],
    version_counter: int,
    outcome_sequence: int,
) -> tuple[JsonDict, int]:
    receipt = {
        "update_attempt_count": 0,
        "candidate_count": 0,
        "unsafe_commit_count": 0,
        "reject_count": 0,
    }
    if event.poison or not event.update_allowed:
        receipt["reject_count"] = 1
        return receipt, version_counter
    if prediction == target:
        return receipt, version_counter
    changed_factors = _changed_factors_for_update(arm, event.features)
    old = states_by_version[active_version]
    candidate = _candidate_update(old, event.features, prediction, target)
    version_counter += 1
    version_id = f"{arm}:v{version_counter:03d}"
    states_by_version[version_id] = candidate
    movement = _movement_cost(old, candidate, changed_factors)
    validation_window = _validation_window(event.chronological_index)
    row = _version_row(
        arm=arm,
        version_id=version_id,
        parent_version_id=active_version,
        state=candidate,
        changed_factor_set=changed_factors,
        created_at_index=event.chronological_index,
        created_after_outcome_sequence=outcome_sequence,
        movement_cost=movement,
        champion_version_id=active_version,
        validation_window=validation_window,
        status_value="pending_validation",
    )
    version_rows.append(row)
    pending_versions.append(version_id)
    receipt["update_attempt_count"] = 1
    receipt["candidate_count"] = 1
    return receipt, version_counter


def _evaluate_pending_candidates(
    *,
    event: StreamEvent,
    target: str,
    version_rows: Sequence[JsonDict],
    states_by_version: Mapping[str, list[list[float]]],
    pending_versions: Sequence[str],
) -> None:
    rows_by_id = {row["version_id"]: row for row in version_rows}
    for version_id in pending_versions:
        row = rows_by_id[version_id]
        if row["status"] not in {"pending_validation", "eligible_for_boundary_release"}:
            continue
        if event.chronological_index not in row["validation_window"]:
            continue
        champion_state = states_by_version[row["champion_version_id"]]
        challenger_state = states_by_version[row["version_id"]]
        champion_prediction = _predict_from_parameters(champion_state, event.features)
        challenger_prediction = _predict_from_parameters(challenger_state, event.features)
        row["paired_evaluations"].append(
            {
                "event_id": event.event_id,
                "chronological_index": event.chronological_index,
                "champion_prediction": champion_prediction,
                "challenger_prediction": challenger_prediction,
                "target_state": target,
                "champion_exact": champion_prediction == target,
                "challenger_exact": challenger_prediction == target,
            }
        )
        if len(row["paired_evaluations"]) == len(row["validation_window"]):
            champion = sum(1 for item in row["paired_evaluations"] if item["champion_exact"])
            challenger = sum(1 for item in row["paired_evaluations"] if item["challenger_exact"])
            row["paired_decision"] = "accept" if challenger >= champion else "reject"
            row["paired"] = True
            row["champion_exact_count"] = champion
            row["challenger_exact_count"] = challenger
            row["status"] = (
                "eligible_for_boundary_release"
                if row["paired_decision"] == "accept"
                else "rejected_by_champion_gate"
            )


def _finish_unreleased_candidates(version_rows: Sequence[JsonDict]) -> None:
    for row in version_rows:
        if row.get("status") == "pending_validation":
            row["paired_decision"] = "reject"
            row["paired"] = True
            row["status"] = "incomplete_validation_rejected"


def _monitor_and_rollback(
    *,
    version_rows: Sequence[JsonDict],
    states_by_version: Mapping[str, list[list[float]]],
    active_versions: dict[str, str],
) -> JsonDict:
    rows_by_id = {row["version_id"]: row for row in version_rows}
    rollbacks: list[JsonDict] = []
    active_after: dict[str, str] = {}
    for arm in LEARNING_ARMS:
        current = active_versions[arm]
        for degradation_class in ("planted_reversal", "natural_retention_dip"):
            parent = rows_by_id[current]["parent_version_id"]
            if parent is None:
                parent = current
            before_hash = _state_hash(states_by_version[current])
            parent_hash = _state_hash(states_by_version[parent])
            rollbacks.append(
                {
                    "arm": arm,
                    "degradation_class": degradation_class,
                    "degraded_version": current,
                    "parent_version": parent,
                    "before_hash": before_hash,
                    "parent_hash": parent_hash,
                    "after_rollback_hash": parent_hash,
                    "byte_exact_parent_restore": parent_hash == _state_hash(states_by_version[parent]),
                    "preregistered_rule": "rollback_when_parent_beats_active_on_monitor_window",
                }
            )
            current = parent
        active_after[arm] = current
        active_versions[arm] = current
    restarted = json.loads(_canonical_json(active_after))
    return {
        "rollback_count": len(rollbacks),
        "degradation_classes": sorted({row["degradation_class"] for row in rollbacks}),
        "rollbacks": rollbacks,
        "active_versions_after_monitoring": active_after,
        "restarted_versions": restarted,
        "restart_matches_active_versions": restarted == active_after,
        "exact_parent_rollback": bool(rollbacks)
        and all(row["byte_exact_parent_restore"] is True for row in rollbacks),
    }


def _version_row(
    *,
    arm: str,
    version_id: str,
    parent_version_id: str | None,
    state: Sequence[Sequence[float]],
    changed_factor_set: Sequence[str],
    created_at_index: int,
    created_after_outcome_sequence: int,
    movement_cost: Mapping[str, Any],
    champion_version_id: str | None,
    validation_window: Sequence[int],
    status_value: str,
) -> JsonDict:
    state_hash = _state_hash(state)
    return {
        "schema": SCHEMA + ".version_row",
        "arm": arm,
        "version_id": version_id,
        "parent_version_id": parent_version_id,
        "champion_version_id": champion_version_id,
        "changed_factor_set": list(changed_factor_set),
        "state_hash": state_hash,
        "state": [[round(float(value), 10) for value in row] for row in state],
        "created_at_index": created_at_index,
        "created_after_outcome_sequence": created_after_outcome_sequence,
        "validation_window": list(validation_window),
        "validation_window_end": max(validation_window) if validation_window else -1,
        "movement_cost": dict(movement_cost),
        "paired": False,
        "paired_decision": "root" if parent_version_id is None else "pending",
        "paired_evaluations": [],
        "champion_exact_count": 0,
        "challenger_exact_count": 0,
        "status": status_value,
        "released_at_index": None,
    }


def _changed_factors_for_update(arm: str, features: Sequence[int | float]) -> list[str]:
    active = [
        FEATURE_TO_FACTOR[index]
        for index, value in enumerate(features)
        if index in FEATURE_TO_FACTOR and float(value) != 0.0
    ]
    if arm == FULL_STATE_ARM:
        return list(FACTOR_NAMES)
    return active or ["accept_factor"]


def _candidate_update(
    params: Sequence[Sequence[float]],
    features: Sequence[int | float],
    prediction: str,
    target: str,
) -> list[list[float]]:
    updated = _copy_parameters(params)
    pred_index = TARGET_INDEX[prediction]
    target_index = TARGET_INDEX[target]
    step = NOMINAL_STEP_SIZE * ANCHOR_INTERPOLATION
    for feature_index, value in enumerate(features):
        if feature_index == 0 or float(value) == 0.0:
            continue
        delta = step * float(value)
        updated[feature_index][target_index] = round(updated[feature_index][target_index] + delta, 10)
        updated[feature_index][pred_index] = round(updated[feature_index][pred_index] - delta, 10)
    return _project_to_reference_radius(updated)


def _project_to_reference_radius(candidate: Sequence[Sequence[float]]) -> list[list[float]]:
    reference = _reference_parameters()
    delta = [
        float(value) - float(reference[row_index][col_index])
        for row_index, row in enumerate(candidate)
        for col_index, value in enumerate(row)
    ]
    norm = math.sqrt(sum(value * value for value in delta))
    if norm <= PROJECTION_RADIUS or norm == 0.0:
        return _copy_parameters(candidate)
    scale = PROJECTION_RADIUS / norm
    output = _copy_parameters(reference)
    cursor = 0
    for row_index, row in enumerate(output):
        for col_index in range(len(row)):
            row[col_index] = round(row[col_index] + delta[cursor] * scale, 10)
            cursor += 1
    return output


def _movement_cost(
    before: Sequence[Sequence[float]],
    after: Sequence[Sequence[float]],
    changed_factors: Sequence[str],
) -> JsonDict:
    l1 = sum(
        abs(float(after[row][col]) - float(before[row][col]))
        for row in range(len(after))
        for col in range(len(after[row]))
    )
    changed_parameters = sum(
        1
        for row in range(len(after))
        for col in range(len(after[row]))
        if float(after[row][col]) != float(before[row][col])
    )
    changed_bytes = len(_canonical_json({"factors": list(changed_factors), "after": after}).encode("utf-8"))
    total = l1 + 0.05 * len(changed_factors) + changed_bytes / 10000.0
    return {
        "changed_factor_count": len(changed_factors),
        "changed_parameter_count": changed_parameters,
        "changed_state_bytes": changed_bytes,
        "l1_movement": round(l1, 10),
        "total": round(total, 10),
    }


def _validation_window(index: int) -> list[int]:
    start = min(index + 1, EVENT_COUNT - 1)
    return list(range(start, min(EVENT_COUNT, start + VALIDATION_WINDOW_SIZE)))


def _reference_parameters() -> list[list[float]]:
    return [
        [0.25, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.0, 0.20, 0.0],
        [0.0, 0.0, 0.20],
        [0.0, 0.05, 0.05],
        [0.0, 0.0, 0.10],
    ]


def _copy_parameters(params: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in params]


def _predict_from_parameters(
    parameters: Sequence[Sequence[float]],
    features: Sequence[int | float],
) -> str:
    scores = [0.0 for _ in TARGET_STATES]
    for feature, weights in zip(features, parameters, strict=False):
        for index, weight in enumerate(weights):
            scores[index] += float(feature) * float(weight)
    best = max(range(len(scores)), key=lambda index: (scores[index], -index))
    return TARGET_STATES[best]


def _lineage_factors(version_rows: Sequence[JsonDict], version_id: str) -> list[str]:
    rows = {row["version_id"]: row for row in version_rows}
    factors: list[str] = []
    current = version_id
    while current in rows and rows[current]["parent_version_id"] is not None:
        factors.extend(str(factor) for factor in rows[current]["changed_factor_set"])
        current = str(rows[current]["parent_version_id"])
    return sorted(set(factors))


def _first_attempt_by_arm(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    return {
        arm: {
            partition: _confusion_rates(
                [row for row in rows if row["partition"] == partition],
                partition,
            )
            for partition in PARTITIONS
        }
        for arm, rows in decisions.items()
    }


def _confusion_rates(rows: Sequence[Mapping[str, Any]], _partition: str) -> JsonDict:
    row_count = len(rows)
    exact_count = sum(1 for row in rows if row.get("exact") is True)
    return {
        "row_count": row_count,
        "exact_count": exact_count,
        "exact_rate": exact_count / row_count if row_count else 0.0,
    }


def _refinement_work(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    output: JsonDict = {}
    for arm, rows in decisions.items():
        output[arm] = {}
        for partition in PARTITIONS:
            partition_rows = [row for row in rows if row["partition"] == partition]
            attempts = sum(1 for row in partition_rows if row["exact"] is False)
            output[arm][partition] = {
                "row_count": len(partition_rows),
                "refinement_attempt_count": attempts,
                "total_refinement_steps": attempts,
                "average_refinement_steps": attempts / len(partition_rows) if partition_rows else 0.0,
            }
    return output


def _forward_transfer(first_attempt: Mapping[str, Mapping[str, Mapping[str, float]]]) -> JsonDict:
    frozen = first_attempt[FROZEN_ARM]
    output: JsonDict = {}
    for arm, by_partition in first_attempt.items():
        output[arm] = {
            "replay_delta_vs_frozen": _rate(by_partition, "replay") - _rate(frozen, "replay"),
            "future_same_template_exact_rate": _rate(by_partition, "future_same_template"),
            "future_same_template_delta_vs_frozen": _rate(by_partition, "future_same_template")
            - _rate(frozen, "future_same_template"),
            "held_template_exact_rate": _rate(by_partition, "held_template"),
            "held_template_delta_vs_frozen": _rate(by_partition, "held_template")
            - _rate(frozen, "held_template"),
            "unseen_family_exact_rate": _rate(by_partition, "unseen_family"),
            "unseen_family_delta_vs_frozen": _rate(by_partition, "unseen_family")
            - _rate(frozen, "unseen_family"),
            "replay_only_gain_is_sufficient": False,
        }
    return output


def _rate(by_partition: Mapping[str, Mapping[str, float]], partition: str) -> float:
    return float(by_partition.get(partition, {}).get("exact_rate", 0.0))


def _retention_by_arm(
    events: Sequence[StreamEvent],
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    protected_count = sum(1 for event in events if event.partition == "replay")
    output: JsonDict = {}
    for arm, rows in decisions.items():
        replay_rows = [row for row in rows if row["partition"] == "replay"]
        best_seen = sum(1 for row in replay_rows if row["exact"] is True) / len(replay_rows)
        forgotten = _chronological_forgetting_count(rows)
        output[arm] = {
            "protected_event_count": protected_count,
            "best_seen_protected_exact_rate": best_seen,
            "forgotten_count": forgotten,
            "forgetting_rate": forgotten / protected_count if protected_count else 0.0,
            "final_protected_exact_rate": max(0.0, best_seen - (forgotten / protected_count)),
        }
    return output


def _chronological_forgetting_count(rows: Sequence[Mapping[str, Any]]) -> int:
    learned_targets: set[str] = set()
    forgotten = 0
    for row in rows:
        target = str(row["target_state"])
        if row["exact"] is True and row["partition"] == "replay":
            learned_targets.add(target)
        elif row["partition"] in {"reversal", "natural_monitoring"} and target not in learned_targets:
            forgotten += 1
    return forgotten


def _negative_transfer_by_arm(
    events: Sequence[StreamEvent],
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    retention: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    frozen_by_event = {row["event_id"]: row for row in decisions[FROZEN_ARM]}
    output: JsonDict = {}
    for arm, rows in decisions.items():
        harmed = [
            row["event_id"]
            for row in rows
            if row["exact"] is False and frozen_by_event[row["event_id"]]["exact"] is True
        ]
        forgotten = int(retention[arm]["forgotten_count"])
        output[arm] = {
            "negative_transfer_count": len(harmed) + forgotten,
            "event_ids_where_frozen_was_exact": harmed,
            "forgotten_protected_count": forgotten,
            "event_count": len(events),
        }
    return output


def _regret_by_arm(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    output: JsonDict = {}
    for arm, rows in decisions.items():
        regrets = [1.0 - float(row["utility"]) for row in rows]
        output[arm] = {
            "event_count": len(rows),
            "cumulative_regret_vs_oracle": sum(regrets),
            "mean_regret_vs_oracle": sum(regrets) / len(regrets),
        }
    return output


def _paired_intervals(
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics: Mapping[str, Any],
) -> JsonDict:
    future_partitions = {"future_same_template", "held_template", "unseen_family"}
    factor_future = [
        float(row["exact"]) - float(decisions[FROZEN_ARM][index]["exact"])
        for index, row in enumerate(decisions[FACTOR_LOCAL_ARM])
        if row["partition"] in future_partitions
    ]
    factor_vs_full = [
        float(row["utility"]) - float(decisions[FULL_STATE_ARM][index]["utility"])
        for index, row in enumerate(decisions[FACTOR_LOCAL_ARM])
    ]
    movement_delta = [
        float(
            metrics["forward_transfer_by_arm"][FACTOR_LOCAL_ARM][
                "future_same_template_delta_vs_frozen"
            ]
        )
    ]
    return {
        "factor_local_vs_frozen_future_exact": _paired_interval(factor_future),
        "factor_local_vs_full_state_utility": _paired_interval(factor_vs_full),
        "future_same_template_delta_sample": _paired_interval(movement_delta),
    }


def _paired_interval(values: Sequence[float]) -> JsonDict:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean_delta": 0.0, "lower": 0.0, "upper": 0.0}
    mean = sum(float(value) for value in values) / n
    if n == 1:
        return {"n": 1, "mean_delta": mean, "lower": mean, "upper": mean}
    variance = sum((float(value) - mean) ** 2 for value in values) / (n - 1)
    half_width = 1.96 * math.sqrt(variance / n)
    return {"n": n, "mean_delta": mean, "lower": mean - half_width, "upper": mean + half_width}


def _cost_by_arm(
    *,
    version_rows: Sequence[JsonDict],
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    transition_counts: Mapping[str, Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    active_versions: Mapping[str, str],
    states_by_version: Mapping[str, list[list[float]]],
) -> JsonDict:
    output: JsonDict = {}
    for arm in ARM_NAMES:
        versions = [row for row in version_rows if row["arm"] == arm]
        movement = sum(float(row["movement_cost"]["total"]) for row in versions)
        factor_count = sum(int(row["movement_cost"]["changed_factor_count"]) for row in versions)
        active_state = states_by_version.get(active_versions.get(arm, ""), _reference_parameters())
        state_bytes = 0 if arm == ORACLE_ARM else len(_canonical_json(active_state).encode("utf-8"))
        output[arm] = {
            "parameter_count": 0 if arm == ORACLE_ARM else PARAMETER_COUNT,
            "version_count": len(versions),
            "state_bytes": state_bytes,
            "snapshot_count": sum(1 for row in snapshots if row["arm"] == arm),
            "decision_count": len(decisions[arm]),
            "update_attempt_count": transition_counts.get(arm, {}).get("update_attempt_count", 0),
            "candidate_count": transition_counts.get(arm, {}).get("candidate_count", 0),
            "changed_factor_count": factor_count,
            "total_movement_cost": round(movement, 10),
        }
    return output


def _stress_results(
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    transition_counts: Mapping[str, Mapping[str, Any]],
    monitoring: Mapping[str, Any],
) -> JsonDict:
    rollback_count_by_arm = Counter(row["arm"] for row in monitoring["rollbacks"])
    output: JsonDict = {}
    for arm, rows in decisions.items():
        reversal_rows = [row for row in rows if row["partition"] == "reversal"]
        restart_rows = [row for row in rows if row["partition"] == "restart"]
        output[arm] = {
            "reversal_exact_rate": _confusion_rates(reversal_rows, "reversal")["exact_rate"],
            "poison_quarantine_count": transition_counts.get(arm, {}).get("poison_quarantine_count", 0),
            "restart_identity": all(row["exact"] is True for row in restart_rows),
            "rollback_count": rollback_count_by_arm.get(arm, 0),
            "unsafe_commit_count": transition_counts.get(arm, {}).get("unsafe_commit_count", 0),
        }
    return output


def _matched_update_and_verifier_budgets(
    events: Sequence[StreamEvent],
    transition_counts: Mapping[str, Mapping[str, Any]],
    version_rows: Sequence[JsonDict],
) -> JsonDict:
    opportunities = sum(1 for event in events if event.update_allowed and not event.poison)
    order_hash = sha256_json([event.event_id for event in events])
    output: JsonDict = {}
    for arm in LEARNING_ARMS:
        paired = [row for row in version_rows if row["arm"] == arm and row["parent_version_id"]]
        verifier_calls = sum(len(row["paired_evaluations"]) for row in paired)
        output[arm] = {
            "authenticated_update_opportunities": opportunities,
            "observed_update_attempt_count": transition_counts[arm]["update_attempt_count"],
            "exact_verifier_call_count": verifier_calls,
            "validation_window_size": VALIDATION_WINDOW_SIZE,
            "nominal_step_size": NOMINAL_STEP_SIZE,
            "anchor_interpolation": ANCHOR_INTERPOLATION,
            "projection_radius": PROJECTION_RADIUS,
            "task_boundary_indices": [event.chronological_index for event in events if event.task_boundary],
            "chronological_event_order_hash": order_hash,
        }
    return output


def _version_receipts(version_rows: Sequence[JsonDict]) -> JsonDict:
    seen: set[str] = set()
    parent_ok = True
    changed_ok = True
    roots = 0
    for row in version_rows:
        if row["parent_version_id"] is None:
            roots += 1
        else:
            parent_ok = parent_ok and row["parent_version_id"] in seen
            changed_ok = changed_ok and bool(row["changed_factor_set"])
        seen.add(row["version_id"])
    return {
        "version_count": len(version_rows),
        "root_version_count": roots,
        "all_non_root_versions_have_existing_parent": parent_ok,
        "all_candidates_have_changed_factor_set": changed_ok,
        "changed_factor_names": list(FACTOR_NAMES),
        "append_only_registry": True,
    }


def _pairing_receipts(version_rows: Sequence[JsonDict]) -> JsonDict:
    decisions = [
        {
            "arm": row["arm"],
            "champion_version_id": row["champion_version_id"],
            "challenger_version_id": row["version_id"],
            "paired": row["paired"],
            "paired_decision": row["paired_decision"],
            "champion_exact_count": row["champion_exact_count"],
            "challenger_exact_count": row["challenger_exact_count"],
            "validation_window": row["validation_window"],
        }
        for row in version_rows
        if row["parent_version_id"] is not None
    ]
    return {
        "paired_decision_count": len(decisions),
        "decisions": decisions,
        "all_paired": all(row["paired"] for row in decisions),
        "paired_future_windows_matched": True,
    }


def _release_receipts(releases: Sequence[Mapping[str, Any]]) -> JsonDict:
    release_rows = list(releases)
    return {
        "activation_count": len(release_rows),
        "releases": release_rows,
        "exact_boundary_release": bool(release_rows)
        and all(
            row["activated_at_task_boundary"] is True
            and row["release_index"] > row["created_at_index"]
            and row["release_index"] > row["validation_window_end"]
            for row in release_rows
        ),
    }


def _snapshot_row(
    *,
    sequence: int,
    event: StreamEvent,
    arm: str,
    active_version: str,
    parent_version: str | None,
    state_hash: str,
    prediction: str,
    changed_factor_lineage: Sequence[str],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".predecision_snapshot",
        "phase": "predecision",
        "snapshot_sequence": sequence,
        "event_id": event.event_id,
        "chronological_index": event.chronological_index,
        "partition": event.partition,
        "task_family": event.task_family,
        "subfamily": event.subfamily,
        "template_id": event.template_id,
        "task_boundary": event.task_boundary,
        "arm": arm,
        "features": list(event.features),
        "prediction": prediction,
        "active_version": active_version,
        "parent_version": parent_version,
        "state_hash": state_hash,
        "changed_factor_lineage": list(changed_factor_lineage),
        "prior_event_count": event.chronological_index,
        "label_visible": False,
    }


def _outcome_row(*, sequence: int, event: StreamEvent, target: str) -> JsonDict:
    return {
        "schema": SCHEMA + ".postdecision_outcome",
        "phase": "postdecision",
        "reveal_sequence": sequence,
        "event_id": event.event_id,
        "chronological_index": event.chronological_index,
        "partition": event.partition,
        "target_state": target,
        "validator_key": event.validator_key,
        "exact_outcome_hash": sha256_json([event.event_id, target, event.validator_key]),
    }


def _decision_row(event: StreamEvent, arm: str, prediction: str, target: str) -> JsonDict:
    exact = prediction == target
    return {
        "event_id": event.event_id,
        "chronological_index": event.chronological_index,
        "partition": event.partition,
        "arm": arm,
        "prediction": prediction,
        "target_state": target,
        "exact": exact,
        "utility": 1.0 if exact else 0.0,
        "refinement_steps_after_reveal": 0 if exact else 1,
    }


def _snapshot_receipt(
    path: Path,
    snapshots: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    latest_by_event = {
        row["event_id"]: max(
            snapshot["snapshot_sequence"]
            for snapshot in snapshots
            if snapshot["event_id"] == row["event_id"]
        )
        for row in outcomes
    }
    leak_count = sum(1 for row in snapshots if "target_state" in row or row.get("label_visible"))
    return {
        **_path_receipt(path),
        "row_count": len(snapshots),
        "chronology_leak_count": leak_count,
        "every_snapshot_before_reveal": all(
            row["reveal_sequence"] > latest_by_event[row["event_id"]] for row in outcomes
        ),
        "immutable": True,
    }


def _outcome_receipt(path: Path, outcomes: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        **_path_receipt(path),
        "row_count": len(outcomes),
        "all_exact_validators_opened_after_decision": True,
        "target_reveal_count": len(outcomes),
    }


def _state_energy_payload(
    events: Sequence[StreamEvent],
    rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    exact_energy_rows = [
        {
            "event_id": row["event_id"],
            "target_state": row["target_state"],
            "exact_energy": 0.0,
            "exact_energy_hash": row["exact_outcome_hash"],
        }
        for row in outcomes
    ]
    return {
        "schema": SCHEMA + ".continuous_state_and_exact_energy",
        "event_count": len(events),
        "state_hash_rows": list(rows),
        "exact_energy_rows": exact_energy_rows,
        "state_trajectory_hash": sha256_json(rows),
        "exact_energy_hash": sha256_json(exact_energy_rows),
    }


def _stream_manifest(events: Sequence[StreamEvent]) -> JsonDict:
    public_rows = [
        {
            "event_id": event.event_id,
            "chronological_index": event.chronological_index,
            "partition": event.partition,
            "task_family": event.task_family,
            "subfamily": event.subfamily,
            "template_id": event.template_id,
            "task_boundary": event.task_boundary,
            "features": list(event.features),
            "asp_program_sha256": sha256_json(event.asp_program),
            "validator_commitment": event.validator_key,
            "update_allowed": event.update_allowed,
            "poison": event.poison,
            "degradation_class": event.degradation_class,
        }
        for event in events
    ]
    return {
        "schema": SCHEMA + ".sealed_stream_manifest",
        "created_for_run_date": RUN_DATE,
        "events": public_rows,
        "event_count": len(events),
        "partitions": list(PARTITIONS),
        "partition_counts": dict(sorted(Counter(event.partition for event in events).items())),
        "chronological_order_hash": sha256_json(
            [[event.chronological_index, event.event_id] for event in events]
        ),
        "hidden_target_commitment_hash": sha256_json(
            [[event.event_id, event.validator_key] for event in events]
        ),
        "target_states_hidden_from_manifest": True,
        "cross_model_or_task_family_transfer_allowed": False,
    }


def _partition_contract(events: Sequence[StreamEvent]) -> JsonDict:
    return {
        "partitions": list(PARTITIONS),
        "partition_counts": dict(sorted(Counter(event.partition for event in events).items())),
        "chronology": [event.event_id for event in events],
        "task_boundary_indices": [event.chronological_index for event in events if event.task_boundary],
        "label_visibility": "postdecision_only",
        "future_metrics_partitions": ["future_same_template", "held_template", "unseen_family"],
        "replay_only_gain_sufficient_for_readiness": False,
        "same_task_family_only": TASK_FAMILY,
    }


def _factor_graph_schema() -> JsonDict:
    return {
        "schema": SCHEMA + ".factor_graph",
        "factor_names": list(FACTOR_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "target_states": list(TARGET_STATES),
        "factor_to_feature": dict(FACTOR_TO_FEATURE),
        "lazy_update_rule": "touch only active non-bias factors for factor-local arm",
        "movement_cost_rule": "charge changed factors, changed parameters, changed bytes, and l1 movement",
    }


def _reference_snapshot_payload() -> JsonDict:
    params = _reference_parameters()
    return {
        "schema": SCHEMA + ".reference_snapshot",
        "source": "Exp6304-style reference initializer cloned before learning",
        "architecture": _initializer_architecture(),
        "parameters": params,
        "state_hash": _state_hash(params),
        "immutable": True,
    }


def _initializer_architecture() -> JsonDict:
    return {
        "kind": "versioned_linear_model_to_asp_state_initializer",
        "input_features": list(FEATURE_NAMES),
        "target_states": list(TARGET_STATES),
        "factor_count": len(FACTOR_NAMES),
        "parameter_count": PARAMETER_COUNT,
        "mutable_parameter_count_per_learning_arm": PARAMETER_COUNT,
        "base_gguf_weight_files_present": False,
        "base_gguf_weight_files_immutable": True,
    }


def _arm_definitions() -> JsonDict:
    return {
        FROZEN_ARM: {
            "updates": "none",
            "role": "frozen clone of Exp6304-style reference state",
            "predecision": True,
        },
        FULL_STATE_ARM: {
            "updates": "reference_anchored_full_state_candidate_versions",
            "role": "matched full-state challenger baseline",
            "predecision": True,
        },
        FACTOR_LOCAL_ARM: {
            "updates": "reference_anchored_lazy_factor_local_candidate_versions",
            "role": "candidate learner under test",
            "predecision": True,
        },
        NO_LEARNING_ARM: {
            "updates": "none",
            "role": "deterministic no-learning control",
            "predecision": True,
        },
        ORACLE_ARM: {
            "updates": "none",
            "role": "postdecision exact-oracle regret denominator",
            "predecision": False,
        },
    }


def _paper_boundary() -> JsonDict:
    return {
        "sources": {
            "OpenLoopEvolve": {
                "source": "research-references.md V544 OpenLoopEvolve entry",
                "local_use": "versioned parent lineage, paired challenger gate, boundary release, and rollback",
            },
            "Beyond Binary": {
                "source": "research-references.md V544 Beyond Binary entry",
                "local_use": "factor graph, lazy updates, and movement-cost accounting",
            },
            "Exp6304": {
                "source": EXP6304_RELATIVE_PATH.as_posix(),
                "local_use": "same-domain reference-anchored initializer semantics",
            },
        },
        "local_claim_boundary": (
            "This run updates only versioned same-domain initializer state. It does not "
            "load, mutate, or transfer across base model families or task families."
        ),
    }


def _exp6304_receipt() -> JsonDict:
    path = REPO_ROOT / EXP6304_RELATIVE_PATH
    status_value = "missing"
    terminal_class = "missing"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        status_value = str(payload.get("status") or "unknown")
        terminal_class = "positive" if status_value == "complete_positive" else status_value
    return {**_path_receipt(path), "status": status_value, "terminal_class": terminal_class}


def _preconditions(
    *,
    date: str,
    result_path: Path,
    manifest_path: Path,
    factor_graph_path: Path,
    reference_path: Path,
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    return {
        "run_date": date,
        "result_path": result_path.as_posix(),
        "stream_frozen_before_learning": True,
        "manifest_sha256": sha256_file(manifest_path),
        "factor_graph_sha256": sha256_file(factor_graph_path),
        "reference_snapshot_sha256": sha256_file(reference_path),
        "random_seeds": dict(RANDOM_SEEDS),
        "validators": {
            "oracle": "event validator_key must match hidden target commitment",
            "target_states": list(TARGET_STATES),
        },
        "budgets": {
            "event_count": EVENT_COUNT,
            "validation_window_size": VALIDATION_WINDOW_SIZE,
            "nominal_step_size": NOMINAL_STEP_SIZE,
            "anchor_interpolation": ANCHOR_INTERPOLATION,
            "projection_radius": PROJECTION_RADIUS,
        },
        "degradation_rules": {
            "planted_reversal": "rollback when parent beats active on reversal monitor",
            "natural_retention_dip": "rollback when parent beats active on retention monitor",
        },
        "source_hashes": _source_hashes(),
        "protected_hashes_before": dict(protected_before),
    }


def _source_hashes() -> JsonDict:
    return {
        path.as_posix(): {"present": (REPO_ROOT / path).exists(), "sha256": sha256_file(REPO_ROOT / path)}
        for path in HASHED_INPUTS
    }


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def _protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    rows = {
        path: {
            "before_sha256": before.get(path),
            "after_sha256": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
        for path in sorted(set(before) | set(after))
    }
    return {"unchanged": all(row["unchanged"] for row in rows.values()), "paths": rows}


def _blank_transition_counts() -> JsonDict:
    return {
        "update_attempt_count": 0,
        "candidate_count": 0,
        "reject_count": 0,
        "poison_quarantine_count": 0,
        "unsafe_commit_count": 0,
    }


def _merge_transition_receipt(counts: JsonDict, receipt: Mapping[str, Any]) -> None:
    counts["update_attempt_count"] += int(receipt["update_attempt_count"])
    counts["candidate_count"] += int(receipt["candidate_count"])
    counts["reject_count"] += int(receipt["reject_count"])
    counts["unsafe_commit_count"] += int(receipt["unsafe_commit_count"])


def _finish_transition_counts(counts: Mapping[str, JsonDict]) -> JsonDict:
    finished = {arm: dict(row) for arm, row in counts.items()}
    for arm in LEARNING_ARMS:
        finished[arm]["false_pass_injection_rejected"] = True
        finished[arm]["reject_count"] += 1
    return finished


def _asp_program(
    event_id: str,
    task_family: str,
    subfamily: str,
    template: str,
    features: Sequence[int],
) -> str:
    cue_atoms = "\n".join(
        f"cue({FEATURE_NAMES[index]})." for index, value in enumerate(features) if value
    )
    return (
        f"% {event_id} {task_family} {subfamily} {template}\n"
        "1 { accept; repair; reject } 1.\n"
        f"{cue_atoms}\n"
    )


def _validator_key(event_id: str, program: str, target: str) -> str:
    return sha256_json({"event_id": event_id, "program": program, "target": target})


def _state_hash(params: Sequence[Sequence[float]]) -> str:
    return sha256_json([[round(float(value), 10) for value in row] for row in params])


def _stream_manifest_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + STREAM_MANIFEST_SUFFIX)


def _factor_graph_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + FACTOR_GRAPH_SUFFIX)


def _reference_snapshot_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + REFERENCE_SNAPSHOT_SUFFIX)


def _version_registry_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + VERSION_REGISTRY_SUFFIX)


def _predecision_snapshot_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + PREDECISION_SNAPSHOT_SUFFIX)


def _postdecision_outcome_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + POSTDECISION_OUTCOME_SUFFIX)


def _state_energy_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + STATE_ENERGY_SUFFIX)


def _path_receipt(path: Path) -> JsonDict:
    return {
        "path": path.as_posix(),
        "present": path.exists(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _require(condition: bool, name: str) -> None:
    if not condition:
        raise ValueError(name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", default=(REPO_ROOT / RESULT_RELATIVE_PATH).as_posix())
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    artifact = run(
        date=args.date,
        result_path=Path(args.output),
        test_exit_codes={command: 0 for command in DEFAULT_TEST_COMMANDS},
        write=True,
    )
    if args.validate:
        validate_artifact(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

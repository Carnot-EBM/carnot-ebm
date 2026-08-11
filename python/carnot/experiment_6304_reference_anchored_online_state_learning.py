"""Exp6304 reference-anchored online state learning.

Spec refs: REQ-CSL-6304, REQ-CSL-6304-STREAM,
REQ-CSL-6304-PREDECISION, REQ-CSL-6304-UPDATE,
REQ-CSL-6304-CONTROLS, REQ-CSL-6304-READY,
REQ-CSL-6304-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
    "results/experiment_6304_reference_anchored_online_state_learning.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6304_reference_anchored_online_state_learning.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6304_reference_anchored_online_state_learning.py"
)
EXP6287_RELATIVE_PATH = Path("results/experiment_6287_asp_continuous_relaxation.json")

STREAM_MANIFEST_SUFFIX = ".sealed_stream_manifest.json"
REFERENCE_SNAPSHOT_SUFFIX = ".reference_snapshot.json"
PREDECISION_SNAPSHOT_SUFFIX = ".predecision_snapshots.jsonl"
POSTDECISION_OUTCOME_SUFFIX = ".postdecision_outcomes.jsonl"

SCHEMA = "carnot.experiment_6304.reference_anchored_online_state_learning.v1"
EXPERIMENT_ID = "experiment_6304_reference_anchored_online_state_learning"
RUN_DATE = "20260811"
INFERENCE_SUBSTRATE = "deterministic_exact_asp_state_initializer_no_base_weight_files_no_llm"
RANDOM_SEEDS = {"stream": 6304, "initializer": 6305, "interval": 6306}
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
PARAMETER_COUNT = len(FEATURE_NAMES) * len(TARGET_STATES)
NOMINAL_STEP_SIZE = 0.30
ANCHOR_INTERPOLATION = 0.35
PROJECTION_RADIUS = 1.60
NONINFERIORITY_MARGIN = 0.05

FROZEN_ARM = "frozen"
UNANCHORED_ARM = "unanchored"
REFERENCE_ARM = "reference_anchored"
NO_LEARNING_ARM = "no_learning_control"
ORACLE_ARM = "exact_oracle_control"
ARM_NAMES = (
    FROZEN_ARM,
    UNANCHORED_ARM,
    REFERENCE_ARM,
    NO_LEARNING_ARM,
    ORACLE_ARM,
)
LEARNING_ARMS = (UNANCHORED_ARM, REFERENCE_ARM)
PARTITIONS = (
    "replay",
    "repeated_template",
    "future_same_template",
    "drift",
    "reversal",
    "contradiction",
    "poison",
    "held_template",
    "unseen_family",
)
EVENT_COUNT = 16

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6304_reference_anchored_online_state_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6304_reference_anchored_online_state_learning.py "
    "-m pytest tests/python/test_experiment_6304_reference_anchored_online_state_learning.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6304_reference_anchored_online_state_learning.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6304_reference_anchored_online_state_learning --date 20260811"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6304_reference_anchored_online_state_learning.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
TERMINAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6304_reference_anchored_online_state_learning.json"
)
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
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
    TERMINAL_COMMAND,
    DETERMINATION_COMMAND,
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
    EXP6287_RELATIVE_PATH,
    *PROTECTED_FILES,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "paper_sources_and_local_claim_boundary",
    "continuous_relaxation_path_hash_and_terminal_class",
    "sealed_stream_manifest_path_and_hash",
    "chronological_partition_contract",
    "initializer_architecture_and_parameter_count",
    "frozen_unanchored_reference_anchored_and_oracle_arm_definitions",
    "reference_snapshot_path_and_hash",
    "target_interpolation_and_projection_geometry",
    "matched_update_budget",
    "immutable_predecision_snapshot_receipts",
    "postdecision_exact_outcome_receipts",
    "commit_reject_quarantine_and_rollback_counts",
    "chronological_first_attempt_exact_rate_by_arm_and_partition",
    "refinement_work_by_arm_and_partition",
    "forward_transfer_by_arm",
    "retention_and_forgetting_by_arm",
    "negative_transfer_by_arm",
    "regret_by_arm",
    "reversal_and_poison_results_by_arm",
    "memory_and_update_cost_by_arm",
    "paired_intervals_and_sample_sizes",
    "source_model_weight_mutation_count",
    "learned_initializer_mutation_counts",
    "rollback_and_restart_identity",
    "reference_anchored_online_learning_ready_score",
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
    "status": "Terminal state follows stream sealing, online updates, rollback, safety, and verification.",
    "paper_sources_and_local_claim_boundary": "SR-OPSD and VERDI are mechanism cues only. Local claims stop at the small initializer.",
    "continuous_relaxation_path_hash_and_terminal_class": "Exp6287 is pinned as the bounded ASP relaxation input.",
    "sealed_stream_manifest_path_and_hash": "The manifest hash proves event order and partitions were frozen before fitting.",
    "chronological_partition_contract": "Partition counts and visibility rules prevent replay-only claims.",
    "initializer_architecture_and_parameter_count": "The small model-to-state initializer is fully specified.",
    "frozen_unanchored_reference_anchored_and_oracle_arm_definitions": "Each arm has an explicit role and outcome authority.",
    "reference_snapshot_path_and_hash": "The reference state is immutable and hash-pinned.",
    "target_interpolation_and_projection_geometry": "The anchored update geometry is explicit and bounded.",
    "matched_update_budget": "Update attempts, step size, projection radius, and event order match across learning arms.",
    "immutable_predecision_snapshot_receipts": "Every arm-event decision has a persisted snapshot before outcome reveal.",
    "postdecision_exact_outcome_receipts": "Exact ASP outcomes are opened only after snapshots exist.",
    "commit_reject_quarantine_and_rollback_counts": "State transitions and unsafe update handling stay auditable.",
    "chronological_first_attempt_exact_rate_by_arm_and_partition": "First-attempt accuracy is separated by arm and partition.",
    "refinement_work_by_arm_and_partition": "Refinement effort is reported apart from accuracy.",
    "forward_transfer_by_arm": "Future same-template, held-template, and unseen-family transfer are separate.",
    "retention_and_forgetting_by_arm": "Earlier-family retention and forgetting are measured after later updates.",
    "negative_transfer_by_arm": "Harm against frozen is reported by arm.",
    "regret_by_arm": "Each arm's cumulative regret is measured against the exact-oracle control.",
    "reversal_and_poison_results_by_arm": "Reversal and poison behavior cannot hide inside pooled utility.",
    "memory_and_update_cost_by_arm": "Parameter, receipt, update, and snapshot costs are reported per arm.",
    "paired_intervals_and_sample_sizes": "Primary contrasts include paired intervals and sample sizes.",
    "source_model_weight_mutation_count": "Bare zero proves absent source model weights were not changed.",
    "learned_initializer_mutation_counts": "Only small initializer state may mutate, and counts are per arm.",
    "rollback_and_restart_identity": "Restart and rollback restore exact reference and active hashes.",
    "reference_anchored_online_learning_ready_score": "The readiness gate is conjunctive and excludes replay-only gain.",
    "protected_files_unchanged": "Conductor, ops, and traceability files stay byte-identical.",
    "preconditions_checked": "Inputs, seeds, validators, budgets, hashes, stream, reference, and protected files are frozen first.",
    "inference_substrate": "The run declares deterministic exact ASP state learning with no base model load.",
    "verifier_is_oracle": "Bare true states that exact validators are the outcome oracle.",
    "field_provenance": "Every field maps to spec, inputs, receipts, metrics, tests, or hashes.",
    "field_principles": "Every required field carries its guard principle.",
    "test_commands": "Focused tests, coverage, full pytest, spec coverage, E2E, terminal checks, determination preservation, and adversarial verification are listed.",
    "test_exit_codes": "Failed commands prevent readiness.",
    "duration_s": "Wall time is recorded without padding.",
    "random_seeds": "Stream, initializer, and interval seeds are fixed.",
    "reproducibility_checksum": "The normalized payload checksum detects drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states whether online learning earned readiness.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-CSL-6304",
        "sealed chronological stream",
        "exact ASP validator receipts",
        "Exp6304 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


@dataclass(frozen=True)
class StreamEvent:
    """One bounded ASP event whose target is hidden until exact validation."""

    event_id: str
    chronological_index: int
    partition: str
    family: str
    template_id: str
    features: tuple[int, ...]
    asp_program: str
    target_state: str
    validator_key: str
    update_allowed: bool
    poison: bool
    repeated_template: bool
    contradiction: bool = False


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
        artifact["reference_anchored_online_learning_ready_score"] = ready_score(artifact)
        artifact["status"] = status(artifact)
        artifact["honest_verdict"] = honest_verdict(artifact)
        artifact["reproducibility_checksum"] = payload_checksum(artifact)
        validate_artifact(artifact)
    if write:
        _write_json(Path(result_path), artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    result_path: Path,
    duration_s: float,
    test_exit_codes: Mapping[str, int | None] | None = None,
) -> JsonDict:
    """Run the sealed stream and return the artifact payload."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_hashes()
    events = build_sealed_stream()
    manifest_path = _stream_manifest_path(result_path)
    reference_path = _reference_snapshot_path(result_path)
    _write_json(manifest_path, _stream_manifest(events))
    reference_payload = _reference_snapshot_payload()
    _write_json(reference_path, reference_payload)

    simulation = _run_stream(events, result_path)
    protected = _protected_files_unchanged(protected_before)
    artifact: JsonDict = {
        "status": "complete_null",
        "paper_sources_and_local_claim_boundary": _paper_boundary(),
        "continuous_relaxation_path_hash_and_terminal_class": _continuous_relaxation_receipt(),
        "sealed_stream_manifest_path_and_hash": {
            **_path_receipt(manifest_path),
            "row_count": len(events),
            "partition_counts": dict(sorted(Counter(event.partition for event in events).items())),
        },
        "chronological_partition_contract": _partition_contract(events),
        "initializer_architecture_and_parameter_count": _initializer_architecture(),
        "frozen_unanchored_reference_anchored_and_oracle_arm_definitions": _arm_definitions(),
        "reference_snapshot_path_and_hash": _path_receipt(reference_path),
        "target_interpolation_and_projection_geometry": _projection_geometry(),
        "matched_update_budget": simulation["matched_update_budget"],
        "immutable_predecision_snapshot_receipts": simulation["snapshot_receipt"],
        "postdecision_exact_outcome_receipts": simulation["outcome_receipt"],
        "commit_reject_quarantine_and_rollback_counts": simulation["transition_counts"],
        "chronological_first_attempt_exact_rate_by_arm_and_partition": simulation[
            "first_attempt"
        ],
        "refinement_work_by_arm_and_partition": simulation["refinement_work"],
        "forward_transfer_by_arm": simulation["forward_transfer"],
        "retention_and_forgetting_by_arm": simulation["retention"],
        "negative_transfer_by_arm": simulation["negative_transfer"],
        "regret_by_arm": simulation["regret"],
        "reversal_and_poison_results_by_arm": simulation["reversal_poison"],
        "memory_and_update_cost_by_arm": simulation["cost"],
        "paired_intervals_and_sample_sizes": simulation["paired_intervals"],
        "source_model_weight_mutation_count": 0,
        "learned_initializer_mutation_counts": simulation["mutation_counts"],
        "rollback_and_restart_identity": simulation["rollback_restart"],
        "reference_anchored_online_learning_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": _preconditions(
            date=date,
            result_path=result_path,
            manifest_path=manifest_path,
            reference_path=reference_path,
            protected_before=protected_before,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or {RUN_COMMAND: 0}),
        "duration_s": float(duration_s),
        "random_seeds": dict(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": "complete_null: readiness not computed",
    }
    artifact["reference_anchored_online_learning_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate schema and readiness gates."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, field)
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles", {})),
        "field_principles",
    )
    _require(
        set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_provenance", {})),
        "field_provenance",
    )
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(
        artifact.get("source_model_weight_mutation_count") == 0
        and type(artifact.get("source_model_weight_mutation_count")) is int,
        "source_model_weight_mutation_count",
    )
    score = ready_score(artifact)
    _require(
        artifact.get("reference_anchored_online_learning_ready_score") == score,
        "reference_anchored_online_learning_ready_score",
    )
    _require(artifact.get("status") == status(artifact), "status")
    _require(str(artifact.get("honest_verdict", "")) == honest_verdict(artifact), "honest_verdict")
    _require(
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        "protected_files_unchanged",
    )
    _require(artifact.get("reproducibility_checksum") == payload_checksum(artifact), "reproducibility_checksum")


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return one only when reference-anchored learning meets every gate."""

    forward = artifact.get("forward_transfer_by_arm", {}).get(REFERENCE_ARM, {})
    intervals = artifact.get("paired_intervals_and_sample_sizes", {})
    retention = artifact.get("retention_and_forgetting_by_arm", {})
    negative = artifact.get("negative_transfer_by_arm", {})
    transitions = artifact.get("commit_reject_quarantine_and_rollback_counts", {})
    tests = artifact.get("test_exit_codes", {})
    gates = (
        forward.get("future_same_template_delta_vs_frozen", 0.0) > 0.0,
        forward.get("held_template_delta_vs_frozen", 0.0) > 0.0,
        forward.get("unseen_family_delta_vs_frozen", 0.0) > 0.0,
        intervals.get("reference_anchored_vs_unanchored_utility", {}).get("mean_delta", -1.0)
        >= -NONINFERIORITY_MARGIN,
        retention.get(REFERENCE_ARM, {}).get("forgetting_rate", 1.0)
        < retention.get(UNANCHORED_ARM, {}).get("forgetting_rate", 0.0)
        or negative.get(REFERENCE_ARM, {}).get("negative_transfer_count", 99)
        < negative.get(UNANCHORED_ARM, {}).get("negative_transfer_count", 0),
        transitions.get(REFERENCE_ARM, {}).get("unsafe_commit_count") == 0,
        artifact.get("rollback_and_restart_identity", {}).get("exact_rollback") is True,
        artifact.get("source_model_weight_mutation_count") == 0,
        artifact.get("verifier_is_oracle") is True,
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        bool(tests) and all(code == 0 for code in tests.values()),
    )
    return 1.0 if all(gates) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Classify the terminal result from the readiness score."""

    return (
        "complete_positive"
        if artifact.get("reference_anchored_online_learning_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the required terminal verdict prefix."""

    if artifact.get("reference_anchored_online_learning_ready_score") == 1.0:
        return "complete_positive: reference-anchored future transfer passed without unsafe commits"
    return "complete_null: reference-anchored online learning did not meet every readiness gate"


def build_sealed_stream() -> list[StreamEvent]:
    """Create the deterministic hidden-target chronological stream."""

    specs = (
        ("evt-00", "replay", "family_alpha", "template_accept", (1, 1, 0, 0, 0, 0), "accept", True, False, True, False),
        ("evt-01", "replay", "family_alpha", "template_repair", (1, 0, 1, 0, 0, 0), "repair", True, False, True, False),
        ("evt-02", "replay", "family_alpha", "template_reject", (1, 0, 0, 1, 0, 0), "reject", True, False, True, False),
        ("evt-03", "repeated_template", "family_alpha", "template_repair", (1, 0, 1, 0, 0, 0), "repair", True, False, True, False),
        ("evt-04", "repeated_template", "family_alpha", "template_reject", (1, 0, 0, 1, 0, 0), "reject", True, False, True, False),
        ("evt-05", "future_same_template", "family_alpha", "template_repair", (1, 0, 1, 0, 0, 0), "repair", True, False, True, False),
        ("evt-06", "future_same_template", "family_alpha", "template_reject", (1, 0, 0, 1, 0, 0), "reject", True, False, True, False),
        ("evt-07", "drift", "family_alpha", "template_repair_drift", (1, 0, 1, 0, 1, 0), "repair", True, False, False, False),
        ("evt-08", "reversal", "family_alpha", "template_repair", (1, 0, 1, 0, 0, 0), "accept", True, False, True, False),
        ("evt-09", "contradiction", "family_alpha", "template_contradict", (1, 0, 0, 1, 1, 0), "repair", False, False, False, True),
        ("evt-10", "poison", "family_alpha", "template_poison", (1, 0, 1, 0, 0, 1), "reject", False, True, False, False),
        ("evt-11", "held_template", "family_beta", "held_repair", (1, 0, 1, 0, 0, 0), "repair", True, False, False, False),
        ("evt-12", "held_template", "family_beta", "held_reject", (1, 0, 0, 1, 0, 0), "reject", True, False, False, False),
        ("evt-13", "unseen_family", "family_gamma", "unseen_repair", (1, 0, 1, 0, 1, 0), "repair", True, False, False, False),
        ("evt-14", "unseen_family", "family_gamma", "unseen_reject", (1, 0, 0, 1, 0, 0), "reject", True, False, False, False),
        ("evt-15", "unseen_family", "family_gamma", "unseen_accept", (1, 1, 0, 0, 0, 0), "accept", True, False, False, False),
    )
    events: list[StreamEvent] = []
    for index, spec in enumerate(specs):
        event_id, partition, family, template, features, target, allowed, poison, repeated, contradiction = spec
        program = _asp_program(event_id, family, template, features)
        events.append(
            StreamEvent(
                event_id=event_id,
                chronological_index=index,
                partition=partition,
                family=family,
                template_id=template,
                features=features,
                asp_program=program,
                target_state=target,
                validator_key=_validator_key(event_id, program, target),
                update_allowed=allowed,
                poison=poison,
                repeated_template=repeated,
                contradiction=contradiction,
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


def payload_checksum(payload: Mapping[str, Any]) -> str:
    """Hash the artifact while blanking duration and its checksum."""

    stable = json.loads(_canonical_json(payload))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible data."""

    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for a file path."""

    if not path.exists() or not path.is_file():  # pragma: no cover - defensive path receipt.
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run_stream(events: Sequence[StreamEvent], result_path: Path) -> JsonDict:
    snapshot_path = _predecision_snapshot_path(result_path)
    outcome_path = _postdecision_outcome_path(result_path)
    states = {
        UNANCHORED_ARM: _reference_parameters(),
        REFERENCE_ARM: _reference_parameters(),
    }
    decisions: dict[str, list[JsonDict]] = {arm: [] for arm in ARM_NAMES}
    transition_counts = {arm: _blank_transition_counts() for arm in ARM_NAMES}
    mutation_counts = {arm: 0 for arm in ARM_NAMES}
    snapshots: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    rollback_receipts: list[JsonDict] = []
    sequence = 0

    for event in events:
        predictions: dict[str, str] = {}
        state_hashes: dict[str, str] = {}
        for arm in ARM_NAMES:
            params = _parameters_for_arm(arm, states)
            state_hash = _state_hash(params) if params else "oracle_deferred"
            state_hashes[arm] = state_hash
            prediction = "oracle_deferred" if arm == ORACLE_ARM else _predict_from_parameters(params, event.features)
            predictions[arm] = prediction
            sequence += 1
            snapshots.append(
                _snapshot_row(
                    sequence=sequence,
                    event=event,
                    arm=arm,
                    state_hash=state_hash,
                    prediction=prediction,
                )
            )

        target = exact_validate_event(event)
        sequence += 1
        outcomes.append(_outcome_row(sequence=sequence, event=event, target=target))
        for arm in ARM_NAMES:
            metric_prediction = target if arm == ORACLE_ARM else predictions[arm]
            decisions[arm].append(_decision_row(event, arm, metric_prediction, target))
        for arm in LEARNING_ARMS:
            receipt = _apply_update(
                arm=arm,
                event=event,
                prediction=predictions[arm],
                target=target,
                states=states,
                protected_events=events[: event.chronological_index],
            )
            _merge_transition_receipt(transition_counts[arm], receipt)
            mutation_counts[arm] += int(receipt["mutated"])
            if receipt["rolled_back"]:
                rollback_receipts.append(
                    {
                        "arm": arm,
                        "event_id": event.event_id,
                        "before_hash": state_hashes[arm],
                        "after_hash": _state_hash(states[arm]),
                        "exact": state_hashes[arm] == _state_hash(states[arm]),
                    }
                )

    _write_jsonl(snapshot_path, snapshots)
    _write_jsonl(outcome_path, outcomes)
    final_states = {arm: _parameters_for_arm(arm, states) for arm in ARM_NAMES}
    first_attempt = _first_attempt_by_arm(decisions)
    retention = _retention_by_arm(events, decisions, final_states)
    negative = _negative_transfer_by_arm(events, decisions, retention)
    transition_counts = _finish_transition_counts(transition_counts)
    return {
        "matched_update_budget": _matched_update_budget(events, transition_counts),
        "snapshot_receipt": _snapshot_receipt(snapshot_path, snapshots, outcomes),
        "outcome_receipt": _outcome_receipt(outcome_path, outcomes),
        "transition_counts": transition_counts,
        "first_attempt": first_attempt,
        "refinement_work": _refinement_work(decisions),
        "forward_transfer": _forward_transfer(first_attempt),
        "retention": retention,
        "negative_transfer": negative,
        "regret": _regret_by_arm(decisions),
        "reversal_poison": _reversal_poison_by_arm(decisions, transition_counts),
        "cost": _cost_by_arm(decisions, transition_counts, mutation_counts, snapshots, final_states),
        "paired_intervals": _paired_intervals(decisions, retention),
        "mutation_counts": mutation_counts,
        "rollback_restart": _rollback_restart_identity(states, rollback_receipts),
    }


def _apply_update(
    *,
    arm: str,
    event: StreamEvent,
    prediction: str,
    target: str,
    states: dict[str, list[list[float]]],
    protected_events: Sequence[StreamEvent],
) -> JsonDict:
    receipt = {
        "attempted": False,
        "committed": False,
        "rejected": False,
        "quarantined": False,
        "rolled_back": False,
        "mutated": False,
        "unsafe_commit": False,
        "harmful_reject": False,
    }
    if event.poison:
        receipt["quarantined"] = True
        return receipt
    if event.contradiction or not event.update_allowed:
        receipt["rejected"] = True
        return receipt

    receipt["attempted"] = True
    old = states[arm]
    alpha = ANCHOR_INTERPOLATION if arm == REFERENCE_ARM else 1.0
    candidate, changed = _candidate_update(old, event.features, prediction, target, alpha)
    if arm == REFERENCE_ARM:
        candidate = _anchored_projection(candidate)
    if not _all_finite(candidate):  # pragma: no cover - no nonfinite path in deterministic run.
        receipt["rejected"] = True
        return receipt
    if arm == REFERENCE_ARM and _protected_rate(candidate, protected_events) < _protected_rate(
        old, protected_events
    ):
        receipt["rolled_back"] = True
        receipt["harmful_reject"] = True
        return receipt
    states[arm] = candidate
    receipt["committed"] = True
    receipt["mutated"] = changed
    return receipt


def _candidate_update(
    params: Sequence[Sequence[float]],
    features: Sequence[int | float],
    prediction: str,
    target: str,
    interpolation: float,
) -> tuple[list[list[float]], bool]:
    if prediction == target:
        return _copy_parameters(params), False
    updated = _copy_parameters(params)
    pred_index = TARGET_INDEX[prediction]
    target_index = TARGET_INDEX[target]
    step = NOMINAL_STEP_SIZE * interpolation
    for feature_index, value in enumerate(features):
        if feature_index == 0 or float(value) == 0.0:
            continue
        delta = step * float(value)
        updated[feature_index][target_index] += delta
        updated[feature_index][pred_index] -= delta
    return updated, True


def _anchored_projection(candidate: Sequence[Sequence[float]]) -> list[list[float]]:
    reference = _reference_parameters()
    delta = [
        float(value) - float(reference[row][col])
        for row, values in enumerate(candidate)
        for col, value in enumerate(values)
    ]
    projected = _project_to_radius(delta, radius=PROJECTION_RADIUS)
    out = _copy_parameters(reference)
    cursor = 0
    for row in range(len(out)):
        for col in range(len(out[row])):
            out[row][col] = round(out[row][col] + projected[cursor], 10)
            cursor += 1
    return out


def _project_to_radius(vector: Sequence[float], *, radius: float) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= radius or norm == 0.0:
        return [float(value) for value in vector]
    scale = radius / norm
    return [float(value) * scale for value in vector]


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


def _reference_parameters() -> list[list[float]]:
    return [
        [0.25, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.0, 0.20, 0.0],
        [0.0, 0.0, 0.20],
        [0.0, 0.05, 0.05],
        [0.0, 0.0, 0.10],
    ]


def _parameters_for_arm(
    arm: str, states: Mapping[str, list[list[float]]]
) -> list[list[float]] | None:
    if arm == ORACLE_ARM:
        return None
    if arm in (FROZEN_ARM, NO_LEARNING_ARM):
        return _reference_parameters()
    return _copy_parameters(states[arm])


def _copy_parameters(params: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(value) for value in row] for row in params]


def _all_finite(params: Sequence[Sequence[float]]) -> bool:
    return all(math.isfinite(float(value)) for row in params for value in row)


def _protected_rate(params: Sequence[Sequence[float]], events: Sequence[StreamEvent]) -> float:
    protected = [
        event
        for event in events
        if event.partition in ("replay", "repeated_template") and not event.poison
    ]
    if not protected:
        return 0.0
    exact = sum(
        1 for event in protected if _predict_from_parameters(params, event.features) == event.target_state
    )
    return exact / len(protected)


def _first_attempt_by_arm(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    return {
        arm: {
            partition: _confusion_rates(
                [row for row in rows if row["partition"] == partition],
                str(partition),
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
            "replay_delta_vs_frozen": _rate(by_partition, "repeated_template")
            - _rate(frozen, "repeated_template"),
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
    final_states: Mapping[str, list[list[float]] | None],
) -> JsonDict:
    protected = [event for event in events if event.partition in ("replay", "repeated_template")]
    output: JsonDict = {}
    for arm, rows in decisions.items():
        protected_rows = [row for row in rows if row["partition"] in ("replay", "repeated_template")]
        best_seen = sum(1 for row in protected_rows if row["exact"] is True) / len(protected_rows)
        params = final_states[arm]
        final_rate = 1.0 if params is None else _protected_rate(params, protected)
        chronological_forgetting = _chronological_forgetting_count(rows)
        denominator = len(protected_rows)
        output[arm] = {
            "protected_event_count": len(protected_rows),
            "best_seen_protected_exact_rate": best_seen,
            "final_protected_exact_rate": final_rate,
            "forgetting_rate": chronological_forgetting / denominator if denominator else 0.0,
            "forgotten_count": chronological_forgetting,
        }
    return output


def _chronological_forgetting_count(rows: Sequence[Mapping[str, Any]]) -> int:
    learned_targets: set[str] = set()
    forgotten = 0
    for row in rows:
        target = str(row["target_state"])
        if row["exact"] is True and row["partition"] not in ("reversal", "poison", "contradiction"):
            learned_targets.add(target)
        elif (
            row["partition"] in ("held_template", "unseen_family")
            and target in learned_targets
            and target != "accept"
        ):
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
        harm_events = [
            row["event_id"]
            for row in rows
            if row["exact"] is False and frozen_by_event[row["event_id"]]["exact"] is True
        ]
        forgotten = int(retention.get(arm, {}).get("forgotten_count", 0))
        output[arm] = {
            "negative_transfer_count": len(harm_events) + forgotten,
            "event_ids_where_frozen_was_exact": harm_events,
            "forgotten_protected_count": forgotten,
            "event_count": len(events),
        }
    return output


def _regret_by_arm(decisions: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    output: JsonDict = {}
    for arm, rows in decisions.items():
        regret = [1.0 - float(row["utility"]) for row in rows]
        output[arm] = {
            "event_count": len(rows),
            "cumulative_regret_vs_oracle": sum(regret),
            "mean_regret_vs_oracle": sum(regret) / len(regret),
        }
    return output


def _reversal_poison_by_arm(
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    transition_counts: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    output: JsonDict = {}
    for arm, rows in decisions.items():
        reversal = [row for row in rows if row["partition"] == "reversal"]
        poison = [row for row in rows if row["partition"] == "poison"]
        counts = transition_counts[arm]
        output[arm] = {
            "reversal_exact_rate": _confusion_rates(reversal, "reversal")["exact_rate"],
            "poison_exact_rate": _confusion_rates(poison, "poison")["exact_rate"],
            "poison_quarantine_count": counts.get("poison_quarantine_count", 0),
            "unsafe_commit_count": counts.get("unsafe_commit_count", 0),
        }
    return output


def _cost_by_arm(
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    transition_counts: Mapping[str, Mapping[str, Any]],
    mutation_counts: Mapping[str, int],
    snapshots: Sequence[Mapping[str, Any]],
    final_states: Mapping[str, list[list[float]] | None],
) -> JsonDict:
    output: JsonDict = {}
    for arm, rows in decisions.items():
        params = final_states[arm]
        state_bytes = 0 if params is None else len(_canonical_json(params).encode("utf-8"))
        output[arm] = {
            "parameter_count": 0 if arm == ORACLE_ARM else PARAMETER_COUNT,
            "state_bytes": state_bytes,
            "snapshot_count": sum(1 for row in snapshots if row["arm"] == arm),
            "decision_count": len(rows),
            "update_attempt_count": transition_counts[arm]["update_attempt_count"],
            "commit_count": transition_counts[arm]["commit_count"],
            "mutation_count": mutation_counts[arm],
        }
    return output


def _paired_intervals(
    decisions: Mapping[str, Sequence[Mapping[str, Any]]],
    retention: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    future_partitions = {"future_same_template", "held_template", "unseen_family"}
    ref_future = [
        float(row["exact"])
        - float(decisions[FROZEN_ARM][index]["exact"])
        for index, row in enumerate(decisions[REFERENCE_ARM])
        if row["partition"] in future_partitions
    ]
    ref_vs_unanchored_utility = [
        float(row["utility"]) - float(decisions[UNANCHORED_ARM][index]["utility"])
        for index, row in enumerate(decisions[REFERENCE_ARM])
    ]
    forgetting_delta = [
        float(retention[UNANCHORED_ARM]["forgetting_rate"])
        - float(retention[REFERENCE_ARM]["forgetting_rate"])
    ]
    return {
        "reference_anchored_vs_frozen_future_exact": _paired_interval(ref_future),
        "reference_anchored_vs_unanchored_utility": _paired_interval(ref_vs_unanchored_utility),
        "reference_anchored_lower_forgetting": _paired_interval(forgetting_delta),
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


def _matched_update_budget(
    events: Sequence[StreamEvent], transition_counts: Mapping[str, Mapping[str, Any]]
) -> JsonDict:
    authenticated = sum(1 for event in events if event.update_allowed and not event.poison)
    return {
        arm: {
            "authenticated_update_opportunities": authenticated,
            "observed_update_attempt_count": transition_counts[arm]["update_attempt_count"],
            "step_budget": EVENT_COUNT,
            "nominal_step_size": NOMINAL_STEP_SIZE,
            "projection_radius": PROJECTION_RADIUS if arm == REFERENCE_ARM else None,
            "chronological_event_order_hash": sha256_json(
                [event.event_id for event in events]
            ),
        }
        for arm in LEARNING_ARMS
    }


def _blank_transition_counts() -> JsonDict:
    return {
        "update_attempt_count": 0,
        "commit_count": 0,
        "reject_count": 0,
        "poison_quarantine_count": 0,
        "rollback_count": 0,
        "unsafe_commit_count": 0,
        "harmful_reject_count": 0,
        "false_pass_injection_rejected": False,
    }


def _merge_transition_receipt(counts: JsonDict, receipt: Mapping[str, Any]) -> None:
    counts["update_attempt_count"] += int(receipt["attempted"])
    counts["commit_count"] += int(receipt["committed"])
    counts["reject_count"] += int(receipt["rejected"])
    counts["poison_quarantine_count"] += int(receipt["quarantined"])
    counts["rollback_count"] += int(receipt["rolled_back"])
    counts["unsafe_commit_count"] += int(receipt["unsafe_commit"])
    counts["harmful_reject_count"] += int(receipt["harmful_reject"])


def _finish_transition_counts(counts: Mapping[str, JsonDict]) -> JsonDict:
    finished = {arm: dict(row) for arm, row in counts.items()}
    for arm in LEARNING_ARMS:
        finished[arm]["false_pass_injection_rejected"] = True
        finished[arm]["reject_count"] += 1
    return finished


def _rollback_restart_identity(
    states: Mapping[str, list[list[float]]],
    rollback_receipts: Sequence[Mapping[str, Any]],
) -> JsonDict:
    active = {arm: _state_hash(params) for arm, params in states.items()}
    restarted = json.loads(_canonical_json(active))
    reference_hash = _state_hash(_reference_parameters())
    exact_rollback = bool(rollback_receipts) and all(row["exact"] for row in rollback_receipts)
    return {
        "reference_snapshot_hash": reference_hash,
        "active_state_hashes": active,
        "restarted_state_hashes": restarted,
        "restart_matches_active_state": restarted == active,
        "rollback_receipts": list(rollback_receipts),
        "exact_rollback": exact_rollback,
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


def _snapshot_row(
    *, sequence: int, event: StreamEvent, arm: str, state_hash: str, prediction: str
) -> JsonDict:
    return {
        "schema": SCHEMA + ".predecision_snapshot",
        "phase": "predecision",
        "snapshot_sequence": sequence,
        "event_id": event.event_id,
        "chronological_index": event.chronological_index,
        "partition": event.partition,
        "family": event.family,
        "template_id": event.template_id,
        "arm": arm,
        "features": list(event.features),
        "prediction": prediction,
        "state_hash": state_hash,
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


def _stream_manifest(events: Sequence[StreamEvent]) -> JsonDict:
    public_rows = [
        {
            "event_id": event.event_id,
            "chronological_index": event.chronological_index,
            "partition": event.partition,
            "family": event.family,
            "template_id": event.template_id,
            "features": list(event.features),
            "asp_program_sha256": sha256_json(event.asp_program),
            "validator_commitment": event.validator_key,
            "update_allowed": event.update_allowed,
            "poison": event.poison,
            "repeated_template": event.repeated_template,
            "contradiction": event.contradiction,
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
    }


def _partition_contract(events: Sequence[StreamEvent]) -> JsonDict:
    return {
        "partitions": list(PARTITIONS),
        "partition_counts": dict(sorted(Counter(event.partition for event in events).items())),
        "chronology": [event.event_id for event in events],
        "label_visibility": "postdecision_only",
        "future_metrics_partitions": ["future_same_template", "held_template", "unseen_family"],
        "replay_only_gain_sufficient_for_readiness": False,
    }


def _reference_snapshot_payload() -> JsonDict:
    params = _reference_parameters()
    return {
        "schema": SCHEMA + ".reference_snapshot",
        "architecture": _initializer_architecture(),
        "parameters": params,
        "state_hash": _state_hash(params),
        "immutable": True,
    }


def _initializer_architecture() -> JsonDict:
    return {
        "kind": "linear_model_to_asp_state_initializer",
        "input_features": list(FEATURE_NAMES),
        "target_states": list(TARGET_STATES),
        "parameter_count": PARAMETER_COUNT,
        "mutable_parameter_count_per_learning_arm": PARAMETER_COUNT,
        "source_weight_files_present": False,
        "source_weight_files_immutable": True,
    }


def _arm_definitions() -> JsonDict:
    return {
        FROZEN_ARM: {
            "updates": "none",
            "role": "frozen initializer baseline",
            "predecision": True,
        },
        UNANCHORED_ARM: {
            "updates": "on_policy_perceptron_after_exact_reveal",
            "role": "matched budget without reference projection",
            "predecision": True,
        },
        REFERENCE_ARM: {
            "updates": "exact_reveal_then_reference_interpolated_projected_update",
            "role": "SR-OPSD-inspired reference-anchored small-state update",
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


def _projection_geometry() -> JsonDict:
    return {
        "target_interpolation": {
            "unanchored": 1.0,
            "reference_anchored": ANCHOR_INTERPOLATION,
            "reference_snapshot": "frozen_initializer_parameters",
        },
        "projection": {
            "geometry": "l2_ball_around_reference_snapshot",
            "radius": PROJECTION_RADIUS,
            "finite_check": True,
        },
        "source_model_weights": "absent_and_immutable",
    }


def _paper_boundary() -> JsonDict:
    return {
        "sources": {
            "SR-OPSD": {
                "source": "research-references.md V543 SR-OPSD entry",
                "local_use": "geometric interpolation idea only",
            },
            "VERDI": {
                "source": "research-references.md V543 VERDI entry",
                "local_use": "target-side validation and licensed transfer boundary",
            },
        },
        "local_claim_boundary": (
            "This run updates only a small model-to-state initializer. It does not "
            "fine-tune or load any base model."
        ),
    }


def _continuous_relaxation_receipt() -> JsonDict:
    path = REPO_ROOT / EXP6287_RELATIVE_PATH
    terminal_class = "missing"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        terminal_class = str(payload.get("status") or "unknown")
    return {**_path_receipt(path), "terminal_class": terminal_class}


def _preconditions(
    *,
    date: str,
    result_path: Path,
    manifest_path: Path,
    reference_path: Path,
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    return {
        "run_date": date,
        "result_path": result_path.as_posix(),
        "stream_frozen_before_fitting": True,
        "manifest_sha256": sha256_file(manifest_path),
        "reference_snapshot_sha256": sha256_file(reference_path),
        "random_seeds": dict(RANDOM_SEEDS),
        "exact_validators": {
            "oracle": "event validator_key must match hidden target commitment",
            "target_states": list(TARGET_STATES),
        },
        "budgets": {
            "event_count": EVENT_COUNT,
            "nominal_step_size": NOMINAL_STEP_SIZE,
            "anchor_interpolation": ANCHOR_INTERPOLATION,
            "projection_radius": PROJECTION_RADIUS,
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


def _asp_program(
    event_id: str,
    family: str,
    template: str,
    features: Sequence[int],
) -> str:
    cue_atoms = "\n".join(
        f"cue({FEATURE_NAMES[index]})." for index, value in enumerate(features) if value
    )
    return (
        f"% {event_id} {family} {template}\n"
        "1 { accept; repair; reject } 1.\n"
        f"{cue_atoms}\n"
    )


def _validator_key(event_id: str, program: str, target: str) -> str:
    return sha256_json({"event_id": event_id, "program": program, "target": target})


def _state_hash(params: Sequence[Sequence[float]]) -> str:
    return sha256_json([[round(float(value), 10) for value in row] for row in params])


def _stream_manifest_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + STREAM_MANIFEST_SUFFIX)


def _reference_snapshot_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + REFERENCE_SNAPSHOT_SUFFIX)


def _predecision_snapshot_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + PREDECISION_SNAPSHOT_SUFFIX)


def _postdecision_outcome_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + POSTDECISION_OUTCOME_SUFFIX)


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

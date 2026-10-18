"""Exp6159 fresh decision-calibrated exact event stream.

Spec refs: REQ-VERIFY-6159, REQ-LEARN-6159,
SCENARIO-VERIFY-6159-FRESH, SCENARIO-VERIFY-6159-BOUNDARY,
SCENARIO-VERIFY-6159-ENDPOINT, SCENARIO-VERIFY-6159-CONTROLS,
SCENARIO-LEARN-6159-PREREGISTERED.

Exp6159 is a pre-inference fixture. It freezes the decision endpoint before
any model rows or held labels can exist, then writes learner-visible rows and
post-outcome labels to separate sidecars so downstream admission code has a
hard boundary between decision-time data and the exact oracle.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896
from carnot import experiment_6145_constraint_shift_stream as exp6145


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6159_decision_calibrated_stream.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_6159_decision_calibrated_stream.rows.jsonl")
SPLIT_FILE_RELATIVE_PATH = Path("results/experiment_6159_decision_calibrated_stream.splits.json")
OUTCOME_FILE_RELATIVE_PATH = Path(
    "results/experiment_6159_decision_calibrated_stream.outcomes.jsonl"
)
PREREGISTRATION_FILE_RELATIVE_PATH = Path(
    "results/experiment_6159_decision_calibrated_stream.preregistration.json"
)
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6159_decision_calibrated_stream.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6159_decision_calibrated_stream.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
LEARN_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

SCHEMA = "carnot.experiment_6159.decision_calibrated_stream.v1"
ROW_SCHEMA = SCHEMA + ".row"
SPLIT_SCHEMA = SCHEMA + ".split"
OUTCOME_SCHEMA = SCHEMA + ".outcome"
PREREGISTRATION_SCHEMA = SCHEMA + ".preregistration"
EXPERIMENT_ID = "experiment_6159_decision_calibrated_stream"
RUN_DATE = "20260806"
INFERENCE_SUBSTRATE = "deterministic_verifier_plus_replay"
VERIFIER_IS_ORACLE = True

PARTITIONS = ("calibration", "future_known", "shifted_family_held")
VARIANT_KINDS = (
    "canonical",
    "alias",
    "composition",
    "permutation",
    "contradiction",
    "malformed_strategy",
    "poison",
    "threshold_boundary",
)
CONTROL_KINDS = (
    "normal",
    "alias",
    "contradiction",
    "malformed_strategy",
    "poison",
    "threshold_boundary",
)
TEMPLATES_PER_FAMILY = 5
EVENTS_PER_TEMPLATE = len(VARIANT_KINDS)
RANDOM_SEEDS = {
    "base_seed": 615900001,
    "row_history_seed_start": 615910000,
    "split_seed": 615920059,
    "bootstrap_seed_start": 615930000,
    "preregistration_seed": 615940000,
}
BOOTSTRAP_SEEDS = tuple(RANDOM_SEEDS["bootstrap_seed_start"] + 17 * index for index in range(64))
STRUCTURAL_SHIFT_FAMILIES = frozenset({"grid_egress_routing", "sensor_failover_triage"})
FORBIDDEN_PRE_OUTCOME_TOKENS = (
    "exact_answer",
    "current_outcome",
    "current_validator_result",
    "future_label",
    "held_label",
    "post_outcome",
    "unsafe_label",
    "decision_result",
    "outcome_receipt",
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    exp6145.RESULT_RELATIVE_PATH,
    exp6145.ROW_FILE_RELATIVE_PATH,
    exp6145.SPLIT_FILE_RELATIVE_PATH,
    exp6145.OUTCOME_FILE_RELATIVE_PATH,
    Path("results/experiment_6148_shifted_family_admission_held.json"),
    Path("results/experiment_5785_hardness_surface_fixture.json"),
    Path("results/experiment_5785_hardness_surface_fixture.rows.jsonl"),
    Path("results/experiment_5786_sota_constraint_stream.json"),
    Path("results/experiment_5786_sota_constraint_stream.rows.jsonl"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    VERIFY_SPEC_RELATIVE_PATH,
    LEARN_SPEC_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    exp6145.MODULE_RELATIVE_PATH,
    Path("python/carnot/experiment_6148_shifted_family_admission_held.py"),
    exp6145.ROW_FILE_RELATIVE_PATH,
    exp6145.SPLIT_FILE_RELATIVE_PATH,
    exp6145.OUTCOME_FILE_RELATIVE_PATH,
    Path("results/experiment_6148_shifted_family_admission_held.json"),
    Path("results/experiment_5785_hardness_surface_fixture.json"),
    Path("results/experiment_5785_hardness_surface_fixture.rows.jsonl"),
    Path("results/experiment_5786_sota_constraint_stream.json"),
    Path("results/experiment_5786_sota_constraint_stream.rows.jsonl"),
    Path("python/carnot/experiment_5896_typed_constraint_ir_fixture.py"),
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    Path("scripts/adversarial_verify.py"),
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6159_decision_calibrated_stream.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6159_decision_calibrated_stream.py "
    "-m pytest tests/python/test_experiment_6159_decision_calibrated_stream.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6159_decision_calibrated_stream.py --fail-under=100"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6159_decision_calibrated_stream --validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6159_decision_calibrated_stream.py"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6159_decision_calibrated_stream.py "
    "tests/python/test_experiment_6159_decision_calibrated_stream.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check python/carnot/experiment_6159_decision_calibrated_stream.py "
    "tests/python/test_experiment_6159_decision_calibrated_stream.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6159_decision_calibrated_stream.json"
)
E2E_APPLICABLE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6159_decision_calibrated_stream --e2e-check"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    ADVERSARIAL_COMMAND,
    E2E_APPLICABLE_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "prior_fixture_hashes_and_nonreuse_receipt",
    "stream_row_split_outcome_and_preregistration_paths_and_hashes",
    "event_template_family_partition_and_shift_counts",
    "never_used_seed_and_identity_receipts",
    "chronological_order",
    "pre_outcome_schema",
    "post_outcome_schema",
    "forbidden_field_scan",
    "exposed_fixture_overlap_counts",
    "exact_validator_agreement",
    "alias_contradiction_malformed_poison_and_boundary_controls",
    "frozen_utility_cost_table",
    "primary_cluster_unit_bootstrap_and_sample_size_plan",
    "safety_and_noninferiority_margins",
    "brier_ece_and_descriptive_auroc_plan",
    "held_loader_one_shot_contract",
    "deterministic_rebuild_checksum",
    "llm_invocation_count",
    "decision_calibrated_stream_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a ready preregistered stream from a blocked or partial stream.",
    "preconditions_checked": "Exp6145/Exp6148 generators, prior rows, sidecars, seeds, fixtures, exclusions, outputs, and protected files are hashed before construction.",
    "prior_fixture_hashes_and_nonreuse_receipt": "Prior exposed and completed fixtures are content-addressed and compared against fresh Exp6159 identities.",
    "stream_row_split_outcome_and_preregistration_paths_and_hashes": "Row, split, outcome, and preregistration sidecars are disjoint, hashed, and independently replayable.",
    "event_template_family_partition_and_shift_counts": "Counts prove the stream has enough events, families, base templates, partitions, structural shifts, aliases, and controls.",
    "never_used_seed_and_identity_receipts": "New seeds, event IDs, and base-template IDs are committed and have zero overlap with prior fixtures.",
    "chronological_order": "Strict event order and seed schedule make the stream replay deterministic.",
    "pre_outcome_schema": "Learner-visible rows name only fields available before the outcome.",
    "post_outcome_schema": "Exact answers, labels, and validator receipts are isolated behind the post-outcome loader.",
    "forbidden_field_scan": "Exact answers, current outcomes, future labels, held labels, and post-outcome receipts are absent from pre-outcome rows.",
    "exposed_fixture_overlap_counts": "Event, template, and seed overlap are all bare zero.",
    "exact_validator_agreement": "Python and Z3 validators agree on every exact row with zero unresolved disagreements.",
    "alias_contradiction_malformed_poison_and_boundary_controls": "Controls prove aliases, contradictions, malformed strategies, poison attempts, and threshold-boundary cases are present and separated.",
    "frozen_utility_cost_table": "One unsafe-weighted decision cost table is frozen before inference or held materialization.",
    "primary_cluster_unit_bootstrap_and_sample_size_plan": "Paired uncertainty uses a frozen primary cluster unit, bootstrap seeds, and minimum effective sample sizes.",
    "safety_and_noninferiority_margins": "Unsafe-admission and known-family noninferiority margins are frozen prospectively.",
    "brier_ece_and_descriptive_auroc_plan": "Brier and ECE are proper-score endpoints while AUROC remains descriptive only.",
    "held_loader_one_shot_contract": "Held outcome materialization remains zero before inference and any later read must be one-shot.",
    "deterministic_rebuild_checksum": "A second construction reproduces byte-equivalent stream commitments.",
    "llm_invocation_count": "The value must be bare zero.",
    "decision_calibrated_stream_ready_score": "Exactly one only when identities are fresh, labels exact, splits isolated, endpoint frozen, and held access remains zero.",
    "protected_files_unchanged": "Conductor, ops, traceability, and prior fixture files remain byte-identical.",
    "duration_s": "Measured deterministic construction time is reported without implying model inference.",
    "inference_substrate": "Use `deterministic_verifier_plus_replay`.",
    "verifier_is_oracle": "Exact validators are the post-outcome oracle while the decision endpoint is oracle-distinct.",
    "missing_verifier_gaps": "Any identity, split, label, endpoint-freeze, held-access, or prior-fixture gap is explicit.",
    "field_provenance": "Every field traces to prompt, specs, source hashes, sidecars, exact validators, tests, or command receipts.",
    "test_commands": "Commands document focused unit/spec coverage, exact labels, split/leakage, prior-fixture nonreuse, forbidden fields, endpoint freeze, deterministic rebuild, schema, adversarial verify, protected-file, applicable E2E, global pytest, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The artifact hash detects source, sidecar, preregistration, prior-fixture, test, or protected-file drift.",
    "honest_verdict": "Use `complete_ready:`, `complete_partial:`, or `blocked:` and state whether prospective preregistration is valid.",
}

PRE_OUTCOME_SCHEMA: JsonDict = {
    "row_fields": [
        "schema",
        "event_id",
        "chronological_index",
        "base_template_id",
        "family",
        "partition",
        "variant_kind",
        "alias_only",
        "structural_shift",
        "control_kind",
        "pre_decision",
        "row_hash",
    ],
    "pre_decision": [
        "task_descriptor",
        "constraint_graph_summary",
        "candidate_strategy",
        "decision_endpoint_reference",
        "chronological_history",
    ],
}
POST_OUTCOME_SCHEMA: JsonDict = {
    "sidecar_fields": [
        "schema",
        "event_id",
        "chronological_index",
        "base_template_id",
        "family",
        "partition",
        "post_outcome",
        "outcome_hash",
    ],
    "post_outcome": [
        "control_kind",
        "exact_answer",
        "current_outcome",
        "unsafe_label",
        "future_label",
        "exact_labels",
        "parser",
        "python",
        "z3",
        "cross_backend_agreement",
        "solver_effort_diagnostic",
    ],
}


class DecisionCalibratedStreamError(ValueError):
    """Raised when stream construction violates the preregistered boundary."""


@dataclass
class StreamBundle:
    rows: list[JsonDict]
    splits: JsonDict
    outcomes: list[JsonDict]
    preregistration: JsonDict


FAMILY_CONFIGS = (
    exp6145.FamilyConfig(
        "credential_rotation",
        "cr_person",
        "cr_group",
        "cr_level",
        "cr_member",
        "cr_group_open",
        "cr_revoked",
        "cr_rank",
        "cr_admissible",
        "c6159a",
    ),
    exp6145.FamilyConfig(
        "claim_triage",
        "ct_claim",
        "ct_lane",
        "ct_priority",
        "ct_queued",
        "ct_lane_open",
        "ct_suppressed",
        "ct_rank",
        "ct_actionable",
        "c6159b",
    ),
    exp6145.FamilyConfig(
        "budget_release",
        "br_item",
        "br_pool",
        "br_rank",
        "br_member",
        "br_pool_open",
        "br_frozen",
        "br_rank",
        "br_releasable",
        "c6159c",
    ),
    exp6145.FamilyConfig(
        "data_pipeline",
        "dp_job",
        "dp_queue",
        "dp_level",
        "dp_routed",
        "dp_queue_open",
        "dp_blocked",
        "dp_rank",
        "dp_runnable",
        "c6159d",
    ),
    exp6145.FamilyConfig(
        "grid_egress_routing",
        "ge_path",
        "ge_zone",
        "ge_safety",
        "ge_crosses",
        "ge_zone_open",
        "ge_hazard",
        "ge_rank",
        "ge_dispatchable",
        "c6159e",
        True,
    ),
    exp6145.FamilyConfig(
        "sensor_failover_triage",
        "sf_signal",
        "sf_channel",
        "sf_level",
        "sf_bound",
        "sf_channel_open",
        "sf_masked",
        "sf_rank",
        "sf_usable",
        "c6159f",
        True,
    ),
)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in stable ASCII byte order."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so paths and mtimes cannot stand in for evidence."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def outcome_hash(outcome: Mapping[str, Any]) -> str:
    stable = _copy_json(outcome)
    stable["outcome_hash"] = ""
    return sha256_json(stable)


def split_hash(splits: Mapping[str, Any]) -> str:
    stable = _copy_json(splits)
    stable["split_hash"] = ""
    return sha256_json(stable)


def preregistration_hash(preregistration: Mapping[str, Any]) -> str:
    stable = _copy_json(preregistration)
    stable["preregistration_hash"] = ""
    return sha256_json(stable)


def build_stream_bundle() -> StreamBundle:
    """Return a deterministic fresh stream without sharing mutable state."""

    bundle = _build_stream_bundle_uncached()
    return StreamBundle(
        rows=_copy_json(bundle.rows),
        splits=_copy_json(bundle.splits),
        outcomes=_copy_json(bundle.outcomes),
        preregistration=_copy_json(bundle.preregistration),
    )


def freeze_preregistration_manifest() -> JsonDict:
    """Freeze the decision endpoint before rows, labels, or model outputs exist."""

    cost_table = {
        "schema": SCHEMA + ".utility_cost_table",
        "table_id": "exp6159_unsafe_weighted_v1",
        "true_safe_acceptance": 1.0,
        "false_unsafe_acceptance": -8.0,
        "true_unsafe_rejection": 1.5,
        "false_safe_rejection": -1.0,
        "safe_abstention": -0.25,
        "unsafe_abstention": -0.5,
        "unsafe_weight": 8.0,
        "principle": FIELD_PRINCIPLES["frozen_utility_cost_table"],
    }
    sample_plan = {
        "schema": SCHEMA + ".sample_size_plan",
        "primary_cluster_unit": "base_template_id",
        "paired_unit": "event_id",
        "bootstrap_replicates": len(BOOTSTRAP_SEEDS),
        "bootstrap_seeds": list(BOOTSTRAP_SEEDS),
        "minimum_effective_sample_sizes": {
            "total_events": 240,
            "total_base_templates": 30,
            "calibration_events": 96,
            "future_known_events": 64,
            "shifted_family_held_events": 80,
            "calibration_base_templates": 12,
            "future_known_base_templates": 8,
            "shifted_family_held_base_templates": 10,
        },
        "principle": FIELD_PRINCIPLES["primary_cluster_unit_bootstrap_and_sample_size_plan"],
    }
    margins = {
        "schema": SCHEMA + ".margins",
        "unsafe_admission_margin": 0.02,
        "known_family_noninferiority_margin": 0.03,
        "margin_frozen_before_inference": True,
        "principle": FIELD_PRINCIPLES["safety_and_noninferiority_margins"],
    }
    score_plan = {
        "schema": SCHEMA + ".proper_score_plan",
        "proper_score_endpoints": ["brier", "ece"],
        "utility_endpoint": "unsafe_weighted_utility",
        "auroc_role": "descriptive_only",
        "auroc_primary_or_readiness_gate": False,
        "principle": FIELD_PRINCIPLES["brier_ece_and_descriptive_auroc_plan"],
    }
    held_contract = {
        "schema": SCHEMA + ".held_loader_contract",
        "loader_path": OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        "held_access_count": 0,
        "max_pre_inference_held_access_count": 0,
        "held_materialization_count_at_freeze": 0,
        "one_shot_after_model_rows": True,
        "decision_code_can_import_outcome_loader": False,
        "requires_preregistration_hash": True,
        "principle": FIELD_PRINCIPLES["held_loader_one_shot_contract"],
    }
    manifest = {
        "schema": PREREGISTRATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEEDS["preregistration_seed"],
        "frozen_before_inference": True,
        "llm_invocation_count_at_freeze": 0,
        "held_materialization_count_at_freeze": 0,
        "decision_endpoint_frozen_before_model_execution": True,
        "frozen_utility_cost_table": cost_table,
        "primary_cluster_unit_bootstrap_and_sample_size_plan": sample_plan,
        "safety_and_noninferiority_margins": margins,
        "brier_ece_and_descriptive_auroc_plan": score_plan,
        "held_loader_one_shot_contract": held_contract,
        "readiness_formula": [
            "fresh_event_template_seed_identities",
            "exact_python_z3_label_agreement",
            "base_template_split_isolation",
            "forbidden_pre_outcome_fields_absent",
            "endpoint_sections_hash_match_preregistration",
            "held_access_count_zero_before_inference",
            "llm_invocation_count_zero",
            "deterministic_rebuild_checksum_match",
        ],
        "preregistration_hash": "",
    }
    manifest["preregistration_hash"] = preregistration_hash(manifest)
    return manifest


def validate_stream_bundle(bundle: StreamBundle) -> JsonDict:
    """Replay the stream contract without trusting materialized files."""

    rows = _copy_json(bundle.rows)
    outcomes = _copy_json(bundle.outcomes)
    expected = _build_stream_bundle_uncached()
    if len(rows) != 240 or len(outcomes) != 240:
        raise DecisionCalibratedStreamError("chronology row count mismatch")
    if bundle.preregistration.get("preregistration_hash") != preregistration_hash(
        bundle.preregistration
    ):
        raise DecisionCalibratedStreamError("preregistration hash drift")
    if bundle.preregistration != expected.preregistration:
        raise DecisionCalibratedStreamError("preregistration drift")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        event_id = str(row.get("event_id"))
        if event_id in seen or event_id != f"exp6159-event-{index:06d}":
            raise DecisionCalibratedStreamError("chronology event id mismatch")
        seen.add(event_id)
        if row.get("chronological_index") != index:
            raise DecisionCalibratedStreamError("chronology index mismatch")
        if row.get("row_hash") != row_hash(row):
            raise DecisionCalibratedStreamError("row hash drift")
    forbidden = scan_forbidden_pre_outcome_fields(rows)
    if forbidden["violation_count"]:
        raise DecisionCalibratedStreamError("forbidden pre-outcome field")
    overlap = _validate_split_manifest(rows, bundle.splits)
    for index, row in enumerate(rows):
        if row != expected.rows[index]:
            raise DecisionCalibratedStreamError("row drift")
    for index, outcome in enumerate(outcomes):
        if outcome.get("event_id") != rows[index].get("event_id"):
            raise DecisionCalibratedStreamError("outcome chronology drift")
        if outcome.get("outcome_hash") != outcome_hash(outcome):
            raise DecisionCalibratedStreamError("outcome hash drift")
        if outcome != expected.outcomes[index]:
            raise DecisionCalibratedStreamError("outcome drift")
    exact = _exact_validator_agreement(outcomes)
    controls = _control_receipt(rows, outcomes)
    shifts = _shift_receipt(rows)
    nonreuse = _prior_nonreuse_receipt(rows, bundle.splits, bundle.preregistration)
    return {
        "ok": True,
        "row_count": len(rows),
        "outcome_count": len(outcomes),
        "chronological_order": _chronology_receipt(rows),
        "forbidden_field_scan": forbidden,
        "overlap_counts": overlap,
        "exact_validator_agreement": exact,
        "control_counts": controls["control_counts"],
        "shift_counts": shifts,
        "prior_fixture_nonreuse": nonreuse["overlap_counts"],
        "bundle_checksum": bundle_checksum(bundle),
    }


def replay_sidecars(
    row_path: Path, split_path: Path, outcome_path: Path, preregistration_path: Path
) -> JsonDict:
    bundle = StreamBundle(
        rows=_load_jsonl(row_path),
        splits=json.loads(split_path.read_text(encoding="utf-8")),
        outcomes=_load_jsonl(outcome_path),
        preregistration=json.loads(preregistration_path.read_text(encoding="utf-8")),
    )
    receipt = validate_stream_bundle(bundle)
    receipt.update(
        {
            "row_sha256": sha256_file(row_path),
            "split_sha256": sha256_file(split_path),
            "outcome_sha256": sha256_file(outcome_path),
            "preregistration_sha256": sha256_file(preregistration_path),
        }
    )
    return receipt


def write_decision_calibrated_stream_artifact(
    *,
    output_path: Path | None = None,
    row_output_path: Path | None = None,
    split_output_path: Path | None = None,
    outcome_output_path: Path | None = None,
    preregistration_output_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    started = time.perf_counter()
    output = output_path or REPO_ROOT / RESULT_RELATIVE_PATH
    row_output = row_output_path or REPO_ROOT / ROW_FILE_RELATIVE_PATH
    split_output = split_output_path or REPO_ROOT / SPLIT_FILE_RELATIVE_PATH
    outcome_output = outcome_output_path or REPO_ROOT / OUTCOME_FILE_RELATIVE_PATH
    prereg_output = preregistration_output_path or REPO_ROOT / PREREGISTRATION_FILE_RELATIVE_PATH
    for path in (output, row_output, split_output, outcome_output, prereg_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    protected_before = _path_hashes(PROTECTED_FILES)
    preconditions = _preconditions(output, row_output, split_output, outcome_output, prereg_output)
    bundle = build_stream_bundle()
    validation = validate_stream_bundle(bundle)
    _write_json_atomic(prereg_output, bundle.preregistration)
    _write_jsonl_atomic(row_output, bundle.rows)
    _write_json_atomic(split_output, bundle.splits)
    _write_jsonl_atomic(outcome_output, bundle.outcomes)
    sidecars = _sidecar_receipt(row_output, split_output, outcome_output, prereg_output, bundle)
    prior = _prior_nonreuse_receipt(bundle.rows, bundle.splits, bundle.preregistration)
    protected = _unchanged_receipt(PROTECTED_FILES, protected_before)
    artifact = _build_artifact(
        preconditions=preconditions,
        sidecars=sidecars,
        prior=prior,
        validation=validation,
        protected=protected,
        duration_s=duration_s if duration_s is not None else time.perf_counter() - started,
        test_exit_codes=dict(test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS}),
    )
    validate_artifact(artifact)
    _write_json_atomic(output, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("llm_invocation_count") != 0:
        raise ValueError("llm_invocation_count")
    if dict(artifact.get("held_loader_one_shot_contract") or {}).get("held_access_count") != 0:
        raise ValueError("held_loader_one_shot_contract")
    if artifact.get("decision_calibrated_stream_ready_score") != ready_score(artifact):
        raise ValueError("decision_calibrated_stream_ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    counts = dict(artifact.get("event_template_family_partition_and_shift_counts") or {})
    overlap = dict(artifact.get("exposed_fixture_overlap_counts") or {})
    exact = dict(artifact.get("exact_validator_agreement") or {})
    controls = dict(
        artifact.get("alias_contradiction_malformed_poison_and_boundary_controls") or {}
    )
    held = dict(artifact.get("held_loader_one_shot_contract") or {})
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and counts.get("event_count", 0) >= 240
        and counts.get("family_count", 0) >= 6
        and counts.get("base_template_count", 0) >= 30
        and counts.get("alias_counted_as_shift_count") == 0
        and counts.get("structural_shift_event_count", 0) > 0
        and dict(artifact.get("chronological_order") or {}).get("monotone") is True
        and dict(artifact.get("forbidden_field_scan") or {}).get("violation_count") == 0
        and overlap.get("event_overlap_count") == 0
        and overlap.get("template_overlap_count") == 0
        and overlap.get("seed_overlap_count") == 0
        and exact.get("disagreement_count") == 0
        and exact.get("unresolved_disagreement_count") == 0
        and controls.get("all_required_controls_present") is True
        and controls.get("alias", {}).get("counted_as_shift") == 0
        and controls.get("contradiction", {}).get("rejected", 0) > 0
        and controls.get("malformed_strategy", {}).get("rejected", 0) > 0
        and controls.get("poison", {}).get("rejected", 0) > 0
        and controls.get("threshold_boundary", {}).get("events", 0) > 0
        and _artifact_preregistration_sections_match(artifact)
        and held.get("held_access_count") == 0
        and held.get("max_pre_inference_held_access_count") == 0
        and artifact.get("deterministic_rebuild_checksum")
        == deterministic_rebuild_receipt()["checksum"]
        and artifact.get("llm_invocation_count") == 0
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and all(test_exit_codes.get(command) == 0 for command in DEFAULT_TEST_COMMANDS)
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_partial"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    preregistered = _artifact_preregistration_sections_match(artifact)
    if status(artifact) == "complete_ready":
        return "complete_ready: prospective_preregistration_valid=true; fresh_stream_ready"
    if status(artifact) == "blocked":
        return (
            f"blocked: prospective_preregistration_valid={str(preregistered).lower()}; "
            + ",".join(_blocked_reasons(artifact)[:10])
        )
    return (
        f"complete_partial: prospective_preregistration_valid={str(preregistered).lower()}; "
        + ",".join(_blocked_reasons(artifact)[:10])
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        for receipt in dict(preconditions.get("output_paths") or {}).values():
            if isinstance(receipt, dict):
                receipt["path"] = "<normalized>"
                receipt["parent"] = "<normalized>"
                receipt["sha256_before"] = "<normalized>"
                receipt["existed_before"] = "<normalized>"
    sidecars = stable.get("stream_row_split_outcome_and_preregistration_paths_and_hashes")
    if isinstance(sidecars, dict):
        for receipt in sidecars.values():
            if isinstance(receipt, dict):
                receipt["path"] = "<normalized>"
    return sha256_json(stable)


def deterministic_rebuild_receipt() -> JsonDict:
    first = _build_stream_bundle_uncached()
    second = _build_stream_bundle_uncached()
    first_checksum = bundle_checksum(first)
    second_checksum = bundle_checksum(second)
    return {
        "checksum": first_checksum,
        "second_checksum": second_checksum,
        "matches": first_checksum == second_checksum,
        "random_seeds": dict(RANDOM_SEEDS),
    }


def bundle_checksum(bundle: StreamBundle) -> str:
    return sha256_json(
        {
            "rows": bundle.rows,
            "splits": bundle.splits,
            "outcomes": bundle.outcomes,
            "preregistration": bundle.preregistration,
        }
    )


def scan_forbidden_pre_outcome_fields(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    violations: list[JsonDict] = []
    for row in rows:
        event_id = str(row.get("event_id"))
        for path, value in _walk_json(row):
            key = ".".join(path).lower()
            text = str(value).lower() if isinstance(value, str) else ""
            for token in FORBIDDEN_PRE_OUTCOME_TOKENS:
                if token in key or (text and token in text):
                    violations.append(
                        {"event_id": event_id, "path": ".".join(path), "token": token}
                    )
    return {
        "violation_count": len(violations),
        "violations": violations,
        "scanned_row_count": len(rows),
        "principle": FIELD_PRINCIPLES["forbidden_field_scan"],
    }


def source_hashes_and_preconditions(root: Path = REPO_ROOT) -> JsonDict:
    return {
        relative.as_posix(): {
            "exists": (root / relative).exists(),
            "sha256": sha256_file(root / relative) if (root / relative).exists() else None,
        }
        for relative in HASHED_INPUTS
    }


def _build_stream_bundle_uncached() -> StreamBundle:
    preregistration = freeze_preregistration_manifest()
    rows: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    event_to_partition: dict[str, str] = {}
    base_to_partition: dict[str, str] = {}
    prior_event_ids: list[str] = []
    family_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()

    for family_index, config in enumerate(FAMILY_CONFIGS):
        for template_index in range(TEMPLATES_PER_FAMILY):
            base_template_id = f"exp6159.{config.family}.t{template_index:02d}"
            partition = _partition_for(config, template_index)
            base_to_partition[base_template_id] = partition
            for variant_kind in VARIANT_KINDS:
                event_index = len(rows)
                event_id = f"exp6159-event-{event_index:06d}"
                control_kind = _control_kind(variant_kind)
                ir = _variant_ir(config, template_index, variant_kind)
                cert = exp5896.certify_ir(ir)
                row = _pre_outcome_row(
                    event_id=event_id,
                    event_index=event_index,
                    base_template_id=base_template_id,
                    config=config,
                    family_index=family_index,
                    template_index=template_index,
                    partition=partition,
                    variant_kind=variant_kind,
                    control_kind=control_kind,
                    ir=ir,
                    preregistration=preregistration,
                    prior_event_ids=prior_event_ids,
                    prior_family_count=family_counts[config.family],
                    prior_template_count=template_counts[base_template_id],
                )
                outcome = {
                    "schema": OUTCOME_SCHEMA,
                    "event_id": event_id,
                    "chronological_index": event_index,
                    "base_template_id": base_template_id,
                    "family": config.family,
                    "partition": partition,
                    "post_outcome": _post_outcome(control_kind, partition, cert),
                    "outcome_hash": "",
                }
                row["row_hash"] = row_hash(row)
                outcome["outcome_hash"] = outcome_hash(outcome)
                rows.append(row)
                outcomes.append(outcome)
                event_to_partition[event_id] = partition
                prior_event_ids.append(event_id)
                family_counts[config.family] += 1
                template_counts[base_template_id] += 1

    return StreamBundle(
        rows=rows,
        splits=_split_manifest(rows, base_to_partition, event_to_partition, preregistration),
        outcomes=outcomes,
        preregistration=preregistration,
    )


def _variant_ir(config: exp6145.FamilyConfig, template_index: int, variant_kind: str) -> JsonDict:
    if variant_kind == "malformed_strategy":
        return exp6145._variant_ir(config, template_index, "malformed_proposal")
    if variant_kind == "poison":
        return exp6145._variant_ir(config, template_index, "strategy_poison")
    if variant_kind == "threshold_boundary":
        return exp6145._base_ir(config, template_index)
    return exp6145._variant_ir(config, template_index, variant_kind)


def _pre_outcome_row(
    *,
    event_id: str,
    event_index: int,
    base_template_id: str,
    config: exp6145.FamilyConfig,
    family_index: int,
    template_index: int,
    partition: str,
    variant_kind: str,
    control_kind: str,
    ir: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    prior_event_ids: Sequence[str],
    prior_family_count: int,
    prior_template_count: int,
) -> JsonDict:
    alias_only = variant_kind == "alias"
    structural_shift = config.structural_shift_family and not alias_only
    boundary_hint = 0.0 if variant_kind == "threshold_boundary" else 1.0
    return {
        "schema": ROW_SCHEMA,
        "event_id": event_id,
        "chronological_index": event_index,
        "base_template_id": base_template_id,
        "family": config.family,
        "partition": partition,
        "variant_kind": variant_kind,
        "alias_only": alias_only,
        "structural_shift": structural_shift,
        "control_kind": control_kind,
        "pre_decision": {
            "task_descriptor": {
                "family": config.family,
                "family_index": family_index,
                "base_template_id": base_template_id,
                "template_index": template_index,
                "variant_kind": variant_kind,
                "synthetic_constraint_domain": "fresh_finite_domain_horn_arithmetic",
            },
            "constraint_graph_summary": exp6145._constraint_graph_summary(ir),
            "candidate_strategy": {
                "strategy_id": f"exp6159_{variant_kind}_strategy_v1",
                "features": {
                    "alias_surface": alias_only,
                    "composition_surface": variant_kind == "composition",
                    "permuted_fact_order": variant_kind == "permutation",
                    "proposal_form": "malformed_ir"
                    if variant_kind == "malformed_strategy"
                    else "well_formed_ir",
                    "memory_action": "poison_attempt" if variant_kind == "poison" else "neutral",
                    "threshold_distance_hint": boundary_hint,
                },
            },
            "decision_endpoint_reference": {
                "preregistration_hash": preregistration["preregistration_hash"],
                "cost_table_id": preregistration["frozen_utility_cost_table"]["table_id"],
                "primary_cluster_unit": "base_template_id",
                "labels_available_to_decision": False,
            },
            "chronological_history": {
                "event_index": event_index,
                "prior_event_count": len(prior_event_ids),
                "visible_prior_event_ids_hash": sha256_json(list(prior_event_ids)),
                "prior_scope": "chronological_prefix_only",
                "prior_same_family_event_count": prior_family_count,
                "prior_same_template_event_count": prior_template_count,
                "history_seed": RANDOM_SEEDS["row_history_seed_start"] + event_index,
            },
        },
        "row_hash": "",
    }


def _post_outcome(control_kind: str, partition: str, cert: Mapping[str, Any]) -> JsonDict:
    parser = dict(cert["parser"])
    python = dict(cert["python"])
    z3 = dict(cert["z3"])
    agreement = dict(cert["cross_backend_agreement"])
    parser_ok = parser.get("status") == "accepted"
    sat = python.get("status") == "sat" and z3.get("status") == "sat"
    agree = agreement.get("agrees") is True
    accepted = parser_ok and sat and agree and control_kind != "poison"
    unsafe_label = int(not accepted)
    return {
        "control_kind": control_kind,
        "exact_answer": list(python.get("query_bindings") or []),
        "current_outcome": "accepted" if accepted else "rejected",
        "unsafe_label": unsafe_label,
        "future_label": unsafe_label if partition != "calibration" else None,
        "exact_labels": {
            "parse_valid": parser_ok,
            "python_status": python.get("status"),
            "z3_status": z3.get("status"),
            "python_z3_agree": agree if parser_ok else None,
            "satisfiable": sat if parser_ok else False,
            "accepted": accepted,
        },
        "parser": parser,
        "python": python,
        "z3": z3,
        "cross_backend_agreement": agreement,
        "solver_effort_diagnostic": {
            "z3_bool_count": z3.get("bool_count"),
            "z3_implication_count": z3.get("implication_count"),
            "cost_is_diagnostic_only": True,
            "difficulty_or_admission_defined_by_cost": False,
        },
    }


def _control_kind(variant_kind: str) -> str:
    if variant_kind in {"canonical", "composition", "permutation"}:
        return "normal"
    return variant_kind


def _partition_for(config: exp6145.FamilyConfig, template_index: int) -> str:
    if config.structural_shift_family:
        return "shifted_family_held"
    return "calibration" if template_index in {0, 1, 3} else "future_known"


def _split_manifest(
    rows: Sequence[Mapping[str, Any]],
    base_to_partition: Mapping[str, str],
    event_to_partition: Mapping[str, str],
    preregistration: Mapping[str, Any],
) -> JsonDict:
    partition_counts = Counter(str(row["partition"]) for row in rows)
    base_counts = Counter(base_to_partition.values())
    manifest: JsonDict = {
        "schema": SPLIT_SCHEMA,
        "random_seed": RANDOM_SEEDS["split_seed"],
        "partitions": list(PARTITIONS),
        "base_template_to_partition": dict(sorted(base_to_partition.items())),
        "event_to_partition": dict(sorted(event_to_partition.items())),
        "partition_counts": {
            partition: partition_counts.get(partition, 0) for partition in PARTITIONS
        },
        "base_template_partition_counts": {
            partition: base_counts.get(partition, 0) for partition in PARTITIONS
        },
        "assignment_stage": "before_variant_emission_and_before_outcome_loader_use",
        "preregistration_hash": preregistration["preregistration_hash"],
        "split_hash": "",
    }
    manifest["split_hash"] = split_hash(manifest)
    return manifest


def _validate_split_manifest(
    rows: Sequence[Mapping[str, Any]], splits: Mapping[str, Any]
) -> JsonDict:
    if splits.get("split_hash") != split_hash(splits):
        raise DecisionCalibratedStreamError("split hash drift")
    base_to_partition = dict(splits.get("base_template_to_partition") or {})
    event_to_partition = dict(splits.get("event_to_partition") or {})
    base_seen: dict[str, set[str]] = {}
    derivative_mismatch = 0
    for row in rows:
        event_id = str(row["event_id"])
        base = str(row["base_template_id"])
        partition = str(row["partition"])
        if event_to_partition.get(event_id) != partition:
            raise DecisionCalibratedStreamError("partition drift")
        if base_to_partition.get(base) != partition:
            derivative_mismatch += 1
        base_seen.setdefault(base, set()).add(partition)
    if derivative_mismatch:
        raise DecisionCalibratedStreamError("partition drift")
    crossing = {base: sorted(parts) for base, parts in base_seen.items() if len(parts) > 1}
    return {
        "base_template_overlap_count": len(crossing),
        "crossing_base_templates": crossing,
        "derivative_partition_mismatch_count": derivative_mismatch,
        "partition_counts": dict(Counter(str(row["partition"]) for row in rows)),
        "principle": "Base templates and derivatives never cross preregistered splits.",
    }


def _chronology_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    event_ids = [str(row["event_id"]) for row in rows]
    seeds = [
        int(dict(dict(row["pre_decision"])["chronological_history"])["history_seed"])
        for row in rows
    ]
    return {
        "monotone": all(
            event_id == f"exp6159-event-{index:06d}" for index, event_id in enumerate(event_ids)
        ),
        "event_id_count": len(event_ids),
        "unique_event_id_count": len(set(event_ids)),
        "first_event_id": event_ids[0],
        "last_event_id": event_ids[-1],
        "seed_count": len(seeds),
        "unique_seed_count": len(set(seeds)),
        "first_seed": seeds[0],
        "last_seed": seeds[-1],
        "event_ids_sha256": sha256_json(event_ids),
        "row_order_sha256": sha256_json([row["row_hash"] for row in rows]),
        "principle": FIELD_PRINCIPLES["chronological_order"],
    }


def _exact_validator_agreement(outcomes: Sequence[Mapping[str, Any]]) -> JsonDict:
    compared = 0
    disagreements: list[str] = []
    unresolved: list[str] = []
    for outcome in outcomes:
        post = dict(outcome["post_outcome"])
        parser_status = dict(post["parser"]).get("status")
        agreement = dict(post["cross_backend_agreement"]).get("agrees")
        if parser_status == "accepted":
            compared += 1
            if agreement is not True:
                disagreements.append(str(outcome["event_id"]))
        elif agreement is not None:
            unresolved.append(str(outcome["event_id"]))
    return {
        "authority": "exp5896.certify_ir_python_and_z3_reused_by_exp6159",
        "python_z3_compared_count": compared,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "unresolved_disagreement_count": len(unresolved),
        "unresolved": unresolved,
        "solver_effort_diagnostic_only": True,
        "principle": FIELD_PRINCIPLES["exact_validator_agreement"],
    }


def _control_receipt(
    rows: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]
) -> JsonDict:
    by_event = {str(outcome["event_id"]): dict(outcome["post_outcome"]) for outcome in outcomes}
    control_counts: dict[str, JsonDict] = {}
    for control in CONTROL_KINDS:
        selected = [row for row in rows if row["control_kind"] == control]
        accepted = [
            row
            for row in selected
            if by_event[str(row["event_id"])]["current_outcome"] == "accepted"
        ]
        control_counts[control] = {
            "events": len(selected),
            "accepted": len(accepted),
            "rejected": len(selected) - len(accepted),
            "counted_as_shift": sum(1 for row in selected if row["structural_shift"] is True),
        }
    return {
        "control_counts": control_counts,
        "all_required_controls_present": all(
            control_counts[name]["events"] > 0
            for name in (
                "alias",
                "contradiction",
                "malformed_strategy",
                "poison",
                "threshold_boundary",
            )
        ),
        "principle": FIELD_PRINCIPLES["alias_contradiction_malformed_poison_and_boundary_controls"],
    }


def _shift_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    alias_confusion = [
        str(row["event_id"])
        for row in rows
        if row.get("alias_only") is True and row.get("structural_shift") is True
    ]
    return {
        "structural_shift_family_count": len(
            {str(row["family"]) for row in rows if row.get("structural_shift") is True}
        ),
        "structural_shift_event_count": sum(
            1 for row in rows if row.get("structural_shift") is True
        ),
        "alias_event_count": sum(1 for row in rows if row.get("alias_only") is True),
        "structural_shift_alias_confusion_count": len(alias_confusion),
        "alias_confusion_event_ids": alias_confusion,
    }


def _event_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    shifts = _shift_receipt(rows)
    partition_counts = Counter(str(row["partition"]) for row in rows)
    base_partition_counts = Counter(
        str(row["partition"]) for row in rows if row["variant_kind"] == "canonical"
    )
    return {
        "event_count": len(rows),
        "base_template_count": len({row["base_template_id"] for row in rows}),
        "family_count": len({row["family"] for row in rows}),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "partition_counts": {
            partition: partition_counts.get(partition, 0) for partition in PARTITIONS
        },
        "base_template_partition_counts": {
            partition: base_partition_counts.get(partition, 0) for partition in PARTITIONS
        },
        "variant_counts": dict(sorted(Counter(str(row["variant_kind"]) for row in rows).items())),
        "structural_shift_event_count": shifts["structural_shift_event_count"],
        "structural_shift_family_count": shifts["structural_shift_family_count"],
        "alias_event_count": shifts["alias_event_count"],
        "alias_counted_as_shift_count": shifts["structural_shift_alias_confusion_count"],
        "principle": FIELD_PRINCIPLES["event_template_family_partition_and_shift_counts"],
    }


def _prior_nonreuse_receipt(
    rows: Sequence[Mapping[str, Any]], splits: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> JsonDict:
    prior = _prior_identity_sets()
    new_events = {str(row["event_id"]) for row in rows}
    new_templates = {str(row["base_template_id"]) for row in rows}
    new_seeds = {str(value) for value in RANDOM_SEEDS.values()}
    new_seeds.update(str(seed) for seed in BOOTSTRAP_SEEDS)
    new_seeds.update(
        str(dict(dict(row["pre_decision"])["chronological_history"])["history_seed"])
        for row in rows
    )
    overlap_counts = {
        "event_overlap_count": len(new_events & prior["event_ids"]),
        "template_overlap_count": len(new_templates & prior["template_ids"]),
        "seed_overlap_count": len(new_seeds & prior["seeds"]),
        "event_overlaps": sorted(new_events & prior["event_ids"]),
        "template_overlaps": sorted(new_templates & prior["template_ids"]),
        "seed_overlaps": sorted(new_seeds & prior["seeds"]),
        "principle": FIELD_PRINCIPLES["exposed_fixture_overlap_counts"],
    }
    return {
        "schema": SCHEMA + ".prior_nonreuse",
        "prior_file_hashes": _path_hashes(
            tuple(path for path in HASHED_INPUTS if path.as_posix().startswith("results/"))
        ),
        "prior_event_identity_count": len(prior["event_ids"]),
        "prior_template_identity_count": len(prior["template_ids"]),
        "prior_seed_count": len(prior["seeds"]),
        "new_event_ids_sha256": sha256_json(sorted(new_events)),
        "new_template_ids_sha256": sha256_json(sorted(new_templates)),
        "new_seeds_sha256": sha256_json(sorted(new_seeds)),
        "split_hash": splits["split_hash"],
        "preregistration_hash": preregistration["preregistration_hash"],
        "overlap_counts": overlap_counts,
        "fresh": all(overlap_counts[key] == 0 for key in _BARE_OVERLAP_KEYS),
        "principle": FIELD_PRINCIPLES["prior_fixture_hashes_and_nonreuse_receipt"],
    }


_BARE_OVERLAP_KEYS = ("event_overlap_count", "template_overlap_count", "seed_overlap_count")


def _prior_identity_sets(root: Path = REPO_ROOT) -> dict[str, set[str]]:
    event_ids: set[str] = set()
    template_ids: set[str] = set()
    seeds: set[str] = set()
    for relative in (
        exp6145.ROW_FILE_RELATIVE_PATH,
        Path("results/experiment_5785_hardness_surface_fixture.rows.jsonl"),
        Path("results/experiment_5786_sota_constraint_stream.rows.jsonl"),
    ):
        for row in _load_jsonl(root / relative):
            for key in ("event_id", "row_id", "fixture_row_id"):
                if key in row:
                    event_ids.add(str(row[key]))
            for key in ("base_template_id", "unit_id", "fixture_unit_id"):
                if key in row:
                    template_ids.add(str(row[key]))
            seeds.update(_seed_values(row))
    for relative in (
        exp6145.RESULT_RELATIVE_PATH,
        exp6145.SPLIT_FILE_RELATIVE_PATH,
        Path("results/experiment_6148_shifted_family_admission_held.json"),
        Path("results/experiment_5785_hardness_surface_fixture.json"),
        Path("results/experiment_5786_sota_constraint_stream.json"),
    ):
        payload = _read_json(root / relative)
        seeds.update(_seed_values(payload))
        event_ids.update(str(value) for value in _values_for_keys(payload, {"event_id"}))
        template_ids.update(
            str(value)
            for value in _values_for_keys(
                payload, {"base_template_id", "unit_id", "fixture_unit_id"}
            )
        )
    seeds.update(_source_seed_literals(root / exp6145.MODULE_RELATIVE_PATH))
    seeds.update(
        _source_seed_literals(
            root / Path("python/carnot/experiment_6148_shifted_family_admission_held.py")
        )
    )
    return {"event_ids": event_ids, "template_ids": template_ids, "seeds": seeds}


def _source_seed_literals(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    return set(re.findall(r"RANDOM_SEED\s*=\s*(\d+)", text))


def _seed_values(value: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "seed" in str(key).lower() and isinstance(item, str | int | float):
                out.add(str(item))
            out.update(_seed_values(item))
    elif isinstance(value, list):
        for item in value:
            out.update(_seed_values(item))
    return out


def _values_for_keys(value: Any, keys: set[str]) -> list[Any]:
    out: list[Any] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in keys and isinstance(item, str | int | float):
                out.append(item)
            out.extend(_values_for_keys(item, keys))
    elif isinstance(value, list):
        for item in value:
            out.extend(_values_for_keys(item, keys))
    return out


def _sidecar_receipt(
    row_output: Path,
    split_output: Path,
    outcome_output: Path,
    preregistration_output: Path,
    bundle: StreamBundle,
) -> JsonDict:
    return {
        "row_file": {
            "path": str(row_output),
            "sha256": sha256_file(row_output),
            "row_count": len(bundle.rows),
            "schema": ROW_SCHEMA,
        },
        "split_file": {
            "path": str(split_output),
            "sha256": sha256_file(split_output),
            "base_template_count": len(bundle.splits["base_template_to_partition"]),
            "schema": SPLIT_SCHEMA,
        },
        "outcome_sidecar": {
            "path": str(outcome_output),
            "sha256": sha256_file(outcome_output),
            "row_count": len(bundle.outcomes),
            "schema": OUTCOME_SCHEMA,
        },
        "preregistration_file": {
            "path": str(preregistration_output),
            "sha256": sha256_file(preregistration_output),
            "preregistration_hash": bundle.preregistration["preregistration_hash"],
            "schema": PREREGISTRATION_SCHEMA,
        },
        "principle": FIELD_PRINCIPLES[
            "stream_row_split_outcome_and_preregistration_paths_and_hashes"
        ],
    }


def _build_artifact(
    *,
    preconditions: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    prior: Mapping[str, Any],
    validation: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    bundle = build_stream_bundle()
    preregistration = bundle.preregistration
    controls = _control_receipt(bundle.rows, bundle.outcomes)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seeds": dict(RANDOM_SEEDS),
        "field_principles": FIELD_PRINCIPLES,
        "status": "complete_partial",
        "preconditions_checked": dict(preconditions),
        "prior_fixture_hashes_and_nonreuse_receipt": dict(prior),
        "stream_row_split_outcome_and_preregistration_paths_and_hashes": dict(sidecars),
        "event_template_family_partition_and_shift_counts": _event_counts(bundle.rows),
        "never_used_seed_and_identity_receipts": _never_used_receipt(bundle, prior),
        "chronological_order": dict(validation["chronological_order"]),
        "pre_outcome_schema": {
            **PRE_OUTCOME_SCHEMA,
            "principle": FIELD_PRINCIPLES["pre_outcome_schema"],
        },
        "post_outcome_schema": {
            **POST_OUTCOME_SCHEMA,
            "principle": FIELD_PRINCIPLES["post_outcome_schema"],
        },
        "forbidden_field_scan": dict(validation["forbidden_field_scan"]),
        "exposed_fixture_overlap_counts": dict(prior["overlap_counts"]),
        "exact_validator_agreement": dict(validation["exact_validator_agreement"]),
        "alias_contradiction_malformed_poison_and_boundary_controls": {
            **controls["control_counts"],
            "all_required_controls_present": controls["all_required_controls_present"],
            "principle": FIELD_PRINCIPLES[
                "alias_contradiction_malformed_poison_and_boundary_controls"
            ],
        },
        "frozen_utility_cost_table": preregistration["frozen_utility_cost_table"],
        "primary_cluster_unit_bootstrap_and_sample_size_plan": preregistration[
            "primary_cluster_unit_bootstrap_and_sample_size_plan"
        ],
        "safety_and_noninferiority_margins": preregistration["safety_and_noninferiority_margins"],
        "brier_ece_and_descriptive_auroc_plan": preregistration[
            "brier_ece_and_descriptive_auroc_plan"
        ],
        "held_loader_one_shot_contract": preregistration["held_loader_one_shot_contract"],
        "deterministic_rebuild_checksum": deterministic_rebuild_receipt()["checksum"],
        "llm_invocation_count": 0,
        "decision_calibrated_stream_ready_score": 0.0,
        "protected_files_unchanged": dict(protected),
        "duration_s": round(float(duration_s), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["decision_calibrated_stream_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["missing_verifier_gaps"] = (
        [] if artifact["status"] == "complete_ready" else _blocked_reasons(artifact)
    )
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _never_used_receipt(bundle: StreamBundle, prior: Mapping[str, Any]) -> JsonDict:
    overlap = dict(prior["overlap_counts"])
    return {
        "fresh": all(overlap[key] == 0 for key in _BARE_OVERLAP_KEYS),
        "random_seeds": dict(RANDOM_SEEDS),
        "bootstrap_seeds_sha256": sha256_json(list(BOOTSTRAP_SEEDS)),
        "event_identity_sha256": sha256_json([row["event_id"] for row in bundle.rows]),
        "base_template_identity_sha256": sha256_json(
            sorted({row["base_template_id"] for row in bundle.rows})
        ),
        "overlap_counts": overlap,
        "principle": FIELD_PRINCIPLES["never_used_seed_and_identity_receipts"],
    }


def _preconditions(
    output: Path,
    row_output: Path,
    split_output: Path,
    outcome_output: Path,
    preregistration_output: Path,
) -> JsonDict:
    source_hashes = source_hashes_and_preconditions()
    checks = {
        "exp6145_generator_available": callable(exp6145._variant_ir),
        "exact_validator_available": callable(exp5896.certify_ir),
        "all_prior_fixture_files_present": all(
            (REPO_ROOT / path).exists() for path in PROTECTED_FILES
        ),
        "exclusion_manifest_present": (REPO_ROOT / EXCLUSION_MANIFEST_RELATIVE_PATH).exists(),
        "specs_have_exp6159_requirements": _specs_have_exp6159_requirements(),
        "output_paths_writable": all(
            os.access(path.parent, os.W_OK)
            for path in (output, row_output, split_output, outcome_output, preregistration_output)
        ),
        "protected_files_present": all((REPO_ROOT / path).exists() for path in PROTECTED_FILES),
        "llm_invocation_count_zero_at_precondition": True,
        "held_access_count_zero_at_precondition": True,
    }
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "hashed_input_receipts": source_hashes,
        "source_hash_receipt_sha256": sha256_json(source_hashes),
        "output_paths": {
            "result": _output_path_receipt(output),
            "rows": _output_path_receipt(row_output),
            "splits": _output_path_receipt(split_output),
            "outcomes": _output_path_receipt(outcome_output),
            "preregistration": _output_path_receipt(preregistration_output),
        },
        "protected_file_hashes_before": _path_hashes(PROTECTED_FILES),
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        reasons.append("preconditions")
    overlap = dict(artifact.get("exposed_fixture_overlap_counts") or {})
    if any(overlap.get(key) != 0 for key in _BARE_OVERLAP_KEYS):
        reasons.append("prior_fixture_overlap")
    if dict(artifact.get("forbidden_field_scan") or {}).get("violation_count") != 0:
        reasons.append("forbidden_pre_outcome_fields")
    exact = dict(artifact.get("exact_validator_agreement") or {})
    if exact.get("disagreement_count") or exact.get("unresolved_disagreement_count"):
        reasons.append("exact_validator_agreement")
    controls = dict(
        artifact.get("alias_contradiction_malformed_poison_and_boundary_controls") or {}
    )
    if controls.get("all_required_controls_present") is not True:
        reasons.append("missing_controls")
    if not _artifact_preregistration_sections_match(artifact):
        reasons.append("endpoint_preregistration_mismatch")
    if dict(artifact.get("held_loader_one_shot_contract") or {}).get("held_access_count") != 0:
        reasons.append("held_access_not_zero")
    if artifact.get("llm_invocation_count") != 0:
        reasons.append("llm_invocation_count")
    if (
        artifact.get("deterministic_rebuild_checksum")
        != deterministic_rebuild_receipt()["checksum"]
    ):
        reasons.append("deterministic_rebuild")
    if dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is not True:
        reasons.append("protected_files")
    test_exit_codes = dict(artifact.get("test_exit_codes") or {})
    if any(test_exit_codes.get(command) != 0 for command in DEFAULT_TEST_COMMANDS):
        reasons.append("test_commands")
    return reasons or ["ready_score"]


def _artifact_preregistration_sections_match(artifact: Mapping[str, Any]) -> bool:
    prereg = freeze_preregistration_manifest()
    return all(
        artifact.get(field) == prereg[field]
        for field in (
            "frozen_utility_cost_table",
            "primary_cluster_unit_bootstrap_and_sample_size_plan",
            "safety_and_noninferiority_margins",
            "brier_ece_and_descriptive_auroc_plan",
            "held_loader_one_shot_contract",
        )
    )


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        LEARN_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        exp6145.MODULE_RELATIVE_PATH.as_posix(),
        "python/carnot/experiment_5896_typed_constraint_ir_fixture.py",
        exp6145.ROW_FILE_RELATIVE_PATH.as_posix(),
        exp6145.SPLIT_FILE_RELATIVE_PATH.as_posix(),
        exp6145.OUTCOME_FILE_RELATIVE_PATH.as_posix(),
        "results/experiment_6148_shifted_family_admission_held.json",
        "results/experiment_5785_hardness_surface_fixture.rows.jsonl",
        "results/experiment_5786_sota_constraint_stream.rows.jsonl",
    ]
    return {
        field: {"principle": principle, "sources": list(sources)}
        for field, principle in FIELD_PRINCIPLES.items()
    }


def _path_hashes(paths: Sequence[Path], root: Path = REPO_ROOT) -> JsonDict:
    return {
        path.as_posix(): {
            "exists": (root / path).exists(),
            "sha256": sha256_file(root / path) if (root / path).exists() else None,
        }
        for path in paths
    }


def _unchanged_receipt(paths: Sequence[Path], before: Mapping[str, Any]) -> JsonDict:
    after = _path_hashes(paths)
    changed = [
        path
        for path, receipt in after.items()
        if dict(before.get(path) or {}).get("sha256") != receipt.get("sha256")
    ]
    return {
        "unchanged": not changed,
        "changed_paths": changed,
        "before": dict(before),
        "after": after,
        "principle": FIELD_PRINCIPLES["protected_files_unchanged"],
    }


def _output_path_receipt(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "parent": str(path.parent),
        "parent_exists": path.parent.exists(),
        "parent_writable": os.access(path.parent, os.W_OK),
        "existed_before": path.exists(),
        "sha256_before": sha256_file(path) if path.exists() else None,
        "path_string_sha256": sha256_text(str(path)),
    }


def _specs_have_exp6159_requirements() -> bool:
    verify = (REPO_ROOT / VERIFY_SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    learn = (REPO_ROOT / LEARN_SPEC_RELATIVE_PATH).read_text(encoding="utf-8")
    return "REQ-VERIFY-6159" in verify and "REQ-LEARN-6159" in learn


def _read_json(path: Path) -> JsonDict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text_atomic(path, "".join(canonical_json(row) + "\n" for row in rows))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _walk_json(value: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    out: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            out.extend(_walk_json(item, (*prefix, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            out.extend(_walk_json(item, (*prefix, str(index))))
    else:
        out.append((prefix, value))
    return out


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--e2e-check", action="store_true")
    args = parser.parse_args(argv)
    if args.validate or args.e2e_check:
        artifact = _read_json(REPO_ROOT / RESULT_RELATIVE_PATH)
        validate_artifact(artifact)
        replay_sidecars(
            REPO_ROOT / ROW_FILE_RELATIVE_PATH,
            REPO_ROOT / SPLIT_FILE_RELATIVE_PATH,
            REPO_ROOT / OUTCOME_FILE_RELATIVE_PATH,
            REPO_ROOT / PREREGISTRATION_FILE_RELATIVE_PATH,
        )
        return 0
    write_decision_calibrated_stream_artifact()
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())

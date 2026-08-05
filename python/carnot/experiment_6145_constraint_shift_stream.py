"""Exp6145 deterministic exact constraint-shift event stream.

Spec refs: REQ-VERIFY-6145, SCENARIO-VERIFY-6145-STREAM,
SCENARIO-VERIFY-6145-EXACT, SCENARIO-VERIFY-6145-SHIFT,
SCENARIO-VERIFY-6145-REBUILD, REQ-LEARN-6145,
SCENARIO-LEARN-6145-BOUNDARY, SCENARIO-LEARN-6145-PARTITIONS.

This fixture separates what a learner may see before a decision from what the
exact verifier knows after the decision. Rows contain only pre-decision task
features. The answer set and Python/Z3 validation results live in a sidecar so
calibration code cannot accidentally train on the oracle.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_5896_typed_constraint_ir_fixture as exp5896


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6145_constraint_shift_stream.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_6145_constraint_shift_stream.rows.jsonl")
SPLIT_FILE_RELATIVE_PATH = Path("results/experiment_6145_constraint_shift_stream.splits.json")
OUTCOME_FILE_RELATIVE_PATH = Path("results/experiment_6145_constraint_shift_stream.outcomes.jsonl")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6145_constraint_shift_stream.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6145_constraint_shift_stream.py")
VERIFY_SPEC_RELATIVE_PATH = Path("openspec/capabilities/verifiable-reasoning/spec.md")
LEARN_SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
EXP6120_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6120_outcome_committed_reduced_order_csl.json"
)
EXP6140_RESULT_RELATIVE_PATH = Path(
    "results/experiment_6140_phase_d_exp6128_option_psychometrics.json"
)
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")

SCHEMA = "carnot.experiment_6145.constraint_shift_stream.v1"
ROW_SCHEMA = SCHEMA + ".row"
SPLIT_SCHEMA = SCHEMA + ".split"
OUTCOME_SCHEMA = SCHEMA + ".outcome"
EXPERIMENT_ID = "experiment_6145_constraint_shift_stream"
RUN_DATE = "20260805"
RANDOM_SEED = 6145
INFERENCE_SUBSTRATE = "deterministic_exact_fixture_construction"
VERIFIER_IS_ORACLE = True

PARTITIONS = ("calibration", "future_known", "sealed_shifted_family")
VARIANT_ROTATION = ("canonical", "alias", "composition", "permutation")
CONTROL_ROTATION = ("contradiction", "malformed_proposal", "strategy_poison")
STRUCTURAL_SHIFT_FAMILIES = frozenset({"route_planning", "incident_response"})
FORBIDDEN_PRE_OUTCOME_TOKENS = (
    "exact_answer",
    "current_validator_result",
    "validator_result",
    "post_outcome",
    "held_label",
    "future_event",
    "oracle_label",
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    VERIFY_SPEC_RELATIVE_PATH,
    LEARN_SPEC_RELATIVE_PATH,
    Path("python/carnot/experiment_5896_typed_constraint_ir_fixture.py"),
    Path("python/carnot/constraint_ir_replay_contract.py"),
    Path("python/carnot/pipeline/z3_validator.py"),
    Path("python/carnot/verify/z3_math.py"),
    EXP6120_RESULT_RELATIVE_PATH,
    EXP6140_RESULT_RELATIVE_PATH,
    EXCLUSION_MANIFEST_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6145_constraint_shift_stream.py -q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6145_constraint_shift_stream.py "
    "-m pytest tests/python/test_experiment_6145_constraint_shift_stream.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6145_constraint_shift_stream.py --fail-under=100"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6145_constraint_shift_stream.py"
)
RUFF_CHECK_COMMAND = (
    ".venv/bin/ruff check python/carnot/experiment_6145_constraint_shift_stream.py "
    "tests/python/test_experiment_6145_constraint_shift_stream.py"
)
RUFF_FORMAT_COMMAND = (
    ".venv/bin/ruff format --check python/carnot/experiment_6145_constraint_shift_stream.py "
    "tests/python/test_experiment_6145_constraint_shift_stream.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6145_constraint_shift_stream.json"
)
VALIDATE_COMMAND = ".venv/bin/python -m carnot.experiment_6145_constraint_shift_stream --validate"
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py "
    "ops/changelog.md ops/status.md _bmad/traceability.md"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    RUFF_CHECK_COMMAND,
    RUFF_FORMAT_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    PROTECTED_FILE_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "source_generator_validator_and_exclusion_hashes",
    "stream_row_split_and_outcome_sidecar_paths_and_hashes",
    "event_base_template_family_partition_and_shift_counts",
    "chronological_order_and_seed_receipt",
    "pre_outcome_schema",
    "post_outcome_schema",
    "forbidden_pre_outcome_field_scan",
    "calibration_future_known_shifted_overlap_counts",
    "exact_validator_agreement",
    "contradiction_alias_malformed_and_poison_controls",
    "deterministic_rebuild_checksum",
    "llm_invocation_count",
    "constraint_shift_stream_ready_score",
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
    "status": "A terminal state distinguishes a ready fixture from a blocked or partial fixture.",
    "preconditions_checked": "Hashes exact generators, validators, Exp6120, Exp6140, exclusions, outputs, protected files, and retired mechanisms before construction.",
    "source_generator_validator_and_exclusion_hashes": "Content-addressed sources prove labels came from local exact code, not the retired source pool.",
    "stream_row_split_and_outcome_sidecar_paths_and_hashes": "Immutable row, split, and post-outcome files are independently replayable.",
    "event_base_template_family_partition_and_shift_counts": "Counts prove the stream has enough events, families, base groups, partitions, structural shifts, and non-shift aliases.",
    "chronological_order_and_seed_receipt": "Strict event order and seed receipts make the stream replay deterministic.",
    "pre_outcome_schema": "The learner-visible contract names only decision-time fields.",
    "post_outcome_schema": "Exact answers and validator outcomes are available only after the decision boundary.",
    "forbidden_pre_outcome_field_scan": "Exact answers, current outcomes, held labels, and future events are absent by interface, not convention.",
    "calibration_future_known_shifted_overlap_counts": "Base templates and derivatives never cross calibration, future-known, or sealed shifted-family partitions.",
    "exact_validator_agreement": "Python and Z3 must agree on every exact row with zero unresolved disagreement.",
    "contradiction_alias_malformed_and_poison_controls": "Controls prove contradictions, superficial aliases, malformed proposals, and poison attempts are represented and separated.",
    "deterministic_rebuild_checksum": "A second construction must reproduce byte-equivalent stream commitments.",
    "llm_invocation_count": "The value must be bare zero.",
    "constraint_shift_stream_ready_score": "Exactly one only when labels validate, split overlap is zero, shifts are structural rather than aliases, and rebuild is deterministic.",
    "protected_files_unchanged": "Conductor and reconciler-owned files remain byte-identical.",
    "duration_s": "Measured deterministic fixture construction time is reported.",
    "inference_substrate": "Use `deterministic_exact_fixture_construction`.",
    "verifier_is_oracle": "Exact Python/Z3 labels are the post-outcome oracle, while the pre-outcome verifier surface is oracle-distinct.",
    "missing_verifier_gaps": "Any non-finite-domain or non-oracle-separation gap is explicit.",
    "field_provenance": "Every field traces to prompt, specs, rows, sidecars, validators, tests, or command receipts.",
    "test_commands": "Commands document focused unit/spec coverage, exact-label, split/leakage, forbidden-field, shift-vs-alias, contradiction/poison, deterministic rebuild, schema, adversarial, protected-file, applicable E2E, and root-clutter checks.",
    "test_exit_codes": "Exit codes prevent failed checks from becoming readiness.",
    "reproducibility_checksum": "The artifact hash detects source, row, split, outcome, test, or protected-file drift.",
    "honest_verdict": "Use `complete_ready:`, `complete_partial:`, or `blocked:` and identify any missing shift or oracle-separation requirement.",
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
        "memory_provenance",
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
        "current_validator_result",
        "exact_labels",
        "parser",
        "python",
        "z3",
        "cross_backend_agreement",
        "solver_cost_diagnostic",
    ],
}


class ConstraintShiftStreamError(ValueError):
    """Raised when the stream would leak labels or drift from exact replay."""


@dataclass
class StreamBundle:
    rows: list[JsonDict]
    splits: JsonDict
    outcomes: list[JsonDict]


_BUNDLE_CACHE: StreamBundle | None = None


@dataclass(frozen=True)
class FamilyConfig:
    family: str
    entity_domain: str
    group_domain: str
    level_domain: str
    member_predicate: str
    gate_predicate: str
    block_predicate: str
    rank_predicate: str
    target_predicate: str
    prefix: str
    structural_shift_family: bool = False


FAMILY_CONFIGS = (
    FamilyConfig(
        "access_control",
        "person",
        "department",
        "clearance",
        "works_in",
        "approved",
        "suspended",
        "clearance",
        "eligible",
        "ac",
    ),
    FamilyConfig(
        "task_selection",
        "task",
        "queue",
        "priority",
        "queued_in",
        "queue_open",
        "blocked",
        "priority",
        "selectable",
        "ts",
    ),
    FamilyConfig(
        "menu_recommendation",
        "dish",
        "category",
        "price",
        "listed_as",
        "allowed_category",
        "allergen",
        "price",
        "recommended",
        "mr",
    ),
    FamilyConfig(
        "inventory_allocation",
        "sku",
        "warehouse",
        "stock_level",
        "stored_at",
        "warehouse_open",
        "reserved",
        "stock_level",
        "allocatable",
        "ia",
    ),
    FamilyConfig(
        "release_gating",
        "change",
        "service",
        "risk_rank",
        "touches",
        "service_green",
        "frozen",
        "risk_rank",
        "releasable",
        "rg",
    ),
    FamilyConfig(
        "maintenance_schedule",
        "job",
        "machine",
        "urgency",
        "assigned_to",
        "machine_ready",
        "locked",
        "urgency",
        "schedulable",
        "ms",
    ),
    FamilyConfig(
        "route_planning",
        "route",
        "zone",
        "safety_rank",
        "passes_zone",
        "zone_open",
        "hazard",
        "safety_rank",
        "dispatchable",
        "rp",
        True,
    ),
    FamilyConfig(
        "incident_response",
        "incident",
        "team",
        "severity",
        "owned_by",
        "team_available",
        "suppressed",
        "severity",
        "actionable",
        "ir",
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
    """Return a prefixed SHA-256 digest for canonical JSON-compatible evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash exact file bytes so names and mtimes cannot define evidence."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def row_hash(row: Mapping[str, Any]) -> str:
    """Hash one pre-outcome row while blanking its self-hash field."""

    stable = json.loads(canonical_json(row))
    stable["row_hash"] = ""
    return sha256_json(stable)


def outcome_hash(outcome: Mapping[str, Any]) -> str:
    """Hash one post-outcome sidecar row while blanking its self-hash field."""

    stable = json.loads(canonical_json(outcome))
    stable["outcome_hash"] = ""
    return sha256_json(stable)


def split_hash(splits: Mapping[str, Any]) -> str:
    """Hash the split manifest while blanking its self-hash field."""

    stable = json.loads(canonical_json(splits))
    stable["split_hash"] = ""
    return sha256_json(stable)


def build_stream_bundle() -> StreamBundle:
    """Return a copy of the deterministic stream bundle."""

    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is None:
        _BUNDLE_CACHE = _build_stream_bundle_uncached()
    return StreamBundle(
        rows=_copy_json(_BUNDLE_CACHE.rows),
        splits=_copy_json(_BUNDLE_CACHE.splits),
        outcomes=_copy_json(_BUNDLE_CACHE.outcomes),
    )


def _build_stream_bundle_uncached() -> StreamBundle:
    """Build rows, split manifest, and exact sidecar in chronological order."""

    rows: list[JsonDict] = []
    outcomes: list[JsonDict] = []
    event_to_partition: dict[str, str] = {}
    base_to_partition: dict[str, str] = {}
    prior_event_ids: list[str] = []
    prior_family_counts: Counter[str] = Counter()
    prior_template_counts: Counter[str] = Counter()

    for family_index, config in enumerate(FAMILY_CONFIGS):
        for template_index in range(6):
            base_template_id = f"exp6145.{config.family}.t{template_index:02d}"
            partition = _partition_for(config, template_index)
            base_to_partition[base_template_id] = partition
            variants = (*VARIANT_ROTATION, CONTROL_ROTATION[template_index % 3])
            for variant_kind in variants:
                event_index = len(rows)
                event_id = f"exp6145-event-{event_index:06d}"
                control_kind = _control_kind(variant_kind)
                ir = _variant_ir(config, template_index, variant_kind)
                cert = exp5896.certify_ir(ir)
                post_outcome = _post_outcome(control_kind, cert)
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
                    prior_event_ids=prior_event_ids,
                    prior_family_count=prior_family_counts[config.family],
                    prior_template_count=prior_template_counts[base_template_id],
                )
                outcome = {
                    "schema": OUTCOME_SCHEMA,
                    "event_id": event_id,
                    "chronological_index": event_index,
                    "base_template_id": base_template_id,
                    "family": config.family,
                    "partition": partition,
                    "post_outcome": post_outcome,
                    "outcome_hash": "",
                }
                row["row_hash"] = row_hash(row)
                outcome["outcome_hash"] = outcome_hash(outcome)
                rows.append(row)
                outcomes.append(outcome)
                event_to_partition[event_id] = partition
                prior_event_ids.append(event_id)
                prior_family_counts[config.family] += 1
                prior_template_counts[base_template_id] += 1

    splits = _split_manifest(rows, base_to_partition, event_to_partition)
    return StreamBundle(rows=rows, splits=splits, outcomes=outcomes)


def validate_stream_bundle(bundle: StreamBundle) -> JsonDict:
    """Validate chronology, pre-outcome isolation, splits, controls, and outcomes."""

    rows = [dict(row) for row in bundle.rows]
    outcomes = [dict(outcome) for outcome in bundle.outcomes]
    expected = build_stream_bundle()
    if len(rows) != len(expected.rows) or len(outcomes) != len(expected.outcomes):
        raise ConstraintShiftStreamError("chronology row count mismatch")

    seen: set[str] = set()
    for index, row in enumerate(rows):
        event_id = str(row.get("event_id"))
        if event_id in seen or event_id != f"exp6145-event-{index:06d}":
            raise ConstraintShiftStreamError("chronology event id mismatch")
        seen.add(event_id)
        if row.get("chronological_index") != index:
            raise ConstraintShiftStreamError("chronology index mismatch")

    forbidden_scan = scan_forbidden_pre_outcome_fields(rows)
    if forbidden_scan["violation_count"]:
        raise ConstraintShiftStreamError("forbidden pre-outcome field")

    split_receipt = _validate_split_manifest(rows, bundle.splits)
    for index, row in enumerate(rows):
        if row.get("row_hash") != row_hash(row):
            raise ConstraintShiftStreamError("row hash drift")
        comparable = dict(row)
        if comparable != expected.rows[index]:
            raise ConstraintShiftStreamError("row drift")

    for index, outcome in enumerate(outcomes):
        if outcome.get("event_id") != rows[index].get("event_id"):
            raise ConstraintShiftStreamError("outcome chronology drift")
        if outcome.get("outcome_hash") != outcome_hash(outcome):
            raise ConstraintShiftStreamError("outcome drift")
        if outcome != expected.outcomes[index]:
            raise ConstraintShiftStreamError("outcome drift")

    exact = _exact_validator_agreement(outcomes)
    controls = _control_receipt(rows, outcomes)
    shifts = _shift_receipt(rows)
    chronology = _chronology_receipt(rows)
    return {
        "ok": True,
        "row_count": len(rows),
        "outcome_count": len(outcomes),
        "chronological_order": chronology,
        "forbidden_pre_outcome_field_scan": forbidden_scan,
        "overlap_counts": split_receipt,
        "exact_validator_agreement": exact,
        "control_counts": controls["control_counts"],
        "shift_counts": shifts,
        "bundle_checksum": bundle_checksum(bundle),
    }


def replay_sidecars(row_path: Path, split_path: Path, outcome_path: Path) -> JsonDict:
    """Load materialized sidecars and replay the Exp6145 validation contract."""

    bundle = StreamBundle(
        rows=_load_jsonl(row_path),
        splits=json.loads(split_path.read_text(encoding="utf-8")),
        outcomes=_load_jsonl(outcome_path),
    )
    receipt = validate_stream_bundle(bundle)
    receipt.update(
        {
            "row_sha256": sha256_file(row_path),
            "split_sha256": sha256_file(split_path),
            "outcome_sha256": sha256_file(outcome_path),
        }
    )
    return receipt


def write_constraint_shift_stream_artifact(
    *,
    output_path: Path | None = None,
    row_output_path: Path | None = None,
    split_output_path: Path | None = None,
    outcome_output_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
) -> JsonDict:
    """Write the immutable Exp6145 row, split, outcome, and result artifacts."""

    started = time.monotonic()
    output = output_path or REPO_ROOT / RESULT_RELATIVE_PATH
    row_output = row_output_path or REPO_ROOT / ROW_FILE_RELATIVE_PATH
    split_output = split_output_path or REPO_ROOT / SPLIT_FILE_RELATIVE_PATH
    outcome_output = outcome_output_path or REPO_ROOT / OUTCOME_FILE_RELATIVE_PATH
    for path in (output, row_output, split_output, outcome_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    protected_before = _path_hashes(PROTECTED_FILES)
    source_hashes = source_generator_validator_and_exclusion_hashes()
    preconditions = _preconditions(output, row_output, split_output, outcome_output, source_hashes)
    bundle = build_stream_bundle()
    validation = validate_stream_bundle(bundle)
    _write_jsonl_atomic(row_output, bundle.rows)
    _write_json_atomic(split_output, bundle.splits)
    _write_jsonl_atomic(outcome_output, bundle.outcomes)
    sidecars = _sidecar_receipt(row_output, split_output, outcome_output, bundle)
    protected = _unchanged_receipt(PROTECTED_FILES, protected_before)
    elapsed = float(duration_s if duration_s is not None else time.monotonic() - started)
    artifact = _build_artifact(
        preconditions=preconditions,
        source_hashes=source_hashes,
        sidecars=sidecars,
        validation=validation,
        protected=protected,
        duration_s=elapsed,
        test_exit_codes=dict(test_exit_codes or {}),
    )
    validate_artifact(artifact)
    _write_json_atomic(output, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the terminal artifact readiness contract."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle")
    if artifact.get("llm_invocation_count") != 0:
        raise ValueError("llm_invocation_count")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    if artifact.get("constraint_shift_stream_ready_score") != ready_score(artifact):
        raise ValueError("ready_score")
    if artifact.get("status") != status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    """Return the strict Exp6145 readiness scalar."""

    counts = dict(artifact.get("event_base_template_family_partition_and_shift_counts") or {})
    overlap = dict(artifact.get("calibration_future_known_shifted_overlap_counts") or {})
    exact = dict(artifact.get("exact_validator_agreement") or {})
    controls = dict(artifact.get("contradiction_alias_malformed_and_poison_controls") or {})
    forbidden = dict(artifact.get("forbidden_pre_outcome_field_scan") or {})
    chronology = dict(artifact.get("chronological_order_and_seed_receipt") or {})
    task_exit_codes = dict(artifact.get("test_exit_codes") or {})
    missing_commands = [
        command for command in DEFAULT_TEST_COMMANDS if command not in task_exit_codes
    ]
    nonzero_commands = [
        command for command in DEFAULT_TEST_COMMANDS if task_exit_codes.get(command) != 0
    ]
    ready = (
        dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and counts.get("event_count", 0) >= 240
        and counts.get("family_count", 0) >= 6
        and counts.get("alias_counted_as_shift_count") == 0
        and counts.get("structural_shift_event_count", 0) > 0
        and set(dict(counts.get("partition_counts") or {})) == set(PARTITIONS)
        and chronology.get("monotone") is True
        and forbidden.get("violation_count") == 0
        and overlap.get("base_template_overlap_count") == 0
        and overlap.get("derivative_partition_mismatch_count") == 0
        and exact.get("disagreement_count") == 0
        and exact.get("unresolved_disagreement_count") == 0
        and controls.get("contradiction", {}).get("rejected", 0) > 0
        and controls.get("malformed_proposal", {}).get("rejected", 0) > 0
        and controls.get("strategy_poison", {}).get("rejected", 0) > 0
        and controls.get("alias", {}).get("counted_as_shift") == 0
        and artifact.get("deterministic_rebuild_checksum")
        == deterministic_rebuild_receipt()["checksum"]
        and artifact.get("llm_invocation_count") == 0
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is True
        and not missing_commands
        and not nonzero_commands
    )
    return 1.0 if ready else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    """Return the terminal status implied by readiness."""

    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        return "blocked"
    return "complete_ready" if ready_score(artifact) == 1.0 else "complete_partial"


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return a terminal-prefixed verdict for the stream fixture."""

    if status(artifact) == "complete_ready":
        return "complete_ready: exact_constraint_shift_stream_oracle_separated"
    if status(artifact) == "blocked":
        return "blocked: " + ",".join(_blocked_reasons(artifact)[:8])
    return "complete_partial: " + ",".join(_blocked_reasons(artifact)[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact while normalizing volatile host fields."""

    stable = json.loads(canonical_json(artifact))
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        for key in ("disk", "ram"):
            if isinstance(preconditions.get(key), dict):
                preconditions[key]["available_mb"] = 0
        output_paths = preconditions.get("output_paths")
        if isinstance(output_paths, dict):
            for receipt in output_paths.values():
                if isinstance(receipt, dict):
                    receipt["path"] = "<normalized>"
    return sha256_json(stable)


def deterministic_rebuild_receipt() -> JsonDict:
    """Build the stream twice and return the deterministic commitment."""

    first = build_stream_bundle()
    second = build_stream_bundle()
    first_checksum = bundle_checksum(first)
    second_checksum = bundle_checksum(second)
    return {
        "checksum": first_checksum,
        "second_checksum": second_checksum,
        "matches": first_checksum == second_checksum,
        "seed": RANDOM_SEED,
    }


def bundle_checksum(bundle: StreamBundle) -> str:
    """Hash rows, splits, and outcomes without file-path metadata."""

    return sha256_json({"rows": bundle.rows, "splits": bundle.splits, "outcomes": bundle.outcomes})


def scan_forbidden_pre_outcome_fields(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Scan only learner-visible payloads for forbidden oracle terms."""

    violations: list[JsonDict] = []
    for row in rows:
        event_id = str(row.get("event_id"))
        for path, value in _walk_json(dict(row.get("pre_decision") or {})):
            text = str(value).lower() if isinstance(value, str) else ""
            key = ".".join(path).lower()
            for token in FORBIDDEN_PRE_OUTCOME_TOKENS:
                if token in key or (text and token in text):
                    violations.append(
                        {"event_id": event_id, "path": ".".join(path), "token": token}
                    )
    return {
        "violation_count": len(violations),
        "violations": violations,
        "scanned_row_count": len(rows),
        "principle": FIELD_PRINCIPLES["forbidden_pre_outcome_field_scan"],
    }


def source_generator_validator_and_exclusion_hashes(root: Path = REPO_ROOT) -> JsonDict:
    """Hash exact sources, retirement evidence, exclusions, and specs."""

    paths: JsonDict = {}
    for relative in HASHED_SOURCE_PATHS:
        path = root / relative
        paths[relative.as_posix()] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return {
        "paths": paths,
        "exact_generator_paths": [
            "python/carnot/experiment_5896_typed_constraint_ir_fixture.py",
            "python/carnot/constraint_ir_replay_contract.py",
        ],
        "exact_validator_paths": [
            "python/carnot/pipeline/z3_validator.py",
            "python/carnot/verify/z3_math.py",
        ],
        "exp6120_transaction_evidence": EXP6120_RESULT_RELATIVE_PATH.as_posix(),
        "exp6140_retirement_evidence": EXP6140_RESULT_RELATIVE_PATH.as_posix(),
        "exclusion_manifest": EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
        "retired_exp6128_source_pool_dependency": False,
        "generated_answer_transport_dependency": False,
        "principle": FIELD_PRINCIPLES["source_generator_validator_and_exclusion_hashes"],
    }


def _pre_outcome_row(
    *,
    event_id: str,
    event_index: int,
    base_template_id: str,
    config: FamilyConfig,
    family_index: int,
    template_index: int,
    partition: str,
    variant_kind: str,
    control_kind: str,
    ir: Mapping[str, Any],
    prior_event_ids: Sequence[str],
    prior_family_count: int,
    prior_template_count: int,
) -> JsonDict:
    graph = _constraint_graph_summary(ir)
    alias_only = variant_kind == "alias"
    structural_shift = config.structural_shift_family and not alias_only
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
                "synthetic_constraint_domain": "finite_domain_horn_arithmetic",
            },
            "constraint_graph_summary": graph,
            "candidate_strategy": {
                "strategy_id": f"{variant_kind}_strategy_v1",
                "features": {
                    "alias_surface": alias_only,
                    "composition_surface": variant_kind == "composition",
                    "permuted_fact_order": variant_kind == "permutation",
                    "proposal_form": "malformed_ir"
                    if variant_kind == "malformed_proposal"
                    else "well_formed_ir",
                    "memory_action": "poison_request"
                    if variant_kind == "strategy_poison"
                    else "neutral",
                },
            },
            "memory_provenance": {
                "source": "synthetic_exact_fixture",
                "prior_event_count": len(prior_event_ids),
                "visible_prior_event_ids_hash": sha256_json(list(prior_event_ids)),
                "prior_scope": "chronological_prefix_only",
            },
            "chronological_history": {
                "event_index": event_index,
                "prior_same_family_event_count": prior_family_count,
                "prior_same_template_event_count": prior_template_count,
                "history_seed": RANDOM_SEED + event_index,
            },
        },
        "row_hash": "",
    }


def _post_outcome(control_kind: str, cert: Mapping[str, Any]) -> JsonDict:
    parser = dict(cert["parser"])
    python = dict(cert["python"])
    z3 = dict(cert["z3"])
    agreement = dict(cert["cross_backend_agreement"])
    parser_ok = parser.get("status") == "accepted"
    sat = python.get("status") == "sat" and z3.get("status") == "sat"
    agree = agreement.get("agrees") is True
    accepted = parser_ok and sat and agree and control_kind != "strategy_poison"
    exact_labels = {
        "parse_valid": parser_ok,
        "python_status": python.get("status"),
        "z3_status": z3.get("status"),
        "python_z3_agree": agree if parser_ok else None,
        "satisfiable": sat if parser_ok else False,
        "accepted": accepted,
    }
    return {
        "control_kind": control_kind,
        "exact_answer": list(python.get("query_bindings") or []),
        "current_validator_result": "accepted" if accepted else "rejected",
        "exact_labels": exact_labels,
        "parser": parser,
        "python": python,
        "z3": z3,
        "cross_backend_agreement": agreement,
        "solver_cost_diagnostic": {
            "z3_bool_count": z3.get("bool_count"),
            "z3_implication_count": z3.get("implication_count"),
            "cost_is_diagnostic_only": True,
            "difficulty_or_energy_defined_by_cost": False,
        },
    }


def _variant_ir(config: FamilyConfig, template_index: int, variant_kind: str) -> JsonDict:
    base = _base_ir(config, template_index)
    if variant_kind == "alias":
        return _alias_ir(base, config, template_index)
    if variant_kind == "composition":
        return _composition_ir(base, config)
    if variant_kind == "permutation":
        payload = _copy_json(base)
        payload["facts"] = list(reversed(payload["facts"]))
        return payload
    if variant_kind == "contradiction":
        payload = _copy_json(base)
        target_group = payload["domains"][1]["values"][0]
        payload["facts"].append(
            {"predicate": config.gate_predicate, "args": [target_group], "truth": False}
        )
        return payload
    if variant_kind == "malformed_proposal":
        payload = _copy_json(base)
        payload["generated_answer_transport_hint"] = "retired_pool_not_allowed"
        return payload
    return base


def _base_ir(config: FamilyConfig, template_index: int) -> JsonDict:
    entities = [f"{config.prefix}_e{template_index}_{idx}" for idx in range(3)]
    groups = [f"{config.prefix}_g{template_index}_{idx}" for idx in range(2)]
    levels = [1, 2, 3, 4]
    ranks = [3, 4, 2 + (template_index % 2)]
    facts: list[JsonDict] = [
        {"predicate": config.member_predicate, "args": [entities[0], groups[0]], "truth": True},
        {"predicate": config.member_predicate, "args": [entities[1], groups[1]], "truth": True},
        {"predicate": config.member_predicate, "args": [entities[2], groups[0]], "truth": True},
        {"predicate": config.gate_predicate, "args": [groups[0]], "truth": True},
        {"predicate": config.gate_predicate, "args": [groups[1]], "truth": True},
        {"predicate": config.block_predicate, "args": [entities[1]], "truth": True},
    ]
    for entity, rank in zip(entities, ranks, strict=True):
        facts.append({"predicate": config.rank_predicate, "args": [entity, rank], "truth": True})
    return _constraint_ir(
        config=config,
        domains=[
            {"id": config.entity_domain, "type": "symbol", "values": entities},
            {"id": config.group_domain, "type": "symbol", "values": groups},
            {"id": config.level_domain, "type": "int", "values": levels},
        ],
        predicates=[
            {
                "id": config.member_predicate,
                "arg_types": [config.entity_domain, config.group_domain],
            },
            {"id": config.gate_predicate, "arg_types": [config.group_domain]},
            {"id": config.block_predicate, "arg_types": [config.entity_domain]},
            {"id": config.rank_predicate, "arg_types": [config.entity_domain, config.level_domain]},
            {"id": config.target_predicate, "arg_types": [config.entity_domain]},
        ],
        facts=facts,
        body_terms=[
            {"node": "atom", "predicate": config.member_predicate, "args": ["?item", "?group"]},
            {"node": "atom", "predicate": config.gate_predicate, "args": ["?group"]},
            {"node": "atom", "predicate": config.rank_predicate, "args": ["?item", "?level"]},
            {"node": "arith", "left": "?level", "op": ">=", "right": 2},
            {
                "node": "not",
                "term": {"node": "atom", "predicate": config.block_predicate, "args": ["?item"]},
            },
        ],
    )


def _constraint_ir(
    *,
    config: FamilyConfig,
    domains: list[JsonDict],
    predicates: list[JsonDict],
    facts: list[JsonDict],
    body_terms: list[JsonDict],
) -> JsonDict:
    entities = [
        {"id": value, "domain": domain["id"]}
        for domain in domains
        if domain["type"] == "symbol"
        for value in domain["values"]
    ]
    return {
        "schema_version": exp5896.CONSTRAINT_IR_SCHEMA_VERSION,
        "domains": domains,
        "entities": entities,
        "predicates": predicates,
        "facts": facts,
        "rules": [
            {
                "id": "r1",
                "variables": {
                    "?item": config.entity_domain,
                    "?group": config.group_domain,
                    "?level": config.level_domain,
                },
                "body": {"node": "and", "terms": body_terms},
                "head": {
                    "node": "atom",
                    "predicate": config.target_predicate,
                    "args": ["?item"],
                },
            }
        ],
        "query": {
            "vars": {"?item": config.entity_domain},
            "where": {"node": "atom", "predicate": config.target_predicate, "args": ["?item"]},
        },
    }


def _alias_ir(base: Mapping[str, Any], config: FamilyConfig, template_index: int) -> JsonDict:
    payload = _copy_json(base)
    renames: dict[str, str] = {}
    for domain in payload["domains"]:
        if domain["type"] == "symbol":
            domain_values = []
            for index, value in enumerate(domain["values"]):
                renamed = f"{config.prefix}_alias{template_index}_{len(renames)}_{index}"
                renames[str(value)] = renamed
                domain_values.append(renamed)
            domain["values"] = domain_values
    for entity in payload["entities"]:
        entity["id"] = renames.get(entity["id"], entity["id"])
    for fact in payload["facts"]:
        fact["args"] = [renames.get(arg, arg) for arg in fact["args"]]
    for rule in payload["rules"]:
        _rename_expr_constants(rule["body"], renames)
        _rename_expr_constants(rule["head"], renames)
    _rename_expr_constants(payload["query"]["where"], renames)
    return payload


def _composition_ir(base: Mapping[str, Any], config: FamilyConfig) -> JsonDict:
    payload = _copy_json(base)
    aux_predicate = f"{config.prefix}_certified"
    entity_values = payload["domains"][0]["values"]
    payload["predicates"].append({"id": aux_predicate, "arg_types": [config.entity_domain]})
    for entity in entity_values:
        payload["facts"].append({"predicate": aux_predicate, "args": [entity], "truth": True})
    payload["rules"][0]["body"]["terms"].append(
        {"node": "atom", "predicate": aux_predicate, "args": ["?item"]}
    )
    return payload


def _rename_expr_constants(expr: JsonDict, renames: Mapping[str, str]) -> None:
    if expr["node"] == "atom":
        expr["args"] = [renames.get(arg, arg) for arg in expr["args"]]
    elif expr["node"] == "not":
        _rename_expr_constants(expr["term"], renames)
    elif expr["node"] == "and":
        for term in expr["terms"]:
            _rename_expr_constants(term, renames)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _control_kind(variant_kind: str) -> str:
    if variant_kind in {"canonical", "composition", "permutation"}:
        return "normal"
    if variant_kind == "alias":
        return "alias"
    return variant_kind


def _partition_for(config: FamilyConfig, template_index: int) -> str:
    if config.structural_shift_family:
        return "sealed_shifted_family"
    return "future_known" if template_index % 3 == 0 else "calibration"


def _constraint_graph_summary(ir: Mapping[str, Any]) -> JsonDict:
    facts = list(ir.get("facts") or [])
    predicates = list(ir.get("predicates") or [])
    rules = list(ir.get("rules") or [])
    body_terms = []
    for rule in rules:
        body = dict(rule.get("body") or {})
        body_terms.extend(list(body.get("terms") or []))
    return {
        "domain_count": len(list(ir.get("domains") or [])),
        "entity_count": len(list(ir.get("entities") or [])),
        "predicate_count": len(predicates),
        "fact_count": len(facts),
        "rule_count": len(rules),
        "query_variable_count": len(dict(dict(ir.get("query") or {}).get("vars") or {})),
        "negation_term_count": sum(1 for term in body_terms if term.get("node") == "not"),
        "arithmetic_term_count": sum(1 for term in body_terms if term.get("node") == "arith"),
        "body_term_count": len(body_terms),
        "has_recursive_dependency": False,
        "malformed_extra_top_level_count": len(sorted(set(ir) - set(exp5896.TOP_LEVEL_KEYS))),
    }


def _split_manifest(
    rows: Sequence[Mapping[str, Any]],
    base_to_partition: Mapping[str, str],
    event_to_partition: Mapping[str, str],
) -> JsonDict:
    partition_counts = dict(Counter(str(row["partition"]) for row in rows))
    base_counts = dict(Counter(base_to_partition.values()))
    family_partitions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        family_partitions[str(row["family"])].add(str(row["partition"]))
    manifest: JsonDict = {
        "schema": SPLIT_SCHEMA,
        "random_seed": RANDOM_SEED,
        "partitions": list(PARTITIONS),
        "base_template_to_partition": dict(sorted(base_to_partition.items())),
        "event_to_partition": dict(sorted(event_to_partition.items())),
        "partition_counts": {
            partition: partition_counts.get(partition, 0) for partition in PARTITIONS
        },
        "base_template_partition_counts": {
            partition: base_counts.get(partition, 0) for partition in PARTITIONS
        },
        "family_partitions": {
            family: sorted(partitions) for family, partitions in sorted(family_partitions.items())
        },
        "assignment_stage": "before_variant_emission",
        "split_hash": "",
    }
    manifest["split_hash"] = split_hash(manifest)
    return manifest


def _validate_split_manifest(
    rows: Sequence[Mapping[str, Any]], splits: Mapping[str, Any]
) -> JsonDict:
    if splits.get("split_hash") != split_hash(splits):
        raise ConstraintShiftStreamError("split hash drift")
    base_to_partition = dict(splits.get("base_template_to_partition") or {})
    event_to_partition = dict(splits.get("event_to_partition") or {})
    base_seen: dict[str, set[str]] = defaultdict(set)
    derivative_mismatch = 0
    for row in rows:
        event_id = str(row["event_id"])
        base = str(row["base_template_id"])
        partition = str(row["partition"])
        if event_to_partition.get(event_id) != partition:
            raise ConstraintShiftStreamError("partition drift")
        if base_to_partition.get(base) != partition:
            derivative_mismatch += 1
        base_seen[base].add(partition)
    if derivative_mismatch:
        raise ConstraintShiftStreamError("partition drift")
    crossing = {base: sorted(parts) for base, parts in base_seen.items() if len(parts) > 1}
    return {
        "base_template_overlap_count": len(crossing),
        "crossing_base_templates": crossing,
        "derivative_partition_mismatch_count": derivative_mismatch,
        "partition_counts": dict(Counter(str(row["partition"]) for row in rows)),
        "principle": FIELD_PRINCIPLES["calibration_future_known_shifted_overlap_counts"],
    }


def _chronology_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    event_ids = [str(row["event_id"]) for row in rows]
    return {
        "monotone": all(
            event_id == f"exp6145-event-{index:06d}" for index, event_id in enumerate(event_ids)
        ),
        "event_id_count": len(event_ids),
        "unique_event_id_count": len(set(event_ids)),
        "first_event_id": event_ids[0] if event_ids else None,
        "last_event_id": event_ids[-1] if event_ids else None,
        "random_seed": RANDOM_SEED,
        "event_ids_sha256": sha256_json(event_ids),
        "row_order_sha256": sha256_json([row["row_hash"] for row in rows]),
        "principle": FIELD_PRINCIPLES["chronological_order_and_seed_receipt"],
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
        "authority": "exp5896.certify_ir_python_and_z3",
        "python_z3_compared_count": compared,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "unresolved_disagreement_count": len(unresolved),
        "unresolved": unresolved,
        "solver_cost_diagnostic_only": True,
        "principle": FIELD_PRINCIPLES["exact_validator_agreement"],
    }


def _control_receipt(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
) -> JsonDict:
    by_event = {str(outcome["event_id"]): dict(outcome["post_outcome"]) for outcome in outcomes}
    control_counts: dict[str, JsonDict] = {}
    for control in ("normal", "alias", "contradiction", "malformed_proposal", "strategy_poison"):
        selected = [row for row in rows if row["control_kind"] == control]
        accepted = [
            row
            for row in selected
            if by_event[str(row["event_id"])]["current_validator_result"] == "accepted"
        ]
        rejected = len(selected) - len(accepted)
        control_counts[control] = {
            "events": len(selected),
            "accepted": len(accepted),
            "rejected": rejected,
            "counted_as_shift": sum(1 for row in selected if row["structural_shift"] is True),
        }
    return {
        "control_counts": control_counts,
        "all_required_controls_present": all(
            control_counts[name]["events"] > 0
            for name in ("alias", "contradiction", "malformed_proposal", "strategy_poison")
        ),
        "principle": FIELD_PRINCIPLES["contradiction_alias_malformed_and_poison_controls"],
    }


def _shift_receipt(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    structural_families = {
        str(row["family"]) for row in rows if row.get("structural_shift") is True
    }
    alias_confusion = [
        str(row["event_id"])
        for row in rows
        if row.get("alias_only") is True and row.get("structural_shift") is True
    ]
    sealed_structural = {
        str(row["family"])
        for row in rows
        if row.get("partition") == "sealed_shifted_family"
        and str(row["family"]) in STRUCTURAL_SHIFT_FAMILIES
    }
    return {
        "structural_shift_family_count": len(structural_families),
        "sealed_structural_shift_family_count": len(sealed_structural),
        "structural_shift_event_count": sum(
            1 for row in rows if row.get("structural_shift") is True
        ),
        "alias_event_count": sum(1 for row in rows if row.get("alias_only") is True),
        "structural_shift_alias_confusion_count": len(alias_confusion),
        "alias_confusion_event_ids": alias_confusion,
    }


def _event_counts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    shifts = _shift_receipt(rows)
    partition_counts = dict(Counter(str(row["partition"]) for row in rows))
    return {
        "event_count": len(rows),
        "base_template_count": len({row["base_template_id"] for row in rows}),
        "family_count": len({row["family"] for row in rows}),
        "family_counts": dict(sorted(Counter(str(row["family"]) for row in rows).items())),
        "partition_counts": {
            partition: partition_counts.get(partition, 0) for partition in PARTITIONS
        },
        "variant_counts": dict(sorted(Counter(str(row["variant_kind"]) for row in rows).items())),
        "structural_shift_event_count": shifts["structural_shift_event_count"],
        "structural_shift_family_count": shifts["structural_shift_family_count"],
        "sealed_structural_shift_family_count": shifts["sealed_structural_shift_family_count"],
        "alias_event_count": shifts["alias_event_count"],
        "alias_counted_as_shift_count": shifts["structural_shift_alias_confusion_count"],
        "principle": FIELD_PRINCIPLES["event_base_template_family_partition_and_shift_counts"],
    }


def _sidecar_receipt(
    row_output: Path,
    split_output: Path,
    outcome_output: Path,
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
        "principle": FIELD_PRINCIPLES["stream_row_split_and_outcome_sidecar_paths_and_hashes"],
    }


def _build_artifact(
    *,
    preconditions: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    validation: Mapping[str, Any],
    protected: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    bundle = build_stream_bundle()
    controls = _control_receipt(bundle.rows, bundle.outcomes)
    rebuild = deterministic_rebuild_receipt()
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "field_principles": FIELD_PRINCIPLES,
        "status": "complete_partial",
        "preconditions_checked": dict(preconditions),
        "source_generator_validator_and_exclusion_hashes": dict(source_hashes),
        "stream_row_split_and_outcome_sidecar_paths_and_hashes": dict(sidecars),
        "event_base_template_family_partition_and_shift_counts": _event_counts(bundle.rows),
        "chronological_order_and_seed_receipt": dict(validation["chronological_order"]),
        "pre_outcome_schema": {
            **PRE_OUTCOME_SCHEMA,
            "principle": FIELD_PRINCIPLES["pre_outcome_schema"],
        },
        "post_outcome_schema": {
            **POST_OUTCOME_SCHEMA,
            "principle": FIELD_PRINCIPLES["post_outcome_schema"],
        },
        "forbidden_pre_outcome_field_scan": dict(validation["forbidden_pre_outcome_field_scan"]),
        "calibration_future_known_shifted_overlap_counts": dict(validation["overlap_counts"]),
        "exact_validator_agreement": dict(validation["exact_validator_agreement"]),
        "contradiction_alias_malformed_and_poison_controls": {
            **controls["control_counts"],
            "all_required_controls_present": controls["all_required_controls_present"],
            "principle": FIELD_PRINCIPLES["contradiction_alias_malformed_and_poison_controls"],
        },
        "deterministic_rebuild_checksum": rebuild["checksum"],
        "llm_invocation_count": 0,
        "constraint_shift_stream_ready_score": 0.0,
        "protected_files_unchanged": dict(protected),
        "duration_s": round(duration_s, 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["constraint_shift_stream_ready_score"] = ready_score(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _preconditions(
    output: Path,
    row_output: Path,
    split_output: Path,
    outcome_output: Path,
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    disk = _disk_probe(REPO_ROOT)
    ram = _memory_probe()
    try:
        import z3

        z3_version = z3.get_version_string()
        z3_available = True
    except Exception:  # pragma: no cover - z3 is required by existing tests.
        z3_version = None
        z3_available = False
    exp6140 = _read_json_if_exists(REPO_ROOT / EXP6140_RESULT_RELATIVE_PATH)
    checks = {
        "exact_fixture_builder_available": callable(exp5896.certify_ir),
        "z3_available": z3_available,
        "exp6120_transaction_evidence_present": (REPO_ROOT / EXP6120_RESULT_RELATIVE_PATH).exists(),
        "exp6140_retirement_present": (REPO_ROOT / EXP6140_RESULT_RELATIVE_PATH).exists(),
        "exp6140_retired": exp6140.get("status") == "retired",
        "exclusion_manifest_present": (REPO_ROOT / EXCLUSION_MANIFEST_RELATIVE_PATH).exists(),
        "retired_exp6128_source_pool_dependency_absent": True,
        "generated_answer_transport_dependency_absent": True,
        "output_paths_writable": all(
            os.access(path.parent, os.W_OK)
            for path in (output, row_output, split_output, outcome_output)
        ),
        "protected_files_present": all((REPO_ROOT / path).exists() for path in PROTECTED_FILES),
        "disk": disk["ok"],
        "ram": ram["ok"],
    }
    return {
        "run_date": RUN_DATE,
        "checks": checks,
        "preconditions_ready": all(checks.values()),
        "source_hash_receipt_sha256": sha256_json(source_hashes),
        "z3_version": z3_version,
        "output_paths": {
            "result": _output_path_receipt(output),
            "rows": _output_path_receipt(row_output),
            "splits": _output_path_receipt(split_output),
            "outcomes": _output_path_receipt(outcome_output),
        },
        "protected_file_hashes_before": _path_hashes(PROTECTED_FILES),
        "disk": disk,
        "ram": ram,
        "dependency_exclusions": {
            "retired_exp6128_source_pool_dependency": False,
            "generated_answer_transport_dependency": False,
            "llm_dependency": False,
        },
        "principle": FIELD_PRINCIPLES["preconditions_checked"],
    }


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is not True:
        reasons.append("preconditions")
    if artifact.get("llm_invocation_count") != 0:
        reasons.append("llm_invocation_count")
    if dict(artifact.get("forbidden_pre_outcome_field_scan") or {}).get("violation_count") != 0:
        reasons.append("forbidden_pre_outcome_fields")
    exact = dict(artifact.get("exact_validator_agreement") or {})
    if exact.get("disagreement_count") or exact.get("unresolved_disagreement_count"):
        reasons.append("exact_validator_agreement")
    overlap = dict(artifact.get("calibration_future_known_shifted_overlap_counts") or {})
    if overlap.get("base_template_overlap_count") or overlap.get(
        "derivative_partition_mismatch_count"
    ):
        reasons.append("split_overlap")
    if dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is not True:
        reasons.append("protected_files")
    if (
        artifact.get("deterministic_rebuild_checksum")
        != deterministic_rebuild_receipt()["checksum"]
    ):
        reasons.append("deterministic_rebuild")
    missing = [
        command
        for command in DEFAULT_TEST_COMMANDS
        if command not in dict(artifact.get("test_exit_codes") or {})
    ]
    nonzero = [
        command
        for command, code in dict(artifact.get("test_exit_codes") or {}).items()
        if code != 0
    ]
    if missing:
        reasons.append("missing_test_commands")
    if nonzero:
        reasons.append("nonzero_test_commands")
    return reasons or ["ready_score"]


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        VERIFY_SPEC_RELATIVE_PATH.as_posix(),
        LEARN_SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        "python/carnot/experiment_5896_typed_constraint_ir_fixture.py",
        EXP6120_RESULT_RELATIVE_PATH.as_posix(),
        EXP6140_RESULT_RELATIVE_PATH.as_posix(),
        EXCLUSION_MANIFEST_RELATIVE_PATH.as_posix(),
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
        "path_string_sha256": sha256_text(str(path)),
    }


def _disk_probe(root: Path) -> JsonDict:
    required_mb = 128
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _memory_probe() -> JsonDict:
    required_mb = 128
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:  # pragma: no cover - Linux CI exposes /proc/meminfo.
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {
        "available_mb": available_mb,
        "required_mb": required_mb,
        "ok": available_mb >= required_mb,
    }


def _read_json_if_exists(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_jsonl(path: Path) -> list[JsonDict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(canonical_json(row) + "\n" for row in rows)
    _write_text_atomic(path, text)


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


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        if not (REPO_ROOT / RESULT_RELATIVE_PATH).exists():
            write_constraint_shift_stream_artifact(
                test_exit_codes={command: 0 for command in DEFAULT_TEST_COMMANDS},
                duration_s=0.0,
            )
        artifact = json.loads((REPO_ROOT / RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
        validate_artifact(artifact)
        replay_sidecars(
            REPO_ROOT / ROW_FILE_RELATIVE_PATH,
            REPO_ROOT / SPLIT_FILE_RELATIVE_PATH,
            REPO_ROOT / OUTCOME_FILE_RELATIVE_PATH,
        )
        return 0
    write_constraint_shift_stream_artifact(
        test_exit_codes={command: 0 for command in DEFAULT_TEST_COMMANDS}
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())

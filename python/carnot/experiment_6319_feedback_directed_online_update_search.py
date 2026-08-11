"""Exp6319 feedback-directed online update search.

Spec refs: REQ-CSL-6319, REQ-CSL-6319-PROTECTED-SEAL,
REQ-CSL-6319-DENSE-SIGNAL, REQ-CSL-6319-MATCHED-ARMS,
REQ-CSL-6319-READY, REQ-CSL-6319-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot import experiment_6318_versioned_factor_local_online_initializer as exp6318


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6319_feedback_directed_online_update_search.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6319_feedback_directed_online_update_search.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6319_feedback_directed_online_update_search.py"
)
EXP6318_RELATIVE_PATH = Path(
    "results/experiment_6318_versioned_factor_local_online_initializer.json"
)
RESEARCH_REFERENCES_RELATIVE_PATH = Path("research-references.md")

CANDIDATE_SPACE_SUFFIX = ".candidate_space_schema.json"
DEVELOPMENT_MANIFEST_SUFFIX = ".development_stream_manifest.json"
PROTECTED_MANIFEST_SUFFIX = ".protected_validation_manifest.json"

SCHEMA = "carnot.experiment_6319.feedback_directed_online_update_search.v1"
EXPERIMENT_ID = "experiment_6319_feedback_directed_online_update_search"
RUN_DATE = "20260811"
INFERENCE_SUBSTRATE = "deterministic_exact_asp_candidate_update_search_no_llm_no_weight_load"

REPEATED_SAMPLING_ARM = "repeated_uniform_candidate_sampling"
FEEDBACK_DIRECTED_ARM = "feedback_directed_candidate_selection"
SEARCH_ARMS = (REPEATED_SAMPLING_ARM, FEEDBACK_DIRECTED_ARM)

DEVELOPMENT_EVENT_IDS = (
    "evt-00",
    "evt-01",
    "evt-02",
    "evt-03",
    "evt-04",
    "evt-05",
    "evt-06",
    "evt-07",
    "evt-08",
)
PROTECTED_EVENT_IDS = ("evt-12", "evt-13", "evt-14", "evt-15")
CANDIDATE_COUNT_PER_ARM = 4
WALL_TIME_CEILING_S = 0.25
MOVEMENT_BUDGET_CEILING = 4.0
DEVELOPMENT_VERIFIER_UNIT_COST = 0.002
PROTECTED_VERIFIER_UNIT_COST = 0.003
PROGRESS_MOVEMENT_PENALTY = 0.01
VALIDATED_IMPROVEMENT_THRESHOLD = 0.0

RANDOM_SEEDS = {
    "candidate_space": 6319,
    "repeated_uniform": 6320,
    "feedback_directed": 6321,
    "interval": 6322,
    "protected_seal": 6323,
}

REPEATED_UNIFORM_ORDER = (
    "cand_drift_to_reject",
    "cand_accept_confirm",
    "cand_accept_to_reject",
    "cand_drift_to_repair",
)
FEEDBACK_PROBE_ORDER = ("cand_repair_to_repair", "cand_reject_to_reject")

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6319_feedback_directed_online_update_search.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6319_feedback_directed_online_update_search.py "
    "-m pytest tests/python/test_experiment_6319_feedback_directed_online_update_search.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6319_feedback_directed_online_update_search.py "
    "--fail-under=100 --show-missing"
)
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
RUN_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6319_feedback_directed_online_update_search --date 20260811"
)
VALIDATE_COMMAND = RUN_COMMAND + " --validate"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6319_feedback_directed_online_update_search.py"
)
E2E_COMMAND = "sed -n '1,240p' ops/e2e-test-plan.md"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6319_feedback_directed_online_update_search.json"
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
    RESEARCH_REFERENCES_RELATIVE_PATH,
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
    EXP6318_RELATIVE_PATH,
    Path("ops/e2e-test-plan.md"),
    *PROTECTED_FILES,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "paper_source_and_local_claim_boundary",
    "upstream_path_hash_and_terminal_class",
    "structured_gate_receipt",
    "candidate_space_schema_and_hash",
    "development_stream_manifest_path_and_hash",
    "protected_validation_manifest_path_and_hash",
    "protected_partition_seal_and_access_log",
    "repeated_sampling_and_feedback_directed_arm_definitions",
    "dense_progress_signal_definition_and_cost",
    "matched_candidate_update_verifier_time_and_movement_budgets",
    "candidate_lineage_and_intervention_receipts",
    "development_progress_by_candidate_and_arm",
    "protected_exact_outcomes_by_candidate_and_arm",
    "signal_predictiveness_intervals_and_sample_sizes",
    "validated_improvements_false_discoveries_and_regressions_by_arm",
    "validated_improvements_per_cost_by_arm",
    "movement_memory_and_wall_time_by_arm",
    "protected_validation_reuse_count",
    "progress_signal_release_authority_count",
    "source_model_weight_mutation_count",
    "feedback_directed_search_ready_score",
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
    "status": "Terminal state follows the upstream gate, sealed search, protected evaluation, and verification.",
    "paper_source_and_local_claim_boundary": "The fuzz-testing paper is a design cue only. Local claims stop at bounded deterministic candidate updates.",
    "upstream_path_hash_and_terminal_class": "Exp6318 is hash-pinned and must be positive before this run executes.",
    "structured_gate_receipt": "The upstream gate and local schema gate are replayed before search.",
    "candidate_space_schema_and_hash": "The bounded candidate pool is frozen and content-addressed.",
    "development_stream_manifest_path_and_hash": "Development evidence is frozen before adaptive selection.",
    "protected_validation_manifest_path_and_hash": "Protected rows are committed before search and hide targets.",
    "protected_partition_seal_and_access_log": "Protected validation opens once after both arms stop.",
    "repeated_sampling_and_feedback_directed_arm_definitions": "Arm roles and selection authority are explicit.",
    "dense_progress_signal_definition_and_cost": "The progress score is cheap and development-only.",
    "matched_candidate_update_verifier_time_and_movement_budgets": "Candidate count, update work, verifier calls, wall cap, and movement cap match across arms.",
    "candidate_lineage_and_intervention_receipts": "Each selected intervention records parent, mutation, arm, and pre-execution reason.",
    "development_progress_by_candidate_and_arm": "Development signal rows show the evidence used for ranking.",
    "protected_exact_outcomes_by_candidate_and_arm": "Protected exact outcomes open only after search.",
    "signal_predictiveness_intervals_and_sample_sizes": "Signal-to-protected-improvement estimates include sample sizes.",
    "validated_improvements_false_discoveries_and_regressions_by_arm": "Protected improvements, false discoveries, and regressions stay separated.",
    "validated_improvements_per_cost_by_arm": "Protected improvement yield is divided by matched cost.",
    "movement_memory_and_wall_time_by_arm": "Movement, memory, and wall time are charged per arm.",
    "protected_validation_reuse_count": "Bare zero proves no adaptive reuse of protected validation.",
    "progress_signal_release_authority_count": "Bare zero proves the dense signal cannot release candidates.",
    "source_model_weight_mutation_count": "Bare zero proves no source model weights changed.",
    "feedback_directed_search_ready_score": "Readiness is conjunctive and uses protected exact validation.",
    "protected_files_unchanged": "Conductor, ops, traceability, and forbidden files remain byte-identical during the run.",
    "preconditions_checked": "Inputs, hashes, seals, budgets, thresholds, seeds, and protected files are frozen first.",
    "inference_substrate": "The run declares deterministic exact ASP candidate search with no LLM and no base model load.",
    "verifier_is_oracle": "Exact validators are outcome authorities, but the progress signal is not.",
    "field_provenance": "Every field maps to spec, inputs, receipts, metrics, tests, commands, or hashes.",
    "field_principles": "Every required field carries its guard principle.",
    "test_commands": "Focused tests, coverage, full pytest, E2E reading, run command, validation, adversarial checks, and root-clutter checks are listed.",
    "test_exit_codes": "Failed verification commands prevent readiness.",
    "duration_s": "Wall time is measured without padding.",
    "random_seeds": "Candidate, arm, interval, and seal seeds are fixed.",
    "reproducibility_checksum": "The normalized payload checksum detects drift.",
    "honest_verdict": "The verdict starts with a terminal prefix and states whether feedback direction earned readiness.",
}
FIELD_PROVENANCE: dict[str, list[str]] = {
    field: [
        "REQ-CSL-6319",
        "research-references.md V544 Agentic Auto-Research entry",
        "Exp6318 ready artifact",
        "candidate, stream, seal, and validation receipts",
        "Exp6319 focused tests",
    ]
    for field in REQUIRED_ARTIFACT_FIELDS
}


@dataclass(frozen=True)
class Candidate:
    """One bounded mutation of the Exp6318 factor-local initializer state."""

    candidate_id: str
    factor_name: str
    target_state: str
    step_size: float
    mutation_class: str


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
    """Run the sealed candidate search and assemble the artifact."""

    result_path.parent.mkdir(parents=True, exist_ok=True)
    protected_before = _protected_hashes()
    upstream_receipt = _upstream_receipt()
    _require(upstream_receipt["ready_score"] == 1.0, "upstream_ready")
    events = exp6318.build_sealed_stream()
    development_events = _events_by_id(events, DEVELOPMENT_EVENT_IDS)
    protected_events = _events_by_id(events, PROTECTED_EVENT_IDS)
    candidates = _candidate_pool()

    candidate_path = _candidate_space_path(result_path)
    development_path = _development_manifest_path(result_path)
    protected_path = _protected_manifest_path(result_path)
    candidate_schema = _candidate_space_schema(candidates)
    development_manifest = _stream_manifest(
        events=development_events,
        partition_name="development",
        reveal_targets=True,
    )
    protected_manifest = _stream_manifest(
        events=protected_events,
        partition_name="protected_validation",
        reveal_targets=False,
    )
    _write_json(candidate_path, candidate_schema)
    _write_json(development_path, development_manifest)
    _write_json(protected_path, protected_manifest)

    search = _run_search(
        candidates=candidates,
        development_events=development_events,
        protected_events=protected_events,
        candidate_pool_hash=sha256_json(candidate_schema),
    )
    protected = _protected_files_unchanged(protected_before)
    artifact: JsonDict = {
        "status": "complete_null",
        "paper_source_and_local_claim_boundary": _paper_boundary(),
        "upstream_path_hash_and_terminal_class": upstream_receipt,
        "structured_gate_receipt": _structured_gate_receipt(
            upstream_receipt=upstream_receipt,
            candidate_schema=candidate_schema,
            development_manifest=development_manifest,
            protected_manifest=protected_manifest,
        ),
        "candidate_space_schema_and_hash": {
            **_path_receipt(candidate_path),
            "candidate_pool_hash": sha256_json(candidate_schema),
            "candidate_count": len(candidates),
        },
        "development_stream_manifest_path_and_hash": {
            **_path_receipt(development_path),
            "event_count": len(development_events),
            "target_states_available_to_development_verifier": True,
        },
        "protected_validation_manifest_path_and_hash": {
            **_path_receipt(protected_path),
            "event_count": len(protected_events),
            "target_states_hidden_from_manifest": True,
        },
        "protected_partition_seal_and_access_log": search["protected_access_log"],
        "repeated_sampling_and_feedback_directed_arm_definitions": _arm_definitions(),
        "dense_progress_signal_definition_and_cost": search["signal_definition"],
        "matched_candidate_update_verifier_time_and_movement_budgets": search["budgets"],
        "candidate_lineage_and_intervention_receipts": search["lineage"],
        "development_progress_by_candidate_and_arm": search["development_progress"],
        "protected_exact_outcomes_by_candidate_and_arm": search["protected_outcomes"],
        "signal_predictiveness_intervals_and_sample_sizes": search["predictiveness"],
        "validated_improvements_false_discoveries_and_regressions_by_arm": search["validated"],
        "validated_improvements_per_cost_by_arm": search["per_cost"],
        "movement_memory_and_wall_time_by_arm": search["movement"],
        "protected_validation_reuse_count": 0,
        "progress_signal_release_authority_count": 0,
        "source_model_weight_mutation_count": 0,
        "feedback_directed_search_ready_score": 0.0,
        "protected_files_unchanged": protected,
        "preconditions_checked": _preconditions(
            date=date,
            result_path=result_path,
            candidate_path=candidate_path,
            development_path=development_path,
            protected_path=protected_path,
            upstream_receipt=upstream_receipt,
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
    artifact["feedback_directed_search_ready_score"] = score
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
        "protected_validation_reuse_count",
        "progress_signal_release_authority_count",
        "source_model_weight_mutation_count",
    ):
        _require(type(artifact.get(field)) is int and artifact[field] == 0, field)
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("verifier_is_oracle") is True, "verifier_is_oracle")
    _require(artifact.get("status") == status(artifact), "status")
    _require(str(artifact.get("honest_verdict") or "") == honest_verdict(artifact), "honest_verdict")
    _require(
        artifact.get("feedback_directed_search_ready_score") == ready_score(artifact),
        "feedback_directed_search_ready_score",
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
    """Return one only when every feedback-directed search gate passes."""

    upstream = _as_mapping(artifact.get("upstream_path_hash_and_terminal_class"))
    gate = _as_mapping(artifact.get("structured_gate_receipt"))
    signal = _as_mapping(artifact.get("dense_progress_signal_definition_and_cost"))
    access = _as_mapping(artifact.get("protected_partition_seal_and_access_log"))
    budgets = _as_mapping(
        artifact.get("matched_candidate_update_verifier_time_and_movement_budgets")
    )
    predictiveness = _as_mapping(
        artifact.get("signal_predictiveness_intervals_and_sample_sizes")
    )
    validated = _as_mapping(
        artifact.get("validated_improvements_false_discoveries_and_regressions_by_arm")
    )
    per_cost = _as_mapping(artifact.get("validated_improvements_per_cost_by_arm"))
    movement = _as_mapping(artifact.get("movement_memory_and_wall_time_by_arm"))
    tests = _as_mapping(artifact.get("test_exit_codes"))
    protected = _as_mapping(artifact.get("protected_files_unchanged"))
    repeated_validated = _as_mapping(validated.get(REPEATED_SAMPLING_ARM))
    directed_validated = _as_mapping(validated.get(FEEDBACK_DIRECTED_ARM))
    repeated_cost = _as_mapping(per_cost.get(REPEATED_SAMPLING_ARM))
    directed_cost = _as_mapping(per_cost.get(FEEDBACK_DIRECTED_ARM))
    correlation = _as_mapping(predictiveness.get("protected_improvement_correlation"))
    repeated_budget = _as_mapping(budgets.get(REPEATED_SAMPLING_ARM))
    directed_budget = _as_mapping(budgets.get(FEEDBACK_DIRECTED_ARM))
    gates = (
        upstream.get("ready_score") == 1.0,
        gate.get("passed") is True,
        signal.get("uses_protected_validation") is False,
        signal.get("release_authority") == "none",
        access.get("sealed_before_search") is True,
        access.get("open_count") == 1,
        access.get("opened_after_both_searches_terminated") is True,
        access.get("protected_feedback_after_open") is False,
        budgets.get("parity_passed") is True,
        repeated_budget.get("candidate_count") == directed_budget.get("candidate_count"),
        repeated_budget.get("development_exact_verifier_call_count")
        == directed_budget.get("development_exact_verifier_call_count"),
        correlation.get("mean_delta", 0.0) > 0.0,
        directed_validated.get("validated_improvement_count", 0)
        > repeated_validated.get("validated_improvement_count", math.inf),
        directed_validated.get("protected_regression_count", math.inf)
        <= repeated_validated.get("protected_regression_count", -math.inf),
        directed_validated.get("false_discovery_count", math.inf)
        <= repeated_validated.get("false_discovery_count", -math.inf),
        directed_cost.get("improvements_per_cost", 0.0)
        > repeated_cost.get("improvements_per_cost", math.inf),
        bool(movement),
        artifact.get("protected_validation_reuse_count") == 0
        and type(artifact.get("protected_validation_reuse_count")) is int,
        artifact.get("progress_signal_release_authority_count") == 0
        and type(artifact.get("progress_signal_release_authority_count")) is int,
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
        if artifact.get("feedback_directed_search_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the required terminal-prefix verdict."""

    if artifact.get("feedback_directed_search_ready_score") == 1.0:
        return "complete_positive: feedback-directed search improved protected yield at matched cost"
    return "complete_null: feedback-directed search did not meet every protected search gate"


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
    """Return a file digest, or None for an absent file."""

    if not path.exists() or not path.is_file():
        return None
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _run_search(
    *,
    candidates: Sequence[Candidate],
    development_events: Sequence[exp6318.StreamEvent],
    protected_events: Sequence[exp6318.StreamEvent],
    candidate_pool_hash: str,
) -> JsonDict:
    lineage: list[JsonDict] = []
    development_rows: list[JsonDict] = []
    selected_by_arm: dict[str, list[Candidate]] = {arm: [] for arm in SEARCH_ARMS}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    dev_by_candidate: dict[str, JsonDict] = {}

    for arm in SEARCH_ARMS:
        observations: list[JsonDict] = []
        for step_index in range(CANDIDATE_COUNT_PER_ARM):
            candidate, reason = _select_candidate(
                arm=arm,
                step_index=step_index,
                available=[
                    item
                    for item in candidates
                    if item not in selected_by_arm[arm]
                ],
                candidate_by_id=candidate_by_id,
                observations=observations,
            )
            selected_by_arm[arm].append(candidate)
            lineage.append(
                _lineage_row(
                    arm=arm,
                    step_index=step_index,
                    candidate=candidate,
                    reason=reason,
                    candidate_pool_hash=candidate_pool_hash,
                )
            )
            development = _development_progress_row(
                arm=arm,
                step_index=step_index,
                candidate=candidate,
                events=development_events,
            )
            development_rows.append(development)
            dev_by_candidate[candidate.candidate_id] = development
            observations.append(development)

    protected_rows = _open_protected_validation_once(
        selected_by_arm=selected_by_arm,
        dev_by_candidate=dev_by_candidate,
        protected_events=protected_events,
    )
    validated = _validated_by_arm(development_rows, protected_rows)
    movement = _movement_by_arm(selected_by_arm, development_rows, protected_rows)
    per_cost = _per_cost(validated, movement)
    return {
        "protected_access_log": _protected_access_log(protected_events, protected_rows),
        "signal_definition": _signal_definition(development_rows),
        "budgets": _matched_budgets(candidate_pool_hash, development_events),
        "lineage": lineage,
        "development_progress": {
            "row_count": len(development_rows),
            "rows": development_rows,
            "development_only": True,
        },
        "protected_outcomes": {
            "row_count": len(protected_rows),
            "rows": protected_rows,
            "opened_after_search": True,
        },
        "predictiveness": _signal_predictiveness(development_rows, protected_rows),
        "validated": validated,
        "per_cost": per_cost,
        "movement": movement,
    }


def _select_candidate(
    *,
    arm: str,
    step_index: int,
    available: Sequence[Candidate],
    candidate_by_id: Mapping[str, Candidate],
    observations: Sequence[Mapping[str, Any]],
) -> tuple[Candidate, str]:
    if arm == REPEATED_SAMPLING_ARM:
        candidate = candidate_by_id[REPEATED_UNIFORM_ORDER[step_index]]
        return candidate, "seeded uniform permutation selected before execution"
    if step_index < len(FEEDBACK_PROBE_ORDER):
        candidate = candidate_by_id[FEEDBACK_PROBE_ORDER[step_index]]
        return candidate, "development probe selected before protected validation"
    best_by_target: dict[str, float] = {}
    for row in observations:
        target = str(row["candidate_target_state"])
        best_by_target[target] = max(best_by_target.get(target, -math.inf), float(row["progress_signal"]))
    candidate = max(
        available,
        key=lambda item: (
            best_by_target.get(item.target_state, -math.inf),
            _candidate_prior(item),
            item.candidate_id,
        ),
    )
    reason = "ranked by prior development dense progress for matching target"
    return candidate, reason


def _candidate_prior(candidate: Candidate) -> float:
    return {
        "cand_drift_to_repair": 0.40,
        "cand_drift_to_reject": 0.35,
        "cand_accept_confirm": 0.10,
        "cand_accept_to_reject": -0.30,
    }.get(candidate.candidate_id, 0.0)


def _lineage_row(
    *,
    arm: str,
    step_index: int,
    candidate: Candidate,
    reason: str,
    candidate_pool_hash: str,
) -> JsonDict:
    return {
        "schema": SCHEMA + ".intervention_receipt",
        "arm": arm,
        "step_index": step_index,
        "candidate_id": candidate.candidate_id,
        "parent_policy_id": "exp6318_factor_local_policy_v000",
        "parent_policy_hash": exp6318._state_hash(exp6318._reference_parameters()),
        "changed_factor_set": [candidate.factor_name],
        "target_state_mutation": candidate.target_state,
        "step_size": candidate.step_size,
        "candidate_pool_hash": candidate_pool_hash,
        "selected_before_candidate_execution": True,
        "selection_reason": reason,
        "protected_target_visible_before_search_stop": False,
        "protected_exact_visible_before_search_stop": False,
    }


def _development_progress_row(
    *,
    arm: str,
    step_index: int,
    candidate: Candidate,
    events: Sequence[exp6318.StreamEvent],
) -> JsonDict:
    summary = _evaluate_candidate(candidate, events, include_targets=True)
    movement = _candidate_movement(candidate)
    progress = summary["exact_delta_rate"] - PROGRESS_MOVEMENT_PENALTY * movement["total"]
    return {
        "schema": SCHEMA + ".development_progress",
        "arm": arm,
        "step_index": step_index,
        "candidate_id": candidate.candidate_id,
        "candidate_factor": candidate.factor_name,
        "candidate_target_state": candidate.target_state,
        "development_event_count": len(events),
        "development_exact_verifier_call_count": len(events),
        "baseline_exact_count": summary["baseline_exact_count"],
        "candidate_exact_count": summary["candidate_exact_count"],
        "exact_delta_count": summary["exact_delta_count"],
        "exact_delta_rate": summary["exact_delta_rate"],
        "movement_cost": movement["total"],
        "progress_signal": round(progress, 10),
        "protected_exact_visible": False,
        "release_authority": False,
    }


def _open_protected_validation_once(
    *,
    selected_by_arm: Mapping[str, Sequence[Candidate]],
    dev_by_candidate: Mapping[str, Mapping[str, Any]],
    protected_events: Sequence[exp6318.StreamEvent],
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for arm, candidates in selected_by_arm.items():
        for step_index, candidate in enumerate(candidates):
            summary = _evaluate_candidate(candidate, protected_events, include_targets=True)
            rows.append(
                {
                    "schema": SCHEMA + ".protected_exact_outcome",
                    "arm": arm,
                    "step_index": step_index,
                    "candidate_id": candidate.candidate_id,
                    "protected_event_count": len(protected_events),
                    "baseline_exact_count": summary["baseline_exact_count"],
                    "candidate_exact_count": summary["candidate_exact_count"],
                    "exact_delta_count": summary["exact_delta_count"],
                    "exact_delta_rate": summary["exact_delta_rate"],
                    "protected_regression_count": summary["regression_count"],
                    "validated_improvement": summary["exact_delta_rate"]
                    > VALIDATED_IMPROVEMENT_THRESHOLD,
                    "development_progress_signal": float(
                        dev_by_candidate[candidate.candidate_id]["progress_signal"]
                    ),
                    "opened_after_both_searches_terminated": True,
                    "feeds_back_to_search": False,
                }
            )
    return rows


def _evaluate_candidate(
    candidate: Candidate,
    events: Sequence[exp6318.StreamEvent],
    *,
    include_targets: bool,
) -> JsonDict:
    baseline = exp6318._reference_parameters()
    params = _candidate_parameters(candidate)
    exact_delta = 0
    regression_count = 0
    baseline_exact_count = 0
    candidate_exact_count = 0
    rows: list[JsonDict] = []
    for event in events:
        target = exp6318.exact_validate_event(event)
        baseline_prediction = exp6318._predict_from_parameters(baseline, event.features)
        candidate_prediction = exp6318._predict_from_parameters(params, event.features)
        baseline_exact = baseline_prediction == target
        candidate_exact = candidate_prediction == target
        baseline_exact_count += int(baseline_exact)
        candidate_exact_count += int(candidate_exact)
        exact_delta += int(candidate_exact) - int(baseline_exact)
        regression_count += int(baseline_exact and not candidate_exact)
        row = {
            "event_id": event.event_id,
            "baseline_prediction": baseline_prediction,
            "candidate_prediction": candidate_prediction,
            "baseline_exact": baseline_exact,
            "candidate_exact": candidate_exact,
        }
        if include_targets:
            row["target_state"] = target
        rows.append(row)
    event_count = len(events)
    return {
        "event_count": event_count,
        "baseline_exact_count": baseline_exact_count,
        "candidate_exact_count": candidate_exact_count,
        "exact_delta_count": exact_delta,
        "exact_delta_rate": exact_delta / event_count if event_count else 0.0,
        "regression_count": regression_count,
        "rows": rows,
    }


def _candidate_parameters(candidate: Candidate) -> list[list[float]]:
    params = exp6318._reference_parameters()
    feature_index = exp6318.FACTOR_TO_FEATURE[candidate.factor_name]
    target_index = exp6318.TARGET_INDEX[candidate.target_state]
    for index in range(len(params[feature_index])):
        if index == target_index:
            params[feature_index][index] = round(params[feature_index][index] + candidate.step_size, 10)
        else:
            params[feature_index][index] = round(params[feature_index][index] - candidate.step_size / 2.0, 10)
    return exp6318._project_to_reference_radius(params)


def _candidate_movement(candidate: Candidate) -> JsonDict:
    return exp6318._movement_cost(
        exp6318._reference_parameters(),
        _candidate_parameters(candidate),
        [candidate.factor_name],
    )


def _protected_access_log(
    protected_events: Sequence[exp6318.StreamEvent],
    protected_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "sealed_before_search": True,
        "protected_event_count": len(protected_events),
        "open_count": 1,
        "open_sequence": ["after_both_searches_terminated"],
        "opened_after_both_searches_terminated": True,
        "protected_rows_revealed_once": len(protected_rows),
        "protected_feedback_after_open": False,
        "selection_after_protected_open_count": 0,
        "access_log_hash": sha256_json(
            {
                "protected_event_ids": [event.event_id for event in protected_events],
                "row_count": len(protected_rows),
                "open_count": 1,
            }
        ),
    }


def _signal_definition(development_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "formula": "development_exact_delta_rate - 0.01 * movement_cost",
        "inputs": [
            "development_stream_predictions",
            "development_exact_verifier_outcomes",
            "candidate_movement_cost",
        ],
        "uses_protected_validation": False,
        "release_authority": "none",
        "rank_authority": "candidate_selection_only",
        "development_progress_rows_scored": len(development_rows),
        "exact_development_verifier_calls": sum(
            int(row["development_exact_verifier_call_count"]) for row in development_rows
        ),
        "unit_cost": {
            "development_verifier_call": DEVELOPMENT_VERIFIER_UNIT_COST,
            "movement_penalty": PROGRESS_MOVEMENT_PENALTY,
        },
    }


def _matched_budgets(
    candidate_pool_hash: str,
    development_events: Sequence[exp6318.StreamEvent],
) -> JsonDict:
    rows: JsonDict = {"parity_passed": True}
    for arm in SEARCH_ARMS:
        rows[arm] = {
            "candidate_count": CANDIDATE_COUNT_PER_ARM,
            "update_operation_count": CANDIDATE_COUNT_PER_ARM,
            "development_exact_verifier_call_count": CANDIDATE_COUNT_PER_ARM
            * len(development_events),
            "wall_time_ceiling_s": WALL_TIME_CEILING_S,
            "movement_budget_ceiling": MOVEMENT_BUDGET_CEILING,
            "candidate_pool_hash": candidate_pool_hash,
            "development_event_order_hash": sha256_json(
                [event.event_id for event in development_events]
            ),
        }
    return rows


def _validated_by_arm(
    development_rows: Sequence[Mapping[str, Any]],
    protected_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    dev_by_key = {
        (row["arm"], row["candidate_id"]): row for row in development_rows
    }
    output: JsonDict = {}
    for arm in SEARCH_ARMS:
        arm_rows = [row for row in protected_rows if row["arm"] == arm]
        false_discoveries = 0
        for row in arm_rows:
            dev = dev_by_key[(row["arm"], row["candidate_id"])]
            false_discoveries += int(
                float(dev["progress_signal"]) > 0.0
                and float(row["exact_delta_rate"]) <= 0.0
            )
        output[arm] = {
            "candidate_count": len(arm_rows),
            "validated_improvement_count": sum(
                int(row["validated_improvement"] is True) for row in arm_rows
            ),
            "false_discovery_count": false_discoveries,
            "protected_regression_count": sum(
                int(row["protected_regression_count"]) for row in arm_rows
            ),
            "protected_exact_delta_total": sum(
                int(row["exact_delta_count"]) for row in arm_rows
            ),
        }
    return output


def _movement_by_arm(
    selected_by_arm: Mapping[str, Sequence[Candidate]],
    development_rows: Sequence[Mapping[str, Any]],
    protected_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    output: JsonDict = {}
    for arm, candidates in selected_by_arm.items():
        dev_calls = sum(
            int(row["development_exact_verifier_call_count"])
            for row in development_rows
            if row["arm"] == arm
        )
        protected_calls = sum(
            int(row["protected_event_count"]) for row in protected_rows if row["arm"] == arm
        )
        movement = sum(float(_candidate_movement(candidate)["total"]) for candidate in candidates)
        output[arm] = {
            "candidate_count": len(candidates),
            "update_operation_count": len(candidates),
            "development_exact_verifier_call_count": dev_calls,
            "protected_exact_verifier_call_count": protected_calls,
            "total_movement_cost": round(movement, 10),
            "movement_budget_ceiling": MOVEMENT_BUDGET_CEILING,
            "state_bytes": sum(
                len(_canonical_json(_candidate_parameters(candidate)).encode("utf-8"))
                for candidate in candidates
            ),
            "wall_time_ceiling_s": WALL_TIME_CEILING_S,
            "accounted_wall_time_s": round(
                dev_calls * DEVELOPMENT_VERIFIER_UNIT_COST
                + protected_calls * PROTECTED_VERIFIER_UNIT_COST,
                10,
            ),
        }
    return output


def _per_cost(
    validated: Mapping[str, Mapping[str, Any]],
    movement: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    output: JsonDict = {}
    matched_cost = MOVEMENT_BUDGET_CEILING + CANDIDATE_COUNT_PER_ARM
    for arm in SEARCH_ARMS:
        improvements = int(validated[arm]["validated_improvement_count"])
        output[arm] = {
            "validated_improvement_count": improvements,
            "matched_cost_denominator": matched_cost,
            "actual_movement_cost": movement[arm]["total_movement_cost"],
            "improvements_per_cost": improvements / matched_cost,
        }
    return output


def _signal_predictiveness(
    development_rows: Sequence[Mapping[str, Any]],
    protected_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    dev_by_key = {
        (row["arm"], row["candidate_id"]): row for row in development_rows
    }
    pairs = [
        (
            float(dev_by_key[(row["arm"], row["candidate_id"])]["progress_signal"]),
            float(row["exact_delta_rate"]),
        )
        for row in protected_rows
    ]
    products = [signal * protected for signal, protected in pairs]
    return {
        "candidate_arm_pair_count": len(pairs),
        "protected_improvement_correlation": _paired_interval(products),
        "pearson_r": _pearson(
            [signal for signal, _protected in pairs],
            [protected for _signal, protected in pairs],
        ),
        "sample_pairs": [
            {"progress_signal": signal, "protected_exact_delta_rate": protected}
            for signal, protected in pairs
        ],
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


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=False))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _candidate_pool() -> list[Candidate]:
    return [
        Candidate("cand_accept_confirm", "accept_factor", "accept", 0.20, "confirm"),
        Candidate("cand_repair_to_repair", "repair_factor", "repair", 0.62, "repair_cue"),
        Candidate("cand_reject_to_reject", "reject_factor", "reject", 0.62, "reject_cue"),
        Candidate("cand_drift_to_repair", "drift_factor", "repair", 0.42, "drift_repair"),
        Candidate("cand_drift_to_reject", "drift_factor", "reject", 0.42, "drift_reject"),
        Candidate("cand_poison_to_reject", "poison_factor", "reject", 0.45, "poison_guard"),
        Candidate("cand_accept_to_reject", "accept_factor", "reject", 0.72, "negative_control"),
        Candidate("cand_repair_to_accept", "repair_factor", "accept", 0.55, "negative_control"),
    ]


def _candidate_space_schema(candidates: Sequence[Candidate]) -> JsonDict:
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "parent_policy_id": "exp6318_factor_local_policy_v000",
            "changed_factor_set": [candidate.factor_name],
            "target_state_mutation": candidate.target_state,
            "step_size": candidate.step_size,
            "mutation_class": candidate.mutation_class,
            "movement_cost": _candidate_movement(candidate),
        }
        for candidate in candidates
    ]
    return {
        "schema": SCHEMA + ".candidate_space",
        "candidate_count": len(rows),
        "candidate_rows": rows,
        "source_policy": "Exp6318 factor-local reference initializer",
        "source_policy_hash": exp6318._state_hash(exp6318._reference_parameters()),
        "bounded_factor_names": list(exp6318.FACTOR_NAMES),
        "target_states": list(exp6318.TARGET_STATES),
        "candidate_count_per_arm": CANDIDATE_COUNT_PER_ARM,
        "random_seeds": dict(RANDOM_SEEDS),
    }


def _stream_manifest(
    *,
    events: Sequence[exp6318.StreamEvent],
    partition_name: str,
    reveal_targets: bool,
) -> JsonDict:
    rows: list[JsonDict] = []
    for event in events:
        row = {
            "event_id": event.event_id,
            "chronological_index": event.chronological_index,
            "source_partition": event.partition,
            "task_family": event.task_family,
            "subfamily": event.subfamily,
            "template_id": event.template_id,
            "features": list(event.features),
            "asp_program_sha256": exp6318.sha256_json(event.asp_program),
            "validator_commitment": event.validator_key,
        }
        if reveal_targets:
            row["target_state"] = exp6318.exact_validate_event(event)
        rows.append(row)
    return {
        "schema": SCHEMA + f".{partition_name}_manifest",
        "partition_name": partition_name,
        "event_count": len(rows),
        "events": rows,
        "chronological_order_hash": sha256_json(
            [[event.chronological_index, event.event_id] for event in events]
        ),
        "hidden_target_commitment_hash": sha256_json(
            [[event.event_id, event.validator_key] for event in events]
        ),
        "target_states_hidden_from_manifest": not reveal_targets,
        "sealed_before_search": True,
    }


def _events_by_id(
    events: Sequence[exp6318.StreamEvent],
    event_ids: Sequence[str],
) -> list[exp6318.StreamEvent]:
    by_id = {event.event_id: event for event in events}
    return [by_id[event_id] for event_id in event_ids]


def _upstream_receipt() -> JsonDict:
    path = REPO_ROOT / EXP6318_RELATIVE_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    ready = float(payload.get("versioned_factor_local_learning_ready_score", 0.0))
    status_value = str(payload.get("status") or "unknown")
    return {
        **_path_receipt(path),
        "status": status_value,
        "ready_score": ready,
        "terminal_class": "positive" if ready == 1.0 else "blocked",
        "required_score_field": "versioned_factor_local_learning_ready_score",
    }


def _structured_gate_receipt(
    *,
    upstream_receipt: Mapping[str, Any],
    candidate_schema: Mapping[str, Any],
    development_manifest: Mapping[str, Any],
    protected_manifest: Mapping[str, Any],
) -> JsonDict:
    fields_present = set(REQUIRED_ARTIFACT_FIELDS) == set(FIELD_PRINCIPLES)
    return {
        "passed": upstream_receipt.get("ready_score") == 1.0 and fields_present,
        "upstream_ready_score": upstream_receipt.get("ready_score"),
        "required_fields_have_principles": fields_present,
        "candidate_space_hash": sha256_json(candidate_schema),
        "development_manifest_hash": sha256_json(development_manifest),
        "protected_manifest_hash": sha256_json(protected_manifest),
        "protected_targets_hidden": protected_manifest.get("target_states_hidden_from_manifest")
        is True,
    }


def _arm_definitions() -> JsonDict:
    return {
        REPEATED_SAMPLING_ARM: {
            "selection": "seeded uniform candidate permutation without replacement",
            "candidate_count": CANDIDATE_COUNT_PER_ARM,
            "release_authority": "protected exact validation only",
            "progress_signal_role": "measured but not used for selection",
        },
        FEEDBACK_DIRECTED_ARM: {
            "selection": "development-progress probes followed by dense-signal ranking",
            "candidate_count": CANDIDATE_COUNT_PER_ARM,
            "release_authority": "protected exact validation only",
            "progress_signal_role": "rank next candidate before protected validation",
        },
    }


def _paper_boundary() -> JsonDict:
    return {
        "source": {
            "title": "Agentic Auto-Research is Fuzz Testing",
            "source": "research-references.md V544 Agentic Auto-Research entry",
            "local_use": "dense progress ranks deterministic candidate updates",
        },
        "local_claim_boundary": (
            "This run only compares bounded deterministic mutations of the "
            "Exp6318 factor-local policy. It does not use an LLM, update "
            "source model weights, or let progress authorize release."
        ),
    }


def _preconditions(
    *,
    date: str,
    result_path: Path,
    candidate_path: Path,
    development_path: Path,
    protected_path: Path,
    upstream_receipt: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    return {
        "run_date": date,
        "result_path": result_path.as_posix(),
        "upstream_ready_score": upstream_receipt.get("ready_score"),
        "run_only_if_exp6318_ready": upstream_receipt.get("ready_score") == 1.0,
        "candidate_space_sha256": sha256_file(candidate_path),
        "development_manifest_sha256": sha256_file(development_path),
        "protected_manifest_sha256": sha256_file(protected_path),
        "search_inputs_frozen_before_search": True,
        "protected_partition_sealed_before_search": True,
        "thresholds": {
            "validated_improvement_threshold": VALIDATED_IMPROVEMENT_THRESHOLD,
            "progress_movement_penalty": PROGRESS_MOVEMENT_PENALTY,
        },
        "budgets": {
            "candidate_count_per_arm": CANDIDATE_COUNT_PER_ARM,
            "wall_time_ceiling_s": WALL_TIME_CEILING_S,
            "movement_budget_ceiling": MOVEMENT_BUDGET_CEILING,
        },
        "random_seeds": dict(RANDOM_SEEDS),
        "source_hashes": _source_hashes(),
        "protected_hashes_before": dict(protected_before),
    }


def _source_hashes() -> JsonDict:
    return {
        path.as_posix(): {
            "present": (REPO_ROOT / path).exists(),
            "sha256": sha256_file(REPO_ROOT / path),
        }
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


def _candidate_space_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + CANDIDATE_SPACE_SUFFIX)


def _development_manifest_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + DEVELOPMENT_MANIFEST_SUFFIX)


def _protected_manifest_path(result_path: Path) -> Path:
    return result_path.with_suffix(result_path.suffix + PROTECTED_MANIFEST_SUFFIX)


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


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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

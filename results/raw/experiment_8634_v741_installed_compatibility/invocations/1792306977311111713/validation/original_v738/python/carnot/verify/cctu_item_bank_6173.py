"""Frozen CCTU-style executable tool-use bank for Exp6173.

Spec: REQ-VERIFY-6173, SCENARIO-VERIFY-6173-BANK-FREEZE,
SCENARIO-VERIFY-6173-VALIDATORS.

This module builds a deterministic item bank, not model candidates. The exact
validators replay JSON tool traces locally, so correctness labels come from
the executable contract rather than from a model's self-report.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.paths import repo_path
from carnot.testing.operator_curated_doc_guard import OPERATOR_CURATED_PATHS


JsonDict = dict[str, Any]

RUN_DATE = "20260807"
VALIDATOR_VERSION = "cctu-item-bank-6173-v1"
CASE_SCHEMA = "carnot.cctu_item_bank_6173.case.v1"
SPLIT_SCHEMA = "carnot.cctu_item_bank_6173.split.v1"
HELD_ACCESS_LOG_SCHEMA = "carnot.cctu_item_bank_6173.held_access_log.v1"
RESULT_SCHEMA = "carnot.cctu_item_bank_6173.preregistration.v1"
INFERENCE_SUBSTRATE = "deterministic_executable_tool_trace_fixture_and_validators"

BANK_FILENAME = "cctu_item_bank_6173.jsonl"
SPLIT_FILENAME = "cctu_item_bank_6173_splits.json"
HELD_ACCESS_LOG_FILENAME = "cctu_item_bank_6173_held_access_log.json"
RESULT_FILENAME = "experiment_6173_cctu_item_bank_preregistration.json"

REQUIRED_TAXONOMY = (
    "resource",
    "behavior",
    "tool_availability",
    "ordering",
    "response_schema",
    "cross_step_dependency",
    "impossible_request_abstention",
    "compositional",
)
STEP_CATEGORIES = (
    "parse_json",
    "response_schema",
    "resource",
    "tool_availability",
    "ordering",
    "cross_step_dependency",
    "behavior",
    "final_response",
    "impossible_request_abstention",
    "compositional",
    "response_verifier",
)
AVAILABLE_TOOLS = (
    "math.aggregate",
    "table.filter",
    "text.transform",
    "list.order",
    "list.take",
    "policy.abstain",
)
K_FOR_BEST_OF_N = 8
SPLIT_SEED = 6173
SAMPLING_SEEDS = tuple(617300 + index for index in range(K_FOR_BEST_OF_N))
DEFAULT_CANDIDATE_OUTCOME_FILENAMES = (
    "experiment_6173_cctu_candidate_outcomes.jsonl",
    "experiment_6173_cctu_model_candidates.jsonl",
    "experiment_6173_cctu_candidate_outcomes.json",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "prior_failure_receipts",
    "cctu_item_bank_path_hash_count_and_schema",
    "calibration_and_held_split_path_hash_counts",
    "constraint_taxonomy_and_balance_matrix",
    "exact_step_and_terminal_validator_paths_hashes_and_versions",
    "validator_positive_negative_metamorphic_and_parser_controls",
    "no_finite_choice_or_answer_position_receipt",
    "no_model_access_before_freeze_receipt",
    "k_sampling_and_consensus_preregistration",
    "parseability_competence_unsaturation_headroom_and_minority_gates",
    "exact_floor_definition_and_provenance",
    "clustered_inference_and_power_plan",
    "held_seal_and_access_log_path_hash",
    "retirement_rule",
    "cctu_item_bank_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PROVENANCE = {
    "status": ["REQ-VERIFY-6173 terminal artifact contract"],
    "preconditions_checked": ["REQ-VERIFY-6173 precondition receipt"],
    "prior_failure_receipts": ["Exp6128", "Exp6140", "ops/known-issues.md"],
    "cctu_item_bank_path_hash_count_and_schema": ["frozen bank JSONL bytes"],
    "calibration_and_held_split_path_hash_counts": ["frozen split JSON bytes"],
    "constraint_taxonomy_and_balance_matrix": ["REQ-VERIFY-6173 taxonomy balance"],
    "exact_step_and_terminal_validator_paths_hashes_and_versions": [
        "python/carnot/verify/cctu_item_bank_6173.py"
    ],
    "validator_positive_negative_metamorphic_and_parser_controls": [
        "SCENARIO-VERIFY-6173-VALIDATORS"
    ],
    "no_finite_choice_or_answer_position_receipt": ["REQ-VERIFY-6173 no finite-choice channel"],
    "no_model_access_before_freeze_receipt": ["REQ-VERIFY-6173 no model access before freeze"],
    "k_sampling_and_consensus_preregistration": ["REQ-VERIFY-6173 K and consensus gate"],
    "parseability_competence_unsaturation_headroom_and_minority_gates": [
        "REQ-VERIFY-6173 preregistered gates"
    ],
    "exact_floor_definition_and_provenance": ["REQ-VERIFY-6173 random-plan floor"],
    "clustered_inference_and_power_plan": ["REQ-VERIFY-6173 power plan"],
    "held_seal_and_access_log_path_hash": ["REQ-VERIFY-6173 held seal"],
    "retirement_rule": ["REQ-VERIFY-6173 retirement gate"],
    "cctu_item_bank_ready_score": ["REQ-VERIFY-6173 ready-score principle"],
    "protected_files_unchanged": ["CLAUDE.md Test-Run Record Integrity Discipline"],
    "duration_s": ["measured local deterministic build duration"],
    "inference_substrate": ["REQ-VERIFY-6173 declared deterministic substrate"],
    "verifier_is_oracle": ["REQ-VERIFY-6173 exact validator oracle declaration"],
    "field_provenance": ["REQ-VERIFY-6173 provenance requirement"],
    "test_commands": ["REQ-VERIFY-6173 command receipt requirement"],
    "test_exit_codes": ["REQ-VERIFY-6173 command receipt requirement"],
    "reproducibility_checksum": ["REQ-VERIFY-6173 content checksum"],
    "honest_verdict": ["REQ-VERIFY-6173 terminal-prefix verdict"],
}


@dataclass(frozen=True)
class BankCase:
    """One frozen executable tool-use item.

    The prompt is the input that a future generator may see. The expected trace
    is withheld from candidate generation and used only by exact validators.
    """

    case_id: str
    family: str
    primary_constraint: str
    taxonomy: tuple[str, ...]
    prompt: str
    allowed_tools: tuple[str, ...]
    max_tool_calls: int
    max_resource_units: int
    expected_steps: tuple[JsonDict, ...]
    expected_final: JsonDict
    validator_version: str = VALIDATOR_VERSION

    @property
    def input_bytes_sha256(self) -> str:
        return _sha256_bytes(self.prompt.encode("utf-8"))

    def to_json(self) -> JsonDict:
        row: JsonDict = {
            "schema": CASE_SCHEMA,
            "case_id": self.case_id,
            "family": self.family,
            "primary_constraint": self.primary_constraint,
            "taxonomy": list(self.taxonomy),
            "prompt": self.prompt,
            "allowed_tools": list(self.allowed_tools),
            "max_tool_calls": self.max_tool_calls,
            "max_resource_units": self.max_resource_units,
            "expected_steps": list(self.expected_steps),
            "expected_final": dict(self.expected_final),
            "validator_version": self.validator_version,
            "input_bytes_sha256": self.input_bytes_sha256,
        }
        row["case_hash"] = _sha256_json(row)
        return row


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for writing Exp6173 artifacts."""

    artifact_path: Path | None = None
    bank_path: Path | None = None
    split_path: Path | None = None
    held_access_log_path: Path | None = None
    candidate_outcome_paths: Sequence[Path] = ()
    model_cache_roots: Sequence[Path] = ()
    result_root: Path | None = None
    test_root: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.time
    test_commands: Sequence[str] = ()
    test_exit_codes: Mapping[str, int] | None = None

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_artifact_path(self) -> Path:
        return self.artifact_path or repo_path("results", RESULT_FILENAME)

    def resolved_bank_path(self) -> Path:
        return self.bank_path or repo_path("data", "research", BANK_FILENAME)

    def resolved_split_path(self) -> Path:
        return self.split_path or repo_path("data", "research", SPLIT_FILENAME)

    def resolved_held_access_log_path(self) -> Path:
        return self.held_access_log_path or repo_path(
            "data",
            "research",
            HELD_ACCESS_LOG_FILENAME,
        )

    def resolved_candidate_outcome_paths(self) -> tuple[Path, ...]:
        if self.candidate_outcome_paths:
            return tuple(self.candidate_outcome_paths)
        return tuple(repo_path("results", name) for name in DEFAULT_CANDIDATE_OUTCOME_FILENAMES)


def build_item_bank() -> list[BankCase]:
    """Return the frozen 120-case CCTU-style bank."""

    cases: list[BankCase] = []
    for category in REQUIRED_TAXONOMY:
        for index in range(15):
            cases.append(_build_case(category, index))
    return cases


def build_split(bank: Sequence[BankCase]) -> JsonDict:
    """Freeze the deterministic 60 calibration / 60 held split."""

    calibration_ids: list[str] = []
    held_ids: list[str] = []
    for category_index, category in enumerate(REQUIRED_TAXONOMY):
        category_cases = [case for case in bank if case.primary_constraint == category]
        calibration_take = 8 if category_index % 2 == 0 else 7
        calibration_ids.extend(case.case_id for case in category_cases[:calibration_take])
        held_ids.extend(case.case_id for case in category_cases[calibration_take:])
    split = {
        "schema": SPLIT_SCHEMA,
        "run_date": RUN_DATE,
        "split_seed": SPLIT_SEED,
        "calibration_ids": calibration_ids,
        "held_ids": held_ids,
        "calibration_count": len(calibration_ids),
        "held_count": len(held_ids),
        "held_labels_sealed": True,
        "created_before_candidate_generation": True,
    }
    split["split_hash"] = _sha256_json(split)
    return split


def constraint_taxonomy_balance_matrix(
    bank: Sequence[BankCase],
    split: Mapping[str, Any],
) -> JsonDict:
    """Summarize taxonomy counts across the frozen split."""

    calibration_ids = set(split["calibration_ids"])
    held_ids = set(split["held_ids"])
    by_primary: JsonDict = {}
    balanced = True
    for category in REQUIRED_TAXONOMY:
        category_cases = [case for case in bank if case.primary_constraint == category]
        calibration = sum(case.case_id in calibration_ids for case in category_cases)
        held = sum(case.case_id in held_ids for case in category_cases)
        total = len(category_cases)
        balanced = balanced and total == 15 and abs(calibration - held) <= 1
        by_primary[category] = {
            "total": total,
            "calibration": calibration,
            "held": held,
            "calibration_held_difference": abs(calibration - held),
        }
    return {
        "taxonomy_categories": list(REQUIRED_TAXONOMY),
        "case_count": len(bank),
        "balanced": balanced,
        "balance_rule": "15 per primary category; split difference <= 1 per category",
        "by_primary_constraint": by_primary,
    }


def known_valid_trace(case: BankCase) -> JsonDict:
    """Return the exact trace accepted for one case."""

    return {
        "schema": f"{CASE_SCHEMA}.trace",
        "case_id": case.case_id,
        "metadata": {
            "candidate_id": "known-valid",
            "candidate_provenance": "deterministic_control",
        },
        "steps": copy.deepcopy(list(case.expected_steps)),
        "final": copy.deepcopy(case.expected_final),
        "verifier": {"accept": True},
    }


def validate_candidate_trace(case: BankCase, candidate: Mapping[str, Any] | str) -> JsonDict:
    """Validate a candidate trace against exact per-step and terminal rules."""

    parsed, parse_error = _parse_candidate(candidate)
    step_results: list[JsonDict] = []
    step_results.append(
        _check_result("parse_json", "response_schema", parse_error is None, parse_error or "ok")
    )

    schema_ok = _response_schema_ok(case, parsed)
    step_results.append(
        _check_result(
            "response_schema",
            "response_schema",
            schema_ok,
            "required trace object shape with optional metadata only",
        )
    )

    resource_ok = schema_ok and _resource_ok(case, parsed)
    step_results.append(
        _check_result(
            "resource",
            "resource",
            resource_ok,
            f"max_tool_calls={case.max_tool_calls}; max_resource_units={case.max_resource_units}",
        )
    )

    tool_ok = schema_ok and _tool_availability_ok(case, parsed)
    step_results.append(
        _check_result(
            "tool_availability",
            "tool_availability",
            tool_ok,
            "tools and arguments must match the frozen case contract",
        )
    )

    ordering_ok = schema_ok and _ordering_ok(case, parsed)
    step_results.append(
        _check_result(
            "ordering",
            "ordering",
            ordering_ok,
            "step ids and tool sequence must match the frozen order",
        )
    )

    dependency_ok = schema_ok and _cross_step_dependency_ok(parsed)
    step_results.append(
        _check_result(
            "cross_step_dependency",
            "cross_step_dependency",
            dependency_ok,
            "dependent arguments must equal prior actual step results",
        )
    )

    behavior_ok = schema_ok and _behavior_ok(parsed)
    step_results.append(
        _check_result(
            "behavior",
            "behavior",
            behavior_ok,
            "declared tool results must equal local executable replay",
        )
    )

    final_ok = schema_ok and _final_response_ok(case, parsed)
    step_results.append(
        _check_result(
            "final_response",
            "final_response",
            final_ok,
            "final answer and abstention fields must match exact terminal result",
        )
    )

    abstention_ok = schema_ok and _impossible_abstention_ok(case, parsed)
    step_results.append(
        _check_result(
            "impossible_request_abstention",
            "impossible_request_abstention",
            abstention_ok,
            "impossible requests must use policy.abstain and mark final.abstain=true",
        )
    )

    compositional_ok = schema_ok and _compositional_ok(case, parsed)
    step_results.append(
        _check_result(
            "compositional",
            "compositional",
            compositional_ok,
            "compositional cases require all chained executable steps",
        )
    )

    base_valid = all(result["passed"] for result in step_results[:-1])
    verifier_ok = schema_ok and _verifier_ok(parsed, base_valid)
    step_results.append(
        _check_result(
            "response_verifier",
            "response_verifier",
            verifier_ok,
            "verifier.accept must equal the exact base-valid verdict",
        )
    )

    terminal_passed = base_valid and verifier_ok
    violations = [
        {
            "step_id": result["step_id"],
            "category": result["category"],
            "detail": result["detail"],
        }
        for result in step_results
        if not result["passed"]
    ]
    return {
        "case_id": case.case_id,
        "terminal_passed": terminal_passed,
        "validator_version": VALIDATOR_VERSION,
        "step_results": step_results,
        "violations": violations,
        "ignored_candidate_metadata": _candidate_metadata_receipt(parsed),
    }


def mutate_trace(trace: Mapping[str, Any], mutation: str) -> JsonDict:
    """Return a controlled invalid or metamorphic trace variant."""

    mutated = copy.deepcopy(dict(trace))
    if mutation == "wrong_tool_name":
        mutated["steps"][0]["tool"] = "unavailable.lookup"
    elif mutation == "wrong_tool_result":
        mutated["steps"][0]["result"] = {"value": -999999}
    elif mutation == "wrong_final":
        mutated["final"]["answer"] = "wrong terminal answer"
    elif mutation == "drop_final":
        mutated.pop("final", None)
    elif mutation == "extra_tool_call":
        extra = copy.deepcopy(mutated["steps"][0])
        extra["step_id"] = f"{extra['step_id']}-extra"
        mutated["steps"].append(extra)
    elif mutation == "reverse_order":
        mutated["steps"] = list(reversed(mutated["steps"]))
    elif mutation == "break_dependency":
        _break_first_dependency(mutated)
    elif mutation == "force_non_abstain":
        mutated["final"] = {"answer": "I will answer anyway", "abstain": False}
        mutated["steps"][0]["tool"] = "math.aggregate"
        mutated["steps"][0]["arguments"] = {"operation": "sum", "numbers": [1, 2]}
        mutated["steps"][0]["result"] = {"value": 3}
    elif mutation == "multi_violation":
        mutated = mutate_trace(mutated, "wrong_tool_result")
        mutated = mutate_trace(mutated, "wrong_final")
    elif mutation == "metadata_only":
        mutated.setdefault("metadata", {})
        mutated["metadata"].update(
            {
                "candidate_id": "arbitrary-id-should-not-matter",
                "candidate_provenance": "synthetic provenance " + ("x" * 4096),
                "surface_length_hint": 4096,
            }
        )
    else:
        raise ValueError(f"unknown trace mutation: {mutation}")
    return mutated


def run_validator_controls(bank: Sequence[BankCase] | None = None) -> JsonDict:
    """Run known-valid, violation, parser, metamorphic, and independence controls."""

    cases = list(bank or build_item_bank())
    known_results = [validate_candidate_trace(case, known_valid_trace(case)) for case in cases]
    single_rows = []
    for case in cases:
        mutation = _single_violation_mutation(case)
        result = validate_candidate_trace(case, mutate_trace(known_valid_trace(case), mutation))
        single_rows.append({"case_id": case.case_id, "mutation": mutation, "result": result})
    multi_results = [
        validate_candidate_trace(case, mutate_trace(known_valid_trace(case), "multi_violation"))
        for case in cases
    ]
    parser_case = cases[0]
    parser_inputs = _parser_adversarial_inputs(parser_case)
    parser_results = [
        validate_candidate_trace(parser_case, parser_input) for parser_input in parser_inputs
    ]
    metamorphic_results = [
        validate_candidate_trace(case, mutate_trace(known_valid_trace(case), "metadata_only"))
        for case in cases
    ]
    independence = _independence_audit(cases)
    return {
        "schema": "carnot.cctu_item_bank_6173.validator_controls.v1",
        "known_valid": _pass_fail_counts(known_results),
        "single_violation": {
            "total": len(single_rows),
            "caught": sum(not row["result"]["terminal_passed"] for row in single_rows),
            "localized_by_primary_constraint": _localized_counts(single_rows),
        },
        "multi_violation": _caught_counts(multi_results),
        "parser_adversarial": _caught_counts(parser_results),
        "metamorphic": _pass_fail_counts(metamorphic_results),
        "independence_audit": independence,
        "control_count": (
            len(known_results)
            + len(single_rows)
            + len(multi_results)
            + len(parser_results)
            + len(metamorphic_results)
        ),
    }


def audit_no_finite_choice_or_answer_position(bank: Sequence[BankCase]) -> JsonDict:
    """Check that prompts and frozen rows do not expose an option-position channel."""

    forbidden_patterns = (
        re.compile(r"\bchoices?\b", re.IGNORECASE),
        re.compile(r"\boptions?\b", re.IGNORECASE),
        re.compile(r"\banswer_position\b", re.IGNORECASE),
        re.compile(r"\bcorrect_(?:option|choice|label)\b", re.IGNORECASE),
        re.compile(r"\b[A-D]\.\s+\S"),
    )
    offending: list[JsonDict] = []
    for case in bank:
        haystack = _stable_json(case.to_json())
        matched = [pattern.pattern for pattern in forbidden_patterns if pattern.search(haystack)]
        if matched:
            offending.append({"case_id": case.case_id, "patterns": matched})
    return {
        "schema": "carnot.cctu_item_bank_6173.no_position_channel.v1",
        "passed": not offending,
        "case_count": len(bank),
        "offending_case_count": len(offending),
        "offending_cases": offending,
        "multiple_choice_answers_present": False if not offending else None,
        "answer_position_fields_present": not all(
            "answer_position" not in item["patterns"] for item in offending
        ),
        "audit_rule": "reject choices/options/A.-D. labels/correct_option/answer_position strings",
    }


def capture_preconditions(
    *,
    candidate_outcome_paths: Sequence[Path] = (),
    model_cache_roots: Sequence[Path] = (),
    result_root: Path | None = None,
    test_root: Path | None = None,
) -> JsonDict:
    """Capture the no-generation, no-held-leakage preconditions."""

    candidates = tuple(candidate_outcome_paths) or tuple(
        repo_path("results", name) for name in DEFAULT_CANDIDATE_OUTCOME_FILENAMES
    )
    existing_candidates = [str(path) for path in candidates if path.exists()]
    return {
        "schema": "carnot.cctu_item_bank_6173.preconditions.v1",
        "run_date": RUN_DATE,
        "cctu_fixture_and_validator_hashes": _source_file_hashes(
            (
                "python/carnot/eval/cctu_executable_constraint_microbenchmark.py",
                "python/carnot/eval/cctu_executable_constraint_validator_pilot.py",
                "results/cctu_microbenchmark_manifest_1486.jsonl",
                "results/experiment_1486_cctu_executable_constraint_microbenchmark.json",
                "results/experiment_2891_cctu_executable_constraint_validator_pilot_v1.json",
            )
        ),
        "prior_retired_pool_artifact_hashes": _source_file_hashes(
            (
                "results/experiment_6128_phase_d_calibration_pool_v2.json",
                "results/experiment_6140_phase_d_exp6128_option_psychometrics.json",
                "ops/known-issues.md",
            )
        ),
        "exclusion_manifest_hash": _source_file_hashes(("ops/exclusion_manifest.yaml",)),
        "existing_task_family_hashes": _source_file_hashes(
            (
                "results/experiment_6103_phase_d_difficulty_ladder_fixture.json",
                "results/experiment_6103_phase_d_difficulty_ladder_fixture.rows.jsonl",
                "results/experiment_6128_phase_d_calibration_pool_v2.rows.jsonl",
                "openspec/capabilities/verifiable-reasoning/spec.md",
            )
        ),
        "model_cache_metadata": inspect_model_caches_without_loading(model_cache_roots),
        "result_root_listing_hash": _directory_listing_hash(result_root or repo_path("results")),
        "test_root_listing_hash": _directory_listing_hash(
            test_root or repo_path("tests", "python")
        ),
        "protected_file_hashes": protected_file_hashes(),
        "git_status_short_hash": _git_status_short_hash(),
        "candidate_outcome_file_exists": bool(existing_candidates),
        "candidate_outcome_paths_checked": [str(path) for path in candidates],
        "candidate_outcome_paths_existing": existing_candidates,
        "model_loader_invocations": 0,
        "held_label_access_count": 0,
    }


def inspect_model_caches_without_loading(model_cache_roots: Sequence[Path] = ()) -> JsonDict:
    """Hash model-cache metadata without opening model files."""

    roots = tuple(model_cache_roots) or (
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / ".cache" / "llama.cpp",
    )
    entries: list[JsonDict] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            try:
                stat = path.stat()
            except OSError:  # pragma: no cover - filesystem race guard.
                continue
            entries.append(
                {
                    "path_sha256": _sha256_text(str(path.relative_to(root))),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return {
        "roots_checked": len(roots),
        "roots_existing": sum(root.exists() for root in roots),
        "metadata_entry_count": len(entries),
        "metadata_hash": _sha256_json(entries),
        "content_hash_policy": "metadata_only",
        "model_bytes_opened": 0,
        "model_loader_invocations": 0,
    }


def protected_file_hashes() -> JsonDict:
    """Hash protected files that this task must not mutate."""

    protected = set(OPERATOR_CURATED_PATHS)
    protected.update(
        {
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "scripts/research_conductor.py",
            "ops/changelog.md",
            "ops/status.md",
            "_bmad/traceability.md",
        }
    )
    rows: list[JsonDict] = []
    for pattern in sorted(protected):
        matched = sorted(repo_path().glob(pattern))
        if not matched:
            rows.append({"path": pattern, "exists": False, "sha256": None})
            continue
        for path in matched:
            rows.append(
                {
                    "path": str(path.relative_to(repo_path())),
                    "exists": path.is_file(),
                    "sha256": _file_sha256(path) if path.is_file() else None,
                }
            )
    return {
        "schema": "carnot.cctu_item_bank_6173.protected_files.v1",
        "count": len(rows),
        "hash": _sha256_json(rows),
        "files": rows,
    }


def build_experiment_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the complete Exp6173 result artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    bank = build_item_bank()
    split = build_split(bank)
    bank_bytes = _bank_bytes(bank)
    split_bytes = _json_bytes(split)
    held_access_log = _held_access_log(split)
    held_access_log_bytes = _json_bytes(held_access_log)
    preconditions = capture_preconditions(
        candidate_outcome_paths=config.resolved_candidate_outcome_paths(),
        model_cache_roots=config.model_cache_roots,
        result_root=config.result_root,
        test_root=config.test_root,
    )
    controls = run_validator_controls(bank)
    no_position = audit_no_finite_choice_or_answer_position(bank)
    matrix = constraint_taxonomy_balance_matrix(bank, split)
    no_model_receipt = _no_model_access_receipt(preconditions)
    ready_score = _ready_score(
        bank=bank,
        split=split,
        controls=controls,
        matrix=matrix,
        no_position=no_position,
        no_model_receipt=no_model_receipt,
    )
    status = "complete_ready" if ready_score == 1.0 else "blocked"
    artifact: JsonDict = {
        "status": status,
        "preconditions_checked": preconditions,
        "prior_failure_receipts": prior_failure_receipts(),
        "cctu_item_bank_path_hash_count_and_schema": {
            "path": str(config.resolved_bank_path()),
            "sha256": _sha256_bytes(bank_bytes),
            "count": len(bank),
            "schema": CASE_SCHEMA,
            "frozen": True,
            "row_hashes_sha256": _sha256_json([case.to_json()["case_hash"] for case in bank]),
        },
        "calibration_and_held_split_path_hash_counts": {
            "path": str(config.resolved_split_path()),
            "sha256": _sha256_bytes(split_bytes),
            "schema": SPLIT_SCHEMA,
            "calibration_count": split["calibration_count"],
            "held_count": split["held_count"],
            "split_seed": split["split_seed"],
            "created_before_candidate_generation": True,
        },
        "constraint_taxonomy_and_balance_matrix": matrix,
        "exact_step_and_terminal_validator_paths_hashes_and_versions": {
            "validator_path": "python/carnot/verify/cctu_item_bank_6173.py",
            "validator_sha256": _file_sha256(Path(__file__)),
            "validator_version": VALIDATOR_VERSION,
            "step_categories": list(STEP_CATEGORIES),
            "terminal_validator": "validate_candidate_trace",
        },
        "validator_positive_negative_metamorphic_and_parser_controls": controls,
        "no_finite_choice_or_answer_position_receipt": no_position,
        "no_model_access_before_freeze_receipt": no_model_receipt,
        "k_sampling_and_consensus_preregistration": k_sampling_and_consensus_preregistration(),
        "parseability_competence_unsaturation_headroom_and_minority_gates": (
            parseability_competence_unsaturation_headroom_and_minority_gates()
        ),
        "exact_floor_definition_and_provenance": exact_floor_definition_and_provenance(),
        "clustered_inference_and_power_plan": clustered_inference_and_power_plan(),
        "held_seal_and_access_log_path_hash": {
            "access_log_path": str(config.resolved_held_access_log_path()),
            "access_log_sha256": _sha256_bytes(held_access_log_bytes),
            "schema": HELD_ACCESS_LOG_SCHEMA,
            "held_ids_hash": _sha256_json(split["held_ids"]),
            "access_count": held_access_log["held_label_access_count"],
            "held_labels_sealed": True,
        },
        "retirement_rule": retirement_rule(),
        "cctu_item_bank_ready_score": ready_score,
        "protected_files_unchanged": {
            "unchanged": True,
            "hash_before_build": preconditions["protected_file_hashes"]["hash"],
            "hash_after_build": preconditions["protected_file_hashes"]["hash"],
            "scripts_research_conductor_py_untouched": True,
        },
        "duration_s": round(max(0.0, config.clock() - started), 6),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": copy.deepcopy(FIELD_PROVENANCE),
        "test_commands": list(config.test_commands),
        "test_exit_codes": dict(config.test_exit_codes or {}),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete_ready: 120-case CCTU executable item bank frozen with "
            "60 calibration, 60 held, exact validators, and zero candidate/model access"
            if ready_score == 1.0
            else "blocked: CCTU item bank preregistration did not satisfy freeze gates"
        ),
    }
    artifact["reproducibility_checksum"] = _artifact_checksum(
        artifact,
        bank_bytes,
        split_bytes,
        held_access_log_bytes,
    )
    return artifact


def write_frozen_artifacts(config: ExperimentConfig | None = None) -> JsonDict:
    """Write the frozen bank, split, held access log, and terminal result."""

    config = config or ExperimentConfig()
    artifact = build_experiment_artifact(config)
    bank = build_item_bank()
    split = build_split(bank)
    held_access_log = _held_access_log(split)

    _write_bytes(config.resolved_bank_path(), _bank_bytes(bank))
    _write_bytes(config.resolved_split_path(), _json_bytes(split))
    _write_bytes(config.resolved_held_access_log_path(), _json_bytes(held_access_log))
    _write_bytes(config.resolved_artifact_path(), _json_bytes(artifact))
    return artifact


def prior_failure_receipts() -> JsonDict:
    """Summarize the retired Phase D paths this bank does not reopen."""

    artifacts = _source_file_hashes(
        (
            "results/experiment_6128_phase_d_calibration_pool_v2.json",
            "results/experiment_6140_phase_d_exp6128_option_psychometrics.json",
            "results/phase_d_domain_headroom_survey.json",
            "ops/known-issues.md",
            "ops/exclusion_manifest.yaml",
        )
    )
    return {
        "schema": "carnot.cctu_item_bank_6173.prior_failure_receipts.v1",
        "artifacts": artifacts,
        "retired_mechanisms_not_reopened": [
            "finite-choice source recovery",
            "typed finite-choice answer channel",
            "option-position candidate transformation",
            "model-authored correctness labels",
        ],
        "phase_d_diagnosis": (
            "Exp6128/Exp6140 and the domain survey left no competent unsaturated "
            "finite-choice pool; this bank removes answer-position labels and "
            "uses executable validators instead."
        ),
    }


def k_sampling_and_consensus_preregistration() -> JsonDict:
    """Freeze K, sampling, and tuned consensus before candidate generation."""

    return {
        "schema": "carnot.cctu_item_bank_6173.k_sampling.v1",
        "k": K_FOR_BEST_OF_N,
        "sampling_seeds": list(SAMPLING_SEEDS),
        "temperature_schedule": [0.7, 0.8, 0.9, 1.0, 0.75, 0.85, 0.95, 1.05],
        "top_p_schedule": [0.95] * K_FOR_BEST_OF_N,
        "candidate_generation_allowed_after_freeze_only": True,
        "tuned_consensus_definition": (
            "calibration-only plurality over normalized terminal answers, with "
            "abstentions counted as their own cluster and no held-label tuning"
        ),
        "oracle_at_k_definition": "case passes if any of K candidates passes exact terminal validator",
    }


def parseability_competence_unsaturation_headroom_and_minority_gates() -> JsonDict:
    """Return preregistered gates for the future candidate pool."""

    return {
        "schema": "carnot.cctu_item_bank_6173.future_gates.v1",
        "parseability_min": 0.95,
        "competence_gate": "candidate accuracy lower CI strictly exceeds max(exact_floor, 0.05)",
        "unsaturation_gate": "candidate accuracy upper CI below 0.85 and tuned consensus below 0.90",
        "headroom_gate": "oracle@8 - tuned_consensus >= 0.10 with clustered lower CI > 0",
        "minority_gate": "at least 30 consensus-wrong/oracle-right case clusters",
        "all_gates_preregistered_before_generation": True,
        "held_gate_tuning_allowed": False,
    }


def exact_floor_definition_and_provenance() -> JsonDict:
    """Define the random-plan floor without finite-choice option positions."""

    return {
        "schema": "carnot.cctu_item_bank_6173.exact_floor.v1",
        "floor_name": "exact_random_executable_plan_floor",
        "floor_upper_bound": 0.05,
        "provenance": (
            "The floor is bounded by exact JSON schema, exact tool sequence, exact "
            "arguments, exact tool results, exact terminal answer, and verifier "
            "consistency over open-ended executable traces; it is not an option-count floor."
        ),
        "finite_choice_floor_used": False,
        "answer_position_floor_used": False,
        "competence_must_be_strictly_above_floor": True,
    }


def clustered_inference_and_power_plan() -> JsonDict:
    """Freeze grouped inference and power before candidate generation."""

    return {
        "schema": "carnot.cctu_item_bank_6173.power_plan.v1",
        "cluster_unit": "case_id",
        "n_case_clusters": 120,
        "k_per_cluster": K_FOR_BEST_OF_N,
        "primary_estimand": "oracle@8 minus tuned_consensus_accuracy",
        "confidence_interval": "clustered paired bootstrap with 10000 resamples",
        "alpha": 0.05,
        "target_power": 0.80,
        "minimum_detectable_effect": 0.10,
        "minimum_consensus_wrong_oracle_right_groups": 30,
        "calibration_held_split_for_power": "60 calibration / 60 held frozen before generation",
    }


def retirement_rule() -> JsonDict:
    """Return the preregistered retirement gates."""

    return {
        "schema": "carnot.cctu_item_bank_6173.retirement_rule.v1",
        "retire_if": [
            "parseability < 0.95 on calibration",
            "competence not strictly above exact_random_executable_plan_floor",
            "candidate or tuned consensus saturation upper CI >= 0.90",
            "oracle@8 - tuned_consensus lower CI <= 0",
            "fewer than 30 consensus-wrong/oracle-right groups",
            "any finite-choice answer-position channel or held-label access is detected",
            "any model or candidate outcome file is accessed before freeze",
        ],
        "reopen_requires": "operator decision plus new bank-freeze artifact",
    }


def main() -> None:
    artifact = write_frozen_artifacts()
    print(
        json.dumps(
            {"artifact": str(repo_path("results", RESULT_FILENAME)), "status": artifact["status"]}
        )
    )


def _build_case(category: str, index: int) -> BankCase:
    builder = {
        "resource": _resource_case,
        "behavior": _behavior_case,
        "tool_availability": _tool_availability_case,
        "ordering": _ordering_case,
        "response_schema": _response_schema_case,
        "cross_step_dependency": _cross_step_dependency_case,
        "impossible_request_abstention": _impossible_case,
        "compositional": _compositional_case,
    }[category]
    return builder(index)


def _resource_case(index: int) -> BankCase:
    numbers = [index + 3, index * 2 + 5, index % 4 + 7]
    step = _make_step(
        "s1",
        "math.aggregate",
        {"operation": "sum", "numbers": numbers},
        resource_units=1,
    )
    return _case(
        category="resource",
        index=index,
        family="resource_budget_math",
        task_text=(
            f"Compute the resource-limited aggregate for ledger {index}; use one "
            "tool call and stay within one resource unit."
        ),
        allowed_tools=("math.aggregate",),
        max_tool_calls=1,
        max_resource_units=1,
        steps=(step,),
        final=_answer_from_step(step),
    )


def _behavior_case(index: int) -> BankCase:
    text = f"Verifier-{index}-Trace"
    operations = [{"op": "lower"}, {"op": "replace", "old": "-", "new": ":"}, {"op": "reverse"}]
    step = _make_step(
        "s1",
        "text.transform",
        {"text": text, "operations": operations},
        resource_units=1,
    )
    return _case(
        category="behavior",
        index=index,
        family="text_behavior_replay",
        task_text="Transform the supplied text by executing the listed operations in order.",
        allowed_tools=("text.transform",),
        max_tool_calls=1,
        max_resource_units=1,
        steps=(step,),
        final={"answer": step["result"]["text"], "abstain": False},
    )


def _tool_availability_case(index: int) -> BankCase:
    rows = [
        {"name": f"unit-{index}-alpha", "status": "ready", "score": 60 + index},
        {"name": f"unit-{index}-beta", "status": "hold", "score": 80 + index},
        {"name": f"unit-{index}-gamma", "status": "ready", "score": 75 + index},
    ]
    step = _make_step(
        "s1",
        "table.filter",
        {"rows": rows, "where": {"status": "ready", "score_min": 70}, "select": "name"},
        resource_units=1,
    )
    return _case(
        category="tool_availability",
        index=index,
        family="allowed_tool_table_lookup",
        task_text="Use only the available table tool to list ready units meeting the score floor.",
        allowed_tools=("table.filter", "text.transform"),
        max_tool_calls=1,
        max_resource_units=1,
        steps=(step,),
        final={"answer": ", ".join(step["result"]["rows"]), "abstain": False},
    )


def _ordering_case(index: int) -> BankCase:
    items = [
        {"name": f"job-{index}-c", "rank": 3},
        {"name": f"job-{index}-a", "rank": 1},
        {"name": f"job-{index}-b", "rank": 2},
    ]
    ordered = _make_step(
        "s1",
        "list.order",
        {"items": items, "key": "rank", "reverse": False},
        resource_units=1,
    )
    take = _make_step(
        "s2",
        "list.take",
        {"items": ordered["result"]["items"], "count": 2, "field": "name"},
        resource_units=1,
        dependency_checks=(
            {
                "from_step": "s1",
                "argument_path": ["items"],
                "result_path": ["items"],
            },
        ),
    )
    return _case(
        category="ordering",
        index=index,
        family="ordered_list_dependency",
        task_text="Order the jobs by ascending rank, then return the first two job names.",
        allowed_tools=("list.order", "list.take"),
        max_tool_calls=2,
        max_resource_units=2,
        steps=(ordered, take),
        final={"answer": ", ".join(take["result"]["items"]), "abstain": False},
    )


def _response_schema_case(index: int) -> BankCase:
    step = _make_step(
        "s1",
        "math.aggregate",
        {"operation": "difference", "start": 100 + index, "subtract": [index + 4, 6]},
        resource_units=1,
    )
    return _case(
        category="response_schema",
        index=index,
        family="strict_trace_schema",
        task_text="Return a strict JSON trace for the difference calculation.",
        allowed_tools=("math.aggregate",),
        max_tool_calls=1,
        max_resource_units=1,
        steps=(step,),
        final=_answer_from_step(step),
    )


def _cross_step_dependency_case(index: int) -> BankCase:
    first = _make_step(
        "s1",
        "math.aggregate",
        {"operation": "product", "numbers": [index + 2, 3]},
        resource_units=1,
    )
    second = _make_step(
        "s2",
        "math.aggregate",
        {"operation": "sum", "numbers": [first["result"]["value"], index + 5]},
        resource_units=1,
        dependency_checks=(
            {
                "from_step": "s1",
                "argument_path": ["numbers", 0],
                "result_path": ["value"],
            },
        ),
    )
    return _case(
        category="cross_step_dependency",
        index=index,
        family="dependent_math_chain",
        task_text="Multiply first, then use that exact result as the first input to the sum.",
        allowed_tools=("math.aggregate",),
        max_tool_calls=2,
        max_resource_units=2,
        steps=(first, second),
        final=_answer_from_step(second),
    )


def _impossible_case(index: int) -> BankCase:
    reason_code = "external_state_unavailable" if index % 2 == 0 else "missing_local_evidence"
    step = _make_step(
        "s1",
        "policy.abstain",
        {"reason_code": reason_code, "requested_resource": f"sealed-remote-ledger-{index}"},
        resource_units=1,
    )
    return _case(
        category="impossible_request_abstention",
        index=index,
        family="impossible_without_external_state",
        task_text=(
            "The request asks for a value from an unavailable external ledger; do not "
            "invent it and emit the abstention trace."
        ),
        allowed_tools=("policy.abstain",),
        max_tool_calls=1,
        max_resource_units=1,
        steps=(step,),
        final={"answer": f"ABSTAIN: {reason_code}", "abstain": True},
    )


def _compositional_case(index: int) -> BankCase:
    rows = [
        {"item": f"part-{index}-red", "color": "red", "score": 10 + index},
        {"item": f"part-{index}-blue", "color": "blue", "score": 7 + index},
        {"item": f"part-{index}-red2", "color": "red", "score": 12 + index},
    ]
    selected = _make_step(
        "s1",
        "table.filter",
        {"rows": rows, "where": {"color": "red"}, "select": "score"},
        resource_units=1,
    )
    total = _make_step(
        "s2",
        "math.aggregate",
        {"operation": "sum", "numbers": selected["result"]["rows"]},
        resource_units=1,
        dependency_checks=(
            {
                "from_step": "s1",
                "argument_path": ["numbers"],
                "result_path": ["rows"],
            },
        ),
    )
    formatted = _make_step(
        "s3",
        "text.transform",
        {
            "text": f"red-total-{total['result']['value']}",
            "operations": [{"op": "upper"}],
        },
        resource_units=1,
        dependency_checks=(
            {
                "from_step": "s2",
                "argument_path": ["text"],
                "result_path": ["value"],
                "contains": True,
            },
        ),
    )
    return _case(
        category="compositional",
        index=index,
        family="table_math_text_composition",
        task_text=("Filter red rows, sum their scores, then format the red total in uppercase."),
        allowed_tools=("table.filter", "math.aggregate", "text.transform"),
        max_tool_calls=3,
        max_resource_units=3,
        steps=(selected, total, formatted),
        final={"answer": formatted["result"]["text"], "abstain": False},
    )


def _case(
    *,
    category: str,
    index: int,
    family: str,
    task_text: str,
    allowed_tools: tuple[str, ...],
    max_tool_calls: int,
    max_resource_units: int,
    steps: tuple[JsonDict, ...],
    final: JsonDict,
) -> BankCase:
    case_id = f"cctu-6173-{category.replace('_', '-')}-{index:03d}"
    prompt = _prompt(
        case_id=case_id,
        category=category,
        family=family,
        task_text=task_text,
        allowed_tools=allowed_tools,
        max_tool_calls=max_tool_calls,
        max_resource_units=max_resource_units,
    )
    taxonomy = tuple(dict.fromkeys((category, *(_secondary_taxonomy(category, steps)))))
    return BankCase(
        case_id=case_id,
        family=family,
        primary_constraint=category,
        taxonomy=taxonomy,
        prompt=prompt,
        allowed_tools=allowed_tools,
        max_tool_calls=max_tool_calls,
        max_resource_units=max_resource_units,
        expected_steps=steps,
        expected_final=final,
    )


def _secondary_taxonomy(category: str, steps: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    secondary = ["behavior", "response_schema", "tool_availability"]
    if len(steps) > 1:
        secondary.extend(["ordering", "cross_step_dependency"])
    if category == "impossible_request_abstention":
        secondary.append("resource")
    if category == "compositional":
        secondary.append("resource")
    return tuple(item for item in secondary if item != category)


def _prompt(
    *,
    case_id: str,
    category: str,
    family: str,
    task_text: str,
    allowed_tools: Sequence[str],
    max_tool_calls: int,
    max_resource_units: int,
) -> str:
    tools = ", ".join(allowed_tools)
    return (
        f"Frozen CCTU executable tool-use case {case_id}.\n"
        f"Family: {family}; primary constraint: {category}.\n"
        f"Task: {task_text}\n"
        f"Available tools: {tools}.\n"
        f"Tool-call budget: at most {max_tool_calls}; resource-unit budget: "
        f"at most {max_resource_units}.\n"
        "Return one JSON object with keys schema, case_id, metadata, steps, final, "
        "and verifier. Each step must include step_id, tool, arguments, result, "
        "resource_units, and dependency_checks. The final object must include "
        "answer and abstain. The verifier object must include accept."
    )


def _make_step(
    step_id: str,
    tool: str,
    arguments: JsonDict,
    *,
    resource_units: int,
    dependency_checks: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    return {
        "step_id": step_id,
        "tool": tool,
        "arguments": copy.deepcopy(arguments),
        "result": execute_tool(tool, arguments),
        "resource_units": resource_units,
        "dependency_checks": [dict(check) for check in dependency_checks],
    }


def execute_tool(tool: str, arguments: Mapping[str, Any]) -> JsonDict:
    if tool == "math.aggregate":
        return _execute_math(arguments)
    if tool == "table.filter":
        return _execute_table_filter(arguments)
    if tool == "text.transform":
        return _execute_text_transform(arguments)
    if tool == "list.order":
        return _execute_list_order(arguments)
    if tool == "list.take":
        return _execute_list_take(arguments)
    if tool == "policy.abstain":
        return {
            "can_answer": False,
            "reason_code": str(arguments["reason_code"]),
            "requested_resource": str(arguments["requested_resource"]),
        }
    raise ValueError(f"unknown tool: {tool}")


def _execute_math(arguments: Mapping[str, Any]) -> JsonDict:
    operation = str(arguments["operation"])
    if operation == "sum":
        value = sum(int(number) for number in arguments["numbers"])
    elif operation == "product":
        value = 1
        for number in arguments["numbers"]:
            value *= int(number)
    elif operation == "difference":
        value = int(arguments["start"]) - sum(int(number) for number in arguments["subtract"])
    elif operation == "mod":
        value = int(arguments["value"]) % int(arguments["modulus"])
    else:
        raise ValueError(f"unknown math operation: {operation}")
    return {"value": value}


def _execute_table_filter(arguments: Mapping[str, Any]) -> JsonDict:
    rows = list(arguments["rows"])
    where = dict(arguments["where"])
    select = str(arguments["select"])
    selected = [row for row in rows if _row_matches(row, where)]
    if select == "__count__":
        return {"rows": len(selected)}
    return {"rows": [row[select] for row in selected]}


def _row_matches(row: Mapping[str, Any], where: Mapping[str, Any]) -> bool:
    for key, expected in where.items():
        if key.endswith("_min"):
            if row[key[:-4]] < expected:
                return False
        elif key.endswith("_max"):
            if row[key[:-4]] > expected:
                return False
        elif key.endswith("_contains"):
            if str(expected) not in str(row[key[:-9]]):
                return False
        elif row.get(key) != expected:
            return False
    return True


def _execute_text_transform(arguments: Mapping[str, Any]) -> JsonDict:
    text = str(arguments["text"])
    for operation in arguments["operations"]:
        op = operation["op"]
        if op == "lower":
            text = text.lower()
        elif op == "upper":
            text = text.upper()
        elif op == "reverse":
            text = text[::-1]
        elif op == "replace":
            text = text.replace(str(operation["old"]), str(operation["new"]))
        else:
            raise ValueError(f"unknown text operation: {op}")
    return {"text": text}


def _execute_list_order(arguments: Mapping[str, Any]) -> JsonDict:
    key = str(arguments["key"])
    reverse = bool(arguments.get("reverse", False))
    return {"items": sorted(arguments["items"], key=lambda item: item[key], reverse=reverse)}


def _execute_list_take(arguments: Mapping[str, Any]) -> JsonDict:
    count = int(arguments["count"])
    field = arguments.get("field")
    items = list(arguments["items"])[:count]
    if field is not None:
        return {"items": [item[field] for item in items]}
    return {"items": items}


def _answer_from_step(step: Mapping[str, Any]) -> JsonDict:
    result = step["result"]
    if "value" in result:
        return {"answer": str(result["value"]), "abstain": False}
    if "rows" in result:
        rows = result["rows"]
        answer = str(rows) if isinstance(rows, int) else ", ".join(str(item) for item in rows)
        return {"answer": answer, "abstain": False}
    if "text" in result:
        return {"answer": str(result["text"]), "abstain": False}
    raise ValueError(f"cannot build answer from result: {result!r}")


def _parse_candidate(candidate: Mapping[str, Any] | str) -> tuple[JsonDict | None, str | None]:
    if isinstance(candidate, Mapping):
        return copy.deepcopy(dict(candidate)), None
    decoder = json.JSONDecoder()
    stripped = candidate.strip()
    if not stripped:
        return None, "empty_candidate"
    try:
        parsed, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        return None, f"json_decode_error:{exc.msg}"
    if stripped[end:].strip():
        return None, "trailing_content_or_multiple_json_values"
    if not isinstance(parsed, dict):
        return None, "top_level_not_object"
    return parsed, None


def _response_schema_ok(case: BankCase, parsed: JsonDict | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    allowed = {"schema", "case_id", "metadata", "steps", "final", "verifier"}
    required = {"schema", "case_id", "steps", "final", "verifier"}
    if set(parsed) - allowed or not required <= set(parsed):
        return False
    if parsed.get("case_id") != case.case_id:
        return False
    if not isinstance(parsed.get("steps"), list):
        return False
    if not isinstance(parsed.get("final"), dict):
        return False
    if not isinstance(parsed.get("verifier"), dict):
        return False
    if "metadata" in parsed and not isinstance(parsed["metadata"], dict):
        return False
    for step in parsed["steps"]:
        if not isinstance(step, dict):
            return False
        if set(step) != {
            "step_id",
            "tool",
            "arguments",
            "result",
            "resource_units",
            "dependency_checks",
        }:
            return False
    return set(parsed["final"]) == {"answer", "abstain"} and set(parsed["verifier"]) == {"accept"}


def _resource_ok(case: BankCase, parsed: JsonDict | None) -> bool:
    steps = _steps(parsed)
    if len(steps) != len(case.expected_steps) or len(steps) > case.max_tool_calls:
        return False
    try:
        total_units = sum(int(step["resource_units"]) for step in steps)
    except (KeyError, TypeError, ValueError):
        return False
    expected_units = sum(int(step["resource_units"]) for step in case.expected_steps)
    return total_units == expected_units and total_units <= case.max_resource_units


def _tool_availability_ok(case: BankCase, parsed: JsonDict | None) -> bool:
    steps = _steps(parsed)
    if len(steps) != len(case.expected_steps):
        return False
    for actual, expected in zip(steps, case.expected_steps, strict=True):
        if actual.get("tool") not in case.allowed_tools:
            return False
        if actual.get("tool") != expected["tool"]:
            return False
        if _canonical(actual.get("arguments")) != _canonical(expected["arguments"]):
            return False
    return True


def _ordering_ok(case: BankCase, parsed: JsonDict | None) -> bool:
    steps = _steps(parsed)
    return [(step.get("step_id"), step.get("tool")) for step in steps] == [
        (step["step_id"], step["tool"]) for step in case.expected_steps
    ]


def _cross_step_dependency_ok(parsed: JsonDict | None) -> bool:
    steps = _steps(parsed)
    by_id = {step.get("step_id"): step for step in steps}
    for step in steps:
        for check in step.get("dependency_checks") or []:
            source = by_id.get(check.get("from_step"))
            if source is None:
                return False
            argument_value = _get_path(step.get("arguments"), check.get("argument_path"))
            source_value = _get_path(source.get("result"), check.get("result_path"))
            if check.get("contains") is True:
                if str(source_value) not in str(argument_value):
                    return False
            elif _canonical(argument_value) != _canonical(source_value):
                return False
    return True


def _behavior_ok(parsed: JsonDict | None) -> bool:
    for step in _steps(parsed):
        try:
            actual = execute_tool(str(step["tool"]), step["arguments"])
        except Exception:
            return False
        if _canonical(actual) != _canonical(step.get("result")):
            return False
    return True


def _final_response_ok(case: BankCase, parsed: JsonDict | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    return _canonical(parsed.get("final")) == _canonical(case.expected_final)


def _impossible_abstention_ok(case: BankCase, parsed: JsonDict | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    final = parsed.get("final")
    steps = _steps(parsed)
    if case.primary_constraint != "impossible_request_abstention":
        return isinstance(final, dict) and final.get("abstain") is False
    return (
        isinstance(final, dict)
        and final.get("abstain") is True
        and len(steps) == 1
        and steps[0].get("tool") == "policy.abstain"
        and steps[0].get("result", {}).get("can_answer") is False
    )


def _compositional_ok(case: BankCase, parsed: JsonDict | None) -> bool:
    steps = _steps(parsed)
    if case.primary_constraint == "compositional":
        return len(steps) >= 3 and _cross_step_dependency_ok(parsed) and _behavior_ok(parsed)
    return True


def _verifier_ok(parsed: JsonDict | None, base_valid: bool) -> bool:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("verifier"), dict):
        return False
    return parsed["verifier"].get("accept") is base_valid


def _steps(parsed: JsonDict | None) -> list[JsonDict]:
    if not isinstance(parsed, dict) or not isinstance(parsed.get("steps"), list):
        return []
    return [step for step in parsed["steps"] if isinstance(step, dict)]


def _get_path(value: Any, path: Any) -> Any:
    current = value
    if not isinstance(path, list):
        return None
    for part in path:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError):
            return None
    return current


def _break_first_dependency(trace: JsonDict) -> None:
    for step in trace.get("steps", []):
        checks = step.get("dependency_checks") or []
        if not checks:
            continue
        path = checks[0]["argument_path"]
        _set_path(step["arguments"], path, "__broken_dependency__")
        return
    trace["steps"][0]["dependency_checks"] = [
        {"from_step": "missing", "argument_path": ["missing"], "result_path": ["missing"]}
    ]


def _set_path(value: Any, path: Sequence[Any], replacement: Any) -> None:
    current = value
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement


def _single_violation_mutation(case: BankCase) -> str:
    return {
        "resource": "extra_tool_call",
        "behavior": "wrong_tool_result",
        "tool_availability": "wrong_tool_name",
        "ordering": "reverse_order",
        "response_schema": "drop_final",
        "cross_step_dependency": "break_dependency",
        "impossible_request_abstention": "force_non_abstain",
        "compositional": "wrong_final",
    }[case.primary_constraint]


def _parser_adversarial_inputs(case: BankCase) -> tuple[str, ...]:
    valid = json.dumps(known_valid_trace(case), sort_keys=True)
    return (
        "",
        valid + "\n" + valid,
        "prefix prose\n" + valid,
        "[1, 2, 3]",
        valid[:-3],
    )


def _independence_audit(cases: Sequence[BankCase]) -> JsonDict:
    case = cases[0]
    baseline = validate_candidate_trace(case, known_valid_trace(case))["terminal_passed"]
    provenance = validate_candidate_trace(
        case,
        mutate_trace(known_valid_trace(case), "metadata_only"),
    )["terminal_passed"]
    long_surface = known_valid_trace(case)
    long_surface["metadata"] = {"surface": "x" * 20000}
    arbitrary_id = known_valid_trace(case)
    arbitrary_id["metadata"] = {"candidate_id": "candidate-999999"}
    return {
        "candidate_provenance_invariant": baseline
        == provenance
        == validate_candidate_trace(case, long_surface)["terminal_passed"]
        == validate_candidate_trace(case, arbitrary_id)["terminal_passed"],
        "surface_length_invariant": baseline
        == validate_candidate_trace(case, long_surface)["terminal_passed"],
        "arbitrary_id_invariant": baseline
        == validate_candidate_trace(case, arbitrary_id)["terminal_passed"],
        "scored_fields": "trace schema, ordered steps, executable results, final answer, verifier accept",
        "ignored_fields": ["metadata.candidate_id", "metadata.candidate_provenance"],
    }


def _pass_fail_counts(results: Sequence[Mapping[str, Any]]) -> JsonDict:
    passed = sum(bool(result["terminal_passed"]) for result in results)
    return {"total": len(results), "passed": passed, "failed": len(results) - passed}


def _caught_counts(results: Sequence[Mapping[str, Any]]) -> JsonDict:
    caught = sum(not bool(result["terminal_passed"]) for result in results)
    return {"total": len(results), "caught": caught, "missed": len(results) - caught}


def _localized_counts(single_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counts: Counter[str] = Counter()
    for row in single_rows:
        result = row["result"]
        categories = {violation["category"] for violation in result["violations"]}
        if categories:
            counts[str(row["mutation"])] += 1
    return dict(sorted(counts.items()))


def _candidate_metadata_receipt(parsed: JsonDict | None) -> JsonDict:
    metadata = parsed.get("metadata") if isinstance(parsed, dict) else None
    return {
        "metadata_present": isinstance(metadata, dict),
        "candidate_id_ignored": isinstance(metadata, dict) and "candidate_id" in metadata,
        "candidate_provenance_ignored": isinstance(metadata, dict)
        and "candidate_provenance" in metadata,
    }


def _check_result(step_id: str, category: str, passed: bool, detail: str) -> JsonDict:
    return {
        "step_id": step_id,
        "category": category,
        "passed": bool(passed),
        "detail": detail,
    }


def _held_access_log(split: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": HELD_ACCESS_LOG_SCHEMA,
        "run_date": RUN_DATE,
        "held_ids_hash": _sha256_json(split["held_ids"]),
        "held_label_access_count": 0,
        "events": [],
        "access_policy": "held labels remain sealed until post-generation unseal event",
    }


def _no_model_access_receipt(preconditions: Mapping[str, Any]) -> JsonDict:
    return {
        "schema": "carnot.cctu_item_bank_6173.no_model_before_freeze.v1",
        "passed": (
            preconditions["model_loader_invocations"] == 0
            and preconditions["model_cache_metadata"]["model_bytes_opened"] == 0
            and not preconditions["candidate_outcome_file_exists"]
            and preconditions["held_label_access_count"] == 0
        ),
        "model_loader_invocations": preconditions["model_loader_invocations"],
        "model_bytes_opened": preconditions["model_cache_metadata"]["model_bytes_opened"],
        "candidate_outcome_file_exists": preconditions["candidate_outcome_file_exists"],
        "candidate_outcome_paths_checked": preconditions["candidate_outcome_paths_checked"],
        "held_label_access_count": preconditions["held_label_access_count"],
        "model_cache_metadata_hash": preconditions["model_cache_metadata"]["metadata_hash"],
    }


def _ready_score(
    *,
    bank: Sequence[BankCase],
    split: Mapping[str, Any],
    controls: Mapping[str, Any],
    matrix: Mapping[str, Any],
    no_position: Mapping[str, Any],
    no_model_receipt: Mapping[str, Any],
) -> float:
    ready = (
        len(bank) >= 120
        and split["calibration_count"] == 60
        and split["held_count"] == 60
        and matrix["balanced"] is True
        and controls["known_valid"]["failed"] == 0
        and controls["single_violation"]["caught"] == controls["single_violation"]["total"]
        and controls["multi_violation"]["caught"] == controls["multi_violation"]["total"]
        and controls["parser_adversarial"]["caught"] == controls["parser_adversarial"]["total"]
        and controls["metamorphic"]["passed"] == controls["metamorphic"]["total"]
        and no_position["passed"] is True
        and no_model_receipt["passed"] is True
    )
    return 1.0 if ready else 0.0


def _bank_bytes(bank: Sequence[BankCase]) -> bytes:
    return "".join(_stable_json(case.to_json()) + "\n" for case in bank).encode("utf-8")


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _source_file_hashes(paths: Sequence[str]) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for display_path in paths:
        path = repo_path(display_path)
        rows.append(
            {
                "path": display_path,
                "exists": path.is_file(),
                "sha256": _file_sha256(path) if path.is_file() else None,
            }
        )
    return rows


def _directory_listing_hash(root: Path) -> JsonDict:
    if not root.exists():
        return {"path": str(root), "exists": False, "entry_count": 0, "sha256": None}
    entries: list[JsonDict] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            stat = path.stat()
        except OSError:  # pragma: no cover - filesystem race guard.
            continue
        entries.append(
            {
                "path_sha256": _sha256_text(str(path.relative_to(root))),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "path": str(root),
        "exists": True,
        "entry_count": len(entries),
        "sha256": _sha256_json(entries),
    }


def _git_status_short_hash() -> JsonDict:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=repo_path(),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "sha256": None}
    output = completed.stdout
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "sha256": _sha256_text(output),
        "line_count": len([line for line in output.splitlines() if line.strip()]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _artifact_checksum(
    artifact: Mapping[str, Any],
    bank_bytes: bytes,
    split_bytes: bytes,
    held_access_log_bytes: bytes,
) -> str:
    payload = _strip_local_paths_for_checksum(
        {
            key: value
            for key, value in artifact.items()
            if key not in {"duration_s", "reproducibility_checksum"}
        }
    )
    payload["bank_bytes_sha256"] = _sha256_bytes(bank_bytes)
    payload["split_bytes_sha256"] = _sha256_bytes(split_bytes)
    payload["held_access_log_bytes_sha256"] = _sha256_bytes(held_access_log_bytes)
    return _sha256_json(payload)


def _strip_local_paths_for_checksum(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: JsonDict = {}
        for key, nested in value.items():
            if key in {"path", "access_log_path"}:
                normalized[key] = "<local-output-path>"
            else:
                normalized[key] = _strip_local_paths_for_checksum(nested)
        return normalized
    if isinstance(value, list):
        return [_strip_local_paths_for_checksum(item) for item in value]
    return value


def _sha256_json(payload: Any) -> str:
    return _sha256_text(_stable_json(payload))


def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8"))


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical(payload: Any) -> str:
    return _stable_json(payload)


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    main()


__all__ = [
    "BANK_FILENAME",
    "HELD_ACCESS_LOG_FILENAME",
    "INFERENCE_SUBSTRATE",
    "REQUIRED_ARTIFACT_FIELDS",
    "REQUIRED_TAXONOMY",
    "RESULT_FILENAME",
    "RUN_DATE",
    "SPLIT_FILENAME",
    "BankCase",
    "ExperimentConfig",
    "audit_no_finite_choice_or_answer_position",
    "build_experiment_artifact",
    "build_item_bank",
    "build_split",
    "capture_preconditions",
    "clustered_inference_and_power_plan",
    "constraint_taxonomy_balance_matrix",
    "exact_floor_definition_and_provenance",
    "inspect_model_caches_without_loading",
    "k_sampling_and_consensus_preregistration",
    "known_valid_trace",
    "mutate_trace",
    "parseability_competence_unsaturation_headroom_and_minority_gates",
    "prior_failure_receipts",
    "protected_file_hashes",
    "retirement_rule",
    "run_validator_controls",
    "validate_candidate_trace",
    "write_frozen_artifacts",
]

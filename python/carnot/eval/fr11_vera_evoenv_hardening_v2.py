"""Exp 3142 FR-11 VeRA EvoEnv hardening v2.

Spec refs: REQ-LEARN-3142, SCENARIO-LEARN-3142,
SCENARIO-LEARN-3142-BLOCKED.

This module hardens the Exp 3128 executable environments without making a
model-weight-learning claim. The VeRA-E variants are equivalent rewrites of the
same exact constraint systems; the VeRA-H variants add deterministic checksum
constraints that the exact solver can verify and that reduce the solution
density. The memory-ledger replay keeps the Exp 3129 ledger blocker visible
instead of hiding it behind fresh solver-only success.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

from carnot.eval import fragment_time_monitor_satisfiable_drift_audit_v1 as monitor
from carnot.eval import fr11_constraint_memory_retention_drift_audit_v1 as memory_audit
from carnot.eval import fr11_evoenv_verifiable_environment_synthesis_v1 as evoenv


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3142_fr11_vera_evoenv_hardening_v2"
SCHEMA = "carnot.fr11.vera_evoenv_hardening.v2"
OUTPUT_REL_PATH = Path("results/experiment_3142_fr11_vera_evoenv_hardening_v2.json")
EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")
EXP3126_REL_PATH = Path(
    "results/experiment_3126_fragment_time_monitor_satisfiable_drift_audit_v1.json"
)
EXP3128_REL_PATH = Path(
    "results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"
)
EXP3129_REL_PATH = Path(
    "results/experiment_3129_fr11_constraint_memory_retention_drift_audit_v1.json"
)
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MANDATED_MODEL_SPECS = evoenv.MANDATED_MODEL_SPECS
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = {
    "fr11_vera_evoenv_v2_ready",
    "continuous_self_learning_targeted",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "live_model_variant_generation",
    "admitted_environment_count",
    "equivalent_variant_count",
    "hardened_variant_count",
    "solve_verify_asymmetry_pass_rate",
    "no_answer_leakage_pass_rate",
    "ledger_consistency_rate",
    "soundness_errors",
    "completeness_errors",
    "no_weight_update_claim",
    "promotion_recommendation",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' tests/python/test_experiment_3142_fr11_vera_evoenv_hardening_v2.py -q",
    ".venv/bin/coverage run --source=python/carnot/eval/fr11_vera_evoenv_hardening_v2.py -m pytest -o addopts='' tests/python/test_experiment_3142_fr11_vera_evoenv_hardening_v2.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_vera_evoenv_hardening_v2.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_authenticity_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("bmad_prd", Path("_bmad/prd.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3123_sota_preconditions", EXP3123_REL_PATH, True),
    ("exp3126_fragment_time_ledger", EXP3126_REL_PATH, False),
    ("exp3128_evoenv_admission", EXP3128_REL_PATH, True),
    ("exp3129_constraint_memory_audit", EXP3129_REL_PATH, True),
    (
        "exp3142_module",
        Path("python/carnot/eval/fr11_vera_evoenv_hardening_v2.py"),
        False,
    ),
    (
        "exp3142_tests",
        Path("tests/python/test_experiment_3142_fr11_vera_evoenv_hardening_v2.py"),
        False,
    ),
)


@dataclass(frozen=True)
class VariantRecord:
    """One generated VeRA variant plus the exact replay evidence for it."""

    source_environment_id: str
    variant_id: str
    variant_kind: str
    environment: evoenv.ConstraintEnvironment
    reference: evoenv.ReferenceSolution
    source_solution_count: int
    source_solver_evaluations: int
    determinism_passed: bool
    solve_verify_asymmetry_passed: bool
    no_answer_leakage_passed: bool
    novelty_passed: bool
    difficulty_passed: bool
    semantic_passed: bool
    solution_density_delta: float
    soundness_errors: int
    completeness_errors: int
    exact_replay_passed: bool

    def to_dict(self) -> JsonDict:
        return {
            "source_environment_id": self.source_environment_id,
            "variant_id": self.variant_id,
            "variant_kind": self.variant_kind,
            "environment": self.environment.to_dict(),
            "reference": self.reference.to_dict(),
            "source_solution_count": self.source_solution_count,
            "source_solver_evaluations": self.source_solver_evaluations,
            "determinism_passed": self.determinism_passed,
            "solve_verify_asymmetry_passed": self.solve_verify_asymmetry_passed,
            "no_answer_leakage_passed": self.no_answer_leakage_passed,
            "novelty_passed": self.novelty_passed,
            "difficulty_passed": self.difficulty_passed,
            "semantic_passed": self.semantic_passed,
            "solution_density_delta": self.solution_density_delta,
            "soundness_errors": self.soundness_errors,
            "completeness_errors": self.completeness_errors,
            "exact_replay_passed": self.exact_replay_passed,
        }


@dataclass(frozen=True)
class VariantSummary:
    """Aggregate metrics across all VeRA-E and VeRA-H records."""

    records: tuple[VariantRecord, ...]

    @property
    def equivalent_variant_count(self) -> int:
        return sum(1 for record in self.records if record.variant_kind == "equivalent")

    @property
    def hardened_variant_count(self) -> int:
        return sum(1 for record in self.records if record.variant_kind == "hardened")

    @property
    def soundness_errors(self) -> int:
        return sum(record.soundness_errors for record in self.records)

    @property
    def completeness_errors(self) -> int:
        return sum(record.completeness_errors for record in self.records)

    @property
    def solve_verify_asymmetry_pass_rate(self) -> float:
        return pass_rate(record.solve_verify_asymmetry_passed for record in self.records)

    @property
    def no_answer_leakage_pass_rate(self) -> float:
        return pass_rate(record.no_answer_leakage_passed for record in self.records)

    @property
    def novelty_pass_rate(self) -> float:
        return pass_rate(record.novelty_passed for record in self.records)

    @property
    def difficulty_pass_rate(self) -> float:
        return pass_rate(record.difficulty_passed for record in self.records)

    @property
    def determinism_pass_rate(self) -> float:
        return pass_rate(record.determinism_passed for record in self.records)

    @property
    def semantic_pass_rate(self) -> float:
        return pass_rate(record.semantic_passed for record in self.records)

    @property
    def exact_replay_pass_rate(self) -> float:
        return pass_rate(record.exact_replay_passed for record in self.records)


def read_json_object(path: Path) -> JsonDict:
    """Read one JSON object and fail closed to empty evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_admitted_environments(exp3128: Mapping[str, Any]) -> tuple[evoenv.ConstraintEnvironment, ...]:
    """REQ-LEARN-3142-1: reconstruct every admitted Exp 3128 environment."""

    rows = [
        row for row in exp3128.get("admitted_environments", []) if isinstance(row, Mapping)
    ]
    return tuple(memory_audit.environment_from_row(row) for row in rows)


def generate_and_validate_variants(
    admitted: Sequence[evoenv.ConstraintEnvironment],
) -> VariantSummary:
    """Generate one VeRA-E and one VeRA-H variant for each admitted environment."""

    prior_signatures = {evoenv.environment_signature(environment) for environment in admitted}
    seen_signatures = set(prior_signatures)
    records: list[VariantRecord] = []
    for index, source in enumerate(admitted):
        equivalent = equivalent_variant(source, index)
        records.append(validate_variant(source, equivalent, "equivalent", seen_signatures))
        seen_signatures.add(evoenv.environment_signature(equivalent))
        hardened = hardened_variant(source, index)
        records.append(validate_variant(source, hardened, "hardened", seen_signatures))
        seen_signatures.add(evoenv.environment_signature(hardened))
    return VariantSummary(records=tuple(records))


def equivalent_variant(
    source: evoenv.ConstraintEnvironment,
    index: int,
) -> evoenv.ConstraintEnvironment:
    """REQ-LEARN-3142-2: rename variables while preserving exact solutions."""

    rename_map = {variable: f"e{index}_{offset}" for offset, variable in enumerate(source.variables)}
    variables = tuple(rename_map[variable] for variable in source.variables)
    domains = {
        rename_map[variable]: tuple(source.domains[variable]) for variable in source.variables
    }
    constraints = tuple(rename_constraint(constraint, rename_map) for constraint in source.constraints)
    return evoenv.ConstraintEnvironment(
        environment_id=f"{source.environment_id}-vera-e-{index}",
        family_id=f"{source.family_id}_vera_equivalent",
        variables=variables,
        domains=domains,
        constraints=constraints,
        prompt=(
            "Find an assignment for variables "
            f"{', '.join(variables)} satisfying the attached equivalent executable "
            f"constraints for {source.family_id}. Respond as JSON."
        ),
    )


def hardened_variant(
    source: evoenv.ConstraintEnvironment,
    index: int,
) -> evoenv.ConstraintEnvironment:
    """REQ-LEARN-3142-3: add a deterministic checksum that reduces density."""

    reference = source.compute_reference()
    modulus = checksum_modulus(source)
    checksum = sum(int(reference.canonical_assignment[var]) for var in source.variables) % modulus
    constraints = source.constraints + (
        evoenv.ConstraintSpec(
            "linear_mod",
            terms={variable: 1 for variable in source.variables},
            modulus=modulus,
            equals=checksum,
        ),
    )
    return evoenv.ConstraintEnvironment(
        environment_id=f"{source.environment_id}-vera-h-{index}",
        family_id=f"{source.family_id}_vera_hardened",
        variables=source.variables,
        domains=source.domains,
        constraints=constraints,
        prompt=(
            "Find an assignment satisfying the original executable constraints plus "
            f"a deterministic VeRA hardening checksum for {source.family_id}. "
            "Respond as JSON."
        ),
    )


def rename_constraint(
    constraint: evoenv.ConstraintSpec,
    rename_map: Mapping[str, str],
) -> evoenv.ConstraintSpec:
    """Rewrite a serializable constraint under a bijective variable rename."""

    terms = {
        rename_map[str(variable)]: int(coef)
        for variable, coef in dict(constraint.terms or {}).items()
    }
    left = rename_map.get(str(constraint.left), constraint.left) if constraint.left else None
    right = rename_map.get(str(constraint.right), constraint.right) if constraint.right else None
    return evoenv.ConstraintSpec(
        kind=constraint.kind,
        terms=terms,
        modulus=constraint.modulus,
        equals=constraint.equals,
        lower=constraint.lower,
        upper=constraint.upper,
        left=left,
        right=right,
    )


def checksum_modulus(environment: evoenv.ConstraintEnvironment) -> int:
    """Choose a stable checksum modulus from the first variable domain."""

    first_domain = environment.domains[environment.variables[0]]
    return max(2, len(first_domain))


def validate_variant(
    source: evoenv.ConstraintEnvironment,
    variant: evoenv.ConstraintEnvironment,
    variant_kind: str,
    seen_signatures: set[str],
) -> VariantRecord:
    """Run every executable gate for a single generated variant."""

    reference = variant.compute_reference()
    source_reference = source.compute_reference()
    replay = exact_variant_replay(variant)
    signature = evoenv.environment_signature(variant)
    source_density = rate(source_reference.solution_count, source_reference.solver_evaluations)
    variant_density = rate(reference.solution_count, reference.solver_evaluations)
    density_delta = round_float(variant_density - source_density)
    semantic_passed = (
        reference.solution_count == source_reference.solution_count
        if variant_kind == "equivalent"
        else 0 < reference.solution_count < source_reference.solution_count
    )
    return VariantRecord(
        source_environment_id=source.environment_id,
        variant_id=variant.environment_id,
        variant_kind=variant_kind,
        environment=variant,
        reference=reference,
        source_solution_count=source_reference.solution_count,
        source_solver_evaluations=source_reference.solver_evaluations,
        determinism_passed=reference == variant.compute_reference(),
        solve_verify_asymmetry_passed=evoenv.solve_verify_asymmetry_passed(reference),
        no_answer_leakage_passed=no_answer_leakage_passed(variant, reference),
        novelty_passed=signature not in seen_signatures,
        difficulty_passed=evoenv.difficulty_calibrated(reference),
        semantic_passed=semantic_passed,
        solution_density_delta=density_delta,
        soundness_errors=int(replay["soundness_errors"]),
        completeness_errors=int(replay["completeness_errors"]),
        exact_replay_passed=bool(replay["exact_replay_passed"]),
    )


def exact_variant_replay(environment: evoenv.ConstraintEnvironment) -> JsonDict:
    """REQ-LEARN-3142-4: compare cheap verify against exact truth everywhere."""

    reference = environment.compute_reference()
    soundness_errors = 0
    completeness_errors = 0
    valid_assignment_count = 0
    for values in itertools.product(*(environment.domains[var] for var in environment.variables)):
        assignment = dict(zip(environment.variables, values, strict=True))
        exact_valid = all(
            evoenv.constraint_holds(constraint, assignment)
            for constraint in environment.constraints
        )
        verifier_accepts = environment.score_response(assignment).accepted
        valid_assignment_count += int(exact_valid)
        soundness_errors += int(verifier_accepts and not exact_valid)
        completeness_errors += int(exact_valid and not verifier_accepts)
    return {
        "variant_id": environment.environment_id,
        "assignment_count": reference.solver_evaluations,
        "valid_assignment_count": valid_assignment_count,
        "soundness_errors": soundness_errors,
        "completeness_errors": completeness_errors,
        "exact_replay_passed": (
            valid_assignment_count > 0 and soundness_errors == 0 and completeness_errors == 0
        ),
    }


def no_answer_leakage_passed(
    environment: evoenv.ConstraintEnvironment,
    reference: evoenv.ReferenceSolution,
) -> bool:
    """Reject prompts that contain the exact canonical response string."""

    return not evoenv.prompt_leaks_answer(environment, reference)


def ledger_replay_summary(
    exp3129: Mapping[str, Any],
    variant_records: Sequence[VariantRecord],
    exp3126: Mapping[str, Any],
) -> JsonDict:
    """REQ-LEARN-3142-5: combine prior and new exact ledger measurements."""

    prior = prior_ledger_counts(exp3129, exp3126)
    new_observed = len(variant_records)
    new_consistent = sum(1 for record in variant_records if variant_ledger_consistent(record))
    combined_observed = int(prior["observed_final_answer_count"]) + new_observed
    combined_consistent = int(prior["ledger_consistent_final_answer_count"]) + new_consistent
    return {
        "prior_ledger_consistency_rate": float(prior["ledger_consistency_rate"]),
        "prior_observed_final_answer_count": int(prior["observed_final_answer_count"]),
        "prior_ledger_consistent_final_answer_count": int(
            prior["ledger_consistent_final_answer_count"]
        ),
        "new_variant_ledger_consistency_rate": rate(new_consistent, new_observed),
        "new_variant_observed_final_answer_count": new_observed,
        "new_variant_ledger_consistent_final_answer_count": new_consistent,
        "combined_observed_final_answer_count": combined_observed,
        "combined_ledger_consistent_final_answer_count": combined_consistent,
        "ledger_consistency_rate": rate(combined_consistent, combined_observed),
        "measurement": "prior_fragment_time_ledger_plus_exact_vera_variant_ledger",
    }


def prior_ledger_counts(exp3129: Mapping[str, Any], exp3126: Mapping[str, Any]) -> JsonDict:
    """Return prior ledger numerator and denominator when monitor events exist."""

    events = exp3126.get("monitor_events")
    if isinstance(events, list):
        replay = monitor.replay_monitor_events(events)
        return {
            "ledger_consistency_rate": float(replay["ledger_consistency_rate"]),
            "observed_final_answer_count": int(replay["observed_final_answer_count"]),
            "ledger_consistent_final_answer_count": int(
                replay["ledger_consistent_final_answer_count"]
            ),
        }
    return {
        "ledger_consistency_rate": float(exp3129.get("ledger_consistency_rate") or 0.0),
        "observed_final_answer_count": 0,
        "ledger_consistent_final_answer_count": 0,
    }


def variant_ledger_consistent(record: VariantRecord) -> bool:
    """A variant ledger row is consistent when the exact reference is accepted."""

    return (
        record.exact_replay_passed
        and record.soundness_errors == 0
        and record.completeness_errors == 0
        and record.environment.score_response(record.reference.canonical_assignment).accepted
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3142 terminal artifact from checked-in evidence."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3123 = read_json_object(root_path / EXP3123_REL_PATH)
    exp3128 = read_json_object(root_path / EXP3128_REL_PATH)
    exp3129 = read_json_object(root_path / EXP3129_REL_PATH)
    exp3126 = read_json_object(root_path / EXP3126_REL_PATH)
    blocker = precondition_blocker(exp3123, exp3128, exp3129)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    admitted = load_admitted_environments(exp3128)
    variants = generate_and_validate_variants(admitted)
    ledger = ledger_replay_summary(exp3129, variants.records, exp3126)
    exp3129_soundness = int(exp3129.get("soundness_errors") or 0)
    exp3129_completeness = int(exp3129.get("completeness_errors") or 0)
    soundness_errors = variants.soundness_errors + exp3129_soundness
    completeness_errors = variants.completeness_errors + exp3129_completeness
    declared_admitted = int(exp3128.get("admitted_environment_count") or 0)
    count_matches = declared_admitted == len(admitted)
    ready = (
        count_matches
        and declared_admitted > 0
        and variants.equivalent_variant_count >= declared_admitted
        and variants.hardened_variant_count >= declared_admitted
        and variants.soundness_errors == 0
        and variants.completeness_errors == 0
        and variants.solve_verify_asymmetry_pass_rate == 1.0
        and variants.no_answer_leakage_pass_rate == 1.0
        and variants.novelty_pass_rate == 1.0
        and variants.difficulty_pass_rate == 1.0
        and variants.semantic_pass_rate == 1.0
        and variants.exact_replay_pass_rate == 1.0
        and exp3129_soundness == 0
        and exp3129_completeness == 0
    )
    recommendation = promotion_recommendation(
        ready,
        soundness_errors,
        completeness_errors,
        float(ledger["ledger_consistency_rate"]),
    )
    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_vera_evoenv_v2_ready": ready,
        "continuous_self_learning_targeted": True,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "selected_model_ids": [],
        "live_call_count": 0,
        "live_model_variant_generation": False,
        "admitted_environment_count": declared_admitted,
        "equivalent_variant_count": variants.equivalent_variant_count,
        "hardened_variant_count": variants.hardened_variant_count,
        "solve_verify_asymmetry_pass_rate": variants.solve_verify_asymmetry_pass_rate,
        "no_answer_leakage_pass_rate": variants.no_answer_leakage_pass_rate,
        "ledger_consistency_rate": float(ledger["ledger_consistency_rate"]),
        "soundness_errors": soundness_errors,
        "completeness_errors": completeness_errors,
        "no_weight_update_claim": True,
        "promotion_recommendation": recommendation,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": inference_substrate(exp3123),
        "precondition_checks": precondition_checks(exp3123, exp3128, exp3129),
        "admitted_environment_ids": [environment.environment_id for environment in admitted],
        "variant_records": [record.to_dict() for record in variants.records],
        "variant_validation_summary": {
            "determinism_pass_rate": variants.determinism_pass_rate,
            "novelty_pass_rate": variants.novelty_pass_rate,
            "difficulty_pass_rate": variants.difficulty_pass_rate,
            "semantic_pass_rate": variants.semantic_pass_rate,
            "exact_replay_pass_rate": variants.exact_replay_pass_rate,
            "exp3129_prior_soundness_errors": exp3129_soundness,
            "exp3129_prior_completeness_errors": exp3129_completeness,
            "admitted_count_matches_exp3128": count_matches,
        },
        "ledger_replay_summary": ledger,
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(ready, recommendation),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    start: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """Return a schema-complete failed-closed artifact."""

    artifact = {
        "artifact": ARTIFACT,
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "fr11_vera_evoenv_v2_ready": False,
        "continuous_self_learning_targeted": True,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "selected_model_ids": [],
        "live_call_count": 0,
        "live_model_variant_generation": False,
        "admitted_environment_count": 0,
        "equivalent_variant_count": 0,
        "hardened_variant_count": 0,
        "solve_verify_asymmetry_pass_rate": 0.0,
        "no_answer_leakage_pass_rate": 0.0,
        "ledger_consistency_rate": 0.0,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "no_weight_update_claim": True,
        "promotion_recommendation": "block_fr11_vera_evoenv_missing_source_evidence",
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root),
        "inference_substrate": inference_substrate({}, mode="blocked_precondition_check"),
        "precondition_checks": {
            "exp3123_manifest_ready": False,
            "exp3128_evoenv_ready": False,
            "exp3129_constraint_memory_ready": False,
        },
        "variant_records": [],
        "variant_validation_summary": {},
        "ledger_replay_summary": {
            "prior_ledger_consistency_rate": 0.0,
            "new_variant_ledger_consistency_rate": 0.0,
        },
        "blocked_reason": blocker,
        "duration_s": duration(start, now_s),
        "honest_verdict": f"blocked_precondition_failed: {blocker}",
    }
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and write the Exp 3142 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    write_json(output, artifact)
    return output


def precondition_blocker(
    exp3123: Mapping[str, Any],
    exp3128: Mapping[str, Any],
    exp3129: Mapping[str, Any],
) -> str:
    """Return the first missing source for the VeRA hardening pass."""

    if exp3123.get("sota_cache_manifest_v2_ready") is not True:
        return "exp3123_precondition_manifest_missing_or_empty"
    if exp3128.get("fr11_evoenv_pilot_v1_ready") is not True:
        return "exp3128_evoenv_artifact_missing_or_not_ready"
    if exp3129.get("fr11_constraint_memory_audit_v1_ready") is not True:
        return "exp3129_constraint_memory_audit_missing_or_not_ready"
    return ""


def precondition_checks(
    exp3123: Mapping[str, Any],
    exp3128: Mapping[str, Any],
    exp3129: Mapping[str, Any],
) -> JsonDict:
    """Expose source readiness in the final artifact."""

    present = [
        model_id
        for model_id in exp3123.get("present_model_ids", [])
        if model_id in MANDATED_MODEL_SPECS
    ]
    return {
        "exp3123_manifest_ready": exp3123.get("sota_cache_manifest_v2_ready") is True,
        "exp3128_evoenv_ready": exp3128.get("fr11_evoenv_pilot_v1_ready") is True,
        "exp3129_constraint_memory_ready": (
            exp3129.get("fr11_constraint_memory_audit_v1_ready") is True
        ),
        "present_mandated_model_ids": present,
        "legacy_small_models_excluded_from_headline": True,
    }


def inference_substrate(
    exp3123: Mapping[str, Any],
    mode: str = "solver_only_vera_variant_generation",
) -> JsonDict:
    """Separate executable variant generation from model-weight learning."""

    present = [
        model_id
        for model_id in exp3123.get("present_model_ids", [])
        if model_id in MANDATED_MODEL_SPECS
    ]
    return {
        "mode": mode,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "present_mandated_model_ids": present,
        "selected_model_ids": [],
        "live_call_count": 0,
        "live_model_variant_generation": False,
        "solver_only_variant_generation": True,
        "executes_exact_solvers": True,
        "executes_model_generation": False,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """List traceable source files and artifacts with stable checksums."""

    rows: list[JsonDict] = []
    for source_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        rows.append(
            {
                "id": source_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    return rows


def promotion_recommendation(
    ready: bool,
    soundness_errors: int,
    completeness_errors: int,
    ledger_consistency_rate: float,
) -> str:
    """REQ-LEARN-3142-6: keep controller memory separate from weight claims."""

    if not ready:
        return "block_fr11_vera_evoenv_hardening_until_exact_variant_gates_pass"
    if soundness_errors or completeness_errors:
        return "block_fr11_promotion_soundness_or_completeness_regression"
    if ledger_consistency_rate < 1.0:
        return (
            "promote_controller_environment_memory_only_"
            "block_model_weight_learning_until_ledger_consistency_is_1.0"
        )
    return "promote_controller_environment_memory_only"


def honest_verdict(ready: bool, recommendation: str) -> str:
    """Return a conductor-compatible terminal verdict string."""

    if ready:
        return (
            "complete: fr11_vera_evoenv_v2_ready=true; "
            f"promotion_recommendation={recommendation}; no model-weight update claimed"
        )
    return "blocked_precondition_failed: fr11_vera_evoenv_v2_ready=false"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise if the Exp 3142 artifact violates the hardening contract."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("live_model_variant_generation") and int(artifact.get("live_call_count") or 0) <= 0:
        raise ValueError("live_call_count must be positive for live variant generation")
    if artifact.get("no_weight_update_claim") is not True:
        raise ValueError("no_weight_update_claim must be true")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or any(
        substrate.get(flag) is True
        for flag in ("model_weight_mutation", "model_weight_training", "base_model_weights_updated")
    ):
        raise ValueError("model_weight_mutation must remain false")
    ledger_rate = float(artifact.get("ledger_consistency_rate", math.nan))
    if not math.isfinite(ledger_rate) or not 0.0 <= ledger_rate <= 1.0:
        raise ValueError("ledger_consistency_rate must be finite and within [0, 1]")
    if artifact.get("fr11_vera_evoenv_v2_ready") is not True:
        return
    if int(artifact["admitted_environment_count"]) <= 0:
        raise ValueError("admitted_environment_count must be positive for readiness")
    if int(artifact["equivalent_variant_count"]) <= 0:
        raise ValueError("equivalent_variant_count must be positive for readiness")
    if int(artifact["hardened_variant_count"]) <= 0:
        raise ValueError("hardened_variant_count must be positive for readiness")
    if int(artifact["soundness_errors"]) != 0:
        raise ValueError("soundness_errors must be zero for readiness")
    if int(artifact["completeness_errors"]) != 0:
        raise ValueError("completeness_errors must be zero for readiness")
    if float(artifact["solve_verify_asymmetry_pass_rate"]) != 1.0:
        raise ValueError("solve_verify_asymmetry_pass_rate must be 1.0 for readiness")
    if float(artifact["no_answer_leakage_pass_rate"]) != 1.0:
        raise ValueError("no_answer_leakage_pass_rate must be 1.0 for readiness")
    if any(
        row.get("required") and not row.get("exists")
        for row in artifact.get("source_artifacts", [])
        if isinstance(row, Mapping)
    ):
        raise ValueError("required source_artifacts must exist")
    if not str(artifact.get("honest_verdict") or "").startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")


def pass_rate(values: Iterable[bool]) -> float:
    """Return the rounded fraction of truthy checks."""

    observed = tuple(values)
    return rate(sum(1 for value in observed if value), len(observed))


def rate(numerator: int | float, denominator: int | float) -> float:
    """Return a rounded ratio with an explicit zero-denominator convention."""

    if denominator == 0:
        return 0.0
    return round_float(float(numerator) / float(denominator))


def round_float(value: float) -> float:
    """Round artifact metrics to the precision used by nearby experiments."""

    return round(float(value), 6)


def duration(started_s: float, now_s: float | None) -> float:
    """Return nonnegative elapsed seconds for reproducible tests."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round_float(max(0.0, end - started_s))


def sha256_file(path: Path) -> str:
    """Return a SHA-256 checksum for a source file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write stable JSON output so experiment artifacts diff cleanly."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    output = write_artifact(REPO_ROOT)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["fr11_vera_evoenv_v2_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

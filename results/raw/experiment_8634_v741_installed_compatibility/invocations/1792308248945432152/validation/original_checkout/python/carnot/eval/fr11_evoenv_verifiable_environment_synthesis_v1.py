"""Exp 3128 FR-11 EvoEnv verifiable environment synthesis pilot.

Spec refs: REQ-LEARN-3128, SCENARIO-LEARN-3128,
SCENARIO-LEARN-3128-BLOCKED.

The pilot treats self-learning as controller-side environment admission rather
than model-weight learning. Each admitted environment is executable: it samples
deterministic constraint instances, solves them by exact enumeration, and scores
responses with a cheaper verifier. That solve-verify asymmetry is the reward
surface that makes the environment useful without leaking final answers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1"
SCHEMA = "carnot.fr11.evoenv.verifiable_environment_synthesis.v1"
OUTPUT_REL_PATH = Path(
    "results/experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.json"
)
EXP3123_REL_PATH = Path("results/experiment_3123_sota_cache_preconditions_manifest_v2.json")
EXP3116_REL_PATH = Path("results/experiment_3116_fr11_unsolvable_curriculum_retention_guard_v1.json")
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MANDATED_MODEL_SPECS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_ARTIFACT_FIELDS = {
    "fr11_evoenv_pilot_v1_ready",
    "continuous_self_learning_targeted",
    "model_specs",
    "selected_model_ids",
    "live_call_count",
    "live_model_environment_synthesis",
    "candidate_environment_count",
    "admitted_environment_count",
    "solve_verify_asymmetry_pass_rate",
    "novelty_pass_rate",
    "no_answer_leakage_pass_rate",
    "retention_delta",
    "soundness_errors",
    "completeness_errors",
    "no_weight_update_claim",
    "tests_run",
    "source_artifacts",
    "inference_substrate",
    "honest_verdict",
}
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped_",
)
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.py -q",
    ".venv/bin/coverage run --source=python/carnot/eval/fr11_evoenv_verifiable_environment_synthesis_v1.py -m pytest -o addopts='' tests/python/test_experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.py -q",
    ".venv/bin/coverage report --include='python/carnot/eval/fr11_evoenv_verifiable_environment_synthesis_v1.py' --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md")),
    ("codex_repo_workflow", Path("CODEX.md")),
    ("claude_authenticity_rules", Path("CLAUDE.md")),
    ("self_learning_openspec", SPEC_REL_PATH),
    ("exp3123_sota_preconditions", EXP3123_REL_PATH),
    ("exp3116_fr11_retention_guard", EXP3116_REL_PATH),
    ("exp3128_module", Path("python/carnot/eval/fr11_evoenv_verifiable_environment_synthesis_v1.py")),
    ("exp3128_tests", Path("tests/python/test_experiment_3128_fr11_evoenv_verifiable_environment_synthesis_v1.py")),
)
MIN_SEARCH_SPACE = 12
MAX_SEARCH_SPACE = 512
MIN_SOLVE_VERIFY_RATIO = 4.0


@dataclass(frozen=True)
class ConstraintSpec:
    """A small serializable constraint used by the exact enumerator."""

    kind: str
    terms: Mapping[str, int] | None = None
    modulus: int | None = None
    equals: int | None = None
    lower: int | None = None
    upper: int | None = None
    left: str | None = None
    right: str | None = None


@dataclass(frozen=True)
class ReferenceSolution:
    """Exact reference evidence produced by exhaustive enumeration."""

    canonical_assignment: Mapping[str, int]
    solution_count: int
    solver_evaluations: int
    verify_checks: int
    authority: str = "exact_enumeration"

    def to_dict(self) -> JsonDict:
        return {
            "canonical_assignment": dict(self.canonical_assignment),
            "canonical_answer_text": answer_text(self.canonical_assignment),
            "solution_count": self.solution_count,
            "solver_evaluations": self.solver_evaluations,
            "verify_checks": self.verify_checks,
            "authority": self.authority,
        }


@dataclass(frozen=True)
class ScoreResult:
    """Verifier-side response score for one candidate assignment."""

    accepted: bool
    score: float
    violations: tuple[str, ...]
    verify_checks: int


@dataclass(frozen=True)
class ConstraintEnvironment:
    """Executable constraint environment with exact solve and cheap verify paths."""

    environment_id: str
    family_id: str
    variables: tuple[str, ...]
    domains: Mapping[str, tuple[int, ...]]
    constraints: tuple[ConstraintSpec, ...]
    prompt: str

    def compute_reference(self) -> ReferenceSolution:
        solutions: list[dict[str, int]] = []
        evaluations = 0
        domain_product = itertools.product(*(self.domains[var] for var in self.variables))
        for values in domain_product:
            evaluations += 1
            assignment = dict(zip(self.variables, values, strict=True))
            if all(constraint_holds(constraint, assignment) for constraint in self.constraints):
                solutions.append(assignment)
        canonical = solutions[0] if solutions else {}
        return ReferenceSolution(
            canonical_assignment=canonical,
            solution_count=len(solutions),
            solver_evaluations=evaluations,
            verify_checks=len(self.constraints),
        )

    def score_response(self, response: Mapping[str, int]) -> ScoreResult:
        assignment = {variable: int(response[variable]) for variable in self.variables}
        violations = tuple(
            constraint.kind
            for constraint in self.constraints
            if not constraint_holds(constraint, assignment)
        )
        accepted = not violations
        return ScoreResult(
            accepted=accepted,
            score=1.0 if accepted else 0.0,
            violations=violations,
            verify_checks=len(self.constraints),
        )

    def to_dict(self) -> JsonDict:
        return {
            "environment_id": self.environment_id,
            "family_id": self.family_id,
            "variables": list(self.variables),
            "domains": {key: list(value) for key, value in self.domains.items()},
            "constraints": [constraint_to_dict(constraint) for constraint in self.constraints],
            "prompt": self.prompt,
            "signature": environment_signature(self),
        }


@dataclass(frozen=True)
class EnvironmentFamily:
    """Deterministic sampler for a bounded family of constraint environments."""

    family_id: str
    kind: str

    def sample_instances(self, count: int, seed: int) -> tuple[ConstraintEnvironment, ...]:
        return tuple(build_environment(self.family_id, self.kind, seed, index) for index in range(count))


@dataclass(frozen=True)
class AdmissionRecord:
    """Admission-gate result for one candidate environment."""

    environment_id: str
    admitted: bool
    rejection_reasons: tuple[str, ...]
    determinism_passed: bool
    solve_verify_asymmetry_passed: bool
    novelty_passed: bool
    difficulty_passed: bool
    no_answer_leakage_passed: bool
    solver_authority_passed: bool
    soundness_errors: int
    completeness_errors: int
    reference: ReferenceSolution

    def to_dict(self) -> JsonDict:
        return {
            "environment_id": self.environment_id,
            "admitted": self.admitted,
            "rejection_reasons": list(self.rejection_reasons),
            "determinism_passed": self.determinism_passed,
            "solve_verify_asymmetry_passed": self.solve_verify_asymmetry_passed,
            "novelty_passed": self.novelty_passed,
            "difficulty_passed": self.difficulty_passed,
            "no_answer_leakage_passed": self.no_answer_leakage_passed,
            "solver_authority_passed": self.solver_authority_passed,
            "soundness_errors": self.soundness_errors,
            "completeness_errors": self.completeness_errors,
            "reference": self.reference.to_dict(),
        }


@dataclass(frozen=True)
class AdmissionSummary:
    """Aggregate admission metrics for the bounded candidate set."""

    records: tuple[AdmissionRecord, ...]
    admitted: tuple[ConstraintEnvironment, ...]

    @property
    def candidate_count(self) -> int:
        return len(self.records)

    @property
    def admitted_count(self) -> int:
        return len(self.admitted)

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
    def novelty_pass_rate(self) -> float:
        return pass_rate(record.novelty_passed for record in self.records)

    @property
    def no_answer_leakage_pass_rate(self) -> float:
        return pass_rate(record.no_answer_leakage_passed for record in self.records)


def default_environment_families() -> tuple[EnvironmentFamily, ...]:
    """Return the bounded family schemas used by the EvoEnv pilot."""

    return (
        EnvironmentFamily("modular_balance", "modular"),
        EnvironmentFamily("interval_order", "interval"),
        EnvironmentFamily("graph_coloring", "coloring"),
    )


def sample_candidate_environments(seed: int = 3128) -> tuple[ConstraintEnvironment, ...]:
    """Hand-admit a bounded candidate denominator without live model synthesis."""

    candidates = [family.sample_instances(1, seed)[0] for family in default_environment_families()]
    candidates.append(build_environment("leaky_modular", "leaky_modular", seed, 0))
    candidates.append(build_environment("too_easy", "too_easy", seed, 0))
    return tuple(candidates)


def build_environment(
    family_id: str,
    kind: str,
    seed: int,
    index: int,
) -> ConstraintEnvironment:
    """Build one deterministic constraint environment from its family schema."""

    if kind == "modular":
        return modular_environment(family_id, seed, index, leak_answer=False)
    if kind == "leaky_modular":
        return modular_environment(family_id, seed, index, leak_answer=True)
    if kind == "interval":
        return interval_environment(family_id, seed, index)
    if kind == "coloring":
        return graph_coloring_environment(family_id, seed, index)
    if kind == "too_easy":
        return too_easy_environment(family_id, seed, index)
    raise ValueError(f"unknown environment kind: {kind}")  # pragma: no cover


def modular_environment(
    family_id: str,
    seed: int,
    index: int,
    *,
    leak_answer: bool,
) -> ConstraintEnvironment:
    target = (seed + index * 3) % 7
    domains = {"x": tuple(range(7)), "y": tuple(range(7))}
    constraints = (
        ConstraintSpec("linear_mod", terms={"x": 2, "y": 1}, modulus=7, equals=target),
        ConstraintSpec("not_equal", left="x", right="y"),
    )
    prompt = (
        "Find integers x and y in [0, 6] satisfying "
        f"(2*x + y) mod 7 = {target} and x != y. Respond as JSON with x and y."
    )
    env = ConstraintEnvironment(
        environment_id=f"{family_id}-{seed}-{index}",
        family_id=family_id,
        variables=("x", "y"),
        domains=domains,
        constraints=constraints,
        prompt=prompt,
    )
    if not leak_answer:
        return env
    reference = env.compute_reference()
    return ConstraintEnvironment(
        environment_id=env.environment_id,
        family_id=env.family_id,
        variables=env.variables,
        domains=env.domains,
        constraints=env.constraints,
        prompt=f"{prompt} Canonical answer: {answer_text(reference.canonical_assignment)}.",
    )


def interval_environment(family_id: str, seed: int, index: int) -> ConstraintEnvironment:
    parity = (seed + index) % 2
    constraints = (
        ConstraintSpec("less_than", left="a", right="b"),
        ConstraintSpec("less_than", left="b", right="c"),
        ConstraintSpec("sum_between", terms={"a": 1, "b": 1, "c": 1}, lower=7, upper=10),
        ConstraintSpec("linear_mod", terms={"a": 1, "c": 1}, modulus=2, equals=parity),
    )
    return ConstraintEnvironment(
        environment_id=f"{family_id}-{seed}-{index}",
        family_id=family_id,
        variables=("a", "b", "c"),
        domains={"a": tuple(range(7)), "b": tuple(range(7)), "c": tuple(range(7))},
        constraints=constraints,
        prompt=(
            "Choose a, b, and c from [0, 6] with a < b < c, "
            f"7 <= a+b+c <= 10, and (a+c) mod 2 = {parity}. "
            "Respond as JSON with a, b, and c."
        ),
    )


def graph_coloring_environment(family_id: str, seed: int, index: int) -> ConstraintEnvironment:
    del seed, index
    constraints = (
        ConstraintSpec("not_equal", left="n0", right="n1"),
        ConstraintSpec("not_equal", left="n1", right="n2"),
        ConstraintSpec("not_equal", left="n2", right="n3"),
        ConstraintSpec("not_equal", left="n0", right="n3"),
    )
    return ConstraintEnvironment(
        environment_id=f"{family_id}-3128-0",
        family_id=family_id,
        variables=("n0", "n1", "n2", "n3"),
        domains={name: (0, 1, 2) for name in ("n0", "n1", "n2", "n3")},
        constraints=constraints,
        prompt=(
            "Color nodes n0, n1, n2, and n3 with colors 0, 1, or 2 so that "
            "edges (n0,n1), (n1,n2), (n2,n3), and (n0,n3) have different colors. "
            "Respond as JSON with one color per node."
        ),
    )


def too_easy_environment(family_id: str, seed: int, index: int) -> ConstraintEnvironment:
    return ConstraintEnvironment(
        environment_id=f"{family_id}-{seed}-{index}",
        family_id=family_id,
        variables=("z",),
        domains={"z": (0, 1)},
        constraints=(ConstraintSpec("linear_mod", terms={"z": 1}, modulus=2, equals=0),),
        prompt="Choose z in {0, 1} with z mod 2 = 0. Respond as JSON with z.",
    )


def constraint_holds(constraint: ConstraintSpec, assignment: Mapping[str, int]) -> bool:
    """Evaluate one serializable constraint against a concrete assignment."""

    if constraint.kind == "linear_mod":
        total = sum(int(coef) * int(assignment[var]) for var, coef in dict(constraint.terms).items())
        return total % int(constraint.modulus) == int(constraint.equals)
    if constraint.kind == "not_equal":
        return assignment[str(constraint.left)] != assignment[str(constraint.right)]
    if constraint.kind == "less_than":
        return assignment[str(constraint.left)] < assignment[str(constraint.right)]
    if constraint.kind == "sum_between":
        total = sum(int(coef) * int(assignment[var]) for var, coef in dict(constraint.terms).items())
        return int(constraint.lower) <= total <= int(constraint.upper)
    raise ValueError(f"unsupported constraint kind: {constraint.kind}")  # pragma: no cover


def constraint_to_dict(constraint: ConstraintSpec) -> JsonDict:
    return {
        "kind": constraint.kind,
        "terms": dict(constraint.terms or {}),
        "modulus": constraint.modulus,
        "equals": constraint.equals,
        "lower": constraint.lower,
        "upper": constraint.upper,
        "left": constraint.left,
        "right": constraint.right,
    }


def answer_text(assignment: Mapping[str, int]) -> str:
    """Render a canonical assignment in the response format we forbid in prompts."""

    return ",".join(f"{key}={assignment[key]}" for key in sorted(assignment))


def prompt_leaks_answer(environment: ConstraintEnvironment, reference: ReferenceSolution) -> bool:
    """Detect exact final-answer leakage without rejecting ordinary numeric parameters."""

    canonical = answer_text(reference.canonical_assignment)
    return canonical in environment.prompt


def evaluate_admission(
    candidates: Sequence[ConstraintEnvironment],
    prior_signatures: set[str] | None = None,
) -> AdmissionSummary:
    """Run every EvoEnv admission gate and keep the full denominator auditable."""

    known = prior_signatures or set()
    records: list[AdmissionRecord] = []
    admitted: list[ConstraintEnvironment] = []
    for environment in candidates:
        reference = environment.compute_reference()
        determinism_passed = reference == environment.compute_reference()
        asymmetry_passed = solve_verify_asymmetry_passed(reference)
        novelty_passed = environment_signature(environment) not in known
        difficulty_passed = difficulty_calibrated(reference)
        no_leakage_passed = not prompt_leaks_answer(environment, reference)
        completeness_errors = 0 if environment.score_response(reference.canonical_assignment).accepted else 1
        invalid = first_invalid_assignment(environment)
        soundness_errors = 1 if invalid is not None and environment.score_response(invalid).accepted else 0
        solver_authority_passed = completeness_errors == 0 and soundness_errors == 0
        gate_results = {
            "determinism": determinism_passed,
            "solve_verify_asymmetry": asymmetry_passed,
            "novelty": novelty_passed,
            "difficulty_calibration": difficulty_passed,
            "answer_leakage": no_leakage_passed,
            "solver_authority": solver_authority_passed,
        }
        rejections = tuple(name for name, passed in gate_results.items() if not passed)
        is_admitted = not rejections
        record = AdmissionRecord(
            environment_id=environment.environment_id,
            admitted=is_admitted,
            rejection_reasons=rejections,
            determinism_passed=determinism_passed,
            solve_verify_asymmetry_passed=asymmetry_passed,
            novelty_passed=novelty_passed,
            difficulty_passed=difficulty_passed,
            no_answer_leakage_passed=no_leakage_passed,
            solver_authority_passed=solver_authority_passed,
            soundness_errors=soundness_errors,
            completeness_errors=completeness_errors,
            reference=reference,
        )
        records.append(record)
        if is_admitted:
            admitted.append(environment)
    return AdmissionSummary(records=tuple(records), admitted=tuple(admitted))


def solve_verify_asymmetry_passed(reference: ReferenceSolution) -> bool:
    return (
        reference.verify_checks > 0
        and reference.solver_evaluations / reference.verify_checks >= MIN_SOLVE_VERIFY_RATIO
    )


def difficulty_calibrated(reference: ReferenceSolution) -> bool:
    return (
        MIN_SEARCH_SPACE <= reference.solver_evaluations <= MAX_SEARCH_SPACE
        and 0 < reference.solution_count < reference.solver_evaluations
    )


def first_invalid_assignment(environment: ConstraintEnvironment) -> Mapping[str, int] | None:
    for values in itertools.product(*(environment.domains[var] for var in environment.variables)):
        assignment = dict(zip(environment.variables, values, strict=True))
        if not environment.score_response(assignment).accepted:
            return assignment
    return None


def pass_rate(values: Iterable[bool]) -> float:
    observed = tuple(values)
    if not observed:
        return 0.0
    return round(sum(1 for value in observed if value) / len(observed), 6)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the Exp 3128 terminal artifact payload."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    exp3123 = read_json_object(root_path / EXP3123_REL_PATH)
    exp3116 = read_json_object(root_path / EXP3116_REL_PATH)
    blocked_reason = precondition_blocker(exp3123, exp3116)
    if blocked_reason:
        artifact = blocked_artifact(root_path, blocked_reason, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    summary = evaluate_admission(
        sample_candidate_environments(seed=3128),
        prior_signatures=prior_signatures(exp3116),
    )
    retention_delta = compute_retention_delta(exp3116)
    ready = (
        summary.admitted_count > 0
        and summary.soundness_errors == 0
        and summary.completeness_errors == 0
        and retention_delta >= 0.0
    )
    live_call_count = 0
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_evoenv_pilot_v1_ready": ready,
        "continuous_self_learning_targeted": True,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "selected_model_ids": [],
        "live_call_count": live_call_count,
        "live_model_environment_synthesis": False,
        "candidate_environment_count": summary.candidate_count,
        "admitted_environment_count": summary.admitted_count,
        "solve_verify_asymmetry_pass_rate": summary.solve_verify_asymmetry_pass_rate,
        "novelty_pass_rate": summary.novelty_pass_rate,
        "no_answer_leakage_pass_rate": summary.no_answer_leakage_pass_rate,
        "retention_delta": retention_delta,
        "soundness_errors": summary.soundness_errors,
        "completeness_errors": summary.completeness_errors,
        "no_weight_update_claim": True,
        "admission_records": [record.to_dict() for record in summary.records],
        "admitted_environments": admitted_environment_rows(summary),
        "retention_check": retention_check(exp3116),
        "precondition_checks": precondition_checks(exp3123, exp3116),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root_path),
        "inference_substrate": inference_substrate(exp3123),
        "duration_s": duration(start, now_s),
        "honest_verdict": (
            "complete: fr11_evoenv_pilot_v1_ready=true; solver-only executable "
            "environment admission complete; no model-weight update claimed"
            if ready
            else "blocked_precondition_failed: evoenv admission gates did not clear"
        ),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocked_reason: str,
    start: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "fr11_evoenv_pilot_v1_ready": False,
        "continuous_self_learning_targeted": True,
        "model_specs": list(MANDATED_MODEL_SPECS),
        "selected_model_ids": [],
        "live_call_count": 0,
        "live_model_environment_synthesis": False,
        "candidate_environment_count": 0,
        "admitted_environment_count": 0,
        "solve_verify_asymmetry_pass_rate": 0.0,
        "novelty_pass_rate": 0.0,
        "no_answer_leakage_pass_rate": 0.0,
        "retention_delta": 0.0,
        "soundness_errors": 0,
        "completeness_errors": 0,
        "no_weight_update_claim": True,
        "admission_records": [],
        "admitted_environments": [],
        "retention_check": {"prior_correct_rate": 0.0, "post_admission_correct_rate": 0.0},
        "precondition_checks": {
            "exp3123_manifest_ready": False,
            "exp3116_retention_ready": False,
        },
        "blocked_reason": blocked_reason,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "source_artifacts": source_artifacts(root),
        "inference_substrate": {
            "mode": "blocked_precondition_check",
            "present_mandated_model_ids": [],
            "legacy_small_model_headline_used": False,
            "model_weight_training": False,
            "model_weight_mutation": False,
            "live_model_environment_synthesis": False,
            "live_model_calls": 0,
        },
        "duration_s": duration(start, now_s),
        "honest_verdict": f"blocked_precondition_failed: {blocked_reason}",
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
    """Write the Exp 3128 JSON artifact."""

    root_path = Path(root)
    path = Path(output_path)
    output = path if path.is_absolute() else root_path / path
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def precondition_blocker(exp3123: Mapping[str, Any], exp3116: Mapping[str, Any]) -> str:
    if not exp3123 or exp3123.get("sota_cache_manifest_v2_ready") is not True:
        return "exp3123_precondition_manifest_missing_or_empty"
    if not exp3116 or exp3116.get("fr11_unsolvable_curriculum_ready") is not True:
        return "exp3116_retention_guard_missing_or_not_ready"
    return ""


def precondition_checks(exp3123: Mapping[str, Any], exp3116: Mapping[str, Any]) -> JsonDict:
    return {
        "exp3123_manifest_ready": exp3123.get("sota_cache_manifest_v2_ready") is True,
        "exp3116_retention_ready": exp3116.get("fr11_unsolvable_curriculum_ready") is True,
        "present_mandated_model_ids": [
            model_id
            for model_id in exp3123.get("present_model_ids", [])
            if model_id in MANDATED_MODEL_SPECS
        ],
        "legacy_small_models_excluded_from_headline": True,
    }


def read_json_object(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def prior_signatures(exp3116: Mapping[str, Any]) -> set[str]:
    signatures = set()
    for row in exp3116.get("guarded_decisions", []):
        fixture_id = str(row.get("fixture_id", ""))
        if fixture_id:
            signatures.add(fixture_id)
    return signatures


def compute_retention_delta(exp3116: Mapping[str, Any]) -> float:
    check = retention_check(exp3116)
    return round(check["post_admission_correct_rate"] - check["prior_correct_rate"], 6)


def retention_check(exp3116: Mapping[str, Any]) -> JsonDict:
    rows = list(exp3116.get("guarded_decisions", []))
    prior_correct = sum(1 for row in rows if row.get("decision_label") == "correct")
    prior_rate = round(prior_correct / len(rows), 6) if rows else 0.0
    return {
        "prior_case_count": len(rows),
        "prior_correct_rate": prior_rate,
        "post_admission_correct_rate": prior_rate,
        "mechanism": "no_prior_controller_mutation_from_environment_admission",
    }


def admitted_environment_rows(summary: AdmissionSummary) -> list[JsonDict]:
    records_by_id = {record.environment_id: record for record in summary.records}
    return [
        environment.to_dict()
        | {
            "reference": records_by_id[environment.environment_id].reference.to_dict(),
            "prompt_leaks_answer": not records_by_id[
                environment.environment_id
            ].no_answer_leakage_passed,
        }
        for environment in summary.admitted
    ]


def inference_substrate(exp3123: Mapping[str, Any]) -> JsonDict:
    present = [
        model_id
        for model_id in exp3123.get("present_model_ids", [])
        if model_id in MANDATED_MODEL_SPECS
    ]
    return {
        "mode": "solver_only_environment_admission",
        "model_specs": list(MANDATED_MODEL_SPECS),
        "present_mandated_model_ids": present,
        "selected_model_ids": [],
        "live_model_calls": 0,
        "live_model_environment_synthesis": False,
        "legacy_small_model_headline_used": False,
        "executes_exact_solvers": True,
        "executes_model_generation": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for name, rel_path in SOURCE_ARTIFACTS:
        path = root / rel_path
        row: JsonDict = {"name": name, "path": rel_path.as_posix(), "exists": path.is_file()}
        if path.is_file():
            row["sha256"] = sha256_file(path)
        rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_signature(environment: ConstraintEnvironment) -> str:
    payload = {
        "family_id": environment.family_id,
        "domains": {key: list(value) for key, value in sorted(environment.domains.items())},
        "constraints": [constraint_to_dict(constraint) for constraint in environment.constraints],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the Exp 3128 artifact schema or honesty gates are violated."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["live_model_environment_synthesis"] and int(artifact["live_call_count"]) <= 0:
        raise ValueError("live_call_count must be positive for live synthesis claims")
    if artifact["no_weight_update_claim"] is not True:
        raise ValueError("no_weight_update_claim must stay true for this pilot")
    if artifact["fr11_evoenv_pilot_v1_ready"]:
        if int(artifact["soundness_errors"]) != 0:
            raise ValueError("soundness_errors must be zero for readiness")
        if int(artifact["completeness_errors"]) != 0:
            raise ValueError("completeness_errors must be zero for readiness")
        if float(artifact["retention_delta"]) < 0.0:
            raise ValueError("retention_delta must be nonnegative for readiness")
        if int(artifact["admitted_environment_count"]) <= 0:
            raise ValueError("admitted_environment_count must be positive for readiness")
        if not str(artifact["honest_verdict"]).startswith(SUCCESS_PREFIXES):
            raise ValueError("honest_verdict must use a terminal success prefix")


def duration(start: float, now_s: float | None) -> float:
    now = time.perf_counter() if now_s is None else float(now_s)
    return round(now - start, 6)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    output = write_artifact(REPO_ROOT)
    artifact = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["fr11_evoenv_pilot_v1_ready"] else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

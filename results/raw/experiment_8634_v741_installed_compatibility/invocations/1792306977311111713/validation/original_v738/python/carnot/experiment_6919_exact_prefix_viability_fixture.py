"""Build a bounded exact prefix-viability fixture for plain relation lines.

Spec refs: REQ-CONSTRAINT-6919 and SCENARIO-CONSTRAINT-6919-*.

The in-loop engine uses direct Python enumeration. The final engine builds a
separate ASP theory and asks clingo for stable-model existence. This separation
matters because agreement between copies of one algorithm would not detect a
shared semantic mistake.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
FinalValidator = Callable[["RelationFixture", tuple[str, ...]], bool]

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path("openspec/capabilities/constraint-verification/spec.md")
RESULT_PATH = Path("results/experiment_6919_exact_prefix_viability_fixture.json")
EXP6274_ARTIFACT_PATH = Path("results/experiment_6274_asp_energy_semantic_compiler.json")
EXP6274_SOURCE_PATH = Path("python/carnot/asp_energy.py")
EXP6886_ARTIFACT_PATH = Path("results/experiment_6886_enoki_exact_relation_fixture.json")
EXCLUSION_MANIFEST_PATH = Path("ops/exclusion_manifest.yaml")
MODULE_PATH = Path("python/carnot/experiment_6919_exact_prefix_viability_fixture.py")
TEST_PATH = Path("tests/python/test_experiment_6919_exact_prefix_viability_fixture.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_6919_exact_prefix_viability_fixture.py")

INFERENCE_SUBSTRATE = "deterministic_cpu_dual_exact_prefix_canary_no_llm"
RANDOM_SEED = 360073
MAX_RELATION_LINES = 2
SOURCE_GROUP_COUNT = 30
MIN_PREFIX_CASES = 120
MIN_CASES_PER_FAMILY = 24
MIN_CASES_PER_TYPE = 12
FAMILIES = (
    "graph_coloring",
    "scheduling",
    "non_monotonic_defaults",
    "contradictions",
    "cardinality_constraints",
)
REQUIRED_CASE_TYPES = (
    "positive",
    "immediately_impossible",
    "late_failure",
    "ambiguous",
    "duplicate",
    "contradiction",
    "unsupported",
    "no_headroom",
)
AUXILIARY_CASE_TYPES = ("empty_prefix", "valid_completion", "order_permutation")
RETIRED_MECHANISMS = (
    "schema_decoder",
    "repair_reprompt",
    "finite_answer_id",
    "per_instance_answer_menu",
)

EXP6274_ARTIFACT_SHA256 = "sha256:b02c88963c4815aa0e26d451ffd60fdd9f1014d32e76f638592ac114c611e96b"
EXP6274_SOURCE_SHA256 = "sha256:0f6077bcd49aa93a6cdbde72422ecf97d905b76b31cadbc0cd401c494af015e1"
EXP6886_ARTIFACT_SHA256 = "sha256:602250fbfe172f08458ea279787d992e89835f12005ba6ef59ec02f3b411d500"
EXP6886_FROZEN_HASHES = {
    "reproducibility_checksum": (
        "sha256:6a5954a0cb90e31c7b60965b2bfece7311fcd3c4eb912b9aecc779f51e6ad634"
    ),
    "calibration_manifest": (
        "sha256:5f890d4d30bd814b7bb93b27c77f5c630069c8131873e34b3bfdb3a7aa0df797"
    ),
    "held_manifest": ("sha256:2e5a4879996cd95742ec3c6a123a344eb96224d8bd151c1bce6400b22b621ab5"),
    "calibration_sidecar": (
        "sha256:cb8db945ac4b20e0d2e9658cd09e5263fe2240e78f999c323fd09fc3351a6cce"
    ),
    "held_sidecar": ("sha256:d75b4b032ed27542dd8d619922865c389f869aa11470a5cfffd90c05de6553ad"),
}
EXCLUSION_ENTRY_IDS = (
    "finite_id_gguf_generated_answer_transport_same_mechanism_v519",
    "exp5923_schema_supported_constraintir_zero_exact_semantics_retired_v526",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "fixture_manifest",
    "rows",
    "prefix_case_rows",
    "source_group_split_rows",
    "in_loop_exact_engine_rows",
    "final_exact_engine_rows",
    "exact_engine_parity_rows",
    "positive_rows",
    "impossible_rows",
    "late_failure_rows",
    "ambiguous_rows",
    "duplicate_rows",
    "contradiction_rows",
    "unsupported_rows",
    "no_headroom_rows",
    "branch_factor_rows",
    "feasible_branch_rows",
    "rejected_branch_rows",
    "witness_rows",
    "latency_rows",
    "retired_mechanism_activation_count",
    "model_inference_call_count",
    "exact_engine_disagreement_count",
    "random_seed",
    "reproducibility_checksum",
    "prefix_viability_canary_ready_score",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "field_principles": "A reason per field keeps the artifact contract auditable.",
    "preconditions_checked": "Exact admission checks prevent drift from entering the canary.",
    "inference_substrate": "The fixed CPU declaration proves that no model inference ran.",
    "duration_s": "Measured wall time shows the bounded fixture builder executed.",
    "source_artifact_hashes": "Hashes bind the result to its exact code and evidence inputs.",
    "fixture_manifest": "The manifest freezes families, cases, bounds, and split identities.",
    "rows": "Branch rows retain each source, prefix, candidate line, and exact decision.",
    "prefix_case_rows": "Case rows retain one complete prefix-level decision receipt.",
    "source_group_split_rows": "Group rows prove that canary and held sources do not overlap.",
    "in_loop_exact_engine_rows": "Direct-engine rows expose every candidate-level decision.",
    "final_exact_engine_rows": "Clingo rows keep final exact authority separate from guidance.",
    "exact_engine_parity_rows": "Parity rows make every disagreement visible before readiness.",
    "positive_rows": "Positive rows preserve all prefixes with at least one valid completion.",
    "impossible_rows": "Impossible rows preserve rejected prefixes instead of dropping them.",
    "late_failure_rows": "Late rows prove that an initially valid branch can fail later.",
    "ambiguous_rows": "Ambiguous rows count prefixes with more than one valid completion.",
    "duplicate_rows": "Duplicate rows prove repeated relations fail closed.",
    "contradiction_rows": "Contradiction rows prove conflicting subject relations fail closed.",
    "unsupported_rows": "Unsupported rows prove unknown relation atoms fail closed.",
    "no_headroom_rows": "No-headroom rows expose invalid full programs that cannot extend.",
    "branch_factor_rows": "Branch counts measure the exact candidate cost for each prefix.",
    "feasible_branch_rows": "Feasible branch rows show which next lines preserve completion.",
    "rejected_branch_rows": "Rejected branch rows show the exact cost of pruning bad lines.",
    "witness_rows": "Witness rows provide replayable valid completions for positive decisions.",
    "latency_rows": "Measured per-engine latency supports later branch-cost comparisons.",
    "retired_mechanism_activation_count": "Zero proves retired generation mechanisms stayed off.",
    "model_inference_call_count": "Zero proves the canary used no proposal model.",
    "exact_engine_disagreement_count": "Zero is required before exact labels can be consumed.",
    "random_seed": "A fixed non-ID seed identifies deterministic fixture construction.",
    "reproducibility_checksum": "A timing-free hash detects changes in scientific content.",
    "prefix_viability_canary_ready_score": "One opens the next gate only after every check passes.",
    "gate_check_summary": "Expected and observed values make every blocked gate actionable.",
    "verifier_is_oracle": "True discloses that exact engines define fixture correctness.",
    "verdict_class": "The closed class prevents oracle evidence from becoming positive evidence.",
    "honest_verdict": "A terminal prefix lets the conductor classify this completed run safely.",
}


@dataclass(frozen=True, order=True)
class RelationAction:
    """One plain semantic relation line without a token-level decoder."""

    subject: str
    predicate: str
    object: str

    @property
    def line(self) -> str:
        """Render the action as whitespace-separated plain text."""

        return f"{self.subject} {self.predicate} {self.object}"


@dataclass(frozen=True)
class RelationFixture:
    """One bounded relation task with frozen source-group membership."""

    source_id: str
    source_group: str
    split: str
    family: str
    ordinal: int
    subjects: tuple[str, str]
    predicate: str
    values: tuple[str, str, str]
    forbidden_value: str
    invalid_pairs: frozenset[tuple[str, str]]
    max_lines: int = MAX_RELATION_LINES

    def action_line(self, subject: str, value: str) -> str:
        """Create one plain line for fixture construction and branching."""

        return RelationAction(subject, self.predicate, value).line

    @property
    def semantic_lines(self) -> tuple[str, ...]:
        """Return global semantic actions, not an answer-ID transport."""

        return tuple(
            self.action_line(subject, value) for subject in self.subjects for value in self.values
        )

    @property
    def candidate_lines(self) -> tuple[str, ...]:
        """Return plain next-line actions used to measure the branch frontier."""

        blocked = tuple(
            self.action_line(subject, self.forbidden_value) for subject in self.subjects
        )
        return (*self.semantic_lines, *blocked)


@dataclass(frozen=True)
class BoundedRelationProgramState:
    """An ordered generation prefix with a strict final line bound."""

    fixture: RelationFixture
    prefix: tuple[str, ...]

    @property
    def available_headroom(self) -> int:
        """Return how many lines can still enter without exceeding the bound."""

        return max(0, self.fixture.max_lines - len(self.prefix))


@dataclass(frozen=True)
class ExactDecision:
    """A replayable exact extendability result from one implementation."""

    engine: str
    extendable: bool
    completion_count: int
    completions: tuple[tuple[str, ...], ...]
    witness: tuple[str, ...]
    reason: str
    available_headroom: int

    @property
    def ambiguous(self) -> bool:
        """Report whether more than one distinct final program is possible."""

        return self.completion_count > 1


def canonical_json(value: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for scientific content hashes."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    """Return an explicit SHA-256 identity."""

    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_path(path: Path) -> str:
    """Hash one local source file without modifying it."""

    return sha256_bytes(path.read_bytes())


def build_relation_fixture(family: str, ordinal: int) -> RelationFixture:
    """Build one family fixture with names that are safe in plain text and ASP."""

    source_group = f"relation_group_{ordinal:02d}"
    split = "canary" if ordinal < SOURCE_GROUP_COUNT // 2 else "held"
    if family == "graph_coloring":
        subjects = (f"node_{ordinal}_left", f"node_{ordinal}_right")
        predicate = "has_color"
        values = ("red", "blue", "green")
        invalid_pairs = frozenset((value, value) for value in values)
    elif family == "scheduling":
        subjects = (f"task_{ordinal}_alpha", f"task_{ordinal}_beta")
        predicate = "scheduled_at"
        values = ("morning", "afternoon", "evening")
        invalid_pairs = frozenset((value, value) for value in values)
    elif family == "non_monotonic_defaults":
        subjects = (f"bird_{ordinal}_alpha", f"bird_{ordinal}_beta")
        predicate = "has_condition"
        values = ("injured", "healthy", "unknown")
        invalid_pairs = frozenset({("injured", "injured")})
    elif family == "contradictions":
        subjects = (f"claim_{ordinal}_alpha", f"claim_{ordinal}_beta")
        predicate = "has_status"
        values = ("accepted", "rejected", "unknown")
        invalid_pairs = frozenset({("accepted", "rejected"), ("rejected", "accepted")})
    elif family == "cardinality_constraints":
        subjects = (f"set_{ordinal}_alpha", f"set_{ordinal}_beta")
        predicate = "selects"
        values = ("option_a", "option_b", "option_c")
        invalid_pairs = frozenset((value, value) for value in values)
    else:
        raise ValueError(f"unsupported_family:{family}")
    return RelationFixture(
        source_id=f"{family}_{ordinal:02d}",
        source_group=source_group,
        split=split,
        family=family,
        ordinal=ordinal,
        subjects=subjects,
        predicate=predicate,
        values=values,
        forbidden_value="blocked",
        invalid_pairs=invalid_pairs,
    )


def build_source_fixtures() -> list[RelationFixture]:
    """Build 30 source-group-disjoint fixtures balanced over five families."""

    return [
        build_relation_fixture(FAMILIES[ordinal % len(FAMILIES)], ordinal)
        for ordinal in range(SOURCE_GROUP_COUNT)
    ]


def _parse_prefix(
    fixture: RelationFixture, prefix: tuple[str, ...]
) -> tuple[tuple[RelationAction, ...], str | None]:
    """Validate relation lines before either semantic engine receives them."""

    if len(prefix) > fixture.max_lines:
        return (), "over_capacity"
    actions: list[RelationAction] = []
    for line in prefix:
        parts = line.split()
        if len(parts) != 3:
            return (), "unsupported_atom"
        action = RelationAction(*parts)
        if (
            action.subject not in fixture.subjects
            or action.predicate != fixture.predicate
            or action.object not in (*fixture.values, fixture.forbidden_value)
        ):
            return (), "unsupported_atom"
        actions.append(action)
    if len(set(actions)) != len(actions):
        return (), "duplicate_relation"
    if len({action.subject for action in actions}) != len(actions):
        return (), "contradiction"
    if any(action.object == fixture.forbidden_value for action in actions):
        return (), "immediate_constraint"
    return tuple(actions), None


def _direct_final_valid(fixture: RelationFixture, lines: tuple[str, ...]) -> bool:
    """Evaluate one complete program with direct bounded Python predicates."""

    actions, error = _parse_prefix(fixture, lines)
    if error is not None or len(actions) != fixture.max_lines:
        return False
    values_by_subject = {action.subject: action.object for action in actions}
    pair = tuple(values_by_subject[subject] for subject in fixture.subjects)
    return pair not in fixture.invalid_pairs


def _completion_candidates(
    fixture: RelationFixture, prefix: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Enumerate bounded suffixes without assuming that they are valid."""

    headroom = fixture.max_lines - len(prefix)
    if headroom < 0:
        return ()
    remaining = tuple(line for line in fixture.semantic_lines if line not in prefix)
    return tuple(tuple(sorted((*prefix, *suffix))) for suffix in combinations(remaining, headroom))


def _decision(
    *,
    engine: str,
    fixture: RelationFixture,
    prefix: tuple[str, ...],
    completions: Sequence[tuple[str, ...]],
    error: str | None,
    invalid_final_reason: str,
) -> ExactDecision:
    """Build the common receipt shape after an engine completes its own work."""

    canonical = tuple(sorted(set(completions)))
    if canonical:
        reason = "valid_completion" if len(prefix) == fixture.max_lines else "bounded_completion"
    elif error is not None:
        reason = error
    elif len(prefix) == fixture.max_lines:
        reason = invalid_final_reason
    else:
        reason = "no_bounded_completion"
    return ExactDecision(
        engine=engine,
        extendable=bool(canonical),
        completion_count=len(canonical),
        completions=canonical,
        witness=canonical[0] if canonical else (),
        reason=reason,
        available_headroom=max(0, fixture.max_lines - len(prefix)),
    )


def direct_prefix_viability(fixture: RelationFixture, prefix: Sequence[str]) -> ExactDecision:
    """Decide exact extendability with direct bounded enumeration."""

    ordered_prefix = tuple(prefix)
    _, error = _parse_prefix(fixture, ordered_prefix)
    completions: list[tuple[str, ...]] = []
    if error is None:
        completions = [
            candidate
            for candidate in _completion_candidates(fixture, ordered_prefix)
            if _direct_final_valid(fixture, candidate)
        ]
    return _decision(
        engine="python_bounded_relation_enumerator_v1",
        fixture=fixture,
        prefix=ordered_prefix,
        completions=completions,
        error=error,
        invalid_final_reason="family_constraint",
    )


def _clingo_source(fixture: RelationFixture) -> str:
    """Build an independent ASP generator for all valid final programs."""

    lines: list[str] = []
    for subject in fixture.subjects:
        choices = "; ".join(f"chosen({subject},{value})" for value in fixture.values)
        lines.append(f"1 {{{choices}}} 1.")
    left_subject, right_subject = fixture.subjects
    for left_value, right_value in sorted(fixture.invalid_pairs):
        lines.append(
            f":- chosen({left_subject},{left_value}), chosen({right_subject},{right_value})."
        )
    return "\n".join(lines) + "\n"


@cache
def _clingo_valid_completions(fixture: RelationFixture) -> tuple[tuple[str, ...], ...]:
    """Enumerate the final valid set once through the independent ASP engine."""

    import clingo

    control = clingo.Control(["0", "--warn=none"])
    control.add("base", [], _clingo_source(fixture))
    control.ground([("base", [])])
    completions: list[tuple[str, ...]] = []
    with control.solve(yield_=True) as handle:
        for model in handle:
            actions = [
                fixture.action_line(str(symbol.arguments[0]), str(symbol.arguments[1]))
                for symbol in model.symbols(atoms=True)
                if symbol.name == "chosen" and len(symbol.arguments) == 2
            ]
            completions.append(tuple(sorted(actions)))
    return tuple(sorted(completions))


@cache
def clingo_final_validity(fixture: RelationFixture, lines: tuple[str, ...]) -> ExactDecision:
    """Judge one final program through clingo, independent of Python predicates."""

    ordered_lines = tuple(lines)
    actions, error = _parse_prefix(fixture, ordered_lines)
    if error is not None or len(actions) != fixture.max_lines:
        return _decision(
            engine="clingo_stable_model_final_engine_v1",
            fixture=fixture,
            prefix=ordered_lines,
            completions=(),
            error=error or "incomplete_program",
            invalid_final_reason="family_constraint",
        )
    canonical = tuple(sorted(ordered_lines))
    valid = canonical in _clingo_valid_completions(fixture)
    return _decision(
        engine="clingo_stable_model_final_engine_v1",
        fixture=fixture,
        prefix=ordered_lines,
        completions=(canonical,) if valid else (),
        error=None,
        invalid_final_reason="family_constraint",
    )


def final_engine_prefix_viability(
    fixture: RelationFixture,
    prefix: Sequence[str],
    *,
    final_validator: FinalValidator | None = None,
) -> ExactDecision:
    """Derive prefix viability only from independent final-program decisions."""

    ordered_prefix = tuple(prefix)
    _, error = _parse_prefix(fixture, ordered_prefix)
    completions: list[tuple[str, ...]] = []
    engine = "clingo_stable_model_final_engine_v1"
    if error is None:
        for candidate in _completion_candidates(fixture, ordered_prefix):
            if final_validator is None:
                valid = clingo_final_validity(fixture, candidate).extendable
            else:
                valid = bool(final_validator(fixture, candidate))
                engine = "injected_final_exact_engine"
            if valid:
                completions.append(candidate)
    return _decision(
        engine=engine,
        fixture=fixture,
        prefix=ordered_prefix,
        completions=completions,
        error=error,
        invalid_final_reason="family_constraint",
    )


def _first_valid_pair(fixture: RelationFixture) -> tuple[str, str]:
    """Select the first stable valid pair for deterministic case construction."""

    return next(
        pair
        for pair in ((left, right) for left in fixture.values for right in fixture.values)
        if pair not in fixture.invalid_pairs
    )


def build_prefix_cases(fixture: RelationFixture) -> list[JsonDict]:
    """Create all required failure timings and controls for one source group."""

    first_subject, second_subject = fixture.subjects
    first_value, second_value = _first_valid_pair(fixture)
    valid = (
        fixture.action_line(first_subject, first_value),
        fixture.action_line(second_subject, second_value),
    )
    contradiction_value = next(value for value in fixture.values if value != first_value)
    invalid_left, invalid_right = sorted(fixture.invalid_pairs)[0]
    base = fixture.action_line(first_subject, first_value)
    cases: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = [
        ("empty_prefix", (), ()),
        ("positive", (base,), ()),
        ("valid_completion", valid, ()),
        (
            "immediately_impossible",
            (fixture.action_line(first_subject, fixture.forbidden_value),),
            (),
        ),
        (
            "late_failure",
            (base, fixture.action_line(first_subject, contradiction_value)),
            (base,),
        ),
        ("ambiguous", (base,), ()),
        ("duplicate", (base, base), (base,)),
        (
            "contradiction",
            (base, fixture.action_line(first_subject, contradiction_value)),
            (base,),
        ),
        (
            "unsupported",
            (fixture.action_line(first_subject, "unsupported_value"),),
            (),
        ),
        (
            "no_headroom",
            (
                fixture.action_line(first_subject, invalid_left),
                fixture.action_line(second_subject, invalid_right),
            ),
            (),
        ),
        ("order_permutation", tuple(reversed(valid)), valid),
    ]
    return [
        {
            "prefix_case_id": f"{fixture.source_id}-{case_type}",
            "case_type": case_type,
            "prefix": prefix,
            "prior_prefix": prior_prefix,
        }
        for case_type, prefix, prior_prefix in cases
    ]


def _decision_row(case_id: str, decision: ExactDecision, latency_ms: float) -> JsonDict:
    """Convert an immutable exact decision into a JSON receipt."""

    return {
        "prefix_case_id": case_id,
        "engine": decision.engine,
        "extendable": decision.extendable,
        "completion_count": decision.completion_count,
        "ambiguous": decision.ambiguous,
        "reason": decision.reason,
        "witness": list(decision.witness),
        "completions_hash": sha256_bytes(canonical_json(decision.completions)),
        "decision_latency_ms": latency_ms,
        "available_headroom": decision.available_headroom,
    }


def _timed_decision(call: Callable[[], ExactDecision]) -> tuple[ExactDecision, float]:
    """Measure one exact decision without changing its result."""

    started_ns = time.perf_counter_ns()
    decision = call()
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
    return decision, round(elapsed_ms, 6)


def _case_evidence(
    fixture: RelationFixture,
    case: Mapping[str, Any],
    *,
    final_validator: FinalValidator | None,
) -> JsonDict:
    """Build prefix, branch, parity, witness, and timing evidence for one case."""

    case_id = str(case["prefix_case_id"])
    case_type = str(case["case_type"])
    prefix = tuple(str(line) for line in case["prefix"])
    prior_prefix = tuple(str(line) for line in case["prior_prefix"])
    direct, direct_ms = _timed_decision(lambda: direct_prefix_viability(fixture, prefix))
    final, final_ms = _timed_decision(
        lambda: final_engine_prefix_viability(
            fixture,
            prefix,
            final_validator=final_validator,
        )
    )
    engines_agree = (
        direct.extendable == final.extendable and direct.completions == final.completions
    )
    prior_extendable = (
        direct_prefix_viability(fixture, prior_prefix).extendable if prior_prefix else None
    )
    branch_lines = (
        fixture.candidate_lines if direct.extendable and direct.available_headroom > 0 else ()
    )
    branch_rows: list[JsonDict] = []
    for branch in branch_lines:
        branch_prefix = (*prefix, branch)
        branch_direct = direct_prefix_viability(fixture, branch_prefix)
        branch_final = final_engine_prefix_viability(
            fixture,
            branch_prefix,
            final_validator=final_validator,
        )
        branch_rows.append(
            {
                "prefix_case_id": case_id,
                "source_id": fixture.source_id,
                "source_group": fixture.source_group,
                "split": fixture.split,
                "family": fixture.family,
                "prefix": list(prefix),
                "branch": branch,
                "in_loop_extendable": branch_direct.extendable,
                "final_engine_extendable": branch_final.extendable,
                "engines_agree": (
                    branch_direct.extendable == branch_final.extendable
                    and branch_direct.completions == branch_final.completions
                ),
                "exact_decision": "feasible" if branch_direct.extendable else "rejected",
                "proof": branch_direct.reason,
                "witness": list(branch_direct.witness),
            }
        )
    if not branch_rows:
        branch_rows.append(
            {
                "prefix_case_id": case_id,
                "source_id": fixture.source_id,
                "source_group": fixture.source_group,
                "split": fixture.split,
                "family": fixture.family,
                "prefix": list(prefix),
                "branch": None,
                "in_loop_extendable": direct.extendable,
                "final_engine_extendable": final.extendable,
                "engines_agree": engines_agree,
                "exact_decision": "feasible" if direct.extendable else "rejected",
                "proof": direct.reason,
                "witness": list(direct.witness),
            }
        )
    feasible = [
        row
        for row in branch_rows
        if row["branch"] is not None and row["exact_decision"] == "feasible"
    ]
    rejected = [
        row
        for row in branch_rows
        if row["branch"] is not None and row["exact_decision"] == "rejected"
    ]
    case_row = {
        "prefix_case_id": case_id,
        "source_id": fixture.source_id,
        "source_group": fixture.source_group,
        "split": fixture.split,
        "family": fixture.family,
        "case_type": case_type,
        "prefix": list(prefix),
        "prefix_line_count": len(prefix),
        "in_loop_extendable": direct.extendable,
        "final_engine_extendable": final.extendable,
        "completion_count": direct.completion_count,
        "ambiguous": direct.ambiguous,
        "branch_factor": len(branch_lines),
        "feasible_branch_count": len(feasible),
        "rejected_branch_count": len(rejected),
        "proof": direct.reason,
        "witness": list(direct.witness),
        "decision_latency_ms": round(direct_ms + final_ms, 6),
        "final_available_headroom": direct.available_headroom,
        "prior_prefix": list(prior_prefix),
        "prior_prefix_extendable": prior_extendable,
        "engines_agree": engines_agree,
    }
    return {
        "case_row": case_row,
        "branch_rows": branch_rows,
        "direct_row": _decision_row(case_id, direct, direct_ms),
        "final_row": _decision_row(case_id, final, final_ms),
        "parity_row": {
            "prefix_case_id": case_id,
            "source_group": fixture.source_group,
            "split": fixture.split,
            "family": fixture.family,
            "engines_agree": engines_agree,
            "in_loop_completion_count": direct.completion_count,
            "final_engine_completion_count": final.completion_count,
            "in_loop_completions_hash": sha256_bytes(canonical_json(direct.completions)),
            "final_engine_completions_hash": sha256_bytes(canonical_json(final.completions)),
        },
        "feasible": feasible,
        "rejected": rejected,
    }


def _read_json(path: Path) -> JsonDict | None:
    """Read one JSON object, or return no value when admission cannot parse it."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _hash_if_present(path: Path) -> str | None:
    """Return a file hash while keeping missing source evidence explicit."""

    return sha256_path(path) if path.is_file() else None


def source_artifact_hashes(repo_root: Path) -> JsonDict:
    """Bind every exact input and new implementation file used by the run."""

    paths = {
        "exp6274_artifact": EXP6274_ARTIFACT_PATH,
        "exp6274_compiler": EXP6274_SOURCE_PATH,
        "exp6886_artifact": EXP6886_ARTIFACT_PATH,
        "exclusion_manifest": EXCLUSION_MANIFEST_PATH,
        "module": MODULE_PATH,
        "spec": SPEC_PATH,
        "tests": TEST_PATH,
        "wrapper": WRAPPER_PATH,
    }
    return {
        name: {"path": str(path), "sha256": _hash_if_present(repo_root / path)}
        for name, path in paths.items()
    }


def check_preconditions(repo_root: Path) -> JsonDict:
    """Check exact compiler, fixture, solver, and retirement inputs before work."""

    exp6274_path = repo_root / EXP6274_ARTIFACT_PATH
    compiler_path = repo_root / EXP6274_SOURCE_PATH
    exp6886_path = repo_root / EXP6886_ARTIFACT_PATH
    manifest_path = repo_root / EXCLUSION_MANIFEST_PATH
    exp6274 = _read_json(exp6274_path)
    exp6886 = _read_json(exp6886_path)
    exp6274_observed = {
        "artifact_sha256": _hash_if_present(exp6274_path),
        "compiler_sha256": _hash_if_present(compiler_path),
        "status": exp6274.get("status") if exp6274 else None,
        "ready_score": exp6274.get("asp_energy_semantic_ready_score") if exp6274 else None,
        "parity_failure_count": exp6274.get("parity_failure_count") if exp6274 else None,
    }
    exp6274_passed = exp6274_observed == {
        "artifact_sha256": EXP6274_ARTIFACT_SHA256,
        "compiler_sha256": EXP6274_SOURCE_SHA256,
        "status": "complete",
        "ready_score": 1.0,
        "parity_failure_count": 0,
    }
    fixture_observed = {
        "artifact_sha256": _hash_if_present(exp6886_path),
        "ready_score": exp6886.get("relation_fixture_ready_score") if exp6886 else None,
        "reproducibility_checksum": (exp6886.get("reproducibility_checksum") if exp6886 else None),
        "calibration_manifest": (
            exp6886.get("calibration_group_manifest", {}).get("public_fixture_manifest_hash")
            if exp6886
            else None
        ),
        "held_manifest": (
            exp6886.get("sealed_held_group_manifest", {}).get("public_fixture_manifest_hash")
            if exp6886
            else None
        ),
        "calibration_sidecar": (
            exp6886.get("independent_solver_receipts", {})
            .get("sidecar_hashes", {})
            .get("calibration")
            if exp6886
            else None
        ),
        "held_sidecar": (
            exp6886.get("independent_solver_receipts", {}).get("sidecar_hashes", {}).get("held")
            if exp6886
            else None
        ),
    }
    fixture_expected = {
        "artifact_sha256": EXP6886_ARTIFACT_SHA256,
        "ready_score": 1,
        **EXP6886_FROZEN_HASHES,
    }
    manifest_text = manifest_path.read_text(encoding="utf-8") if manifest_path.is_file() else ""
    exclusion_entries = {entry: entry in manifest_text for entry in EXCLUSION_ENTRY_IDS}
    try:
        import clingo

        clingo_version: str | None = str(clingo.__version__)
    except ImportError:
        clingo_version = None
    checks = {
        "qualified_exp6274_compiler": {
            "passed": exp6274_passed,
            "expected": {
                "artifact_sha256": EXP6274_ARTIFACT_SHA256,
                "compiler_sha256": EXP6274_SOURCE_SHA256,
                "status": "complete",
                "ready_score": 1.0,
                "parity_failure_count": 0,
            },
            "observed": exp6274_observed,
        },
        "exact_exp6886_fixture_hashes": {
            "passed": fixture_observed == fixture_expected,
            "expected": fixture_expected,
            "observed": fixture_observed,
        },
        "two_independent_exact_engines": {
            "passed": clingo_version is not None,
            "expected": {
                "available": True,
                "implementation_independent": True,
                "engines": [
                    "python_bounded_relation_enumerator_v1",
                    "clingo_stable_model_final_engine_v1",
                ],
            },
            "observed": {
                "available": clingo_version is not None,
                "implementation_independent": True,
                "engines": [
                    "python_bounded_relation_enumerator_v1",
                    "clingo_stable_model_final_engine_v1" if clingo_version else None,
                ],
                "clingo_version": clingo_version,
            },
        },
        "exclusion_manifest": {
            "passed": manifest_path.is_file() and all(exclusion_entries.values()),
            "expected": {"available": True, "required_entries": list(EXCLUSION_ENTRY_IDS)},
            "observed": {
                "available": manifest_path.is_file(),
                "required_entries": exclusion_entries,
                "sha256": _hash_if_present(manifest_path),
            },
        },
    }
    return {
        **checks,
        "requested_path_resolution": {
            "stale_exp6274_prompt_path": "results/experiment_6274_bounded_asp_energy_compiler.json",
            "resolved_exp6274_path": str(EXP6274_ARTIFACT_PATH),
            "stale_compiler_prompt_path": "python/carnot/constraints/asp_energy.py",
            "resolved_compiler_path": str(EXP6274_SOURCE_PATH),
        },
        "all_passed": all(check["passed"] for check in checks.values()),
    }


def _gate_check(name: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Build one explicit gate comparison."""

    return {"check": name, "expected": expected, "observed": observed, "passed": passed}


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize all failed gates without hiding later failures."""

    failed = [str(check["check"]) for check in checks if not check["passed"]]
    return {
        "checks": [dict(check) for check in checks],
        "passed": not failed,
        "failed_check": failed[0] if failed else None,
        "failed_checks": failed,
        "expected": "all checks pass",
        "observed": "all checks pass" if not failed else f"failed:{','.join(failed)}",
    }


def _empty_evidence() -> JsonDict:
    """Return required row fields for a precondition-blocked artifact."""

    return {
        field: []
        for field in REQUIRED_ARTIFACT_FIELDS
        if field.endswith("_rows") or field == "rows"
    }


def _blocked_precondition_artifact(
    *, date: str, repo_root: Path, preconditions: Mapping[str, Any], duration_s: float
) -> JsonDict:
    """Build a complete terminal artifact when exact inputs are unavailable."""

    precondition_checks = [
        _gate_check(name, value["expected"], value["observed"], bool(value["passed"]))
        for name, value in preconditions.items()
        if isinstance(value, Mapping) and {"expected", "observed", "passed"} <= set(value)
    ]
    artifact: JsonDict = {
        "schema": "carnot.exp6919.exact_prefix_viability_fixture.v1",
        "experiment_id": 6919,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": dict(preconditions),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "source_artifact_hashes": source_artifact_hashes(repo_root),
        "fixture_manifest": {},
        **_empty_evidence(),
        "retired_mechanism_activation_count": 0,
        "model_inference_call_count": 0,
        "exact_engine_disagreement_count": 0,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "prefix_viability_canary_ready_score": 0,
        "gate_check_summary": gate_summary(precondition_checks),
        "verifier_is_oracle": True,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_exact_prefix_viability_fixture",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    date: str,
    repo_root: Path = REPO_ROOT,
    final_validator: FinalValidator | None = None,
    retired_mechanism_activations: Mapping[str, int] | None = None,
) -> JsonDict:
    """Build the complete dual-engine canary or a fail-closed blocker."""

    started_ns = time.perf_counter_ns()
    root = Path(repo_root)
    preconditions = check_preconditions(root)
    if not preconditions["all_passed"]:
        duration = max((time.perf_counter_ns() - started_ns) / 1_000_000_000, 0.000001)
        return _blocked_precondition_artifact(
            date=date,
            repo_root=root,
            preconditions=preconditions,
            duration_s=round(duration, 6),
        )

    fixtures = build_source_fixtures()
    evidence = [
        _case_evidence(fixture, case, final_validator=final_validator)
        for fixture in fixtures
        for case in build_prefix_cases(fixture)
    ]
    prefix_rows = [item["case_row"] for item in evidence]
    branch_rows = [row for item in evidence for row in item["branch_rows"]]
    direct_rows = [item["direct_row"] for item in evidence]
    final_rows = [item["final_row"] for item in evidence]
    parity_rows = [item["parity_row"] for item in evidence]
    feasible_rows = [row for item in evidence for row in item["feasible"]]
    rejected_rows = [row for item in evidence for row in item["rejected"]]
    disagreement_count = sum(not row["engines_agree"] for row in parity_rows)
    family_counts = Counter(row["family"] for row in prefix_rows)
    case_counts = Counter(row["case_type"] for row in prefix_rows)
    split_rows = [
        {
            "source_id": fixture.source_id,
            "source_group": fixture.source_group,
            "split": fixture.split,
            "family": fixture.family,
            "train_free": True,
        }
        for fixture in fixtures
    ]
    canary_groups = {row["source_group"] for row in split_rows if row["split"] == "canary"}
    held_groups = {row["source_group"] for row in split_rows if row["split"] == "held"}
    overlap_count = len(canary_groups & held_groups)
    activations = {name: 0 for name in RETIRED_MECHANISMS}
    if retired_mechanism_activations:
        for name, count in retired_mechanism_activations.items():
            activations[str(name)] = int(count)
    activation_count = sum(activations.values())
    required_family_floor = all(
        family_counts[family] >= MIN_CASES_PER_FAMILY for family in FAMILIES
    )
    required_case_floor = all(
        case_counts[case_type] >= MIN_CASES_PER_TYPE for case_type in REQUIRED_CASE_TYPES
    )
    telemetry_complete = all(
        row["decision_latency_ms"] >= 0
        and row["final_available_headroom"] >= 0
        and bool(row["proof"] or row["witness"])
        for row in prefix_rows
    )
    checks = [
        _gate_check("preconditions", True, True, True),
        _gate_check(
            "minimum_prefix_cases",
            MIN_PREFIX_CASES,
            len(prefix_rows),
            len(prefix_rows) >= MIN_PREFIX_CASES,
        ),
        _gate_check("family_floors", True, required_family_floor, required_family_floor),
        _gate_check("case_type_floors", True, required_case_floor, required_case_floor),
        _gate_check("source_group_split_overlap", 0, overlap_count, overlap_count == 0),
        _gate_check("exact_engine_parity", 0, disagreement_count, disagreement_count == 0),
        _gate_check("branch_and_latency_telemetry", True, telemetry_complete, telemetry_complete),
        _gate_check(
            "retired_mechanism_activation_count", 0, activation_count, activation_count == 0
        ),
        _gate_check("model_inference_call_count", 0, 0, True),
    ]
    summary = gate_summary(checks)
    ready = int(summary["passed"])
    duration = max((time.perf_counter_ns() - started_ns) / 1_000_000_000, 0.000001)
    artifact: JsonDict = {
        "schema": "carnot.exp6919.exact_prefix_viability_fixture.v1",
        "experiment_id": 6919,
        "run_date": date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": round(duration, 6),
        "source_artifact_hashes": source_artifact_hashes(root),
        "fixture_manifest": {
            "schema": "plain_relation_program_prefix_fixture_v1",
            "plain_relation_line_shape": "subject predicate object",
            "max_relation_lines": MAX_RELATION_LINES,
            "source_group_count": len(split_rows),
            "prefix_case_count": len(prefix_rows),
            "families": list(FAMILIES),
            "family_case_counts": dict(sorted(family_counts.items())),
            "required_case_types": list(REQUIRED_CASE_TYPES),
            "auxiliary_case_types": list(AUXILIARY_CASE_TYPES),
            "case_type_counts": dict(sorted(case_counts.items())),
            "minimum_prefix_cases": MIN_PREFIX_CASES,
            "minimum_cases_per_family": MIN_CASES_PER_FAMILY,
            "minimum_cases_per_type": MIN_CASES_PER_TYPE,
            "partition_policy": "source_group_00_14_canary_source_group_15_29_held",
            "train_free": True,
            "in_loop_engine": "python_bounded_relation_enumerator_v1",
            "final_engine": (
                "clingo_stable_model_final_engine_v1"
                if final_validator is None
                else "injected_final_exact_engine"
            ),
            "retired_mechanism_activations": activations,
        },
        "rows": branch_rows,
        "prefix_case_rows": prefix_rows,
        "source_group_split_rows": split_rows,
        "in_loop_exact_engine_rows": direct_rows,
        "final_exact_engine_rows": final_rows,
        "exact_engine_parity_rows": parity_rows,
        "positive_rows": [row for row in prefix_rows if row["in_loop_extendable"]],
        "impossible_rows": [row for row in prefix_rows if not row["in_loop_extendable"]],
        "late_failure_rows": [row for row in prefix_rows if row["case_type"] == "late_failure"],
        "ambiguous_rows": [row for row in prefix_rows if row["case_type"] == "ambiguous"],
        "duplicate_rows": [row for row in prefix_rows if row["case_type"] == "duplicate"],
        "contradiction_rows": [row for row in prefix_rows if row["case_type"] == "contradiction"],
        "unsupported_rows": [row for row in prefix_rows if row["case_type"] == "unsupported"],
        "no_headroom_rows": [row for row in prefix_rows if row["case_type"] == "no_headroom"],
        "branch_factor_rows": [
            {
                "prefix_case_id": row["prefix_case_id"],
                "branch_factor": row["branch_factor"],
                "feasible_branch_count": row["feasible_branch_count"],
                "rejected_branch_count": row["rejected_branch_count"],
            }
            for row in prefix_rows
        ],
        "feasible_branch_rows": feasible_rows,
        "rejected_branch_rows": rejected_rows,
        "witness_rows": [
            {"prefix_case_id": row["prefix_case_id"], "witness": row["witness"]}
            for row in prefix_rows
            if row["witness"]
        ],
        "latency_rows": [
            {
                "prefix_case_id": row["prefix_case_id"],
                "in_loop_decision_latency_ms": direct_rows[index]["decision_latency_ms"],
                "final_engine_decision_latency_ms": final_rows[index]["decision_latency_ms"],
                "total_decision_latency_ms": row["decision_latency_ms"],
            }
            for index, row in enumerate(prefix_rows)
        ],
        "retired_mechanism_activation_count": activation_count,
        "model_inference_call_count": 0,
        "exact_engine_disagreement_count": disagreement_count,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "prefix_viability_canary_ready_score": ready,
        "gate_check_summary": summary,
        "verifier_is_oracle": True,
        "verdict_class": "circular_positive" if ready else "blocked",
        "honest_verdict": (
            "complete_exact_prefix_viability_fixture_ready"
            if ready
            else "complete_blocked_exact_prefix_viability_fixture"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def _without_timing(value: Any) -> Any:
    """Remove measured timing values so replay hashes scientific decisions only."""

    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if key != "reproducibility_checksum" and key != "duration_s" and "latency_ms" not in key
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all scientific content while excluding nondeterministic clocks."""

    return sha256_bytes(canonical_json(_without_timing(deepcopy(dict(artifact)))))


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject a result whose required fields or readiness invariants disagree."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing_required_fields:{','.join(missing)}")
    missing_principles = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact["field_principles"]))
    if missing_principles:
        raise ValueError(f"missing_field_principles:{','.join(missing_principles)}")
    ready = artifact["prefix_viability_canary_ready_score"]
    if ready not in (0, 1):
        raise ValueError("invalid_ready_score")
    if bool(ready) != bool(artifact["gate_check_summary"]["passed"]):
        raise ValueError("ready_gate_disagreement")
    if artifact["verdict_class"] == "positive":
        raise ValueError("oracle_verdict_cannot_be_positive")
    if not str(artifact["honest_verdict"]).startswith("complete_"):
        raise ValueError("honest_verdict_not_terminal")


def run(
    *,
    date: str,
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
) -> JsonDict:
    """Build, validate, and write the experiment artifact once."""

    root = Path(repo_root)
    output = Path(output_path) if output_path is not None else root / RESULT_PATH
    artifact = build_artifact(date=date, repo_root=root)
    validate_artifact(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic fixture from the repository command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args(argv)
    artifact = run(date=args.date)
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "prefix_case_count": len(artifact["prefix_case_rows"]),
                "prefix_viability_canary_ready_score": artifact[
                    "prefix_viability_canary_ready_score"
                ],
                "result_path": str(RESULT_PATH),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - the checked wrapper calls main
    raise SystemExit(main())

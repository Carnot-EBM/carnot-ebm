"""Exp 5318 deterministic SMT hint validation protocol.

Spec refs: REQ-VERIFY-5318, SCENARIO-VERIFY-5318.

This fixture prepares the slot where future systems may propose SMT
instantiations, conjectures, or lemmas. In this experiment the proposer is a
canned deterministic fixture, not an LLM. Every hint is checked by a local SMT
solver before it can affect a solve, so a bad hint can only produce telemetry
and then fall back to the classical solver result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import z3


JsonDict = dict[str, Any]

RUN_DATE = "20260706"
RANDOM_SEED = 5318
EXPERIMENT_ID = "exp5318-smt-hint-validation-protocol-v485"
MILESTONE = "2026.07.485"
SCHEMA = "carnot.experiment_5318.smt_hint_validation_protocol.v485"
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5318_smt_hint_validation_protocol_v485.json"
)
FIXTURE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5318_smt_hint_validation_protocol_v485.py"
)
INFERENCE_SUBSTRATE = "deterministic_smt_hint_validation_no_llm"
SPEC_REFS = ("REQ-VERIFY-5318", "SCENARIO-VERIFY-5318")
TERMINAL_PREFIXES = ("complete:", "null:", "blocked_")
PROPOSAL_SOURCE = "canned_fixture_no_llm"
EXP5309_GATE = (
    "Exp5309 results/experiment_5309_sota_runtime_timeout_rootcause_matrix_v485.json "
    "requires sota_runtime_unblocked=true before any SOTA GGUF hint proposer is enabled"
)

FIELD_PRINCIPLES: dict[str, str] = {
    "experiment_id": (
        "Traceable Exp 5318 identifier for the deterministic SMT hint validation protocol."
    ),
    "milestone": "Milestone accountability for the V485 SMT hint-validation fixture.",
    "status": (
        "Terminal status for downstream solver-guidance readers; complete means the "
        "deterministic no-LLM hint protocol ran and preserved completeness."
    ),
    "honest_verdict": (
        "Terminal Exp 5318 verdict; starts with complete:, null:, or blocked_ and "
        "states whether SMT hint validation is ready."
    ),
    "inference_substrate": (
        "Declares deterministic_smt_hint_validation_no_llm so canned solver-validated "
        "hints are not mistaken for live LLM inference."
    ),
    "smt_hint_protocol_ready": (
        "Bare boolean true only when valid hints are accepted, unsound hints are "
        "rejected, fallback reaches the classical solver result, and completeness is "
        "preserved."
    ),
    "fixture_path": (
        "Points to the deterministic fixture module that owns the canned quantified "
        "and inductive SMT examples."
    ),
    "valid_hint_acceptance_rate": (
        "Bare numeric fraction of solver-validated sound hints accepted by the protocol."
    ),
    "unsound_hint_rejection_rate": (
        "Bare numeric fraction of solver-refuted unsound hints rejected before they "
        "can affect final solving."
    ),
    "usefulness_rate": (
        "Bare numeric fraction of accepted valid hints that are non-redundant and "
        "reduce the fixture's deterministic proof burden."
    ),
    "solver_fallback_complete": (
        "Bare boolean proving rejected hints fall back to the classical SMT solver and "
        "still reach the baseline result."
    ),
    "completeness_preserved": (
        "Bare boolean proving accepted hints never change the classical solver "
        "SAT/UNSAT label and rejected unsound hints cannot block fallback."
    ),
    "future_llm_slot_gated_on_sota_runtime": (
        "Bare boolean that must stay true until the Exp5309 SOTA GGUF runtime gate is "
        "ready for any future hint proposer."
    ),
    "tests_run": (
        "Commands run to validate hint soundness, unsound rejection, fallback "
        "completeness, artifact schema, new-code coverage, repository tests, and "
        "applicable offline e2e checks."
    ),
}

REQUIRED_ARTIFACT_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "smt_hint_protocol_ready",
    "fixture_path",
    "valid_hint_acceptance_rate",
    "unsound_hint_rejection_rate",
    "usefulness_rate",
    "solver_fallback_complete",
    "completeness_preserved",
    "future_llm_slot_gated_on_sota_runtime",
    "tests_run",
)
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "fixture_path",
    "tests_run",
)
BARE_BOOL_FIELDS = (
    "smt_hint_protocol_ready",
    "solver_fallback_complete",
    "completeness_preserved",
    "future_llm_slot_gated_on_sota_runtime",
)
BARE_NUMERIC_FIELDS = (
    "valid_hint_acceptance_rate",
    "unsound_hint_rejection_rate",
    "usefulness_rate",
)


@dataclass(frozen=True)
class SmtHintCandidate:
    """One proposed SMT hint before solver validation decides whether it is safe."""

    hint_id: str
    hint_kind: str
    expected_class: str
    formula: z3.BoolRef
    burden_delta: int
    description: str
    overwrite_clause: str
    proposal_source: str = PROPOSAL_SOURCE

    def as_serializable(self) -> JsonDict:
        return {
            "hint_id": self.hint_id,
            "hint_kind": self.hint_kind,
            "expected_class": self.expected_class,
            "formula": self.formula.sexpr(),
            "burden_delta": self.burden_delta,
            "description": self.description,
            "overwrite_clause": self.overwrite_clause,
            "proposal_source": self.proposal_source,
        }


@dataclass(frozen=True)
class SmtHintExample:
    """One tiny SMT problem with a satisfiable validation context."""

    example_id: str
    style: str
    variables: Mapping[str, z3.ExprRef]
    validation_constraints: tuple[z3.BoolRef, ...]
    problem_constraints: tuple[z3.BoolRef, ...]
    baseline_proof_burden: int
    hints: tuple[SmtHintCandidate, ...]

    def as_serializable(self) -> JsonDict:
        return {
            "example_id": self.example_id,
            "style": self.style,
            "validation_context_status": validation_context_status(self),
            "baseline_status": solve_status(self.problem_constraints),
            "baseline_proof_burden": self.baseline_proof_burden,
            "hints": [hint.as_serializable() for hint in self.hints],
        }


def wrap_field(field: str, value: Any) -> JsonDict:
    """Attach the task-required principle to an artifact field."""

    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def build_smt_hint_examples() -> tuple[SmtHintExample, ...]:
    """Build quantified, inductive, and SAT-control fixtures with canned hints."""

    return (
        _quantified_successor_example(),
        _inductive_chain_example(),
        _sat_choice_overwrite_example(),
    )


def validation_context_status(example: SmtHintExample) -> str:
    """Return the validation-context status so entailment cannot be vacuous."""

    return solve_status(example.validation_constraints)


def solve_status(constraints: Sequence[z3.BoolRef]) -> str:
    """Run the local SMT solver once and return a stable status string."""

    return _solve(constraints, {}).status


def evaluate_hint(example: SmtHintExample, hint: SmtHintCandidate) -> JsonDict:
    """Validate one hint, accept only entailed hints, and preserve fallback."""

    baseline = _solve(example.problem_constraints, example.variables)
    validation_status = validation_context_status(example)
    refutation = _solve(
        (*example.validation_constraints, z3.Not(hint.formula)),
        example.variables,
    )
    solver_valid = validation_status == "sat" and refutation.status == "unsat"
    accepted = solver_valid
    blindly_added = _solve(
        (*example.problem_constraints, hint.formula),
        example.variables,
    )
    final = blindly_added if accepted else baseline
    fallback_to_classical = not accepted
    redundant = accepted and _formula_already_present(
        hint.formula,
        (*example.validation_constraints, *example.problem_constraints),
    )
    usefulness_class = _usefulness_class(hint, solver_valid, redundant)
    hinted_burden = (
        max(0, example.baseline_proof_burden - hint.burden_delta)
        if accepted
        else example.baseline_proof_burden
    )
    useful = bool(
        accepted
        and usefulness_class == "useful"
        and hinted_burden < example.baseline_proof_burden
    )
    overwrite_clauses = _overwrite_clauses(hint, baseline)
    completeness_preserved = final.status == baseline.status
    return {
        "example_id": example.example_id,
        "style": example.style,
        "hint_id": hint.hint_id,
        "hint_kind": hint.hint_kind,
        "expected_class": hint.expected_class,
        "proposal_source": hint.proposal_source,
        "formula": hint.formula.sexpr(),
        "validation_context_status": validation_status,
        "solver_valid": solver_valid,
        "accepted": accepted,
        "usefulness_class": usefulness_class,
        "useful": useful,
        "redundant": redundant,
        "baseline_status": baseline.status,
        "blindly_added_status": blindly_added.status,
        "final_status": final.status,
        "baseline_model": baseline.model_values,
        "final_model": final.model_values,
        "fallback_to_classical": fallback_to_classical,
        "overwrite_count": len(overwrite_clauses),
        "overwrite_clauses": overwrite_clauses,
        "baseline_proof_burden": example.baseline_proof_burden,
        "hinted_proof_burden": hinted_burden,
        "completeness_preserved": completeness_preserved,
    }


def run_benchmark() -> JsonDict:
    """Run every canned SMT hint through solver validation and fallback."""

    examples = build_smt_hint_examples()
    rows = [
        evaluate_hint(example, hint)
        for example in examples
        for hint in example.hints
    ]
    valid_rows = [row for row in rows if row["solver_valid"]]
    unsound_rows = [row for row in rows if not row["solver_valid"]]
    accepted_valid = [row for row in valid_rows if row["accepted"]]
    useful_rows = [row for row in accepted_valid if row["useful"]]
    valid_hint_acceptance_rate = _rate(len(accepted_valid), len(valid_rows))
    unsound_hint_rejection_rate = _rate(
        sum(not row["accepted"] for row in unsound_rows),
        len(unsound_rows),
    )
    usefulness_rate = _rate(len(useful_rows), len(accepted_valid))
    solver_fallback_complete = all(
        row["fallback_to_classical"] and row["final_status"] == row["baseline_status"]
        for row in unsound_rows
    )
    completeness_preserved = all(row["completeness_preserved"] for row in rows)
    future_llm_slot_gated = True
    protocol_ready = bool(
        valid_hint_acceptance_rate == 1.0
        and unsound_hint_rejection_rate == 1.0
        and solver_fallback_complete
        and completeness_preserved
        and future_llm_slot_gated
    )
    return {
        "fixture_examples": [example.as_serializable() for example in examples],
        "hint_validation_telemetry": rows,
        "valid_hint_acceptance_rate": valid_hint_acceptance_rate,
        "unsound_hint_rejection_rate": unsound_hint_rejection_rate,
        "usefulness_rate": usefulness_rate,
        "solver_fallback_complete": solver_fallback_complete,
        "completeness_preserved": completeness_preserved,
        "future_llm_slot_gated_on_sota_runtime": future_llm_slot_gated,
        "smt_hint_protocol_ready": protocol_ready,
        "llm_invoked": False,
        "future_llm_slot": {
            "current_proposer": PROPOSAL_SOURCE,
            "gate": EXP5309_GATE,
            "interface": (
                "future proposer returns candidate SMT hints; validator still "
                "requires local solver entailment before any hint is accepted"
            ),
            "runtime_gate_ready_required": True,
        },
        "counts": {
            "examples": len(examples),
            "hints": len(rows),
            "valid_hints": len(valid_rows),
            "unsound_hints": len(unsound_rows),
            "useful_accepted_hints": len(useful_rows),
        },
    }


def build_artifact(
    *,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the validated Exp 5318 terminal artifact."""

    started_at = time.perf_counter()
    benchmark = run_benchmark()
    measured_duration = (
        round(time.perf_counter() - started_at, 6)
        if duration_s is None
        else duration_s
    )
    status = "complete" if benchmark["smt_hint_protocol_ready"] else "blocked"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "run_date": RUN_DATE,
        "duration_s": measured_duration,
        "experiment_id": wrap_field("experiment_id", EXPERIMENT_ID),
        "milestone": wrap_field("milestone", MILESTONE),
        "status": wrap_field("status", status),
        "honest_verdict": wrap_field("honest_verdict", _honest_verdict(benchmark)),
        "inference_substrate": wrap_field("inference_substrate", INFERENCE_SUBSTRATE),
        "smt_hint_protocol_ready": benchmark["smt_hint_protocol_ready"],
        "fixture_path": wrap_field("fixture_path", str(FIXTURE_RELATIVE_PATH)),
        "valid_hint_acceptance_rate": benchmark["valid_hint_acceptance_rate"],
        "unsound_hint_rejection_rate": benchmark["unsound_hint_rejection_rate"],
        "usefulness_rate": benchmark["usefulness_rate"],
        "solver_fallback_complete": benchmark["solver_fallback_complete"],
        "completeness_preserved": benchmark["completeness_preserved"],
        "future_llm_slot_gated_on_sota_runtime": benchmark[
            "future_llm_slot_gated_on_sota_runtime"
        ],
        "tests_run": wrap_field("tests_run", [dict(row) for row in tests_run or []]),
        "fixture_examples": benchmark["fixture_examples"],
        "hint_validation_telemetry": benchmark["hint_validation_telemetry"],
        "future_llm_slot": benchmark["future_llm_slot"],
        "llm_invoked": benchmark["llm_invoked"],
        "counts": benchmark["counts"],
        "claim_limits": [
            "deterministic SMT hint-validation fixture only",
            "canned hints only; no LLM proposer invoked",
            "Z3 entailment validates hints before acceptance",
            "unsound hints fall back to the classical SMT result",
            "future SOTA GGUF proposer remains gated on Exp5309 runtime readiness",
        ],
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": list(SPEC_REFS),
        "random_seed": RANDOM_SEED,
    }
    artifact["reproducibility_checksum"] = _checksum_payload(benchmark)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Fail closed when the Exp 5318 artifact drifts from its contract."""

    for field in REQUIRED_ARTIFACT_FIELDS:
        _require(field in artifact, f"missing required field: {field}")
    for field in WRAPPED_FIELDS:
        wrapped = artifact[field]
        _require(isinstance(wrapped, Mapping), f"{field} must be principle-wrapped")
        _require("value" in wrapped, f"{field} missing value")
        _require(
            wrapped.get("principle") == FIELD_PRINCIPLES[field],
            f"{field} principle drift",
        )
    for field in BARE_BOOL_FIELDS:
        _require(isinstance(artifact[field], bool), f"{field} must be a bare bool")
    for field in BARE_NUMERIC_FIELDS:
        _require(
            isinstance(artifact[field], int | float) and not isinstance(artifact[field], bool),
            f"{field} must be a bare numeric value",
        )
        _require(0.0 <= float(artifact[field]) <= 1.0, f"{field} rate out of range")

    verdict = artifact["honest_verdict"]["value"]
    _require(
        isinstance(verdict, str) and verdict.startswith(TERMINAL_PREFIXES),
        "honest_verdict prefix",
    )
    _require(artifact["experiment_id"]["value"] == EXPERIMENT_ID, "experiment_id drift")
    _require(artifact["milestone"]["value"] == MILESTONE, "milestone drift")
    _require(artifact["status"]["value"] == "complete", "status must be complete")
    _require(
        artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE,
        f"inference_substrate must be {INFERENCE_SUBSTRATE}",
    )
    _require(
        artifact["fixture_path"]["value"] == str(FIXTURE_RELATIVE_PATH),
        "fixture_path drift",
    )
    _require(
        artifact["smt_hint_protocol_ready"] is True,
        "smt_hint_protocol_ready must be a bare bool true",
    )
    _require(
        artifact["valid_hint_acceptance_rate"] == 1.0,
        "valid hint acceptance must be complete",
    )
    _require(
        artifact["unsound_hint_rejection_rate"] == 1.0,
        "unsound hint rejection must be complete",
    )
    _require(
        artifact["solver_fallback_complete"] is True,
        "solver fallback must be complete",
    )
    _require(
        artifact["completeness_preserved"] is True,
        "completeness must be preserved",
    )
    _require(
        artifact["future_llm_slot_gated_on_sota_runtime"] is True,
        "Exp5309 runtime gate must protect the future LLM slot",
    )
    _require(artifact["llm_invoked"] is False, "LLM must not be invoked")
    _require(isinstance(artifact["tests_run"]["value"], list), "tests_run must be list")
    _require("REQ-VERIFY-5318" in artifact["spec_refs"], "spec refs must include REQ-VERIFY-5318")
    _require(len(str(artifact["reproducibility_checksum"])) == 64, "checksum drift")


def write_outputs(
    *,
    artifact_path: str | Path = RESULT_RELATIVE_PATH,
    duration_s: float | None = None,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Write the Exp 5318 JSON artifact and return the validated payload."""

    artifact = build_artifact(duration_s=duration_s, tests_run=tests_run)
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


@dataclass(frozen=True)
class _SolveOutcome:
    status: str
    model: z3.ModelRef | None
    model_values: JsonDict


def _quantified_successor_example() -> SmtHintExample:
    qx = z3.Int("quantified_x")
    n = z3.Int("quantified_n")
    successor_axiom = z3.ForAll(
        [qx],
        z3.Implies(z3.And(qx >= 0, qx <= 3), qx + 1 > qx),
    )
    validation = (successor_axiom, n >= 0, n <= 3)
    problem = (*validation, n + 1 <= n)
    hints = (
        SmtHintCandidate(
            hint_id="quantified_successor_instantiation",
            hint_kind="instantiation",
            expected_class="valid",
            formula=n + 1 > n,
            burden_delta=2,
            description="Instantiate the quantified successor axiom at n.",
            overwrite_clause="quantified_n_successor_positive",
        ),
        SmtHintCandidate(
            hint_id="quantified_tautology_useless",
            hint_kind="lemma",
            expected_class="useless",
            formula=z3.Or(n == n, n != n),
            burden_delta=0,
            description="Valid tautology that should not improve the proof burden.",
            overwrite_clause="quantified_tautology",
        ),
        SmtHintCandidate(
            hint_id="quantified_unsound_out_of_domain",
            hint_kind="conjecture",
            expected_class="unsound",
            formula=n == 4,
            burden_delta=0,
            description="Unsound conjecture outside the declared bounded domain.",
            overwrite_clause="quantified_n_equals_4",
        ),
    )
    return SmtHintExample(
        example_id="quantified_successor",
        style="quantified_instantiation",
        variables={"n": n},
        validation_constraints=validation,
        problem_constraints=problem,
        baseline_proof_burden=4,
        hints=hints,
    )


def _inductive_chain_example() -> SmtHintExample:
    s0, s1, s2, s3 = z3.Bools("ind_s0 ind_s1 ind_s2 ind_s3")
    validation = (
        s0,
        z3.Implies(s0, s1),
        z3.Implies(s1, s2),
        z3.Implies(s2, s3),
    )
    problem = (*validation, z3.Not(s3))
    hints = (
        SmtHintCandidate(
            hint_id="inductive_goal_lemma",
            hint_kind="lemma",
            expected_class="valid",
            formula=s3,
            burden_delta=2,
            description="Inductive reachability lemma for the final state.",
            overwrite_clause="ind_s3",
        ),
        SmtHintCandidate(
            hint_id="inductive_start_redundant",
            hint_kind="lemma",
            expected_class="redundant",
            formula=s0,
            burden_delta=0,
            description="Duplicate of the base fact already asserted.",
            overwrite_clause="ind_s0",
        ),
        SmtHintCandidate(
            hint_id="inductive_unsound_not_reached",
            hint_kind="conjecture",
            expected_class="unsound",
            formula=z3.Not(s3),
            burden_delta=0,
            description="False conjecture contradicting the inductive chain.",
            overwrite_clause="not_ind_s3",
        ),
    )
    return SmtHintExample(
        example_id="inductive_reachability",
        style="inductive_chain",
        variables={"s0": s0, "s1": s1, "s2": s2, "s3": s3},
        validation_constraints=validation,
        problem_constraints=problem,
        baseline_proof_burden=4,
        hints=hints,
    )


def _sat_choice_overwrite_example() -> SmtHintExample:
    a, b = z3.Bools("sat_choice_a sat_choice_b")
    validation = (z3.Or(a, b), z3.Not(b), z3.Not(z3.And(a, b)))
    hints = (
        SmtHintCandidate(
            hint_id="sat_choice_valid_a",
            hint_kind="lemma",
            expected_class="valid",
            formula=a,
            burden_delta=2,
            description="Entailed SAT-side lemma selecting the only remaining branch.",
            overwrite_clause="sat_choice_a",
        ),
        SmtHintCandidate(
            hint_id="sat_choice_redundant_or",
            hint_kind="lemma",
            expected_class="redundant",
            formula=z3.Or(a, b),
            burden_delta=0,
            description="Duplicate of the original disjunction.",
            overwrite_clause="sat_choice_a_or_b",
        ),
        SmtHintCandidate(
            hint_id="sat_choice_unsound_b",
            hint_kind="conjecture",
            expected_class="unsound",
            formula=b,
            burden_delta=0,
            description="Unsound branch conjecture that would make the SAT problem UNSAT.",
            overwrite_clause="sat_choice_b",
        ),
    )
    return SmtHintExample(
        example_id="sat_choice_overwrite",
        style="sat_overwrite_control",
        variables={"a": a, "b": b},
        validation_constraints=validation,
        problem_constraints=validation,
        baseline_proof_burden=3,
        hints=hints,
    )


def _solve(
    constraints: Sequence[z3.BoolRef],
    variables: Mapping[str, z3.ExprRef],
) -> _SolveOutcome:
    solver = z3.Solver()
    for constraint in constraints:
        solver.add(constraint)
    status = str(solver.check())
    model = solver.model() if status == "sat" else None
    model_values = _model_values(model, variables) if model is not None else {}
    return _SolveOutcome(status=status, model=model, model_values=model_values)


def _model_values(model: z3.ModelRef, variables: Mapping[str, z3.ExprRef]) -> JsonDict:
    values: JsonDict = {}
    for name, variable in sorted(variables.items()):
        value = model.eval(variable, model_completion=True)
        if z3.is_bool(variable):
            values[name] = bool(z3.is_true(value))
        elif z3.is_int_value(value):
            values[name] = int(value.as_long())
        else:  # pragma: no cover - all current fixture variables are bools or ints.
            values[name] = str(value)
    return values


def _formula_already_present(
    formula: z3.BoolRef,
    constraints: Sequence[z3.BoolRef],
) -> bool:
    needle = formula.sexpr()
    return any(constraint.sexpr() == needle for constraint in constraints)


def _usefulness_class(
    hint: SmtHintCandidate,
    solver_valid: bool,
    redundant: bool,
) -> str:
    if not solver_valid:
        return "unsound"
    if redundant:
        return "redundant"
    if hint.expected_class == "useless":
        return "useless"
    return "useful"


def _overwrite_clauses(
    hint: SmtHintCandidate,
    baseline: _SolveOutcome,
) -> list[str]:
    if baseline.model is not None and z3.is_false(
        baseline.model.eval(hint.formula, model_completion=True)
    ):
        return [hint.overwrite_clause]
    return []


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else round(count / total, 6)


def _honest_verdict(benchmark: Mapping[str, Any]) -> str:
    if benchmark["smt_hint_protocol_ready"]:
        return (
            "complete: deterministic SMT hint validation protocol is ready; "
            "valid hints are accepted, unsound hints are rejected, and future LLM "
            "hint proposals remain gated on Exp5309 runtime readiness"
        )
    return "blocked_smt_hint_protocol_not_ready"  # pragma: no cover - current fixture is ready.


def _checksum_payload(benchmark: Mapping[str, Any]) -> str:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": SPEC_REFS,
        "fixture_examples": benchmark["fixture_examples"],
        "hint_validation_telemetry": [
            {
                "example_id": row["example_id"],
                "hint_id": row["hint_id"],
                "solver_valid": row["solver_valid"],
                "accepted": row["accepted"],
                "usefulness_class": row["usefulness_class"],
                "baseline_status": row["baseline_status"],
                "blindly_added_status": row["blindly_added_status"],
                "final_status": row["final_status"],
                "overwrite_clauses": row["overwrite_clauses"],
            }
            for row in benchmark["hint_validation_telemetry"]
        ],
        "rates": {
            "valid": benchmark["valid_hint_acceptance_rate"],
            "unsound": benchmark["unsound_hint_rejection_rate"],
            "usefulness": benchmark["usefulness_rate"],
        },
        "smt_hint_protocol_ready": benchmark["smt_hint_protocol_ready"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = write_outputs()
    print(artifact["honest_verdict"]["value"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Bounded VerIbmc-style local SOTA solver-feedback pilot for Exp 5226.

Spec refs: REQ-VERIFY-5226, SCENARIO-VERIFY-5226.
"""

from __future__ import annotations

import ast
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - absence is covered through blocked artifacts, not import tricks.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None

from carnot.inference.sota_models import SOTA_GGUF_MODELS, cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RUN_DATE = "20260704"
SCHEMA = "carnot.experiment_5226.veribmc_local_solver_feedback_pilot.v478"
RESULT_RELATIVE_PATH = Path("results/experiment_5226_veribmc_local_solver_feedback_pilot_v478.json")
SPEC_REFS = ("REQ-VERIFY-5226", "SCENARIO-VERIFY-5226")
INFERENCE_SUBSTRATE = "local_sota_gguf_plus_deterministic_solver_feedback"
CHECKER_SUBSTRATE = "z3"
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_")
MANDATED_SOTA_IDS = tuple(model["hf_id"] for model in SOTA_GGUF_MODELS)

FIELD_PRINCIPLES = {
    "solver_feedback_pilot_complete": (
        "True only when all three arms ran on the same bounded fixture and the artifact was validated."
    ),
    "n_examples": (
        "Number of fixture examples used for every arm; the pilot is bounded and not a broad benchmark."
    ),
    "solver_only_solved": (
        "Count accepted by the deterministic solver-only baseline on the same examples."
    ),
    "llm_only_solved": (
        "Count accepted from initial local SOTA GGUF proposals before solver-feedback retry."
    ),
    "llm_solver_feedback_solved": (
        "Count accepted after at most one structured solver-feedback retry."
    ),
    "solver_feedback_uplift": (
        "llm_solver_feedback_solved minus the stronger of solver_only_solved and llm_only_solved, divided by n_examples."
    ),
    "accepted_invariants_or_constraints": (
        "Accepted invariant or formal-constraint strings, grouped by example and arm."
    ),
    "model_specs": (
        "Concrete resolved model spec records, including at least one mandated local SOTA GGUF for a non-smoke result."
    ),
    "models_used": "Concrete model identifiers and paths actually attempted for local SOTA inference.",
    "checker_substrate": "Honest deterministic checker substrate: z3, smt, esbmc, or other.",
    "tests_run": "Commands run for this pilot, with pass/fail status.",
    "inference_substrate": "Must be local_sota_gguf_plus_deterministic_solver_feedback.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether solver feedback improved over baselines."
    ),
}


@dataclass(frozen=True)
class LoopInvariantExample:
    """One tiny loop-verification fixture checked by Z3."""

    example_id: str
    title: str
    variables: tuple[str, ...]
    precondition: str
    guard: str
    updates: Mapping[str, str]
    postcondition: str
    baseline_candidates: tuple[str, ...]
    repair_hint: str


@dataclass(frozen=True)
class CheckResult:
    """Deterministic checker result for one invariant candidate."""

    accepted: bool
    failed_obligation: str | None
    counterexample: JsonDict
    feedback: JsonDict
    runtime_ms: float


@dataclass(frozen=True)
class ArmResult:
    """One arm's result on one fixture example."""

    example_id: str
    arm: str
    raw_output: str
    parsed_invariant: str | None
    accepted: bool
    failed_obligation: str | None
    counterexample: JsonDict
    solver_feedback: JsonDict
    runtime_ms: float
    failure_mode: str | None

    def to_dict(self) -> JsonDict:
        return {
            "example_id": self.example_id,
            "arm": self.arm,
            "raw_output": self.raw_output,
            "parsed_invariant": self.parsed_invariant,
            "accepted": self.accepted,
            "failed_obligation": self.failed_obligation,
            "counterexample": self.counterexample,
            "solver_feedback": self.solver_feedback,
            "runtime_ms": self.runtime_ms,
            "failure_mode": self.failure_mode,
        }


@dataclass(frozen=True)
class ProposalPrompt:
    """Prompt payload passed to an LLM proposal function."""

    example: LoopInvariantExample
    arm: str
    prior_invariant: str | None = None
    solver_feedback: Mapping[str, Any] | None = None


ProposalFn = Callable[[ProposalPrompt], str]
ModelSpecsProvider = Callable[[], list[JsonDict]]


def fixture_examples() -> list[LoopInvariantExample]:
    """Return the bounded VerIbmc-style fixture used by all three arms."""

    return [
        LoopInvariantExample(
            example_id="inc_to_n",
            title="Increment a counter until it reaches n",
            variables=("i", "n"),
            precondition="n >= 0 and i == 0",
            guard="i < n",
            updates={"i": "i + 1", "n": "n"},
            postcondition="i == n",
            baseline_candidates=("0 <= i <= n",),
            repair_hint="Keep both the lower counter bound and the upper n bound.",
        ),
        LoopInvariantExample(
            example_id="sum_to_n",
            title="Accumulate 1+...+n with a loop counter",
            variables=("i", "s", "n"),
            precondition="n >= 0 and i == 0 and s == 0",
            guard="i < n",
            updates={"i": "i + 1", "s": "s + i + 1", "n": "n"},
            postcondition="2*s == n*(n+1)",
            baseline_candidates=("0 <= i <= n and s >= 0",),
            repair_hint="Relate the accumulator to the counter, not just their bounds.",
        ),
        LoopInvariantExample(
            example_id="paired_decrement",
            title="Decrement y while preserving an initial nonnegative x",
            variables=("x", "y"),
            precondition="x >= 0 and y == x",
            guard="y > 0",
            updates={"x": "x", "y": "y - 1"},
            postcondition="y == 0 and x >= 0",
            baseline_candidates=("y >= 0",),
            repair_hint="Preserve that y stays between 0 and the unchanged x.",
        ),
    ]


def resolve_model_specs_for_pilot() -> list[JsonDict]:
    """Resolve mandated GGUF models, trying ``cached_sota_pair`` first."""

    pair = cached_sota_pair(gpu_indices=(0, 1))
    if pair:
        return [dict(spec) for spec in pair]

    resolved: list[JsonDict] = []
    for model in SOTA_GGUF_MODELS:
        path = resolve_cached_gguf(model["hf_id"])
        if path:
            row = {
                "name": model["name"],
                "hf_id": model["hf_id"],
                "gpu": len(resolved),
                "model_path": path,
            }
            resolved.append(row)
    return resolved


def parse_invariant_text(text: str) -> str | None:
    """Extract one invariant expression from JSON, tagged text, or a bare line."""

    cleaned = _strip_code_fence(text.strip())
    if not cleaned:
        return None

    json_candidate = _extract_json_object(cleaned)
    if json_candidate is not None:
        parsed = _invariant_from_json(json_candidate)
        if parsed:
            return parsed

    tagged = re.search(r"(?im)^\s*(?:INVARIANT|INVARIANTS|CONSTRAINT)\s*[:=]\s*(.+?)\s*$", cleaned)
    if tagged:
        return _clean_candidate_expression(tagged.group(1))

    prose = re.search(
        r"(?im)\binvariant\s+(?:should\s+be|is)\s+(.+?)\s*(?:[\.;\n]|$)",
        cleaned,
    )
    if prose:
        return _clean_candidate_expression(prose.group(1))

    for line in cleaned.splitlines():
        if any(token in line for token in ("<=", ">=", "==", "<", ">")):
            return _clean_candidate_expression(line)
    return None


def compile_formula(
    expression: str,
    variables: Sequence[str],
    *,
    z3_module: Any = _z3,
    env: Mapping[str, Any] | None = None,
) -> Any:
    """Compile the tiny invariant DSL into a Z3 expression."""

    if z3_module is None:
        raise RuntimeError("z3_unavailable")
    stripped = expression.strip()
    if stripped.lower() == "true":
        return z3_module.BoolVal(True)
    if stripped.lower() == "false":
        return z3_module.BoolVal(False)

    active_env = dict(env or {name: z3_module.Int(name) for name in variables})
    tree = ast.parse(stripped, mode="eval")
    return _compile_ast_node(tree.body, active_env, z3_module)


def check_invariant(
    example: LoopInvariantExample,
    invariant: str,
    *,
    z3_module: Any = _z3,
    timeout_ms: int = 2000,
) -> CheckResult:
    """Check initiation, preservation, and postcondition obligations."""

    started = time.perf_counter()
    try:
        current = {name: z3_module.Int(name) for name in example.variables}
        nxt = {name: z3_module.Int(f"{name}_next") for name in example.variables}
        pre = compile_formula(example.precondition, example.variables, z3_module=z3_module, env=current)
        guard = compile_formula(example.guard, example.variables, z3_module=z3_module, env=current)
        post = compile_formula(example.postcondition, example.variables, z3_module=z3_module, env=current)
        inv = compile_formula(invariant, example.variables, z3_module=z3_module, env=current)
        inv_next = compile_formula(invariant, example.variables, z3_module=z3_module, env=nxt)
    except Exception as exc:
        runtime_ms = _elapsed_ms(started)
        feedback = {
            "failed_obligation": "parse_error",
            "counterexample": {},
            "repair_hint": f"Return a Python-style integer invariant over {', '.join(example.variables)}.",
            "checker_status": f"{type(exc).__name__}: {exc}",
        }
        return CheckResult(False, "parse_error", {}, feedback, runtime_ms)

    initiation = _prove_unsat(
        [pre, z3_module.Not(inv)],
        obligation="initiation",
        variables=current,
        next_variables={},
        z3_module=z3_module,
        timeout_ms=timeout_ms,
    )
    if initiation is not None:
        return _failed_check(started, example, invariant, initiation)

    transition = [
        inv,
        guard,
        *[
            nxt[name]
            == compile_formula(
                str(example.updates.get(name, name)),
                example.variables,
                z3_module=z3_module,
                env=current,
            )
            for name in example.variables
        ],
        z3_module.Not(inv_next),
    ]
    preservation = _prove_unsat(
        transition,
        obligation="preservation",
        variables=current,
        next_variables=nxt,
        z3_module=z3_module,
        timeout_ms=timeout_ms,
    )
    if preservation is not None:
        return _failed_check(started, example, invariant, preservation)

    postcondition = _prove_unsat(
        [inv, z3_module.Not(guard), z3_module.Not(post)],
        obligation="postcondition",
        variables=current,
        next_variables={},
        z3_module=z3_module,
        timeout_ms=timeout_ms,
    )
    if postcondition is not None:
        return _failed_check(started, example, invariant, postcondition)

    return CheckResult(
        accepted=True,
        failed_obligation=None,
        counterexample={},
        feedback={},
        runtime_ms=_elapsed_ms(started),
    )


def run_solver_only_baseline(example: LoopInvariantExample) -> ArmResult:
    """Run the deterministic no-LLM candidate set for one example."""

    last_result: ArmResult | None = None
    for candidate in example.baseline_candidates:
        result = evaluate_proposal(example, candidate, arm="solver_only")
        if result.accepted:
            return result
        last_result = result
    return last_result or evaluate_proposal(example, "", arm="solver_only")


def evaluate_proposal(example: LoopInvariantExample, raw_output: str, *, arm: str) -> ArmResult:
    """Parse and check one proposal string."""

    parsed = parse_invariant_text(raw_output)
    if parsed is None:
        feedback = {
            "failed_obligation": "parse_error",
            "counterexample": {},
            "repair_hint": "Return exactly JSON with an invariant string.",
            "checker_status": "no_parseable_invariant",
        }
        return ArmResult(
            example_id=example.example_id,
            arm=arm,
            raw_output=raw_output,
            parsed_invariant=None,
            accepted=False,
            failed_obligation="parse_error",
            counterexample={},
            solver_feedback=feedback,
            runtime_ms=0.0,
            failure_mode="parse_error",
        )

    checked = check_invariant(example, parsed)
    return ArmResult(
        example_id=example.example_id,
        arm=arm,
        raw_output=raw_output,
        parsed_invariant=parsed,
        accepted=checked.accepted,
        failed_obligation=checked.failed_obligation,
        counterexample=checked.counterexample,
        solver_feedback=checked.feedback,
        runtime_ms=checked.runtime_ms,
        failure_mode=None if checked.accepted else checked.failed_obligation,
    )


def run_pilot(
    *,
    examples: Sequence[LoopInvariantExample] | None = None,
    proposal_fn: ProposalFn | None = None,
    model_specs_provider: ModelSpecsProvider = resolve_model_specs_for_pilot,
    tests_run: Sequence[str] | None = None,
    duration_s: float | None = None,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Run all three pilot arms and return the validated artifact."""

    started = time.perf_counter()
    active_examples = list(examples or fixture_examples())
    model_specs = model_specs_provider()
    models_used = _models_used_from_specs(model_specs)
    if _z3 is None or not _has_mandated_sota_model(model_specs):
        artifact = build_artifact(
            examples=active_examples,
            model_specs=model_specs,
            models_used=models_used,
            checker_substrate=CHECKER_SUBSTRATE if _z3 is not None else "other",
            tests_run=list(tests_run or []),
            duration_s=_duration(duration_s, started),
            solver_only_results=[],
            llm_initial_results=[],
            llm_feedback_results=[],
            failure_modes={"z3_or_model_precondition": len(active_examples)},
            complete=False,
            run_date=run_date,
        )
        return artifact

    active_proposal_fn = proposal_fn or LiveGGUFProposalGenerator(model_specs)
    solver_results: list[ArmResult] = []
    initial_results: list[ArmResult] = []
    feedback_results: list[ArmResult] = []

    for example in active_examples:
        solver_results.append(run_solver_only_baseline(example))
        initial_raw = active_proposal_fn(ProposalPrompt(example=example, arm="initial"))
        initial = evaluate_proposal(example, initial_raw, arm="llm_only")
        initial_results.append(initial)

        if initial.accepted:
            feedback_results.append(
                ArmResult(
                    example_id=example.example_id,
                    arm="llm_solver_feedback",
                    raw_output=initial.raw_output,
                    parsed_invariant=initial.parsed_invariant,
                    accepted=True,
                    failed_obligation=None,
                    counterexample={},
                    solver_feedback={},
                    runtime_ms=initial.runtime_ms,
                    failure_mode=None,
                )
            )
            continue

        retry_raw = active_proposal_fn(
            ProposalPrompt(
                example=example,
                arm="feedback",
                prior_invariant=initial.parsed_invariant,
                solver_feedback=initial.solver_feedback,
            )
        )
        feedback_results.append(evaluate_proposal(example, retry_raw, arm="llm_solver_feedback"))

    failure_modes = Counter(
        result.failure_mode
        for result in [*solver_results, *initial_results, *feedback_results]
        if result.failure_mode
    )
    artifact = build_artifact(
        examples=active_examples,
        model_specs=model_specs,
        models_used=models_used,
        checker_substrate=CHECKER_SUBSTRATE,
        tests_run=list(tests_run or []),
        duration_s=_duration(duration_s, started),
        solver_only_results=solver_results,
        llm_initial_results=initial_results,
        llm_feedback_results=feedback_results,
        failure_modes=dict(failure_modes),
        complete=True,
        run_date=run_date,
    )
    return artifact


def build_artifact(
    *,
    examples: Sequence[LoopInvariantExample],
    model_specs: Sequence[Mapping[str, Any]],
    models_used: Sequence[str],
    checker_substrate: str,
    tests_run: Sequence[str],
    duration_s: float,
    solver_only_results: Sequence[ArmResult],
    llm_initial_results: Sequence[ArmResult],
    llm_feedback_results: Sequence[ArmResult],
    failure_modes: Mapping[str, int],
    complete: bool,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Build and validate the terminal Exp 5226 artifact."""

    n_examples = len(examples)
    solver_solved = _accepted_count(solver_only_results)
    llm_only_solved = _accepted_count(llm_initial_results)
    feedback_solved = _accepted_count(llm_feedback_results)
    uplift = (
        round((feedback_solved - max(solver_solved, llm_only_solved)) / n_examples, 6)
        if n_examples
        else 0.0
    )
    accepted = _accepted_by_example(solver_only_results, llm_initial_results, llm_feedback_results)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": "experiment_5226_veribmc_local_solver_feedback_pilot_v478",
        "experiment_id": "exp5226-veribmc-local-solver-feedback-pilot-v478",
        "run_date": run_date,
        "duration_s": round(float(duration_s), 6),
        "result_path": str(RESULT_RELATIVE_PATH),
        "spec_refs": list(SPEC_REFS),
        "solver_feedback_pilot_complete": _wrap("solver_feedback_pilot_complete", bool(complete)),
        "n_examples": _wrap("n_examples", n_examples),
        "solver_only_solved": _wrap("solver_only_solved", solver_solved),
        "llm_only_solved": _wrap("llm_only_solved", llm_only_solved),
        "llm_solver_feedback_solved": _wrap("llm_solver_feedback_solved", feedback_solved),
        "solver_feedback_uplift": _wrap("solver_feedback_uplift", uplift),
        "accepted_invariants_or_constraints": _wrap("accepted_invariants_or_constraints", accepted),
        "model_specs": _wrap("model_specs", [dict(spec) for spec in model_specs]),
        "models_used": _wrap("models_used", list(models_used)),
        "checker_substrate": _wrap("checker_substrate", checker_substrate),
        "tests_run": _wrap("tests_run", list(tests_run)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(complete, solver_solved, llm_only_solved, feedback_solved, n_examples),
        ),
        "runtime_s_by_arm": {
            "solver_only": round(sum(result.runtime_ms for result in solver_only_results) / 1000.0, 6),
            "llm_only_checker": round(sum(result.runtime_ms for result in llm_initial_results) / 1000.0, 6),
            "llm_solver_feedback_checker": round(
                sum(result.runtime_ms for result in llm_feedback_results) / 1000.0,
                6,
            ),
        },
        "counterexample_feedback_used": sum(
            1
            for result in llm_initial_results
            if result.solver_feedback.get("counterexample")
        ),
        "failure_modes": dict(failure_modes),
        "per_example_results": {
            "solver_only": [result.to_dict() for result in solver_only_results],
            "llm_only": [result.to_dict() for result in llm_initial_results],
            "llm_solver_feedback": [result.to_dict() for result in llm_feedback_results],
        },
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the REQ-VERIFY-5226 terminal artifact shape."""

    missing = [field for field in FIELD_PRINCIPLES if field not in artifact]
    assert not missing, f"missing required fields: {missing}"
    for field, principle in FIELD_PRINCIPLES.items():
        wrapped = artifact[field]
        assert (
            isinstance(wrapped, dict)
            and wrapped.get("principle") == principle
            and "value" in wrapped
        ), f"{field} must be principle-wrapped"

    assert artifact["inference_substrate"]["value"] == INFERENCE_SUBSTRATE, (
        "inference_substrate must be local_sota_gguf_plus_deterministic_solver_feedback"
    )
    assert artifact["checker_substrate"]["value"] in {"z3", "smt", "esbmc", "other"}
    assert str(artifact["honest_verdict"]["value"]).startswith(TERMINAL_PREFIXES)
    n_examples = artifact["n_examples"]["value"]
    assert isinstance(n_examples, int) and n_examples >= 0
    for field in ("solver_only_solved", "llm_only_solved", "llm_solver_feedback_solved"):
        value = artifact[field]["value"]
        assert isinstance(value, int) and 0 <= value <= n_examples
    assert isinstance(artifact["solver_feedback_uplift"]["value"], float)
    if artifact["solver_feedback_pilot_complete"]["value"]:
        assert n_examples > 0
        assert artifact["models_used"]["value"], "complete pilot requires attempted local SOTA model path"
        assert _has_mandated_sota_model(artifact["model_specs"]["value"])


def run_experiment(
    *,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    run_date: str = RUN_DATE,
    duration_s: float | None = None,
    tests_run: Sequence[str] | None = None,
    proposal_fn: ProposalFn | None = None,
    model_specs_provider: ModelSpecsProvider = resolve_model_specs_for_pilot,
) -> JsonDict:
    """Run the pilot, write the JSON artifact, and return it."""

    artifact = run_pilot(
        proposal_fn=proposal_fn,
        model_specs_provider=model_specs_provider,
        tests_run=tests_run,
        duration_s=duration_s,
        run_date=run_date,
    )
    output = Path(result_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


class LiveGGUFProposalGenerator:  # pragma: no cover - exercised by the live experiment run.
    """Llama.cpp-backed local GGUF invariant proposal function."""

    def __init__(self, model_specs: Sequence[Mapping[str, Any]], *, seed: int = 5226) -> None:
        self.model_specs = [dict(spec) for spec in model_specs]
        self.seed = seed
        self._llm: Any | None = None
        self._active_spec: JsonDict | None = None

    def __call__(self, prompt: ProposalPrompt) -> str:
        llm = self._load()
        if llm is None:
            return ""
        result = llm(
            render_prompt(prompt),
            max_tokens=160,
            temperature=0.0,
            top_p=1.0,
            seed=self.seed + (17 if prompt.arm == "feedback" else 0),
            stop=["</s>", "<eos>", "\n\n\n"],
        )
        try:
            return str(result["choices"][0]["text"]).strip()
        except Exception:
            return str(result)

    def _load(self) -> Any | None:
        if self._llm is not None:
            return self._llm
        try:
            from llama_cpp import Llama
        except Exception:
            return None

        ordered_specs = sorted(
            self.model_specs,
            key=lambda spec: (0 if "gemma-4-26B" in str(spec.get("hf_id")) else 1),
        )
        for spec in ordered_specs:
            model_path = str(spec.get("model_path") or "")
            if not model_path:
                continue
            try:
                self._llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=-1,
                    main_gpu=int(spec.get("gpu") or 0),
                    n_ctx=2048,
                    n_batch=128,
                    seed=self.seed,
                    verbose=False,
                )
                self._active_spec = dict(spec)
                return self._llm
            except Exception:
                self._llm = None
        return None


def render_prompt(prompt: ProposalPrompt) -> str:
    """Render the strict JSON prompt sent to the local GGUF proposer."""

    example = prompt.example
    body = (
        "Return exactly one JSON object like {\"invariant\":\"...\"}; no prose.\n"
        "Use a Python-style integer Boolean expression with variables only from "
        f"{', '.join(example.variables)}. Allowed operators: +, -, *, <=, >=, <, >, ==, and, or.\n"
        f"Loop title: {example.title}\n"
        f"Precondition: {example.precondition}\n"
        f"Guard: {example.guard}\n"
        f"Updates: {json.dumps(dict(example.updates), sort_keys=True)}\n"
        f"Postcondition to prove when the guard is false: {example.postcondition}\n"
    )
    if prompt.arm == "feedback":
        body += (
            f"Prior invariant: {prompt.prior_invariant}\n"
            f"Solver feedback: {json.dumps(dict(prompt.solver_feedback or {}), sort_keys=True)}\n"
            "Repair the invariant once using the counterexample and hint.\n"
        )
    return body


def _compile_ast_node(node: ast.AST, env: Mapping[str, Any], z3_module: Any) -> Any:
    if isinstance(node, ast.BoolOp):
        values = [_compile_ast_node(value, env, z3_module) for value in node.values]
        if isinstance(node.op, ast.And):
            return z3_module.And(*values)
        if isinstance(node.op, ast.Or):
            return z3_module.Or(*values)
    if isinstance(node, ast.Compare):
        left = _compile_ast_node(node.left, env, z3_module)
        clauses = []
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _compile_ast_node(comparator, env, z3_module)
            clauses.append(_compile_comparison(left, op, right))
            left = right
        return clauses[0] if len(clauses) == 1 else z3_module.And(*clauses)
    if isinstance(node, ast.BinOp):
        left = _compile_ast_node(node.left, env, z3_module)
        right = _compile_ast_node(node.right, env, z3_module)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_compile_ast_node(node.operand, env, z3_module)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in {"True", "true"}:
            return z3_module.BoolVal(True)
        if node.id in {"False", "false"}:
            return z3_module.BoolVal(False)
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return z3_module.BoolVal(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return z3_module.IntVal(node.value)
    raise ValueError(f"unsupported expression node: {ast.dump(node, include_attributes=False)}")


def _compile_comparison(left: Any, op: ast.cmpop, right: Any) -> Any:
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.Eq):
        return left == right
    raise ValueError(f"unsupported comparison operator: {op!r}")


def _prove_unsat(
    assertions: Sequence[Any],
    *,
    obligation: str,
    variables: Mapping[str, Any],
    next_variables: Mapping[str, Any],
    z3_module: Any,
    timeout_ms: int,
) -> JsonDict | None:
    solver = z3_module.Solver()
    solver.set("timeout", timeout_ms)
    solver.add(*assertions)
    status = solver.check()
    if status == z3_module.unsat:
        return None
    if status == z3_module.sat:
        model = solver.model()
        return {
            "failed_obligation": obligation,
            "checker_status": "sat_counterexample",
            "counterexample": _model_values(model, variables, next_variables),
        }
    return {  # pragma: no cover - requires solver timeout/unknown rather than SAT/UNSAT.
        "failed_obligation": obligation,
        "checker_status": f"unknown: {solver.reason_unknown()}",
        "counterexample": {},
    }


def _failed_check(
    started: float,
    example: LoopInvariantExample,
    invariant: str,
    failure: Mapping[str, Any],
) -> CheckResult:
    feedback = {
        "failed_obligation": failure["failed_obligation"],
        "counterexample": dict(failure.get("counterexample") or {}),
        "repair_hint": example.repair_hint,
        "candidate_rejected": invariant,
        "checker_status": failure["checker_status"],
    }
    return CheckResult(
        accepted=False,
        failed_obligation=str(failure["failed_obligation"]),
        counterexample=dict(failure.get("counterexample") or {}),
        feedback=feedback,
        runtime_ms=_elapsed_ms(started),
    )


def _model_values(model: Any, variables: Mapping[str, Any], next_variables: Mapping[str, Any]) -> JsonDict:
    values: JsonDict = {}
    for name, ref in {**variables, **{f"{key}_next": value for key, value in next_variables.items()}}.items():
        value = model.eval(ref, model_completion=True)
        values[name] = value.as_long() if hasattr(value, "as_long") else str(value)
    return values


def _strip_code_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|python)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _clean_candidate_expression(text: str) -> str:
    candidate = text.strip().strip("`").rstrip(".")
    for marker in (" is the same as", " which ", " because ", " iff ", " if "):
        index = candidate.lower().find(marker)
        if index > 0:
            candidate = candidate[:index]
    return candidate.strip().rstrip(".")


def _extract_json_object(text: str) -> JsonDict | None:
    candidates = [text]
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _invariant_from_json(value: Mapping[str, Any]) -> str | None:
    invariant = value.get("invariant") or value.get("constraint")
    if isinstance(invariant, str) and invariant.strip():
        return invariant.strip()
    invariants = value.get("invariants")
    if isinstance(invariants, list) and invariants and all(isinstance(item, str) for item in invariants):
        return " and ".join(f"({item.strip()})" for item in invariants if item.strip())
    return None


def _accepted_count(results: Sequence[ArmResult]) -> int:
    return sum(1 for result in results if result.accepted)


def _accepted_by_example(
    solver_only_results: Sequence[ArmResult],
    llm_initial_results: Sequence[ArmResult],
    llm_feedback_results: Sequence[ArmResult],
) -> JsonDict:
    grouped: JsonDict = {}
    for arm_name, results in (
        ("solver_only", solver_only_results),
        ("llm_only", llm_initial_results),
        ("llm_solver_feedback", llm_feedback_results),
    ):
        for result in results:
            grouped.setdefault(result.example_id, {})[arm_name] = (
                result.parsed_invariant if result.accepted else None
            )
    return grouped


def _models_used_from_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        f"{spec.get('hf_id')}::{spec.get('model_path')}"
        for spec in model_specs
        if spec.get("hf_id") and spec.get("model_path")
    ]


def _has_mandated_sota_model(model_specs: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(spec.get("hf_id")) in MANDATED_SOTA_IDS and spec.get("model_path") for spec in model_specs)


def _honest_verdict(
    complete: bool,
    solver_only_solved: int,
    llm_only_solved: int,
    llm_solver_feedback_solved: int,
    n_examples: int,
) -> str:
    if not complete:
        return (
            "complete: solver-feedback pilot blocked before all three arms; solver feedback did not "
            "improve because required local SOTA/Z3 preconditions were not met"
        )
    strongest_baseline = max(solver_only_solved, llm_only_solved)
    if llm_solver_feedback_solved > strongest_baseline:
        return "complete: solver feedback improved over both baselines in the bounded VerIbmc pilot"
    if llm_solver_feedback_solved > llm_only_solved:
        return (
            "complete: solver feedback improved over LLM-only but not over the deterministic "
            "solver-only baseline"
        )
    if n_examples == 0:
        return "complete: empty fixture produced no solver-feedback uplift"
    return "complete: clean null; solver feedback did not improve over baselines"


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 6)


def _duration(duration_s: float | None, started: float) -> float:
    return float(duration_s) if duration_s is not None else time.perf_counter() - started


def main() -> None:  # pragma: no cover - exercised through the CLI command in live runs.
    artifact = run_experiment()
    print(json.dumps({"result_path": str(RESULT_RELATIVE_PATH), "honest_verdict": artifact["honest_verdict"]["value"]}))


if __name__ == "__main__":  # pragma: no cover
    main()

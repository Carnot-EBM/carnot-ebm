"""Exp 1536 SATQuest-style CNF verifier benchmark.

Spec: REQ-BENCH-1536, SCENARIO-BENCH-1536.

The benchmark keeps SAT instances small enough that Carnot can always fall
back to exhaustive local solving when PySAT is unavailable.  That fallback is
deliberate: the model may propose answers and self-verifier decisions, but the
solver oracle remains the authority for labels, false accepts, and repair
metrics.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from carnot.eval import cctu_executable_constraint_microbenchmark as cctu

JsonDict = dict[str, Any]

RUN_DATE = "20260508"
MAX_EXHAUSTIVE_VARS = 6
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1536_satquest_cnf_verifier_benchmark.json")
DEFAULT_MANIFEST_PATH = Path("results/satquest_cnf_verifier_1536.jsonl")
FORMAT_ORDER: tuple[str, ...] = ("machine", "symbolic", "narrative")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "role": "headline_flagship_moe_cnf_reasoner",
        "gpu": 0,
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "role": "headline_flagship_dense_cnf_reasoner",
        "gpu": 1,
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma4-26B-A4B-it",
        "role": "headline_middle_moe_cnf_reasoner",
        "gpu": 1,
    },
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "milestone",
    "satquest_benchmark_ready",
    "model_specs",
    "live_sota_model_inference_used",
    "cnf_instances",
    "formats_tested",
    "solver_oracle_used",
    "solver_oracle_false_accepts",
    "baseline_accuracy",
    "energy_ranked_accuracy",
    "repair_hint_accuracy",
    "false_accept_rate",
    "benchmark_manifest_path",
    "focused_tests_passed",
    "honest_verdict",
)


@dataclass(frozen=True)
class OracleResult:
    """Local SAT solver label and optional satisfying assignment."""

    is_satisfiable: bool
    satisfying_assignment: tuple[bool, ...] | None
    checked_assignments: int
    backend: str

    @property
    def label(self) -> str:
        return "SAT" if self.is_satisfiable else "UNSAT"


@dataclass(frozen=True)
class CNFInstance:
    """One bounded CNF instance with a deterministic local oracle label."""

    instance_id: str
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]
    family: str
    oracle: OracleResult


@dataclass(frozen=True)
class PromptCase:
    """One prompt format for one CNF instance."""

    case_id: str
    instance: CNFInstance
    format_name: str
    prompt: str

    @property
    def oracle_label(self) -> str:
        return self.instance.oracle.label


@dataclass(frozen=True)
class CandidateAnswer:
    """A parsed SAT/UNSAT answer plus an optional SAT assignment certificate."""

    label: str | None
    assignment: tuple[bool, ...] | None = None


@dataclass(frozen=True)
class ParsedModelAnswer:
    """Structured answer extracted from a model completion."""

    parse_ok: bool
    baseline: CandidateAnswer
    model_declared_accept: bool | None
    candidates: tuple[CandidateAnswer, ...] = ()
    repair_hint: CandidateAnswer = CandidateAnswer(None)
    parse_error: str | None = None


ResolverFn = Callable[[str], str | None]
LlamaImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectModelOutputsFn = Callable[[JsonDict, list[PromptCase]], JsonDict]
CachedPairFn = Callable[..., list[JsonDict] | None]


RAW_INSTANCES: tuple[tuple[str, int, tuple[tuple[int, ...], ...], str], ...] = (
    (
        "cnf-1536-sat-chain",
        3,
        ((1,), (-1, 2), (-2, 3)),
        "unit_propagation_sat",
    ),
    (
        "cnf-1536-unsat-unit-clash",
        1,
        ((1,), (-1,)),
        "direct_contradiction_unsat",
    ),
    (
        "cnf-1536-sat-choice",
        4,
        ((1, -2), (2, 3), (-3, 4), (-1, 4)),
        "branching_sat",
    ),
    (
        "cnf-1536-unsat-two-var-cover",
        2,
        ((1, 2), (1, -2), (-1, 2), (-1, -2)),
        "truth_table_cover_unsat",
    ),
    (
        "cnf-1536-sat-negated-path",
        4,
        ((-1, 2), (-2, -3), (3, 4), (-4, 1)),
        "mixed_polarity_sat",
    ),
    (
        "cnf-1536-unsat-implication-cycle",
        2,
        ((-1, 2), (-2, 1), (1, 2), (-1, -2)),
        "xor_equivalence_clash_unsat",
    ),
)


def build_cnf_instances(run_date: str = RUN_DATE) -> list[CNFInstance]:
    """Return the fixed bounded SATQuest CNF suite for the run date."""

    del run_date
    return [
        CNFInstance(
            instance_id=instance_id,
            n_vars=n_vars,
            clauses=clauses,
            family=family,
            oracle=solve_cnf(n_vars, clauses),
        )
        for instance_id, n_vars, clauses, family in RAW_INSTANCES
    ]


def solve_cnf(n_vars: int, clauses: tuple[tuple[int, ...], ...]) -> OracleResult:
    """Prefer PySAT when installed; otherwise use the bounded exact fallback."""

    try:
        return solve_cnf_pysat(n_vars, clauses)
    except ModuleNotFoundError:
        return solve_cnf_exact(n_vars, clauses)
    except Exception:
        return solve_cnf_exact(n_vars, clauses)


def solve_cnf_pysat(n_vars: int, clauses: tuple[tuple[int, ...], ...]) -> OracleResult:
    """Solve a CNF with PySAT, preserving the same result shape as the fallback."""

    from pysat.solvers import Solver  # type: ignore[import-not-found]  # noqa: PLC0415

    with Solver(bootstrap_with=[list(clause) for clause in clauses]) as solver:
        sat = bool(solver.solve())
        if not sat:
            return OracleResult(False, None, 1, "pysat")
        model = solver.get_model() or []
    positive = {abs(lit): lit > 0 for lit in model}
    assignment = tuple(bool(positive.get(index, False)) for index in range(1, n_vars + 1))
    return OracleResult(True, assignment, 1, "pysat")


def solve_cnf_exact(
    n_vars: int,
    clauses: tuple[tuple[int, ...], ...],
    *,
    backend: str = "exact_exhaustive_fallback",
) -> OracleResult:
    """Exhaustively solve a small CNF and return the first satisfying assignment."""

    if n_vars > MAX_EXHAUSTIVE_VARS:
        raise ValueError("bounded exhaustive solver refuses CNFs above MAX_EXHAUSTIVE_VARS")

    checked = 0
    for assignment in itertools.product((False, True), repeat=n_vars):
        checked += 1
        candidate = tuple(bool(value) for value in assignment)
        if assignment_satisfies(clauses, candidate):
            return OracleResult(True, candidate, checked, backend)
    return OracleResult(False, None, checked, backend)


def assignment_satisfies(
    clauses: tuple[tuple[int, ...], ...],
    assignment: tuple[bool, ...],
) -> bool:
    """Return whether a Boolean assignment satisfies every CNF clause."""

    return all(any(_literal_value(literal, assignment) for literal in clause) for clause in clauses)


def build_prompt_cases(instances: Iterable[CNFInstance] | None = None) -> list[PromptCase]:
    """Emit machine, symbolic, and narrative prompt formats for each CNF."""

    selected = list(instances or build_cnf_instances())
    cases: list[PromptCase] = []
    for instance in selected:
        for format_name in FORMAT_ORDER:
            cases.append(
                PromptCase(
                    case_id=f"{instance.instance_id}-{format_name}",
                    instance=instance,
                    format_name=format_name,
                    prompt=_build_prompt(instance, format_name),
                )
            )
    return cases


def parse_model_answer(text: str) -> ParsedModelAnswer:
    """Extract the SATQuest JSON answer from a model response."""

    obj = cctu.extract_json_object(text)
    if obj is None:
        return ParsedModelAnswer(
            False,
            CandidateAnswer(None),
            None,
            parse_error="no_json_object",
        )
    baseline = _parse_candidate(obj)
    if baseline.label is None:
        return ParsedModelAnswer(
            False,
            baseline,
            _declared_accept(obj),
            parse_error="answer_not_sat_or_unsat",
        )
    candidates_raw = obj.get("candidate_answers")
    candidates = (
        tuple(
            candidate
            for candidate in (_parse_candidate(item) for item in candidates_raw)
            if candidate.label is not None
        )
        if isinstance(candidates_raw, list)
        else ()
    )
    repair_hint = _parse_candidate(obj.get("repair_hint_answer"))
    return ParsedModelAnswer(
        True,
        baseline,
        _declared_accept(obj),
        candidates=candidates,
        repair_hint=repair_hint,
    )


def gold_answer_for_prompt_case(case: PromptCase) -> str:
    """Return a solver-derived JSON answer for tests and manifest sanity checks."""

    oracle = case.instance.oracle
    if oracle.is_satisfiable and oracle.satisfying_assignment is not None:
        assignment = _assignment_dict(oracle.satisfying_assignment)
        payload: JsonDict = {
            "answer": "SAT",
            "assignment": assignment,
            "verifier": {"accept": True},
            "candidate_answers": [
                {"answer": "UNSAT"},
                {"answer": "SAT", "assignment": assignment},
            ],
            "repair_hint_answer": {"answer": "SAT", "assignment": assignment},
        }
    else:
        payload = {
            "answer": "UNSAT",
            "assignment": None,
            "verifier": {"accept": True},
            "candidate_answers": [
                {"answer": "SAT", "assignment": _assignment_dict(tuple(True for _ in range(case.instance.n_vars)))},
                {"answer": "UNSAT"},
            ],
            "repair_hint_answer": {"answer": "UNSAT"},
        }
    return json.dumps(payload, sort_keys=True)


def build_manifest_row(case: PromptCase, generation_row: JsonDict) -> JsonDict:
    """Join one raw model output with local solver-authoritative scoring."""

    output_text = str(generation_row.get("output_text") or "")
    parsed = parse_model_answer(output_text)
    baseline = _evaluate_candidate(case, parsed.baseline, parsed.parse_error)
    ranked_candidate = _energy_rank_candidate(
        case,
        (parsed.baseline, *parsed.candidates),
    )
    energy_ranked = _evaluate_candidate(case, ranked_candidate, None)
    repair_candidate = parsed.repair_hint if parsed.repair_hint.label else parsed.baseline
    repair_hint = _evaluate_candidate(case, repair_candidate, None)
    false_accept = parsed.model_declared_accept is True and not bool(baseline["correct"])

    return {
        "case_id": case.case_id,
        "instance_id": case.instance.instance_id,
        "format_name": case.format_name,
        "family": case.instance.family,
        "n_vars": case.instance.n_vars,
        "clauses": [list(clause) for clause in case.instance.clauses],
        "prompt": case.prompt,
        "model_hf_id": generation_row.get("model_hf_id"),
        "model_name": generation_row.get("model_name"),
        "generation_source": generation_row.get("generation_source"),
        "elapsed_seconds": generation_row.get("elapsed_seconds"),
        "blocker": generation_row.get("blocker"),
        "model_output": output_text,
        "solver_oracle": {
            "backend": case.instance.oracle.backend,
            "label": case.instance.oracle.label,
            "checked_assignments": case.instance.oracle.checked_assignments,
            "satisfying_assignment": (
                list(case.instance.oracle.satisfying_assignment)
                if case.instance.oracle.satisfying_assignment is not None
                else None
            ),
        },
        "parse_result": {
            "parse_ok": parsed.parse_ok,
            "parse_error": parsed.parse_error,
            "baseline_answer": parsed.baseline.label,
            "model_declared_accept": parsed.model_declared_accept,
        },
        "baseline": baseline,
        "energy_ranked": energy_ranked,
        "repair_hint": repair_hint,
        "verifier": {
            "self_declared_accept": parsed.model_declared_accept,
            "self_verifier_false_accept": false_accept,
        },
    }


def aggregate_manifest_metrics(rows: list[JsonDict]) -> JsonDict:
    """Compute baseline, energy-ranked, repair-hint, and false-accept rates."""

    if not rows:
        return {
            "baseline_accuracy": 0.0,
            "energy_ranked_accuracy": 0.0,
            "repair_hint_accuracy": 0.0,
            "solver_oracle_false_accepts": 0,
            "false_accept_rate": 0.0,
            "classification_counts": {},
            "ordinary_wrong_answers": 0,
        }
    total = len(rows)
    false_accepts = sum(bool(row["verifier"]["self_verifier_false_accept"]) for row in rows)
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["baseline"]["classification"])
        counts[key] = counts.get(key, 0) + 1
    baseline_correct = sum(bool(row["baseline"]["correct"]) for row in rows)
    return {
        "baseline_accuracy": round(baseline_correct / total, 6),
        "energy_ranked_accuracy": round(
            sum(bool(row["energy_ranked"]["correct"]) for row in rows) / total,
            6,
        ),
        "repair_hint_accuracy": round(
            sum(bool(row["repair_hint"]["correct"]) for row in rows) / total,
            6,
        ),
        "solver_oracle_false_accepts": false_accepts,
        "false_accept_rate": round(false_accepts / total, 6),
        "classification_counts": counts,
        "ordinary_wrong_answers": total - baseline_correct - false_accepts,
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable in-progress artifact required before execution."""

    payload: JsonDict = {
        "status": "in_progress",
        "milestone": run_date,
        "satquest_benchmark_ready": False,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "cnf_instances": 0,
        "formats_tested": [],
        "solver_oracle_used": "pending",
        "solver_oracle_false_accepts": 0,
        "baseline_accuracy": 0.0,
        "energy_ranked_accuracy": 0.0,
        "repair_hint_accuracy": 0.0,
        "false_accept_rate": 0.0,
        "benchmark_manifest_path": _display_path(DEFAULT_MANIFEST_PATH),
        "focused_tests_passed": False,
        "honest_verdict": "in_progress: SATQuest CNF verifier benchmark initialized",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_benchmark(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] = MANDATED_MODEL_SPECS,
    collect_model_outputs_fn: CollectModelOutputsFn | None = None,
    cached_pair_fn: CachedPairFn | None = None,
    gpu_probe_fn: Callable[[], JsonDict] | None = None,
    max_models: int = 1,
    focused_tests_passed: bool = False,
) -> JsonDict:
    """Run the SATQuest benchmark and persist the JSON artifact plus JSONL manifest."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, run_date=run_date)

    instances = build_cnf_instances(run_date=run_date)
    prompt_cases = build_prompt_cases(instances)
    specs, cached_pair_details, cached_pair_error = _resolve_headline_specs(
        [dict(spec) for spec in model_specs],
        cached_pair_fn,
    )
    collector = collect_model_outputs_fn or collect_live_model_outputs
    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []
    case_by_id = {case.case_id: case for case in prompt_cases}

    for index, spec in enumerate(specs):
        if index >= max_models:
            model_attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_name": spec.get("name"),
                    "model_used": False,
                    "blocker": "not_attempted_runtime_budget",
                }
            )
            continue
        collection = collector(spec, prompt_cases)
        summary = dict(collection.get("summary") or {})
        model_attempts.append(summary)
        for generation_row in collection.get("rows") or []:
            case = case_by_id.get(str(generation_row.get("case_id") or ""))
            if case is not None:
                rows.append(build_manifest_row(case, generation_row))

    _write_jsonl(manifest, rows)
    metrics = aggregate_manifest_metrics(rows)
    mandated = {str(spec["hf_id"]) for spec in MANDATED_MODEL_SPECS}
    live_used = any(
        row.get("generation_source") == "live_sota_llamacpp"
        and not row.get("blocker")
        and row.get("model_hf_id") in mandated
        for row in rows
    )
    models_used = [
        str(summary["hf_id"])
        for summary in model_attempts
        if summary.get("model_used") is True and summary.get("hf_id")
    ]
    solver_backends = sorted({instance.oracle.backend for instance in instances})
    ready = bool(rows) and bool(live_used) and len({row["format_name"] for row in rows}) == 3
    status = "complete" if ready else "blocked"
    artifact: JsonDict = {
        "status": status,
        "milestone": run_date,
        "schema_version": 1,
        "satquest_benchmark_ready": bool(ready),
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(live_used),
        "cnf_instances": len(instances),
        "formats_tested": sorted({case.format_name for case in prompt_cases}),
        "solver_oracle_used": solver_backends[0] if len(solver_backends) == 1 else f"mixed:{','.join(solver_backends)}",
        "solver_oracle_false_accepts": metrics["solver_oracle_false_accepts"],
        "baseline_accuracy": metrics["baseline_accuracy"],
        "energy_ranked_accuracy": metrics["energy_ranked_accuracy"],
        "repair_hint_accuracy": metrics["repair_hint_accuracy"],
        "false_accept_rate": metrics["false_accept_rate"],
        "benchmark_manifest_path": _display_path(manifest),
        "focused_tests_passed": bool(focused_tests_passed),
        "honest_verdict": (
            "complete: SATQuest CNF verifier benchmark ready with live local SOTA GGUF rows"
            if ready
            else "complete_blocked: SATQuest CNF verifier benchmark written but live SOTA rows unavailable"
        ),
        "model_attempts": model_attempts,
        "models_used": models_used,
        "gpu_probe": (gpu_probe_fn or probe_gpu)(),
        "blockers": _collect_blockers(model_attempts, cached_pair_error),
        "cached_sota_pair": cached_pair_details,
        "classification_counts": metrics["classification_counts"],
        "ordinary_wrong_answers": metrics["ordinary_wrong_answers"],
        "prompt_cases": len(prompt_cases),
    }
    _write_json(output, artifact)
    return artifact


def collect_live_model_outputs(
    spec: JsonDict,
    prompt_cases: list[PromptCase],
    *,
    resolver: ResolverFn | None = None,
    llama_importer: LlamaImporterFn | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:
    """Collect raw SATQuest answers from one mandated local GGUF model."""

    hf_id = str(spec.get("hf_id") or "")
    resolver_fn = resolver or _default_resolver
    model_path = spec.get("model_path") or resolver_fn(hf_id)
    if not model_path:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_used": False,
                "blocker": "model_not_cached",
            },
            "rows": [],
        }

    env_details = (env_preparer or cctu.prepare_llama_environment)()
    ok, llama_class, import_error = (llama_importer or cctu._default_llama_importer)()
    if not ok or llama_class is None:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": import_error or "llama_cpp_import_failed",
                "env_details": env_details,
            },
            "rows": [],
        }

    llm = None
    rows: list[JsonDict] = []
    load_start = time.monotonic()
    try:
        llm = llama_class(
            model_path=str(model_path),
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=3072,
            seed=1536,
            verbose=False,
        )
    except Exception as exc:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": str(model_path),
                "model_used": False,
                "blocker": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - load_start, 6),
                "env_details": env_details,
            },
            "rows": [],
        }

    try:
        for case in prompt_cases:
            started = time.monotonic()
            try:
                result = llm(
                    case.prompt,
                    max_tokens=192,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                output_text = cctu._completion_text(result)
                blocker = None if output_text.strip() else "empty_generation"
            except Exception as exc:
                output_text = ""
                blocker = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "case_id": case.case_id,
                    "instance_id": case.instance.instance_id,
                    "format_name": case.format_name,
                    "model_hf_id": hf_id,
                    "model_name": spec.get("name"),
                    "model_path": str(model_path),
                    "generation_source": "live_sota_llamacpp",
                    "output_text": output_text,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                    "blocker": blocker,
                }
            )
    finally:
        cctu._close_llama(llm)

    model_used = any(row.get("blocker") is None for row in rows)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": str(model_path),
            "model_used": model_used,
            "blocker": None if model_used else "no_usable_generations",
            "env_details": env_details,
        },
        "rows": rows,
    }


def probe_gpu() -> JsonDict:  # pragma: no cover - host-specific probe.
    """Return a compact GPU provenance snapshot for the artifact."""

    try:
        from scripts.experiment_template import (  # noqa: PLC0415
            _cuda_is_available,
            _detect_gpu_count_rocm_aware,
        )

        return {
            "cuda_available": _cuda_is_available(),
            "gpu_count": _detect_gpu_count_rocm_aware(),
        }
    except Exception as exc:
        return {"probe_error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for conductor and manual SATQuest runs."""

    args = list(sys.argv[1:] if argv is None else argv)
    max_models = int(os.getenv("CARNOT_SATQUEST_1536_MAX_MODELS", "1"))
    if "--all-models" in args:
        max_models = len(MANDATED_MODEL_SPECS)
    artifact = run_benchmark(max_models=max_models, focused_tests_passed=False)
    print(
        "[exp1536] "
        f"ready={artifact['satquest_benchmark_ready']} "
        f"cnf_instances={artifact['cnf_instances']} "
        f"formats={artifact['formats_tested']} "
        f"baseline={artifact['baseline_accuracy']} "
        f"energy_ranked={artifact['energy_ranked_accuracy']} "
        f"repair_hint={artifact['repair_hint_accuracy']} "
        f"false_accepts={artifact['solver_oracle_false_accepts']}"
    )
    return 0


def _evaluate_candidate(
    case: PromptCase,
    candidate: CandidateAnswer,
    parse_error: str | None,
) -> JsonDict:
    if parse_error:
        return {
            "answer": candidate.label,
            "correct": False,
            "classification": "parse_failure",
            "energy": _candidate_energy(case, candidate),
            "parse_error": parse_error,
        }
    if candidate.label is None:
        return {
            "answer": None,
            "correct": False,
            "classification": "no_answer",
            "energy": _candidate_energy(case, candidate),
            "parse_error": None,
        }
    if candidate.label != case.instance.oracle.label:
        return {
            "answer": candidate.label,
            "correct": False,
            "classification": "wrong_label",
            "energy": _candidate_energy(case, candidate),
            "parse_error": None,
        }
    if candidate.label == "SAT":
        if candidate.assignment is None:
            return {
                "answer": candidate.label,
                "correct": False,
                "classification": "missing_assignment",
                "energy": _candidate_energy(case, candidate),
                "parse_error": None,
            }
        if not assignment_satisfies(case.instance.clauses, candidate.assignment):
            return {
                "answer": candidate.label,
                "correct": False,
                "classification": "invalid_assignment",
                "energy": _candidate_energy(case, candidate),
                "parse_error": None,
            }
    return {
        "answer": candidate.label,
        "correct": True,
        "classification": "oracle_agreement",
        "energy": _candidate_energy(case, candidate),
        "parse_error": None,
    }


def _energy_rank_candidate(
    case: PromptCase,
    candidates: tuple[CandidateAnswer, ...],
) -> CandidateAnswer:
    if not candidates:
        return CandidateAnswer(None)
    return min(candidates, key=lambda candidate: _candidate_energy(case, candidate))


def _candidate_energy(case: PromptCase, candidate: CandidateAnswer) -> float:
    if candidate.label is None:
        return 99.0
    oracle_label = case.instance.oracle.label
    if candidate.label == "UNSAT":
        return 0.0 if oracle_label == "UNSAT" else 50.0
    if oracle_label == "UNSAT":
        return 50.0 + float(_clause_violations(case.instance.clauses, candidate.assignment))
    if candidate.assignment is None:
        return 25.0
    return float(_clause_violations(case.instance.clauses, candidate.assignment))


def _clause_violations(
    clauses: tuple[tuple[int, ...], ...],
    assignment: tuple[bool, ...] | None,
) -> int:
    if assignment is None:
        return len(clauses)
    return sum(not any(_literal_value(literal, assignment) for literal in clause) for clause in clauses)


def _literal_value(literal: int, assignment: tuple[bool, ...]) -> bool:
    var_index = abs(literal) - 1
    if var_index < 0 or var_index >= len(assignment):
        return False
    value = assignment[var_index]
    return not value if literal < 0 else value


def _parse_candidate(value: Any) -> CandidateAnswer:
    if isinstance(value, str):
        return CandidateAnswer(_normalise_label(value), None)
    if not isinstance(value, dict):
        return CandidateAnswer(None, None)
    return CandidateAnswer(
        _normalise_label(value.get("answer")),
        _parse_assignment(value.get("assignment")),
    )


def _normalise_label(value: Any) -> str | None:
    if isinstance(value, bool):
        return "SAT" if value else "UNSAT"
    if not isinstance(value, str):
        return None
    upper = value.strip().upper()
    if upper in {"SAT", "SATISFIABLE", "TRUE"}:
        return "SAT"
    if upper in {"UNSAT", "UNSATISFIABLE", "FALSE"}:
        return "UNSAT"
    return None


def _parse_assignment(value: Any) -> tuple[bool, ...] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not all(isinstance(item, bool) for item in value):
            return None
        return tuple(bool(item) for item in value)
    if isinstance(value, dict):
        entries: list[tuple[int, bool]] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key.startswith("x") or not isinstance(item, bool):
                return None
            suffix = key[1:]
            if not suffix.isdigit():
                return None
            entries.append((int(suffix), bool(item)))
        if not entries:
            return None
        max_index = max(index for index, _value in entries)
        assignment = [False] * max_index
        for index, item in entries:
            assignment[index - 1] = item
        return tuple(assignment)
    return None


def _declared_accept(obj: JsonDict) -> bool | None:
    verifier = obj.get("verifier")
    if isinstance(verifier, dict) and isinstance(verifier.get("accept"), bool):
        return bool(verifier["accept"])
    return None


def _assignment_dict(assignment: tuple[bool, ...]) -> JsonDict:
    return {f"x{index}": bool(value) for index, value in enumerate(assignment, start=1)}


def _build_prompt(instance: CNFInstance, format_name: str) -> str:
    schema = {
        "answer": "SAT or UNSAT",
        "assignment": {"x1": "boolean values required when answer is SAT"},
        "verifier": {"accept": "true only if your answer is solver-consistent"},
        "candidate_answers": [{"answer": "SAT or UNSAT", "assignment": {"x1": True}}],
        "repair_hint_answer": {"answer": "SAT or UNSAT"},
    }
    body = {
        "machine": _dimacs_text(instance),
        "symbolic": _symbolic_text(instance),
        "narrative": _narrative_text(instance),
    }[format_name]
    return (
        "Solve this bounded CNF SATQuest verifier case.\n"
        f"Case: {instance.instance_id}\n"
        f"Format: {format_name}\n"
        "Return exactly one JSON object and no prose.\n"
        "Use this schema:\n"
        f"{json.dumps(schema, sort_keys=True)}\n"
        "If SAT, include a full assignment for x1..xN. If UNSAT, assignment may be null.\n"
        "Include candidate_answers and a repair_hint_answer after checking your own answer.\n\n"
        f"{body}\n"
    )


def _dimacs_text(instance: CNFInstance) -> str:
    lines = [
        f"c {instance.instance_id}",
        f"p cnf {instance.n_vars} {len(instance.clauses)}",
    ]
    lines.extend(" ".join(str(literal) for literal in clause) + " 0" for clause in instance.clauses)
    return "\n".join(lines)


def _symbolic_text(instance: CNFInstance) -> str:
    clauses = [
        "(" + " OR ".join(_literal_symbol(literal) for literal in clause) + ")"
        for clause in instance.clauses
    ]
    return f"Variables: x1..x{instance.n_vars}\nFormula: " + " AND ".join(clauses)


def _narrative_text(instance: CNFInstance) -> str:
    lines = [f"A lab panel has binary switches x1 through x{instance.n_vars}."]
    for index, clause in enumerate(instance.clauses, start=1):
        choices = ", ".join(_literal_narrative(literal) for literal in clause)
        lines.append(f"Rule {index}: at least one of these statements must hold: {choices}.")
    lines.append("Decide whether all rules can be true at the same time.")
    return "\n".join(lines)


def _literal_symbol(literal: int) -> str:
    return f"not x{abs(literal)}" if literal < 0 else f"x{literal}"


def _literal_narrative(literal: int) -> str:
    state = "off" if literal < 0 else "on"
    return f"x{abs(literal)} is {state}"


def _resolve_headline_specs(
    specs: list[JsonDict],
    cached_pair_fn: CachedPairFn | None,
) -> tuple[list[JsonDict], list[JsonDict], str | None]:
    pair_details: list[JsonDict] = []
    pair_error: str | None = None
    try:
        if cached_pair_fn is None:  # pragma: no cover - real cache path is host-specific.
            from carnot.inference.sota_models import cached_sota_pair  # noqa: PLC0415

            pair = cached_sota_pair(gpu_indices=(0, 1))
        else:
            pair = cached_pair_fn(gpu_indices=(0, 1))
    except Exception as exc:  # pragma: no cover - host cache/import state.
        pair = None
        pair_error = f"{type(exc).__name__}: {exc}"
    if pair:
        pair_details = [dict(item) for item in pair]
        paths = {item.get("hf_id"): item.get("model_path") for item in pair if item.get("model_path")}
        for spec in specs:
            if spec.get("hf_id") in paths:
                spec["model_path"] = paths[spec.get("hf_id")]
    return specs, pair_details, pair_error


def _collect_blockers(model_attempts: list[JsonDict], cached_pair_error: str | None) -> list[str]:
    blockers: list[str] = []
    if cached_pair_error:
        blockers.append(f"cached_sota_pair_error:{cached_pair_error}")
    for attempt in model_attempts:
        blocker = attempt.get("blocker")
        if blocker and blocker != "not_attempted_runtime_budget" and str(blocker) not in blockers:
            blockers.append(str(blocker))
    return blockers


def _default_resolver(hf_id: str) -> str | None:  # pragma: no cover - thin external resolver.
    from carnot.inference.sota_models import resolve_cached_gguf  # noqa: PLC0415

    return resolve_cached_gguf(hf_id)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _display_path(path: Path | str) -> str:
    as_path = Path(path)
    try:
        return str(as_path.resolve().relative_to(_repo_root()))
    except ValueError:
        return str(as_path)


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised by conductor/manual run.
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "FORMAT_ORDER",
    "MANDATED_MODEL_SPECS",
    "MAX_EXHAUSTIVE_VARS",
    "REQUIRED_ARTIFACT_FIELDS",
    "CNFInstance",
    "CandidateAnswer",
    "OracleResult",
    "ParsedModelAnswer",
    "PromptCase",
    "aggregate_manifest_metrics",
    "assignment_satisfies",
    "build_cnf_instances",
    "build_manifest_row",
    "build_prompt_cases",
    "collect_live_model_outputs",
    "gold_answer_for_prompt_case",
    "main",
    "parse_model_answer",
    "run_benchmark",
    "solve_cnf_exact",
    "solve_cnf_pysat",
    "write_in_progress_artifact",
]

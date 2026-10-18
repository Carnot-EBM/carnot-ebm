"""Exp 2919 ConstraintBench mini direct-optimization benchmark.

Spec: REQ-BENCH-2919, SCENARIO-BENCH-2919.

This module keeps the benchmark deliberately tiny and exact.  The model is
asked to output a structured JSON solution for each direct optimization task,
then Carnot verifies that answer by exhaustive enumeration rather than by
trusting the model's self-verification.  That separation matters because the
ConstraintBench failure mode is usually feasibility first and objective quality
second: an answer can be valid JSON and even near a good objective while still
violating a hard constraint.
"""

from __future__ import annotations

import itertools
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
RUN_DATE = "20260523"
RANDOM_SEED = 2919
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2919_constraintbench_mini_direct_optimization_v1.json"
MANIFEST_FILENAME = "constraintbench_mini_direct_optimization_2919_manifest.json"
INFERENCE_SUBSTRATE = "live_llm_inference_plus_exact_verifier"
MANDATED_MODEL_IDS: tuple[str, ...] = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "gpu": 0,
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "gpu": 0,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "gpu": 0,
    },
)
_SPEC_BY_HF_ID = {spec["hf_id"]: spec for spec in MANDATED_MODEL_SPECS}
EXACT_VERIFIER_TYPES: tuple[str, ...] = (
    "bounded_integer_exhaustive",
    "binary_subset_exhaustive",
    "color_assignment_exhaustive",
)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "constraintbench_mini_ready",
    "model_specs",
    "models_used",
    "cached_sota_pair_used",
    "random_seed",
    "task_manifest_path",
    "n_tasks",
    "exact_verifier_types",
    "feasibility_rate",
    "optimality_rate",
    "syntax_valid_rate",
    "violation_classes",
    "per_task_results",
    "inference_substrate",
    "duration_s",
    "run_date",
)

CachedPairProvider = Callable[..., list[JsonDict] | None]
IndividualResolver = Callable[[str], str | None]
CollectModelOutputs = Callable[[JsonDict, list["OptimizationTask"]], JsonDict]
LlamaImporter = Callable[[], tuple[bool, type[Any] | None, str | None]]


@dataclass(frozen=True)
class OptimizationTask:
    """One bounded direct-optimization task with a prompt and exact verifier."""

    task_id: str
    task_type: str
    exact_verifier_type: str
    objective_sense: str
    payload: JsonDict
    prompt: str


@dataclass(frozen=True)
class SolverResult:
    """Exact exhaustive solution set summary for one task."""

    feasible_count: int
    optimal_solution: JsonDict
    optimum_value: int
    feasible_solutions: tuple[tuple[JsonDict, int], ...]


@dataclass(frozen=True)
class ParsedSolution:
    """Structured JSON solution parsed from a model completion."""

    syntax_valid: bool
    solution: JsonDict | None
    parse_error: str | None


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime knobs for writing the Exp 2919 manifest and artifact."""

    output_path: Path | None = None
    manifest_path: Path | None = None
    max_models: int = 1
    random_seed: int = RANDOM_SEED
    tests_run: Sequence[str] = ()
    started_at: float | None = None
    clock: Callable[[], float] = time.time

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or REPO_ROOT / "results" / OUTPUT_FILENAME

    def task_manifest_path(self) -> Path:
        return self.manifest_path or REPO_ROOT / "results" / MANIFEST_FILENAME


def build_task_manifest() -> list[OptimizationTask]:
    """Return the fixed 18-task mini benchmark in deterministic family order."""

    linear = _linear_tasks()
    knapsack = _knapsack_tasks()
    graph = _graph_tasks()
    tasks: list[OptimizationTask] = []
    for index in range(6):
        tasks.extend([linear[index], knapsack[index], graph[index]])
    return [_with_prompt(task) for task in tasks]


def write_task_manifest(tasks: Sequence[OptimizationTask], path: Path | str) -> JsonDict:
    """Write the task manifest with exact optima so the verifier is auditable."""

    payload = {
        "schema": "carnot.constraintbench_mini_direct_optimization.v1",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "n_tasks": len(tasks),
        "exact_verifier_types": list(EXACT_VERIFIER_TYPES),
        "tasks": [_task_to_manifest_row(task) for task in tasks],
    }
    _write_json(Path(path), payload)
    return payload


def solve_task(task: OptimizationTask) -> SolverResult:
    """Solve one bounded task by exhaustive enumeration."""

    if task.task_type == "linear_integer":
        return _solve_linear(task)
    if task.task_type == "knapsack_binary":
        return _solve_knapsack(task)
    if task.task_type == "graph_coloring":
        return _solve_graph_coloring(task)
    raise ValueError(f"unknown task type: {task.task_type}")


def compliant_answer_for_task(task: OptimizationTask) -> str:
    """Return the exact optimal solution in the model's expected JSON shape."""

    return json.dumps({"solution": solve_task(task).optimal_solution}, sort_keys=True)


def suboptimal_answer_for_task(task: OptimizationTask) -> JsonDict:
    """Return a feasible but non-optimal answer for verifier tests."""

    solved = solve_task(task)
    for solution, value in solved.feasible_solutions:
        if value != solved.optimum_value:
            return solution
    return solved.optimal_solution


def infeasible_answer_for_task(task: OptimizationTask) -> JsonDict:
    """Return a syntactically valid answer that violates at least one hard constraint."""

    if task.task_type == "linear_integer":
        variables = task.payload["variables"]
        first = str(next(iter(variables)))
        return {
            "variables": {name: int(bounds[0]) for name, bounds in variables.items()}
            | {first: int(variables[first][1]) + 1}
        }
    if task.task_type == "knapsack_binary":
        return {"selected_items": [item["name"] for item in task.payload["items"]]}
    if task.task_type == "graph_coloring":
        return {"colors": {str(node): 0 for node in range(int(task.payload["n_nodes"]))}}
    raise ValueError(f"unknown task type: {task.task_type}")


def parse_model_output(task: OptimizationTask, text: str) -> ParsedSolution:
    """Extract and schema-check the first JSON object from a model completion."""

    obj, error = _extract_json_object(text)
    if error is not None:
        return ParsedSolution(False, None, error)
    if "solution" in obj and not isinstance(obj["solution"], dict):
        return ParsedSolution(False, None, "solution_not_object")
    solution = obj["solution"] if "solution" in obj else obj
    if task.task_type == "linear_integer":
        return _parse_linear_solution(task, solution)
    if task.task_type == "knapsack_binary":
        return _parse_knapsack_solution(solution)
    if task.task_type == "graph_coloring":
        return _parse_graph_solution(task, solution)
    return ParsedSolution(False, None, "unknown_task_type")


def evaluate_model_output(task: OptimizationTask, text: str) -> JsonDict:
    """Parse, verify feasibility, compute objective, and compare to optimum."""

    parsed = parse_model_output(task, text)
    if not parsed.syntax_valid or parsed.solution is None:
        return _evaluation_row(
            task,
            syntax_valid=False,
            feasible=False,
            optimal=False,
            objective_value=None,
            violation_class="syntax_invalid",
            violation_reasons=(parsed.parse_error or "parse_error",),
        )

    feasible, reasons = _check_feasibility(task, parsed.solution)
    objective_value = _objective_value(task, parsed.solution) if feasible else None
    optimum = solve_task(task).optimum_value
    optimal = bool(feasible and objective_value == optimum)
    if optimal:
        violation_class = "none"
    elif feasible:
        violation_class = "suboptimal"
    else:
        violation_class = "infeasible"
    return _evaluation_row(
        task,
        syntax_valid=True,
        feasible=feasible,
        optimal=optimal,
        objective_value=objective_value,
        violation_class=violation_class,
        violation_reasons=reasons,
    )


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute separate syntax, feasibility, and optimality rates over attempts."""

    if not rows:
        return {
            "syntax_valid_rate": 0.0,
            "feasibility_rate": 0.0,
            "optimality_rate": 0.0,
            "violation_classes": {},
        }
    total = len(rows)
    counts = Counter(str(row["violation_class"]) for row in rows)
    return {
        "syntax_valid_rate": round(sum(bool(row["syntax_valid"]) for row in rows) / total, 6),
        "feasibility_rate": round(sum(bool(row["feasible"]) for row in rows) / total, 6),
        "optimality_rate": round(sum(bool(row["optimal"]) for row in rows) / total, 6),
        "violation_classes": dict(sorted(counts.items())),
    }


def resolve_model_specs(
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
) -> tuple[list[JsonDict], bool, str | None]:
    """Call cached_sota_pair first, then fall back only to mandated GGUFs."""

    cache_error = None
    try:
        pair = cached_pair_provider(gpu_indices=(0, 1))
        if pair:
            return (
                [dict(spec) for spec in pair if spec.get("hf_id") in MANDATED_MODEL_IDS],
                True,
                None,
            )
    except Exception as exc:
        cache_error = f"{type(exc).__name__}: {exc}"

    specs: list[JsonDict] = []
    for hf_id in MANDATED_MODEL_IDS:
        path = individual_model_resolver(hf_id)
        if path:
            base = dict(_SPEC_BY_HF_ID[hf_id])
            base["model_path"] = str(path)
            specs.append(base)
    return specs, False, cache_error


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    cached_pair_provider: CachedPairProvider = cached_sota_pair,
    individual_model_resolver: IndividualResolver = resolve_cached_gguf,
    collect_model_outputs_fn: CollectModelOutputs = None,
) -> JsonDict:
    """Run the mini benchmark and write the terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    tasks = build_task_manifest()
    manifest_path = active.task_manifest_path()
    write_task_manifest(tasks, manifest_path)
    specs, cached_pair_used, cache_error = resolve_model_specs(
        cached_pair_provider=cached_pair_provider,
        individual_model_resolver=individual_model_resolver,
    )
    collector = collect_model_outputs_fn or collect_live_model_outputs
    if not specs:
        artifact = _base_artifact(
            active,
            started,
            manifest_path,
            tasks,
            _blocked_model_specs(),
            cached_pair_used=False,
            models_used=[],
            per_task_results=[],
            model_attempts=[],
            honest_verdict="blocked_sota_gguf_cache_missing",
            constraintbench_mini_ready=False,
            cached_pair_error=cache_error,
        )
        _write_json(active.artifact_path(), artifact)
        return artifact

    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []
    task_by_id = {task.task_id: task for task in tasks}
    for index, spec in enumerate(specs):
        if index >= active.max_models:
            model_attempts.append(
                {
                    "hf_id": spec.get("hf_id"),
                    "model_name": spec.get("name"),
                    "model_used": False,
                    "blocker": "not_attempted_runtime_budget",
                }
            )
            continue
        collection = collector(spec, tasks)
        summary = dict(collection.get("summary") or {})
        model_attempts.append(summary)
        for generation_row in collection.get("rows") or []:
            task = task_by_id.get(generation_row.get("task_id"))
            if task is None:
                continue
            evaluated = evaluate_model_output(task, str(generation_row.get("output_text") or ""))
            evaluated.update(
                {
                    "model_hf_id": generation_row.get("model_hf_id"),
                    "model_name": generation_row.get("model_name"),
                    "generation_source": generation_row.get("generation_source"),
                    "generation_blocker": generation_row.get("blocker"),
                    "elapsed_seconds": generation_row.get("elapsed_seconds"),
                }
            )
            rows.append(evaluated)

    metrics = aggregate_results(rows)
    models_used = [
        str(attempt["hf_id"])
        for attempt in model_attempts
        if attempt.get("model_used") is True and attempt.get("hf_id") in MANDATED_MODEL_IDS
    ]
    ready = bool(models_used) and bool(rows)
    verdict = (
        "complete: constraintbench mini direct optimization measured with live local SOTA GGUF"
        if ready
        else "blocked_sota_runtime_unavailable"
    )
    artifact = _base_artifact(
        active,
        started,
        manifest_path,
        tasks,
        specs,
        cached_pair_used=cached_pair_used,
        models_used=models_used,
        per_task_results=rows,
        model_attempts=model_attempts,
        honest_verdict=verdict,
        constraintbench_mini_ready=ready,
        cached_pair_error=cache_error,
    )
    artifact.update(metrics)
    _write_json(active.artifact_path(), artifact)
    return artifact


def collect_live_model_outputs(
    spec: JsonDict,
    tasks: list[OptimizationTask],
    *,
    llama_importer: LlamaImporter | None = None,
) -> JsonDict:
    """Collect structured JSON answers from one local GGUF through llama.cpp."""

    hf_id = str(spec.get("hf_id") or "")
    model_path = str(spec.get("model_path") or "")
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
    ok, llama_class, import_error = (llama_importer or _default_llama_importer)()
    if not ok or llama_class is None:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": model_path,
                "model_used": False,
                "blocker": import_error or "llama_cpp_import_failed",
            },
            "rows": [],
        }

    load_started = time.monotonic()
    try:
        llm = llama_class(
            model_path=model_path,
            n_gpu_layers=-1,
            main_gpu=int(spec.get("gpu") or 0),
            n_ctx=4096,
            seed=RANDOM_SEED,
            verbose=False,
        )
    except Exception as exc:
        return {
            "summary": {
                "hf_id": hf_id,
                "model_name": spec.get("name"),
                "model_path": model_path,
                "model_used": False,
                "blocker": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.monotonic() - load_started, 6),
            },
            "rows": [],
        }

    rows: list[JsonDict] = []
    try:
        for task in tasks:
            started = time.monotonic()
            try:
                result = llm(
                    task.prompt,
                    max_tokens=256,
                    temperature=0.0,
                    top_p=1.0,
                    stop=["</s>", "<eos>"],
                    echo=False,
                )
                output_text = _completion_text(result)
                blocker = None if output_text.strip() else "empty_generation"
            except Exception as exc:
                output_text = ""
                blocker = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "task_id": task.task_id,
                    "model_hf_id": hf_id,
                    "model_name": spec.get("name"),
                    "model_path": model_path,
                    "generation_source": "live_sota_llamacpp",
                    "output_text": output_text,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                    "blocker": blocker,
                }
            )
    finally:
        _close_llama(llm)

    model_used = any(not row.get("blocker") for row in rows)
    return {
        "summary": {
            "hf_id": hf_id,
            "model_name": spec.get("name"),
            "model_path": model_path,
            "model_used": model_used,
            "blocker": None if model_used else "no_usable_generations",
        },
        "rows": rows,
    }


def _linear_tasks() -> list[OptimizationTask]:
    return [
        _task(
            "cbmini-2919-linear-001",
            "linear_integer",
            "bounded_integer_exhaustive",
            "max",
            {
                "variables": {"x": [0, 4], "y": [0, 4]},
                "objective": {"x": 3, "y": 2},
                "constraints": [
                    {"coefficients": {"x": 1, "y": 2}, "op": "<=", "rhs": 6},
                    {"coefficients": {"x": 2, "y": 1}, "op": "<=", "rhs": 7},
                ],
            },
        ),
        _task(
            "cbmini-2919-linear-002",
            "linear_integer",
            "bounded_integer_exhaustive",
            "min",
            {
                "variables": {"x": [0, 5], "y": [0, 5]},
                "objective": {"x": 2, "y": 3},
                "constraints": [
                    {"coefficients": {"x": 1, "y": 1}, "op": ">=", "rhs": 4},
                    {"coefficients": {"x": 1}, "op": ">=", "rhs": 1},
                ],
            },
        ),
        _task(
            "cbmini-2919-linear-003",
            "linear_integer",
            "bounded_integer_exhaustive",
            "max",
            {
                "variables": {"a": [0, 3], "b": [0, 3], "c": [0, 2]},
                "objective": {"a": 4, "b": 3, "c": 5},
                "constraints": [
                    {"coefficients": {"a": 2, "b": 1, "c": 2}, "op": "<=", "rhs": 8},
                    {"coefficients": {"a": 1, "b": -1}, "op": ">=", "rhs": 0},
                ],
            },
        ),
        _task(
            "cbmini-2919-linear-004",
            "linear_integer",
            "bounded_integer_exhaustive",
            "min",
            {
                "variables": {"p": [0, 4], "q": [0, 4]},
                "objective": {"p": 5, "q": 1},
                "constraints": [
                    {"coefficients": {"p": 2, "q": 1}, "op": ">=", "rhs": 5},
                    {"coefficients": {"q": 1}, "op": ">=", "rhs": 1},
                ],
            },
        ),
        _task(
            "cbmini-2919-linear-005",
            "linear_integer",
            "bounded_integer_exhaustive",
            "max",
            {
                "variables": {"u": [0, 5], "v": [0, 3]},
                "objective": {"u": 1, "v": 6},
                "constraints": [
                    {"coefficients": {"u": 1, "v": 1}, "op": "<=", "rhs": 6},
                    {"coefficients": {"u": 1, "v": -1}, "op": ">=", "rhs": 1},
                ],
            },
        ),
        _task(
            "cbmini-2919-linear-006",
            "linear_integer",
            "bounded_integer_exhaustive",
            "min",
            {
                "variables": {"r": [0, 4], "s": [0, 4], "t": [0, 3]},
                "objective": {"r": 2, "s": 2, "t": 4},
                "constraints": [
                    {"coefficients": {"r": 1, "s": 1, "t": 1}, "op": ">=", "rhs": 5},
                    {"coefficients": {"r": 1, "s": -1}, "op": "<=", "rhs": 1},
                ],
            },
        ),
    ]


def _knapsack_tasks() -> list[OptimizationTask]:
    raw = [
        (
            "cbmini-2919-knapsack-001",
            7,
            [("map", 2, 5), ("rope", 3, 6), ("lamp", 4, 7), ("kit", 1, 3)],
            ["kit"],
            [("lamp", "rope")],
        ),
        (
            "cbmini-2919-knapsack-002",
            9,
            [("cpu", 3, 8), ("gpu", 5, 11), ("ssd", 2, 5), ("wifi", 1, 2)],
            [],
            [("gpu", "wifi")],
        ),
        (
            "cbmini-2919-knapsack-003",
            8,
            [("sensor", 2, 4), ("radio", 3, 7), ("battery", 4, 8), ("case", 1, 1)],
            ["case"],
            [("radio", "battery")],
        ),
        (
            "cbmini-2919-knapsack-004",
            10,
            [("atlas", 4, 8), ("guide", 2, 3), ("water", 3, 6), ("med", 2, 5)],
            ["water"],
            [("atlas", "guide")],
        ),
        (
            "cbmini-2919-knapsack-005",
            6,
            [("red", 2, 5), ("blue", 2, 4), ("green", 3, 7), ("white", 1, 2)],
            [],
            [("red", "green")],
        ),
        (
            "cbmini-2919-knapsack-006",
            11,
            [("forge", 5, 9), ("drill", 4, 8), ("meter", 2, 4), ("guard", 3, 6)],
            ["meter"],
            [("forge", "drill")],
        ),
    ]
    return [
        _task(
            task_id,
            "knapsack_binary",
            "binary_subset_exhaustive",
            "max",
            {
                "capacity": capacity,
                "items": [
                    {"name": name, "weight": weight, "value": value}
                    for name, weight, value in items
                ],
                "required_items": required,
                "excludes": [[left, right] for left, right in excludes],
            },
        )
        for task_id, capacity, items, required, excludes in raw
    ]


def _graph_tasks() -> list[OptimizationTask]:
    raw = [
        ("cbmini-2919-graph-001", 4, 3, [(0, 1), (1, 2), (2, 3)]),
        ("cbmini-2919-graph-002", 4, 3, [(0, 1), (1, 2), (2, 0), (2, 3)]),
        ("cbmini-2919-graph-003", 5, 3, [(0, 1), (0, 2), (0, 3), (0, 4)]),
        ("cbmini-2919-graph-004", 5, 4, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
        ("cbmini-2919-graph-005", 4, 4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]),
        ("cbmini-2919-graph-006", 6, 3, [(0, 1), (1, 2), (3, 4), (4, 5), (2, 5)]),
    ]
    return [
        _task(
            task_id,
            "graph_coloring",
            "color_assignment_exhaustive",
            "min",
            {
                "n_nodes": n_nodes,
                "n_colors": n_colors,
                "edges": [[left, right] for left, right in edges],
            },
        )
        for task_id, n_nodes, n_colors, edges in raw
    ]


def _task(
    task_id: str,
    task_type: str,
    verifier: str,
    objective_sense: str,
    payload: JsonDict,
) -> OptimizationTask:
    return OptimizationTask(task_id, task_type, verifier, objective_sense, payload, "")


def _with_prompt(task: OptimizationTask) -> OptimizationTask:
    return OptimizationTask(
        task.task_id,
        task.task_type,
        task.exact_verifier_type,
        task.objective_sense,
        task.payload,
        _build_prompt(task),
    )


def _build_prompt(task: OptimizationTask) -> str:
    schema = _solution_schema(task)
    return (
        "Solve this direct constrained optimization task.\n"
        f"Task id: {task.task_id}\n"
        f"Task type: {task.task_type}\n"
        f"Objective sense: {task.objective_sense}\n"
        "Return exactly one JSON object and no prose.\n"
        f"The JSON object must follow this shape: {json.dumps({'solution': schema}, sort_keys=True)}\n"
        f"Task data: {json.dumps(task.payload, sort_keys=True)}\n"
    )


def _solution_schema(task: OptimizationTask) -> JsonDict:
    if task.task_type == "linear_integer":
        return {"variables": {name: "<integer>" for name in task.payload["variables"]}}
    if task.task_type == "knapsack_binary":
        return {"selected_items": ["item names"]}
    if task.task_type == "graph_coloring":
        return {
            "colors": {str(node): "<integer color>" for node in range(int(task.payload["n_nodes"]))}
        }
    return {}


def _task_to_manifest_row(task: OptimizationTask) -> JsonDict:
    solved = solve_task(task)
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "exact_verifier_type": task.exact_verifier_type,
        "objective_sense": task.objective_sense,
        "payload": task.payload,
        "prompt": task.prompt,
        "feasible_count": solved.feasible_count,
        "optimum_value": solved.optimum_value,
        "optimal_solution": solved.optimal_solution,
    }


def _solve_linear(task: OptimizationTask) -> SolverResult:
    variables = task.payload["variables"]
    names = list(variables)
    feasible: list[tuple[JsonDict, int]] = []
    domains = [range(int(variables[name][0]), int(variables[name][1]) + 1) for name in names]
    for values in itertools.product(*domains):
        assignment = dict(zip(names, values, strict=True))
        solution = {"variables": assignment}
        ok, _reasons = _check_linear_feasibility(task, solution)
        if ok:
            feasible.append((solution, _linear_objective(task, assignment)))
    return _best_result(task, feasible)


def _solve_knapsack(task: OptimizationTask) -> SolverResult:
    items = [item["name"] for item in task.payload["items"]]
    feasible: list[tuple[JsonDict, int]] = []
    for mask in itertools.product([False, True], repeat=len(items)):
        selected = [item for item, keep in zip(items, mask, strict=True) if keep]
        solution = {"selected_items": selected}
        ok, _reasons = _check_knapsack_feasibility(task, solution)
        if ok:
            feasible.append((solution, _knapsack_objective(task, selected)))
    return _best_result(task, feasible)


def _solve_graph_coloring(task: OptimizationTask) -> SolverResult:
    n_nodes = int(task.payload["n_nodes"])
    n_colors = int(task.payload["n_colors"])
    feasible: list[tuple[JsonDict, int]] = []
    for colors in itertools.product(range(n_colors), repeat=n_nodes):
        solution = {"colors": {str(node): color for node, color in enumerate(colors)}}
        ok, _reasons = _check_graph_feasibility(task, solution)
        if ok:
            feasible.append((solution, sum(colors)))
    return _best_result(task, feasible)


def _best_result(task: OptimizationTask, feasible: list[tuple[JsonDict, int]]) -> SolverResult:
    if not feasible:
        return SolverResult(0, {}, 0, ())
    reverse = task.objective_sense == "max"
    ordered = sorted(
        feasible,
        key=lambda item: (item[1], json.dumps(item[0], sort_keys=True)),
        reverse=reverse,
    )
    best_solution, best_value = ordered[0]
    return SolverResult(len(feasible), best_solution, best_value, tuple(ordered))


def _extract_json_object(text: str) -> tuple[JsonDict, str | None]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj, None
    return {}, "no_json_object"


def _parse_linear_solution(task: OptimizationTask, solution: JsonDict) -> ParsedSolution:
    variables = solution.get("variables")
    if not isinstance(variables, dict):
        return ParsedSolution(False, None, "variables_not_object")
    parsed: dict[str, int] = {}
    for name in task.payload["variables"]:
        if name not in variables:
            return ParsedSolution(False, None, f"missing_variable:{name}")
        value = variables[name]
        if isinstance(value, bool) or not isinstance(value, int):
            return ParsedSolution(False, None, f"variable_not_integer:{name}")
        parsed[name] = value
    return ParsedSolution(True, {"variables": parsed}, None)


def _parse_knapsack_solution(solution: JsonDict) -> ParsedSolution:
    selected = solution.get("selected_items")
    if not isinstance(selected, list):
        return ParsedSolution(False, None, "selected_items_not_list")
    if not all(isinstance(item, str) for item in selected):
        return ParsedSolution(False, None, "selected_item_not_string")
    return ParsedSolution(True, {"selected_items": selected}, None)


def _parse_graph_solution(task: OptimizationTask, solution: JsonDict) -> ParsedSolution:
    colors = solution.get("colors")
    if not isinstance(colors, dict):
        return ParsedSolution(False, None, "colors_not_object")
    parsed: dict[str, int] = {}
    for node in range(int(task.payload["n_nodes"])):
        key = str(node)
        if key not in colors:
            return ParsedSolution(False, None, f"missing_color:{key}")
        value = colors[key]
        if isinstance(value, bool) or not isinstance(value, int):
            return ParsedSolution(False, None, f"color_not_integer:{key}")
        parsed[key] = value
    return ParsedSolution(True, {"colors": parsed}, None)


def _check_feasibility(task: OptimizationTask, solution: JsonDict) -> tuple[bool, tuple[str, ...]]:
    if task.task_type == "linear_integer":
        return _check_linear_feasibility(task, solution)
    if task.task_type == "knapsack_binary":
        return _check_knapsack_feasibility(task, solution)
    if task.task_type == "graph_coloring":
        return _check_graph_feasibility(task, solution)
    return False, ("unknown_task_type",)


def _check_linear_feasibility(
    task: OptimizationTask, solution: JsonDict
) -> tuple[bool, tuple[str, ...]]:
    variables = solution.get("variables", {})
    reasons: list[str] = []
    for name, bounds in task.payload["variables"].items():
        value = variables.get(name)
        if value is None:
            reasons.append(f"missing_variable:{name}")
            continue
        if value < int(bounds[0]) or value > int(bounds[1]):
            reasons.append(f"domain:{name}")
    for index, constraint in enumerate(task.payload["constraints"]):
        if not _constraint_holds(variables, constraint):
            reasons.append(f"linear_constraint:{index}")
    return not reasons, tuple(reasons)


def _constraint_holds(assignment: Mapping[str, int], constraint: Mapping[str, Any]) -> bool:
    lhs = sum(
        int(coef) * int(assignment.get(name, 0))
        for name, coef in constraint["coefficients"].items()
    )
    rhs = int(constraint["rhs"])
    op = constraint["op"]
    if op == "<=":
        return lhs <= rhs
    if op == ">=":
        return lhs >= rhs
    if op == "==":
        return lhs == rhs
    raise ValueError(f"unsupported operator: {op}")


def _check_knapsack_feasibility(
    task: OptimizationTask, solution: JsonDict
) -> tuple[bool, tuple[str, ...]]:
    selected = list(solution.get("selected_items", []))
    selected_set = set(selected)
    known = {item["name"] for item in task.payload["items"]}
    reasons: list[str] = []
    if len(selected) != len(selected_set):
        reasons.append("duplicate_item")
    for item in selected_set - known:
        reasons.append(f"unknown_item:{item}")
    for item in task.payload.get("required_items", []):
        if item not in selected_set:
            reasons.append(f"missing_required:{item}")
    if _knapsack_weight(task, selected_set) > int(task.payload["capacity"]):
        reasons.append("capacity_exceeded")
    for left, right in task.payload.get("excludes", []):
        if left in selected_set and right in selected_set:
            reasons.append(f"excludes:{left},{right}")
    return not reasons, tuple(reasons)


def _check_graph_feasibility(
    task: OptimizationTask, solution: JsonDict
) -> tuple[bool, tuple[str, ...]]:
    colors = solution.get("colors", {})
    n_colors = int(task.payload["n_colors"])
    reasons: list[str] = []
    for node in range(int(task.payload["n_nodes"])):
        key = str(node)
        color = colors.get(key)
        if color is None:
            reasons.append(f"missing_color:{key}")
        elif color < 0 or color >= n_colors:
            reasons.append(f"color_range:{key}")
    for left, right in task.payload["edges"]:
        if colors.get(str(left)) == colors.get(str(right)):
            reasons.append(f"edge_conflict:{left}-{right}")
    return not reasons, tuple(reasons)


def _objective_value(task: OptimizationTask, solution: JsonDict) -> int:
    if task.task_type == "linear_integer":
        return _linear_objective(task, solution["variables"])
    if task.task_type == "knapsack_binary":
        return _knapsack_objective(task, solution["selected_items"])
    return sum(solution["colors"][str(node)] for node in range(int(task.payload["n_nodes"])))


def _linear_objective(task: OptimizationTask, variables: Mapping[str, int]) -> int:
    return sum(int(coef) * int(variables[name]) for name, coef in task.payload["objective"].items())


def _knapsack_objective(task: OptimizationTask, selected: Iterable[str]) -> int:
    selected_set = set(selected)
    return sum(int(item["value"]) for item in task.payload["items"] if item["name"] in selected_set)


def _knapsack_weight(task: OptimizationTask, selected: Iterable[str]) -> int:
    selected_set = set(selected)
    return sum(
        int(item["weight"]) for item in task.payload["items"] if item["name"] in selected_set
    )


def _evaluation_row(
    task: OptimizationTask,
    *,
    syntax_valid: bool,
    feasible: bool,
    optimal: bool,
    objective_value: int | None,
    violation_class: str,
    violation_reasons: Sequence[str],
) -> JsonDict:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "exact_verifier_type": task.exact_verifier_type,
        "syntax_valid": bool(syntax_valid),
        "feasible": bool(feasible),
        "objective_value": objective_value,
        "optimum_value": solve_task(task).optimum_value,
        "optimal": bool(optimal),
        "violation_class": violation_class,
        "violation_reasons": list(violation_reasons),
    }


def _base_artifact(
    config: ExperimentConfig,
    started: float,
    manifest_path: Path,
    tasks: Sequence[OptimizationTask],
    model_specs: Sequence[JsonDict],
    *,
    cached_pair_used: bool,
    models_used: Sequence[str],
    per_task_results: Sequence[JsonDict],
    model_attempts: Sequence[JsonDict],
    honest_verdict: str,
    constraintbench_mini_ready: bool,
    cached_pair_error: str | None,
) -> JsonDict:
    metrics = aggregate_results(per_task_results)
    return {
        "artifact": "experiment_2919_constraintbench_mini_direct_optimization_v1",
        "schema": "carnot.constraintbench_mini_direct_optimization.v1",
        "honest_verdict": honest_verdict,
        "constraintbench_mini_ready": bool(constraintbench_mini_ready),
        "model_specs": list(model_specs),
        "models_used": list(models_used),
        "cached_sota_pair_used": bool(cached_pair_used),
        "cached_sota_pair_error": cached_pair_error,
        "random_seed": int(config.random_seed),
        "task_manifest_path": str(manifest_path),
        "n_tasks": len(tasks),
        "exact_verifier_types": list(EXACT_VERIFIER_TYPES),
        "syntax_valid_rate": metrics["syntax_valid_rate"],
        "feasibility_rate": metrics["feasibility_rate"],
        "optimality_rate": metrics["optimality_rate"],
        "violation_classes": metrics["violation_classes"],
        "per_task_results": list(per_task_results),
        "model_attempts": list(model_attempts),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "tests_run": list(config.tests_run),
        "run_date": RUN_DATE,
        "duration_s": max(0.0, config.clock() - started),
    }


def _blocked_model_specs() -> list[JsonDict]:
    return [{"hf_id": hf_id, "available": False} for hf_id in MANDATED_MODEL_IDS]


def _completion_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return ""
    choices = result.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    if isinstance(first.get("text"), str):
        return first["text"]
    message = first.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def _close_llama(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def _default_llama_importer() -> tuple[
    bool, type[Any] | None, str | None
]:  # pragma: no cover - host runtime.
    try:
        from llama_cpp import Llama  # type: ignore[import]  # noqa: PLC0415
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(
        "[exp2919] "
        f"ready={artifact['constraintbench_mini_ready']} "
        f"models={artifact['models_used']} "
        f"syntax={artifact['syntax_valid_rate']} "
        f"feasible={artifact['feasibility_rate']} "
        f"optimal={artifact['optimality_rate']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())

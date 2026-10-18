"""Exp 1486 CCTU-style executable constraint micro-benchmark.

Spec: REQ-VERIFY-1486, SCENARIO-VERIFY-1486.

The benchmark is intentionally local and deterministic.  The model is asked to
emit one JSON transcript containing a tool call, the claimed local tool result,
the final answer, and a verifier accept/reject decision.  Carnot then executes
the same tool call locally and checks the transcript instead of trusting the
model's self-report.
"""

from __future__ import annotations

import heapq
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

JsonDict = dict[str, Any]

RUN_DATE = "20260507"
BENCHMARK_CASE_COUNT = 20
VALIDATORS_PATH = "python/carnot/eval/cctu_executable_constraint_microbenchmark.py"
DEFAULT_ARTIFACT_PATH = Path(
    "results/experiment_1486_cctu_executable_constraint_microbenchmark.json"
)
DEFAULT_MANIFEST_PATH = Path("results/cctu_microbenchmark_manifest_1486.jsonl")

MANDATED_MODEL_SPECS: tuple[JsonDict, ...] = (
    {
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "name": "Qwen3.6-35B-A3B",
        "role": "flagship_moe_primary_tool_use_model",
        "gpu": 0,
    },
    {
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "name": "Gemma4-31B-it",
        "role": "flagship_dense_secondary_tool_use_model",
        "gpu": 1,
    },
    {
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "name": "Gemma4-26B-A4B-it",
        "role": "middle_moe_secondary_tool_use_model",
        "gpu": 1,
    },
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "model_specs",
    "live_sota_model_inference_used",
    "executable_constraint_benchmark_ready",
    "benchmark_cases",
    "validators_path",
    "manifest_path",
    "tool_call_validity_rate",
    "final_answer_validity_rate",
    "verifier_catch_rate",
    "verifier_false_accept_rate",
    "models_used",
    "tests_run",
    "honest_verdict",
)


@dataclass(frozen=True)
class BenchmarkCase:
    """One deterministic tool-use case.

    The expected result is stored next to the prompt so every manifest row can
    be audited without rerunning the model.  The value is still recomputed by
    ``validate_transcript`` from the declared tool call to catch inconsistent
    model-supplied tool results.
    """

    case_id: str
    family: str
    tool_name: str
    tool_arguments: JsonDict
    expected_tool_result: Any
    expected_final_answer: str
    prompt: str


ResolverFn = Callable[[str], str | None]
LlamaImporterFn = Callable[[], tuple[bool, type[Any] | None, str | None]]
CollectModelOutputsFn = Callable[[JsonDict, list[BenchmarkCase]], JsonDict]


def build_benchmark_cases() -> list[BenchmarkCase]:
    """Return the fixed 20-case executable constraint suite."""

    raw_cases: list[tuple[str, str, str, JsonDict]] = [
        (
            "cctu-1486-arith-001",
            "arithmetic",
            "arithmetic.evaluate",
            {"operation": "sum", "numbers": [12, 18, 15]},
        ),
        (
            "cctu-1486-arith-002",
            "arithmetic",
            "arithmetic.evaluate",
            {"operation": "difference", "start": 81, "subtract": [9, 14, 8]},
        ),
        (
            "cctu-1486-arith-003",
            "arithmetic",
            "arithmetic.evaluate",
            {"operation": "product_plus", "numbers": [4, 7], "add": 6},
        ),
        (
            "cctu-1486-arith-004",
            "arithmetic",
            "arithmetic.evaluate",
            {
                "operation": "weighted_total",
                "items": {"alpha": 3, "beta": 2, "gamma": 5},
                "weights": {"alpha": 4, "beta": 7, "gamma": 1},
            },
        ),
        (
            "cctu-1486-arith-005",
            "arithmetic",
            "arithmetic.evaluate",
            {"operation": "mod", "value": 137, "modulus": 11},
        ),
        (
            "cctu-1486-table-001",
            "table_filter",
            "table.filter_rows",
            {
                "rows": [
                    {"item": "bolt", "color": "red", "qty": 8},
                    {"item": "nut", "color": "blue", "qty": 12},
                    {"item": "gasket", "color": "red", "qty": 5},
                    {"item": "clip", "color": "red", "qty": 3},
                ],
                "where": {"color": "red", "qty_min": 5},
                "select": "item",
            },
        ),
        (
            "cctu-1486-table-002",
            "table_filter",
            "table.filter_rows",
            {
                "rows": [
                    {"name": "Ada", "region": "east", "score": 91},
                    {"name": "Ben", "region": "west", "score": 87},
                    {"name": "Cy", "region": "east", "score": 82},
                    {"name": "Di", "region": "east", "score": 76},
                ],
                "where": {"region": "east", "score_min": 80},
                "select": "name",
            },
        ),
        (
            "cctu-1486-table-003",
            "table_filter",
            "table.filter_rows",
            {
                "rows": [
                    {"ticket": "T1", "status": "open", "priority": 1},
                    {"ticket": "T2", "status": "closed", "priority": 2},
                    {"ticket": "T3", "status": "open", "priority": 3},
                    {"ticket": "T4", "status": "open", "priority": 2},
                ],
                "where": {"status": "open", "priority_max": 2},
                "select": "__count__",
            },
        ),
        (
            "cctu-1486-table-004",
            "table_filter",
            "table.filter_rows",
            {
                "rows": [
                    {"name": "Ira", "dept": "ops", "shift": "night-a"},
                    {"name": "Jo", "dept": "ops", "shift": "day"},
                    {"name": "Kai", "dept": "eng", "shift": "night-b"},
                    {"name": "Lux", "dept": "ops", "shift": "night-c"},
                ],
                "where": {"dept": "ops", "shift_contains": "night"},
                "select": "name",
            },
        ),
        (
            "cctu-1486-table-005",
            "table_filter",
            "table.filter_rows",
            {
                "rows": [
                    {"city": "Reno", "state": "NV", "temp": 61},
                    {"city": "Austin", "state": "TX", "temp": 73},
                    {"city": "Dallas", "state": "TX", "temp": 69},
                    {"city": "Miami", "state": "FL", "temp": 80},
                ],
                "where": {"state": "TX", "temp_min": 70},
                "select": "city",
            },
        ),
        (
            "cctu-1486-string-001",
            "string_constraint",
            "string.transform",
            {
                "text": "Carnolot Shift",
                "operations": [{"op": "lower"}, {"op": "remove_spaces"}, {"op": "reverse"}],
            },
        ),
        (
            "cctu-1486-string-002",
            "string_constraint",
            "string.transform",
            {
                "text": "AA-BB-cc-11",
                "operations": [{"op": "keep_alnum"}, {"op": "lower"}, {"op": "sort_chars"}],
            },
        ),
        (
            "cctu-1486-string-003",
            "string_constraint",
            "string.transform",
            {
                "text": "red blue red",
                "operations": [
                    {"op": "replace", "old": "red", "new": "green"},
                    {"op": "upper"},
                ],
            },
        ),
        (
            "cctu-1486-string-004",
            "string_constraint",
            "string.transform",
            {"text": "abcdefghi", "operations": [{"op": "take_every_n", "n": 2}]},
        ),
        (
            "cctu-1486-string-005",
            "string_constraint",
            "string.transform",
            {
                "text": "Verifier",
                "operations": [
                    {"op": "lower"},
                    {"op": "replace", "old": "e", "new": "3"},
                    {"op": "reverse"},
                ],
            },
        ),
        (
            "cctu-1486-graph-001",
            "graph_path",
            "graph.shortest_path",
            {
                "edges": [["A", "B", 3], ["B", "D", 4], ["A", "C", 10], ["C", "D", 1]],
                "start": "A",
                "end": "D",
            },
        ),
        (
            "cctu-1486-graph-002",
            "graph_path",
            "graph.shortest_path",
            {
                "edges": [["S", "A", 2], ["A", "T", 5], ["S", "B", 4], ["B", "T", 1]],
                "start": "S",
                "end": "T",
            },
        ),
        (
            "cctu-1486-graph-003",
            "graph_path",
            "graph.shortest_path",
            {
                "edges": [["R", "Q", 6], ["R", "P", 2], ["P", "Q", 2], ["Q", "Z", 3]],
                "start": "R",
                "end": "Z",
            },
        ),
        (
            "cctu-1486-graph-004",
            "graph_path",
            "graph.shortest_path",
            {
                "edges": [["M", "N", 1], ["N", "O", 1], ["M", "O", 5], ["O", "P", 2]],
                "start": "M",
                "end": "P",
            },
        ),
        (
            "cctu-1486-graph-005",
            "graph_path",
            "graph.shortest_path",
            {
                "edges": [["H", "I", 2], ["I", "K", 2], ["H", "J", 1], ["J", "K", 6]],
                "start": "H",
                "end": "K",
            },
        ),
    ]

    cases: list[BenchmarkCase] = []
    for case_id, family, tool_name, arguments in raw_cases:
        expected_result = execute_tool(tool_name, arguments)
        expected_answer = _final_answer_from_result(family, expected_result)
        prompt = _build_prompt(
            case_id=case_id,
            family=family,
            tool_name=tool_name,
            arguments=arguments,
        )
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                family=family,
                tool_name=tool_name,
                tool_arguments=arguments,
                expected_tool_result=expected_result,
                expected_final_answer=expected_answer,
                prompt=prompt,
            )
        )
    return cases


def execute_tool(tool_name: str, arguments: JsonDict) -> Any:
    """Execute one of the local deterministic benchmark tools."""

    if tool_name == "arithmetic.evaluate":
        return _execute_arithmetic(arguments)
    if tool_name == "table.filter_rows":
        return _execute_table_filter(arguments)
    if tool_name == "string.transform":
        return _execute_string_transform(arguments)
    if tool_name == "graph.shortest_path":
        return _execute_shortest_path(arguments)
    raise ValueError(f"unknown tool: {tool_name}")


def compliant_transcript_for_case(case: BenchmarkCase) -> str:
    """Return a gold transcript used by tests and validator sanity checks."""

    payload = {
        "tool_call": {"name": case.tool_name, "arguments": case.tool_arguments},
        "tool_result": case.expected_tool_result,
        "final_answer": case.expected_final_answer,
        "verifier": {"accept": True},
    }
    return json.dumps(payload, sort_keys=True)


def extract_json_object(text: str) -> JsonDict | None:
    """Extract the first JSON object from a model response."""

    decoder = json.JSONDecoder()
    stripped = text.strip()
    if not stripped:
        return None
    for start, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(stripped[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def validate_transcript(case: BenchmarkCase, output_text: str) -> JsonDict:
    """Validate a model transcript against one benchmark case."""

    obj = extract_json_object(output_text)
    if obj is None:
        validator = _validator_payload(
            parse_error="no_json_object",
            tool_call_structure_valid=False,
            tool_result_consistent=False,
            final_answer_valid=False,
            verifier_outcome_valid=False,
            tool_result_error="missing_json",
            model_declared_accept=None,
        )
        return {
            "validator_result": validator,
            "verifier_result": _verifier_payload(
                base_valid=False,
                accepted=False,
                model_declared_accept=None,
                verifier_outcome_valid=False,
            ),
        }

    tool_call = obj.get("tool_call")
    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else None
    tool_args = tool_call.get("arguments") if isinstance(tool_call, dict) else None
    tool_call_structure_valid = (
        tool_name == case.tool_name and _canonical(tool_args) == _canonical(case.tool_arguments)
    )

    tool_result_error = None
    tool_result_consistent = False
    if isinstance(tool_name, str) and isinstance(tool_args, dict):
        try:
            declared_result = obj.get("tool_result")
            actual_result = execute_tool(tool_name, tool_args)
            tool_result_consistent = _canonical(declared_result) == _canonical(actual_result)
        except Exception as exc:
            tool_result_error = f"{type(exc).__name__}: {exc}"
    else:
        tool_result_error = "missing_or_malformed_tool_call"

    final_answer = obj.get("final_answer")
    final_answer_valid = (
        isinstance(final_answer, str)
        and _normalise_final_answer(final_answer) == _normalise_final_answer(case.expected_final_answer)
    )

    verifier = obj.get("verifier")
    model_declared_accept = (
        verifier.get("accept") if isinstance(verifier, dict) and isinstance(verifier.get("accept"), bool) else None
    )
    base_valid = tool_call_structure_valid and tool_result_consistent and final_answer_valid
    verifier_outcome_valid = model_declared_accept is not None and model_declared_accept == base_valid
    accepted = base_valid and verifier_outcome_valid

    validator = _validator_payload(
        parse_error=None,
        tool_call_structure_valid=tool_call_structure_valid,
        tool_result_consistent=tool_result_consistent,
        final_answer_valid=final_answer_valid,
        verifier_outcome_valid=verifier_outcome_valid,
        tool_result_error=tool_result_error,
        model_declared_accept=model_declared_accept,
    )
    return {
        "validator_result": validator,
        "verifier_result": _verifier_payload(
            base_valid=base_valid,
            accepted=accepted,
            model_declared_accept=model_declared_accept,
            verifier_outcome_valid=verifier_outcome_valid,
        ),
    }


def build_manifest_row(case: BenchmarkCase, generation_row: JsonDict) -> JsonDict:
    """Join a raw model generation row with deterministic validation results."""

    output_text = str(generation_row.get("output_text") or "")
    validation = validate_transcript(case, output_text)
    return {
        "case_id": case.case_id,
        "family": case.family,
        "prompt": case.prompt,
        "model_hf_id": generation_row.get("model_hf_id"),
        "model_name": generation_row.get("model_name"),
        "generation_source": generation_row.get("generation_source"),
        "elapsed_seconds": generation_row.get("elapsed_seconds"),
        "blocker": generation_row.get("blocker"),
        "model_output": output_text,
        "validator_result": validation["validator_result"],
        "verifier_result": validation["verifier_result"],
    }


def aggregate_manifest_metrics(rows: list[JsonDict]) -> JsonDict:
    """Compute artifact-level rates from manifest rows."""

    if not rows:
        return {
            "tool_call_validity_rate": 0.0,
            "final_answer_validity_rate": 0.0,
            "verifier_catch_rate": 1.0,
            "verifier_false_accept_rate": 0.0,
        }
    total = len(rows)
    tool_valid = sum(
        bool(row["validator_result"]["tool_call_structure_valid"]) for row in rows
    )
    answer_valid = sum(bool(row["validator_result"]["final_answer_valid"]) for row in rows)
    invalid_rows = [row for row in rows if not bool(row["verifier_result"]["base_valid"])]
    caught = sum(bool(row["verifier_result"]["caught_invalid"]) for row in invalid_rows)
    false_accepts = sum(bool(row["verifier_result"]["false_accept"]) for row in invalid_rows)
    invalid_total = len(invalid_rows)
    return {
        "tool_call_validity_rate": round(tool_valid / total, 6),
        "final_answer_validity_rate": round(answer_valid / total, 6),
        "verifier_catch_rate": round(caught / invalid_total, 6) if invalid_total else 1.0,
        "verifier_false_accept_rate": (
            round(false_accepts / invalid_total, 6) if invalid_total else 0.0
        ),
    }


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """Write the durable bootstrap artifact required by the experiment prompt."""

    payload: JsonDict = {
        "status": "in_progress",
        "run_date": run_date,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": False,
        "executable_constraint_benchmark_ready": False,
        "benchmark_cases": 0,
        "validators_path": VALIDATORS_PATH,
        "manifest_path": _display_path(DEFAULT_MANIFEST_PATH),
        "tool_call_validity_rate": None,
        "final_answer_validity_rate": None,
        "verifier_catch_rate": None,
        "verifier_false_accept_rate": None,
        "models_used": [],
        "tests_run": [],
        "honest_verdict": "in_progress",
    }
    _write_json(Path(output_path), payload)
    return payload


def run_microbenchmark(
    *,
    output_path: Path | str = DEFAULT_ARTIFACT_PATH,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    run_date: str = RUN_DATE,
    model_specs: Iterable[JsonDict] = MANDATED_MODEL_SPECS,
    collect_model_outputs_fn: CollectModelOutputsFn = None,  # type: ignore[assignment]
    max_models: int = 1,
    tests_run: list[str] | None = None,
) -> JsonDict:
    """Run the benchmark and write both the manifest JSONL and final artifact."""

    output = Path(output_path)
    manifest = Path(manifest_path)
    write_in_progress_artifact(output, run_date=run_date)

    cases = build_benchmark_cases()
    collector = collect_model_outputs_fn or collect_live_model_outputs
    specs = [dict(spec) for spec in model_specs]
    rows: list[JsonDict] = []
    model_attempts: list[JsonDict] = []
    case_by_id = {case.case_id: case for case in cases}

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
        collection = collector(spec, cases)
        summary = dict(collection.get("summary") or {})
        model_attempts.append(summary)
        for generation_row in collection.get("rows") or []:
            case_id = generation_row.get("case_id")
            case = case_by_id.get(case_id)
            if case is None:
                continue
            rows.append(build_manifest_row(case, generation_row))

    _write_jsonl(manifest, rows)
    metrics = aggregate_manifest_metrics(rows)
    models_used = [
        str(summary["hf_id"])
        for summary in model_attempts
        if summary.get("model_used") is True and summary.get("hf_id")
    ]
    live_used = any(
        row.get("generation_source") == "live_sota_llamacpp"
        and not row.get("blocker")
        and row.get("model_hf_id") in {spec["hf_id"] for spec in MANDATED_MODEL_SPECS}
        for row in rows
    )
    ready = len(cases) == BENCHMARK_CASE_COUNT and bool(rows) and live_used
    artifact: JsonDict = {
        "status": "complete",
        "run_date": run_date,
        "schema_version": 1,
        "model_specs": [spec["hf_id"] for spec in MANDATED_MODEL_SPECS],
        "live_sota_model_inference_used": bool(live_used),
        "executable_constraint_benchmark_ready": bool(ready),
        "benchmark_cases": len(cases),
        "validators_path": VALIDATORS_PATH,
        "manifest_path": _display_path(manifest),
        "tool_call_validity_rate": metrics["tool_call_validity_rate"],
        "final_answer_validity_rate": metrics["final_answer_validity_rate"],
        "verifier_catch_rate": metrics["verifier_catch_rate"],
        "verifier_false_accept_rate": metrics["verifier_false_accept_rate"],
        "models_used": models_used,
        "tests_run": list(tests_run or []),
        "honest_verdict": (
            "complete: executable CCTU micro-benchmark ready with live local SOTA GGUF inference"
            if ready
            else "complete: executable CCTU micro-benchmark written but live SOTA inference incomplete"
        ),
        "model_attempts": model_attempts,
        "manifest_rows": len(rows),
    }
    _write_json(output, artifact)
    return artifact


def collect_live_model_outputs(
    spec: JsonDict,
    cases: list[BenchmarkCase],
    *,
    resolver: ResolverFn | None = None,
    llama_importer: LlamaImporterFn | None = None,
    env_preparer: Callable[[], JsonDict] | None = None,
) -> JsonDict:
    """Collect raw outputs from one local GGUF model through llama.cpp."""

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

    prepare_env = env_preparer or prepare_llama_environment
    env_details = prepare_env()
    importer = llama_importer or _default_llama_importer
    ok, llama_class, import_error = importer()
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
            n_ctx=4096,
            seed=1486,
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
        for case in cases:
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
                output_text = _completion_text(result)
                blocker = None if output_text.strip() else "empty_generation"
            except Exception as exc:
                output_text = ""
                blocker = f"{type(exc).__name__}: {exc}"
            rows.append(
                {
                    "case_id": case.case_id,
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
        _close_llama(llm)

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


def prepare_llama_environment() -> JsonDict:
    """Prepend the project venv CUDA library dirs needed by llama.cpp."""

    root = _repo_root()
    candidate_dirs = [
        str(path)
        for pattern in (
            ".venv/lib/python*/site-packages/nvidia/cuda_runtime/lib",
            ".venv/lib/python*/site-packages/nvidia/cublas/lib",
        )
        for path in sorted(root.glob(pattern))
        if path.is_dir()
    ]
    before = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [path for path in candidate_dirs if path]
    if before:
        parts.append(before)
    after = ":".join(dict.fromkeys(parts))
    if after:
        os.environ["LD_LIBRARY_PATH"] = after
    return {
        "candidate_library_dirs": candidate_dirs,
        "ld_library_path_before": before,
        "ld_library_path_after": os.environ.get("LD_LIBRARY_PATH", ""),
    }


def _execute_arithmetic(arguments: JsonDict) -> JsonDict:
    operation = arguments.get("operation")
    if operation == "sum":
        return {"value": sum(int(value) for value in arguments["numbers"])}
    if operation == "difference":
        return {
            "value": int(arguments["start"])
            - sum(int(value) for value in arguments["subtract"])
        }
    if operation == "product_plus":
        product = 1
        for value in arguments["numbers"]:
            product *= int(value)
        return {"value": product + int(arguments.get("add", 0))}
    if operation == "weighted_total":
        items = arguments["items"]
        weights = arguments["weights"]
        return {"value": sum(int(items[key]) * int(weights[key]) for key in sorted(items))}
    if operation == "mod":
        return {"value": int(arguments["value"]) % int(arguments["modulus"])}
    raise ValueError(f"unknown arithmetic operation: {operation}")


def _execute_table_filter(arguments: JsonDict) -> JsonDict:
    rows = list(arguments["rows"])
    where = dict(arguments.get("where") or {})
    filtered = [row for row in rows if _row_matches(row, where)]
    select = str(arguments.get("select") or "__count__")
    if select == "__count__":
        return {"count": len(filtered)}
    return {"rows": [row[select] for row in filtered]}


def _row_matches(row: JsonDict, where: JsonDict) -> bool:
    for key, expected in where.items():
        if key.endswith("_min"):
            field = key[: -len("_min")]
            if row.get(field) is None or row[field] < expected:
                return False
        elif key.endswith("_max"):
            field = key[: -len("_max")]
            if row.get(field) is None or row[field] > expected:
                return False
        elif key.endswith("_contains"):
            field = key[: -len("_contains")]
            if str(expected) not in str(row.get(field, "")):
                return False
        elif row.get(key) != expected:
            return False
    return True


def _execute_string_transform(arguments: JsonDict) -> JsonDict:
    value = str(arguments["text"])
    for operation in arguments.get("operations") or []:
        op = operation.get("op")
        if op == "lower":
            value = value.lower()
        elif op == "upper":
            value = value.upper()
        elif op == "remove_spaces":
            value = value.replace(" ", "")
        elif op == "reverse":
            value = value[::-1]
        elif op == "replace":
            value = value.replace(str(operation["old"]), str(operation["new"]))
        elif op == "keep_alnum":
            value = "".join(char for char in value if char.isalnum())
        elif op == "sort_chars":
            value = "".join(sorted(value))
        elif op == "take_every_n":
            value = value[:: int(operation["n"])]
        else:
            raise ValueError(f"unknown string operation: {op}")
    return {"value": value}


def _execute_shortest_path(arguments: JsonDict) -> JsonDict:
    start = str(arguments["start"])
    end = str(arguments["end"])
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for left, right, weight in arguments["edges"]:
        adjacency.setdefault(str(left), []).append((str(right), int(weight)))
        adjacency.setdefault(str(right), []).append((str(left), int(weight)))
    queue: list[tuple[int, tuple[str, ...], str]] = [(0, (start,), start)]
    best: dict[str, int] = {}
    while queue:
        cost, path, node = heapq.heappop(queue)
        if node in best and best[node] <= cost:
            continue
        best[node] = cost
        if node == end:
            return {"cost": cost, "path": list(path)}
        for neighbor, weight in sorted(adjacency.get(node, [])):
            if neighbor not in best:
                heapq.heappush(queue, (cost + weight, (*path, neighbor), neighbor))
    raise ValueError(f"no path from {start} to {end}")


def _final_answer_from_result(family: str, result: Any) -> str:
    if family == "arithmetic":
        return str(result["value"])
    if family == "table_filter":
        if "count" in result:
            return str(result["count"])
        return ", ".join(str(value) for value in result["rows"])
    if family == "string_constraint":
        return str(result["value"])
    if family == "graph_path":
        return f"{'->'.join(result['path'])} cost={result['cost']}"
    raise ValueError(f"unknown family: {family}")


def _build_prompt(*, case_id: str, family: str, tool_name: str, arguments: JsonDict) -> str:
    schema = {
        "tool_call": {"name": tool_name, "arguments": arguments},
        "tool_result": "<execute the local tool and place JSON result here>",
        "final_answer": "<compose from tool_result only>",
        "verifier": {"accept": "<true only if every field is consistent>"},
    }
    return (
        "You are evaluating a CCTU-style executable constraint case.\n"
        f"Case: {case_id}\n"
        f"Constraint family: {family}\n"
        f"Use exactly one local tool named {tool_name} with these arguments:\n"
        f"{json.dumps(arguments, sort_keys=True)}\n"
        "Return exactly one JSON object and no prose. The object must follow this shape:\n"
        f"{json.dumps(schema, sort_keys=True)}"
    )


def _validator_payload(
    *,
    parse_error: str | None,
    tool_call_structure_valid: bool,
    tool_result_consistent: bool,
    final_answer_valid: bool,
    verifier_outcome_valid: bool,
    tool_result_error: str | None,
    model_declared_accept: bool | None,
) -> JsonDict:
    return {
        "parse_error": parse_error,
        "tool_call_structure_valid": bool(tool_call_structure_valid),
        "tool_result_consistent": bool(tool_result_consistent),
        "final_answer_valid": bool(final_answer_valid),
        "verifier_outcome_valid": bool(verifier_outcome_valid),
        "tool_result_error": tool_result_error,
        "model_declared_accept": model_declared_accept,
    }


def _verifier_payload(
    *,
    base_valid: bool,
    accepted: bool,
    model_declared_accept: bool | None,
    verifier_outcome_valid: bool,
) -> JsonDict:
    return {
        "base_valid": bool(base_valid),
        "accepted": bool(accepted),
        "model_declared_accept": model_declared_accept,
        "verifier_outcome_valid": bool(verifier_outcome_valid),
        "caught_invalid": not bool(base_valid) and not bool(accepted),
        "false_accept": not bool(base_valid) and bool(accepted),
    }


def _canonical(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _normalise_final_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


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
    return str(first.get("text") or "")


def _close_llama(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        close()


def _default_llama_importer() -> tuple[bool, type[Any] | None, str | None]:
    try:
        from llama_cpp import Llama  # noqa: PLC0415
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"
    return True, Llama, None


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


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the conductor and manual runs."""

    args = list(sys.argv[1:] if argv is None else argv)
    max_models = int(os.getenv("CARNOT_CCTU_1486_MAX_MODELS", "1"))
    if "--all-models" in args:
        max_models = len(MANDATED_MODEL_SPECS)
    artifact = run_microbenchmark(max_models=max_models)
    print(
        "[exp1486] "
        f"ready={artifact['executable_constraint_benchmark_ready']} "
        f"cases={artifact['benchmark_cases']} "
        f"models={artifact['models_used']} "
        f"final_answer_rate={artifact['final_answer_validity_rate']} "
        f"false_accept={artifact['verifier_false_accept_rate']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by conductor.
    raise SystemExit(main())


__all__ = [
    "BENCHMARK_CASE_COUNT",
    "DEFAULT_ARTIFACT_PATH",
    "DEFAULT_MANIFEST_PATH",
    "MANDATED_MODEL_SPECS",
    "REQUIRED_ARTIFACT_FIELDS",
    "BenchmarkCase",
    "aggregate_manifest_metrics",
    "build_benchmark_cases",
    "build_manifest_row",
    "collect_live_model_outputs",
    "compliant_transcript_for_case",
    "execute_tool",
    "extract_json_object",
    "prepare_llama_environment",
    "run_microbenchmark",
    "validate_transcript",
    "write_in_progress_artifact",
]
